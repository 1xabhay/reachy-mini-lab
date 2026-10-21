"""The Sage kernel as the mind of a Pipecat pipeline.

The LLM is not in the body's pipeline. ``SageProcessor`` sits between STT and
TTS: a final transcript starts a kernel turn, ``reply_delta`` text goes straight
to TTS (whose first-clause aggregator starts the voice early), every delivered
effect gets one ``result``, and an interruption reports what was actually
played. ``BotSpeech`` sits after the output transport, where frames arrive as
they are played, and feeds that playback into the interruption policy.
"""

import asyncio
import contextlib
import inspect
import logging
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterruptionFrame,
    InterruptionWorkerFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    StartFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    TTSTextFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.text.base_text_aggregator import AggregationType

from reachy_sage_voice.interruption import InterruptionTracker
from reachy_sage_voice.kernel import EMOTIONS, NO_SOUND, KernelUnavailableError, Session
from reachy_sage_voice.speech import TOOLS, SpeechStream, is_self_echo, move_sound, trim_echo
from reachy_sage_voice.voice import ENGINES, VoiceStyleFrame, style_settings, vocal_action

logger = logging.getLogger(__name__)

#: Said locally when the kernel cannot be reached, and when another body holds it.
OFFLINE_LINE = "I can't reach my memory right now."
BUSY_LINE = "One moment."
#: Long-poll length. Short enough that a stop reaches a playing body quickly.
POLL_S = 2.0
#: Bot replies remembered for the self-echo check (an earlier voice bot's window).
RECENT_REPLIES = 5
#: Longest a short reply's move waits for the voice before playing muted.
VOICE_WAIT_S = 10.0
#: Run states after which a turn will never deliver a reply.
_ENDED = ("failed", "cancelled", "unknown")
#: How the kernel's own turns (greetings, nudges) are named (protocol v1 §9).
KERNEL_TURN = "kernel-"


class Playback:
    """The pieces of the current response that have actually been played.

    Word by word with Kokoro's captioned speech: the output transport releases
    each word's TTSTextFrame as it is played (``tts.py``). Without captions, a
    clause counts only once all of it has played.
    """

    def __init__(self) -> None:
        """Start with nothing played."""
        self.spoken: list[str] = []
        self.speaking = False  # audio is playing
        self.queued = False  # text is on its way to the voice
        self._boundary = asyncio.Event()
        self._voice_free = asyncio.Event()
        self._voice_free.set()

    @property
    def busy(self) -> bool:
        """The voice is speaking or about to."""
        return self.speaking or self.queued

    async def idle(self) -> None:
        """Return once the voice has nothing left to say."""
        while self.busy:
            await self._boundary.wait()

    def hold_voice(self) -> None:
        """Hold the voice while a move sounds: sounds never overlap speech."""
        self._voice_free.clear()

    def release_voice(self) -> None:
        """Let the voice speak again: the move's sound is over."""
        self._voice_free.set()

    async def voice_free(self) -> None:
        """Return once no move is sounding."""
        await self._voice_free.wait()

    async def boundary(self) -> None:
        """Return at the next audio boundary: between sentences, or now if silent."""
        if self.speaking:
            await self._boundary.wait()

    def mark_boundary(self) -> None:
        """Release everyone waiting for a boundary."""
        self._boundary.set()
        self._boundary = asyncio.Event()

    @property
    def text(self) -> str:
        """What the person heard of this response."""
        return " ".join(s.strip() for s in self.spoken)


def _ms(seconds: float) -> int:
    return max(0, round(seconds * 1000))


@dataclass
class _Turn:
    turn_id: str
    text: str = ""  # what the person said
    sent_at: float | None = None
    first_text_at: float | None = None
    first_audio_at: float | None = None
    reply_seq: int | None = None
    reply_text: str | None = None  # the final reply, once delivered
    reported: bool = False
    streamed: bool = False
    ended: bool = False  # played and reported, or never admitted
    emotes: dict[int, asyncio.Task] = field(default_factory=dict)
    speech: SpeechStream = field(default_factory=lambda: SpeechStream((*TOOLS, *EMOTIONS)))


class SageProcessor(FrameProcessor):
    """Final transcripts to the kernel; its reply, emotes and results back out."""

    def __init__(self, session: Session, play_emote: Callable, playback: Playback,
                 now: Callable[[], float] = time.monotonic, play_sound: Callable | None = None,
                 engine: dict = ENGINES["kokoro"], **kwargs):
        """Talk through ``session``; ``play_emote(name, sound)`` performs a move, muted or not.

        ``play_sound(move)`` plays a move's own sound for a vocal (None: no sounds),
        and ``engine`` maps voice styles and vocals onto the TTS (``voice.ENGINES``).
        """
        super().__init__(**kwargs)
        self._session, self._play_emote, self._playback, self._now = (
            session, play_emote, playback, now)
        self._play_sound, self._engine = play_sound, engine
        self._recent: deque[str] = deque(maxlen=RECENT_REPLIES)
        self._speech_ended: float | None = None
        self._turn: _Turn | None = None
        self._task: asyncio.Task | None = None
        self._audio_done = asyncio.Event()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Route speech, playback and interruptions; pass every frame on."""
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            self._task = self.create_task(self._listen())
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            # The VAD decides stop_secs after the person actually stopped.
            self._speech_ended = self._now() - frame.stop_secs
        elif isinstance(frame, TranscriptionFrame) and not is_self_echo(frame.text, self._recent):
            # The mic hears the speaker late: its own closing words can lead a turn.
            heard = trim_echo(frame.text, [*self._recent, self._playback.text])
            if heard:
                await self._submit(heard)
        elif isinstance(frame, BotStartedSpeakingFrame):
            if self._turn and self._turn.first_audio_at is None:
                self._turn.first_audio_at = self._now()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._audio_done.set()
        elif isinstance(frame, InterruptionFrame):
            await self._interrupt()
            self._task = self.create_task(self._listen())  # the kernel may still speak first
        await self.push_frame(frame, direction)

    async def cleanup(self):
        """Stop talking to the kernel when the pipeline ends."""
        await super().cleanup()
        if self._task:
            await self.cancel_task(self._task)
        for task in self._turn.emotes.values() if self._turn else ():
            await self.cancel_task(task)

    # --- a turn ----------------------------------------------------------------

    async def _submit(self, heard: str) -> None:
        turn = self._turn
        running = turn is not None and not turn.ended
        if running and turn.reply_seq is None and not turn.streamed:
            # Still thinking: the kernel forgets a turn stopped before its reply,
            # so its words lead the one that replaces it, as one message.
            heard = f"{turn.text} {heard}"
        timing = {}
        if self._speech_ended is not None:
            timing["stt_ms"] = _ms(self._now() - self._speech_ended)
        await self._interrupt()  # a new turn replaces one still running
        if running and turn.streamed:
            # Its words are queued in the TTS: interrupt the voice, as barge-in does,
            # or they lead the next reply ("OhOh, a messy screen!").
            await self.push_frame(InterruptionFrame())
        self._task = self.create_task(self._converse(heard, timing))

    async def _converse(self, heard: str, timing: dict) -> None:
        """Take one turn of the person's, then listen for the kernel's own."""
        turn = self._turn = _Turn("reachy-" + uuid.uuid4().hex, heard)
        self._audio_done.clear()
        try:
            admitted = await self._session.start(turn.turn_id, heard, timing)
        except KernelUnavailableError as error:
            logger.warning("kernel unreachable: %s", error)
            admitted = None
            await self._say(OFFLINE_LINE)
        if admitted == "busy":
            await self._say(BUSY_LINE)
        if admitted == "admitted":
            turn.sent_at = self._now()
            await self._play(turn)
        turn.ended = True
        await self._listen()

    async def _listen(self) -> None:
        """Between turns, follow any turn the kernel starts itself: a greeting, a nudge."""
        while True:
            try:
                body = await self._session.poll(0, POLL_S)
            except KernelUnavailableError as error:
                logger.warning("kernel unreachable between turns: %s", error)
                await asyncio.sleep(POLL_S)
                continue
            turn_id = next((e["turn_id"] for e in body["effects"]
                            if (e.get("turn_id") or "").startswith(KERNEL_TURN)), None)
            if turn_id is not None:
                turn = self._turn = _Turn(turn_id)
                self._audio_done.clear()
                await self._play(turn, body)
                turn.ended = True

    async def _play(self, turn: _Turn, body: dict | None = None) -> None:
        try:
            await self._follow(turn, body)
        except KernelUnavailableError as error:
            logger.warning("kernel unreachable mid-turn: %s", error)
            if turn.streamed:
                await self.push_frame(LLMFullResponseEndFrame())

    async def _follow(self, turn: _Turn, first: dict | None = None) -> None:
        """Stream this turn's reply, then report it once it has played.

        ``first`` is a poll already in hand that holds the turn's first effects.
        """
        offset = 0
        while turn.reply_seq is None:
            body, first = first or await self._session.poll(offset, POLL_S), None
            tools = []
            for effect in body["effects"]:
                kind = effect["kind"]
                if kind == "stop":
                    await self._stop_body()
                    return
                if effect.get("turn_id") != turn.turn_id:
                    continue
                if kind == "reply_delta":
                    # The kernel sends from our offset on; skip anything already had.
                    piece = effect["text"][max(0, offset - effect["offset"]):]
                    offset = effect["offset"] + len(effect["text"])
                    await self._speak(turn, piece)
                elif kind == "reply":
                    turn.reply_seq, turn.reply_text = effect["sequence"], effect["text"]
                    if effect["stale"]:
                        await self._report(turn.reply_seq, "failed", reason="stale")
                        turn.reported = True
                        break
                    self._recent.append(effect["text"])
                    await self._speak(turn, effect["text"][offset:], final=True)
                elif kind in ("voice_style", "vocal"):
                    await self._voice_tool(turn, effect)  # in order with the words
                else:
                    tools.append(effect)
            # After the whole batch, so a move knows the reply delivered with it.
            for effect in tools:
                self._start_tool(turn, effect)
            run = body.get("turn") or {}
            if run.get("turn_id") == turn.turn_id and run.get("run_status") in _ENDED:
                break
        if turn.streamed:
            await self.push_frame(LLMFullResponseEndFrame())
            if not await self._played(offset):
                return
        if turn.reply_seq is not None and not turn.reported:
            turn.reported = True
            await self._report(turn.reply_seq, "done", timing=self._reply_timing(turn))

    async def _speak(self, turn: _Turn, piece: str, final: bool = False) -> None:
        if piece and turn.first_text_at is None:
            turn.first_text_at = self._now()
        piece = turn.speech.feed(piece) + (turn.speech.flush() if final else "")
        if not piece.strip():
            return
        if not turn.streamed:
            await self._before_speaking()
            turn.streamed = True
            await self.push_frame(LLMFullResponseStartFrame())
        await self.push_frame(LLMTextFrame(piece))

    async def _before_speaking(self) -> None:
        """Let a sounding move finish, then mark the voice as about to speak."""
        await self._playback.voice_free()
        self._playback.queued = True

    async def _say(self, line: str) -> None:
        await self._before_speaking()
        await self.push_frame(TTSSpeakFrame(line))

    async def _played(self, offset: int) -> bool:
        """Wait for the audio to finish, still listening for a stop. False if stopped."""
        done = asyncio.ensure_future(self._audio_done.wait())
        try:
            while not done.done():
                poll = asyncio.ensure_future(self._session.poll(offset, POLL_S))
                await asyncio.wait({done, poll}, return_when=asyncio.FIRST_COMPLETED)
                if not poll.done():
                    poll.cancel()  # its effects stay in the journal for the next poll
                    break
                if any(e["kind"] == "stop" for e in poll.result()["effects"]):
                    await self._stop_body()
                    return False
            return True
        finally:
            done.cancel()

    async def _stop_body(self) -> None:
        """Obey a stop from the kernel: interrupt the whole pipeline, as barge-in does."""
        await self.push_frame(InterruptionWorkerFrame(), FrameDirection.UPSTREAM)

    def _reply_timing(self, turn: _Turn) -> dict:
        timing = {}
        if turn.sent_at is not None and turn.first_text_at is not None:
            timing["ttft_ms"] = _ms(turn.first_text_at - turn.sent_at)
        # Only for the person's own turns: the kernel's follow no speech of theirs.
        if (turn.sent_at is not None and self._speech_ended is not None
                and turn.first_audio_at is not None):
            timing["first_audio_ms"] = _ms(turn.first_audio_at - self._speech_ended)
        return timing

    # --- tools -----------------------------------------------------------------

    def _start_tool(self, turn: _Turn, effect: dict) -> None:
        name = (effect.get("input") or {}).get("name")
        ok = effect["kind"] == "emote" and name in EMOTIONS and not effect.get("stale")
        turn.emotes[effect["sequence"]] = self.create_task(
            self._emote(turn, effect["sequence"], name if ok else None))

    async def _voice_tool(self, turn: _Turn, effect: dict) -> None:
        sequence, value = effect["sequence"], next(iter((effect.get("input") or {}).values()), "")
        if effect.get("stale"):
            await self._report(sequence, "failed", reason="stale")
            return
        if effect["kind"] == "voice_style":
            settings = style_settings(self._engine, value)
            if settings is None:
                await self._report(sequence, "failed", reason="unsupported")
                return
            await self.push_frame(VoiceStyleFrame(settings))
            await self._report(sequence, "done")
            return
        action = vocal_action(self._engine, value)
        if action is None or (action[0] == "sound" and self._play_sound is None):
            await self._report(sequence, "failed", reason="unsupported")
        elif action[0] == "say":
            await self._speak(turn, action[1] + " ")
            await self._report(sequence, "done")
        else:
            turn.emotes[sequence] = self.create_task(self._vocal(sequence, action[1]))

    async def _vocal(self, sequence: int, move: str) -> None:
        """Play a move's own sound between words, holding the voice while it sounds."""
        playback = self._playback
        await playback.boundary()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(playback.idle(), VOICE_WAIT_S)
        playback.hold_voice()
        status = "done"
        try:
            await self._play_sound(move)
        except Exception:
            logger.exception("vocal %s failed", move)
            status = "failed"
        finally:
            playback.release_voice()
        await self._report(sequence, status)

    async def _emote(self, turn: _Turn, sequence: int, name: str | None) -> None:
        status, duration_ms = "failed", None
        if name is not None:
            playback = self._playback
            await playback.boundary()  # a move must not land mid-word
            if move_sound(turn.reply_text, voice_busy=False) and playback.busy:
                # Little to say, and it is about to be said: say it, then sound.
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(playback.idle(), VOICE_WAIT_S)
            sound = move_sound(turn.reply_text, playback.busy) and name not in NO_SOUND
            if sound:
                playback.hold_voice()
            began = self._now()
            try:
                result = self._play_emote(name, sound)
                if inspect.isawaitable(result):
                    await result
                status, duration_ms = "done", _ms(self._now() - began)
            except Exception:
                logger.exception("emote %s failed", name)
            finally:
                if sound:
                    playback.release_voice()
        await self._report(sequence, status, duration_ms=duration_ms)

    # --- interruption ----------------------------------------------------------

    async def _interrupt(self) -> None:
        """Cancel the turn in hand and report, off the frame path, what it got to."""
        turn, task = self._turn, self._task
        self._turn = self._task = None
        if task and not task.done():
            await self.cancel_task(task)
        if turn is not None and not (turn.reported and all(t.done() for t in turn.emotes.values())):
            self.create_task(self._report_interrupted(turn))

    async def _report_interrupted(self, turn: _Turn) -> None:
        for sequence, task in turn.emotes.items():
            if not task.done():
                await self.cancel_task(task)
                await self._report(sequence, "interrupted")
        if turn.reply_seq is None:
            # Still streaming. From its first piece the body is acting on the reply,
            # so it reports what was heard, by turn, then stops the turn.
            if turn.streamed and not turn.reported:
                turn.reported = True
                await self._report(None, "interrupted", text=self._playback.text,
                                   timing=self._reply_timing(turn), turn_id=turn.turn_id)
            try:
                await self._session.stop(turn.turn_id)
            except KernelUnavailableError as error:
                logger.warning("could not stop turn %s: %s", turn.turn_id, error)
        elif not turn.reported:
            turn.reported = True
            await self._report(turn.reply_seq, "interrupted", text=self._playback.text,
                               timing=self._reply_timing(turn))

    async def _report(self, sequence: int | None, status: str, text: str | None = None,
                      timing: dict | None = None, reason: str | None = None,
                      turn_id: str | None = None, duration_ms: int | None = None) -> None:
        try:
            await self._session.result(sequence, status, text=text, timing=timing,
                                       reason=reason, turn_id=turn_id, duration_ms=duration_ms)
        except KernelUnavailableError as error:
            # The kernel times the effect out itself (max_s + 2 s); nothing is lost here.
            logger.warning("could not report %s for effect %s: %s", status, sequence, error)


def _ends_clause(frame: TTSTextFrame) -> bool:
    return frame.aggregated_by != AggregationType.WORD or frame.text.rstrip()[-1:] in ",;:.!?"


class BotSpeech(FrameProcessor):
    """After the output transport: playback into the interruption policy.

    Frames reach this point as their audio is played. Each played sentence is a
    sentence boundary, where the policy re-checks whether someone is still
    talking, and is recorded as heard.
    """

    def __init__(self, tracker: InterruptionTracker, playback: Playback,
                 now: Callable[[], float] = time.monotonic, **kwargs):
        """Feed ``tracker`` and ``playback`` from what the speaker played."""
        super().__init__(**kwargs)
        self._tracker, self._playback, self._now = tracker, playback, now
        self._speaking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Update the policy, then pass the frame on; yield if the policy says so."""
        await super().process_frame(frame, direction)
        t, must_yield = self._now(), False
        if isinstance(frame, BotStartedSpeakingFrame):
            self._speaking, self._playback.spoken = True, []
            self._playback.speaking = True
            must_yield = self._tracker.bot_started(t)
        elif isinstance(frame, TTSTextFrame):
            self._playback.spoken.append(frame.text)
            # Captioned speech arrives word by word; the policy and the body
            # language act at clause ends, as they did when text came by clause.
            if _ends_clause(frame):
                self._playback.mark_boundary()
                # A system frame overtakes queued data, so the last clause's text can
                # land after BotStopped: it must not reopen a finished response.
                if self._speaking:
                    must_yield = self._tracker.bot_started(t)  # the next clause starts now
        elif isinstance(frame, BotStoppedSpeakingFrame):
            # Pipecat 1.12 reports TTS audio stopped once per response (TTSStoppedFrame).
            self._speaking = self._playback.speaking = self._playback.queued = False
            self._playback.mark_boundary()
            self._tracker.bot_stopped(t)
            self._tracker.assistant_turn_stopped(t)
        elif isinstance(frame, InterruptionFrame):
            self._speaking = self._playback.speaking = self._playback.queued = False
            self._playback.mark_boundary()
            self._tracker.assistant_turn_stopped(t)
        await self.push_frame(frame, direction)
        if must_yield:
            await self.push_frame(InterruptionWorkerFrame(), FrameDirection.UPSTREAM)
