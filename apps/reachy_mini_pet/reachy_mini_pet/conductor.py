"""The pipeline's decision-making, with no threads and no clock of its own.

A realtime voice pipeline is threads shovelling bytes between a microphone,
three models and a speaker. That arrangement is impossible to test and easy to
get subtly wrong: the classic bugs are answering a question the user has
already abandoned, hearing your own voice and replying to it, and believing
you said something the user never heard.

So none of the decisions live in the threads. They all live here, in a
synchronous state machine that takes frames and events in and returns actions
out, with `now` passed in rather than read from a clock. The threads become
dumb shells that call this and do as they are told, and every rule about
listening, speaking and being interrupted becomes assertable at an exact frame.

Cancellation is one integer. Every action carries the `turn` it was made for,
and anything arriving stamped with an older turn is discarded - which is what
makes an interruption total, without a single lock or cancel token.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: The robot's real frame: 320 samples at 16 kHz.
FRAME_S = 0.02


@dataclass(frozen=True)
class Action:
    """Something the threads should do, stamped with the turn it belongs to."""

    turn: int


@dataclass(frozen=True)
class Transcribe(Action):
    """A complete utterance is ready to be turned into text."""

    frames: int
    #: How many of those frames were actually voiced. The transcriber uses this
    #: to refuse a clip that is mostly silence, which is what Whisper
    #: hallucinates on.
    voiced_frames: int = 0


@dataclass(frozen=True)
class Speak(Action):
    """One clause to synthesise and play."""

    text: str


@dataclass(frozen=True)
class Duck(Action):
    """Someone started talking - drop the volume, but do not commit yet."""


@dataclass(frozen=True)
class StopPlayback(Action):
    """They really are talking. Stop, and throw away what was queued."""


@dataclass(frozen=True)
class Resume(Action):
    """It was nothing. Carry on speaking."""


@dataclass
class Conductor:
    """Decides when to listen, when to speak, and when it has been interrupted."""

    #: Speech shorter than this is a cough, a door, a chair.
    min_speech_s: float = 0.3
    #: Silence this long ends the utterance. The single biggest latency term.
    hangover_s: float = 0.6
    #: Continuous speech this long during playback is a real interruption.
    barge_in_s: float = 0.3
    #: Ignore the microphone this long after speaking, so the tail of its own
    #: voice is never mistaken for yours.
    tail_gate_s: float = 0.25
    #: Hard ceiling on one utterance, so a stuck detector cannot allocate for ever.
    max_speech_s: float = 20.0
    frame_s: float = FRAME_S

    turn: int = 1
    speaking: bool = False
    pending_frames: int = 0

    _voiced_frames: int = field(default=0, repr=False)
    _first_voiced: float | None = field(default=None, repr=False)
    _last_voiced: float | None = field(default=None, repr=False)
    _playback_ended: float | None = field(default=None, repr=False)
    _voice_started: float | None = field(default=None, repr=False)
    _ducked: bool = field(default=False, repr=False)
    _spoken: dict[int, list[str]] = field(default_factory=dict, repr=False)

    @property
    def max_speech_frames(self) -> int:
        """Frames one utterance may hold before it is force-flushed."""
        return int(self.max_speech_s / self.frame_s)

    # ------------------------------------------------------------ audio

    def feed_audio(self, voiced: bool, now: float) -> list[Action]:
        """Consume one frame of microphone audio and decide what follows."""
        if self.speaking:
            return self._while_speaking(voiced, now)
        if self._in_tail_gate(now):
            return []
        return self._while_listening(voiced, now)

    def _in_tail_gate(self, now: float) -> bool:
        """Whether this frame is close enough to its own voice to distrust."""
        if self._playback_ended is None:
            return False
        return now - self._playback_ended < self.tail_gate_s

    def _while_listening(self, voiced: bool, now: float) -> list[Action]:
        """Accumulate an utterance, and close it when the talking stops."""
        if voiced:
            if self._first_voiced is None:
                self._first_voiced = now
            self._last_voiced = now
            self._voiced_frames += 1

        if self._first_voiced is None:
            return []  # nothing started yet; nothing to buffer

        self.pending_frames += 1

        if self.pending_frames >= self.max_speech_frames:
            return self._close(now)

        assert self._last_voiced is not None
        if now - self._last_voiced >= self.hangover_s:
            return self._close(now)
        return []

    def _close(self, now: float) -> list[Action]:
        """End the current utterance, discarding it if it was only a noise.

        The test is how much speech there was, not how long ago it started.
        Measuring the span instead let two clicks half a second apart through
        as an "utterance", and Whisper invents words when handed a second of
        near-silence - so a spurious transcript was being manufactured rather
        than merely leaking past.
        """
        frames = self.pending_frames
        voiced = self._voiced_frames
        self._first_voiced = self._last_voiced = None
        self.pending_frames = 0
        self._voiced_frames = 0
        if voiced * self.frame_s < self.min_speech_s:
            return []
        return [Transcribe(turn=self.turn, frames=frames, voiced_frames=voiced)]

    def _while_speaking(self, voiced: bool, now: float) -> list[Action]:
        """Watch for a real interruption without twitching at every noise."""
        if not voiced:
            self._voice_started = None
            if self._ducked:
                self._ducked = False
                return [Resume(turn=self.turn)]
            return []

        if self._voice_started is None:
            self._voice_started = now
        held = now - self._voice_started

        if held >= self.barge_in_s:
            turn = self.turn
            self.interrupt(now)
            return [StopPlayback(turn=turn)]

        if not self._ducked:
            self._ducked = True
            return [Duck(turn=self.turn)]
        return []

    # ----------------------------------------------------------- events

    def on_reply(self, clauses: list[str], turn: int, now: float) -> list[Action]:
        """Queue a reply, unless the turn it answers has been abandoned."""
        if turn != self.turn:
            return []  # they moved on; answering now would be answering the past
        return [Speak(turn=turn, text=c) for c in clauses]

    def on_playback_started(self, turn: int, now: float) -> list[Action]:
        """Note that its own voice is now in the room."""
        if turn == self.turn:
            self.speaking = True
        return []

    def on_playback_finished(self, turn: int, now: float) -> list[Action]:
        """Note that it has stopped, and start distrusting the microphone."""
        self.speaking = False
        self._playback_ended = now
        self._voice_started = None
        self._ducked = False
        return []

    def on_spoken(self, text: str, turn: int, now: float) -> None:
        """Record what actually reached the speaker, for the memory.

        Only this is written to history. Anything cut off by an interruption
        was never heard, and a pet that believes it said it will make no sense
        on the next turn.
        """
        self._spoken.setdefault(turn, []).append(text)

    def spoken_text(self, turn: int) -> str:
        """Return what was actually said out loud on a given turn."""
        return " ".join(self._spoken.get(turn, []))

    def interrupt(self, now: float) -> None:
        """Abandon the current turn. Everything stamped with it is now stale."""
        self.turn += 1
        self.speaking = False
        self._voice_started = None
        self._ducked = False
        self._first_voiced = self._last_voiced = None
        self.pending_frames = 0
        self._voiced_frames = 0
