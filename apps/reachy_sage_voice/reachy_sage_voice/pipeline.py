"""Reachy's voice, assembled: mic -> onset VAD -> STT -> Sage -> TTS -> speaker.

Services are the bench-chosen local ones, each behind an OpenAI-compatible URL
on this machine, in Docker (services/compose.yaml): Parakeet TDT for speech
to text on 8781, Kokoro for speech on 8782, captioned so an interruption
reports the exact words heard. Every URL
and the kernel token come from the environment, with no defaults, so nothing
here can reach the internet unless someone configures it to.

    SAGE_VOICE_KERNEL_URL    the kernel's device listener (protocol v1)
    SAGE_VOICE_KERNEL_TOKEN  this robot's token from the kernel's SAGE_DEVICES
    SAGE_VOICE_STT_URL       Parakeet, OpenAI-compatible (with or without ``/v1``)
    SAGE_VOICE_TTS_URL       Kokoro (``.../v1``; captions at ``/dev/captioned_speech``)
    SAGE_VOICE_STT_MODEL     default parakeet-tdt-0.6b-v2
    SAGE_VOICE_TTS_MODEL     default kokoro
    SAGE_VOICE_TTS_VOICE     default af_heart
    SAGE_VOICE_PRESENCE      1 to report seen/lost from the face tracker; default 0
    SAGE_VOICE_TURN_END_MS   silence that ends a turn; default 450, then
                             calibrated on the owner's voice (``turn_end.py``)
    SAGE_VOICE_TURN          silence (default): the VAD's silence ends a turn; smart:
                             Smart Turn v3 decides, after a 0.2 s VAD stop,
                             and the STT transcribes the whole turn once
    SAGE_VOICE_TURN_TIMEOUT_MS  smart only: silence that ends a turn Smart Turn
                             calls incomplete; default 1500
    SAGE_VOICE_MIC_CHANNEL   mean (default), 0 or 1 of the mic array's stereo
    SAGE_VOICE_RECORD_DIR    save each utterance's raw stereo there as WAV; default off
    SAGE_VOICE_LISTEN_CUES   1 for listening and thinking cues (listen_cues.py); default 0
    SAGE_VOICE_ADDRESSED     1 to answer only when addressed (addressed.py); default 0
    SAGE_VOICE_ALIASES       names besides Sage it answers to, comma-separated; default Reachy
    SAGE_VOICE_EXCHANGE_S    seconds after it spoke that need no name or face; default 8
"""

import asyncio
import logging
import os
import time
from collections.abc import Callable, Mapping

from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import UserStartedSpeakingFrame, UserStoppedSpeakingFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transports.base_transport import TransportParams
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_processor import UserTurnProcessor
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from reachy_sage_voice import clause, turn_end
from reachy_sage_voice.addressed import addressed_filter
from reachy_sage_voice.interruption import InterruptionTracker
from reachy_sage_voice.kernel import KernelClient, KernelUnavailableError, Session, manifest
from reachy_sage_voice.onset import build_speech_onset_silero, make_speech_onset_handler
from reachy_sage_voice.presence import watch_presence
from reachy_sage_voice.sage import BotSpeech, Playback, SageProcessor
from reachy_sage_voice.transport import (
    MIC_CHANNELS,
    ReachyInputTransport,
    ReachyMedia,
    ReachyOutputTransport,
)
from reachy_sage_voice.tts import KokoroTTSService

logger = logging.getLogger(__name__)

REQUIRED = ("SAGE_VOICE_KERNEL_URL", "SAGE_VOICE_KERNEL_TOKEN",
            "SAGE_VOICE_STT_URL", "SAGE_VOICE_TTS_URL")
#: One person, one mic: the speaker the interruption policy tracks.
SPEAKER = "you"
#: Local services check no key, but the OpenAI client insists on one.
_NO_KEY = "local"
#: What ends a user turn (SAGE_VOICE_TURN).
TURNS = ("silence", "smart")
#: The VAD stop Smart Turn is tuned for (Pipecat's default).
SMART_VAD_STOP_S = 0.2


def openai_base(url: str) -> str:
    """Return an OpenAI client base URL (``.../v1``), whether or not it had one."""
    url = url.rstrip("/")
    return url if url.endswith("/v1") else url + "/v1"


class ConfigError(Exception):
    """A required setting is missing from the environment."""


def smart_turn_analyzer(stop_secs: float) -> BaseTurnAnalyzer:
    """Smart Turn v3, the weights bundled with Pipecat, on CPU."""
    from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

    return LocalSmartTurnAnalyzerV3(params=SmartTurnParams(stop_secs=stop_secs))


def user_turns(analyzer: BaseTurnAnalyzer) -> UserTurnProcessor:
    """Pipecat's user turns, ended by ``analyzer`` alone. Barge-in stays with the onset policy.

    It runs before the STT, so the turn ends on the analyzer's verdict, not on a
    transcript (``wait_for_transcript=False``), and no transcript can start a turn.
    """
    return UserTurnProcessor(user_turn_strategies=UserTurnStrategies(
        start=[VADUserTurnStartStrategy(enable_interruptions=False)],
        stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=analyzer,
                                               wait_for_transcript=False)]))


class TurnSTTService(OpenAISTTService):
    """The segmented STT, segmenting on the user turn rather than on each VAD segment.

    A 0.2 s VAD stop cuts a turn into pieces too short to transcribe alone. Here
    the segment opens on ``UserStartedSpeakingFrame`` (with Pipecat's 1 s of
    pre-roll) and closes on ``UserStoppedSpeakingFrame``: one transcript per turn.
    ponytail: overrides SegmentedSTTService's two private segment hooks (Pipecat
    1.12, pinned); Pipecat has no turn-keyed segmented STT to use instead.
    """

    _in_turn = False

    async def _handle_user_started_speaking(self, frame):
        pass  # a VAD segment: the turn goes on

    async def _handle_user_stopped_speaking(self, frame):
        pass

    async def process_frame(self, frame, direction):
        """Segment on the user turn frames, then as any segmented STT."""
        await super().process_frame(frame, direction)
        if isinstance(frame, UserStartedSpeakingFrame):
            self._in_turn = True
            await SegmentedSTTService._handle_user_started_speaking(self, frame)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._in_turn = False
            await SegmentedSTTService._handle_user_stopped_speaking(self, frame)

    async def process_audio_frame(self, frame, direction):
        """Buffer as SegmentedSTTService does, keeping the whole turn, pauses included."""
        # STTService clears _user_speaking at every VAD stop, and Segmented trims on it.
        self._user_speaking = self._in_turn
        await super().process_audio_frame(frame, direction)


def build(media: ReachyMedia, play_emote: Callable, moves: Mapping,
          env: Mapping[str, str] = os.environ, play_sound: Callable | None = None,
          face: Callable[[], bool] | None = None) -> Pipeline:
    """Assemble the pipeline. Connects to nothing until it runs.

    ``moves`` is the declared vocabulary from the moves library (``load_moves``);
    ``face`` reads the face tracker, for SAGE_VOICE_ADDRESSED.
    """
    missing = [name for name in REQUIRED if not env.get(name)]
    if missing:
        raise ConfigError("set " + ", ".join(missing))
    mic_channel = env.get("SAGE_VOICE_MIC_CHANNEL", "mean")
    if mic_channel not in MIC_CHANNELS:
        raise ConfigError("SAGE_VOICE_MIC_CHANNEL must be one of " + ", ".join(MIC_CHANNELS))
    turn = env.get("SAGE_VOICE_TURN", "silence")
    if turn not in TURNS:
        raise ConfigError("SAGE_VOICE_TURN must be one of " + ", ".join(TURNS))
    smart = turn == "smart"

    tracker, playback = InterruptionTracker(), Playback()

    # Silence that ends a turn: an earlier voice bot's 450 ms until recordings of the owner set
    # it (turn_end.py), or Smart Turn (SAGE_VOICE_TURN=smart).
    # With Smart Turn, the VAD only marks pauses and the model ends the turn.
    turn_end_ms = int(env.get("SAGE_VOICE_TURN_END_MS", turn_end.DEFAULT_MS))
    stop_secs = SMART_VAD_STOP_S if smart else turn_end_ms / 1000
    analyzer = build_speech_onset_silero(VADParams(stop_secs=stop_secs))
    vad = VADProcessor(vad_analyzer=analyzer)
    # Pushed upstream from the VAD, the interruption reaches the pipeline
    # worker, which injects it at the source ahead of any queue.
    on_onset = make_speech_onset_handler(tracker, vad.push_frame)
    analyzer.set_onset_handlers(
        on_detected=lambda: on_onset(SPEAKER),
        on_cleared=lambda: tracker.user_offset(time.monotonic(), SPEAKER))

    stt = (TurnSTTService if smart else OpenAISTTService)(
        api_key=_NO_KEY, base_url=openai_base(env["SAGE_VOICE_STT_URL"]),
        settings=OpenAISTTService.Settings(
            model=env.get("SAGE_VOICE_STT_MODEL", "parakeet-tdt-0.6b-v2")))

    tts = KokoroTTSService(base_url=env["SAGE_VOICE_TTS_URL"],
                           voice=env.get("SAGE_VOICE_TTS_VOICE", "af_heart"),
                           model=env.get("SAGE_VOICE_TTS_MODEL", "kokoro"))
    clause.install(tts)

    session = Session(KernelClient(env["SAGE_VOICE_KERNEL_URL"], env["SAGE_VOICE_KERNEL_TOKEN"]),
                      manifest(moves))
    gate = addressed_filter(env, face)
    return Pipeline([
        ReachyInputTransport(media, TransportParams(audio_in_enabled=True,
                                                    audio_in_sample_rate=16000),
                             mic_channel, env.get("SAGE_VOICE_RECORD_DIR") or None),
        vad,
        *([user_turns(smart_turn_analyzer(
            int(env.get("SAGE_VOICE_TURN_TIMEOUT_MS", 1500)) / 1000))] if smart else []),
        stt,
        *([gate] if gate else []),
        SageProcessor(session, play_emote, playback, play_sound=play_sound),
        tts,
        ReachyOutputTransport(media, TransportParams(
            audio_out_enabled=True, audio_out_sample_rate=media.get_output_audio_samplerate())),
        BotSpeech(tracker, playback),
    ])


async def report_presence(session, event: dict) -> None:
    """Send one presence edge; an unreachable kernel costs the edge, never the voice."""
    try:
        await session.presence(event["state"], event["confidence"])
    except KernelUnavailableError as error:
        logger.warning("could not report presence %s: %s", event["state"], error)


async def run(media: ReachyMedia, play_emote: Callable, moves: Mapping,
              env: Mapping[str, str] = os.environ, play_sound: Callable | None = None,
              presence: Callable[[], bool] | None = None, observers: list | None = None,
              face: Callable[[], bool] | None = None) -> None:
    """Run the voice until cancelled. Always on: no idle timeout.

    ``presence`` reads the face tracker; when given, seen/lost edges go to the kernel.
    ``observers`` watch the frames (``listen_cues.py``).
    """
    runner = WorkerRunner()
    pipeline = build(media, play_emote, moves, env, play_sound, face)
    await runner.add_workers(PipelineWorker(pipeline, idle_timeout_secs=None,
                                                observers=observers))
    if presence is None:
        await runner.run()
        return
    session = Session(KernelClient(env["SAGE_VOICE_KERNEL_URL"], env["SAGE_VOICE_KERNEL_TOKEN"]),
                      manifest(moves))
    watch = asyncio.ensure_future(
        watch_presence(presence, lambda event: report_presence(session, event)))
    try:
        await runner.run()
    finally:
        watch.cancel()
