"""Barge-in on the VAD's QUIET -> STARTING edge, several frames before SPEAKING.

The analyzer here is Pipecat's own VADAnalyzer state machine with a scripted
confidence, so the edge under test is the real one, not a stand-in.
"""

import asyncio

from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams, VADState
from pipecat.frames.frames import InterruptionWorkerFrame
from pipecat.processors.frame_processor import FrameDirection

from reachy_sage_voice.interruption import InterruptionTracker
from reachy_sage_voice.onset import (
    SpeechOnsetMixin,
    build_speech_onset_silero,
    make_speech_onset_handler,
    speech_edge,
)

FRAME = 320  # samples per Reachy mic frame at 16 kHz
LOUD = (b"\xff\x3f" * FRAME)  # int16 ~0.5 full scale, over min_volume


class ScriptedVAD(VADAnalyzer):
    def __init__(self, confidences):
        super().__init__(sample_rate=16000, params=VADParams(start_secs=0.2, min_volume=0.0))
        self.set_sample_rate(16000)
        self._confidences = list(confidences)

    def num_frames_required(self) -> int:
        return FRAME

    def voice_confidence(self, buffer: bytes) -> float:
        return self._confidences.pop(0) if self._confidences else 0.0


class OnsetVAD(SpeechOnsetMixin, ScriptedVAD):
    pass


def run_frames(vad, n):
    async def go():
        return [await vad.analyze_audio(LOUD) for _ in range(n)]
    return asyncio.run(go())


def test_edges():
    assert speech_edge(VADState.QUIET, VADState.STARTING) == "detected"
    assert speech_edge(VADState.QUIET, VADState.SPEAKING) == "detected"
    assert speech_edge(VADState.STARTING, VADState.SPEAKING) is None
    assert speech_edge(VADState.SPEAKING, VADState.STOPPING) is None
    assert speech_edge(VADState.STARTING, VADState.QUIET) == "cleared"
    assert speech_edge(VADState.STOPPING, VADState.QUIET) == "cleared"
    for s in VADState:
        assert speech_edge(s, s) is None


def test_onset_fires_on_starting_well_before_speaking():
    vad = OnsetVAD([1.0] * 20)
    fired_at: list[int] = []
    states: list[VADState] = []
    vad.set_onset_handlers(on_detected=lambda: fired_at.append(len(states)))

    async def go():
        for _ in range(20):
            states.append(await vad.analyze_audio(LOUD))
    asyncio.run(go())

    first_speaking = states.index(VADState.SPEAKING)
    assert fired_at == [0]  # once, on the very first voiced frame
    assert first_speaking * FRAME / 16 >= 160  # SPEAKING is >= 160 ms later


def test_observing_does_not_change_the_states_the_stt_reads():
    script = [1.0] * 15 + [0.0] * 20
    plain, observed = ScriptedVAD(script), OnsetVAD(script)
    observed.set_onset_handlers(on_detected=lambda: None, on_cleared=lambda: None)
    assert run_frames(plain, 35) == run_frames(observed, 35)


def test_a_false_start_is_retracted():
    vad = OnsetVAD([1.0, 1.0, 0.0])
    seen: list[str] = []
    vad.set_onset_handlers(on_detected=lambda: seen.append("on"),
                           on_cleared=lambda: seen.append("off"))
    run_frames(vad, 3)
    assert seen == ["on", "off"]


def test_a_failing_handler_costs_an_interruption_not_the_audio_path():
    vad = OnsetVAD([1.0])

    def boom():
        raise RuntimeError("bad handler")
    vad.set_onset_handlers(on_detected=boom)
    assert run_frames(vad, 1) == [VADState.STARTING]


def test_onset_while_the_bot_talks_pushes_one_interruption_upstream():
    tracker = InterruptionTracker()
    tracker.bot_started(10.0)
    pushed = []

    async def enqueue(frame, direction):
        pushed.append((type(frame), direction))

    clock = iter([11.0, 11.5])
    handler = make_speech_onset_handler(tracker, enqueue, now=lambda: next(clock))
    asyncio.run(handler("you"))
    asyncio.run(handler("you"))
    assert pushed == [(InterruptionWorkerFrame, FrameDirection.UPSTREAM)]


def test_onset_with_the_bot_silent_interrupts_nothing():
    pushed = []

    async def enqueue(frame, direction):
        pushed.append(frame)

    handler = make_speech_onset_handler(InterruptionTracker(), enqueue, now=lambda: 1.0)
    asyncio.run(handler("you"))
    assert pushed == []


def test_a_broken_enqueue_never_raises_into_the_audio_path():
    tracker = InterruptionTracker()
    tracker.bot_started(0.0)

    async def enqueue(frame, direction):
        raise RuntimeError("pipeline gone")

    asyncio.run(make_speech_onset_handler(tracker, enqueue, now=lambda: 1.0)("you"))


def test_real_silero_runs_on_the_overridden_onnxruntime():
    """The onnxruntime override (1.27, Pipecat asks ~=1.24) must not break Silero."""
    import onnxruntime

    vad = build_speech_onset_silero(VADParams())
    vad.set_sample_rate(16000)
    states = run_frames(vad, 50)  # one second of 20 ms frames
    assert onnxruntime.__version__ == "1.27.0"
    assert set(states) <= set(VADState)
