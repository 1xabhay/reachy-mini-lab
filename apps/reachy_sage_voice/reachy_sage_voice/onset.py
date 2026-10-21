"""Barge-in on the earliest edge the VAD will commit to.

A VADAnalyzer moves QUIET -> STARTING -> SPEAKING. Pipecat's VADProcessor only
reports SPEAKING and QUIET, so STARTING, the first evidence someone is talking,
is discarded. An earlier voice bot measured that edge at 160 ms (5 frames of 32 ms) ahead of
SPEAKING over real speech (``speech_onset_vad.py``). This keeps one analyzer
with two consumers: interruption reads STARTING, segmentation keeps SPEAKING /
QUIET with its thresholds untouched.

Raw audio energy was rejected as a trigger: Reachy's mic hears its own speaker,
and the confidence and volume gates are what keep that from counting as speech.
"""

import inspect
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from pipecat.audio.vad.vad_analyzer import VADState
from pipecat.frames.frames import Frame, InterruptionWorkerFrame
from pipecat.processors.frame_processor import FrameDirection

from reachy_sage_voice.interruption import InterruptionTracker

logger = logging.getLogger(__name__)


def speech_edge(prev: VADState, cur: VADState) -> str | None:
    """Classify a VAD state change: "detected", "cleared", or None.

    Any move off QUIET is one onset, even straight to SPEAKING (one call can
    consume several frames). Any return to QUIET clears, even from STARTING: a
    false start must be retracted. STARTING -> SPEAKING is not a new onset.
    """
    if prev == cur:
        return None
    if prev == VADState.QUIET:
        return "detected"
    if cur == VADState.QUIET:
        return "cleared"
    return None


class SpeechOnsetMixin:
    """Reports onset and offset from a VADAnalyzer without changing what it returns.

    Mix in ahead of the analyzer class. Handlers take no arguments and may be async.
    """

    _on_detected: Callable[[], Any] | None = None
    _on_cleared: Callable[[], Any] | None = None

    def set_onset_handlers(self, *, on_detected: Callable[[], Any] | None = None,
                           on_cleared: Callable[[], Any] | None = None) -> None:
        """Set what runs on the onset and offset edges."""
        self._on_detected = on_detected
        self._on_cleared = on_cleared

    async def analyze_audio(self, buffer: bytes) -> VADState:
        """Run the analyzer, then fire a handler if this frame crossed an edge."""
        prev = self._vad_state
        state = await super().analyze_audio(buffer)
        edge = speech_edge(prev, state)
        if edge:
            await _fire(self._on_detected if edge == "detected" else self._on_cleared, edge)
        return state


async def _fire(handler: Callable[[], Any] | None, label: str) -> None:
    # On the live audio path: a bad handler costs one interruption, never the session.
    if handler is None:
        return
    try:
        result = handler()
        if inspect.isawaitable(result):
            await result
    except Exception:
        logger.exception("speech onset: %s handler failed", label)


def build_speech_onset_silero(params) -> Any:
    """Return a SileroVADAnalyzer that also reports the STARTING edge."""
    from pipecat.audio.vad.silero import SileroVADAnalyzer

    class SpeechOnsetSileroVAD(SpeechOnsetMixin, SileroVADAnalyzer):
        pass

    return SpeechOnsetSileroVAD(params=params)


def make_speech_onset_handler(
    tracker: InterruptionTracker,
    enqueue_frame: Callable[[Frame, FrameDirection], Awaitable[None]],
    now: Callable[[], float] = time.monotonic,
) -> Callable[[str], Awaitable[None]]:
    """Return the onset handler that makes the bot yield when the policy says so.

    The interruption goes upstream as an ``InterruptionWorkerFrame`` (the 1.4+
    name for an earlier voice bot's ``InterruptionTaskFrame``): the pipeline worker turns it
    into an ``InterruptionFrame`` injected at the source, bypassing the queue a
    downstream frame would wait in. ``enqueue_frame`` is ``PipelineTask.queue_frame``.
    """

    async def on_speech_onset(sid: str) -> None:
        if not tracker.user_onset(now(), sid):
            return
        try:
            await enqueue_frame(InterruptionWorkerFrame(), FrameDirection.UPSTREAM)
        except Exception:
            logger.exception("speech onset: could not interrupt the bot for %s", sid)

    return on_speech_onset
