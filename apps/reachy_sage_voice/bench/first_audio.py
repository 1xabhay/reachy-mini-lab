"""First audio: "Generating TTS" (a reply's first ``run_tts``) to "Bot started speaking".

Replays the 8 replies of live run 3 through this app's own Kokoro service and output
transport, against a running Kokoro (``SAGE_VOICE_TTS_URL``), with a fake robot and
pacing off. No LLM and no kernel: each reply's tokens arrive at once, so the gap is TTS
plus transport. Results go in sage-kernel bench/LEDGER.md, "Voice first audio".

    make bench-first-audio            # REPS=5 by default
"""

import asyncio
import logging
import os
import re
import statistics
import sys
import time

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transports.base_transport import TransportParams
from pipecat.utils.text.base_text_aggregator import Aggregation, AggregationType
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator

from reachy_sage_voice.clause import CLAUSE_ENDINGS, ClauseTextAggregator
from reachy_sage_voice.transport import ReachyOutputTransport
from reachy_sage_voice.tts import KokoroTTSService

REPLIES = (
    "Yes, I'm here! Let's talk.", "Great, let's improve together.",
    "I'm looking around to see what's happening.", "I'm trying to see in all directions.",
    "Yes, I can see you.", "I can see you because I'm here. I see you.",
    "Yes, that's true, plus two.", "2 plus 2 is 4.",
)


class Cut(SimpleTextAggregator):
    """A comparison rule: clauses of ``min_words``, or a first chunk of ``first_words``."""

    def __init__(self, min_words=4, first_words=None, **kwargs):
        """Cut clauses at ``min_words``; optionally the first chunk at ``first_words``."""
        super().__init__(aggregation_type=AggregationType.SENTENCE, **kwargs)
        self.min_words, self.first_words, self._first = min_words, first_words, True

    async def aggregate(self, text):
        """Yield sentences and clauses as the rule allows."""
        for char in text:
            async for sentence in super().aggregate(char):
                self._first = False
                yield sentence
            if char != " ":
                continue
            n = len(self._text.split())
            if ((self._text[-2:-1] in CLAUSE_ENDINGS and n >= self.min_words)
                    or (self._first and self.first_words and n >= self.first_words)):
                chunk, self._text, self._first = self._text.strip(), "", False
                yield Aggregation(text=chunk, type=AggregationType.SENTENCE)

    async def flush(self):
        """Start the next reply afresh."""
        self._first = True
        return await super().flush()


VARIANTS = {
    "app default (ClauseTextAggregator)":
        lambda: ClauseTextAggregator(aggregation_type=AggregationType.SENTENCE),
    "clauses of 4+ words throughout": lambda: Cut(4),
    "first 3 words, any punctuation": lambda: Cut(4, first_words=3),
}


class _Timed(KokoroTTSService):
    started = None

    async def run_tts(self, text, context_id):
        if _Timed.started is None:
            _Timed.started = time.monotonic()
        async for frame in super().run_tts(text, context_id):
            yield frame


class _Probe(FrameProcessor):
    audio = None

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, BotStartedSpeakingFrame) and _Probe.audio is None:
            _Probe.audio = time.monotonic()
        await self.push_frame(frame, direction)


class _Robot:
    def start_playing(self): ...
    def stop_playing(self): ...
    def push_audio_sample(self, data): ...
    def get_output_audio_samplerate(self): return 16000


async def _no_wait(_seconds):  # pacing does not change the gap; skip real playback time
    await asyncio.sleep(0)


async def first_audio_ms(url: str, reply: str, aggregator) -> float | None:
    """Milliseconds from the first TTS request of ``reply`` to the first audio out."""
    _Timed.started = _Probe.audio = None
    tts = _Timed(base_url=url, voice=os.environ.get("SAGE_VOICE_TTS_VOICE", "af_heart"))
    tts._text_aggregator = aggregator
    out = ReachyOutputTransport(_Robot(), TransportParams(
        audio_out_enabled=True, audio_out_sample_rate=16000), sleep=_no_wait)
    tokens = re.findall(r" ?[\w']+|[^\w\s]", reply)
    await run_test(Pipeline([tts, out, _Probe()]), frames_to_send=[
        LLMFullResponseStartFrame(), *[LLMTextFrame(t) for t in tokens],
        LLMFullResponseEndFrame(), SleepFrame(1.5)])
    if _Probe.audio is None or _Timed.started is None:
        return None
    return (_Probe.audio - _Timed.started) * 1000


async def main(url: str, reps: int) -> None:
    """Print p50 and p95 per variant."""
    logging.disable(logging.CRITICAL)
    from loguru import logger

    logger.remove()
    await first_audio_ms(url, "Warm up.", Cut(4))
    for name, make in VARIANTS.items():
        ms = [m for _ in range(reps) for r in REPLIES
              if (m := await first_audio_ms(url, r, make())) is not None]
        p95 = statistics.quantiles(ms, n=20)[18]
        print(f"{name}: n={len(ms)} p50={statistics.median(ms):.0f} ms p95={p95:.0f} ms "
              f"min={min(ms):.0f} max={max(ms):.0f}", flush=True)


if __name__ == "__main__":
    url = os.environ.get("SAGE_VOICE_TTS_URL")
    if not url:
        sys.exit("set SAGE_VOICE_TTS_URL to a running Kokoro (e.g. http://127.0.0.1:8782)")
    asyncio.run(main(url, int(os.environ.get("REPS", "5"))))
