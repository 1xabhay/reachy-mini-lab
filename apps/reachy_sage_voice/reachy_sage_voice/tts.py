"""Kokoro, with captions: each word timed, so an interruption knows what was heard.

The services/tts app serves ``POST /dev/captioned_speech`` (Kokoro-FastAPI's
path and shape): NDJSON, one sentence per line, ``audio`` as base64 16-bit
24 kHz PCM and ``timestamps`` as ``[{word, start_time, end_time}]`` timed from
the start of the request. Each word becomes a TTSTextFrame with a presentation
time, which the output transport releases as that word is played, so Playback
records exactly the words heard up to an interruption.

Without captions (the endpoint answers 404) the plain ``/v1/audio/speech``
speaks instead, and each clause counts whole once its audio has played.
"""

import base64
import json
import logging

import httpx
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    InterruptionFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService

from reachy_sage_voice.voice import VoiceStyleFrame

logger = logging.getLogger(__name__)

RATE = 24000
_BYTES_PER_S = RATE * 2
#: The captioned endpoint is absent, not failing: stop asking for it.
_NO_CAPTIONS = (404, 405, 501)


class _NoCaptionsError(Exception):
    pass


class KokoroTTSService(TTSService):
    """Kokoro over HTTP, word-timed when its server can caption."""

    def __init__(self, *, base_url: str, voice: str = "af_heart", model: str = "kokoro",
                 **kwargs):
        """Speak through the Kokoro server at ``base_url`` (``.../v1`` or its root)."""
        super().__init__(push_text_frames=False, push_stop_frames=True, push_start_frame=True,
                         sample_rate=RATE, settings=TTSSettings(model=model, voice=voice),
                         **kwargs)
        self._root = base_url.rstrip("/").removesuffix("/v1")
        self._captions = True
        self._offset_s = 0.0  # audio already produced in this response
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0))
        self._style: dict = {}  # voice_style settings for the current reply

    def can_generate_metrics(self) -> bool:
        """Report TTFB like any TTS service."""
        return True

    async def cleanup(self):
        """Close the HTTP client."""
        await super().cleanup()
        await self._client.aclose()

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push a frame; a response ending or cut off restarts word timing."""
        await super().push_frame(frame, direction)
        if isinstance(frame, (InterruptionFrame, TTSStoppedFrame)):
            self._offset_s, self._style = 0.0, {}

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Take a voice style for the rest of this reply; everything else as usual."""
        if isinstance(frame, VoiceStyleFrame):
            self._style = dict(frame.settings)
        await super().process_frame(frame, direction)

    async def run_tts(self, text: str, context_id: str):
        """Synthesize one clause, captioned if the server can."""
        body = {"input": text, "voice": self._settings.voice, "model": self._settings.model,
                "response_format": "pcm", "speed": self._style.get("speed", 1.0)}
        try:
            if self._captions:
                try:
                    async for frame in self._captioned(body, context_id):
                        yield frame
                    return
                except _NoCaptionsError:
                    logger.warning("Kokoro has no captioned speech: whole clauses from now on")
                    self._captions = False
            async for frame in self._plain(body, text, context_id):
                yield frame
        except httpx.HTTPError as error:
            yield ErrorFrame(error=f"Kokoro unreachable: {error}")

    async def _captioned(self, body: dict, context_id: str):
        url = self._root + "/dev/captioned_speech"
        async with self._client.stream("POST", url, json={**body, "stream": True}) as response:
            if response.status_code in _NO_CAPTIONS:
                raise _NoCaptionsError
            if response.status_code != 200:
                yield ErrorFrame(error=f"Kokoro captioned speech: {response.status_code}")
                return
            await self.start_tts_usage_metrics(body["input"])
            produced = 0
            async for raw in response.aiter_lines():
                if not raw.strip():
                    continue
                sentence = json.loads(raw)
                audio = base64.b64decode(sentence["audio"])
                await self.stop_ttfb_metrics()
                yield TTSAudioRawFrame(audio, RATE, 1, context_id=context_id)
                produced += len(audio)
                await self.add_word_timestamps(
                    [(w["word"], self._offset_s + w["start_time"])
                     for w in sentence["timestamps"]], context_id)
            self._offset_s += produced / _BYTES_PER_S

    async def _plain(self, body: dict, text: str, context_id: str):
        url = self._root + "/v1/audio/speech"
        async with self._client.stream("POST", url, json=body) as response:
            if response.status_code != 200:
                yield ErrorFrame(error=f"Kokoro speech: {response.status_code}")
                return
            await self.start_tts_usage_metrics(text)
            held, produced = b"", 0
            async for chunk in response.aiter_bytes():
                held += chunk
                whole = len(held) - len(held) % 2  # never split a 16-bit sample
                if whole:
                    await self.stop_ttfb_metrics()
                    yield TTSAudioRawFrame(held[:whole], RATE, 1, context_id=context_id)
                    produced += whole
                    held = held[whole:]
        # The whole clause as one "word" timed at its end: released once it has played.
        self._offset_s += produced / _BYTES_PER_S
        await self.add_word_timestamps([(text.strip(), self._offset_s)], context_id)
