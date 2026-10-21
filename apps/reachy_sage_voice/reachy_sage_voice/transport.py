"""Pipecat input and output over ``ReachyMini.media``.

Input (mic -> pipeline)
    After ``start_recording()``, a task polls ``media.get_audio_sample()``
    (float32, ``(n, channels)``, 16 kHz on Reachy Mini; None when nothing is
    ready), mixes it to mono int16 (or keeps one channel: ``mic_channel``) and
    pushes ``InputAudioRawFrame``. Frames go
    in as the robot delivers them: the VAD and the segmented STT buffer to the
    sizes they need, so nothing here reframes to 320 samples.

    With ``record_dir``, each VAD segment's raw stereo, with the same 1 s
    before it that Pipecat's segmented STT keeps, is saved there as a WAV for
    the offline channel A/B (``bench/mic_channels.py``). It is the owner's
    voice: a local directory, never committed.

Output (pipeline -> speaker)
    Pipecat resamples TTS audio (Kokoro, 24 kHz) to the robot's output rate,
    set from ``media.get_output_audio_samplerate()``; each chunk goes to
    ``media.push_audio_sample()`` as float32 after ``start_playing()``. On an
    interruption Pipecat drops what is queued here; audio already handed to the
    robot's WebRTC/GStreamer buffer still plays. That tail is the talk-over
    this transport cannot remove: measure it on the robot.
    ``media.play_sound(path)`` stays for the robot's own recorded sounds,
    outside the TTS stream.

Echo
    The mic hears the speaker late, through WebRTC. Hardware echo
    cancellation is the first defence; ``speech.is_self_echo`` against recent
    bot speech is the backstop. Measure the lag on the robot before trusting
    either.
"""

import asyncio
import time
import wave
from collections import deque
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    InputAudioRawFrame,
    StartFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams

#: How long to wait before asking an empty mic again.
_MIC_IDLE_S = 0.005

#: How far ahead of real playback audio may be handed to the robot. Writes wait
#: out the rest, so Pipecat's "bot speaking" matches what the speaker plays.
#: ponytail: a guess at 150 ms. It must be measured live against the robot's
#: buffer depth: too small and the speaker starves between chunks, too large
#: and an interruption still lets that much of the reply play.
PLAYBACK_LEAD_S = 0.15

#: The XVF3800's stereo is two processed outputs; which carries the ASR beam is
#: undocumented, so the A/B decides. "mean" mixes both.
MIC_CHANNELS = ("mean", "0", "1")
#: Audio kept before a segment starts, as SegmentedSTTService keeps it.
PRE_ROLL_S = 1.0


class ReachyMedia(Protocol):
    """The slice of ``ReachyMini.media`` this transport uses, and nothing more."""

    def start_recording(self) -> None: ...  # noqa: D102
    def stop_recording(self) -> None: ...  # noqa: D102
    def get_audio_sample(self) -> npt.NDArray[np.float32] | None: ...  # noqa: D102
    def start_playing(self) -> None: ...  # noqa: D102
    def stop_playing(self) -> None: ...  # noqa: D102
    def push_audio_sample(self, data: npt.NDArray[np.float32]) -> None: ...  # noqa: D102
    def get_output_audio_samplerate(self) -> int: ...  # noqa: D102
    def play_sound(self, sound_file: str) -> None: ...  # noqa: D102


def _pcm16(samples: npt.NDArray[np.float32]) -> bytes:
    return (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def to_pcm16(samples: npt.NDArray[np.float32], channel: str = "mean") -> bytes:
    """Mic samples, any channel count, as mono int16 PCM: the mean or one channel."""
    if samples.ndim > 1:
        samples = samples.mean(axis=1) if channel == "mean" else samples[:, int(channel)]
    return _pcm16(samples)


class UtteranceRecorder:
    """Raw mic chunks from ``PRE_ROLL_S`` before a VAD start to its stop, saved as WAV."""

    def __init__(self, directory: str, sample_rate: int) -> None:
        """Save into ``directory``, created if missing."""
        self._dir, self._rate = Path(directory), sample_rate
        self._dir.mkdir(parents=True, exist_ok=True)
        self._chunks: deque[npt.NDArray[np.float32]] = deque()
        self._samples = 0
        self._speaking = False

    def feed(self, samples: npt.NDArray[np.float32]) -> None:
        """Keep a chunk; between segments only the last ``PRE_ROLL_S``."""
        self._chunks.append(samples)
        self._samples += len(samples)
        keep = PRE_ROLL_S * self._rate
        while not self._speaking and self._samples - len(self._chunks[0]) >= keep:
            self._samples -= len(self._chunks.popleft())

    def start(self) -> None:
        """Keep everything from here until the segment stops."""
        self._speaking = True

    def stop(self) -> tuple[Path, bytes] | None:
        """End the segment; return where it goes and its PCM, to be written off the loop."""
        if not (self._speaking and self._chunks):
            self._speaking = False
            return None
        self._speaking = False
        audio = np.concatenate(self._chunks)
        self._chunks.clear()
        self._samples = 0
        path = self._dir / time.strftime(f"utt-%Y%m%d-%H%M%S-{time.time_ns() % 10**9:09d}.wav")
        return path, _pcm16(audio)

    def write(self, path: Path, pcm: bytes, channels: int) -> None:
        """Write one segment as 16-bit WAV."""
        with wave.open(str(path), "wb") as w:
            w.setnchannels(channels)
            w.setsampwidth(2)
            w.setframerate(self._rate)
            w.writeframes(pcm)


def to_float32(pcm: bytes) -> npt.NDArray[np.float32]:
    """Int16 PCM as float32 in [-1, 1), as the robot's speaker takes it."""
    return (np.frombuffer(pcm, dtype=np.int16) / 32768.0).astype(np.float32)


class ReachyInputTransport(BaseInputTransport):
    """The robot's microphone as Pipecat audio input."""

    def __init__(self, media: ReachyMedia, params: TransportParams, mic_channel: str = "mean",
                 record_dir: str | None = None, **kwargs):
        """Read ``mic_channel`` (one of ``MIC_CHANNELS``) from ``media``; maybe record."""
        super().__init__(params, **kwargs)
        self._media, self._mic_channel = media, mic_channel
        self._record_dir = record_dir
        self._recorder: UtteranceRecorder | None = None
        self._channels = 1
        self._reader: asyncio.Task | None = None

    async def start(self, frame: StartFrame):
        """Start recording and reading."""
        await super().start(frame)
        if self._record_dir:
            self._recorder = UtteranceRecorder(self._record_dir, self.sample_rate)
        self._media.start_recording()
        self._reader = self.create_task(self._read())
        await self.set_transport_ready(frame)

    async def stop(self, frame: EndFrame):
        """Stop reading and recording."""
        await super().stop(frame)
        await self._halt()

    async def cancel(self, frame: CancelFrame):
        """Stop reading and recording."""
        await super().cancel(frame)
        await self._halt()

    async def process_frame(self, frame, direction):
        """Mark segments for the recorder: the VAD broadcasts its frames upstream too."""
        await super().process_frame(frame, direction)
        if self._recorder and isinstance(frame, VADUserStartedSpeakingFrame):
            self._recorder.start()
        elif self._recorder and isinstance(frame, VADUserStoppedSpeakingFrame):
            segment = self._recorder.stop()
            if segment:
                await asyncio.to_thread(self._recorder.write, *segment, self._channels)

    async def _halt(self) -> None:
        if self._reader:
            await self.cancel_task(self._reader)
            self._reader = None
            self._media.stop_recording()

    async def _read(self) -> None:
        # ponytail: assumes get_audio_sample() does not block (it pulls from a
        # GStreamer appsink); move it to a thread if the robot shows otherwise.
        while True:
            samples = self._media.get_audio_sample()
            if samples is None:
                await asyncio.sleep(_MIC_IDLE_S)
                continue
            if self._recorder:
                self._recorder.feed(samples)
                self._channels = samples.shape[1] if samples.ndim > 1 else 1
            await self.push_audio_frame(InputAudioRawFrame(
                audio=to_pcm16(samples, self._mic_channel), sample_rate=self.sample_rate,
                num_channels=1))
            await asyncio.sleep(0)  # a mic that always has audio must not starve the loop


class ReachyOutputTransport(BaseOutputTransport):
    """The robot's speaker as Pipecat audio output."""

    def __init__(self, media: ReachyMedia, params: TransportParams,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep, **kwargs):
        """Play through ``media``, paced by ``clock``."""
        super().__init__(params, **kwargs)
        self._media, self._clock, self._sleep = media, clock, sleep
        self._played_until = 0.0  # when the audio handed over so far ends

    async def start(self, frame: StartFrame):
        """Start playing."""
        await super().start(frame)
        self._media.start_playing()
        await self.set_transport_ready(frame)

    async def stop(self, frame: EndFrame):
        """Stop playing."""
        await super().stop(frame)
        self._media.stop_playing()

    async def cancel(self, frame: CancelFrame):
        """Stop playing."""
        await super().cancel(frame)
        self._media.stop_playing()

    async def write_audio_frame(self, frame) -> bool:
        """Hand one chunk of audio to the robot, then wait until it is nearly played.

        The robot buffers whatever it is given and returns at once; waiting here
        is what makes Pipecat's queue, its bot-speaking state and its
        interruptions follow the real speaker.
        """
        samples = to_float32(frame.audio)
        now = self._clock()
        self._played_until = max(now, self._played_until) + len(samples) / frame.sample_rate
        self._media.push_audio_sample(samples)
        wait = self._played_until - PLAYBACK_LEAD_S - now
        if wait > 0:
            await self._sleep(wait)
        return True
