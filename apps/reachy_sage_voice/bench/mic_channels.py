"""Mic channel A/B: transcribe each recorded utterance as channel 0, channel 1 and their mean.

Record a live run with ``SAGE_VOICE_RECORD_DIR=<dir>`` (raw stereo per VAD segment,
``transport.py``), then, against the Parakeet service on :8781 (``make -C services up``):

    uv run python bench/mic_channels.py <dir>

Each mono variant is built exactly as the pipeline builds it (``to_pcm16``) and padded
with the same 0.5 s of trailing silence Pipecat's segmented STT adds. The recordings are
the owner's voice: keep them local, never commit them.
"""

import io
import os
import sys
import wave
from pathlib import Path

import numpy as np
from openai import OpenAI

from reachy_sage_voice.pipeline import openai_base
from reachy_sage_voice.transport import MIC_CHANNELS, to_pcm16

TRAILING_SILENCE_S = 0.5  # SegmentedSTTService's default


def variants(path: Path) -> dict[str, bytes]:
    """One recorded stereo utterance as a mono WAV per channel choice."""
    with wave.open(str(path)) as w:
        rate, channels = w.getframerate(), w.getnchannels()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    samples = (pcm.reshape(-1, channels) / 32768.0).astype(np.float32)
    out = {}
    for channel in MIC_CHANNELS:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(to_pcm16(samples, channel) + bytes(int(rate * TRAILING_SILENCE_S) * 2))
        out[channel] = buf.getvalue()
    return out


def main(directory: str) -> None:
    """Print each utterance's three transcripts, marking those where the channels disagree."""
    client = OpenAI(api_key="local",
                    base_url=openai_base(os.environ.get("SAGE_VOICE_STT_URL",
                                                        "http://127.0.0.1:8781")))
    model = os.environ.get("SAGE_VOICE_STT_MODEL", "parakeet-tdt-0.6b-v2")
    paths = sorted(Path(directory).glob("*.wav"))
    differ = 0
    for path in paths:
        texts = {channel: client.audio.transcriptions.create(
                     model=model, file=(path.name, wav)).text.strip()
                 for channel, wav in variants(path).items()}
        mark = "  " if len(set(texts.values())) == 1 else "* "
        differ += mark == "* "
        print(mark + path.name)
        for channel, text in texts.items():
            print(f"    {channel:>4}: {text}")
    print(f"{differ} of {len(paths)} utterances differ between channels")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python bench/mic_channels.py <SAGE_VOICE_RECORD_DIR>")
    main(sys.argv[1])
