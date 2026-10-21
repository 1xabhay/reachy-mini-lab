"""Text to speech: Kokoro 82M (int8, ONNX, CPU) behind OpenAI's speech endpoint.

Pipecat's OpenAITTSService posts JSON to POST /v1/audio/speech with `input`, `model`, `voice`,
`response_format: "pcm"` and optionally `speed` and `instructions`, then reads the body as a
stream of raw 16-bit little-endian mono PCM at 24 kHz. Here the body streams one sentence at a
time, so the first sentence plays while the rest is synthesised (measured: first audio about
0.21 s instead of 0.38 s for a two-sentence reply, sage-kernel bench/LEDGER.md).

POST /dev/captioned_speech (Kokoro-FastAPI's path and shape) adds when each word is spoken, so
an interrupted voice knows exactly what was heard. It runs the timestamped export of the same
int8 weights, whose extra `durations` output costs about 1 ms per sentence (bench/LEDGER.md).
"""
import base64
import glob
import json
import os
import re
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

RATE = 24000
REPO = "onnx-community/Kokoro-82M-v1.0-ONNX"
TIMED_REPO = "onnx-community/Kokoro-82M-v1.0-ONNX-timestamped"  # same weights, plus phoneme durations
WEIGHTS = "onnx/model_quantized.onnx"  # int8: half the latency of fp32 on an M2 Pro
DIR = os.environ.get("KOKORO_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".local/kokoro"))
_kokoro = _timed = None


def fetch(directory=DIR):
    """Download the weights and every voice, and pack the voices into the npz kokoro-onnx reads."""
    from huggingface_hub import snapshot_download

    snapshot_download(REPO, allow_patterns=[WEIGHTS, "voices/*.bin"], local_dir=directory)
    snapshot_download(TIMED_REPO, allow_patterns=[WEIGHTS], local_dir=os.path.join(directory, "timed"))
    voices = {os.path.basename(p)[:-4]: np.fromfile(p, dtype=np.float32).reshape(-1, 1, 256)
              for p in glob.glob(os.path.join(directory, "voices/*.bin"))}
    np.savez(os.path.join(directory, "voices.npz"), **voices)


def load(directory=DIR):
    """(plain, timed). Two sessions: with timings on, kokoro-onnx also pauses at every comma, which
    would change the plain endpoint's audio."""
    from kokoro_onnx import Kokoro

    if not os.path.exists(os.path.join(directory, "timed", WEIGHTS)):
        fetch(directory)
    voices = os.path.join(directory, "voices.npz")
    timed = Kokoro(os.path.join(directory, "timed", WEIGHTS), voices)
    # ponytail: this export names the output `durations`, kokoro-onnx looks for `duration`.
    timed.has_timings = True
    return Kokoro(os.path.join(directory, WEIGHTS), voices), timed


@asynccontextmanager
async def lifespan(_app):
    global _kokoro, _timed
    _kokoro, _timed = load()
    for k in (_kokoro, _timed):
        k.create("Ready.", voice="af_heart")  # the first call pays for graph setup, not a user
    yield


app = FastAPI(lifespan=lifespan)


class Speech(BaseModel):  # OpenAI's fields
    input: str
    voice: str = "af_heart"
    model: str = "kokoro"  # one model; accepted and ignored
    response_format: str = "pcm"
    speed: float = 1.0
    instructions: str | None = None  # OpenAI's acting notes; Kokoro has none, ignored


class CaptionedSpeech(Speech):
    stream: bool = True  # one JSON line per sentence; false: one object for the whole reply


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


@app.get("/health")
def health():
    return {"ready": _kokoro is not None}


@app.get("/v1/audio/voices")
def voices():
    return {"voices": sorted(_kokoro.get_voices())}


def validated(req: Speech) -> list[str]:
    if req.response_format != "pcm":
        raise HTTPException(400, "only response_format 'pcm' (16-bit mono 24 kHz) is supported")
    if req.voice not in _kokoro.get_voices():
        raise HTTPException(400, f"unknown voice {req.voice!r}; GET /v1/audio/voices lists them")
    if not 0.5 <= req.speed <= 2.0:
        raise HTTPException(400, "speed must be between 0.5 and 2.0")
    parts = sentences(req.input)
    if not parts:
        raise HTTPException(400, "input is empty")
    return parts


def pcm16(audio):
    return (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()


def words_spoken(sentence, timings, offset):
    """[{word, start_time, end_time}] for each word of the sentence as written.

    The model times phonemes; espeak separates words with a space phoneme, and a written word can
    be spoken as several ("25" is "twenty five"), so each word takes as many spoken groups as
    espeak gives it on its own."""
    groups, cur = [], []
    for t in timings:
        if t.phoneme == " ":
            if cur:
                groups.append(cur)
            cur = []
        elif t.phoneme not in PUNCTUATION:
            cur.append(t)
    if cur:
        groups.append(cur)
    words = sentence.split()
    counts = [max(1, len(_timed.tokenizer.phonemize(w, "en-us").split())) for w in words]
    if sum(counts) != len(groups):  # ponytail: espeak disagreed with itself; spread the words evenly
        counts = None
        span = (groups[0][0].start, groups[-1][-1].end) if groups else (0.0, 0.0)
    out, i = [], 0
    for n, w in enumerate(words):
        if counts:
            g = groups[i:i + counts[n]]
            i += counts[n]
            start, end = g[0][0].start, g[-1][-1].end
        else:
            step = (span[1] - span[0]) / len(words)
            start, end = span[0] + n * step, span[0] + (n + 1) * step
        out.append({"word": w, "start_time": round(offset + start, 3), "end_time": round(offset + end, 3)})
    return out


PUNCTUATION = set(';:,.!?¡¿—…"«»“”(){}[]')


@app.post("/v1/audio/speech")
def speech(req: Speech):
    parts = validated(req)

    def pcm():  # a sync generator: Starlette runs each step on its thread pool
        for part in parts:
            try:
                audio, _ = _kokoro.create(part, voice=req.voice, speed=req.speed)
            except ValueError:  # a sentence with nothing to say ("..."): skip it, the rest still plays
                continue
            yield pcm16(audio)

    return StreamingResponse(pcm(), media_type="audio/pcm")


@app.post("/dev/captioned_speech")
def captioned_speech(req: CaptionedSpeech):
    parts = validated(req)

    def chunks():
        offset = 0.0
        for part in parts:
            try:
                audio, _, timings = _timed.create_timed(part, voice=req.voice, speed=req.speed)
            except ValueError:
                continue
            yield pcm16(audio), words_spoken(part, timings, offset)
            offset += len(audio) / RATE

    def line(audio, stamps):
        return json.dumps({"audio": base64.b64encode(audio).decode(), "audio_format": "audio/pcm",
                           "timestamps": stamps}) + "\n"

    if req.stream:
        return StreamingResponse((line(a, t) for a, t in chunks()), media_type="application/x-ndjson")
    done = list(chunks())
    return Response(line(b"".join(a for a, _ in done), [w for _, t in done for w in t]), media_type="application/json")


if __name__ == "__main__":
    import sys

    import uvicorn

    if sys.argv[1:] == ["fetch"]:
        fetch()
    else:
        uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")))
