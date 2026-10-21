"""Speech to text: Parakeet TDT 0.6B v2 (int8, ONNX, CPU) behind OpenAI's transcription endpoint.

Pipecat's OpenAISTTService posts a WAV segment to POST /v1/audio/transcriptions as multipart
(`file`, `model`, and optionally `language`, `prompt`, `temperature`, `response_format`) and
reads `.text` from the JSON. The other fields are accepted and ignored: this server has one model.
Adapted from an earlier Pipecat voice bot's STT service.
"""
import io
import os
from contextlib import asynccontextmanager

import numpy as np
import onnx_asr
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

MODEL = "nemo-parakeet-tdt-0.6b-v2"
RATE = 16000
_model = None


def load():
    """The model, from the Hugging Face cache (baked into the image; fetched on first native run)."""
    return onnx_asr.load_model(MODEL, quantization="int8", providers=["CPUExecutionProvider"])


@asynccontextmanager
async def lifespan(_app):
    global _model
    _model = load()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"ready": _model is not None}


@app.post("/v1/audio/transcriptions")
def transcribe(file: UploadFile = File(...), model: str = Form(""), language: str = Form(""),
               prompt: str = Form(""), temperature: str = Form(""), response_format: str = Form("json")):
    """Sync on purpose: FastAPI runs it on its thread pool and onnxruntime releases the GIL."""
    if response_format not in ("json", "verbose_json"):  # both answer {"text": ...}
        raise HTTPException(400, f"response_format {response_format!r} is not supported")
    try:
        audio, rate = sf.read(io.BytesIO(file.file.read()), dtype="float32", always_2d=True)
    except Exception:
        raise HTTPException(400, "not a readable audio file")
    audio = audio.mean(axis=1)
    if rate != RATE:  # ponytail: linear resampling; Pipecat sends 16 kHz, so this is rarely used
        audio = np.interp(np.arange(0, len(audio), rate / RATE), np.arange(len(audio)), audio).astype("float32")
    text = _model.recognize(audio, sample_rate=RATE) if len(audio) else ""
    return {"text": text}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")))
