"""The STT server's contract, as Pipecat's OpenAISTTService uses it. Loads the real model.

    cd services && uv run python -m unittest discover -s stt -t stt
"""
import io
import pathlib
import re
import unittest

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

import app

# SYNTHETIC: Kokoro bf_emma saying these words, 16 kHz mono (made by sage-kernel bench/).
FIXTURE = pathlib.Path(__file__).parent / "fixtures/good-morning-little-robot.wav"


def wav(audio, rate):
    buf = io.BytesIO()
    sf.write(buf, audio, rate, format="WAV")
    return buf.getvalue()


class TranscriptionContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app.app).__enter__()  # runs the lifespan: the model loads once

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def post(self, data: bytes, **fields):
        return self.client.post("/v1/audio/transcriptions", files={"file": ("a.wav", data, "audio/wav")},
                                data={"model": "parakeet-tdt-0.6b-v2", **fields})

    def test_health_is_ready_once_the_model_is_loaded(self):
        self.assertEqual(self.client.get("/health").json(), {"ready": True})

    def test_speech_comes_back_as_its_words(self):
        r = self.post(FIXTURE.read_bytes())
        self.assertEqual(r.status_code, 200)
        self.assertEqual(re.findall(r"[a-z]+", r.json()["text"].lower()), ["good", "morning", "little", "robot"])

    def test_openai_fields_pipecat_may_send_are_accepted(self):
        r = self.post(FIXTURE.read_bytes(), language="en", prompt="", temperature="0", response_format="json")
        self.assertEqual(r.status_code, 200)

    def test_silence_is_empty_text_not_an_error(self):
        r = self.post(wav(np.zeros(16000, dtype="float32"), 16000))
        self.assertEqual((r.status_code, r.json()["text"].strip()), (200, ""))

    def test_other_sample_rates_are_resampled(self):
        a, rate = sf.read(FIXTURE, dtype="float32")
        a24 = np.interp(np.arange(0, len(a), rate / 24000), np.arange(len(a)), a).astype("float32")
        self.assertIn("morning", self.post(wav(a24, 24000)).json()["text"].lower())

    def test_not_audio_is_a_400(self):
        self.assertEqual(self.post(b"not a wav").status_code, 400)

    def test_a_format_it_cannot_answer_is_a_400(self):
        self.assertEqual(self.post(FIXTURE.read_bytes(), response_format="srt").status_code, 400)


if __name__ == "__main__":
    unittest.main()
