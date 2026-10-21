"""The TTS server's contract, as Pipecat's OpenAITTSService uses it. Loads the real model.

    cd services && uv run python -m unittest discover -s tts -t tts
"""
import base64
import json
import unittest

from fastapi.testclient import TestClient

import app

RATE, WIDTH = 24000, 2  # what Pipecat assumes for "pcm": 16-bit mono at 24 kHz


class SpeechContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app.app).__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def speak(self, **body):
        # Exactly the fields Pipecat sends (services/openai/tts.py run_tts).
        return self.client.post("/v1/audio/speech", json={"model": "kokoro", "voice": "af_heart",
                                                          "response_format": "pcm", **body})

    def test_health_is_ready_once_the_model_is_loaded(self):
        self.assertEqual(self.client.get("/health").json(), {"ready": True})

    def test_a_reply_comes_back_as_raw_pcm_of_a_plausible_length(self):
        r = self.speak(input="Good morning! I was watching the light on your desk.")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["content-type"], "audio/pcm")
        self.assertEqual(len(r.content) % WIDTH, 0)
        self.assertTrue(1.5 < len(r.content) / WIDTH / RATE < 6, len(r.content))

    def test_each_sentence_is_streamed_as_soon_as_it_is_made(self):
        with self.client.stream("POST", "/v1/audio/speech", json={
                "voice": "af_heart", "response_format": "pcm", "input": "Hello there. How are you today?"}) as r:
            chunks = [c for c in r.iter_raw() if c]
        self.assertGreaterEqual(len(chunks), 2)

    def test_speed_is_honoured(self):
        slow = len(self.speak(input="Sleep well.", speed=0.7).content)
        fast = len(self.speak(input="Sleep well.", speed=1.4).content)
        self.assertGreater(slow, fast * 1.4)

    def test_instructions_are_accepted_and_ignored(self):
        self.assertEqual(self.speak(input="Okay.", instructions="cheerfully").status_code, 200)

    def test_an_unknown_voice_is_a_400_naming_the_list(self):
        r = self.speak(input="Hi.", voice="alloy")
        self.assertEqual(r.status_code, 400)
        self.assertIn("/v1/audio/voices", r.text)
        self.assertIn("af_heart", self.client.get("/v1/audio/voices").json()["voices"])

    def test_a_format_other_than_pcm_is_a_400(self):
        self.assertEqual(self.speak(input="Hi.", response_format="mp3").status_code, 400)

    def test_empty_input_is_a_400(self):
        self.assertEqual(self.speak(input="   ").status_code, 400)



class CaptionedSpeechContract(unittest.TestCase):
    """POST /dev/captioned_speech, Kokoro-FastAPI's shape: audio plus [{word, start_time, end_time}],
    so the voice knows exactly which words were heard when it is interrupted."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app.app).__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def caption(self, text, **body):
        r = self.client.post("/dev/captioned_speech", json={"voice": "af_heart", "response_format": "pcm",
                                                            "input": text, **body})
        self.assertEqual(r.status_code, 200, r.text)
        return [json.loads(line) for line in r.text.splitlines() if line.strip()]

    def test_every_input_word_comes_back_in_order_with_a_span_inside_the_audio(self):
        text = "Good morning! I was watching the light on your desk."
        chunks = self.caption(text, stream=False)
        self.assertEqual(len(chunks), 1)
        c = chunks[0]
        self.assertEqual(c["audio_format"], "audio/pcm")
        seconds = len(base64.b64decode(c["audio"])) / WIDTH / RATE
        self.assertEqual([t["word"] for t in c["timestamps"]], text.split())
        ends = 0.0
        for t in c["timestamps"]:
            self.assertLessEqual(ends, t["start_time"] + 1e-6)
            self.assertLess(t["start_time"], t["end_time"])
            ends = t["end_time"]
        self.assertLessEqual(ends, seconds + 0.01)

    def test_streamed_one_sentence_per_line_with_times_from_the_start_of_the_reply(self):
        chunks = self.caption("Hello there. How are you today?", stream=True)
        self.assertEqual([[t["word"] for t in c["timestamps"]] for c in chunks],
                         [["Hello", "there."], ["How", "are", "you", "today?"]])
        first_len = len(base64.b64decode(chunks[0]["audio"])) / WIDTH / RATE
        self.assertGreaterEqual(chunks[1]["timestamps"][0]["start_time"], first_len - 1e-6)

    def test_a_word_spoken_as_several_is_still_one_entry(self):
        words = [t["word"] for c in self.caption("Set a timer for 25 minutes.") for t in c["timestamps"]]
        self.assertEqual(words, ["Set", "a", "timer", "for", "25", "minutes."])

    def test_when_espeak_disagrees_with_itself_the_words_are_spread_evenly(self):
        from kokoro_onnx.sliding import Timing

        three_groups = [Timing("a", 0.0, 0.2), Timing(" ", 0.2, 0.2), Timing("b", 0.2, 0.4),
                        Timing(" ", 0.4, 0.4), Timing("c", 0.4, 0.6)]
        self.assertEqual(app.words_spoken("one two", three_groups, 1.0),
                         [{"word": "one", "start_time": 1.0, "end_time": 1.3},
                          {"word": "two", "start_time": 1.3, "end_time": 1.6}])

    def test_the_same_validation_as_plain_speech(self):
        r = self.client.post("/dev/captioned_speech", json={"input": "Hi.", "voice": "alloy"})
        self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main()
