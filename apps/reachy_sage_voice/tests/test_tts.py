"""Kokoro through its captioned endpoint, so an interruption knows the exact words heard.

The server here is scripted on localhost with Kokoro-FastAPI's shapes, as the
services/tts app serves them: NDJSON lines of base64 PCM plus
[{word, start_time, end_time}] timed from the start of the request.
"""

import asyncio
import base64
import json

import numpy as np
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    InterruptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSTextFrame,
)
from pipecat.tests.utils import SleepFrame, run_test

from reachy_sage_voice.interruption import InterruptionTracker
from reachy_sage_voice.sage import BotSpeech, Playback
from reachy_sage_voice.transport import ReachyOutputTransport
from reachy_sage_voice.tts import KokoroTTSService

RATE = 24000


def pcm(seconds):
    return (np.ones(int(RATE * seconds), dtype=np.int16) * 1000).tobytes()


def line(seconds, words):
    """One sentence: its audio and words, timed from the start of the request."""
    return {"audio": base64.b64encode(pcm(seconds)).decode(), "audio_format": "audio/pcm",
            "timestamps": [{"word": w, "start_time": s, "end_time": e} for w, s, e in words]}


def ndjson(*lines):
    return (200, b"".join(json.dumps(x).encode() + b"\n" for x in lines))


def tts(server, **kw):
    return KokoroTTSService(base_url=server.url + "/v1", voice="bf_emma", **kw)


def texts(down):
    return [f.text for f in down if isinstance(f, TTSTextFrame)]


def test_captioned_words_come_out_one_by_one_in_time_order(http_server):
    http_server.script("POST", "/dev/captioned_speech", ndjson(
        line(0.6, [("Hello", 0.0, 0.3), ("there.", 0.3, 0.6)]),
        line(0.4, [("Bye.", 0.6, 1.0)])))
    down, _ = asyncio.run(run_test(tts(http_server), frames_to_send=[
        TTSSpeakFrame("Hello there. Bye."), SleepFrame(0.3)]))
    words = [f for f in down if isinstance(f, TTSTextFrame)]
    assert [w.text for w in words] == ["Hello", "there.", "Bye."]
    assert [w.pts for w in words] == sorted(w.pts for w in words)
    audio = b"".join(f.audio for f in down if isinstance(f, TTSAudioRawFrame))
    assert len(audio) >= len(pcm(1.0))
    (req,) = [r for r in http_server.requests if r["path"] == "/dev/captioned_speech"]
    assert req["body"]["input"] == "Hello there. Bye." and req["body"]["voice"] == "bf_emma"


def test_without_captions_the_plain_endpoint_speaks_and_the_clause_counts_whole(http_server):
    http_server.script("POST", "/dev/captioned_speech", (404, {"detail": "Not Found"}))
    http_server.script("POST", "/v1/audio/speech", (200, pcm(0.5)))
    service = tts(http_server)
    down, _ = asyncio.run(run_test(service, frames_to_send=[
        TTSSpeakFrame("Hello there."), SleepFrame(0.2), TTSSpeakFrame("Bye now."),
        SleepFrame(0.2)]))
    assert texts(down) == ["Hello there.", "Bye now."]
    assert any(isinstance(f, TTSAudioRawFrame) for f in down)
    captioned = [r for r in http_server.requests if r["path"] == "/dev/captioned_speech"]
    assert len(captioned) == 1  # asked once; the plain endpoint from then on


def test_word_frames_reach_playback_and_rejoin_as_text():
    playback = Playback()
    asyncio.run(run_test(BotSpeech(InterruptionTracker(), playback), frames_to_send=[
        *[TTSTextFrame(w, aggregated_by="word") for w in ("Hello,", "I", "was")],
        SleepFrame(0.05)]))
    assert playback.text == "Hello, I was"


def test_the_policy_rechecks_at_clause_ends_not_at_every_word():
    tracker = InterruptionTracker(min_bot_speech_ms=0)
    playback = Playback()
    rechecks = []
    original = tracker.bot_started
    tracker.bot_started = lambda t: rechecks.append(t) or original(t)
    asyncio.run(run_test(BotSpeech(tracker, playback), frames_to_send=[
        BotStartedSpeakingFrame(),
        *[TTSTextFrame(w, aggregated_by="word") for w in ("Well", "then,", "maybe", "not.")],
        SleepFrame(0.05)]))
    assert len(rechecks) == 3  # the start, then after "then," and "not."


def test_an_interrupted_reply_reports_only_the_words_already_played(http_server):
    words = [(f"w{i}", i * 0.2, (i + 1) * 0.2) for i in range(10)]  # 2 s of speech
    http_server.script("POST", "/dev/captioned_speech", ndjson(line(2.0, words)))

    class Media:
        def start_playing(self): pass
        def stop_playing(self): pass
        def push_audio_sample(self, data): pass
        def get_output_audio_samplerate(self): return 16000

    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.transports.base_transport import TransportParams

    playback = Playback()
    pipeline = Pipeline([
        tts(http_server),
        ReachyOutputTransport(Media(), TransportParams(audio_out_enabled=True,
                                                       audio_out_sample_rate=16000)),
        BotSpeech(InterruptionTracker(), playback)])
    asyncio.run(run_test(pipeline, frames_to_send=[
        TTSSpeakFrame("w0 w1 w2 w3 w4 w5 w6 w7 w8 w9"), SleepFrame(0.9),
        InterruptionFrame(), SleepFrame(0.2)]))
    heard = playback.text.split()
    assert heard == [f"w{i}" for i in range(len(heard))]  # a prefix, in order
    assert 2 <= len(heard) <= 7  # about 0.9 s of a 2 s reply, not all of it
