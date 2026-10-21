"""The voice and the real Sage kernel together: what was said, and what was journaled.

The kernel is sage-kernel's own (``SAGE_KERNEL_DIR``, default the sibling
checkout, as ``apps/reachy_sim`` finds it), started by its conformance suite on
a free loopback port with a throwaway state directory, thinking with that
suite's fake Ollama, whose answers each test scripts. The voice is the app's
real ``Session``, ``SageProcessor``, Kokoro TTS service with its clause
aggregator, output transport and ``BotSpeech``. Faked: the transcripts (sent as
frames), Kokoro's HTTP server (captioned speech, 0.1 s a word), and the robot
(its speaker and moves).

Each scenario checks both sides of the wire: what the voice gave Kokoro or the
speaker played, and the kernel's journal (or the prompt it built from it).
"""

import asyncio
import base64
import importlib.util
import json
import os
import sqlite3
import tempfile
import time
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest
from pipecat.frames.frames import InterruptionFrame, TranscriptionFrame, TTSTextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transports.base_transport import TransportParams

from reachy_sage_voice import clause
from reachy_sage_voice.interruption import InterruptionTracker
from reachy_sage_voice.kernel import EMOTIONS, KernelClient, Session, manifest
from reachy_sage_voice.pipeline import report_presence
from reachy_sage_voice.sage import BotSpeech, Playback, SageProcessor
from reachy_sage_voice.transport import ReachyOutputTransport
from reachy_sage_voice.tts import RATE, KokoroTTSService

pytestmark = pytest.mark.integration

APP = Path(__file__).resolve().parents[2]
KERNEL_DIR = Path(os.environ.get("SAGE_KERNEL_DIR", APP.parents[2] / "sage-kernel"))
WORD_S = 0.1  # how long the fake Kokoro takes to say each word


def _conformance():
    path = KERNEL_DIR / "conformance" / "test_conformance.py"
    if not path.exists():
        pytest.fail(f"no Sage kernel at {KERNEL_DIR}: set SAGE_KERNEL_DIR to its checkout")
    spec = importlib.util.spec_from_file_location("sage_conformance", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Move:
    duration, description = 0.3, "A move."


class Robot:
    """The body: a speaker that takes anything, and moves that take ``Move.duration``."""

    def __init__(self):
        self.moves: list[str] = []

    def start_playing(self): pass
    def stop_playing(self): pass
    def push_audio_sample(self, data): pass
    def get_output_audio_samplerate(self): return 16000

    async def play_emote(self, name, sound):
        self.moves.append(name)
        await asyncio.sleep(Move.duration)


class Rig:
    def __init__(self, conformance, provider, process, state, tts):
        self.conformance, self.provider, self.process = conformance, provider, process
        self.state, self.tts = state, tts
        self.replies: dict[str, tuple[str, dict]] = {}  # the model's answer, by question
        self.robot = Robot()
        fallback = conformance.answer

        def answer(messages):
            question = conformance.words(messages[-1])
            # The kernel quotes the transcript ('heard (may be misheard): "..."').
            return next((said for asked, said in self.replies.items() if asked in question),
                        None) or fallback(messages)
        conformance.answer = answer  # the fake Ollama looks it up per request

    # --- driving ---------------------------------------------------------------

    def device(self, method, path, body=None, device="reachy"):
        return self.process.device(method, path, body, token=self.conformance.TOKENS[device])

    def run(self, frames, during=None, retry_s=5.0):
        """Run the voice over ``frames``; ``during()`` runs beside it, for the kernel's side."""
        client = KernelClient(f"http://127.0.0.1:{self.process.device_port}",
                              self.conformance.TOKENS["reachy"])
        self.session = Session(client, manifest(dict.fromkeys(EMOTIONS, Move())), retry_s=retry_s)
        tts = KokoroTTSService(base_url=self.tts.url + "/v1", voice="af_heart")
        clause.install(tts)
        playback = Playback()
        pipeline = Pipeline([
            SageProcessor(self.session, self.robot.play_emote, playback),
            tts,
            ReachyOutputTransport(self.robot, TransportParams(
                audio_out_enabled=True, audio_out_sample_rate=16000)),
            BotSpeech(InterruptionTracker(), playback)])

        async def go():
            jobs = [run_test(pipeline, frames_to_send=frames)]
            if during:
                jobs.append(asyncio.to_thread(during))
            return (await asyncio.gather(*jobs))[0]
        down, _ = asyncio.run(go())
        return down

    # --- what happened -----------------------------------------------------------

    def journal(self, *kinds):
        """Read the kernel's journal rows straight from its SQLite file."""
        uri = (Path(self.state) / "kernel.sqlite3").as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as db:
            rows = db.execute("SELECT sequence, kind, source, target, turn_id, body FROM journal "
                              "ORDER BY sequence").fetchall()
        return [{"sequence": s, "kind": k, "source": src, "target": t, "turn_id": tid,
                 **json.loads(body)} for s, k, src, t, tid, body in rows if not kinds or k in kinds]

    def tts_inputs(self):
        """List each piece of text the voice was given to say, in order."""
        return [r["body"]["input"] for r in self.tts.requests
                if r["path"] == "/dev/captioned_speech"]

    def prompt(self, index=-1):
        """Return the messages the kernel sent the model on one call."""
        return self.provider.requests[index]["messages"]


def _captioned(fake):
    """Answer as Kokoro-FastAPI's captioned speech: one NDJSON line, every word WORD_S long."""
    text = fake.requests[-1]["body"]["input"]
    said = text.split()
    audio = (np.ones(int(RATE * WORD_S * len(said)), dtype=np.int16) * 1000).tobytes()
    stamps = [{"word": w, "start_time": i * WORD_S, "end_time": (i + 1) * WORD_S}
              for i, w in enumerate(said)]
    line = {"audio": base64.b64encode(audio).decode(), "audio_format": "audio/pcm",
            "timestamps": stamps}
    return 200, json.dumps(line).encode() + b"\n"


@pytest.fixture
def rig(request, http_server):
    """Start the kernel and the fake Kokoro, with the voice to run between them.

    ``@pytest.mark.kernel_env(NAME="value")`` sets kernel settings beyond the suite's defaults.
    """
    conformance = _conformance()
    http_server.script("POST", "/dev/captioned_speech", _captioned)
    marker = request.node.get_closest_marker("kernel_env")
    provider = conformance.Provider()
    with tempfile.TemporaryDirectory() as state:
        process = conformance.KernelProcess(state, provider, **(marker.kwargs if marker else {}))
        process.start()
        try:
            yield Rig(conformance, provider, process, state, http_server)
        finally:
            process.stop()
            provider.open.set()
            provider.midway.set()
            provider.server.shutdown()
            provider.server.server_close()


# --- the scenarios ------------------------------------------------------------


def heard(text):
    return TranscriptionFrame(text=text, user_id="you", timestamp="")


def spoken(down):
    """Words as the speaker played them: the transport releases each as it plays."""
    return [f.text for f in down if isinstance(f, TTSTextFrame)]


def wait_for(predicate, within=10.0):
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        if value := predicate():
            return value
        time.sleep(0.02)
    raise AssertionError("timed out waiting")


def test_fragments_while_thinking_join_one_turn(rig):
    rig.provider.open.clear()  # the model is thinking

    def release():
        wait_for(lambda: len(rig.journal("submitted")) == 2)
        rig.provider.open.set()

    down = rig.run([heard("What's the pin?"), SleepFrame(0.3), heard("Really important."),
                    SleepFrame(2.0)], during=release)
    (reply,) = rig.journal("reply")
    assert reply["text"] == "Heard: What's the pin? Really important."
    first, second = rig.journal("submitted")
    assert second["text"] == "What's the pin? Really important."
    stopped = [r["turn_id"] for r in rig.journal("stop") if r["source"] == "reachy"]
    assert stopped == [first["turn_id"]]
    assert " ".join(rig.tts_inputs()).split() == reply["text"].split()
    assert spoken(down) == reply["text"].split()


STORY = ("Once upon a time there was a robot. It lived on a desk and watched the world. "
         "Every day it learned a new word.")


def interrupted(rig, reply, sequence=True):
    """Return the words the voice reported heard of ``reply`` (by sequence, or by turn)."""
    key = ("sequence", reply["sequence"]) if sequence else ("turn_id", reply["turn_id"])
    (result,) = [r for r in rig.journal("result") if r.get(key[0]) == key[1]]
    assert result["status"] == "interrupted"
    return result["text"].split()


def remembers(rig, played, unheard):
    """Check the next prompt holds what the person heard of a reply, and none of the rest."""
    prompt = " ".join(m["content"] for m in rig.prompt() if isinstance(m.get("content"), str))
    assert f'"{" ".join(played)}"' in prompt
    assert " ".join(unheard[:2]) not in prompt


def test_a_new_transcript_mid_reply_interrupts_it_cleanly(rig):
    old = ("One two three four five six seven eight. Nine ten eleven twelve. "
           "Thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty.")
    new = "Sure. Here is a new answer. And a second sentence too."
    rig.replies = {"Count for me.": (old, {}), "Actually, something else.": (new, {})}
    rig.provider.midway.clear()  # the first sentence streams, the rest is still being written

    def release():
        wait_for(lambda: len(rig.journal("submitted")) == 2)
        rig.provider.midway.set()

    down = rig.run([heard("Count for me."), SleepFrame(0.6), heard("Actually, something else."),
                    SleepFrame(3.5)], during=release)
    first, _ = rig.journal("submitted")
    played = interrupted(rig, first, sequence=False)
    assert 0 < len(played) < 8 and played == old.split()[:len(played)]
    assert spoken(down) == played + new.split()  # nothing doubled, nothing stale, nothing dropped
    assert [r["text"] for r in rig.journal("reply")] == [new]
    remembers(rig, played, unheard=old.split()[len(played):])
    # The old reply's first sentence went to the voice; its held clause never did.
    assert " ".join(rig.tts_inputs()).split() == old.split()[:8] + new.split()


def test_barge_in_after_the_reply_is_journaled_keeps_what_was_heard(rig):
    rig.replies = {"Tell me a story.": (STORY, {})}
    down = rig.run([heard("Tell me a story."), SleepFrame(1.2), InterruptionFrame(),
                    SleepFrame(0.5), heard("What happened next?"), SleepFrame(1.5)])
    reply = rig.journal("reply")[0]
    assert reply["text"] == STORY
    played = interrupted(rig, reply)
    assert 0 < len(played) < len(STORY.split()) and played == STORY.split()[:len(played)]
    assert spoken(down) == [*played, "Heard:", "What", "happened", "next?"]
    remembers(rig, played, unheard=STORY.split()[len(played):])


def test_stop_from_another_device_silences_the_voice(rig):
    rig.replies = {"Tell me a story.": (STORY, {})}

    def stop():
        wait_for(lambda: rig.journal("reply"))
        time.sleep(1.0)  # part way through the reply
        assert rig.device("POST", "/v1/events", {"kind": "stop"}, device="mock")[0] == 200

    down = rig.run([heard("Tell me a story."), SleepFrame(3.5)], during=stop)
    stops = rig.journal("stop")
    assert [(s["source"], s["target"]) for s in stops] == [("mock", None), ("kernel", "reachy")]
    played = interrupted(rig, rig.journal("reply")[0])
    assert 0 < len(played) < len(STORY.split())
    assert spoken(down) == played


@pytest.mark.kernel_env(SAGE_GREET_ON_START="1")
def test_someone_seen_is_greeted(rig):
    def seen():
        time.sleep(0.5)  # the voice is up and listening
        asyncio.run(report_presence(rig.session, {"state": "seen", "confidence": 0.9}))

    down = rig.run([SleepFrame(4.0)], during=seen)
    assert rig.journal("proactive")
    (reply,) = rig.journal("reply")
    assert reply["turn_id"].startswith("kernel-") and reply["target"] == "reachy"
    assert spoken(down) == reply["text"].split()
    (result,) = [r for r in rig.journal("result") if r.get("sequence") == reply["sequence"]]
    assert result["status"] == "done"


def test_a_busy_kernel_is_retried_then_one_moment(rig):
    rig.provider.open.clear()  # another device's turn is thinking
    assert rig.device("POST", "/v1/hello", {"protocols": [1], "manifest": {"senses": []}},
                      device="mock")[0] == 200
    assert rig.device("POST", "/v1/events", {"kind": "submitted", "turn_id": "mock-1",
                                             "text": "Hold on."}, device="mock")[0] == 200
    started = time.monotonic()
    down = rig.run([heard("Are you there?"), SleepFrame(2.0)], retry_s=1.0)
    assert time.monotonic() - started >= 1.0
    assert rig.tts_inputs() == ["One moment."]
    assert spoken(down) == ["One", "moment."]
    assert [r["source"] for r in rig.journal("submitted")] == ["mock"]  # busy admits nothing
    assert rig.journal("stop") == []  # and the other turn was left alone


def test_a_reply_cut_at_the_cap_streams_only_what_is_journaled(rig):
    long = ("The first sentence is short. The second one is a little longer than the first "
            "one was. The third sentence goes on and on and on until it runs well past the cap "
            "that the kernel sets.")
    assert len(long) > 160
    rig.replies = {"Talk a lot.": (long[:160], {})}  # the decoder stops at the cap, mid-word
    down = rig.run([heard("Talk a lot."), SleepFrame(4.0)])
    (reply,) = rig.journal("reply")
    assert len(reply["text"]) <= 160 and long.startswith(reply["text"])
    streamed = " ".join(rig.tts_inputs()).split()
    assert streamed == reply["text"].split()[:len(streamed)]
    assert spoken(down) == reply["text"].split()


def test_an_emote_result_carries_how_long_the_move_took(rig):
    down = rig.run([heard("Feel happy"), SleepFrame(2.0)])
    assert rig.robot.moves == ["cheerful1"]
    (emote,) = rig.journal("emote")
    (result,) = [r for r in rig.journal("result") if r.get("sequence") == emote["sequence"]]
    assert result["status"] == "done"
    assert 250 <= (result.get("duration_ms") or 0) <= 2000  # the move takes 0.3 s
    assert spoken(down) == ["Yay!"]
