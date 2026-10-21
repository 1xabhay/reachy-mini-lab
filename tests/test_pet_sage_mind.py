"""The pet as a light client of the Sage kernel.

Identity, memory and every decision live in the kernel, shared with every other
body. The pet hears, speaks and moves; this checks that it carries words in and
a reply and a feeling out, and behaves well when the kernel is busy, confused
or gone. No robot, no network, no sleeps: the kernel and the clock are fakes.
"""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from reachy_mini_pet import live
from reachy_mini_pet.bus import Bus
from reachy_mini_pet.events import Emote, Heard, Reply, VoiceStarted, VoiceStopped
from reachy_mini_pet.layers import KernelUnavailableError, SageMind


class FakeKernel:
    """The kernel's side of device protocol v0, scripted."""

    def __init__(self, busy=0, unknown=None, down=False, stale=False):
        self.busy, self.unknown, self.down, self.stale = busy, unknown, down, stale
        self.sequence, self.sent, self.hellos, self.deliveries, self.turn = 1, [], [], [], None
        self.respond = lambda text: ("", f"Heard: {text}")

    def _check(self):
        if self.down:
            raise KernelUnavailableError("connection refused")

    def hello(self, manifest):
        self._check()
        self.hellos.append(manifest)
        return 200, {"protocol": 0, "sequence": self.sequence}

    def send(self, kind, turn_id=None, text=""):
        self._check()
        self.sent.append((kind, turn_id, text))
        if kind == "acknowledged":
            self.unknown = None
        if kind == "submitted":
            if self.unknown:
                return 409, {"code": "unknown_pending", "message": "acknowledge it"}
            if self.busy:
                self.busy -= 1
                return 409, {"code": "busy", "message": "retry"}
            feeling, words = self.respond(text)
            self.sequence += 1
            if words is None:
                self.turn = {"turn_id": turn_id, "run_status": "failed"}
            else:
                self.turn = {"turn_id": turn_id, "run_status": "completed"}
                if feeling:
                    self.deliveries.append(
                        {
                            "sequence": self.sequence,
                            "kind": "emote",
                            "turn_id": turn_id,
                            "text": feeling,
                            "stale": self.stale,
                        }
                    )
                self.deliveries.append(
                    {
                        "sequence": self.sequence,
                        "kind": "reply",
                        "turn_id": turn_id,
                        "text": words,
                        "stale": self.stale,
                    }
                )
        return 200, {"sequence": self.sequence}

    def effects(self, since, wait):
        self._check()
        turn = self.turn or (
            {"turn_id": self.unknown, "run_status": "unknown"} if self.unknown else None
        )
        return 200, {
            "sequence": self.sequence,
            "effects": [d for d in self.deliveries if d["sequence"] > since],
            **({"turn": turn} if turn else {}),
        }

    def submitted(self):
        return [turn_id for kind, turn_id, _ in self.sent if kind == "submitted"]


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(seconds, 0.01)


def wired(kernel, vocabulary=("cheerful1", "sad1"), **options):
    bus, clock = Bus(), Clock()
    heard = bus.subscribe((Emote, Reply))
    mind = SageMind(kernel, vocabulary=vocabulary, clock=clock, sleep=clock.sleep, **options)
    mind.start(bus)
    return bus, mind, heard, clock


def hear(bus, mind, text):
    bus.publish(Heard(text=text))
    mind.step()


def test_words_go_to_the_kernel_and_its_reply_is_spoken():
    kernel = FakeKernel()
    bus, mind, out, _ = wired(kernel)
    hear(bus, mind, "hello there")
    (turn_id,) = kernel.submitted()
    assert turn_id.startswith("reachy-")
    replies = [event for event in bus.drain(out) if isinstance(event, Reply)]
    assert [reply.clauses for reply in replies] == [["Heard: hello there"]]


def test_hello_declares_speech_and_exactly_the_feelings_this_body_has():
    kernel = FakeKernel()
    bus, mind, _, _ = wired(kernel, vocabulary=("yes1", "sad1"))
    hear(bus, mind, "hi")
    assert kernel.hellos[0] == {
        "senses": ["speech"],
        "effects": {"reply": {}, "emote": {"vocabulary": ["yes1", "sad1"]}},
    }


def test_the_body_sets_the_tone_before_the_voice_starts():
    kernel = FakeKernel()
    kernel.respond = lambda text: ("cheerful1", "Yay!")
    bus, mind, out, _ = wired(kernel)
    hear(bus, mind, "good news")
    events = bus.drain(out)
    assert [type(event).__name__ for event in events] == ["Emote", "Reply"]
    assert events[0].name == "cheerful1"
    assert events[1].emotion == "cheerful1"


def test_a_feeling_this_body_does_not_have_is_never_played():
    kernel = FakeKernel()
    kernel.respond = lambda text: ("rage1", "Hmph.")
    bus, mind, out, _ = wired(kernel)
    hear(bus, mind, "hi")
    assert [type(event).__name__ for event in bus.drain(out)] == ["Reply"]


def test_a_busy_kernel_is_retried_with_the_same_turn_id():
    kernel = FakeKernel(busy=3)
    bus, mind, out, _ = wired(kernel)
    hear(bus, mind, "are you there")
    assert len(set(kernel.submitted())) == 1 and len(kernel.submitted()) == 4
    assert [event.clauses for event in bus.drain(out)] == [["Heard: are you there"]]


def test_a_kernel_busy_for_too_long_gets_one_moment_said_once():
    kernel = FakeKernel(busy=10_000)
    bus, mind, out, _ = wired(kernel, retry_s=5.0)
    hear(bus, mind, "hello")
    assert len(set(kernel.submitted())) == 1
    assert [event.clauses for event in bus.drain(out)] == [["One moment."]]


def test_its_own_interrupted_turn_is_acknowledged_before_new_words_go_in():
    kernel = FakeKernel(unknown="reachy-old")
    bus, mind, out, _ = wired(kernel)
    hear(bus, mind, "hello again")
    assert ("acknowledged", "reachy-old", "") in kernel.sent
    assert [event.clauses for event in bus.drain(out)] == [["Heard: hello again"]]


def test_an_unreachable_kernel_never_looks_like_the_robot_dying():
    """A kernel outage must stay inside the mind: OSError would end the robot session."""
    assert not issubclass(KernelUnavailableError, OSError)
    kernel = FakeKernel(down=True)
    bus, mind, out, clock = wired(kernel)
    hear(bus, mind, "hello")
    hear(bus, mind, "hello?")
    assert [event.clauses for event in bus.drain(out)] == [["I can't reach my memory right now."]]
    clock.now += 61
    hear(bus, mind, "anyone?")
    assert len(bus.drain(out)) == 1


def test_a_stale_reply_is_remembered_by_the_kernel_but_not_spoken():
    kernel = FakeKernel(stale=True)
    bus, mind, out, _ = wired(kernel)
    hear(bus, mind, "hello")
    assert bus.drain(out) == []


def test_a_failed_turn_says_nothing_and_frees_the_next_one():
    kernel = FakeKernel()
    kernel.respond = lambda text: ("", None)
    bus, mind, out, _ = wired(kernel)
    hear(bus, mind, "one")
    kernel.respond = lambda text: ("", "Two.")
    hear(bus, mind, "two")
    assert len(kernel.submitted()) == 2
    assert [event.clauses for event in bus.drain(out)] == [["Two."]]


def test_its_own_voice_is_reported_as_playback_of_that_turn():
    kernel = FakeKernel()
    bus, mind, out, _ = wired(kernel)
    hear(bus, mind, "hello")
    (reply,) = bus.drain(out)
    (turn_id,) = kernel.submitted()
    bus.publish(VoiceStarted(turn=reply.turn))
    bus.publish(VoiceStopped(turn=reply.turn))
    mind.step()
    assert kernel.sent[-2:] == [
        ("playback_started", turn_id, ""),
        ("playback_finished", turn_id, ""),
    ]


def test_the_pet_keeps_no_conversation_of_its_own():
    assert not hasattr(live, "JsonMemory")


def test_waking_turns_the_motors_on_before_any_gesture(monkeypatch):
    from conftest import FakePetMini

    mini = FakePetMini()
    live.wake(mini)
    assert mini.names()[:2] == ["enable_motors", "wake_up"]


def test_waking_fails_loudly_when_the_motors_will_not_turn_on():
    from conftest import FakePetMini

    class Limp(FakePetMini):
        def enable_motors(self):
            raise RuntimeError("motor bus down")

    with pytest.raises(RuntimeError):
        live.wake(Limp())


def test_the_client_speaks_protocol_v0_with_its_token():
    seen = []

    class Kernel(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append(
                (
                    self.path,
                    self.headers["Authorization"],
                    self.rfile.read(int(self.headers["Content-Length"])),
                )
            )
            body = b'{"protocol": 0, "sequence": 7}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Kernel)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = live.SageClient(f"http://127.0.0.1:{server.server_port}", "t" * 32)
        assert client.hello({"effects": {}}) == (200, {"protocol": 0, "sequence": 7})
    finally:
        server.shutdown()
        server.server_close()
    path, authorization, body = seen[0]
    assert (path, authorization) == ("/v0/hello", "Bearer " + "t" * 32)
    assert b'"protocols": [0]' in body


def test_an_unreachable_kernel_is_kernel_unavailable_not_a_network_error():
    client = live.SageClient("http://127.0.0.1:9", "t" * 32, timeout_s=1)
    with pytest.raises(KernelUnavailableError):
        client.send("connected")


def test_emoji_become_body_language_and_are_never_spoken():
    kernel = FakeKernel()
    kernel.respond = lambda text: ("", "Great news 😊! I knew it 👍")
    bus, mind, out, _ = wired(kernel, vocabulary=("cheerful1", "yes1"))
    hear(bus, mind, "I passed")
    events = bus.drain(out)
    assert [type(event).__name__ for event in events] == ["Emote", "Reply"]
    assert events[0].name == "cheerful1"
    assert events[1].clauses == ["Great news!", "I knew it"]


def test_the_kernel_s_chosen_feeling_wins_over_an_emoji():
    kernel = FakeKernel()
    kernel.respond = lambda text: ("sad1", "Oh no 😊")
    bus, mind, out, _ = wired(kernel, vocabulary=("cheerful1", "sad1"))
    hear(bus, mind, "hi")
    events = bus.drain(out)
    assert events[0].name == "sad1"
    assert events[1].clauses == ["Oh no"]


def test_a_reply_of_only_emoji_is_shown_not_spoken():
    kernel = FakeKernel()
    kernel.respond = lambda text: ("", "😊😊")
    bus, mind, out, _ = wired(kernel, vocabulary=("cheerful1",))
    hear(bus, mind, "hi")
    assert [type(event).__name__ for event in bus.drain(out)] == ["Emote"]
