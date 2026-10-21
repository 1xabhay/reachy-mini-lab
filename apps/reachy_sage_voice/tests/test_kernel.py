"""Device protocol v1 over stdlib HTTP, against a scripted kernel on localhost.

Only the wire is real here; clocks and sleeps are injected so retries run instantly.
"""

import asyncio
import json
import re
from typing import NamedTuple

import pytest

from reachy_sage_voice.kernel import (
    EMOTIONS,
    KernelClient,
    KernelUnavailableError,
    Session,
    manifest,
)

from conftest import TOKEN

# --- manifest -----------------------------------------------------------------


class Move(NamedTuple):
    duration: float
    description: str


#: A stand-in library: the real one is checked in test_vocabulary.py.
MOVES = {name: Move(2.0, f"Look {name}. Use it sparingly.") for name in EMOTIONS}
MOVES["boredom1"] = Move(15.7, "You're about to fall asleep because the conversation is boring. "
                               "Use it when no one talks to you for 2 minutes.")


def test_manifest_offers_emote_as_an_mcp_tool_over_the_declared_moves():
    m = manifest(MOVES)
    assert m["senses"] == ["speech"]
    emote = m["tools"][0]
    assert emote["name"] == "emote"
    assert emote["inputSchema"]["type"] == "object"
    assert emote["inputSchema"]["required"] == ["name"]
    assert emote["inputSchema"]["properties"]["name"]["enum"] == list(EMOTIONS)
    assert len(EMOTIONS) == len(set(EMOTIONS)) and "cheerful1" in EMOTIONS
    assert emote["interruptible"] is True
    assert re.fullmatch(r"[a-z][a-z0-9_]{0,63}", emote["name"])


def test_max_s_fits_the_longest_declared_move_with_a_second_to_spare():
    assert manifest(MOVES)["tools"][0]["max_s"] == 17  # ceil(15.7) + 1
    short = {name: MOVES[name] for name in ("cheerful1", "yes1")}
    assert manifest(short)["tools"][0]["max_s"] == 3  # ceil(2.0) + 1


def test_each_name_carries_the_first_line_of_its_library_description():
    name = manifest(MOVES)["tools"][0]["inputSchema"]["properties"]["name"]
    assert "boredom1: You're about to fall asleep because the conversation is boring." \
        in name["description"]
    assert "2 minutes" not in name["description"]  # one line each: the bench's finding
    assert all(f"{n}:" in name["description"] for n in EMOTIONS)


def test_the_schema_stays_in_the_subset_the_kernel_and_ollama_check():
    """Use enum, which the kernel validates and reads to keep labels unspoken.

    The kernel's checker ignores oneOf/const, so an oneOf vocabulary would let any
    name through. Ollama's format grammar takes enum. Descriptions are plain text.
    """
    name = manifest(MOVES)["tools"][0]["inputSchema"]["properties"]["name"]
    assert set(name) == {"type", "enum", "description"} and name["type"] == "string"
    assert json.loads(json.dumps(manifest(MOVES))) == manifest(MOVES)


# --- the client ---------------------------------------------------------------


def client(kernel, **kw):
    return KernelClient(kernel.url, TOKEN, **kw)


def test_hello_declares_v1_and_the_manifest_with_the_token(kernel):
    kernel.script("POST", "/v1/hello", (200, {"protocol": 1, "sequence": 7}))
    assert client(kernel).hello(manifest(MOVES)) == (200, {"protocol": 1, "sequence": 7})
    (req,) = kernel.requests
    assert req["headers"]["Authorization"] == f"Bearer {TOKEN}"
    assert req["body"] == {"protocols": [1], "manifest": manifest(MOVES)}


def test_effects_long_polls_from_a_cursor_and_a_reply_offset(kernel):
    kernel.script("GET", "/v1/effects", (200, {"sequence": 9, "effects": []}))
    client(kernel).effects(since=9, wait=2, offset=12)
    assert kernel.requests[0]["query"] == {"since": ["9"], "wait": ["2"], "offset": ["12"]}


def test_rate_limited_events_wait_retry_after_then_go_through(kernel):
    kernel.script("POST", "/v1/events",
                  (429, {"code": "rate_limited", "retry_after": 0.25}), (200, {"sequence": 3}))
    slept = []
    status, _ = client(kernel, sleep=slept.append).event({"kind": "result", "sequence": 2,
                                                          "status": "done"})
    assert status == 200 and slept == [0.25]
    assert len(kernel.events("result")) == 2  # the same event, resent


def test_an_unreachable_kernel_is_not_an_oserror():
    dead = KernelClient("http://127.0.0.1:9", TOKEN, timeout_s=0.5)
    with pytest.raises(KernelUnavailableError) as caught:
        dead.hello(manifest(MOVES))
    assert not isinstance(caught.value, OSError)


def test_a_garbled_answer_is_an_outage_too(kernel):
    kernel.script("POST", "/v1/hello", (200, b"<html>proxy error</html>"))
    with pytest.raises(KernelUnavailableError):
        client(kernel).hello(manifest(MOVES))


def test_a_coded_error_comes_back_as_its_status_and_body(kernel):
    kernel.script("POST", "/v1/events", (409, {"code": "forbidden", "message": "not yours"}))
    assert client(kernel).event({"kind": "acknowledged", "turn_id": "x"})[0] == 409


# --- the session: one conversation's rules ------------------------------------


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def session(kernel, clock=None):
    clock = clock or Clock()

    async def sleep(s):
        clock.t += s

    return Session(client(kernel), manifest(MOVES), retry_s=5.0, clock=clock, sleep=sleep)


def hello_ok(kernel, sequence=40):
    kernel.script("POST", "/v1/hello", (200, {"protocol": 1, "sequence": sequence}))


def test_submitting_says_hello_first_then_sends_the_turn_with_timing(kernel):
    hello_ok(kernel)
    kernel.script("POST", "/v1/events", (200, {"sequence": 41}))
    s = session(kernel)
    assert asyncio.run(s.start("reachy-1", "hi", {"stt_ms": 180})) == "admitted"
    assert [r["path"] for r in kernel.requests] == ["/v1/hello", "/v1/events"]
    assert kernel.events() == [{"kind": "submitted", "turn_id": "reachy-1", "text": "hi",
                                "timing": {"stt_ms": 180}}]
    assert s.cursor == 40  # a fresh body starts at the head, never replaying old replies


def test_busy_retries_the_same_turn_id(kernel):
    hello_ok(kernel)
    kernel.script("POST", "/v1/events", (409, {"code": "busy"}), (409, {"code": "busy"}),
                  (200, {"sequence": 41}))
    assert asyncio.run(session(kernel).start("reachy-1", "hi", {})) == "admitted"
    assert [e["turn_id"] for e in kernel.events("submitted")] == ["reachy-1"] * 3


def test_busy_for_about_5_s_gives_up(kernel):
    hello_ok(kernel)
    kernel.script("POST", "/v1/events", (409, {"code": "busy"}))
    clock = Clock()
    assert asyncio.run(session(kernel, clock).start("reachy-1", "hi", {})) == "busy"
    assert 5.0 <= clock.t < 6.0


def test_unknown_pending_acknowledges_this_bodys_own_turn_then_retries(kernel):
    hello_ok(kernel)
    kernel.script("POST", "/v1/events", (409, {"code": "unknown_pending"}),
                  (200, {"sequence": 42}), (200, {"sequence": 43}))
    kernel.script("GET", "/v1/effects", (200, {"sequence": 40, "effects": [],
                                              "turn": {"turn_id": "reachy-0",
                                                       "run_status": "unknown"}}))
    assert asyncio.run(session(kernel).start("reachy-1", "hi", {})) == "admitted"
    assert [(e["kind"], e["turn_id"]) for e in kernel.events()] == [
        ("submitted", "reachy-1"), ("acknowledged", "reachy-0"), ("submitted", "reachy-1")]


def test_another_bodys_pending_turn_is_not_ours_to_acknowledge(kernel):
    hello_ok(kernel)
    kernel.script("POST", "/v1/events", (409, {"code": "unknown_pending"}))
    kernel.script("GET", "/v1/effects", (200, {"sequence": 40, "effects": []}))
    assert asyncio.run(session(kernel).start("reachy-1", "hi", {})) == "refused"
    assert kernel.events("acknowledged") == []


def test_poll_advances_the_cursor_and_passes_the_offset(kernel):
    hello_ok(kernel)
    kernel.script("POST", "/v1/events", (200, {"sequence": 41}))
    kernel.script("GET", "/v1/effects", (200, {"sequence": 44, "effects": [
        {"kind": "reply_delta", "turn_id": "reachy-1", "offset": 0, "text": "Hel"}]}))
    s = session(kernel)

    async def go():
        await s.start("reachy-1", "hi", {})
        return await s.poll(offset=0, wait=2)
    body = asyncio.run(go())
    assert body["effects"][0]["text"] == "Hel" and s.cursor == 44
    assert kernel.requests[-1]["query"]["since"] == ["40"]


def test_a_reset_restarts_from_the_kernels_sequence(kernel):
    hello_ok(kernel, sequence=900)
    kernel.script("POST", "/v1/events", (200, {"sequence": 901}))
    kernel.script("GET", "/v1/effects", (409, {"code": "reset", "sequence": 12}))
    s = session(kernel)

    async def go():
        await s.start("reachy-1", "hi", {})
        return await s.poll(offset=0, wait=0)
    assert asyncio.run(go())["effects"] == [] and s.cursor == 12


def test_results_and_stop_go_out_as_events(kernel):
    kernel.script("POST", "/v1/events", (200, {"sequence": 50}))
    s = session(kernel)

    async def go():
        await s.result(42, "interrupted", text="Hello, I was", timing={"first_audio_ms": 700})
        await s.result(41, "done", duration_ms=2300)
        await s.result(43, "failed", reason="stale")
        await s.result(None, "interrupted", text="Once upon", turn_id="reachy-2")
        await s.stop("reachy-1")
    asyncio.run(go())
    assert kernel.events() == [
        {"kind": "result", "sequence": 42, "status": "interrupted", "text": "Hello, I was",
         "timing": {"first_audio_ms": 700}},
        {"kind": "result", "sequence": 41, "status": "done", "duration_ms": 2300},
        {"kind": "result", "sequence": 43, "status": "failed", "reason": "stale"},
        # A reply still streaming has no sequence yet: it is named by its turn
        # (assumed shape of the pending v1 amendment for streaming devices).
        {"kind": "result", "turn_id": "reachy-2", "status": "interrupted", "text": "Once upon"},
        {"kind": "stop", "turn_id": "reachy-1"}]


def test_presence_goes_out_as_an_event(kernel):
    kernel.script("POST", "/v1/events", (200, {"sequence": 7}))
    asyncio.run(session(kernel).presence("seen", 1.0))
    assert kernel.events() == [{"kind": "presence", "state": "seen", "confidence": 1.0}]


def test_a_presence_send_never_takes_the_voice_down():
    from reachy_sage_voice.pipeline import report_presence

    class Down:
        async def presence(self, state, confidence):
            raise KernelUnavailableError("refused")
    asyncio.run(report_presence(Down(), {"kind": "presence", "state": "seen",
                                         "confidence": 1.0}))
