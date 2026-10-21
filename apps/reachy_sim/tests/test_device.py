"""The sim as a Sage device: protocol v1 against the kernel and its conformance fake model."""

import ast
import importlib.util
import math
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import device
import sim

from conftest import API, api, present

APP = Path(__file__).parent.parent
VOICE_KERNEL = APP.parent / "reachy_sage_voice" / "reachy_sage_voice" / "kernel.py"
SAGE_KERNEL = APP.parents[2] / "sage-kernel"


def test_the_emote_vocabulary_is_the_voice_apps():
    tree = ast.parse(VOICE_KERNEL.read_text())
    voice = next(
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "EMOTIONS"
    )
    assert voice == device.EMOTIONS


def test_the_manifest_offers_one_emote_tool_lasting_the_longest_move_plus_a_second():
    durations = {name: 2.0 for name in device.EMOTIONS} | {"dance1": 7.3}
    (tool,) = device.manifest(durations)["tools"]
    assert tool["name"] == "emote"
    assert tool["inputSchema"]["properties"]["name"]["enum"] == list(device.EMOTIONS)
    assert tool["max_s"] == 9 and tool["interruptible"] is True


@pytest.mark.parametrize(
    ("effect", "action"),
    [
        ({"kind": "emote", "input": {"name": "cheerful1"}, "stale": False}, ("play", "cheerful1")),
        ({"kind": "emote", "input": {"name": "rage1"}, "stale": False}, ("failed", "unknown_move")),
        ({"kind": "emote", "input": {"name": "cheerful1"}, "stale": True}, ("failed", "stale")),
        ({"kind": "reply", "text": "Hi.", "stale": False}, ("done", None)),
        ({"kind": "reply", "text": "Hi.", "stale": True}, ("failed", "stale")),
        ({"kind": "wave", "input": {}, "stale": False}, ("failed", "unavailable")),
        ({"kind": "stop", "turn_id": None, "stale": False}, None),
    ],
)
def test_what_the_sim_does_with_each_effect(effect, action):
    assert device.plan(effect) == action


@pytest.fixture
def kernel():
    """Run the kernel from sage-kernel, thinking with the conformance suite's fake Ollama."""
    path = SAGE_KERNEL / "conformance" / "test_conformance.py"
    spec = importlib.util.spec_from_file_location("sage_conformance", path)
    conformance = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conformance)
    provider = conformance.Provider()
    with tempfile.TemporaryDirectory() as state:
        process = conformance.KernelProcess(state, provider).start()
        try:
            yield SimpleNamespace(
                url=f"http://127.0.0.1:{process.device_port}",
                token=conformance.TOKENS["reachy"],
                provider=provider,
                process=process,
                mock=conformance.TOKENS["mock"],
            )
        finally:
            process.stop()
            provider.server.shutdown()
            provider.server.server_close()


def test_an_emote_effect_moves_the_simulated_head_and_is_reported_done(kernel, neutral):
    samples, playing = [], threading.Event()

    def watch():
        while not playing.is_set():
            samples.append(present())
            time.sleep(0.1)

    watcher = threading.Thread(target=watch)
    watcher.start()
    try:
        results = device.turn(kernel.url, kernel.token, API, "Feel happy", "sim-1")
    finally:
        playing.set()
        watcher.join()

    by_kind = {r["kind"]: r for r in results}
    assert by_kind["emote"] == by_kind["emote"] | {"name": "cheerful1", "status": "done"}
    assert 2000 < by_kind["emote"]["duration_ms"] < 6000  # cheerful1 is 2.8 s
    # The kernel journals the move time under the effect's sequence.
    assert by_kind["emote"]["journaled_ms"] == by_kind["emote"]["duration_ms"]
    assert by_kind["reply"]["status"] == "done"
    # Moves play beside the poll, so the first text is not held back by the move.
    assert by_kind["reply"]["timing"]["ttft_ms"] < 1000

    def joints_delta(state):
        pairs = zip(state["head_joints"], neutral["head_joints"], strict=True)
        return max(abs(a - b) for a, b in pairs)

    assert math.degrees(max(map(joints_delta, samples))) > 5
    before = sim.eye_camera(neutral["head_joints"], neutral["antennas"])
    changes = [
        np.abs(sim.eye_camera(s["head_joints"], s["antennas"]).astype(int) - before).mean()
        for s in samples[::4]
    ]
    assert max(changes) > 5


def test_stop_from_another_device_interrupts_the_playing_move(kernel, neutral):
    def stop_once_moving():
        while not api("/move/running"):
            time.sleep(0.05)
        time.sleep(0.5)
        status, _ = kernel.process.device("POST", "/v1/events", {"kind": "stop"}, token=kernel.mock)
        assert status == 200

    stopper = threading.Thread(target=stop_once_moving)
    stopper.start()
    results = device.turn(kernel.url, kernel.token, API, "Feel happy", "sim-stop")
    stopper.join()

    (emote,) = [r for r in results if r["kind"] == "emote"]
    assert emote["status"] == "interrupted" and emote["duration_ms"] < 2000  # cheerful1 is 2.8 s
    assert not api("/move/running")


def test_a_busy_kernel_is_retried_with_the_same_turn_until_it_is_free(kernel, sim):
    kernel.provider.open.clear()  # the model holds every answer: the mock's turn runs on
    status, _ = kernel.process.device(
        "POST",
        "/v1/events",
        {"kind": "submitted", "turn_id": "mock-1", "text": "Hi"},
        token=kernel.mock,
    )
    assert status == 200
    threading.Timer(1.5, kernel.provider.open.set).start()

    results = device.turn(kernel.url, kernel.token, API, "Hello", "sim-busy")

    assert [r["kind"] for r in results] == ["reply"]


def test_a_kernel_busy_for_more_than_about_5_s_is_given_up(kernel, sim):
    kernel.provider.open.clear()
    kernel.process.device(
        "POST",
        "/v1/events",
        {"kind": "submitted", "turn_id": "mock-1", "text": "Hi"},
        token=kernel.mock,
    )
    start = time.monotonic()
    with pytest.raises(device.KernelBusyError):
        device.turn(kernel.url, kernel.token, API, "Hello", "sim-busy")
    assert 4.5 < time.monotonic() - start < 7
    kernel.provider.open.set()


def test_a_rate_limited_event_is_resent_after_retry_after():
    answers = [(429, {"code": "rate_limited", "retry_after": 0.3}), (200, {"sequence": 7})]
    slept = []
    link = device.Kernel("http://unused", "t", sleep=slept.append)
    link.call = lambda method, path, body=None: answers.pop(0)

    assert link.event({"kind": "connected"}) == (200, {"sequence": 7})
    assert slept == [0.3]


def test_a_newer_emote_supersedes_the_playing_one(sim, neutral):
    reported = []
    body = device.Body(API, lambda record, event: reported.append((record, event)))
    body.emote({"sequence": 1}, "cheerful1")
    while not api("/move/running"):
        time.sleep(0.05)
    time.sleep(0.5)
    body.emote({"sequence": 2}, "yes1")
    body.wait()

    (old, old_event), (new, new_event) = sorted(reported, key=lambda r: r[1]["sequence"])
    assert old_event | {"status": "interrupted", "reason": "superseded"} == old_event
    assert old["duration_ms"] < 2000 and old_event["duration_ms"] == old["duration_ms"]
    assert new_event["status"] == "done" and new["name"] == "yes1"
    assert new["duration_ms"] > 3000  # yes1 is 3.4 s: played in full
