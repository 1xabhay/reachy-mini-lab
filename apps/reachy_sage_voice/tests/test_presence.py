"""Presence from the daemon's face tracker: an event on seen and lost edges only.

A dropped frame is not someone leaving: the last sighting is held, then faded
(as the pet's FaceMemory does), and only then is "lost" reported. Pure, with
time passed in; the send sits behind one function.
"""

import asyncio

from reachy_sage_voice.presence import Presence, watch_presence


def run(presence, frames):
    """frames: (now, detected) pairs; returns the events, with when they fired."""
    return [(now, e) for now, seen in frames if (e := presence.update(seen, now))]


def test_two_sightings_in_a_row_are_seen():
    assert run(Presence(), [(0.0, True), (0.2, True)]) == [
        (0.2, {"kind": "presence", "state": "seen", "confidence": 1.0})]


def test_one_frame_is_not_someone_arriving():
    # Arrivals trigger greetings: a one-frame false positive must not greet, then leave.
    assert run(Presence(), [(0.0, True), *[(t / 5, False) for t in range(1, 60)]]) == []
    assert run(Presence(), [(0.0, True), (0.2, False), (0.4, True), (0.6, False)]) == []


def test_seen_and_lost_sit_either_side_of_the_kernels_half_confidence():
    events = [e for _, e in run(Presence(), [(0.0, True), (0.2, True),
                                             *[(t / 2, False) for t in range(1, 20)]])]
    assert [e["state"] for e in events] == ["seen", "lost"]
    assert events[0]["confidence"] >= 0.5 > events[1]["confidence"]


def test_nothing_more_while_someone_stays():
    assert len(run(Presence(), [(t / 10, True) for t in range(50)])) == 1


def test_a_dropped_frame_is_not_a_loss():
    frames = [(0.0, True), (0.05, True), (0.1, False), (0.2, False), (0.3, True), (1.0, False)]
    assert [e["state"] for _, e in run(Presence(), frames)] == ["seen"]


def test_lost_only_after_the_hold_and_the_fade():
    p = Presence(hold_s=2.0, fade_s=6.0)
    events = run(p, [(-0.2, True), (0.0, True), *[(t / 2, False) for t in range(1, 20)]])
    assert [(now, e["state"], e["confidence"]) for now, e in events] == [
        (0.0, "seen", 1.0), (8.0, "lost", 0.0)]


def test_seen_again_after_being_lost():
    frames = [(0.0, True), (0.2, True), (9.0, False), (10.0, True), (10.2, True)]
    assert [e["state"] for _, e in run(Presence(), frames)] == ["seen", "lost", "seen"]


def test_never_seen_is_never_lost():
    assert run(Presence(), [(t, False) for t in range(20)]) == []


def test_the_watch_sends_only_edges_through_one_function():
    faces = iter([True, True, False, True] + [False] * 100)
    clock = {"t": 0.0}
    sent = []

    async def sleep(s):
        clock["t"] += s

    async def send(event):
        sent.append(event["state"])

    asyncio.run(watch_presence(lambda: next(faces, None), send, lambda: clock["t"], sleep,
                               period_s=0.25, until=lambda: clock["t"] > 20))
    assert sent == ["seen", "lost"]
