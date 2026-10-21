"""A dropped command websocket heals itself: one fresh client, motors and tracker back on.

The SDK's WSClient never reconnects: once its keepalive ping times out every
command raises ConnectionError("Lost connection with the server.") for good.
"""

import asyncio

import pytest

from reachy_sage_voice.robot import Reconnecting


class Client:
    def __init__(self, alive=True):
        self.host, self.port, self.alive, self.closed = "reachy-mini.lan", 8000, alive, False

    def disconnect(self):
        self.closed = True

    def is_connected(self):
        return self.alive


class Mini:
    def __init__(self):
        self.client, self.calls, self.media = Client(alive=False), [], object()

    def _send(self, name):
        if not self.client.alive:
            raise ConnectionError("Lost connection with the server.")
        self.calls.append(name)
        return name

    def goto_target(self):
        return self._send("goto")

    def enable_motors(self):
        self._send("enable_motors")

    def start_head_tracking(self, weight=1.0):
        self._send(("track", weight))

    async def async_play_move(self, move):
        self._send(("move", move))

    def get_tracked_face(self, wait=True):  # the cached status: never raises, may be stale
        return type("Face", (), {"detected": True})()


def robot(now=lambda: 100.0):
    mini, clients = Mini(), []

    def connect(host, port):
        clients.append((host, port))
        return Client()

    return Reconnecting(mini, connect=connect, now=now), mini, clients


def test_a_lost_connection_reconnects_once_rearms_the_robot_and_retries():
    pet, mini, clients = robot()
    old = mini.client
    assert pet.goto_target() == "goto"
    assert clients == [("reachy-mini.lan", 8000)] and old.closed
    assert mini.calls == ["enable_motors", ("track", 1.0), "goto"]


def test_an_emote_reconnects_then_is_skipped_since_a_late_emote_is_wrong():
    pet, mini, clients = robot()
    assert asyncio.run(pet.async_play_move("cheerful1")) is None
    assert len(clients) == 1 and mini.calls == ["enable_motors", ("track", 1.0)]


def test_a_face_read_on_a_dead_connection_reconnects_and_reports_no_face():
    pet, mini, clients = robot()
    assert pet.get_tracked_face(wait=False).detected is False
    assert len(clients) == 1 and mini.calls == ["enable_motors", ("track", 1.0)]
    assert pet.get_tracked_face(wait=False).detected is True


def test_at_most_one_reconnect_per_ten_seconds():
    clock = [100.0]
    pet, mini, clients = robot(now=lambda: clock[0])
    pet.goto_target()
    mini.client.alive = False
    clock[0] += 5
    with pytest.raises(ConnectionError):
        pet.goto_target()
    clock[0] += 6
    assert pet.goto_target() == "goto" and len(clients) == 2


def test_everything_else_passes_through():
    pet, mini, _ = robot()
    assert pet.media is mini.media
