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


def test_an_emote_reconnects_too():
    pet, mini, clients = robot()
    asyncio.run(pet.async_play_move("cheerful1"))
    assert len(clients) == 1 and mini.calls[-1] == ("move", "cheerful1")


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
