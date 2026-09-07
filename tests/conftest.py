"""Shared fixtures: a stub robot and the testbench API wired to it.

The stub implements only the slice of `ReachyMini` the app touches, which
lets the whole HTTP surface be exercised on a laptop with no robot attached.
"""

import threading
import time
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from reachy_mini_testbench import store
from reachy_mini_testbench.main import TestbenchApp

SAMPLERATE = 16000
CHANNELS = 2


class FakeMedia:
    """Stands in for `MediaManager`."""

    def __init__(self) -> None:
        self.frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.circle(self.frame, (160, 120), 60, (255, 255, 255), -1)
        self.recording = False
        self.played: list[str] = []
        self.camera = type("Cam", (), {"K": np.eye(3)})()

    def get_frame(self) -> np.ndarray:
        return self.frame

    def get_frame_jpeg(self) -> bytes:
        return cv2.imencode(".jpg", self.frame)[1].tobytes()

    def start_recording(self) -> None:
        self.recording = True

    def stop_recording(self) -> None:
        self.recording = False

    def get_audio_sample(self) -> np.ndarray | None:
        if not self.recording:
            return None
        time.sleep(0.001)
        return np.full(CHANNELS * 64, 0.25, dtype=np.float32)

    def get_input_audio_samplerate(self) -> int:
        return SAMPLERATE

    def get_input_channels(self) -> int:
        return CHANNELS

    def play_sound(self, path: str) -> None:
        self.played.append(path)


class FakeMini:
    """Stands in for `ReachyMini`."""

    def __init__(self) -> None:
        self.media = FakeMedia()
        self.head_pose = np.eye(4)
        self.antennas = np.zeros(2)
        self.calls: list[tuple[str, Any]] = []

    def get_current_head_pose(self) -> np.ndarray:
        return self.head_pose

    def get_present_antenna_joint_positions(self) -> np.ndarray:
        return self.antennas

    def get_current_joint_positions(self) -> tuple[list[float], list[float]]:
        return [0.0] * 7, list(self.antennas)

    @property
    def imu(self) -> dict[str, Any]:
        return {
            "accelerometer": [0.0, 0.0, 9.81],
            "gyroscope": [0.0, 0.0, 0.0],
            "quaternion": [1.0, 0.0, 0.0, 0.0],
            "temperature": 31.5,
        }

    def goto_target(self, head=None, antennas=None, duration=0.5, **kw) -> None:
        self.calls.append(("goto_target", {"head": head, "antennas": antennas, **kw}))
        if head is not None:
            self.head_pose = np.asarray(head)
        if antennas is not None:
            self.antennas = np.asarray(antennas, dtype=float)

    def wake_up(self) -> None:
        self.calls.append(("wake_up", None))

    def goto_sleep(self) -> None:
        self.calls.append(("goto_sleep", None))


@pytest.fixture
def robot() -> FakeMini:
    """Build a stub robot whose recorded calls the tests can assert against."""
    return FakeMini()


@pytest.fixture
def client(robot: FakeMini, tmp_path, monkeypatch) -> TestClient:
    """Serve the testbench API, backed by the stub robot and a temp storage tree."""
    monkeypatch.setattr(store, "ROOT", tmp_path)
    monkeypatch.setattr(store, "CAPTURES", tmp_path / "captures")
    monkeypatch.setattr(store, "RECORDINGS", tmp_path / "recordings")
    monkeypatch.setattr(store, "REPORTS", tmp_path / "reports")

    app = TestbenchApp()
    stop_event = threading.Event()
    worker = threading.Thread(target=app.run, args=(robot, stop_event), daemon=True)
    worker.start()

    # run() registers the routes and then blocks on stop_event.
    deadline = time.time() + 5.0
    while not any(getattr(r, "path", "") == "/api/status" for r in app.settings_app.routes):
        if time.time() > deadline:
            raise RuntimeError("routes were never registered")
        time.sleep(0.01)

    with TestClient(app.settings_app) as test_client:
        yield test_client

    stop_event.set()
    worker.join(timeout=5.0)


# --------------------------------------------------------------------------
# Desk pet fixtures
#
# The pet reaches the outside world through four seams - the mic/speaker, the
# HuggingFace inference client, the Anthropic client, and the emotions library.
# Each gets a stub that records what it was asked to do, so a whole
# conversation can be played through the real loop with nothing plugged in.
# --------------------------------------------------------------------------

from types import SimpleNamespace  # noqa: E402

from reachy_mini_pet import live  # noqa: E402


class FakePetMedia:
    """A mic playing a scripted timeline, and a speaker that remembers what it played.

    A `None` in the script means "the mic buffer is empty right now", which is
    what the real `get_audio_sample` returns between callbacks. The pet relies
    on that to know when it has finished discarding its own echo, so the stub
    has to reproduce it rather than handing back frames forever.
    """

    def __init__(self, script=None, samplerate=16000, stop_event=None, idle_polls=0):
        self.script = list(script or [])
        self.samplerate = samplerate
        self.stop_event = stop_event
        self.idle_polls = idle_polls
        self.recording = False
        self.played: list[str] = []
        self.polls = 0
        self.exhausted_polls = 0

    def start_recording(self) -> None:
        self.recording = True

    def stop_recording(self) -> None:
        self.recording = False

    def get_input_audio_samplerate(self) -> int:
        return self.samplerate

    def get_audio_sample(self):
        self.polls += 1
        if self.script:
            return self.script.pop(0)
        # Script over: idle for a while so the face/doze logic gets a turn,
        # then let the app's loop finish.
        self.exhausted_polls += 1
        if self.stop_event is not None and self.exhausted_polls > self.idle_polls:
            self.stop_event.set()
        return None

    def play_sound(self, path: str) -> None:
        self.played.append(path)


class FakePetMini:
    """Stands in for `ReachyMini`, recording every call in order."""

    def __init__(self, media=None):
        self.media = media if media is not None else FakePetMedia()
        self.calls: list[tuple[str, Any]] = []
        self.face_detected = False
        self.moves_played: list[str] = []

    def _record(self, name: str, payload: Any = None) -> None:
        self.calls.append((name, payload))

    def names(self) -> list[str]:
        """Just the call names, for asserting on ordering."""
        return [name for name, _ in self.calls]

    def wake_up(self) -> None:
        self._record("wake_up")

    def goto_sleep(self) -> None:
        self._record("goto_sleep")

    def enable_wobbling(self) -> None:
        self._record("enable_wobbling")

    def disable_wobbling(self) -> None:
        self._record("disable_wobbling")

    def start_head_tracking(self, weight: float = 1.0) -> None:
        self._record("start_head_tracking", weight)

    def stop_head_tracking(self) -> None:
        self._record("stop_head_tracking")

    def get_tracked_face(self, wait: bool = True, timeout: float = 5.0):
        return SimpleNamespace(detected=self.face_detected)

    def play_move(self, move, sound: bool = True) -> None:
        self._record("play_move", (move.name, sound))
        self.moves_played.append(move.name)


class FakeMoves:
    """Stands in for the recorded emotions library."""

    def __init__(self, names=None):
        self.names = list(names if names is not None else live.EMOTIONS)

    def list_moves(self) -> list[str]:
        return list(self.names)

    def get(self, name: str):
        if name not in self.names:
            raise ValueError(f"Move {name} not found")
        return SimpleNamespace(name=name)
