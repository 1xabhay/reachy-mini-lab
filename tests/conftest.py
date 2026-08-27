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
