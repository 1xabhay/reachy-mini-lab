"""Reachy Mini debug and CI testbench.

Serves a browser toolbox for poking at a Reachy Mini: live status, manual
motion, camera and audio capture, plus the calibration and rotation-validation
suites that also back the `reachy-diag` CLI.

Run it as a Reachy Mini app (`reachy-mini app run reachy_mini_testbench`) or
directly (`python -m reachy_mini_testbench.main`), then open
http://localhost:8042.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import soundfile as sf
from fastapi import HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from reachy_mini import ReachyMini, ReachyMiniApp
from reachy_mini.utils import create_head_pose

from . import calibration, store
from .rotation_test import calibrate_visual_scale, camera_matrix, validate_rotation

PORT = 8042

#: Order the daemon reports head joints in - `[body_yaw] + stewart_1..6`
#: (`RobotBackend.get_all_joint_positions` returns `[yaw] + list(dofs)`).
HEAD_JOINT_NAMES = ("body_rotation", *(f"stewart_{i}" for i in range(1, 7)))

#: Joint angles (rad) that put the head at its neutral pose, from the SDK's
#: own constant in `ReachyMini.goto_sleep`.
NEUTRAL_HEAD_JOINTS = (
    0.0,
    0.5251518455536499,
    -0.668710345667336,
    0.6067086443974802,
    -0.606711497194891,
    0.6687148024583701,
    -0.5251586523105128,
)

#: Cap on browser-driven motion so a stray slider cannot fling the head.
MAX_HEAD_ANGLE_DEG = 40.0
MAX_HEAD_TRANSLATION_MM = 25.0
MAX_ANTENNA_ANGLE_DEG = 150.0
MAX_RECORDING_SECONDS = 120.0


class HeadTarget(BaseModel):
    """A head pose request from the UI, in degrees and millimetres."""

    roll: float = Field(0.0, ge=-MAX_HEAD_ANGLE_DEG, le=MAX_HEAD_ANGLE_DEG)
    pitch: float = Field(0.0, ge=-MAX_HEAD_ANGLE_DEG, le=MAX_HEAD_ANGLE_DEG)
    yaw: float = Field(0.0, ge=-MAX_HEAD_ANGLE_DEG, le=MAX_HEAD_ANGLE_DEG)
    x: float = Field(0.0, ge=-MAX_HEAD_TRANSLATION_MM, le=MAX_HEAD_TRANSLATION_MM)
    y: float = Field(0.0, ge=-MAX_HEAD_TRANSLATION_MM, le=MAX_HEAD_TRANSLATION_MM)
    z: float = Field(0.0, ge=-MAX_HEAD_TRANSLATION_MM, le=MAX_HEAD_TRANSLATION_MM)
    body_yaw: float = Field(0.0, ge=-MAX_HEAD_ANGLE_DEG, le=MAX_HEAD_ANGLE_DEG)
    duration: float = Field(0.5, gt=0.0, le=10.0)


class AntennaTarget(BaseModel):
    """An antenna request from the UI, in degrees."""

    left: float = Field(0.0, ge=-MAX_ANTENNA_ANGLE_DEG, le=MAX_ANTENNA_ANGLE_DEG)
    right: float = Field(0.0, ge=-MAX_ANTENNA_ANGLE_DEG, le=MAX_ANTENNA_ANGLE_DEG)
    duration: float = Field(0.5, gt=0.0, le=10.0)


class RotationTestRequest(BaseModel):
    """Parameters for one rotation-validation run."""

    axis: Literal["roll", "pitch", "yaw"] = "yaw"
    angle: float = Field(20.0, ge=-MAX_HEAD_ANGLE_DEG, le=MAX_HEAD_ANGLE_DEG)
    tolerance: float = Field(3.0, gt=0.0, le=15.0)


class CalibrationRequest(BaseModel):
    """Parameters for a calibration sweep."""

    axes: list[Literal["roll", "pitch", "yaw"]] = ["roll", "pitch", "yaw"]
    amplitude: float = Field(15.0, ge=5.0, le=MAX_HEAD_ANGLE_DEG)
    steps: int = Field(5, ge=3, le=11)
    antennas: bool = True


class RangeOfMotionRequest(BaseModel):
    """Parameters for a range-of-motion sweep."""

    axes: list[Literal["roll", "pitch", "yaw"]] = ["roll", "pitch", "yaw"]
    step: float = Field(5.0, ge=1.0, le=15.0)
    # None means "use the documented limit for this axis".
    limit: float | None = Field(None, ge=5.0, le=65.0)


class VisualScaleRequest(BaseModel):
    """Parameters for the camera-to-kinematics scale fit."""

    axis: Literal["pitch", "yaw"] = "yaw"
    angles: list[float] = [-20.0, -10.0, 10.0, 20.0]


class AudioRecorder:
    """Pulls microphone samples on a worker thread until asked to stop.

    `MediaManager.get_audio_sample()` is a non-blocking poll, so recording has
    to be driven by someone; doing it here keeps the HTTP handlers responsive
    and lets a recording outlive the request that started it.
    """

    def __init__(self, mini: ReachyMini) -> None:
        """Attach a recorder to a robot; nothing starts until `start()`."""
        self._mini = mini
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self.started_at: float | None = None

    @property
    def recording(self) -> bool:
        """Report whether the worker thread is running."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Begin recording. Raises if a recording is already in progress."""
        if self.recording:
            raise RuntimeError("already recording")
        self._stop.clear()
        with self._lock:
            self._chunks = []
        self.started_at = time.time()
        self._mini.media.start_recording()
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        deadline = time.time() + MAX_RECORDING_SECONDS
        while not self._stop.is_set() and time.time() < deadline:
            sample = self._mini.media.get_audio_sample()
            if sample is None:
                time.sleep(0.005)
                continue
            with self._lock:
                self._chunks.append(np.asarray(sample, dtype=np.float32))

    def stop(self) -> tuple[np.ndarray, int, int]:
        """Stop and return `(samples, samplerate, channels)`."""
        if not self.recording and not self._chunks:
            raise RuntimeError("not recording")
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._mini.media.stop_recording()
        self.started_at = None

        with self._lock:
            chunks, self._chunks = self._chunks, []

        channels = max(self._mini.media.get_input_channels(), 1)
        rate = self._mini.media.get_input_audio_samplerate()
        if not chunks:
            return np.zeros((0, channels), dtype=np.float32), rate, channels

        audio = np.concatenate([c.reshape(-1, channels) for c in chunks], axis=0)
        return audio, rate, channels


class TestbenchApp(ReachyMiniApp):
    """Web toolbox for debugging and validating a Reachy Mini."""

    custom_app_url: str | None = f"http://0.0.0.0:{PORT}"

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        """Register the HTTP API, then idle until asked to stop."""
        store.init()
        api = self.settings_app
        assert api is not None, "settings_app is created whenever custom_app_url is set"

        recorder = AudioRecorder(reachy_mini)
        # Everything that moves the robot takes this lock, so a manual slider
        # drag can never interleave with a running test sweep.
        robot_lock = threading.Lock()
        last_results: dict[str, Any] = {
            "rotation": None,
            "calibration": None,
            "visual_scale": None,
            "range_of_motion": None,
        }
        latest_frame: dict[str, np.ndarray] = {}

        def exclusive(fn: Callable[[], Any]) -> Any:
            """Run `fn` while holding the robot, or 409 if it is already busy."""
            if not robot_lock.acquire(blocking=False):
                raise HTTPException(409, "robot is busy with another movement or test")
            try:
                return fn()
            finally:
                robot_lock.release()

        # ---------------------------------------------------------------- status

        @api.get("/api/status")
        def status() -> dict[str, Any]:
            pose = calibration.decompose(reachy_mini.get_current_head_pose())
            return {
                "connected": True,
                "head_pose": pose,
                "antennas_deg": [
                    float(a) for a in np.rad2deg(reachy_mini.get_present_antenna_joint_positions())
                ],
                "imu": reachy_mini.imu,
                "recording": recorder.recording,
                "busy": robot_lock.locked(),
            }

        @api.get("/api/motor_status")
        def motor_status() -> dict[str, Any]:
            head, antennas = reachy_mini.get_current_joint_positions()
            motors = [
                {
                    "name": name,
                    "position_rad": float(v),
                    "position_deg": float(np.rad2deg(v)),
                    # How far this joint sits from where a neutral head puts it.
                    # A single leg with a large, persistent delta is the
                    # signature of a mis-indexed servo horn or a stalling motor.
                    "delta_from_neutral_deg": float(np.rad2deg(v - z)),
                }
                for name, v, z in zip(HEAD_JOINT_NAMES, head, NEUTRAL_HEAD_JOINTS, strict=True)
            ]
            motors += [
                {
                    "name": name,
                    "position_rad": float(v),
                    "position_deg": float(np.rad2deg(v)),
                    "delta_from_neutral_deg": float(np.rad2deg(v)),
                }
                for name, v in zip(("left_antenna", "right_antenna"), antennas, strict=True)
            ]
            return {"count": len(motors), "motors": motors}

        # -------------------------------------------------------------- movement

        @api.post("/api/move_head")
        def move_head(target: HeadTarget) -> dict[str, Any]:
            pose = create_head_pose(
                x=target.x,
                y=target.y,
                z=target.z,
                roll=target.roll,
                pitch=target.pitch,
                yaw=target.yaw,
                mm=True,
                degrees=True,
            )
            exclusive(
                lambda: reachy_mini.goto_target(
                    head=pose,
                    body_yaw=float(np.deg2rad(target.body_yaw)),
                    duration=target.duration,
                )
            )
            return {"ok": True}

        @api.post("/api/move_antennas")
        def move_antennas(target: AntennaTarget) -> dict[str, Any]:
            exclusive(
                lambda: reachy_mini.goto_target(
                    antennas=np.deg2rad([target.left, target.right]),
                    duration=target.duration,
                )
            )
            return {"ok": True}

        @api.post("/api/go_to_zero")
        def go_to_zero(duration: float = 1.0) -> dict[str, Any]:
            exclusive(
                lambda: reachy_mini.goto_target(
                    head=create_head_pose(),
                    antennas=[0.0, 0.0],
                    body_yaw=0.0,
                    duration=max(duration, 0.1),
                )
            )
            return {"ok": True}

        @api.post("/api/level_head")
        def level_head() -> dict[str, Any]:
            """Drive the head to a genuinely level neutral pose."""
            return exclusive(lambda: calibration.settle_to_pose(reachy_mini))

        @api.post("/api/wake_up")
        def wake_up() -> dict[str, Any]:
            exclusive(reachy_mini.wake_up)
            return {"ok": True}

        @api.post("/api/go_to_sleep")
        def go_to_sleep() -> dict[str, Any]:
            exclusive(reachy_mini.goto_sleep)
            return {"ok": True}

        # ---------------------------------------------------------------- camera

        def grab_frame() -> np.ndarray:
            frame = reachy_mini.media.get_frame()
            if frame is None:
                raise HTTPException(503, "no camera frame available")
            latest_frame["frame"] = frame
            return frame

        @api.get("/api/camera/stream")
        def camera_stream() -> StreamingResponse:
            def frames() -> Any:
                while not stop_event.is_set():
                    jpeg = reachy_mini.media.get_frame_jpeg()
                    if jpeg is None:
                        time.sleep(0.05)
                        continue
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"

            return StreamingResponse(
                frames(), media_type="multipart/x-mixed-replace; boundary=frame"
            )

        @api.get("/api/camera/capture")
        def camera_capture() -> Response:
            jpeg = reachy_mini.media.get_frame_jpeg()
            if jpeg is None:
                raise HTTPException(503, "no camera frame available")
            return Response(content=jpeg, media_type="image/jpeg")

        @api.post("/api/camera/save")
        def camera_save() -> dict[str, Any]:
            jpeg = reachy_mini.media.get_frame_jpeg()
            if jpeg is None:
                raise HTTPException(503, "no camera frame available")
            path = store.CAPTURES / f"capture_{store.stamp()}.jpg"
            path.write_bytes(jpeg)
            return {"filename": path.name, "size": path.stat().st_size}

        @api.get("/api/camera/list")
        def camera_list() -> dict[str, Any]:
            return {"captures": store.listing(store.CAPTURES, (".jpg", ".jpeg", ".png"))}

        @api.get("/api/camera/download/{filename}")
        def camera_download(filename: str) -> FileResponse:
            return _serve(store.CAPTURES, filename, "image/jpeg")

        @api.delete("/api/camera/delete/{filename}")
        def camera_delete(filename: str) -> dict[str, Any]:
            return _delete(store.CAPTURES, filename)

        # ----------------------------------------------------------------- audio

        @api.post("/api/audio/start_recording")
        def audio_start() -> dict[str, Any]:
            try:
                recorder.start()
            except RuntimeError as exc:
                raise HTTPException(409, str(exc)) from exc
            return {"recording": True, "max_seconds": MAX_RECORDING_SECONDS}

        @api.post("/api/audio/stop_recording")
        def audio_stop() -> dict[str, Any]:
            try:
                audio, rate, channels = recorder.stop()
            except RuntimeError as exc:
                raise HTTPException(409, str(exc)) from exc
            if audio.size == 0:
                raise HTTPException(503, "recording captured no audio")

            path = store.RECORDINGS / f"recording_{store.stamp()}.wav"
            sf.write(path, audio, rate)
            return {
                "filename": path.name,
                "seconds": audio.shape[0] / rate,
                "samplerate": rate,
                "channels": channels,
                "peak": float(np.max(np.abs(audio))),
            }

        @api.get("/api/audio/list")
        def audio_list() -> dict[str, Any]:
            return {"recordings": store.listing(store.RECORDINGS, (".wav",))}

        @api.get("/api/audio/download/{filename}")
        def audio_download(filename: str) -> FileResponse:
            return _serve(store.RECORDINGS, filename, "audio/wav")

        @api.post("/api/audio/play/{filename}")
        def audio_play(filename: str) -> dict[str, Any]:
            path = _resolve(store.RECORDINGS, filename)
            reachy_mini.media.play_sound(str(path))
            return {"playing": filename}

        @api.delete("/api/audio/delete/{filename}")
        def audio_delete(filename: str) -> dict[str, Any]:
            return _delete(store.RECORDINGS, filename)

        # ------------------------------------------------------- tests and calib

        @api.post("/api/test/rotation_validation")
        def rotation_validation(req: RotationTestRequest) -> dict[str, Any]:
            def go() -> dict[str, Any]:
                return validate_rotation(
                    reachy_mini,
                    camera_matrix(reachy_mini),
                    axis=req.axis,
                    angle_deg=req.angle,
                    tolerance_deg=req.tolerance,
                )

            result = exclusive(go)
            last_results["rotation"] = result
            return result

        @api.get("/api/test/last_rotation_result")
        def last_rotation_result() -> dict[str, Any]:
            return {"result": last_results["rotation"]}

        @api.post("/api/test/calibration")
        def run_calibration(req: CalibrationRequest) -> dict[str, Any]:
            def go() -> dict[str, Any]:
                return calibration.run_all(
                    reachy_mini,
                    axes=tuple(req.axes),
                    amplitude_deg=req.amplitude,
                    steps=req.steps,
                    with_antennas=req.antennas,
                )

            result = exclusive(go)
            last_results["calibration"] = result
            return result

        @api.post("/api/test/range_of_motion")
        def run_range_of_motion(req: RangeOfMotionRequest) -> dict[str, Any]:
            def go() -> dict[str, Any]:
                checks = [
                    calibration.range_of_motion(
                        reachy_mini, axis=axis, limit_deg=req.limit, step_deg=req.step
                    )
                    for axis in req.axes
                ]
                return {
                    "passed": all(c.passed for c in checks),
                    "checks": [asdict(c) for c in checks],
                }

            result = exclusive(go)
            last_results["range_of_motion"] = result
            return result

        @api.get("/api/test/last_range_of_motion_result")
        def last_range_of_motion_result() -> dict[str, Any]:
            return {"result": last_results["range_of_motion"]}

        @api.get("/api/test/last_calibration_result")
        def last_calibration_result() -> dict[str, Any]:
            return {"result": last_results["calibration"]}

        @api.post("/api/test/visual_scale")
        def visual_scale(req: VisualScaleRequest) -> dict[str, Any]:
            def go() -> dict[str, Any]:
                return calibrate_visual_scale(
                    reachy_mini,
                    camera_matrix(reachy_mini),
                    axis=req.axis,
                    angles_deg=tuple(req.angles),
                )

            result = exclusive(go)
            last_results["visual_scale"] = result
            return result

        self.logger.info("Testbench UI on http://localhost:%d", PORT)
        stop_event.wait()

        if recorder.recording:
            recorder.stop()


def _resolve(directory: Path, filename: str) -> Path:
    """Resolve a user-supplied filename inside `directory`, or raise 400/404."""
    try:
        path = store.safe_path(directory, filename)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not path.is_file():
        raise HTTPException(404, f"{filename} not found")
    return path


def _serve(directory: Path, filename: str, media_type: str) -> FileResponse:
    return FileResponse(_resolve(directory, filename), media_type=media_type, filename=filename)


def _delete(directory: Path, filename: str) -> dict[str, Any]:
    _resolve(directory, filename).unlink()
    return {"deleted": filename}


if __name__ == "__main__":
    app = TestbenchApp()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
