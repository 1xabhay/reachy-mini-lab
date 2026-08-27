"""Headless diagnostics for Reachy Mini.

Run it with no flags for a read-only health report; add `--motion` to include
the calibration sweeps and `--vision` for the camera cross-check.  Motion is
opt-in because this moves a physical robot, and a diagnostic tool should not
surprise anyone standing next to it.

Exit code is 0 when every check passed and 1 otherwise, so it drops straight
into CI.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from typing import Any

import numpy as np
import requests
from reachy_mini import ReachyMini

from . import calibration, store
from .calibration import Check
from .daemon_api import DEFAULT_HOST, DEFAULT_PORT, DaemonClient
from .rotation_test import camera_matrix, validate_rotation

EXPECTED_MOTORS = 9  # body_rotation + stewart_1..6 + two antennas


def check_daemon(client: DaemonClient) -> tuple[Check, dict[str, Any] | None]:
    """Confirm the daemon answers and report what it says about itself."""
    if not client.reachable():
        return (
            Check(
                name="daemon",
                passed=False,
                detail=f"no response from {client.base} - is the robot powered and on the network?",
            ),
            None,
        )

    status = client.status()
    running = status.get("state") == "running"
    return (
        Check(
            name="daemon",
            passed=True,
            detail=(
                f"v{status.get('version')} at {status.get('wlan_ip')}, "
                f"hardware {status.get('hardware_id')}, backend {status.get('state')}"
                + ("" if running else " (start it to reach the motors)")
            ),
            data=status,
        ),
        status,
    )


def check_joints(mini: ReachyMini) -> Check:
    """Read back every joint and confirm the expected motor count is present."""
    head, antennas = mini.get_current_joint_positions()
    pose = calibration.decompose(mini.get_current_head_pose())
    total = len(head) + len(antennas)

    return Check(
        name="joints",
        passed=total == EXPECTED_MOTORS,
        detail=f"{total} joints reporting (expected {EXPECTED_MOTORS})",
        data={
            # head is [stewart_1..6, body_rotation]; keep raw radians alongside
            # the human-readable pose so CI can diff either representation.
            "head_joints_rad": [float(v) for v in head],
            "antennas_rad": [float(v) for v in antennas],
            "head_pose": pose,
        },
    )


def check_imu(mini: ReachyMini) -> Check:
    """Sanity-check the IMU: gravity should read about 1 g while sitting still."""
    imu = mini.imu
    if imu is None:
        return Check(
            name="imu",
            passed=True,
            detail="no IMU on this unit (Lite version) - skipped",
            data={"present": False},
        )

    accel = np.asarray(imu["accelerometer"], dtype=float)
    magnitude = float(np.linalg.norm(accel))
    passed = 8.0 <= magnitude <= 11.5
    return Check(
        name="imu",
        passed=passed,
        detail=f"|accel| {magnitude:.2f} m/s^2 (expect ~9.81 at rest), {imu.get('temperature')} C",
        data={"present": True, "magnitude": magnitude, **{k: v for k, v in imu.items()}},
    )


def check_camera(mini: ReachyMini, frames: int = 30) -> Check:
    """Pull a burst of frames to confirm the stream is live and measure its rate."""
    start = time.time()
    received = 0
    shape: tuple[int, ...] | None = None

    deadline = start + 10.0
    while received < frames and time.time() < deadline:
        frame = mini.media.get_frame()
        if frame is None:
            time.sleep(0.02)
            continue
        shape = frame.shape
        received += 1

    elapsed = time.time() - start
    fps = received / elapsed if elapsed > 0 else 0.0
    passed = received >= frames

    return Check(
        name="camera",
        passed=passed,
        detail=(
            f"{received}/{frames} frames in {elapsed:.1f}s ({fps:.1f} fps)"
            + (f", {shape[1]}x{shape[0]}" if shape else ", no frame decoded")
        ),
        data={"frames": received, "fps": fps, "shape": list(shape) if shape else None},
    )


def check_audio(mini: ReachyMini, seconds: float = 1.5) -> Check:
    """Record briefly and confirm the microphone delivers non-silent samples."""
    rate = mini.media.get_input_audio_samplerate()
    channels = mini.media.get_input_channels()
    if rate <= 0:
        return Check(name="audio", passed=False, detail="no audio input device reported")

    mini.media.start_recording()
    chunks: list[np.ndarray] = []
    deadline = time.time() + seconds
    try:
        while time.time() < deadline:
            sample = mini.media.get_audio_sample()
            if sample is None:
                time.sleep(0.01)
                continue
            chunks.append(np.asarray(sample, dtype=np.float32))
    finally:
        mini.media.stop_recording()

    if not chunks:
        return Check(
            name="audio",
            passed=False,
            detail=f"{rate} Hz / {channels} ch reported, but no samples arrived",
            data={"samplerate": rate, "channels": channels},
        )

    audio = np.concatenate([c.reshape(-1) for c in chunks])
    rms = float(np.sqrt(np.mean(audio**2)))
    peak = float(np.max(np.abs(audio)))
    # A dead mic reads as exact digital silence; room tone always has something.
    passed = peak > 1e-5

    return Check(
        name="audio",
        passed=passed,
        detail=f"{rate} Hz / {channels} ch, {audio.size} samples, rms {rms:.5f}, peak {peak:.5f}",
        data={
            "samplerate": rate,
            "channels": channels,
            "samples": int(audio.size),
            "rms": rms,
            "peak": peak,
        },
    )


def enable_torque(mini: ReachyMini, client: DaemonClient) -> Check:
    """Enable motor torque before any motion check, reporting the prior mode.

    A freshly started backend leaves the motors limp, and a limp head tracks
    nothing - the sweeps would report enormous errors and a confident FAIL that
    says nothing about calibration.
    """
    try:
        before = client.get("/api/motors/status").get("mode")
    except requests.RequestException as exc:
        # The prior mode is advisory; failing to read it must not abort the run.
        before = f"unknown ({exc})"

    mini.enable_motors()
    time.sleep(0.5)
    after = client.get("/api/motors/status").get("mode")

    return Check(
        name="motor_torque",
        passed=after == "enabled",
        detail=f"control mode {before} -> {after}",
        data={"before": before, "after": after},
    )


def check_storage() -> Check:
    """Make sure there is room to write captures and recordings."""
    free = store.disk_free_mb()
    return Check(
        name="storage",
        passed=free > 100,
        detail=f"{free:.0f} MiB free at {store.ROOT}",
        data={"free_mb": free, "root": str(store.ROOT)},
    )


def run(
    host: str,
    port: int,
    with_motion: bool,
    with_vision: bool,
    vision_angle: float,
) -> dict[str, Any]:
    """Run the selected checks and return a JSON-serialisable report."""
    client = DaemonClient(host=host, port=port)
    daemon_check, status = check_daemon(client)
    checks: list[Check] = [daemon_check, check_storage()]

    report: dict[str, Any] = {
        "host": host,
        "timestamp": store.stamp(),
        "daemon_status": status,
        "checks": [],
        "calibration": None,
        "rotation_validation": None,
    }

    if not daemon_check.passed:
        report["checks"] = [asdict(c) for c in checks]
        report["passed"] = False
        return report

    if not client.backend_running():
        checks.append(
            Check(
                name="backend",
                passed=False,
                detail="motor backend is stopped - start it from the dashboard "
                "or POST /api/daemon/start, then re-run",
            )
        )
        report["checks"] = [asdict(c) for c in checks]
        report["passed"] = False
        return report

    with ReachyMini(host=host, port=port, connection_mode="network") as mini:
        checks += [check_joints(mini), check_imu(mini), check_camera(mini), check_audio(mini)]

        if with_motion or with_vision:
            torque = enable_torque(mini, client)
            checks.append(torque)
            if not torque.passed:
                report["checks"] = [asdict(c) for c in checks]
                report["passed"] = False
                return report

        if with_motion:
            report["calibration"] = calibration.run_all(mini)

        if with_vision:
            report["rotation_validation"] = validate_rotation(
                mini, camera_matrix(mini), axis="yaw", angle_deg=vision_angle
            )

    report["checks"] = [asdict(c) for c in checks]
    report["passed"] = (
        all(c.passed for c in checks)
        and (report["calibration"] or {"passed": True})["passed"]
        and (report["rotation_validation"] or {"passed": True})["passed"]
    )
    return report


def render(report: dict[str, Any]) -> str:
    """Format a report for a terminal."""
    lines = [f"Reachy Mini diagnostics - {report['host']} - {report['timestamp']}", ""]

    def row(name: str, passed: bool, detail: str) -> str:
        return f"  [{'PASS' if passed else 'FAIL'}] {name:<24} {detail}"

    for c in report["checks"]:
        lines.append(row(c["name"], c["passed"], c["detail"]))

    if report.get("calibration"):
        lines += ["", "Calibration:"]
        for c in report["calibration"]["checks"]:
            lines.append(row(c["name"], c["passed"], c["detail"]))

    if report.get("rotation_validation"):
        rv = report["rotation_validation"]
        lines += [
            "",
            "Vision:",
            row("rotation_validation", rv.get("passed", False), rv.get("detail", "")),
        ]

    lines += ["", f"Overall: {'PASS' if report['passed'] else 'FAIL'}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (`reachy-diag`)."""
    parser = argparse.ArgumentParser(description="Reachy Mini diagnostics and calibration")
    parser.add_argument("--host", default=DEFAULT_HOST, help="daemon host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="daemon port")
    parser.add_argument(
        "--motion", action="store_true", help="run calibration sweeps (MOVES THE ROBOT)"
    )
    parser.add_argument(
        "--vision",
        action="store_true",
        help="run the camera rotation check (MOVES THE ROBOT)",
    )
    parser.add_argument(
        "--vision-angle",
        type=float,
        default=20.0,
        help="yaw angle for the camera check, in degrees",
    )
    parser.add_argument("--json", metavar="PATH", help="also write the full report as JSON")
    parser.add_argument("--quiet", action="store_true", help="only print the overall verdict")
    args = parser.parse_args(argv)

    report = run(args.host, args.port, args.motion, args.vision, args.vision_angle)

    print(f"Overall: {'PASS' if report['passed'] else 'FAIL'}" if args.quiet else render(report))

    if args.json:
        store.init()
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        if not args.quiet:
            print(f"\nreport written to {args.json}")

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
