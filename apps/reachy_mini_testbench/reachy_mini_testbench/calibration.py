"""Calibration and kinematic health checks for Reachy Mini.

Everything here is *non-destructive*: it commands poses through the normal
motion API and compares them against what the robot reports back.  Nothing
writes to motor EEPROM or to `hardware_config.yaml`; use the vendor tools in
`reachy_mini.tools` for that.

The three checks answer three different questions:

* `zero_offset`    - does "neutral" actually look neutral?
* `axis_sweep`     - is each rotation axis tracking with the right gain, and
                     how much backlash is in the linkage?
* `antenna_check`  - do the two antenna servos agree with their commands?
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
from scipy.spatial.transform import Rotation

Axis = Literal["roll", "pitch", "yaw"]
AXES: tuple[Axis, ...] = ("roll", "pitch", "yaw")

#: Default per-axis sweep amplitude in degrees.  Deliberately conservative -
#: the head has far more range than this, but a small sweep keeps the test
#: quick and stays clear of the Stewart platform's singular corners.
DEFAULT_AMPLITUDE_DEG = 15.0

# Tolerances used to turn measurements into PASS/FAIL.
#
# These are NOT vendor specification - Pollen publish no positioning-accuracy
# figure. They are the numbers you would want from a head that can be pointed
# open-loop. Measurements from unit 2f102f4682d69822 (see README) sit well
# outside several of them, so treat a FAIL as "worth understanding", not
# "return the robot". Retune them for your own unit once you know its baseline;
# `TOL_REPEAT_DEG` is the one that reflects a genuine fault if exceeded.
TOL_ZERO_DEG = 2.0
TOL_ZERO_MM = 3.0
TOL_RMS_DEG = 2.5
TOL_GAIN = 0.15  # |gain - 1| must stay under this
TOL_BACKLASH_DEG = 2.0
TOL_REPEAT_DEG = 0.5


def decompose(pose: np.ndarray) -> dict[str, float]:
    """Split a 4x4 head pose into translation (mm) and roll/pitch/yaw (deg).

    Uses the same extrinsic "xyz" convention as `create_head_pose`, so the
    output can be compared directly against what was commanded.
    """
    roll, pitch, yaw = Rotation.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True)
    x, y, z = pose[:3, 3] * 1000.0
    return {
        "x_mm": float(x),
        "y_mm": float(y),
        "z_mm": float(z),
        "roll": float(roll),
        "pitch": float(pitch),
        "yaw": float(yaw),
    }


def _settled_pose(mini: ReachyMini, settle: float, samples: int = 5) -> dict[str, float]:
    """Wait for the head to settle, then average a few pose readings."""
    time.sleep(settle)
    poses = []
    for _ in range(samples):
        poses.append(decompose(mini.get_current_head_pose()))
        time.sleep(0.02)
    keys = poses[0].keys()
    return {k: float(np.mean([p[k] for p in poses])) for k in keys}


@dataclass
class Check:
    """One named measurement with a verdict attached."""

    name: str
    passed: bool
    detail: str
    data: dict[str, Any] = field(default_factory=dict)


def zero_offset(mini: ReachyMini, settle: float = 0.4) -> Check:
    """Command the neutral pose and measure how far the head actually sits from it.

    A persistent offset here usually means a motor zero is off, or the head
    was assembled with a rod at the wrong length.
    """
    mini.goto_target(head=create_head_pose(), body_yaw=0.0, duration=1.0)
    measured = _settled_pose(mini, settle)

    ang_err = {a: measured[a] for a in AXES}
    lin_err = {k: measured[k] for k in ("x_mm", "y_mm", "z_mm")}
    worst_ang = max(abs(v) for v in ang_err.values())
    worst_lin = max(abs(v) for v in lin_err.values())

    passed = worst_ang <= TOL_ZERO_DEG and worst_lin <= TOL_ZERO_MM
    return Check(
        name="zero_offset",
        passed=passed,
        detail=(
            f"worst angular residual {worst_ang:.2f} deg (tol {TOL_ZERO_DEG}), "
            f"worst linear residual {worst_lin:.2f} mm (tol {TOL_ZERO_MM})"
        ),
        data={"measured": measured, "worst_angular_deg": worst_ang, "worst_linear_mm": worst_lin},
    )


def axis_sweep(
    mini: ReachyMini,
    axis: Axis,
    amplitude_deg: float = DEFAULT_AMPLITUDE_DEG,
    steps: int = 5,
    duration: float = 0.6,
    settle: float = 0.35,
) -> Check:
    """Sweep one rotation axis up and back down, comparing command to feedback.

    Returns gain and offset from a least-squares fit, the RMS tracking error,
    and the mean up-vs-down difference (backlash) at matching setpoints.
    """
    up = list(np.linspace(-amplitude_deg, amplitude_deg, steps))
    down = list(reversed(up))

    def _run(setpoints: list[float]) -> list[dict[str, float]]:
        out = []
        for target in setpoints:
            mini.goto_target(
                head=create_head_pose(**{axis: target}), body_yaw=0.0, duration=duration
            )
            out.append(_settled_pose(mini, settle))
        return out

    up_meas = _run(up)
    down_meas = _run(down)
    mini.goto_target(head=create_head_pose(), body_yaw=0.0, duration=duration)

    cmd = np.array(up + down)
    act = np.array([m[axis] for m in up_meas + down_meas])

    gain, offset = np.polyfit(cmd, act, 1)
    residual = act - cmd
    rms = float(np.sqrt(np.mean(residual**2)))
    max_err = float(np.max(np.abs(residual)))

    # down_meas walks the same setpoints in reverse, so index i pairs with -1-i.
    backlash = float(
        np.mean([abs(up_meas[i][axis] - down_meas[-1 - i][axis]) for i in range(steps)])
    )

    passed = rms <= TOL_RMS_DEG and abs(gain - 1.0) <= TOL_GAIN and backlash <= TOL_BACKLASH_DEG
    return Check(
        name=f"axis_sweep_{axis}",
        passed=passed,
        detail=(
            f"gain {gain:.3f} (tol 1±{TOL_GAIN}), offset {offset:+.2f} deg, "
            f"rms {rms:.2f} deg (tol {TOL_RMS_DEG}), backlash {backlash:.2f} deg "
            f"(tol {TOL_BACKLASH_DEG})"
        ),
        data={
            "axis": axis,
            "amplitude_deg": amplitude_deg,
            "gain": float(gain),
            "offset_deg": float(offset),
            "rms_deg": rms,
            "max_error_deg": max_err,
            "backlash_deg": backlash,
            "commanded_deg": [float(c) for c in cmd],
            "measured_deg": [float(a) for a in act],
        },
    )


def antenna_check(
    mini: ReachyMini,
    amplitude_deg: float = 30.0,
    steps: int = 3,
    duration: float = 0.5,
    settle: float = 0.3,
) -> Check:
    """Command both antennas over a small range and compare against feedback."""
    setpoints = list(np.linspace(-amplitude_deg, amplitude_deg, steps))
    commanded: list[list[float]] = []
    measured: list[list[float]] = []

    for target in setpoints:
        # Mirrored so a swapped-cable fault shows up as a sign error, not a tie.
        pair = [target, -target]
        mini.goto_target(antennas=np.deg2rad(pair), duration=duration)
        time.sleep(settle)
        commanded.append(pair)
        measured.append(list(np.rad2deg(mini.get_present_antenna_joint_positions())))

    mini.goto_target(antennas=[0.0, 0.0], duration=duration)

    err = np.array(measured) - np.array(commanded)
    per_antenna_rms = [float(np.sqrt(np.mean(err[:, i] ** 2))) for i in range(2)]
    passed = all(e <= TOL_RMS_DEG for e in per_antenna_rms)

    return Check(
        name="antennas",
        passed=passed,
        detail=(
            f"rms error left {per_antenna_rms[0]:.2f} deg, "
            f"right {per_antenna_rms[1]:.2f} deg (tol {TOL_RMS_DEG})"
        ),
        data={
            "commanded_deg": commanded,
            "measured_deg": measured,
            "rms_deg": per_antenna_rms,
        },
    )


def repeatability(
    mini: ReachyMini,
    axis: Axis = "roll",
    setpoint: float = 0.0,
    approach_from: float = DEFAULT_AMPLITUDE_DEG,
    trials: int = 3,
    duration: float = 0.8,
    settle: float = 0.6,
) -> Check:
    """Return to one setpoint repeatedly from a fixed direction and measure the spread.

    This is the check that separates a robot which is merely *inaccurate* from
    one which is *erratic*. A head that lands 4 degrees off but does so within a
    tenth of a degree every time is correctable with a calibration map;
    one that scatters is not. Accuracy against the command is deliberately not
    judged here - `axis_sweep` covers that.
    """
    landings: list[float] = []
    for _ in range(trials):
        mini.goto_target(
            head=create_head_pose(**{axis: approach_from}), body_yaw=0.0, duration=duration
        )
        time.sleep(settle)
        mini.goto_target(head=create_head_pose(**{axis: setpoint}), body_yaw=0.0, duration=duration)
        landings.append(_settled_pose(mini, settle)[axis])

    mini.goto_target(head=create_head_pose(), body_yaw=0.0, duration=duration)

    spread = float(np.max(landings) - np.min(landings))
    mean = float(np.mean(landings))
    return Check(
        name=f"repeatability_{axis}",
        passed=spread <= TOL_REPEAT_DEG,
        detail=(
            f"{trials} approaches from {approach_from:+.1f} deg landed within "
            f"{spread:.2f} deg of each other (tol {TOL_REPEAT_DEG}), "
            f"mean {mean:+.2f} vs commanded {setpoint:+.1f}"
        ),
        data={
            "axis": axis,
            "setpoint_deg": setpoint,
            "approach_from_deg": approach_from,
            "landings_deg": landings,
            "spread_deg": spread,
            "mean_deg": mean,
            "offset_from_command_deg": mean - setpoint,
        },
    )


def run_all(
    mini: ReachyMini,
    axes: tuple[Axis, ...] = AXES,
    amplitude_deg: float = DEFAULT_AMPLITUDE_DEG,
    steps: int = 5,
    with_antennas: bool = True,
) -> dict[str, Any]:
    """Run the full calibration suite and return a JSON-serialisable report."""
    checks: list[Check] = [zero_offset(mini)]
    for axis in axes:
        checks.append(axis_sweep(mini, axis, amplitude_deg=amplitude_deg, steps=steps))
    if axes:
        checks.append(repeatability(mini, axis=axes[0], approach_from=amplitude_deg))
    if with_antennas:
        checks.append(antenna_check(mini))

    return {
        "passed": all(c.passed for c in checks),
        "checks": [asdict(c) for c in checks],
    }
