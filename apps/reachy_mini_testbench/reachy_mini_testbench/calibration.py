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

#: Documented head limits, from Pollen's troubleshooting FAQ ("What are the
#: safety limits (Head & Body)?"): head pitch and roll are +/-40 deg, and head
#: yaw must stay within 65 deg of body yaw. The daemon silently clamps anything
#: beyond these, so probing further measures the clamp rather than the robot.
ROM_LIMIT_DEG: dict[str, float] = {"roll": 40.0, "pitch": 40.0, "yaw": 65.0}

#: A direction is called exhausted once the head stops making at least this
#: fraction of each commanded step. Two such steps in a row end that sweep.
ROM_MIN_PROGRESS_FRACTION = 0.25

#: Fraction of the documented limit the head should actually reach.
TOL_ROM_FRACTION = 0.85

#: Allowed difference between the two directions of one axis.
TOL_ROM_ASYMMETRY_DEG = 8.0


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


def _goto(
    mini: ReachyMini,
    pose: np.ndarray,
    duration: float,
    body_yaw: float | None = 0.0,
) -> bool:
    """Command a pose, tolerating a missed completion notification.

    The SDK waits only `duration + 1s` for the daemon to acknowledge a goto and
    raises `TimeoutError` past that, which happens under load even though the
    move itself succeeds. Every caller here verifies the result by reading the
    pose back, so a lost acknowledgement is not worth aborting a sweep for.
    Returns False when the acknowledgement was missed.
    """
    try:
        mini.goto_target(head=pose, body_yaw=body_yaw, duration=duration)
        return True
    except TimeoutError:
        return False


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


POSE_KEYS = ("x_mm", "y_mm", "z_mm", "roll", "pitch", "yaw")


def _pose_from(components: dict[str, float]) -> np.ndarray:
    """Build a head pose from the same component names `decompose` returns."""
    return create_head_pose(
        x=components["x_mm"],
        y=components["y_mm"],
        z=components["z_mm"],
        roll=components["roll"],
        pitch=components["pitch"],
        yaw=components["yaw"],
        mm=True,
        degrees=True,
    )


def _present_body_yaw_deg(mini: ReachyMini) -> float:
    """Present body rotation in degrees.

    Body yaw is a separate degree of freedom and does *not* appear in the head
    pose, which the daemon reports relative to the body - so it has to be read
    from the joints, where it is element 0 of the head chain.
    """
    head_joints, _ = mini.get_current_joint_positions()
    return float(np.rad2deg(head_joints[0]))


def settle_to_pose(
    mini: ReachyMini,
    target: dict[str, float] | None = None,
    body_yaw_deg: float | None = 0.0,
    tolerance_deg: float = 1.0,
    tolerance_mm: float = 1.0,
    max_iterations: int = 6,
    gain: float = 0.8,
    duration: float = 0.6,
    settle: float = 0.35,
) -> dict[str, Any]:
    """Command a head pose, then correct whatever the servos leave behind.

    The Stewart motors are configured P-only (P=300, I=0, D=0 in the SDK's
    `hardware_config.yaml`), so each one stops where friction balances the
    proportional term and nothing ever removes the residual. The head therefore
    settles several degrees from the commanded pose, on whichever side it
    approached from.

    The robot repeats to a fraction of a degree though, so that residual is not
    noise - it is a measurable, correctable offset. Measuring it and folding it
    back into the command supplies, in software and per-pose, the integral
    action the servo loop does not have. `gain` below 1 keeps the correction
    from ringing between the two sides of the friction band.
    """
    goal = {k: 0.0 for k in POSE_KEYS}
    if target:
        unknown = set(target) - set(POSE_KEYS)
        if unknown:
            raise ValueError(f"unknown pose components: {sorted(unknown)}")
        goal.update(target)

    def _residual(error: dict[str, float]) -> tuple[float, float]:
        angular = [abs(error[k]) for k in AXES]
        if "body_yaw" in error:
            angular.append(abs(error["body_yaw"]))
        return (
            max(angular),
            max(abs(error[k]) for k in ("x_mm", "y_mm", "z_mm")),
        )

    command = dict(goal)
    # Body yaw rides alongside the head pose: same P-only problem, but it is a
    # separate joint and needs its own command and its own correction.
    body_goal = body_yaw_deg
    body_command = body_yaw_deg
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    missed_acks = 0
    for _ in range(max_iterations):
        body_rad = None if body_command is None else float(np.deg2rad(body_command))
        if not _goto(mini, _pose_from(command), duration, body_rad):
            missed_acks += 1
        achieved = _settled_pose(mini, settle)
        error = {k: achieved[k] - goal[k] for k in POSE_KEYS}
        if body_goal is not None:
            achieved = {**achieved, "body_yaw": _present_body_yaw_deg(mini)}
            error["body_yaw"] = achieved["body_yaw"] - body_goal
        worst_ang, worst_lin = _residual(error)
        step = {
            "command": dict(command),
            "body_command_deg": body_command,
            "achieved": dict(achieved),
            "error": error,
            "worst_angular_deg": worst_ang,
            "worst_linear_mm": worst_lin,
        }
        history.append(step)

        # Corrections ring inside the friction band rather than converging onto
        # it, so the last attempt is not reliably the best one - keep the best.
        if best is None or worst_ang < best["worst_angular_deg"]:
            best = step

        if worst_ang <= tolerance_deg and worst_lin <= tolerance_mm:
            break

        # Push the command the other way by most of the observed error.
        command = {k: command[k] - gain * error[k] for k in POSE_KEYS}
        if body_command is not None:
            body_command -= gain * error["body_yaw"]

    assert best is not None
    if history[-1] is not best:
        # Leave the head at the best pose found, not at the last one tried.
        best_body = best["body_command_deg"]
        _goto(
            mini,
            _pose_from(best["command"]),
            duration,
            None if best_body is None else float(np.deg2rad(best_body)),
        )
        _settled_pose(mini, settle)

    return {
        "converged": best["worst_angular_deg"] <= tolerance_deg
        and best["worst_linear_mm"] <= tolerance_mm,
        "iterations": len(history),
        "goal": goal if body_goal is None else {**goal, "body_yaw": body_goal},
        "achieved": best["achieved"],
        "worst_angular_deg": best["worst_angular_deg"],
        "worst_linear_mm": best["worst_linear_mm"],
        "missed_acknowledgements": missed_acks,
        "history": history,
    }


def level_head(mini: ReachyMini, **kwargs: Any) -> Check:
    """Drive the head to a genuinely level neutral pose and report the residual."""
    result = settle_to_pose(mini, **kwargs)
    return Check(
        name="level_head",
        passed=bool(result["converged"]),
        detail=(
            f"settled to {result['worst_angular_deg']:.2f} deg / "
            f"{result['worst_linear_mm']:.2f} mm of neutral "
            f"in {result['iterations']} iteration(s)"
        ),
        data=result,
    )


def range_of_motion(
    mini: ReachyMini,
    axis: Axis = "yaw",
    limit_deg: float | None = None,
    step_deg: float = 5.0,
    duration: float = 0.5,
    settle: float = 0.3,
    body_yaw: float | None = 0.0,
) -> Check:
    """Walk one axis out to its limit in both directions and measure what it reaches.

    Range is measured as the span the head *achieves*, not as per-point
    accuracy, which makes it immune to the constant offset and hysteresis that
    dominate this robot's pose error. Each direction stops as soon as the head
    stops making progress, so a mechanically blocked axis is detected instead
    of being driven into its end stop - Pollen's motor guide warns that a head
    which cannot move freely makes "the motors force too much and can be
    damaged".
    """
    limit = ROM_LIMIT_DEG[axis] if limit_deg is None else limit_deg
    min_progress = ROM_MIN_PROGRESS_FRACTION * step_deg

    def _go(target: float) -> float:
        _goto(mini, create_head_pose(**{axis: target}), duration, body_yaw)
        return _settled_pose(mini, settle)[axis]

    directions: dict[str, dict[str, Any]] = {}
    for name, sign in (("positive", 1.0), ("negative", -1.0)):
        origin = _go(0.0)
        extreme = origin
        stalled_at: float | None = None
        no_progress = 0
        commanded = 0.0

        while commanded + step_deg <= limit + 1e-9:
            commanded += step_deg
            measured = _go(sign * commanded)
            progress = (measured - extreme) * sign
            if progress > min_progress:
                extreme = measured
                no_progress = 0
            else:
                no_progress += 1
                extreme = measured if progress > 0 else extreme
                if no_progress >= 2:
                    stalled_at = commanded
                    break

        directions[name] = {
            "commanded_limit_deg": sign * limit,
            "reached_deg": extreme,
            "travel_deg": abs(extreme - origin),
            "stalled_at_deg": None if stalled_at is None else sign * stalled_at,
        }

    mini.goto_target(head=create_head_pose(), body_yaw=0.0, duration=duration)

    span = directions["positive"]["reached_deg"] - directions["negative"]["reached_deg"]
    expected_span = 2 * limit
    coverage = span / expected_span if expected_span else 0.0
    asymmetry = abs(
        directions["positive"]["travel_deg"] - directions["negative"]["travel_deg"]
    )
    stalled = [n for n, d in directions.items() if d["stalled_at_deg"] is not None]

    passed = coverage >= TOL_ROM_FRACTION and asymmetry <= TOL_ROM_ASYMMETRY_DEG
    detail = (
        f"span {span:.1f} deg of {expected_span:.0f} expected ({coverage:.0%}, "
        f"tol {TOL_ROM_FRACTION:.0%}), reached {directions['negative']['reached_deg']:+.1f} "
        f"to {directions['positive']['reached_deg']:+.1f}, asymmetry {asymmetry:.1f} deg "
        f"(tol {TOL_ROM_ASYMMETRY_DEG})"
    )
    if stalled:
        detail += f"; stopped making progress on the {', '.join(stalled)} side"

    return Check(
        name=f"range_of_motion_{axis}",
        passed=passed,
        detail=detail,
        data={
            "axis": axis,
            "limit_deg": limit,
            "step_deg": step_deg,
            "span_deg": span,
            "coverage": coverage,
            "asymmetry_deg": asymmetry,
            "directions": directions,
        },
    )


def run_all(
    mini: ReachyMini,
    axes: tuple[Axis, ...] = AXES,
    amplitude_deg: float = DEFAULT_AMPLITUDE_DEG,
    steps: int = 5,
    with_antennas: bool = True,
    with_range_of_motion: bool = False,
) -> dict[str, Any]:
    """Run the full calibration suite and return a JSON-serialisable report."""
    checks: list[Check] = [zero_offset(mini)]
    for axis in axes:
        checks.append(axis_sweep(mini, axis, amplitude_deg=amplitude_deg, steps=steps))
    if axes:
        checks.append(repeatability(mini, axis=axes[0], approach_from=amplitude_deg))
    if with_range_of_motion:
        checks += [range_of_motion(mini, axis=a) for a in axes]
    if with_antennas:
        checks.append(antenna_check(mini))

    return {
        "passed": all(c.passed for c in checks),
        "checks": [asdict(c) for c in checks],
    }
