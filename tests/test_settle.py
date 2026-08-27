"""Checks for the closed-loop pose correction against a simulated sticky head.

The robot's Stewart servos are P-only, so they stop inside a friction band and
leave a steady-state error whose sign depends on the approach. `settle_to_pose`
is the software integral term that removes it; these tests model exactly that
failure mode so the correction is verified against the thing it exists for.
"""

import numpy as np
import pytest
from reachy_mini_testbench import calibration
from reachy_mini_testbench.calibration import settle_to_pose
from scipy.spatial.transform import Rotation


class StickyHead:
    """A head that only moves when the command beats stiction, then stops short.

    This model converges cleanly; the real robot instead plateaus around 0.7
    deg, which is a hardware floor rather than anything this fixture captures.
    """

    def __init__(self, deadband_deg=6.0, start_roll=12.0):
        self.deadband = deadband_deg
        self.roll = start_roll

    def goto_target(self, head=None, antennas=None, duration=0.5, **kw):
        commanded = float(Rotation.from_matrix(head[:3, :3]).as_euler("xyz", degrees=True)[0])
        delta = commanded - self.roll
        if abs(delta) > self.deadband:
            # Travels most of the way, stopping a band short of the goal.
            self.roll = commanded - np.sign(delta) * self.deadband

    def get_current_head_pose(self):
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler("xyz", [self.roll, 0, 0], degrees=True).as_matrix()
        return pose


class DriftingHead:
    """A head whose accuracy degrades with every command.

    Nothing on the real robot behaves quite like this; it exists to force the
    case where the final attempt is worse than an earlier one, which is the
    only situation in which keeping the best result changes the answer.
    """

    def __init__(self, start_roll=2.0, drift_per_call=3.0):
        self.roll = start_roll
        self.drift = drift_per_call
        self.calls = 0

    def goto_target(self, head=None, antennas=None, duration=0.5, **kw):
        self.calls += 1
        self.roll = self.calls * self.drift

    def get_current_head_pose(self):
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler("xyz", [self.roll, 0, 0], degrees=True).as_matrix()
        return pose


@pytest.fixture(autouse=True)
def _no_waiting(monkeypatch):
    monkeypatch.setattr(calibration.time, "sleep", lambda _s: None)


def test_correction_beats_the_open_loop_error():
    head = StickyHead(deadband_deg=6.0, start_roll=12.0)

    result = settle_to_pose(head, tolerance_deg=1.0, max_iterations=8)

    assert abs(result["achieved"]["roll"]) < 6.0, "should improve on the raw deadband"
    assert result["worst_angular_deg"] == pytest.approx(
        abs(result["achieved"]["roll"]), abs=1e-6
    )


def test_reports_the_best_iteration_not_the_last():
    head = DriftingHead(drift_per_call=3.0)

    result = settle_to_pose(head, tolerance_deg=0.01, max_iterations=5)

    residuals = [h["worst_angular_deg"] for h in result["history"]]
    assert min(residuals) < residuals[-1], "fixture must degrade, or the test proves nothing"
    assert result["worst_angular_deg"] == pytest.approx(min(residuals))
    # An unreachable tolerance must be reported honestly, not rounded away.
    assert not result["converged"]


def test_an_already_level_head_stops_after_one_iteration():
    head = StickyHead(deadband_deg=6.0, start_roll=0.0)

    result = settle_to_pose(head, tolerance_deg=1.0)

    assert result["converged"]
    assert result["iterations"] == 1


def test_non_zero_target_is_honoured():
    head = StickyHead(deadband_deg=4.0, start_roll=0.0)

    result = settle_to_pose(head, target={"roll": 20.0}, tolerance_deg=1.0, max_iterations=8)

    assert result["goal"]["roll"] == 20.0
    assert abs(result["achieved"]["roll"] - 20.0) < 4.0


def test_unknown_pose_component_is_rejected():
    with pytest.raises(ValueError, match="unknown pose components"):
        settle_to_pose(StickyHead(), target={"tilt": 5.0})


def test_a_missed_completion_acknowledgement_does_not_abort_the_correction():
    """The SDK raises TimeoutError when the daemon is slow to acknowledge a goto.

    The move still happens, and the loop reads the pose back anyway, so this
    must be absorbed rather than ending the correction part-way.
    """
    head = StickyHead(deadband_deg=6.0, start_roll=12.0)
    real_goto = head.goto_target
    calls = {"n": 0}

    def flaky_goto(*args, **kwargs):
        calls["n"] += 1
        real_goto(*args, **kwargs)
        if calls["n"] == 1:
            raise TimeoutError("Task did not complete in time.")

    head.goto_target = flaky_goto

    result = settle_to_pose(head, tolerance_deg=1.0, max_iterations=8)

    assert result["missed_acknowledgements"] == 1
    assert result["converged"]
