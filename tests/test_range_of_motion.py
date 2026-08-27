"""Checks for the range-of-motion sweep against simulated heads.

The stall guard is the part that matters: it is what stops the sweep driving a
mechanically blocked axis into its end stop, so it is exercised directly rather
than inferred from a passing happy path.
"""

import numpy as np
import pytest
from reachy_mini_testbench import calibration
from reachy_mini_testbench.calibration import ROM_LIMIT_DEG, range_of_motion
from scipy.spatial.transform import Rotation


class SimHead:
    """A head that follows commands on one axis until it hits a hard stop.

    `offset` biases what the head *reports* by a constant, standing in for a
    mis-zeroed robot: the range is unchanged, only its centre moves.
    """

    def __init__(self, axis="roll", stop_positive=90.0, stop_negative=-90.0, offset=0.0):
        self.axis = axis
        self.stop_positive = stop_positive
        self.stop_negative = stop_negative
        self.offset = offset
        self.commands: list[float] = []
        self._reached = 0.0

    def goto_target(self, head=None, antennas=None, duration=0.5, **kw):
        idx = ("roll", "pitch", "yaw").index(self.axis)
        commanded = float(Rotation.from_matrix(head[:3, :3]).as_euler("xyz", degrees=True)[idx])
        self.commands.append(commanded)
        self._reached = float(np.clip(commanded, self.stop_negative, self.stop_positive))

    def get_current_head_pose(self):
        idx = ("roll", "pitch", "yaw").index(self.axis)
        euler = [0.0, 0.0, 0.0]
        euler[idx] = self._reached + self.offset
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler("xyz", euler, degrees=True).as_matrix()
        return pose


@pytest.fixture(autouse=True)
def _no_waiting(monkeypatch):
    monkeypatch.setattr(calibration.time, "sleep", lambda _s: None)


def test_healthy_axis_reports_full_span_and_passes():
    head = SimHead("roll")

    result = range_of_motion(head, axis="roll", step_deg=5.0)

    assert result.passed, result.detail
    assert result.data["span_deg"] == pytest.approx(2 * ROM_LIMIT_DEG["roll"], abs=1.0)
    assert result.data["coverage"] == pytest.approx(1.0, abs=0.05)


def test_hard_stop_is_detected_and_sweep_gives_up_early():
    head = SimHead("roll", stop_positive=12.0)

    result = range_of_motion(head, axis="roll", step_deg=5.0)

    assert not result.passed
    assert result.data["directions"]["positive"]["stalled_at_deg"] is not None
    assert result.data["directions"]["positive"]["reached_deg"] == pytest.approx(12.0, abs=1.0)
    # It must stop shortly past the stop, not grind on to the full limit.
    assert max(head.commands) < ROM_LIMIT_DEG["roll"]


def test_asymmetric_travel_fails_even_when_total_span_looks_fine():
    # Shifted range: same total travel, but all of it on one side.
    head = SimHead("pitch", stop_positive=40.0, stop_negative=-15.0)

    result = range_of_motion(head, axis="pitch", step_deg=5.0)

    assert not result.passed
    assert result.data["asymmetry_deg"] > calibration.TOL_ROM_ASYMMETRY_DEG


def test_constant_offset_does_not_reduce_measured_range():
    """Range is a span, so a head that is 15 deg mis-zeroed still measures full travel."""
    head = SimHead("yaw", offset=15.0)

    result = range_of_motion(head, axis="yaw", step_deg=5.0)

    assert result.passed, result.detail
    assert result.data["coverage"] == pytest.approx(1.0, abs=0.05)


def test_uses_documented_limit_per_axis_by_default():
    for axis, limit in ROM_LIMIT_DEG.items():
        head = SimHead(axis)
        result = range_of_motion(head, axis=axis, step_deg=5.0)
        assert result.data["limit_deg"] == limit
        assert max(abs(c) for c in head.commands) <= limit + 1e-9


def test_explicit_limit_overrides_the_documented_one():
    head = SimHead("roll")

    range_of_motion(head, axis="roll", limit_deg=20.0, step_deg=5.0)

    assert max(abs(c) for c in head.commands) == pytest.approx(20.0)
