"""Ground-truth checks for the vision-based rotation estimator.

Synthesises the exact homography a known head rotation would produce and
asserts the estimator recovers it - sign included, which is the part that is
easy to get subtly wrong and impossible to notice on hardware.
"""

import cv2
import numpy as np
import pytest
from reachy_mini_testbench.rotation_test import R_HEAD_CAM, measure_rotation

W, H = 1280, 720
K = np.array([[900.0, 0, W / 2], [0, 900.0, H / 2], [0, 0, 1.0]])
AXIS_VECTORS = {"roll": [1.0, 0, 0], "pitch": [0, 1.0, 0], "yaw": [0, 0, 1.0]}


def _scene() -> np.ndarray:
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (H, W), dtype=np.uint8)
    blob = cv2.resize(cv2.resize(base, (W // 4, H // 4)), (W, H), interpolation=cv2.INTER_CUBIC)
    return cv2.addWeighted(blob, 0.6, base, 0.4, 0)


def _rotate_head(scene: np.ndarray, axis: str, deg: float) -> np.ndarray:
    """Warp `scene` the way the camera would see it after a head rotation."""
    n_cam = R_HEAD_CAM.T @ np.array(AXIS_VECTORS[axis])
    R_cam = cv2.Rodrigues(np.deg2rad(deg) * n_cam)[0]
    A = R_cam.T  # scene motion is the inverse of camera motion
    return cv2.warpPerspective(scene, K @ A @ np.linalg.inv(K), (W, H))


@pytest.mark.parametrize("axis", ["roll", "pitch", "yaw"])
@pytest.mark.parametrize("deg", [-15.0, -7.0, 7.0, 15.0])
def test_homography_recovers_commanded_rotation(axis: str, deg: float) -> None:
    scene = _scene()
    result = measure_rotation(scene, _rotate_head(scene, axis, deg), K)

    assert result["ok"], result["error"]
    assert result["homography"][axis] == pytest.approx(deg, abs=0.5)
    # The other two axes must stay near zero, or the axis mapping is crossed.
    for other in set(AXIS_VECTORS) - {axis}:
        assert abs(result["homography"][other]) < 1.0


@pytest.mark.parametrize("axis", ["pitch", "yaw"])
@pytest.mark.parametrize("deg", [-10.0, 10.0])
def test_shift_estimator_agrees_in_sign_and_magnitude(axis: str, deg: float) -> None:
    scene = _scene()
    result = measure_rotation(scene, _rotate_head(scene, axis, deg), K)

    assert result["ok"], result["error"]
    assert result["shift"][axis] == pytest.approx(deg, abs=1.5)


def test_featureless_scene_reports_error_rather_than_a_verdict() -> None:
    blank = np.full((H, W), 128, dtype=np.uint8)
    result = measure_rotation(blank, blank, K)

    assert not result["ok"]
    assert result["error"]
