"""Command the head left, read the real state back, and see the world move in the eye."""

import numpy as np

import sim

from conftest import goto


def centroid_x(frame, geom_name, scene="minimal"):
    """Column of the centre of a geom's pixels in a segmentation frame."""
    cols = np.nonzero(frame[:, :, 0] == sim.geom_id(scene, geom_name))[1]
    assert cols.size, f"{geom_name} not in view"
    return cols.mean()


def test_head_yaw_left_is_within_limits_and_moves_the_duck_in_the_eye(neutral):
    left = goto(yaw=30)

    assert abs(left["yaw"] - 30) < 1.5
    assert abs(left["pitch"]) < 40 and abs(left["roll"]) < 40
    assert abs(left["body_yaw"] - left["yaw"]) <= 65

    before = sim.eye_camera(neutral["head_joints"], neutral["antennas"], segment=True)
    after = sim.eye_camera(left["head_joints"], left["antennas"], segment=True)
    # Turning left brings the duck (on the robot's left) towards the image centre.
    assert centroid_x(after, "duck_geom") - centroid_x(before, "duck_geom") > 30

    rgb_before = sim.eye_camera(neutral["head_joints"], neutral["antennas"])
    rgb_after = sim.eye_camera(left["head_joints"], left["antennas"])
    assert rgb_before.shape == (180, 320, 3)
    assert np.abs(rgb_after.astype(int) - rgb_before).mean() > 5
