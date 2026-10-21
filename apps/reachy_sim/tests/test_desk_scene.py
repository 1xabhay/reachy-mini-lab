"""The desk scene: a screen and a person on the table, for look-at tests."""

import numpy as np

import sim

NEUTRAL_HEAD = [0.0] + [0.6244, -0.6244] * 3


def test_the_person_is_in_view_ahead_and_the_screen_is_off_to_one_side():
    frame = sim.eye_camera(NEUTRAL_HEAD, [0.0, 0.0], scene="desk", segment=True)
    ids = frame[:, :, 0]
    person = np.nonzero(ids == sim.geom_id("desk", "person"))[1]
    screen = np.nonzero(ids == sim.geom_id("desk", "screen"))[1]

    assert person.size and abs(person.mean() - 160) < 20
    assert screen.size and screen.mean() - 160 > 40  # to the right
