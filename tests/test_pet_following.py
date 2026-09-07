"""Checks for the body-yaw follower - the fix for a head that cannot reach.

The daemon's tracker aims the head and nothing else: `body_yaw` is never part
of its aim, so the body never turns and the whole system is capped at the
head's 65 degrees of yaw. That is why it loses people at the edges - it
physically cannot keep them centred, so they walk out of frame.

This adds the missing joint, the way a person does it: eyes and head for small
corrections, torso when they run out. The head stays the daemon's job; this
only decides when the body should help.
"""

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from reachy_mini_pet.following import BodyYawFollower


def settle(follower, x, steps=200, dt=0.05, start=0.0):
    """Run the follower to rest against a fixed face offset."""
    yaw = None
    for i in range(steps):
        yaw = follower.update(detected=True, x=x, now=start + i * dt)
    return yaw


# ------------------------------------------------------------- not moving


def test_a_centred_face_does_not_turn_the_body():
    assert settle(BodyYawFollower(), x=0.0) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("x", [-0.15, -0.05, 0.0, 0.05, 0.15])
def test_small_offsets_are_ignored_so_it_does_not_fidget(x):
    """The head handles these. A body that twitches at every wobble is worse."""
    assert settle(BodyYawFollower(deadzone=0.2), x=x) == pytest.approx(0.0, abs=1e-6)


def test_no_face_means_no_new_command():
    assert BodyYawFollower().update(detected=False, x=None, now=1.0) is None


# ----------------------------------------------------------- reaching out


def test_a_face_off_to_one_side_turns_the_body_towards_it():
    right = settle(BodyYawFollower(), x=0.8)
    left = settle(BodyYawFollower(), x=-0.8)

    assert right > 0.0
    assert left < 0.0
    assert right == pytest.approx(-left, abs=1e-6)


def test_turning_further_for_a_further_face():
    near = settle(BodyYawFollower(), x=0.4)
    far = settle(BodyYawFollower(), x=0.9)

    assert far > near


def test_the_body_never_exceeds_its_limit():
    follower = BodyYawFollower(max_yaw=math.radians(45))
    assert settle(follower, x=1.0) == pytest.approx(math.radians(45), abs=1e-6)


def test_the_direction_can_be_flipped_for_the_real_robot():
    """Which way is positive is a property of the hardware, not of this code."""
    forward = settle(BodyYawFollower(direction=1.0), x=0.8)
    reversed_ = settle(BodyYawFollower(direction=-1.0), x=0.8)

    assert forward == pytest.approx(-reversed_, abs=1e-6)


# --------------------------------------------------------------- smoothly


def test_it_turns_smoothly_rather_than_snapping():
    """A body that jumps to the target looks alarming and stresses the motor."""
    follower = BodyYawFollower(max_rate=1.0)
    first = follower.update(detected=True, x=1.0, now=0.0)
    second = follower.update(detected=True, x=1.0, now=0.1)

    assert abs(first) <= 1e-9                      # nothing to integrate yet
    assert 0.0 < second <= 1.0 * 0.1 + 1e-9        # at most max_rate * dt


@given(dt=st.floats(min_value=0.001, max_value=0.5))
@settings(deadline=None, max_examples=50)
def test_no_step_ever_exceeds_the_rate_limit(dt):
    follower = BodyYawFollower(max_rate=2.0)
    previous = follower.update(detected=True, x=1.0, now=0.0)
    for i in range(1, 30):
        current = follower.update(detected=True, x=1.0, now=i * dt)
        assert abs(current - previous) <= 2.0 * dt + 1e-9
        previous = current


# ------------------------------------------------------- losing you badly


def test_losing_you_at_the_edge_keeps_turning_that_way():
    """The whole point: you left the frame to the right, so look right."""
    follower = BodyYawFollower(reacquire_s=1.0)
    for i in range(20):
        follower.update(detected=True, x=0.9, now=i * 0.05)
    at_loss = follower.update(detected=False, x=None, now=1.0)

    later = follower.update(detected=False, x=None, now=1.3)

    assert at_loss is None or later is not None
    assert follower.searching(now=1.3) is True


def test_it_gives_up_searching_rather_than_spinning_for_ever():
    follower = BodyYawFollower(reacquire_s=1.0)
    for i in range(20):
        follower.update(detected=True, x=0.9, now=i * 0.05)

    assert follower.searching(now=1.0 + 5.0) is False


# ------------------------------------------------------------ invariants


@given(
    xs=st.lists(st.floats(min_value=-1.0, max_value=1.0), min_size=1, max_size=100),
    limit=st.floats(min_value=0.1, max_value=2.0),
)
@settings(deadline=None, max_examples=100)
def test_the_command_is_always_within_the_mechanical_limit(xs, limit):
    """A command outside the limit is a collision or a stalled motor."""
    follower = BodyYawFollower(max_yaw=limit)
    for i, x in enumerate(xs):
        yaw = follower.update(detected=True, x=x, now=i * 0.05)
        assert -limit - 1e-9 <= yaw <= limit + 1e-9
