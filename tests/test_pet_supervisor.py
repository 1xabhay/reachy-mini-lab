"""Checks for the supervisor that survives the robot dying underneath it.

The watchdog handles a sense going quiet. This handles something worse: the
robot itself stopping. Its daemon hung, every connection timed out, and the pet
had nothing to reconnect to - so it was simply gone until a person noticed and
power-cycled the hardware. For something meant to sit in a room for weeks,
"until a person notices" is not a recovery strategy.

The rules that matter are about not making things worse: retry for ever,
because the robot may come back at any time, but back off so a dead robot is
not hammered, and reset the moment it answers.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from reachy_mini_pet.supervisor import State, Supervisor


def test_it_tries_immediately_on_a_cold_start():
    """Nothing has failed yet - there is no reason to wait."""
    assert Supervisor().should_retry(now=0.0) is True


def test_a_healthy_robot_is_left_alone():
    supervisor = Supervisor()
    supervisor.connected(now=1.0)

    assert supervisor.state is State.CONNECTED
    assert supervisor.should_retry(now=100.0) is False


def test_a_failure_is_not_retried_instantly():
    """Reconnecting in a tight loop is how you turn one fault into a log flood."""
    supervisor = Supervisor(initial_backoff_s=1.0)
    supervisor.connected(now=0.0)
    supervisor.failed(now=10.0)

    assert supervisor.state is State.LOST
    assert supervisor.should_retry(now=10.5) is False
    assert supervisor.should_retry(now=11.0) is True


def test_repeated_failures_back_off_further_each_time():
    """A robot that has been dead for an hour is not about to answer this second."""
    supervisor = Supervisor(initial_backoff_s=1.0)
    waits = []
    now = 0.0
    for _ in range(5):
        supervisor.failed(now=now)
        wait = 0.0
        while not supervisor.should_retry(now=now + wait):
            wait += 0.1
        waits.append(round(wait, 1))
        now += wait

    assert waits == sorted(waits)
    assert waits[-1] > waits[0]


def test_the_backoff_has_a_ceiling():
    """It must keep trying for ever - the robot may be plugged back in at any time."""
    supervisor = Supervisor(initial_backoff_s=1.0, max_backoff_s=8.0)
    for i in range(20):
        supervisor.failed(now=i * 1000.0)

    assert supervisor.backoff == pytest.approx(8.0)


def test_a_session_that_lasted_forgets_the_backoff():
    """A working session then a fault is a fresh problem, worth trying at once.

    Note what resets it: a session that *lasted*, not a connection that
    succeeded. Resetting on connect alone was the bug - a connect that fell
    over immediately would clear the backoff it had just earned, and the pet
    would spin through connect-fail-connect-fail about once a second for ever.
    """
    supervisor = Supervisor(initial_backoff_s=1.0, min_stable_s=10.0)
    for i in range(6):
        supervisor.failed(now=i * 100.0)
    assert supervisor.backoff > 1.0

    supervisor.connected(now=1000.0)
    supervisor.failed(now=1100.0)              # after a good long session

    assert supervisor.backoff == pytest.approx(1.0)
    assert supervisor.should_retry(now=1101.5) is True


def test_it_counts_reconnections_for_the_log():
    """A rising count is how an intermittent robot gets diagnosed."""
    supervisor = Supervisor()
    supervisor.connected(now=0.0)
    supervisor.failed(now=1.0)
    supervisor.connected(now=2.0)

    assert supervisor.reconnections == 1


def test_it_reports_a_change_of_state_only_once():
    """Otherwise every loop iteration logs that the robot is still missing."""
    supervisor = Supervisor()
    supervisor.failed(now=1.0)

    assert supervisor.take_transition() is State.LOST
    assert supervisor.take_transition() is None


def test_coming_back_is_reported_too():
    supervisor = Supervisor()
    supervisor.failed(now=1.0)
    supervisor.take_transition()
    supervisor.connected(now=5.0)

    assert supervisor.take_transition() is State.CONNECTED


def test_how_long_it_has_been_gone_is_available():
    """Worth saying out loud: 'back after four minutes' is useful, 'back' is not."""
    supervisor = Supervisor()
    supervisor.connected(now=0.0)
    supervisor.failed(now=10.0)

    assert supervisor.lost_for(now=70.0) == pytest.approx(60.0)


def test_a_robot_that_was_never_there_has_no_downtime():
    assert Supervisor().lost_for(now=100.0) == pytest.approx(0.0)


@given(
    failures=st.integers(min_value=1, max_value=40),
    initial=st.floats(min_value=0.1, max_value=5.0),
    ceiling=st.floats(min_value=5.0, max_value=60.0),
)
@settings(deadline=None, max_examples=100)
def test_the_wait_never_exceeds_the_ceiling(failures, initial, ceiling):
    supervisor = Supervisor(initial_backoff_s=initial, max_backoff_s=ceiling)
    for i in range(failures):
        supervisor.failed(now=i * 10_000.0)
        assert initial - 1e-9 <= supervisor.backoff <= ceiling + 1e-9


@given(gap=st.floats(min_value=0.0, max_value=0.9))
@settings(deadline=None, max_examples=50)
def test_it_never_retries_faster_than_its_own_backoff(gap):
    supervisor = Supervisor(initial_backoff_s=1.0)
    supervisor.failed(now=0.0)
    supervisor.should_retry(now=1.0)          # the one allowed attempt

    assert supervisor.should_retry(now=1.0 + gap) is False


# ------------------------------------- a session that dies the moment it starts

# The backoff protected the case where the robot is absent, and not the likelier
# one: connecting fine and then failing immediately. Because `connected` reset
# the backoff and `failed` only doubled it when already LOST, connect-fail-
# connect-fail cycled about once a second for ever - each attempt loading a
# whisper model, a piper voice and the emotions library, and writing two lines
# to the log. A missing model file on a fresh install would do it, and the log
# would be useless for diagnosing it.
#
# So what matters is not whether it connected, but whether the session lasted.


def test_a_session_that_never_worked_backs_off_like_any_other_failure():
    supervisor = Supervisor(initial_backoff_s=1.0, min_stable_s=5.0)
    waits = []
    now = 0.0
    for _ in range(5):
        supervisor.connected(now=now)          # the connect itself succeeds...
        supervisor.failed(now=now + 0.2)       # ...and the session dies at once
        waits.append(supervisor.backoff)
        now += 10.0

    assert waits == sorted(waits)
    assert waits[-1] > waits[0], "a crash loop must slow down"


def test_a_session_that_ran_properly_retries_straight_away():
    """An hour of conversation then a dropped link deserves a fast retry."""
    supervisor = Supervisor(initial_backoff_s=1.0, min_stable_s=5.0)
    for i in range(4):                          # get the backoff up
        supervisor.connected(now=i * 100.0)
        supervisor.failed(now=i * 100.0 + 0.2)
    assert supervisor.backoff > 1.0

    supervisor.connected(now=1000.0)
    supervisor.failed(now=4600.0)               # an hour later

    assert supervisor.backoff == pytest.approx(1.0)
    assert supervisor.should_retry(now=4601.5) is True


def test_the_crash_loop_still_reaches_the_ceiling():
    supervisor = Supervisor(initial_backoff_s=1.0, max_backoff_s=8.0, min_stable_s=5.0)
    now = 0.0
    for _ in range(20):
        supervisor.connected(now=now)
        supervisor.failed(now=now + 0.1)
        now += 100.0

    assert supervisor.backoff == pytest.approx(8.0)
