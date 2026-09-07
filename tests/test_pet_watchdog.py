"""Checks for the watchdog that notices a sense has gone dead.

The pet ran happily for forty minutes and then its WebRTC audio stream timed
out. The process stayed up, every thread kept running, the tests stayed green -
and it was stone deaf, because a microphone that stops delivering looks exactly
like a quiet room. Nothing in the system could tell the difference.

That is the failure mode this exists to catch: not a crash, which is loud and
obvious, but a sense that quietly stops while everything around it insists it
is fine.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from reachy_mini_pet.watchdog import Watchdog


def test_a_fresh_watchdog_is_not_stalled():
    """Give it the benefit of the doubt until something should have arrived."""
    assert Watchdog(timeout_s=2.0).stalled(now=0.0) is False


def test_a_sense_that_keeps_delivering_is_healthy():
    dog = Watchdog(timeout_s=2.0)
    for i in range(100):
        dog.feed(now=i * 0.1)
        assert dog.stalled(now=i * 0.1) is False


def test_silence_short_of_the_timeout_is_not_a_stall():
    """Rooms go quiet. That is not the same as a dead microphone."""
    dog = Watchdog(timeout_s=2.0)
    dog.feed(now=10.0)

    assert dog.stalled(now=11.9) is False


def test_silence_past_the_timeout_is_a_stall():
    dog = Watchdog(timeout_s=2.0)
    dog.feed(now=10.0)

    assert dog.stalled(now=12.0) is True


def test_a_watchdog_that_never_saw_anything_stalls_from_the_start():
    """A mic that never worked is as broken as one that stopped."""
    dog = Watchdog(timeout_s=2.0, now=0.0)

    assert dog.stalled(now=1.9) is False
    assert dog.stalled(now=2.1) is True


def test_recovery_is_only_reported_once_per_stall():
    """The caller restarts hardware on this. It must not be told twice."""
    dog = Watchdog(timeout_s=1.0)
    dog.feed(now=0.0)

    assert dog.should_recover(now=2.0) is True
    assert dog.should_recover(now=2.1) is False
    assert dog.should_recover(now=3.0) is False


def test_a_recovered_sense_can_stall_again_later():
    """Hardware that wedges once will wedge again."""
    dog = Watchdog(timeout_s=1.0)
    dog.feed(now=0.0)
    assert dog.should_recover(now=2.0) is True

    dog.feed(now=3.0)                       # the restart worked
    assert dog.stalled(now=3.0) is False
    assert dog.should_recover(now=5.0) is True


def test_recovery_backs_off_rather_than_thrashing():
    """If restarting does not help, stop hammering the device."""
    dog = Watchdog(timeout_s=1.0, backoff_s=4.0)
    dog.feed(now=0.0)

    assert dog.should_recover(now=2.0) is True
    assert dog.should_recover(now=3.5) is False     # inside the backoff
    assert dog.should_recover(now=6.5) is True      # past it, still dead


def test_it_counts_how_often_it_has_had_to_intervene():
    """A rising count in the log is how a wedging device gets diagnosed."""
    dog = Watchdog(timeout_s=1.0, backoff_s=1.0)
    dog.feed(now=0.0)
    dog.should_recover(now=2.0)
    dog.should_recover(now=4.0)

    assert dog.recoveries == 2


@given(
    gaps=st.lists(st.floats(min_value=0.0, max_value=10.0), min_size=1, max_size=50),
    timeout=st.floats(min_value=0.1, max_value=5.0),
)
@settings(deadline=None, max_examples=100)
def test_it_never_claims_a_stall_while_frames_are_arriving(gaps, timeout):
    dog = Watchdog(timeout_s=timeout)
    now = 0.0
    for gap in gaps:
        now += gap
        dog.feed(now=now)
        assert dog.stalled(now=now) is False


def test_the_timeout_must_be_positive():
    """A zero timeout would declare every quiet moment a hardware fault."""
    with pytest.raises(ValueError):
        Watchdog(timeout_s=0.0)
