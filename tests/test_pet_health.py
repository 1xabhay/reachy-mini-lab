"""Checks for deciding when a session's connection is beyond saving.

Third distinct way this has broken on real hardware, and the subtlest. The
first was a microphone that stopped delivering. The second was the robot's
daemon hanging. This one is worse than both: the daemon answered HTTP
perfectly, so every health check was green, while the WebSocket that actually
carries commands was dead. The pet sat at 28% CPU throwing a traceback every
iteration and nothing noticed, because the supervisor was watching the wrong
channel.

The lesson encoded here: the layers' own failures are the health signal. A
module that cannot talk to the robot is better evidence than any status
endpoint.
"""

import pytest
from reachy_mini_pet.health import SessionHealth


def test_a_new_session_is_healthy():
    assert SessionHealth().is_broken() is False


def test_work_that_succeeds_keeps_it_healthy():
    health = SessionHealth(tolerance=3)
    for i in range(100):
        health.record_success(now=i * 0.1)

    assert health.is_broken() is False


def test_one_lost_connection_is_not_yet_a_dead_session():
    """A single dropped command may be a blip. Tearing down on one is churn."""
    health = SessionHealth(tolerance=3)
    health.record_error(ConnectionError("blip"), now=1.0)

    assert health.is_broken() is False


def test_repeated_lost_connections_mean_the_session_is_gone():
    health = SessionHealth(tolerance=3, window_s=10.0)
    for i in range(3):
        health.record_error(ConnectionError("gone"), now=1.0 + i)

    assert health.is_broken() is True


def test_other_layers_succeeding_does_not_mask_a_dead_command_channel():
    """The exact bug this exists for.

    When the WebSocket died, vision failed on every step because it sends
    commands, while hearing kept succeeding because audio arrives over a
    different transport. A counter that any success resets would have been held
    open for ever by the healthy layer, which is what let the pet sit there
    broken.
    """
    health = SessionHealth(tolerance=3, window_s=10.0)
    for i in range(3):
        health.record_success(now=i * 2.0)              # hearing, still fine
        health.record_error(ConnectionError("gone"), now=i * 2.0)   # vision, not

    assert health.is_broken() is True


def test_failures_long_past_are_forgotten():
    """A blip an hour ago says nothing about the link now."""
    health = SessionHealth(tolerance=2, window_s=10.0)
    health.record_error(ConnectionError("blip"), now=1.0)
    health.record_error(ConnectionError("blip"), now=2.0)
    assert health.is_broken() is True

    health.record_error(ConnectionError("blip"), now=500.0)
    assert health.is_broken() is False                  # only one recent failure


def test_an_unrelated_failure_does_not_condemn_the_connection():
    """Whisper falling over is not the robot going away."""
    health = SessionHealth(tolerance=2)
    for i in range(10):
        health.record_error(ValueError("bad audio"), now=float(i))

    assert health.is_broken() is False


def test_an_os_error_counts_as_a_lost_connection():
    """Sockets fail as OSError as often as ConnectionError."""
    health = SessionHealth(tolerance=2)
    health.record_error(OSError("broken pipe"), now=1.0)
    health.record_error(OSError("broken pipe"), now=2.0)

    assert health.is_broken() is True


def test_it_says_what_went_wrong():
    """The log needs to name the cause, not just report a teardown."""
    health = SessionHealth(tolerance=1)
    health.record_error(ConnectionError("Lost connection with the server."), now=1.0)

    assert "Lost connection" in health.reason


def test_a_healthy_session_has_no_reason():
    assert SessionHealth().reason == ""


def test_errors_are_counted_for_the_log():
    health = SessionHealth(tolerance=99)
    health.record_error(ConnectionError("x"), now=1.0)
    health.record_error(ValueError("y"), now=2.0)

    assert health.connection_errors == 1
    assert health.other_errors == 1


def test_it_only_reports_a_flood_once():
    """A module failing every 5ms must not write a traceback every 5ms."""
    health = SessionHealth(tolerance=100, report_every=10)
    reported = [health.should_report() for _ in range(25)]

    assert reported.count(True) == 3       # the 1st, 11th and 21st


def test_the_tolerance_must_be_positive():
    with pytest.raises(ValueError):
        SessionHealth(tolerance=0)


# ------------------------------------- the signal has to actually arrive

# The health mechanism was written for one incident: vision could not send
# commands and nothing noticed. It could not have caught it. Every layer wraps
# its own work in `except Exception`, so a ConnectionError was logged and
# swallowed inside the layer and never reached the code watching for it.
#
# These use the REAL layers. The previous test used a fake whose `step` raised,
# which no real module does - so it passed while the mechanism was inert.


class DeadEyes:
    """A camera whose robot has gone away."""

    def look(self):
        raise ConnectionError("Lost connection with the server.")

    def turn_body(self, yaw):
        raise ConnectionError("Lost connection with the server.")


class BlindEyes:
    """A camera that is merely broken, which is not the robot going away."""

    def look(self):
        raise ValueError("garbled frame")

    def turn_body(self, yaw):
        pass


def test_a_real_layer_losing_the_robot_reports_it():
    """The actual incident, through the actual Vision layer."""
    from reachy_mini_pet.bus import Bus
    from reachy_mini_pet.layers import Vision

    vision = Vision(DeadEyes())
    vision.start(Bus())

    with pytest.raises(ConnectionError):
        vision.step(now=1.0)


def test_a_real_layer_failing_for_its_own_reasons_stays_quiet():
    """A broken camera must not tear down a session that is otherwise fine."""
    from reachy_mini_pet.bus import Bus
    from reachy_mini_pet.layers import Vision

    vision = Vision(BlindEyes())
    vision.start(Bus())

    vision.step(now=1.0)          # must not raise


def test_a_dead_robot_reaches_health_through_the_real_runner():
    """End to end: real layer, real runner, real health object."""
    import threading

    from reachy_mini_pet import live
    from reachy_mini_pet.bus import Bus
    from reachy_mini_pet.layers import Vision

    vision = Vision(DeadEyes())
    vision.start(Bus())
    health = SessionHealth(tolerance=3, window_s=60.0)
    stop = threading.Event()
    guard = threading.Timer(5.0, stop.set)     # only fires if the fix failed
    guard.start()

    live.run(None, [vision], stop, rate=0.001, health=health)
    guard.cancel()

    assert health.connection_errors >= 3
    assert health.is_broken() is True
    assert stop.is_set(), "the session should have been ended"
