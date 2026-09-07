"""Checks that a dead microphone is noticed and restarted.

This is the test that was missing. The pet ran for forty minutes, its WebRTC
audio stream timed out, and it went deaf while every thread kept running and
the whole suite stayed green - because every fake microphone in the suite
either delivers frames or returns `None` politely, which is exactly what a
silent room looks like too.

So this one models the actual failure: a microphone that stops for good.
"""

import numpy as np
import pytest
from reachy_mini_pet import live


class DyingMedia:
    """A microphone that works, then stops for ever."""

    def __init__(self, frames_before_death=5):
        self.left = frames_before_death
        self.recording = False
        self.starts = 0
        self.stops = 0

    def get_input_audio_samplerate(self):
        return 16000

    def start_recording(self):
        self.recording = True
        self.starts += 1

    def stop_recording(self):
        self.recording = False
        self.stops += 1

    def get_audio_sample(self):
        if self.left <= 0:
            return None                       # dead, and indistinguishable from quiet
        self.left -= 1
        return np.zeros((320, 2), dtype=np.float32)


class FakeMini:
    """Just enough robot for the ears."""

    def __init__(self, media):
        self.media = media
        self.released = 0
        self.acquired = 0

    def release_media(self):
        self.released += 1

    def acquire_media(self):
        self.acquired += 1


@pytest.fixture
def clock(monkeypatch):
    """Build a clock the test drives, so no test ever waits for a timeout."""
    holder = {"t": 1000.0}
    monkeypatch.setattr(live.time, "monotonic", lambda: holder["t"])
    return holder


def test_a_working_microphone_is_left_alone(clock):
    media = DyingMedia(frames_before_death=10_000)
    ears = live.RobotEars(FakeMini(media), audio_timeout_s=8.0)

    for _ in range(50):
        clock["t"] += 0.02
        ears.listen()

    assert media.starts == 1, "it should not have been restarted"


def test_a_quiet_room_is_not_mistaken_for_a_dead_microphone(clock):
    """Frames keep arriving; they are just silent. Nothing to fix."""
    media = DyingMedia(frames_before_death=10_000)
    ears = live.RobotEars(FakeMini(media), audio_timeout_s=8.0)

    for _ in range(2000):
        clock["t"] += 0.02
        ears.listen()

    assert media.stops == 0


def test_a_dead_microphone_is_restarted(clock):
    media = DyingMedia(frames_before_death=5)
    mini = FakeMini(media)
    ears = live.RobotEars(mini, audio_timeout_s=8.0)

    for _ in range(5):
        clock["t"] += 0.02
        ears.listen()
    clock["t"] += 10.0                        # past the timeout
    ears.listen()

    assert media.stops == 1
    assert media.starts == 2, "capture should have been restarted"


def test_a_microphone_that_stays_dead_gets_the_bigger_hammer(clock):
    """If re-opening the stream did not help, rebuild the whole pipeline."""
    media = DyingMedia(frames_before_death=5)
    mini = FakeMini(media)
    ears = live.RobotEars(mini, audio_timeout_s=8.0)

    for _ in range(5):
        clock["t"] += 0.02
        ears.listen()
    for _ in range(3):
        clock["t"] += 20.0                    # past timeout and past the backoff
        ears.listen()

    assert mini.released >= 1
    assert mini.acquired >= 1


def test_recovery_is_not_attempted_in_a_tight_loop(clock):
    """Hammering a wedged device makes it worse and floods the log."""
    media = DyingMedia(frames_before_death=1)
    ears = live.RobotEars(FakeMini(media), audio_timeout_s=1.0)

    clock["t"] += 0.02
    ears.listen()
    for _ in range(500):                      # half a second of frantic polling
        clock["t"] += 0.001
        ears.listen()

    assert media.stops <= 1


def test_a_revived_microphone_is_used_again(clock):
    """After a successful restart the pet must actually hear again."""
    media = DyingMedia(frames_before_death=5)
    ears = live.RobotEars(FakeMini(media), audio_timeout_s=8.0)

    for _ in range(5):
        clock["t"] += 0.02
        ears.listen()
    clock["t"] += 10.0
    ears.listen()                             # triggers the restart
    media.left = 100                          # the restart worked

    heard = None
    for _ in range(10):
        clock["t"] += 0.02
        heard = ears.listen() if heard is None else heard

    assert heard is not None
    assert len(heard) == live.WINDOW


# ------------------------------------------------- the threaded shell itself

# `run` is the one genuinely concurrent piece, and it went untested - which is
# how a call passing `health=` to a `run()` that did not accept it shipped with
# 363 other tests green. These assert what happens and in what order, never
# when, and use a real (brief) thread rather than pretending.


class Layer:
    """A module that does what the test tells it to."""

    def __init__(self, name="layer", raises=None):
        self.name = name
        self.raises = raises
        self.steps = 0

    def step(self, now=0.0):
        self.steps += 1
        if self.raises is not None:
            raise self.raises

    def stop(self):
        pass


def test_run_accepts_the_health_it_is_given():
    """The signature bug: `session` passed `health=` and `run` refused it."""
    import inspect

    assert "health" in inspect.signature(live.run).parameters


def test_run_steps_every_layer():
    import threading

    stop = threading.Event()
    layers = [Layer("a"), Layer("b")]
    threading.Timer(0.15, stop.set).start()

    live.run(None, layers, stop, rate=0.001)

    assert all(layer.steps > 0 for layer in layers)


def test_a_layer_losing_the_robot_ends_the_session():
    """The actual failure: vision could not send commands, and nothing noticed."""
    import threading

    from reachy_mini_pet.health import SessionHealth

    stop = threading.Event()
    broken = Layer("vision", raises=ConnectionError("Lost connection with the server."))
    health = SessionHealth(tolerance=3, window_s=60.0)
    guard = threading.Timer(5.0, stop.set)      # only fires if the fix failed
    guard.start()

    live.run(None, [broken], stop, rate=0.001, health=health)
    guard.cancel()

    assert stop.is_set(), "the session should have been ended"
    assert health.is_broken() is True
    assert "Lost connection" in health.reason


def test_a_layer_failing_for_its_own_reasons_does_not_end_the_session():
    """Whisper falling over is not the robot going away."""
    import threading

    stop = threading.Event()
    grumpy = Layer("transcriber", raises=ValueError("bad audio"))
    threading.Timer(0.2, stop.set).start()

    live.run(None, [grumpy], stop, rate=0.001)

    assert grumpy.steps > 5, "it should have kept trying"


def test_the_runner_honours_each_layers_own_pace():
    """One global rate made vision flood the robot at the microphone's rate."""
    import threading

    class Slow(Layer):
        period = 0.05

    class Fast(Layer):
        period = 0.0

    stop = threading.Event()
    slow, fast = Slow("slow"), Fast("fast")
    threading.Timer(0.4, stop.set).start()

    live.run(None, [slow, fast], stop, rate=0.001)

    assert fast.steps > slow.steps * 3, (
        f"fast={fast.steps} slow={slow.steps}: the pace is not being respected"
    )
