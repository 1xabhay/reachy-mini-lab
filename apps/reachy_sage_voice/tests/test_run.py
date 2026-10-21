"""`make run`: one command from the environment to a live pipeline, with no robot here.

The robot and the pipeline runner are fakes; what is checked is the order of
operations and that nothing falls back to a network default.
"""

import pytest

from reachy_sage_voice.__main__ import main
from reachy_sage_voice.kernel import EMOTIONS

ENV = {
    "SAGE_VOICE_KERNEL_URL": "http://127.0.0.1:8776",
    "SAGE_VOICE_KERNEL_TOKEN": "t" * 32,
    "SAGE_VOICE_STT_URL": "http://127.0.0.1:8781",
    "SAGE_VOICE_TTS_URL": "http://127.0.0.1:8782",
    "REACHY_MINI_HOST": "reachy-mini.lan",
}


class Robot:
    def __init__(self, calls, fail_wake=False, fail_track=False, **kwargs):
        self.calls, self.fail_wake, self.fail_track = calls, fail_wake, fail_track
        calls.append(("connect", kwargs))
        self.media = object()
        self.client = type("Client", (), {"is_connected": lambda self: True})()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.calls.append(("disconnect",))

    def enable_motors(self):
        self.calls.append(("enable_motors",))

    def wake_up(self):
        self.calls.append(("wake_up",))
        if self.fail_wake:
            raise TimeoutError("still settling")

    def start_head_tracking(self, weight=1.0):
        self.calls.append(("track", weight))
        if self.fail_track:
            raise TimeoutError("tracker not ready")

    def get_tracked_face(self, wait=True):
        self.calls.append(("face", wait))
        return type("Face", (), {"detected": True})()

    async def async_play_move(self, move, sound):
        self.calls.append(("move", move, sound))


def start(env, fail_wake=False, fail_track=False):
    calls, ran = [], {}

    async def run(media, play_emote, moves, env, play_sound=None, presence=None, observers=(),
            face=None):
        ran.update(media=media, moves=moves, env=env, play_sound=play_sound, presence=presence,
                   observers=observers, face=face)
        await play_emote("cheerful1", True)

    main(env=env, robot=lambda **kw: Robot(calls, fail_wake, fail_track, **kw), run=run,
         moves=lambda: dict.fromkeys(EMOTIONS, "a move"), reach=lambda env: [])
    return calls, ran


def test_it_connects_to_the_named_host_only_then_enables_before_waking():
    calls, ran = start(ENV)
    assert calls[0] == ("connect", {"host": "reachy-mini.lan", "connection_mode": "network"})
    assert [c[0] for c in calls[1:3]] == ["enable_motors", "wake_up"]
    assert ran["env"] is ENV and set(ran["moves"]) == set(EMOTIONS)


def test_an_emote_plays_the_recorded_move_muted_or_not():
    calls, _ = start(ENV)
    assert ("move", "a move", True) in calls


def test_it_follows_faces_after_waking_whether_or_not_presence_is_reported():
    # Live run 5: the daemon reported no face ever, because nothing switched its tracker on.
    for env in (ENV, {**ENV, "SAGE_VOICE_PRESENCE": "1"}):
        calls, _ = start(env)
        names = [c[0] for c in calls]
        assert ("track", 1.0) in calls
        assert names.index("track") > names.index("wake_up")


def test_a_tracker_that_will_not_start_does_not_stop_the_voice():
    _, ran = start(ENV, fail_track=True)
    assert ran


def test_a_slow_wake_up_does_not_stop_the_voice():
    _, ran = start(ENV, fail_wake=True)
    assert ran  # the pipeline still ran


@pytest.mark.parametrize("missing", sorted(ENV))
def test_every_address_and_the_token_must_be_set_before_touching_the_robot(missing):
    env = {k: v for k, v in ENV.items() if k != missing}
    calls = []
    with pytest.raises(SystemExit, match=missing):
        main(env=env, robot=lambda **kw: Robot(calls, **kw), run=None, moves=dict)
    assert calls == []


def test_the_stt_url_may_be_given_with_or_without_v1():
    from reachy_sage_voice.pipeline import openai_base

    assert openai_base("http://127.0.0.1:8781") == "http://127.0.0.1:8781/v1"
    assert openai_base("http://127.0.0.1:8781/v1/") == "http://127.0.0.1:8781/v1"


def test_presence_is_off_unless_asked_for():
    _, ran = start(ENV)
    assert ran["presence"] is None  # the deployed kernel refuses presence events


def test_presence_on_reads_the_daemons_face_tracker_without_waiting():
    calls, ran = start({**ENV, "SAGE_VOICE_PRESENCE": "1"})
    assert ran["presence"]() is True
    assert ("face", False) in calls


def test_the_face_tracker_is_handed_to_the_voice_for_addressing():
    calls, ran = start(ENV)
    assert ran["face"]() is True
    assert ("face", False) in calls


def test_listening_cues_are_off_unless_asked_for():
    _, ran = start(ENV)
    assert ran["observers"] == []


def test_listening_cues_on_watch_the_pipeline_and_still_play_emotes():
    from pipecat.observers.speaking_observer import SpeakingObserver

    calls, ran = start({**ENV, "SAGE_VOICE_LISTEN_CUES": "1"})
    assert [type(o) for o in ran["observers"]] == [SpeakingObserver]
    assert ("move", "a move", True) in calls
