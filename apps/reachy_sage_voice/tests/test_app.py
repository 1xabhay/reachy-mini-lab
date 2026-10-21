"""The voice as a Reachy Mini app: what the daemon launches runs the same flow as `make run`.

The daemon finds the app by its ``reachy_mini_apps`` entry point, connects to
the robot itself and hands ``run`` the robot and a stop event. Robot, pipeline
and network are fakes here.
"""

import asyncio
import threading
import urllib.error
from importlib.metadata import entry_points

import pytest
from reachy_mini import ReachyMiniApp
from test_run import ENV, Robot

import reachy_sage_voice.__main__ as voice
from reachy_sage_voice.kernel import EMOTIONS

SETTINGS = {k: v for k, v in ENV.items() if k != voice.HOST}


def test_the_daemon_finds_the_app_by_its_entry_point():
    (entry,) = [e for e in entry_points(group="reachy_mini_apps") if e.name == "reachy-sage-voice"]
    assert entry.value == "reachy_sage_voice.main:ReachySageVoice"
    assert issubclass(entry.load(), ReachyMiniApp)


@pytest.fixture
def app(monkeypatch):
    """Build the app with the settings in the environment, a fake pipeline, every service up."""
    from reachy_sage_voice import kernel, pipeline
    from reachy_sage_voice.main import ReachySageVoice

    for name in (*ENV, "SAGE_VOICE_PRESENCE", "SAGE_VOICE_LISTEN_CUES"):
        monkeypatch.delenv(name, raising=False)
    for name, value in SETTINGS.items():
        monkeypatch.setenv(name, value)
    ran = {}

    async def run(media, play_emote, moves, env, play_sound=None, presence=None, observers=()):
        ran.update(env=dict(env), moves=moves)
        try:
            await asyncio.Event().wait() if ran.get("forever") else None
        except asyncio.CancelledError:
            ran["cancelled"] = True
            raise

    monkeypatch.setattr(pipeline, "run", run)
    monkeypatch.setattr(kernel, "load_moves", lambda: dict.fromkeys(EMOTIONS, "a move"))
    monkeypatch.setattr(voice, "unreachable", lambda env: [])
    return ReachySageVoice(), ran


def test_the_app_runs_the_voice_on_the_robot_the_daemon_connected(app):
    app, ran = app
    calls = []
    app.run(Robot(calls), threading.Event())
    assert [c[0] for c in calls[1:3]] == ["enable_motors", "wake_up"]
    assert SETTINGS.items() <= ran["env"].items() and set(ran["moves"]) == set(EMOTIONS)


def test_the_app_stops_when_the_daemon_says_so(app):
    app, ran = app
    ran["forever"] = True
    stop = threading.Event()
    threading.Timer(0.2, stop.set).start()
    app.run(Robot([]), stop)
    assert ran["cancelled"]


@pytest.mark.parametrize("missing", sorted(SETTINGS))
def test_the_app_needs_the_kernel_and_service_settings_but_no_host(app, monkeypatch, missing):
    app, _ = app
    monkeypatch.delenv(missing)
    calls = []
    with pytest.raises(SystemExit, match=missing):
        app.run(Robot(calls), threading.Event())
    assert calls[1:] == []


def test_an_unreachable_service_is_named_before_the_robot_moves(app, monkeypatch):
    app, _ = app
    monkeypatch.setattr(voice, "unreachable", lambda env: ["SAGE_VOICE_STT_URL"])
    calls = []
    with pytest.raises(SystemExit, match="cannot reach SAGE_VOICE_STT_URL"):
        app.run(Robot(calls), threading.Event())
    assert calls[1:] == []
    with pytest.raises(SystemExit, match="cannot reach SAGE_VOICE_STT_URL"):
        voice.main(env=ENV, robot=lambda **kw: Robot(calls, **kw), run=None, moves=dict)
    assert calls[1:] == []


def test_a_service_is_reachable_if_it_answers_at_all():
    asked = []

    def get(url, timeout):
        asked.append(url)
        if "8776" in url:  # the kernel: no health route, any HTTP answer will do
            raise urllib.error.HTTPError(url, 401, "unauthorized", {}, None)
        if "8781" in url:
            raise urllib.error.URLError("connection refused")
        return type("Answer", (), {"__enter__": lambda s: s, "__exit__": lambda s, *e: None})()

    env = {**ENV, "SAGE_VOICE_TTS_URL": "http://127.0.0.1:8782/v1/"}
    assert voice.unreachable(env, get=get) == ["SAGE_VOICE_STT_URL"]
    assert asked == ["http://127.0.0.1:8776", "http://127.0.0.1:8781/health",
                     "http://127.0.0.1:8782/health"]
