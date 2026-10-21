"""Settings from a .env file, as Pollen's conversation app does it.

The daemon launches the app with only its own environment, so a .env in the
reachy_mini config directory (or above the working directory, for ``make run``)
fills the settings the environment leaves unset. Only temporary files here: a
real .env is never read.
"""

import os
import re
import subprocess
import threading
from pathlib import Path

import pytest
from test_app import SETTINGS, app  # noqa: F401 (a fixture)
from test_run import Robot

import reachy_sage_voice.__main__ as voice

APP = Path(__file__).parent.parent
TOKEN = "SAGE_VOICE_KERNEL_TOKEN"
NAME = re.compile(r"SAGE_VOICE_[A-Z_]*[A-Z]")


@pytest.fixture
def files(tmp_path, monkeypatch):
    """Two .env files, config directory first; the environment is restored after."""
    saved = dict(os.environ)
    paths = [tmp_path / "config.env", tmp_path / "cwd.env"]
    monkeypatch.setattr(voice, "env_files", lambda: paths, raising=False)
    yield paths
    os.environ.clear()
    os.environ.update(saved)


def test_a_dotenv_fills_what_the_environment_leaves_unset(files):
    os.environ.pop("SAGE_VOICE_KERNEL_URL", None)
    os.environ["SAGE_VOICE_TTS_VOICE"] = "af_heart"
    files[0].write_text("SAGE_VOICE_KERNEL_URL=http://from-file\nSAGE_VOICE_TTS_VOICE=am_adam\n")
    voice.load_settings()
    assert os.environ["SAGE_VOICE_KERNEL_URL"] == "http://from-file"
    assert os.environ["SAGE_VOICE_TTS_VOICE"] == "af_heart"  # the real environment wins


def test_the_config_directory_file_wins_over_the_working_directory_one(files):
    os.environ.pop("SAGE_VOICE_KERNEL_URL", None)
    os.environ.pop("SAGE_VOICE_STT_URL", None)
    files[0].write_text("SAGE_VOICE_KERNEL_URL=http://config\n")
    files[1].write_text("SAGE_VOICE_KERNEL_URL=http://cwd\nSAGE_VOICE_STT_URL=http://cwd-stt\n")
    voice.load_settings()
    assert os.environ["SAGE_VOICE_KERNEL_URL"] == "http://config"
    assert os.environ["SAGE_VOICE_STT_URL"] == "http://cwd-stt"


def test_the_app_the_daemon_launches_reads_the_dotenv(app, files):  # noqa: F811
    app, ran = app
    del os.environ[TOKEN]
    files[0].write_text(f"{TOKEN}={SETTINGS[TOKEN]}\n")
    app.run(Robot([]), threading.Event())
    assert ran["env"][TOKEN] == SETTINGS[TOKEN]


def test_the_example_lists_every_setting_and_no_token():
    code = "".join(p.read_text() for p in (APP / "reachy_sage_voice").glob("*.py"))
    example = (APP / ".env.example").read_text()
    assert set(NAME.findall(code)) <= set(NAME.findall(example))
    assert re.search(rf"^{TOKEN}=$", example, re.M)


def test_a_dotenv_is_never_committed():
    assert subprocess.run(["git", "check-ignore", "-q", str(APP / ".env")]).returncode == 0
