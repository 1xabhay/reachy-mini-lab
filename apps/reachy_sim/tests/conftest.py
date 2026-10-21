"""One sim daemon for the session, on localhost, on a port of its own."""

import json
import math
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

# Not 8000 (the SDK default) and not 8742 (`make sim`), so tests run beside either.
PORT = 8743
API = f"http://127.0.0.1:{PORT}/api"
APP = Path(__file__).parent.parent


def api(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(API + path, data, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.load(resp)


def pose(roll=0.0, pitch=0.0, yaw=0.0):
    """Build a head pose from degrees, in the radians the REST API wants."""
    rad = {"roll": roll, "pitch": pitch, "yaw": yaw}
    return {"x": 0, "y": 0, "z": 0} | {k: math.radians(v) for k, v in rad.items()}


def present():
    """Read the state back, in degrees, plus raw joints for rendering."""
    s = api("/state/full?with_head_joints=true&with_body_yaw=true")
    deg = {k: math.degrees(s["head_pose"][k]) for k in ("roll", "pitch", "yaw")}
    return deg | {
        "body_yaw": math.degrees(s["body_yaw"]),
        "head_joints": s["head_joints"],
        "antennas": s["antennas_position"],
    }


def goto(settle=4.0, body_yaw=None, **head):
    """Move over 1 s and wait for the slow tail of the sim's actuators (to ~0.5°)."""
    body = {"head_pose": pose(**head), "duration": 1.0}
    if body_yaw is not None:
        body["body_yaw"] = math.radians(body_yaw)
    api("/move/goto", body)
    time.sleep(settle)
    return present()


@pytest.fixture(scope="session")
def sim():
    proc = subprocess.Popen(
        [sys.executable, str(APP / "sim.py"), str(PORT), "minimal"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(120):
            try:
                if api("/daemon/status")["state"] == "running":
                    break
            except OSError:
                pass
            time.sleep(0.5)
        else:
            pytest.fail("sim daemon did not come up")
        yield
    finally:
        proc.terminate()
        proc.wait(timeout=30)


@pytest.fixture
def neutral(sim):
    """Every test starts from head and body at zero."""
    return goto(body_yaw=0)
