"""Commands beyond each documented limit (Pollen's docs) and what the sim actually reaches."""

import time

import limits

from conftest import api, goto, pose, present

# (case, command in degrees, reached value to read, documented limit)
BEYOND = [
    ("head pitch +60", {"pitch": 60}, "pitch", 40),
    ("head pitch -60", {"pitch": -60}, "pitch", 40),
    ("head roll +60", {"roll": 60}, "roll", 40),
    ("head roll -60", {"roll": -60}, "roll", 40),
    ("head yaw +200 (wraps)", {"yaw": 200}, "yaw", 180),
    ("body yaw +200, head with it", {"yaw": 179, "body_yaw": 200}, "body_yaw", 180),
    ("body yaw -200, head with it", {"yaw": -179, "body_yaw": -200}, "body_yaw", 180),
    ("body yaw +170, head at 0", {"body_yaw": 170}, "body_yaw - head_yaw", 65),
]


def reached(state, key):
    if key == "body_yaw - head_yaw":
        return state["body_yaw"] - state["yaw"]
    return state[key]


def test_commands_beyond_each_documented_limit_are_clamped(neutral):
    rows, failures = [], []
    for case, command, key, limit in BEYOND:
        goto(body_yaw=0)
        value = reached(goto(**command), key)
        asked = command.get(key, command.get("body_yaw"))
        rows.append(f"| {case} | {key} | ±{limit}° | {value:+.1f}° |")
        # Clamped: well short of the command. The sim's stops are not exactly the
        # documented ones (see README), so the documented limit gets 12° of slack
        # (roll measured up to 48.9°).
        if abs(value) > abs(asked) - 10 or abs(value) > limit + 12:
            failures.append(case)
    print("\n| command | reads | documented | reached |\n|---|---|---|---|", *rows, sep="\n")
    assert not failures, failures


def test_a_single_target_beyond_the_pitch_limit_is_dropped_not_clamped(neutral):
    """set_target (the streaming path apps use) has no trajectory to stop part-way."""
    api("/move/set_target", {"target_head_pose": pose(pitch=60)})
    time.sleep(3)
    assert abs(present()["pitch"]) < 2


def test_the_app_clamps_pitch_and_roll_to_40_and_body_yaw_to_160():
    assert limits.clamp({"roll": -55.0, "pitch": 60.0, "yaw": 120.0, "body_yaw": 200.0}) == {
        "roll": -40.0,
        "pitch": 40.0,
        "yaw": 120.0,
        "body_yaw": 160.0,
    }
    assert limits.clamp({"pitch": -12.5}) == {"pitch": -12.5}
    assert limits.clamp({"body_yaw": -170.0}) == {"body_yaw": -160.0}


def test_a_clamped_target_is_reached_where_the_unclamped_one_was_dropped(neutral):
    api("/move/set_target", {"target_head_pose": pose(**limits.clamp({"pitch": 60}))})
    time.sleep(4)
    assert abs(present()["pitch"] - 40) < 2
