"""End-to-end checks of the testbench HTTP surface against a stub robot."""

import numpy as np
import pytest

EXPECTED_MOTOR_NAMES = [
    "stewart_1",
    "stewart_2",
    "stewart_3",
    "stewart_4",
    "stewart_5",
    "stewart_6",
    "body_rotation",
    "left_antenna",
    "right_antenna",
]


def test_status_reports_pose_and_imu(client):
    body = client.get("/api/status").json()

    assert body["connected"] is True
    assert body["head_pose"]["yaw"] == pytest.approx(0.0)
    assert body["imu"]["temperature"] == 31.5
    assert body["busy"] is False


def test_motor_status_lists_all_nine_motors(client):
    body = client.get("/api/motor_status").json()

    assert body["count"] == 9
    assert [m["name"] for m in body["motors"]] == EXPECTED_MOTOR_NAMES


def test_move_head_converts_degrees_and_millimetres(client, robot):
    r = client.post("/api/move_head", json={"yaw": 30, "z": 10, "duration": 0.1})

    assert r.status_code == 200
    head = robot.calls[-1][1]["head"]
    # 30 deg about Z, and 10 mm expressed in metres.
    assert np.degrees(np.arctan2(head[1, 0], head[0, 0])) == pytest.approx(30.0)
    assert head[2, 3] == pytest.approx(0.010)


def test_move_antennas_converts_to_radians(client, robot):
    client.post("/api/move_antennas", json={"left": 90, "right": -90, "duration": 0.1})

    assert robot.calls[-1][1]["antennas"] == pytest.approx([np.pi / 2, -np.pi / 2])


@pytest.mark.parametrize(
    ("payload", "field"),
    [({"yaw": 400}, "yaw"), ({"duration": 0}, "duration"), ({"z": 900}, "z")],
)
def test_out_of_range_motion_is_rejected(client, robot, payload, field):
    r = client.post("/api/move_head", json=payload)

    assert r.status_code == 422, f"{field} should be clamped by the schema"
    assert robot.calls == []


@pytest.mark.parametrize("path", ["/api/go_to_zero", "/api/wake_up", "/api/go_to_sleep"])
def test_quick_actions_reach_the_robot(client, robot, path):
    assert client.post(path).status_code == 200
    assert robot.calls


def test_camera_capture_returns_a_jpeg(client):
    r = client.get("/api/camera/capture")

    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert r.content[:2] == b"\xff\xd8"  # JPEG SOI marker


def test_capture_save_list_download_delete_round_trip(client):
    saved = client.post("/api/camera/save").json()
    listing = client.get("/api/camera/list").json()["captures"]

    assert [c["filename"] for c in listing] == [saved["filename"]]

    download = client.get(f"/api/camera/download/{saved['filename']}")
    assert download.status_code == 200
    assert len(download.content) == saved["size"]

    assert client.delete(f"/api/camera/delete/{saved['filename']}").status_code == 200
    assert client.get("/api/camera/list").json()["captures"] == []


def test_audio_record_produces_a_playable_wav(client, robot):
    assert client.post("/api/audio/start_recording").json()["recording"] is True

    stopped = client.post("/api/audio/stop_recording").json()
    assert stopped["samplerate"] == 16000
    assert stopped["channels"] == 2
    assert stopped["peak"] == pytest.approx(0.25, abs=1e-3)

    client.post(f"/api/audio/play/{stopped['filename']}")
    assert robot.media.played and robot.media.played[-1].endswith(stopped["filename"])


def test_double_start_recording_is_rejected(client):
    client.post("/api/audio/start_recording")
    try:
        assert client.post("/api/audio/start_recording").status_code == 409
    finally:
        client.post("/api/audio/stop_recording")


def test_stop_without_start_is_rejected(client):
    assert client.post("/api/audio/stop_recording").status_code == 409


@pytest.mark.parametrize("filename", ["../../etc/passwd", "..%2f..%2fsecret", "nested/file.jpg"])
def test_path_traversal_is_refused(client, filename):
    r = client.get(f"/api/camera/download/{filename}")

    assert r.status_code in (400, 404), r.text
    assert b"root:" not in r.content


def test_missing_file_is_a_404(client):
    assert client.get("/api/camera/download/nope.jpg").status_code == 404


def test_last_results_start_empty(client):
    assert client.get("/api/test/last_rotation_result").json()["result"] is None
    assert client.get("/api/test/last_calibration_result").json()["result"] is None
