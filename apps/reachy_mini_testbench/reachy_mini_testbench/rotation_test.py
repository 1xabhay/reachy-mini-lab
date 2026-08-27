"""Vision-based validation of head rotation.

Commands a known head rotation, compares the camera view before and after,
and recovers the rotation the *image* says happened.  Agreement between the
commanded angle and the visually measured one cross-checks the kinematics,
the motor calibration and the camera mounting in a single shot - none of
which the joint encoders alone can catch, since they happily report a
perfectly tracked pose on a head that is mechanically mis-assembled.

Axis convention is taken from the SDK's CAD extrinsics
(`DEFAULT_HEAD_TO_CAMERA_TRANSFORM`) rather than hand-derived, so it stays
correct if Pollen ever remounts the camera.
"""

from __future__ import annotations

import math
import time
from typing import Any

import cv2
import numpy as np
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
from reachy_mini.vision.look_at import DEFAULT_HEAD_TO_CAMERA_TRANSFORM

#: Rotation from camera-optical axes to head axes, straight from the SDK's CAD.
R_HEAD_CAM = DEFAULT_HEAD_TO_CAMERA_TRANSFORM[:3, :3]

#: A rotation this small is swamped by feature-matching noise.
MIN_TEST_ANGLE_DEG = 5.0

#: Matching quality floors - below these the measurement is not trustworthy
#: and the test reports an error rather than a bogus PASS/FAIL.
MIN_MATCHES = 30
MIN_INLIERS = 15

#: Default acceptance: measured angle must be within this of commanded.
TOL_ANGLE_DEG = 3.0


def _to_gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame


def camera_matrix(mini: ReachyMini) -> np.ndarray:
    """Return the intrinsics matching the frames `mini.media.get_frame()` returns.

    The daemon publishes `K` for the full sensor readout; the SDK's camera
    object already rescales it for the streamed resolution *and* the digital
    crop, so take it from there rather than rescaling by hand.
    """
    camera = mini.media.camera
    if camera is None or camera.K is None:
        raise RuntimeError("camera intrinsics unavailable - is the media backend up?")
    return np.asarray(camera.K, dtype=np.float64)


def match_frames(ref: np.ndarray, cur: np.ndarray, n_features: int = 2000) -> dict[str, Any]:
    """ORB-match two frames and return the corresponding point sets."""
    orb = cv2.ORB_create(nfeatures=n_features)
    kp1, des1 = orb.detectAndCompute(_to_gray(ref), None)
    kp2, des2 = orb.detectAndCompute(_to_gray(cur), None)

    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return {
            "n_keypoints": (len(kp1 or []), len(kp2 or [])),
            "matches": 0,
            "src": np.empty((0, 2)),
            "dst": np.empty((0, 2)),
        }

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    # Lowe's ratio test - crossCheck would be cheaper but keeps far more
    # ambiguous matches on the repetitive textures typical of a desk scene.
    good = [m for m, n in matcher.knnMatch(des1, des2, k=2) if m.distance < 0.75 * n.distance]

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)
    return {"n_keypoints": (len(kp1), len(kp2)), "matches": len(good), "src": src, "dst": dst}


def measure_rotation(ref: np.ndarray, cur: np.ndarray, K: np.ndarray) -> dict[str, Any]:
    """Recover the head rotation between two frames.

    Two independent estimators are reported:

    * `homography` - fits `H = K A K^-1` with RANSAC and pulls the rotation
      vector out of `A`.  Uses the whole field of view and gives all three
      axes, so it is the primary estimate.
    * `shift` - median pixel displacement through `atan(dx/fx)`.  Only valid
      near the image centre, but it depends on almost nothing, so it is a
      useful sanity check on the homography.
    """
    m = match_frames(ref, cur)
    result: dict[str, Any] = {
        "matches": m["matches"],
        "keypoints": list(m["n_keypoints"]),
        "inliers": 0,
        "ok": False,
        "error": None,
    }

    if m["matches"] < MIN_MATCHES:
        result["error"] = (
            f"only {m['matches']} good matches (need {MIN_MATCHES}); scene too plain or too blurred"
        )
        return result

    H, mask = cv2.findHomography(m["src"], m["dst"], cv2.RANSAC, 3.0)
    if H is None:
        result["error"] = "homography estimation failed"
        return result

    inlier_mask = mask.ravel().astype(bool)
    result["inliers"] = int(inlier_mask.sum())
    if result["inliers"] < MIN_INLIERS:
        result["error"] = f"only {result['inliers']} RANSAC inliers (need {MIN_INLIERS})"
        return result

    # A is the scene's apparent rotation in camera coordinates.  Re-orthonormalise
    # because K^-1 H K is only a rotation up to estimation noise.
    A = np.linalg.inv(K) @ H @ K
    u, _, vt = np.linalg.svd(A)
    A = u @ vt
    if np.linalg.det(A) < 0:
        A = u @ np.diag([1.0, 1.0, -1.0]) @ vt
    rotvec_cam = cv2.Rodrigues(A)[0].ravel()

    # `A` is how the *scene* turned, which is the inverse of how the head
    # turned - hence the negation - and it lives in camera axes, so rotate it
    # into the head frame before naming the components roll/pitch/yaw.
    roll_h, pitch_h, yaw_h = np.degrees(-R_HEAD_CAM @ rotvec_cam)

    d = m["dst"][inlier_mask] - m["src"][inlier_mask]
    dx, dy = float(np.median(d[:, 0])), float(np.median(d[:, 1]))
    fx, fy = float(K[0, 0]), float(K[1, 1])

    result.update(
        ok=True,
        homography={"roll": float(roll_h), "pitch": float(pitch_h), "yaw": float(yaw_h)},
        shift={
            "dx_px": dx,
            "dy_px": dy,
            "yaw": math.degrees(math.atan2(dx, fx)),
            # image-down is head -Z, so a nose-down pitch slides the scene up
            "pitch": -math.degrees(math.atan2(dy, fy)),
        },
    )
    return result


def _grab(mini: ReachyMini, retries: int = 15, delay: float = 0.1) -> np.ndarray:
    """Pull a camera frame, tolerating the stream not being warm yet."""
    for _ in range(retries):
        frame = mini.media.get_frame()
        if frame is not None:
            return frame
        time.sleep(delay)
    raise RuntimeError("no camera frame available - is the media backend up?")


def validate_rotation(
    mini: ReachyMini,
    K: np.ndarray,
    axis: str = "yaw",
    angle_deg: float = 20.0,
    tolerance_deg: float = TOL_ANGLE_DEG,
    duration: float = 1.0,
    settle: float = 0.6,
) -> dict[str, Any]:
    """Rotate the head by a known angle and check the camera agrees.

    The head is returned to neutral before and after, so the test is
    repeatable and leaves the robot where it found it.
    """
    if axis not in ("yaw", "pitch", "roll"):
        raise ValueError(f"axis must be yaw, pitch or roll, got {axis!r}")
    if abs(angle_deg) < MIN_TEST_ANGLE_DEG:
        raise ValueError(
            f"|angle| must be at least {MIN_TEST_ANGLE_DEG} deg to rise above matching noise"
        )

    mini.goto_target(head=create_head_pose(), body_yaw=0.0, duration=duration)
    time.sleep(settle)
    ref = _grab(mini)

    mini.goto_target(head=create_head_pose(**{axis: angle_deg}), body_yaw=0.0, duration=duration)
    time.sleep(settle)
    cur = _grab(mini)

    mini.goto_target(head=create_head_pose(), body_yaw=0.0, duration=duration)

    h, w = ref.shape[:2]
    meas = measure_rotation(ref, cur, K)
    meas["frame_size"] = [w, h]
    meas["axis"] = axis
    meas["expected_deg"] = angle_deg

    if not meas["ok"]:
        meas["passed"] = False
        meas["detail"] = meas["error"]
        return meas

    measured = meas["homography"][axis]
    error = measured - angle_deg
    meas["measured_deg"] = measured
    meas["error_deg"] = error
    meas["passed"] = abs(error) <= tolerance_deg
    meas["detail"] = (
        f"commanded {angle_deg:+.1f} deg {axis}, camera measured {measured:+.2f} deg "
        f"(error {error:+.2f}, tol ±{tolerance_deg}) from {meas['inliers']} inliers"
    )
    return meas


def calibrate_visual_scale(
    mini: ReachyMini,
    K: np.ndarray,
    axis: str = "yaw",
    angles_deg: tuple[float, ...] = (-20.0, -10.0, 10.0, 20.0),
    duration: float = 1.0,
    settle: float = 0.6,
) -> dict[str, Any]:
    """Fit the camera-to-kinematics angular scale over several angles.

    The published intrinsics are calibrated at full sensor resolution, and the
    streamed frames are a cropped and rescaled view of that, so the effective
    focal length is not simply a ratio away.  Rather than guess the crop, this
    measures pixels-per-degree directly: it fits `dx = k * angle` through the
    origin and reports the implied effective `fx`, which callers can feed back
    into `measure_rotation`.
    """
    mini.goto_target(head=create_head_pose(), body_yaw=0.0, duration=duration)
    time.sleep(settle)
    ref = _grab(mini)

    samples: list[dict[str, Any]] = []
    for angle in angles_deg:
        mini.goto_target(head=create_head_pose(**{axis: angle}), body_yaw=0.0, duration=duration)
        time.sleep(settle)
        meas = measure_rotation(ref, _grab(mini), K)
        samples.append(
            {
                "angle_deg": angle,
                "ok": meas["ok"],
                "dx_px": meas["shift"]["dx_px"] if meas["ok"] else None,
                "dy_px": meas["shift"]["dy_px"] if meas["ok"] else None,
                "inliers": meas["inliers"],
                "error": meas["error"],
            }
        )

    mini.goto_target(head=create_head_pose(), body_yaw=0.0, duration=duration)

    usable = [s for s in samples if s["ok"]]
    if len(usable) < 2:
        return {
            "ok": False,
            "error": "not enough usable samples to fit a scale",
            "samples": samples,
        }

    a = np.array([s["angle_deg"] for s in usable], dtype=float)
    d = np.array([s["dx_px"] if axis == "yaw" else s["dy_px"] for s in usable], dtype=float)
    # Through the origin: zero rotation must mean zero shift.
    px_per_deg = float(a @ d / (a @ a))
    residual = d - px_per_deg * a

    h, w = ref.shape[:2]
    return {
        "ok": True,
        "axis": axis,
        "frame_size": [w, h],
        "px_per_deg": px_per_deg,
        "fx_effective": px_per_deg / math.tan(math.radians(1.0)),
        "rms_residual_px": float(np.sqrt(np.mean(residual**2))),
        "samples": samples,
    }
