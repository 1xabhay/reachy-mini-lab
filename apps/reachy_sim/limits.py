"""The body's limits, applied by the app before it sends.

The daemon drops a single target beyond them instead of clamping it (see README).
"""

HEAD_PITCH_ROLL_DEG = 40.0  # Pollen's docs
# The kinematics' 2.8 rad, as measured in the sim; Pollen's docs say 180.
# Verify on the real robot.
BODY_YAW_DEG = 160.0
LIMITS = {"roll": HEAD_PITCH_ROLL_DEG, "pitch": HEAD_PITCH_ROLL_DEG, "body_yaw": BODY_YAW_DEG}


def clamp(pose: dict[str, float]) -> dict[str, float]:
    """Clamp the limited fields of a pose in degrees; pass the others through."""
    return {k: max(-LIMITS[k], min(LIMITS[k], v)) if k in LIMITS else v for k, v in pose.items()}
