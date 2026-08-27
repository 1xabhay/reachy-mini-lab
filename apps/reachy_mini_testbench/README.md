# Reachy Mini Debug & CI Testbench

Web toolbox for debugging, calibrating and validating a Reachy Mini. The same
checks are available headless through `reachy-diag` for CI.

## Running

```sh
uv run python -m reachy_mini_testbench.main   # http://localhost:8042
reachy-mini app run reachy_mini_testbench     # as a daemon-managed app
```

## What it measures

### Calibration sweep — `POST /api/test/calibration`

Non-destructive. Commands poses through the normal motion API and compares them
against the robot's own feedback; nothing is written to motor EEPROM or
`hardware_config.yaml` (use `reachy_mini.tools` for that).

| Check | Measures | Fails when |
| --- | --- | --- |
| `zero_offset` | residual pose at the neutral command | > 2° or > 3 mm off |
| `axis_sweep_{roll,pitch,yaw}` | gain, offset, RMS error, backlash over an up-and-down sweep | gain off by > 0.15, RMS > 2.5°, or backlash > 2° |
| `antennas` | per-antenna tracking error, commanded mirrored so a swapped cable shows as a sign error | RMS > 2.5° |

### Rotation validation — `POST /api/test/rotation_validation`

Rotates by a known angle and recovers the rotation the *camera* saw, using ORB
features and a RANSAC homography (`H = K A K⁻¹`). This catches what encoders
cannot: a head that tracks its commanded pose perfectly while being mechanically
mis-assembled. Camera-to-head axes come from the SDK's CAD extrinsics rather than
being hand-derived.

Reports two independent estimates — the homography (primary, uses the full field)
and the median pixel shift (a cheap sanity check on it). If matching is too weak
it reports an error rather than a bogus verdict.

### Visual scale calibration — `POST /api/test/visual_scale`

Fits pixels-per-degree across several angles and reports the implied effective
`fx`, cross-checking the published intrinsics against reality.

## API

| Method | Path | Notes |
| --- | --- | --- |
| `GET` | `/api/status` | head pose, antennas, IMU, busy flag |
| `GET` | `/api/motor_status` | all 9 motors: `stewart_1..6`, `body_rotation`, both antennas |
| `POST` | `/api/move_head` | roll/pitch/yaw (deg), x/y/z (mm), `body_yaw`, `duration` |
| `POST` | `/api/move_antennas` | `left`, `right` (deg), `duration` |
| `POST` | `/api/go_to_zero` · `/api/wake_up` · `/api/go_to_sleep` | |
| `GET` | `/api/camera/stream` | MJPEG |
| `GET` | `/api/camera/capture` | single JPEG |
| `POST` `GET` `DELETE` | `/api/camera/save` · `/list` · `/download/{f}` · `/delete/{f}` | |
| `POST` | `/api/audio/start_recording` · `/stop_recording` | WAV, 120 s cap |
| `GET` `POST` `DELETE` | `/api/audio/list` · `/download/{f}` · `/play/{f}` · `/delete/{f}` | |
| `POST` | `/api/test/calibration` · `/rotation_validation` · `/visual_scale` | |
| `GET` | `/api/test/last_calibration_result` · `/last_rotation_result` | |

Anything that moves the robot takes an exclusive lock and returns **409** if a
movement or test is already running, so a slider drag can never interleave with a
sweep. Slider ranges are enforced server-side (±40° head, ±25 mm, ±150° antennas),
not just in the browser.

## Storage

Set by `TESTBENCH_DATA_DIR`, default `/tmp/reachy_mini_testbench`:

```
captures/     JPEG frames
recordings/   WAV recordings
reports/      diagnostic JSON
```

Filenames are validated against path traversal before any read or delete.

Built for [Pollen Robotics](https://www.pollen-robotics.com/) Reachy Mini.
