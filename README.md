# reachy-mini

Apps, diagnostics and experiments for a Reachy Mini robot.

## Setup

Requires [uv](https://docs.astral.sh/uv/). One virtualenv at the repo root covers
every app in the workspace:

```sh
uv sync
```

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `REACHY_MINI_HOST` | `reachy-mini.local` | Daemon hostname or IP |

| `REACHY_MINI_PORT` | `8000` | Daemon port |
| `TESTBENCH_DATA_DIR` | `/tmp/reachy_mini_testbench` | Where captures and recordings are written |

`reachy-mini.local` resolves intermittently on some networks — a
[known mDNS issue](https://huggingface.co/docs/reachy_mini/en/troubleshooting).
Set `REACHY_MINI_HOST` to the robot's IP if it drops.

The robot must be powered on, on the same network, and its **motor backend
started** before anything can reach the motors. `reachy-diag` says so explicitly
when the backend is stopped. Start it from the robot dashboard
(`http://reachy-mini.local:8000`) or:

```sh
curl -X POST http://reachy-mini.local:8000/api/daemon/start
```

Starting the backend energises the motors and the head will hold position.

## Diagnostics

```sh
uv run reachy-diag                       # read-only health report, robot stays still
uv run reachy-diag --motion              # + calibration sweeps      (MOVES THE ROBOT)
uv run reachy-diag --vision              # + camera rotation check   (MOVES THE ROBOT)
uv run reachy-diag --json out/report.json
```

Exits `0` when every check passes and `1` otherwise, so it drops straight into CI.
Motion is opt-in: the bare command never commands a movement.

## Testbench UI

```sh
uv run python -m reachy_mini_testbench.main    # then open http://localhost:8042
```

Or install it on the robot and run it as a managed app:

```sh
reachy-mini app run reachy_mini_testbench
```

See [apps/reachy_mini_testbench/README.md](apps/reachy_mini_testbench/README.md)
for the API surface and what each check measures.

## Layout

```
apps/<app_name>/          one installable Reachy Mini app per directory
tests/                    tests for the workspace
```

New apps go in `apps/`; `uv sync` picks them up automatically via
`[tool.uv.workspace]`. Each needs a `reachy_mini_apps` entry point in its own
`pyproject.toml` to be launchable by the robot daemon — copy the shape of
`apps/reachy_mini_testbench/pyproject.toml`, or scaffold one with
`uv run reachy-mini app create`.

## Development

```sh
uv run pytest              # runs without a robot attached
uv run ruff check apps tests
uv run ruff format apps tests
```

The test suite uses a stub robot, so everything above passes on a laptop with
nothing plugged in.

## Updates

```sh
uv sync --upgrade                       # refresh all dependencies within their ranges
uv lock --upgrade-package reachy-mini   # bump just the SDK
```

Keep the SDK version aligned with the daemon running on the robot — the SDK warns
on a mismatch at connect time. Check the robot's version with:

```sh
curl -s http://reachy-mini.local:8000/api/daemon/status
```

## Baseline for unit `2f102f4682d69822`

Measured 2026-08-27, SDK/daemon 1.10.0. Healthy: 9 joints, IMU 9.76 m/s² at rest,
camera 1280x720 at ~19 fps over WebRTC, mic 16 kHz stereo, control loop 49.7 Hz
with 0 errors. Antennas track to 0.2° RMS.

### The head does not return to level, and why

Commanding neutral leaves the head several degrees off, and **where it lands
depends entirely on the direction it approached from**:

| Approach to a 0° roll command | Lands at |
| --- | --- |
| from +15° / +20° | **+7 to +14°** |
| from −15° / −20° | **−3 to −5°** |

That is a friction band roughly 11–17° wide. Inside it, repeating the same
command does not move the head at all — the error is not noise, and the robot
repeats to **0.15°** across trials.

The cause is in the shipped configuration, not this unit: all six Stewart
servos run **P=300, I=0, D=0** (`hardware_config.yaml`). With no integral term,
each motor stops where the proportional term balances friction and nothing ever
removes the residual. Both the daemon's forward kinematics and the camera
(ORB/RANSAC homography) agree on the resulting pose, and it does not drift with
duty cycle or temperature — 14 cycles held −5.05° ± 0.15° at a flat 44 °C.

**This is why the head looks crooked after `wake_up()`**: its final gesture is a
+20° roll before returning to neutral, so it always lands on the high side of
the band.

### The fix

`settle_to_pose()` supplies the missing integral action in software: command,
measure the residual, fold it back in. Measured end to end:

| | roll |
| --- | --- |
| after `wake_up()` | **+10.4°** |
| after `settle_to_pose()` | **−0.9°** |

It converges in 3–6 iterations and plateaus around 0.7°, which is the band's
floor — below that a correction is too small to break stiction. Available as
`POST /api/level_head`, the "Level head" button, and `calibration.level_head()`.
Use it wherever pose accuracy matters; plain `goto_target` is good to about ±5°.

Two other consequences worth knowing:

- `zero_offset` in the calibration suite reads differently run to run, because
  it inherits wherever the previous test left the head. That variation *is* the
  band.
- **Head yaw is not a separate problem.** The gain of ≈0.70 over ±15° was the
  friction band and nothing else: with closed-loop settling the Stewart platform
  hits every yaw target from 5° to 30° to within 0.85°. No saturation, no
  shortfall.
- **Body rotation has the same P-only shortfall**, and it is worse — around 25%
  open-loop. It is a separate joint that does *not* appear in the head pose
  (which the daemon reports relative to the body), so it has to be read from
  joint 0 and corrected on its own. `settle_to_pose` now does.

  | Commanded body yaw | Open loop | Corrected |
  | --- | --- | --- |
  | +10° | +6.2° | **+9.3°** |
  | +20° | +14.8° | **+20.9°** |
  | −20° | −14.8° | **−20.0°** |
  | +30° | +23.5° | **+31.3°** |

Physical inspection found nothing wrong — no red LEDs, no stiff leg, no detached
rods, no squeaking — which rules out the assembly and motor faults in Pollen's
[motors diagnosis guide](https://huggingface.co/docs/reachy_mini/en/troubleshooting/motors_diagnosis).
For motor-level faults use [Pollen's own testbench app](https://huggingface.co/spaces/pollen-robotics/reachy_mini_testbench),
which can scan, verify and reflash motors; this one cannot, because those need
the daemon stopped to take the serial port.

The accuracy tolerances in
[calibration.py](apps/reachy_mini_testbench/reachy_mini_testbench/calibration.py)
are ours, not a Pollen specification, and several are tighter than this hardware
can meet open-loop. `repeatability_*` is the check that would indicate a real
fault if it started failing.

## Robot notes

- Unit: Reachy Mini **wireless** (`wireless_version: true`), so it has an IMU;
  the Lite has none and the IMU check reports "skipped" rather than failing.
- Apps run on the robot's own Pi when launched through the daemon; run from a
  laptop they connect over the network and media arrives via WebRTC instead of
  local IPC. Both paths are supported, WebRTC is the slower one.
