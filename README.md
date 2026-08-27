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

## Robot notes

- Unit: Reachy Mini **wireless** (`wireless_version: true`), so it has an IMU;
  the Lite has none and the IMU check reports "skipped" rather than failing.
- Apps run on the robot's own Pi when launched through the daemon; run from a
  laptop they connect over the network and media arrives via WebRTC instead of
  local IPC. Both paths are supported, WebRTC is the slower one.
