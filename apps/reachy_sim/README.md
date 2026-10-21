# reachy_sim

Pollen's MuJoCo simulator of Reachy Mini (SDK 1.10.0, mujoco 3.3.0) as a test
layer: try what the robot app asks of the body without the robot. Its own uv
project and `.venv`, outside the root workspace.

```sh
make sim                 # daemon on 127.0.0.1:8742, scene `minimal` (table, duck, food)
make sim SCENE=desk      # scenes/desk.xml: a screen and a person, for look-at tests
make test                # starts its own daemon on 127.0.0.1:8743
make limits              # the limits table below
```

`sim.py` runs Pollen's daemon with `--sim --headless --no-media`, bound to
127.0.0.1, named `reachy_sim`, and with its mDNS advertisement switched off: the
stock daemon announces itself on the LAN unconditionally, and a sim must never
be found as a robot. Clients reach it over the daemon's REST API; an SDK client
must pass `host="127.0.0.1", port=8742, connection_mode="localhost_only"`, since
`auto` falls back to `reachy-mini.local`.

The headless daemon renders no camera (Pollen starts the render thread only with
the viewer window), so `sim.eye_camera()` renders the eye offscreen in-process
at 320x180 from the joints read back from the daemon, RGB or per-geom
segmentation.

## Numbers (M2 Pro, CPU physics, 2026-10-20)

- Daemon ready: 13 s after launch. Idle: ~70 MB RSS, ~8 % of one core.
- Tracking: a 1 s `goto` to head yaw 30° reads 27.5° at 1.5 s, 29.9° at 3 s,
  30.3° at 6 s; 60° reads 55.4°, 58.5°, 60.4°. The tests settle 4 s.
- Eye camera: 3 s of physics plus one render ≈ 0.3 s (first call 2.7 s, GL setup).
  Yawing left 30° moves the duck > 30 px towards the image centre.
- Suite: 23 tests, about 2–3 min (each limits case settles twice).

Limits, each from neutral via a 1 s `goto`, read back after 4 s:

| command | reads | documented | reached |
|---|---|---|---|
| head pitch +60 | pitch | ±40° | +42.5° |
| head pitch -60 | pitch | ±40° | -43.8° |
| head roll +60 | roll | ±40° | +45.6° to +48.1° |
| head roll -60 | roll | ±40° | -47.2° to -48.9° |
| head yaw +200 | yaw | ±180° | -159.4° (wraps, then the body limit) |
| body yaw ±200, head with it | body_yaw | ±180° | ±160.0° |
| body yaw +170, head at 0 | body_yaw − head_yaw | ±65° | +64.4° |

- Pitch and roll are not clamped at 40°: `goto` stops wherever IK first fails on
  the way, 42–49°, and that varies by a few degrees run to run.
- Body yaw stops at ±160° (the kinematics' 2.8 rad), not ±180°.
- Automatic body yaw is a deadband: the body only follows once |body − head|
  passes 65°, so body yaw depends on the path taken.
- A single `set_target` beyond the pitch limit (the streaming path) is dropped,
  not clamped: the head stays where it was.

## The sim as a Sage device

`device.turn()` speaks device protocol v1 to the kernel: hello with one `emote`
tool (the voice app's 60 recorded moves, `max_s` = longest move + 1 s = 21),
a `submitted` text turn (retried with the same `turn_id` while the kernel is
busy, given up after 5 s; `rate_limited` resent after `retry_after`), then each
effect. An emote plays as the recorded move in the sim daemon (offline, from
the local Hugging Face cache), beside the poll as on the robot, and is reported
`done`, `failed` or, on a Stop from any device, `interrupted`. A reply is
`done` once received (no speaker), with `ttft_ms`. `limits.clamp()` is what the
app sends through: pitch and roll ±40°, body yaw ±160°.

The tests run sage-kernel with its conformance fake model, never Ollama.
"Feel happy" → `cheerful1`: played in 3.0 s (the move is 2.8 s) while the reply
arrived at `ttft_ms` 11 (3131 when moves played inline); the head joints moved
up to 44° and the eye image changed. A Stop 0.5 s into the move ended it at
1.0 s. A newer emote supersedes the playing one (`interrupted`, reason
`superseded`), as on the pet. Every move result carries its `duration_ms`,
which the kernel journals in `turn.timing.durations_ms`.

## What the sim does not do

No audio (no mic, speaker or direction of arrival), no camera stream or media
server (`--no-media`; the eye is rendered in-process instead), no faces (the
desk "person" is a sphere; the SDK's face detector has nothing to find), no
motor torque or control modes (always "enabled").
