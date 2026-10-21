---
title: Reachy Sage Voice
emoji: 🎙️
colorFrom: red
colorTo: blue
sdk: static
pinned: false
short_description: Local real-time voice for Reachy Mini, Sage as its mind
license: apache-2.0
tags:
 - reachy_mini
---

# Reachy Sage Voice

Real-time voice for Reachy Mini, with a [Sage kernel](https://github.com/1xabhay/sage-kernel) (v0.1.0)
as its mind. It runs on [Pipecat](https://github.com/pipecat-ai/pipecat): mic,
barge-in on speech onset, Parakeet speech to text, the kernel, Kokoro speech
one clause at a time, then the speaker, with the robot's recorded moves as
emotes. Every model runs on your own machine.

**Install from source (`uv sync`); dashboard install is pending an onnxruntime
pin alignment between pipecat-ai and reachy-mini.**

## Quick start

Prerequisites:

- macOS or Linux, with [uv](https://docs.astral.sh/uv/) and Docker (Compose v2).
- A Reachy Mini (Lite or Wireless) with its daemon running, on your network.
- [Ollama](https://ollama.com) with the kernel's model:
  `ollama pull qwen3:4b-instruct` (Qwen3-4B-Instruct-2507, the live model).
- A Sage kernel ([github.com/1xabhay/sage-kernel](https://github.com/1xabhay/sage-kernel))
  running its device listener (protocol v1), with a token for this robot in its
  `SAGE_DEVICES`. Its README covers both.

Then, from this folder:

```sh
uv sync                  # this app's own .venv
make -C services up      # STT on 127.0.0.1:8781, TTS on 127.0.0.1:8782 (Docker)

cp .env.example .env     # then set SAGE_VOICE_KERNEL_TOKEN, this robot's token from the kernel
export REACHY_MINI_HOST=reachy-mini.local     # the robot's daemon (or set it in .env)
make run
```

Stop anything else driving the robot first. The run checks every setting and
that the kernel, STT and TTS answer, then enables the motors, wakes the robot
and talks until Ctrl-C. The first services build downloads the models into
the images; after that the services run offline.

## Where it runs

`make run` runs on your computer and reaches the robot over the network, for
both Lite and Wireless. The STT and TTS services listen on 127.0.0.1 only, so
the app must run on the same machine as them.

The app also has the standard `reachy_mini_apps` entry point
(`reachy_sage_voice.main:ReachySageVoice`), so a Reachy Mini daemon can launch
it. Launched that way, it runs wherever the daemon runs (your computer for a
Lite, the robot's Raspberry Pi for a Wireless). It inherits only the daemon's
environment, so put its settings in `~/.config/reachy_mini/sage_voice/.env`
(see "Settings"), and the daemon connects to the robot, so `REACHY_MINI_HOST`
is not needed. Installing it from the dashboard does not work yet, so the Space
is not tagged `reachy_mini_python_app` and the dashboard does not offer it: see
"Known limits". Running on the Pi is not a goal for 0.1.0.

## Settings

Every setting is listed, with its default, in [`.env.example`](.env.example).
Copy it to `.env` and fill it in, as for Pollen's conversation app. The app
reads, first one wins:

1. the environment;
2. `.env` in the reachy_mini config directory, beside the daemon's own config:
   `~/.config/reachy_mini/sage_voice/.env` on Linux
   (`~/Library/Application Support/reachy_mini/sage_voice/.env` on macOS). Use
   this one when the daemon launches the app;
3. the nearest `.env` in or above the working directory (this folder, for `make run`).

`.env` is gitignored: never commit a token. Addresses and the token have no
defaults, so nothing reaches the network unless you configure it to, and a
missing one stops the app with its name.

| Variable | Default | Purpose |
| --- | --- | --- |
| `SAGE_VOICE_KERNEL_URL` | required | The kernel's device listener (protocol v1) |
| `SAGE_VOICE_KERNEL_TOKEN` | required | This robot's token from the kernel's `SAGE_DEVICES` |
| `SAGE_VOICE_STT_URL` | required | Parakeet, OpenAI-compatible, with or without `/v1` |
| `SAGE_VOICE_TTS_URL` | required | Kokoro, OpenAI-compatible |
| `REACHY_MINI_HOST` | required for `make run` | The robot's daemon |
| `SAGE_VOICE_STT_MODEL` | `parakeet-tdt-0.6b-v2` | Model name sent to the STT |
| `SAGE_VOICE_TTS_MODEL` | `kokoro` | Model name sent to the TTS |
| `SAGE_VOICE_TTS_VOICE` | `af_heart` | Any Kokoro voice |
| `SAGE_VOICE_TURN` | `silence` | `silence`: a pause ends your turn. `smart`: Smart Turn v3 decides, and the STT hears the turn whole |
| `SAGE_VOICE_TURN_END_MS` | `450` | Silence that ends a turn; calibrate with `uv run sage-voice-turn-end turn-*.wav` |
| `SAGE_VOICE_TURN_TIMEOUT_MS` | `1500` | `smart` only: silence that ends a turn Smart Turn calls incomplete |
| `SAGE_VOICE_MIC_CHANNEL` | `mean` | `mean`, `0` or `1` of the mic array's stereo |
| `SAGE_VOICE_PRESENCE` | `0` | `1`: report seen/lost from the face tracker (needs a kernel that accepts presence events) |
| `SAGE_VOICE_LISTEN_CUES` | `0` | `1`: hold still while you speak, glance away while thinking |
| `SAGE_VOICE_ADDRESSED` | `0` | `1`: answer only when addressed: a face is tracked, its name is said, or it spoke in the last few seconds. Other speech (lyrics, TV) is not sent |
| `SAGE_VOICE_ALIASES` | `Reachy` | Names besides Sage it answers to, comma-separated |
| `SAGE_VOICE_EXCHANGE_S` | `8` | Seconds after it spoke during which no name or face is needed |
| `SAGE_VOICE_RECORD_DIR` | off | A local folder: save each utterance as a WAV |

## Privacy

- Speech to text, speech and the kernel's model all run on your machine. The
  services bind to 127.0.0.1, run in sealed containers and run offline once built.
- The app sends audio only to the STT URL and text only to the kernel and TTS
  URLs you set.
- Recordings are off by default. `SAGE_VOICE_RECORD_DIR` saves your voice to
  that folder: keep it local and never commit it.
- The robot's recorded moves come from Pollen's public dataset, read from the
  local Hugging Face cache (downloaded once if it is missing).

## With the simulator

The simulator ([`apps/reachy_sim`](https://github.com/1xabhay/reachy-mini-lab/tree/main/apps/reachy_sim)
in reachy-mini-lab, beside this app) has no mic or speaker, so it
cannot run the voice. It runs the rest without a robot: the kernel's replies and
the recorded moves as emotes, over the same device protocol.

```sh
make -C ../reachy_sim sim     # Pollen's MuJoCo daemon on 127.0.0.1:8742, headless
cd ../reachy_sim && uv run python -c 'import device, os; print(device.turn(
    "http://127.0.0.1:8776", os.environ["SAGE_VOICE_KERNEL_TOKEN"],
    "http://127.0.0.1:8742/api", "Feel happy", "sim-1"))'
```

## Development

```sh
make test                # the app's tests: no robot, no network, no GPU
make -C services test    # the STT and TTS contract tests (loads the real models)
```

## Known limits

- Dashboard install fails today. The daemon installs an app with
  `uv pip install` into its shared apps environment, and `pipecat-ai` (which
  pins `onnxruntime~=1.24.3`) cannot resolve beside `reachy-mini` (which pins
  `onnxruntime==1.27.0`). This project's `uv sync` overrides that pin
  (`[tool.uv]` in `pyproject.toml`); `uv pip install` does not read it.
- The voice needs the robot's audio: the simulator cannot stand in for it.

## Licence

Apache-2.0 ([LICENSE](LICENSE)). Third-party models and software keep their own
licences: see [NOTICE](NOTICE).
