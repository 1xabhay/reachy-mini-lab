---
title: Reachy Mini Pet
emoji: 🐣
colorFrom: indigo
colorTo: pink
sdk: static
pinned: false
short_description: An always-on desk pet that watches, listens and answers - all on your own machine
tags:
  - reachy_mini
  - reachy_mini_python_app
---

# Reachy Mini Pet

An always-on desk pet. It follows you around the room, listens, answers out
loud, and reacts with its body. **Nothing leaves your machine** — the voice
activity detector, the speech recognition, the language model and the voice all
run locally, and the app needs no API keys.

## Install

```sh
pip install -e .
```

Then three model files, none of which the app can fetch for you silently:

```sh
# 1. Voice activity detection (~2 MB)
mkdir -p ~/.reachy_pet
curl -Lo ~/.reachy_pet/silero_vad.onnx \
  https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx

# 2. A voice (~63 MB). Any piper voice works; this is the default.
python -m piper.download_voices en_GB-jenny_dioco-medium \
  --download-dir ~/.reachy_pet/voices

# 3. A language model, served by Ollama (https://ollama.com)
ollama pull qwen3:1.7b
```

Speech-to-text downloads itself on first use.

## Run

From the robot's dashboard, or:

```sh
reachy-pet
```

Run from a terminal it supervises its own connection and survives the robot
going away and coming back. Launched by the daemon it runs a single session and
lets the daemon handle restarts.

## How it works

Six layers, each of which does one thing and announces it on a bus. None of
them can name another, so **any one runs alone and any combination runs
together**:

| Layer | Alone it is… | Publishes | Listens for |
| --- | --- | --- | --- |
| `Vision` | a robot that follows you round the room | `FaceSeen` `FaceLost` | — |
| `Hearing` | knows *when* it is spoken to, without understanding | `Utterance` | `VoiceStarted` `VoiceStopped` |
| `Transcriber` | sound → words | `Heard` | `Utterance` |
| `Mind` | words → a reply and a feeling | `Reply` `Emote` | `Heard` |
| `Voice` | speaks | `Spoke` `VoiceStarted` `VoiceStopped` | `Reply` |
| `Body` | body language, from 85 recorded moves | — | `Emote` |

The decisions — when to listen, when an utterance has ended, when you have
interrupted it — all live in `conductor.py`, a pure state machine that takes
`now` as an argument instead of reading a clock. Threads only ever call
`step()`. That is why the behaviour can be tested exhaustively without a robot,
a thread, or a sleep.

## The local stack

| Stage | What | Why |
| --- | --- | --- |
| Voice detection | [silero-vad](https://github.com/snakers4/silero-vad) (ONNX) | ~2 MB, no torch, tells a voice from a vacuum cleaner |
| Speech to text | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | ~10× realtime on CPU |
| Mind | [Ollama](https://ollama.com) | any local model, ~1.9 GB by default; asked for a feeling and a reply together |
| Voice | [piper](https://github.com/OHF-Voice/piper1-gpl) | ~270 ms to first audio, streams in chunks |

## Choosing a model for your hardware

```sh
reachy-pet-bench                        # everything Ollama has
reachy-pet-bench qwen3:1.7b gemma3:1b   # just these
```

It runs the pet's own prompts and grades what a pet actually needs: a short
spoken reply, a feeling that fairly matches what it heard, and an answer soon
enough that the pause is not awkward. The emotion is a label from a closed set,
so grading is exact and free - no judge model whose own reliability would then
need establishing.

Measured on an M2 Pro, which is why the default is what it is:

| model | apt emotion | valid name | median | resident |
| --- | --- | --- | --- | --- |
| `qwen3:1.7b` | 4/8 | **8/8** | **0.23 s** | **1.9 GB** |
| `qwen2.5:7b` | **5/8** | 6/8 | 0.69 s | 4.7 GB |
| `gemma3:1b` | 2/8 | 2/8 | 0.20 s | 0.9 GB |
| `qwen3:4b` | 0/8 | 0/8 | 3.73 s | 3.2 GB |

The 1.7B names a *valid* emotion more reliably than the 7B, three times faster
and in 40% of the memory. Whole pet: **75 MB of app plus the model**.

Reasoning models are asked not to reason (`think: false`): given the choice,
`qwen3:1.7b` spent its whole token budget inside `<think>` tags and returned
nothing at all. Any traces that do arrive are stripped before the voice sees
them.

## Configuration

Every knob is an environment variable, because rooms, voices and microphones
differ and no default survives contact with all of them.

| Variable | Default | Purpose |
| --- | --- | --- |
| `REACHY_MINI_HOST` | `localhost` | Daemon host |
| `REACHY_MINI_PORT` | `8000` | Daemon port |
| `REACHY_PET_DATA_DIR` | `~/.reachy_pet` | Models and the remembered conversation |
| `REACHY_PET_VOICE` | `en_GB-jenny_dioco-medium.onnx` | Piper voice file |
| `REACHY_PET_WHISPER` | `base.en` | Whisper size, or any faster-whisper model id |
| `REACHY_PET_LLM` | `qwen3:1.7b` | Ollama model - run `reachy-pet-bench` to pick for your hardware |
| `REACHY_PET_OLLAMA` | `http://localhost:11434/api/chat` | Ollama endpoint |

## What it does not do yet

- **No barge-in.** The plumbing exists — the conductor has ducking and
  interruption, and the voice announces itself — but `Voice.say()` still blocks
  until a clause finishes, so you cannot cut it off mid-sentence.
- **Whisper still fabricates occasionally.** Non-speech audio can produce
  fluent, confident sentences nobody said. Most are caught by requiring enough
  *voiced* audio and by rejecting repetition; a novel one-off can get through.
- **A DC-biased microphone jams the gate open.** The level test has no DC
  blocker, so a constant offset reads as continuous speech.
- **Session teardown does not release the models.** If the robot flaps
  repeatedly, memory grows.

These are recorded here rather than in a issue tracker because each one has a
test that pins the current behaviour, and those tests should start failing when
someone fixes them.

## Licence

Not yet chosen - add a `LICENSE` file and the matching `license` field in
`pyproject.toml` before publishing this anywhere public.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The short version: add a layer, don't
grow one. Tests run with no robot, no network and no API key.

```sh
pip install -e ".[dev]"
pytest
```
