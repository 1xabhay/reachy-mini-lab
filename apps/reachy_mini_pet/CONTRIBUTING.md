# Contributing

The short version: **add a layer, don't grow one.** If a change makes an
existing layer know about something new, it probably wants to be its own layer.

Everything runs without a robot, without a network and without an API key:

```sh
pip install -e ".[dev]"
pytest
```

## The shape of the thing

Six layers meet on a bus and know nothing about each other. That is the whole
design, and it is what makes a change small:

```
Vision ──FaceSeen/FaceLost──┐
Hearing ────Utterance───────┤
                            ├──► bus ──►  whoever subscribed
Transcriber ────Heard───────┤
Mind ───────Reply/Emote─────┤
Voice ───Spoke/VoiceStarted─┘
```

Two rules hold it together:

1. **A layer never imports another layer.** It publishes what it observed and
   subscribes to what concerns it. If you need a reference to another layer,
   add an event instead.
2. **Decisions don't live in threads.** Every module does its work in a
   synchronous `step()`; `live.run` gives each one a thread that only calls
   `step()` in a loop. Anything that decides *when* something happens belongs
   in `conductor.py`, which takes `now` as an argument rather than reading a
   clock.

That second rule is why barge-in — normally a nest of timing races — is
asserted at an exact frame index, with no threads and no sleeps.

## Adding a sense or a behaviour

A layer is about thirty lines. Give it a `name`, a `period` (how often it wants
stepping), subscribe to what it needs, and do the work in `step`:

```python
class Nose(Module):
    """Smells things. Reports what it smelled."""

    name = "nose"
    period = 0.5          # twice a second is plenty for a nose

    def __init__(self, sensor):
        super().__init__()
        self.sensor = sensor          # inject the hardware, don't import it

    def subscribe(self, bus):
        pass                          # a pure sense needs no subscriptions

    def step(self, now=0.0):
        smell = self.sensor.sniff()
        if smell:
            self._publish(Smelled(what=smell, at=now))
```

Then add it to `build()` in `live.py`. Nothing else changes, and nothing else
needs to know it exists.

Testing it needs no hardware, because the sensor is injected:

```python
def test_the_nose_reports_what_it_smelled():
    bus = Bus()
    inbox = bus.subscribe(Smelled)
    nose = Nose(FakeNose(["toast"]))
    nose.start(bus)

    nose.step(now=1.0)

    assert [e.what for e in bus.drain(inbox)] == ["toast"]
```

## Swapping a model

The layers don't know that hearing is Silero or that the voice is Piper. Those
are adapters in `live.py`, and each is a small class or closure:

| To replace | Write | Shape |
| --- | --- | --- |
| Speech to text | a function | `(audio, samplerate) -> str` |
| The mind | a function | `(text, history) -> (clauses, emotion)` |
| The voice | an object | `.say(text)` |
| The eyes | an object | `.look() -> (detected, x, y)`, `.turn_body(yaw)` |
| The ears | an object | `.listen() -> 512-sample window or None` |
| The memory | an object | `.load()`, `.save(history)` |

No layer changes. Point `build()` at yours.

## Rules that came from being bitten

Each of these is in the code because something failed on a real robot at two in
the morning. Please keep them.

- **Bound everything.** Every queue has a `maxsize`, every buffer a ceiling,
  every `while True` a deadline. This runs for weeks.
- **Take `now` as an argument.** Never call `time.monotonic()` in decision
  logic. A week of outage is then tested in microseconds.
- **Let robot failures out.** `ConnectionError` and `OSError` from a layer that
  talks to the robot must propagate — see `ROBOT_GONE` in `layers.py`. Every
  recovery mechanism is downstream of `run()` seeing them, and for a while none
  of it could fire because each layer caught its own exceptions.
- **Don't swallow your own model's failures as the robot's.** `Mind` and
  `Transcriber` talk to local models; their outages are their own business.
- **Leave the calibration knob.** A real room, a real microphone and a real
  motor are never the ideal one. `deadzone`, `direction`, `max_rate`,
  `VAD_CEILING` all exist because a value that was right on a desk was wrong in
  a kitchen. Prefer an environment variable over a better constant.
- **Pin a limitation with a test.** Where the pet knowingly cannot cope — a
  DC-biased microphone, an echo arriving after the drain window — there is a
  test asserting the current behaviour and saying so in its docstring. Those
  tests are *meant* to fail when someone fixes the gap; rewrite them then.

## Tests

- `test_pet_conductor.py` — every listening/speaking decision, at exact frames
- `test_pet_layers.py` — each layer alone, then combined
- `test_pet_bus.py` — including two threads, because the module is about threads
- `test_pet_vad.py`, `test_pet_streaming.py`, `test_pet_transcripts.py` — the
  pure pieces, with Hypothesis properties where an invariant is worth stating
- `test_pet_recovery.py`, `test_pet_health.py`, `test_pet_supervisor.py`,
  `test_pet_watchdog.py` — hardware that dies underneath the app
- `test_pet_contract.py` — that the test doubles still match the real SDK, so an
  upgrade cannot leave the suite green and the pet broken

Two habits worth copying. **Use the real class in the test that proves a
mechanism** — a health test using a fake that raises where no real layer does
passed for two hours while the mechanism was inert. And **check your fix by
breaking it on purpose**: mutate the line and confirm a test goes red.

## Publishing

```sh
reachy-mini-app-assistant check .
reachy-mini-app-assistant publish
```

`check` installs the app in a throwaway virtualenv and confirms the entry point
registers, so it catches a dependency you forgot to declare. Run it before
publishing — that failure is otherwise invisible until someone else installs it.
