"""The layers: vision, hearing, transcription, mind, voice.

Each is a specialist that does one thing and announces it. None of them can
name another - they meet only on the bus - so any one runs alone and any
combination runs together, which is the whole point. Vision tracking you with
nothing else switched on is a complete, useful animal; add hearing and it can
be spoken to; add a mind and it answers.

Every module does its work in `step()`, synchronously. `start()` only wires up
subscriptions, and the thread that a live pet gives each module does nothing
but call `step()` in a loop. All the behaviour is therefore testable with no
threads, no sleeps and no timing.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

from .bus import Bus, Subscription
from .conductor import Conductor, Transcribe
from .events import (
    Emote,
    FaceLost,
    FaceSeen,
    Heard,
    Reply,
    Spoke,
    Utterance,
    VoiceStarted,
    VoiceStopped,
)
from .following import BodyYawFollower
from .streaming import FaceMemory
from .transcripts import is_trustworthy

logger = logging.getLogger("reachy_mini.pet.layers")

#: Conversation kept in the mind, in messages.
MEMORY_TURNS = 40


def remember_recently(history: list[dict]) -> list[dict]:
    """Keep the recent past, never opening on the assistant.

    A conversation whose first turn is the assistant's is rejected outright by
    some model APIs, and trimming to a fixed length can easily land in the
    middle of an exchange - so the dangling half is dropped.
    """
    recent = history[-MEMORY_TURNS:]
    if recent and recent[0].get("role") != "user":
        recent = recent[1:]
    return recent

#: Failures that mean the robot itself has gone, rather than that one piece of
#: work failed. These must NOT be swallowed - every recovery mechanism in this
#: package is downstream of `run` seeing them, and for a long time none of it
#: could fire because each layer caught its own exceptions and logged them.
#:
#: Only layers that talk to the robot re-raise these. The mind and the
#: transcriber talk to models on this machine; an outage there is their own
#: business and must not be mistaken for the robot dying.
ROBOT_GONE = (ConnectionError, OSError)


class Module:
    """One specialist. Subscribes on `start`, does its work in `step`."""

    name = "module"

    #: Shortest sensible gap between steps, in seconds. Zero means "every time
    #: round the loop" - right for hearing, which must not miss a frame, and
    #: wrong for everything that sends commands to the robot.
    period = 0.0

    def __init__(self) -> None:
        """Create the module unattached to any bus."""
        self.bus: Bus | None = None

    def start(self, bus: Bus) -> None:
        """Attach to a bus and take out any subscriptions."""
        self.bus = bus
        self.subscribe(bus)

    def subscribe(self, bus: Bus) -> None:
        """Take out subscriptions. Senses that only publish need none."""

    def step(self, now: float = 0.0) -> None:
        """Do one unit of work."""

    def stop(self) -> None:
        """Release anything held."""

    def _publish(self, event) -> None:
        """Announce something, if anyone is listening."""
        if self.bus is not None:
            self.bus.publish(event)


class Vision(Module):
    """Watches. Reports who is there, and turns the body to keep them in frame.

    Useful entirely on its own: with nothing else running, this is a robot that
    follows you around the room.
    """

    name = "vision"
    #: 20 looks a second is far more than a head can usefully act on, and the
    #: body-yaw channel is the one whose death this pet has already suffered.
    period = 0.05

    #: Don't re-send an aim that differs from the last one by less than this
    #: (radians). Below about a degree the robot is already there.
    aim_epsilon = 0.017

    def __init__(self, eyes, follower: BodyYawFollower | None = None) -> None:
        """Take something that can `look()` and `turn_body()`."""
        super().__init__()
        self.eyes = eyes
        self.memory = FaceMemory()
        self.follower = follower if follower is not None else BodyYawFollower()
        self._present = False
        self._last_aim: float | None = None

    def step(self, now: float = 0.0) -> None:
        """Look once, and act on what is there."""
        try:
            detected, x, y = self.eyes.look()
        except ROBOT_GONE:
            raise  # the robot is gone; the session must end, not limp on
        except Exception:
            logger.exception("vision: could not look")
            # Still no sighting - say so, or downstream believes you are
            # standing there for ever.
            if self._present and not self.memory.is_present(now):
                self._present = False
                self._publish(FaceLost(at=now))
            return

        self.memory.update(detected, x, y, now)

        if detected and x is not None:
            self._present = True
            self._publish(FaceSeen(x=float(x), y=float(y or 0.0), at=now))
        elif self._present and not self.memory.is_present(now):
            self._present = False
            self._publish(FaceLost(at=now))

        yaw = self.follower.update(detected, x, now)
        if yaw is None:
            return
        if self._last_aim is not None and abs(yaw - self._last_aim) < self.aim_epsilon:
            return  # already pointing there; saying so again is pure noise
        try:
            self.eyes.turn_body(yaw)
            self._last_aim = yaw
        except ROBOT_GONE:
            raise
        except Exception:
            logger.exception("vision: could not turn the body")


class Hearing(Module):
    """Listens. Reports complete utterances, and nothing else.

    Alone, this is a robot that knows when it is being spoken to - which is
    enough to wake up, or look round, without understanding a word.
    """

    name = "hearing"
    #: Every frame matters: a missed one is audio that cannot be recovered.
    period = 0.0

    def __init__(
        self,
        ears,
        is_speech: Callable[[object], bool],
        conductor: Conductor | None = None,
        reset_detector: Callable[[], None] | None = None,
    ) -> None:
        """Take something that can `listen()`, and a way to judge a frame.

        `reset_detector` is called when an utterance closes. Silero is
        recurrent - its state carries the sense of an utterance in progress -
        so without this the previous utterance colours the next one.
        """
        super().__init__()
        self.ears = ears
        self.is_speech = is_speech
        self.reset_detector = reset_detector
        self.conductor = conductor if conductor is not None else Conductor()
        self._buffer: list[np.ndarray] = []
        self._clock = 0.0
        self._voice: Subscription | None = None

    def subscribe(self, bus: Bus) -> None:
        """Listen for the pet's own voice, so it does not answer itself."""
        self._voice = bus.subscribe((VoiceStarted, VoiceStopped))

    def _follow_the_voice(self) -> None:
        """Tell the conductor when the pet's own voice is in the room.

        Both use the audio clock, deliberately: the conductor measures
        everything in samples heard, and mixing in wall time would make the
        tail gate compare two unrelated origins and never expire.
        """
        if self.bus is None or self._voice is None:
            return
        for event in self.bus.drain(self._voice):
            if isinstance(event, VoiceStarted):
                self.conductor.on_playback_started(turn=self.conductor.turn, now=self._clock)
            else:
                self.conductor.on_playback_finished(turn=self.conductor.turn, now=self._clock)

    def step(self, now: float = 0.0) -> None:
        """Take one frame from the microphone and decide what it means."""
        self._follow_the_voice()
        frame = self.ears.listen()
        if frame is None:
            return

        self._clock += self.conductor.frame_s
        try:
            voiced = bool(self.is_speech(frame))
        except Exception:
            logger.exception("hearing: detector failed")
            return

        self._buffer.append(frame)
        # Never hold more than the conductor would ever ask for.
        if len(self._buffer) > self.conductor.max_speech_frames + 1:
            self._buffer.pop(0)

        for action in self.conductor.feed_audio(voiced=voiced, now=self._clock):
            if isinstance(action, Transcribe):
                audio = self._collect(action.frames)
                self._publish(
                    Utterance(
                        audio=audio,
                        samplerate=getattr(self.ears, "samplerate", 16000),
                        at=self._clock,
                        voiced_fraction=(
                            action.voiced_frames / action.frames if action.frames else 0.0
                        ),
                    )
                )
            if self.reset_detector is not None:
                try:
                    self.reset_detector()
                except Exception:
                    logger.exception("hearing: could not reset the detector")

        if not self.conductor.pending_frames:
            self._buffer.clear()

    def _collect(self, frames: int) -> np.ndarray:
        """Gather the audio the conductor says made up the utterance."""
        taken = self._buffer[-frames:] if frames else []
        usable = [
            np.asarray(f, dtype=np.float32).reshape(-1) for f in taken if isinstance(f, np.ndarray)
        ]
        return np.concatenate(usable) if usable else np.zeros(0, dtype=np.float32)


class Transcriber(Module):
    """Turns an utterance into words. Knows nothing about conversation."""

    name = "transcriber"
    period = 0.05

    def __init__(self, transcribe: Callable[[np.ndarray, int], str]) -> None:
        """Take a function from audio to text."""
        super().__init__()
        self.transcribe = transcribe
        self._inbox: Subscription | None = None

    def subscribe(self, bus: Bus) -> None:
        """Listen for finished utterances."""
        self._inbox = bus.subscribe(Utterance)

    def step(self, now: float = 0.0) -> None:
        """Transcribe whatever has arrived."""
        if self.bus is None or self._inbox is None:
            return
        for utterance in self.bus.drain(self._inbox):
            try:
                text = self.transcribe(utterance.audio, utterance.samplerate).strip()
            except Exception:
                logger.exception("transcriber: could not transcribe")
                continue
            if not text:
                continue
            if not is_trustworthy(text, utterance.voiced_fraction):
                # Whisper fabricates from non-speech, confidently. Answering
                # something that was never said is worse than saying nothing.
                logger.info("ignored (not trustworthy): %s", text)
                continue
            logger.info("heard: %s", text)
            self._publish(Heard(text=text, at=utterance.at))


class Mind(Module):
    """Decides what to say, and how to hold itself while saying it.

    The words and the body language come out of the same thought, which is why
    it publishes both.
    """

    name = "mind"
    period = 0.05

    def __init__(
        self,
        think: Callable[[str, list], tuple[list[str], str]],
        memory=None,
    ) -> None:
        """Take a function from what was heard to what to say.

        `memory` is anything with `load()` and `save(history)`. It is injected
        because the layer's business is thinking, not filing: it knows there is
        somewhere to write the conversation down, not where that is.
        """
        super().__init__()
        self.think = think
        self.memory = memory
        self.history: list[dict] = []
        if memory is not None:
            try:
                self.history = remember_recently(list(memory.load()))
            except Exception:
                logger.exception("mind: could not read the conversation back")
        self._inbox: Subscription | None = None
        self._turn = 0

    def subscribe(self, bus: Bus) -> None:
        """Listen for words."""
        self._inbox = bus.subscribe(Heard)

    def step(self, now: float = 0.0) -> None:
        """Answer whatever was said."""
        if self.bus is None or self._inbox is None:
            return
        for heard in self.bus.drain(self._inbox):
            try:
                clauses, emotion = self.think(heard.text, self.history)
            except Exception:
                logger.exception("mind: could not think")
                continue
            self._turn += 1
            self.history.append({"role": "user", "content": heard.text})
            self.history.append({"role": "assistant", "content": " ".join(clauses)})
            self.history = remember_recently(self.history)
            if self.memory is not None:
                try:
                    self.memory.save(self.history)
                except Exception:
                    # A full disk must not make the pet mute.
                    logger.exception("mind: could not write the conversation down")
            logger.info("reply: %s%s", " ".join(clauses), f" [{emotion}]" if emotion else "")
            if emotion:
                self._publish(Emote(name=emotion))
            self._publish(Reply(clauses=list(clauses), emotion=emotion, turn=self._turn))


class Voice(Module):
    """Speaks. Knows nothing about why."""

    name = "voice"
    period = 0.02

    def __init__(self, mouth) -> None:
        """Take something that can `say()`."""
        super().__init__()
        self.mouth = mouth
        self._inbox: Subscription | None = None

    def subscribe(self, bus: Bus) -> None:
        """Listen for replies."""
        self._inbox = bus.subscribe(Reply)

    def step(self, now: float = 0.0) -> None:
        """Say whatever has been decided."""
        if self.bus is None or self._inbox is None:
            return
        for reply in self.bus.drain(self._inbox):
            if reply.clauses:
                self._publish(VoiceStarted(turn=reply.turn))
            for clause in reply.clauses:
                try:
                    self.mouth.say(clause)
                except ROBOT_GONE:
                    raise
                except Exception:
                    logger.exception("voice: could not speak")
                    continue
                logger.info("said: %s", clause)
                self._publish(Spoke(text=clause))
            if reply.clauses:
                self._publish(VoiceStopped(turn=reply.turn))
