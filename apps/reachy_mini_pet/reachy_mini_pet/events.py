"""The vocabulary the layers share.

This is the whole contract between vision, hearing, voice and mind. Each layer
knows these names and nothing else about the others, which is what lets any of
them run alone or in any combination.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from .bus import Event


@dataclass
class FaceSeen(Event):
    """Someone is in view, at this position in the frame."""

    x: float
    y: float
    at: float = 0.0


@dataclass
class FaceLost(Event):
    """Nobody in view, and the memory of where they were has run out."""

    at: float = 0.0


@dataclass
class Utterance(Event):
    """A complete stretch of speech, ready to be turned into words."""

    audio: npt.NDArray[np.float32]
    samplerate: int
    at: float = 0.0
    #: How much of the clip the detector called speech. A clip that was almost
    #: all silence produces confident nonsense from Whisper, so whoever
    #: transcribes it needs to know.
    voiced_fraction: float = 1.0


@dataclass
class Heard(Event):
    """What the utterance turned out to say."""

    text: str
    at: float = 0.0


@dataclass
class Reply(Event):
    """What to say back, and how to hold yourself while saying it."""

    clauses: list[str] = field(default_factory=list)
    emotion: str = ""
    turn: int = 0


@dataclass
class Emote(Event):
    """Body language, by name, from the emotions library."""

    name: str


@dataclass
class Spoke(Event):
    """One clause has actually reached the speaker."""

    text: str


@dataclass
class VoiceStarted(Event):
    """Its own voice is now in the room.

    Hearing needs this: while the pet is speaking, the microphone is picking up
    the pet. Without it the conductor's whole speaking half - tail gate,
    ducking, interruption - is unreachable, and the only defence against the
    pet answering itself is the hardware echo canceller.
    """

    turn: int = 0


@dataclass
class VoiceStopped(Event):
    """It has stopped speaking, but the tail of it is still in the room."""

    turn: int = 0
