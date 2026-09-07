"""Realtime primitives, so the pet stops being turn-based.

The original loop was a sequence of blocking phases: hear a whole sentence,
transcribe it, wait for a whole reply, then speak it while deaf. Every stage
idled the ones either side of it, and the robot was inert between them.

These are the pure pieces that make the pipeline continuous. They hold no
threads and no sockets so they can be tested exhaustively; the wiring that
uses them lives in `pipeline.py`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from itertools import chain

#: A clause ends at punctuation followed by space, or at a newline. There is
#: deliberately no "end of buffer" case: mid-stream the buffer ends in the
#: middle of a word, and treating that as a sentence end cuts replies short.
_ENDS = re.compile(r"[.!?][\"')\]]*\s|\n")

#: Words that take a full stop without ending a sentence. A pause after "Mr."
#: is audible and wrong.
_ABBREVIATIONS = frozenset(
    [
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "st",
        "sr",
        "jr",
        "vs",
        "etc",
        "eg",
        "ie",
        "e.g",
        "i.e",
        "fig",
        "approx",
        "dept",
        "inc",
        "ltd",
    ]
)

#: Trailing characters that belong to the sentence they follow.
_TRAILING = "\"')]} \t\n"

#: Say something rather than hold a long clause waiting for punctuation the
#: model may never produce.
MAX_CLAUSE_CHARS = 180


def _ends_sentence(clause: str) -> bool:
    """Whether a full stop here really ends a sentence, or just an abbreviation."""
    text = clause.rstrip(_TRAILING)
    if not text.endswith("."):
        return True  # "!" and "?" are unambiguous
    word = text[:-1].rsplit(" ", 1)[-1].strip("\"'([{")
    if len(word) == 1 and word.isalpha():
        return False  # an initial, as in "A. J. Smith"
    return word.lower() not in _ABBREVIATIONS


def _cut(buffer: str, max_chars: int) -> int | None:
    """Where to end the next spoken clause, or `None` to keep buffering."""
    for match in _ENDS.finditer(buffer):
        if _ends_sentence(buffer[: match.end()]):
            return match.end()
    if len(buffer) >= max_chars:
        # No punctuation in sight - break at a word boundary so the voice does
        # not stop mid-word. A single word longer than the cap has no boundary
        # to use, so it is held and flushed at the end.
        space = buffer.rfind(" ")
        if space > 0:
            return space
    return None


def sentences(tokens: Iterable[str], max_chars: int = MAX_CLAUSE_CHARS) -> Iterator[str]:
    """Turn a stream of model tokens into speakable clauses, as they complete.

    This is what removes the dead air: the first clause goes to the voice while
    the model is still writing the second. Whatever is left when the stream
    ends is flushed, so no words are dropped.
    """
    buffer = ""
    for token in tokens:
        buffer += token
        while (cut := _cut(buffer, max_chars)) is not None:
            head, buffer = buffer[:cut].strip(), buffer[cut:]
            if head:
                yield head
    tail = buffer.strip()
    if tail:
        yield tail


@dataclass
class FaceMemory:
    """Where you were last seen, and how much that is still worth.

    A detector that reports nothing is not the same as an empty room - you
    lean out of frame, the exposure shifts, someone walks past. Snapping the
    head back to neutral on every dropped frame is what makes a robot look
    broken. So the last known location is held, then given up slowly.
    """

    #: Keep aiming exactly where you were last seen for this long.
    hold_s: float = 2.0
    #: Then fade back towards neutral over this long, rather than snapping.
    fade_s: float = 6.0

    x: float = 0.0
    y: float = 0.0
    last_seen: float | None = field(default=None)

    def update(self, detected: bool, x: float | None, y: float | None, now: float) -> None:
        """Fold in one observation from the tracker."""
        if detected and x is not None and y is not None:
            self.x, self.y, self.last_seen = float(x), float(y), now

    def confidence(self, now: float) -> float:
        """How much the remembered location is still worth, from 1.0 to 0.0."""
        if self.last_seen is None:
            return 0.0
        age = now - self.last_seen
        if age <= self.hold_s:
            return 1.0
        if age >= self.hold_s + self.fade_s:
            return 0.0
        return 1.0 - (age - self.hold_s) / self.fade_s

    def aim(self, now: float) -> tuple[float, float] | None:
        """Where to point the head now, or `None` once the memory has faded.

        The remembered position is scaled by how stale it is, so the head
        drifts back to centre instead of either freezing or snapping.
        """
        weight = self.confidence(now)
        if weight <= 0.0:
            return None
        return self.x * weight, self.y * weight

    def is_present(self, now: float) -> bool:
        """Whether someone should still be treated as being in the room."""
        return self.confidence(now) > 0.0


#: How far to read looking for an emotion line before concluding there is not
#: one. Long enough for any move name, short enough not to swallow a sentence.
_EMOTION_LOOKAHEAD = 60

_EMOTION_LINE = re.compile(r"^\s*(emotion\s*:)?\s*([a-z0-9_\-]+)\s*$", re.IGNORECASE)


def take_emotion(tokens: Iterable[str], allowed: Iterable[str]) -> tuple[str, Iterator[str]]:
    """Split a leading emotion line off a token stream.

    The model is asked to name how it feels on the first line and then speak.
    That keeps one request and keeps the speech streaming - asking for JSON
    instead would mean parsing the whole reply before a word could be said -
    and it makes the body move before the voice starts, which is the order a
    living thing does it in.

    Returns the emotion (empty if there was not a recognisable one) and the
    rest of the stream, with nothing dropped either way.
    """
    permitted = set(allowed)
    stream = iter(tokens)
    buffer = ""

    for token in stream:
        buffer += token
        if "\n" in buffer:
            head, _, rest = buffer.partition("\n")
            match = _EMOTION_LINE.match(head)
            if match:
                declared, name = match.group(1), match.group(2).lower()
                if name in permitted:
                    return name, chain([rest], stream)
                if declared:
                    # It meant to name a feeling and got the name wrong. Drop
                    # the line - saying "EMOTION: jubilant7" out loud is worse
                    # than losing it.
                    return "", chain([rest], stream)
            # Not an emotion line after all - hand every word back.
            return "", chain([buffer], stream)
        if len(buffer) >= _EMOTION_LOOKAHEAD:
            return "", chain([buffer], stream)

    return "", iter([buffer])


#: Roughly how much speech the pet may commit to in one reply, in characters.
#: It cannot be interrupted while talking, so this is a budget on how long the
#: room has to wait, not a style preference.
SPOKEN_BUDGET = 320


def clamp_clauses(clauses: Iterable[str], budget: int = SPOKEN_BUDGET) -> list[str]:
    """Keep whole clauses up to a spoken-length budget.

    Whole clauses, because a sentence cut in half sounds broken - and always at
    least the first one, because saying nothing is worse than going over.
    """
    kept: list[str] = []
    spent = 0
    for clause in clauses:
        if kept and spent + len(clause) > budget:
            break
        kept.append(clause)
        spent += len(clause)
    return kept
