"""Refusing transcripts the microphone never actually heard.

Whisper invents text when handed non-speech, and it does so *confidently* -
which is why its own `no_speech_prob` and `log_prob` thresholds do not save
you. Those only mark a decode as suspect and re-run it at a higher
temperature; if every attempt looks bad the loop returns the least-bad one
regardless. A fluent, high-confidence fabrication sails straight through.

These are real lines out of the pet's log, none of which were said by anyone:

    "You can't see it, can't see it, can't see it, can't see it. It's a good thing."
    "and our phenomenal numbers are all up to this year."
    "I've got total air"

The pet answered all three. Replying to things that were never said damages
the illusion far more than missing a phrase does, so this prefers silence when
in doubt.
"""

from __future__ import annotations

import re

#: Phrases Whisper produces from its training data rather than from the room -
#: video outros, caption credits, sound-effect labels. Matched against the whole
#: transcript only: as a substring test this would eat real sentences like
#: "thanks for watching the dog".
#: Deliberately excludes phrases a person might really say to a desk pet -
#: "bye", "thank you", "you" are all common Whisper phantoms, but they are also
#: things you say to a pet, and refusing a genuine goodbye is worse than
#: occasionally answering one that was imagined.
PHANTOM_PHRASES = frozenset(
    {
        "thanks for watching",
        "thank you for watching",
        "thank you for watching please subscribe",
        "please subscribe",
        "please subscribe to my channel",
        "subscribe to my channel",
        "like and subscribe",
        "subtitles by the amara.org community",
        "subtitles by the amara org community",
        "transcription by castingwords",
        # Normalisation strips punctuation, so "[BLANK_AUDIO]" arrives as two words.
        "blank audio",
        "applause",
        "music",
        "bell rings",
        "birds chirping",
    }
)

#: A phrase repeated this many times is a stuck decoder, not a person. Set
#: above the two or three repeats that real emphatic speech uses ("no, no,
#: that's not what I meant").
MAX_REPEATS = 4

#: Below this share of voiced audio, the clip was mostly silence - the regime
#: where Whisper hallucinates about half the time - so whatever it produced is
#: not worth trusting.
MIN_VOICED_FRACTION = 0.15

_WORD = re.compile(r"[a-z0-9']+")


def _words(text: str) -> list[str]:
    """Reduce a transcript to bare lowercase words."""
    return _WORD.findall(text.lower())


def repetition_score(text: str) -> int:
    """Return how many times the most-repeated short phrase appears.

    Looks at one-, two- and three-word phrases, because a stuck decoder loops
    on all three ("the the the", "and then and then", "can't see it can't see
    it"). This is independent of the model's own confidence, which is the whole
    point: it catches the fabrications that report themselves as certain.
    """
    words = _words(text)
    if not words:
        return 0

    best = 1
    for size in (1, 2, 3):
        if len(words) < size * 2:
            continue
        counts: dict[tuple[str, ...], int] = {}
        for start in range(len(words) - size + 1):
            phrase = tuple(words[start : start + size])
            counts[phrase] = counts.get(phrase, 0) + 1
        best = max(best, max(counts.values()))
    return best


def is_trustworthy(text: str, voiced_fraction: float | None = None) -> bool:
    """Whether this transcript is worth acting on.

    `voiced_fraction` is how much of the clip the detector called speech, when
    the caller knows it. A clip that was almost all silence produced whatever
    this says out of nothing.
    """
    words = _words(text)
    if not words:
        return False

    if voiced_fraction is not None and voiced_fraction < MIN_VOICED_FRACTION:
        return False

    if " ".join(words) in PHANTOM_PHRASES:
        return False

    return repetition_score(text) < MAX_REPEATS
