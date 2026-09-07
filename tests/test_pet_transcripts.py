"""Checks for refusing transcripts the microphone never actually heard.

Whisper invents text when handed non-speech, and it does so confidently, which
is why its own `no_speech_prob` and `log_prob` thresholds do not catch it -
they only trigger a re-decode at a higher temperature and then return the
least-bad attempt anyway. These are the real examples out of the pet's log:

    "You can't see it, can't see it, can't see it, can't see it. It's a good thing."
    "and our phenomenal numbers are all up to this year."
    "I've got total air"

Nobody said any of them, and the pet answered all three. Replying to things
that were never said is far more corrosive to the illusion than missing a
phrase, so the bar here is deliberately set to prefer silence.
"""

import pytest
from reachy_mini_pet.transcripts import is_trustworthy, repetition_score

# ------------------------------------------------------------ real speech


@pytest.mark.parametrize(
    "text",
    [
        "good morning",
        "what's the weather like today",
        "yes",
        "no",
        "put the kettle on please",
        "I had a really long day and I just want to sit down",
        "no, no, that's not what I meant",       # emphatic repetition is real
    ],
)
def test_real_speech_is_trusted(text):
    assert is_trustworthy(text) is True


# --------------------------------------------------------- the phantoms


def test_the_actual_logged_hallucination_is_refused():
    """The one that made the pet answer a sentence nobody said."""
    text = (
        "You can't see it, can't see it, can't see it, can't see it, "
        "can't see it. It's a good thing."
    )
    assert is_trustworthy(text) is False


@pytest.mark.parametrize(
    "text",
    [
        "can't see it can't see it can't see it can't see it",
        "the the the the the the",
        "and then and then and then and then",
        "yeah yeah yeah yeah yeah yeah yeah",
    ],
)
def test_a_stuck_decoder_is_refused(text):
    assert is_trustworthy(text) is False


@pytest.mark.parametrize(
    "text",
    [
        "Thank you for watching!",
        "thanks for watching",
        "Subtitles by the Amara.org community",
        "Please subscribe to my channel",
        "[BLANK_AUDIO]",
        "(applause)",
        "bell rings",
    ],
)
def test_whispers_stock_phantom_phrases_are_refused(text):
    """These come from its training data, not from the room."""
    assert is_trustworthy(text) is False


@pytest.mark.parametrize("text", ["", "   ", "\n", ".", "...", "-"])
def test_nothing_at_all_is_refused(text):
    assert is_trustworthy(text) is False


@pytest.mark.parametrize("text", ["bye", "Bye.", "thank you", "you", "goodbye"])
def test_things_a_person_really_says_to_a_pet_are_kept(text):
    """These are common Whisper phantoms *and* real speech to a desk robot.

    Refusing a genuine goodbye is worse than occasionally answering an
    imagined one, so they are deliberately not on the blocklist.
    """
    assert is_trustworthy(text) is True


def test_a_phantom_phrase_inside_real_speech_is_kept():
    """Substring matching would eat legitimate sentences."""
    assert is_trustworthy("thanks for watching the dog while I was out") is True


# ------------------------------------------------------- how it measures


def test_repetition_is_scored_not_guessed():
    assert repetition_score("hello there friend") == 1
    assert repetition_score("hello hello hello") == 3
    assert repetition_score("can't see it can't see it can't see it") == 3


def test_an_empty_string_scores_nothing():
    assert repetition_score("") == 0


# ---------------------------------------------- how much of it was speech


def test_a_clip_that_was_mostly_silence_is_refused():
    """Whisper hallucinates about half the time on a second of near-silence."""
    assert is_trustworthy("hello there", voiced_fraction=0.05) is False


def test_a_clip_that_was_mostly_speech_is_trusted():
    assert is_trustworthy("hello there", voiced_fraction=0.8) is True


def test_the_fraction_is_optional():
    """Callers that do not know it should not be penalised."""
    assert is_trustworthy("hello there") is True
