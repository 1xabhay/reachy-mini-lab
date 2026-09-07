"""Checks for the realtime primitives.

These two are what turn the pet from a sequence of blocking phases into a
pipeline: `sentences` lets the voice start on the first clause while the model
is still writing, and `FaceMemory` keeps the head pointed where you were last
seen instead of snapping to neutral on every dropped detection.

Both are pure, so they can be pinned exhaustively here and the threaded wiring
that uses them has nothing left to prove but its plumbing.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from reachy_mini_pet.streaming import (
    FaceMemory,
    clamp_clauses,
    sentences,
    take_emotion,
)

ALLOWED = ("cheerful1", "sad1", "curious1", "welcoming1")


def drip(text: str, size: int = 3):
    """Deliver text the way a model does - a few characters at a time."""
    return [text[i : i + size] for i in range(0, len(text), size)]


# ------------------------------------------------------------ sentences


def test_a_finished_sentence_is_released_immediately():
    """The whole point: speech starts before generation ends."""
    assert list(sentences(drip("Hello there. "))) == ["Hello there."]


def test_each_sentence_comes_out_separately():
    out = list(sentences(drip("One. Two! Three?")))
    assert out == ["One.", "Two!", "Three?"]


def test_a_trailing_fragment_is_never_dropped():
    """The model stops without punctuation more often than you would think."""
    assert list(sentences(drip("All done. And then"))) == ["All done.", "And then"]


def test_an_empty_stream_yields_nothing():
    assert list(sentences([])) == []
    assert list(sentences(["   ", "\n"])) == []


def test_a_decimal_number_is_not_a_sentence_end():
    """'It is 3.5 degrees' must not be spoken as two sentences."""
    assert list(sentences(drip("It is 3.5 degrees outside. "))) == ["It is 3.5 degrees outside."]


def test_the_word_no_still_ends_a_sentence():
    """Check that "Oh no." is two sentences, not one.

    "No." is an abbreviation for "number", but in speech it is almost always
    the word - and treating it as an abbreviation swallowed the break.
    """
    assert list(sentences(drip("Oh no. That's awful. "))) == ["Oh no.", "That's awful."]


def test_an_abbreviation_is_not_a_sentence_end():
    """A pause after 'Mr.' is audible and wrong."""
    assert list(sentences(drip("Mr. Smith is here. "))) == ["Mr. Smith is here."]


def test_an_initial_is_not_a_sentence_end():
    assert list(sentences(drip("It was A. J. who called. ")))  == ["It was A. J. who called."]


def test_closing_quotes_stay_with_their_sentence():
    out = list(sentences(drip('He said "go." Then he left. ')))
    assert out == ['He said "go."', "Then he left."]


def test_a_newline_ends_a_clause():
    assert list(sentences(drip("First line\nSecond line"))) == ["First line", "Second line"]


def test_an_endless_clause_is_broken_so_the_voice_can_start():
    """A model that never punctuates must not mean silence forever."""
    rambling = "and then " * 60
    out = list(sentences(rambling))

    assert len(out) > 1
    assert all(len(s) <= 200 for s in out)


def test_a_single_enormous_word_cannot_wedge_it():
    """No spaces and no punctuation - the pathological input."""
    out = list(sentences(["x" * 5000]))

    assert "".join(out) == "x" * 5000


# ------------------------------------------------- sentences: properties


@given(text=st.text(min_size=0, max_size=400), size=st.integers(min_value=1, max_value=20))
@settings(deadline=None, max_examples=100)
def test_no_word_is_ever_lost_however_the_tokens_arrive(text, size):
    """The invariant that matters: chunking must not silently eat the reply."""
    out = list(sentences(drip(text, size)))

    assert " ".join(out).split() == text.split()


@given(text=st.text(min_size=0, max_size=400), size=st.integers(min_value=1, max_value=20))
@settings(deadline=None, max_examples=100)
def test_nothing_blank_is_ever_sent_to_the_voice(text, size):
    assert all(s.strip() for s in sentences(drip(text, size)))


# ----------------------------------------------------------- FaceMemory


def test_someone_never_seen_is_not_remembered():
    assert FaceMemory().aim(now=100.0) is None
    assert FaceMemory().is_present(now=100.0) is False


def test_a_fresh_sighting_is_aimed_at_exactly():
    memory = FaceMemory()
    memory.update(detected=True, x=0.4, y=-0.2, now=100.0)

    assert memory.aim(now=100.0) == pytest.approx((0.4, -0.2))
    assert memory.confidence(now=100.0) == 1.0


def test_a_dropped_frame_does_not_lose_you():
    """Detectors blink. Snapping to neutral on every blink looks broken."""
    memory = FaceMemory(hold_s=2.0)
    memory.update(detected=True, x=0.4, y=-0.2, now=100.0)
    memory.update(detected=False, x=None, y=None, now=101.0)

    assert memory.aim(now=101.0) == pytest.approx((0.4, -0.2))


def test_the_memory_fades_towards_centre_rather_than_snapping():
    memory = FaceMemory(hold_s=2.0, fade_s=6.0)
    memory.update(detected=True, x=1.0, y=1.0, now=0.0)

    held = memory.aim(now=2.0)
    half = memory.aim(now=5.0)

    assert held == pytest.approx((1.0, 1.0))
    assert half == pytest.approx((0.5, 0.5))
    assert memory.aim(now=8.0) is None


def test_being_seen_again_restores_full_confidence():
    memory = FaceMemory(hold_s=2.0, fade_s=6.0)
    memory.update(detected=True, x=1.0, y=0.0, now=0.0)
    assert memory.confidence(now=5.0) == pytest.approx(0.5)

    memory.update(detected=True, x=1.0, y=0.0, now=5.0)
    assert memory.confidence(now=5.0) == 1.0


@given(age=st.floats(min_value=0.0, max_value=60.0))
@settings(deadline=None, max_examples=100)
def test_confidence_only_ever_decays(age):
    """It must never rebound on its own - only a real sighting restores it."""
    memory = FaceMemory(hold_s=2.0, fade_s=6.0)
    memory.update(detected=True, x=1.0, y=1.0, now=0.0)

    assert 0.0 <= memory.confidence(now=age) <= 1.0
    assert memory.confidence(now=age + 0.5) <= memory.confidence(now=age) + 1e-9


# ------------------------------------------------------- emotion, then speech

# The local rewrite lost body language: the model was asked for words only, so
# the 85-move emotion library went unused and the pet became a talking head.
#
# Asking for structured JSON would get the emotion back but destroy streaming -
# you cannot speak the first clause of a reply that has to be parsed whole. So
# the model emits the emotion on its own first line and the speech after it.
# One call, streaming intact, and the body moves *before* the voice starts,
# which is the right order anyway.


def test_the_emotion_comes_off_the_front():
    emotion, rest = take_emotion(drip("EMOTION: cheerful1\nHello there. "), ALLOWED)

    assert emotion == "cheerful1"
    assert "".join(rest) == "Hello there. "


def test_the_speech_after_it_still_streams_as_clauses():
    """The point of the whole arrangement: speech is not held up."""
    emotion, rest = take_emotion(drip("EMOTION: sad1\nOh no. That's awful. "), ALLOWED)

    assert emotion == "sad1"
    assert list(sentences(rest)) == ["Oh no.", "That's awful."]


def test_an_emotion_it_does_not_have_is_refused():
    """The model will invent names. An unknown move would only fail later."""
    emotion, rest = take_emotion(drip("EMOTION: jubilant7\nHello. "), ALLOWED)

    assert emotion == ""
    assert "".join(rest) == "Hello. "


def test_a_reply_with_no_emotion_line_keeps_all_its_words():
    """A model that ignores the instruction must not cost the user their reply."""
    emotion, rest = take_emotion(drip("Hello there, how are you doing today? "), ALLOWED)

    assert emotion == ""
    assert "Hello there" in "".join(rest)


def test_a_long_first_line_is_not_mistaken_for_an_emotion():
    """Give up quickly rather than swallowing a whole paragraph looking for one."""
    long_reply = "I was thinking about what you said earlier and honestly " * 3
    emotion, rest = take_emotion(drip(long_reply), ALLOWED)

    assert emotion == ""
    assert "".join(rest).startswith("I was thinking")


def test_it_is_not_fussy_about_the_format():
    for line in ("EMOTION: cheerful1", "emotion: cheerful1", "EMOTION:cheerful1",
                 "  EMOTION:  cheerful1  ", "cheerful1"):
        emotion, _ = take_emotion(drip(line + "\nHi. "), ALLOWED)
        assert emotion == "cheerful1", line


def test_an_empty_stream_is_harmless():
    emotion, rest = take_emotion([], ALLOWED)

    assert emotion == ""
    assert "".join(rest) == ""


def test_nothing_is_lost_when_the_emotion_line_is_absent():
    text = "Just talking. And more talking. "
    _, rest = take_emotion(drip(text), ALLOWED)

    assert "".join(rest) == text


# ------------------------------------------------------- a spoken-length budget

# The local rewrite also lost the reply-length cap. It matters more here than
# it looks: the pet is deaf while it talks, so a long reply is a long time in
# which it cannot be interrupted. Whole clauses are dropped rather than cutting
# one in half, because a sentence that stops mid-word sounds broken.


def test_a_short_reply_is_left_alone():
    assert clamp_clauses(["Hello there.", "How are you?"], budget=100) == [
        "Hello there.",
        "How are you?",
    ]


def test_clauses_past_the_budget_are_dropped_whole():
    clauses = ["One two three.", "Four five six.", "Seven eight nine."]
    kept = clamp_clauses(clauses, budget=30)

    assert kept == ["One two three.", "Four five six."]
    assert all(c in clauses for c in kept)


def test_the_first_clause_is_always_said():
    """Silence is worse than going slightly over. It must always answer."""
    kept = clamp_clauses(["A very long opening remark indeed."], budget=5)

    assert kept == ["A very long opening remark indeed."]


def test_nothing_in_means_nothing_out():
    assert clamp_clauses([], budget=100) == []
