"""Reply only when addressed: a face in view, the robot's name, or an exchange in progress."""

import asyncio

from pipecat.frames.frames import BotSpeakingFrame, BotStoppedSpeakingFrame, TranscriptionFrame
from pipecat.processors.filters.function_filter import FunctionFilter

from reachy_sage_voice.addressed import Addressed, addressed_filter


def heard(text):
    return TranscriptionFrame(text, "you", "now")


class Clock:
    t = 100.0

    def __call__(self):
        return self.t


def passes(gate, text):
    return asyncio.run(gate(heard(text)))


def test_lyrics_with_no_face_no_name_and_no_exchange_are_not_answered():
    # Live: "Sister, sister, that's how I missed you" got "Sister, I missed you too!".
    assert not passes(Addressed(face=lambda: False), "Sister, sister, that's how I missed you")


def test_saying_the_name_or_an_alias_is_addressing_it():
    gate = Addressed(aliases=("Reachy",), face=lambda: False)
    assert passes(gate, "Sage, what time is it?")
    assert passes(gate, "hey SAGE")
    assert passes(gate, "Reachy, look here")
    assert not passes(gate, "a sagebrush in the desert")


def test_a_tracked_face_is_someone_talking_to_it():
    assert passes(Addressed(face=lambda: True), "what's the weather")


def test_no_face_reader_means_no_face():
    assert not passes(Addressed(), "what's the weather")


def test_the_exchange_stays_open_for_a_while_after_the_pet_spoke():
    clock = Clock()
    gate = Addressed(face=lambda: False, exchange_s=8.0, now=clock)
    asyncio.run(gate(BotSpeakingFrame()))
    asyncio.run(gate(BotStoppedSpeakingFrame()))
    clock.t += 7.9
    assert passes(gate, "and tomorrow?")
    clock.t += 8.1  # the transcript above refreshed nothing: only the pet's own speech does
    assert not passes(gate, "and the day after?")


def test_other_frames_always_pass():
    assert asyncio.run(Addressed(face=lambda: False)(BotSpeakingFrame()))


def test_the_filter_is_configured_from_the_environment():
    assert addressed_filter({}, None) is None  # off by default
    gate = addressed_filter({"SAGE_VOICE_ADDRESSED": "1", "SAGE_VOICE_ALIASES": "Reachy, Mini",
                             "SAGE_VOICE_EXCHANGE_S": "5"}, lambda: False)
    assert isinstance(gate, FunctionFilter)
    addressed = gate._filter
    assert addressed.exchange_s == 5.0
    assert passes(addressed, "mini, hello") and passes(addressed, "sage hello")
    assert addressed_filter({"SAGE_VOICE_ADDRESSED": "1"}, None)._filter.exchange_s == 8.0
