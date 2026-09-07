"""Checks for the layers: each alone, then combined every way that makes sense.

The requirement is that vision, hearing, voice and mind are independent - each
useful with nothing else switched on - and that any combination of them works
without special-casing. So each layer is exercised on its own here, and then
the pairs, and then the whole animal.

Every module exposes a synchronous `step()`. Threads only ever loop it, so all
of this runs without a single thread, sleep, or timing assumption.
"""

import numpy as np
import pytest
from reachy_mini_pet.bus import Bus
from reachy_mini_pet.conductor import Conductor
from reachy_mini_pet.events import (
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
from reachy_mini_pet.layers import Hearing, Mind, Transcriber, Vision, Voice


class FakeEyes:
    """A camera that sees whatever the test says it sees."""

    def __init__(self, script=None):
        self.script = list(script or [])
        self.aimed = []

    def look(self):
        return self.script.pop(0) if self.script else (False, None, None)

    def turn_body(self, yaw):
        self.aimed.append(yaw)


class FakeEars:
    """A microphone that hands over scripted frames."""

    def __init__(self, script=None):
        self.script = list(script or [])
        self.samplerate = 16000

    def listen(self):
        return self.script.pop(0) if self.script else None


class FakeMouth:
    """A speaker that remembers what it was asked to say."""

    def __init__(self):
        self.said = []

    def say(self, text):
        self.said.append(text)


def voiced(n):
    """Build n frames that the detector will call speech."""
    return [("voiced", i) for i in range(n)]


# ---------------------------------------------------------- vision alone


def test_vision_alone_reports_seeing_you():
    bus = Bus()
    inbox = bus.subscribe(FaceSeen)
    eyes = FakeEyes([(True, 0.1, 0.0)])
    vision = Vision(eyes)
    vision.start(bus)

    vision.step(now=1.0)

    seen = bus.drain(inbox)
    assert len(seen) == 1
    assert seen[0].x == pytest.approx(0.1)


def test_vision_alone_reports_losing_you():
    bus = Bus()
    inbox = bus.subscribe(FaceLost)
    vision = Vision(FakeEyes([(True, 0.1, 0.0), (False, None, None)]))
    vision.start(bus)

    vision.step(now=1.0)
    vision.step(now=20.0)          # long enough for the memory to fade

    assert len(bus.drain(inbox)) == 1


def test_vision_alone_turns_the_body_to_keep_you_in_frame():
    """Vision is useful with nothing else running at all."""
    eyes = FakeEyes([(True, 0.9, 0.0)] * 40)
    vision = Vision(eyes)
    vision.start(Bus())
    for i in range(40):
        vision.step(now=i * 0.05)

    assert eyes.aimed, "the body should have been asked to turn"
    assert abs(eyes.aimed[-1]) > 0.0


# --------------------------------------------------------- hearing alone


def test_hearing_alone_publishes_a_finished_utterance():
    bus = Bus()
    inbox = bus.subscribe(Utterance)
    ears = FakeEars(voiced(30) + [("quiet", i) for i in range(60)])
    hearing = Hearing(ears, is_speech=lambda f: f[0] == "voiced")
    hearing.start(bus)

    for _ in range(90):
        hearing.step()

    assert len(bus.drain(inbox)) == 1


def test_hearing_alone_ignores_a_quiet_room():
    bus = Bus()
    inbox = bus.subscribe(Utterance)
    hearing = Hearing(FakeEars([("quiet", i) for i in range(200)]),
                      is_speech=lambda f: f[0] == "voiced")
    hearing.start(bus)

    for _ in range(200):
        hearing.step()

    assert bus.drain(inbox) == []


# ----------------------------------------------------------- voice alone


def test_voice_alone_speaks_when_told():
    bus = Bus()
    mouth = FakeMouth()
    voice = Voice(mouth)
    voice.start(bus)

    bus.publish(Reply(clauses=["Hello there."], emotion="cheerful1", turn=1))
    voice.step()

    assert mouth.said == ["Hello there."]


def test_voice_alone_says_nothing_unprompted():
    mouth = FakeMouth()
    voice = Voice(mouth)
    voice.start(Bus())
    voice.step()

    assert mouth.said == []


# ------------------------------------------------------ combined: pairs


def test_hearing_and_transcription_turn_sound_into_words():
    bus = Bus()
    inbox = bus.subscribe(Heard)
    transcriber = Transcriber(transcribe=lambda pcm, sr: "put the kettle on")
    transcriber.start(bus)

    bus.publish(Utterance(audio=np.zeros(16000, dtype=np.float32), samplerate=16000))
    transcriber.step()

    heard = bus.drain(inbox)
    assert [h.text for h in heard] == ["put the kettle on"]


def test_words_become_a_reply_with_a_feeling():
    bus = Bus()
    inbox = bus.subscribe(Reply)
    mind = Mind(think=lambda text, history: (["Kettle's on."], "helpful1"))
    mind.start(bus)

    bus.publish(Heard(text="put the kettle on"))
    mind.step()

    replies = bus.drain(inbox)
    assert replies[0].clauses == ["Kettle's on."]
    assert replies[0].emotion == "helpful1"


def test_a_reply_moves_the_body_as_well_as_the_mouth():
    """Body language and speech come from the same thought, not bolted on."""
    bus = Bus()
    inbox = bus.subscribe(Emote)
    mind = Mind(think=lambda text, history: (["Oh no."], "sad1"))
    mind.start(bus)

    bus.publish(Heard(text="I had an awful day"))
    mind.step()

    assert [e.name for e in bus.drain(inbox)] == ["sad1"]


def test_transcription_alone_does_nothing_without_hearing():
    """Each layer is inert until something it cares about happens."""
    bus = Bus()
    inbox = bus.subscribe(Heard)
    Transcriber(transcribe=lambda pcm, sr: "should not happen").start(bus)

    assert bus.drain(inbox) == []


# -------------------------------------------------- combined: the animal


def test_the_whole_chain_from_a_noise_to_a_spoken_reply():
    bus = Bus()
    spoken = bus.subscribe(Spoke)
    mouth = FakeMouth()

    ears = FakeEars(voiced(30) + [("quiet", i) for i in range(60)])
    hearing = Hearing(ears, is_speech=lambda f: f[0] == "voiced")
    transcriber = Transcriber(transcribe=lambda pcm, sr: "good morning")
    mind = Mind(think=lambda text, history: (["Morning!"], "cheerful1"))
    voice = Voice(mouth)

    for module in (hearing, transcriber, mind, voice):
        module.start(bus)

    for _ in range(90):
        hearing.step()
    transcriber.step()
    mind.step()
    voice.step()

    assert mouth.said == ["Morning!"]
    assert [s.text for s in bus.drain(spoken)] == ["Morning!"]


def test_the_mind_remembers_the_conversation():
    bus = Bus()
    asked = []
    def remember(text, history):
        asked.append((text, list(history)))
        return [text], "yes1"

    mind = Mind(think=remember)
    mind.start(bus)

    bus.publish(Heard(text="my name is Abhay"))
    mind.step()
    bus.publish(Heard(text="what is my name"))
    mind.step()

    assert asked[1][1], "the second thought should have seen the first exchange"


def test_a_broken_layer_does_not_stop_the_others():
    """One failing sense must not take the animal down."""
    bus = Bus()
    inbox = bus.subscribe(Heard)

    def explode(pcm, sr):
        raise RuntimeError("whisper fell over")

    transcriber = Transcriber(transcribe=explode)
    transcriber.start(bus)
    bus.publish(Utterance(audio=np.zeros(10, dtype=np.float32), samplerate=16000))

    transcriber.step()          # must not raise

    assert bus.drain(inbox) == []


def test_a_fabricated_transcript_never_reaches_the_mind():
    """Whisper invents fluent text from non-speech; the pet must not answer it."""
    bus = Bus()
    inbox = bus.subscribe(Heard)
    transcriber = Transcriber(
        transcribe=lambda pcm, sr: "can't see it can't see it can't see it can't see it"
    )
    transcriber.start(bus)

    bus.publish(Utterance(audio=np.zeros(16000, dtype=np.float32), samplerate=16000))
    transcriber.step()

    assert bus.drain(inbox) == []


def test_a_transcript_from_a_mostly_silent_clip_is_ignored():
    """The regime where Whisper hallucinates about half the time."""
    bus = Bus()
    inbox = bus.subscribe(Heard)
    transcriber = Transcriber(transcribe=lambda pcm, sr: "and our numbers are up this year")
    transcriber.start(bus)

    bus.publish(
        Utterance(
            audio=np.zeros(16000, dtype=np.float32),
            samplerate=16000,
            voiced_fraction=0.03,
        )
    )
    transcriber.step()

    assert bus.drain(inbox) == []


def test_hearing_reports_how_much_of_the_clip_was_speech():
    """The transcriber cannot judge a clip without knowing this."""
    bus = Bus()
    inbox = bus.subscribe(Utterance)
    ears = FakeEars(voiced(30) + [("quiet", i) for i in range(60)])
    hearing = Hearing(ears, is_speech=lambda f: f[0] == "voiced")
    hearing.start(bus)

    for _ in range(90):
        hearing.step()

    sent = bus.drain(inbox)
    assert sent, "an utterance should have been published"
    assert 0.0 < sent[0].voiced_fraction <= 1.0


# ------------------------------------------------- the voice tells the ears

# The conductor has a whole half about speaking - tail gate, ducking,
# interruption - and none of it was wired to anything. `on_playback_started`,
# `on_playback_finished` and `interrupt` were called by no one, so
# `Conductor.speaking` was permanently False and the tail gate never armed.
# The pet was relying entirely on the hardware echo canceller to avoid
# answering itself, with no defence if that ever stopped converging.
#
# The layers still may not know about each other, so the voice announces
# itself on the bus and hearing listens - the same way everything else here
# is joined up.


def test_the_voice_announces_when_it_starts_and_stops():
    bus = Bus()
    inbox = bus.subscribe(VoiceStarted)
    ended = bus.subscribe(VoiceStopped)
    voice = Voice(FakeMouth())
    voice.start(bus)

    bus.publish(Reply(clauses=["Hello."], emotion="", turn=1))
    voice.step()

    assert len(bus.drain(inbox)) == 1
    assert len(bus.drain(ended)) == 1


def test_hearing_stops_listening_while_the_voice_is_talking():
    """Otherwise its own sentence is transcribed as though you had said it."""
    bus = Bus()
    hearing = Hearing(FakeEars(voiced(50)), is_speech=lambda f: True)
    hearing.start(bus)

    bus.publish(VoiceStarted(turn=1))
    hearing.step()

    assert hearing.conductor.speaking is True


def test_hearing_distrusts_the_microphone_just_after_the_voice_stops():
    """The tail of its own sentence is still in the room."""
    bus = Bus()
    inbox = bus.subscribe(Utterance)
    hearing = Hearing(
        FakeEars(voiced(200)),
        is_speech=lambda f: True,
        conductor=Conductor(min_speech_s=0.1, hangover_s=0.2, tail_gate_s=1.0),
    )
    hearing.start(bus)

    bus.publish(VoiceStarted(turn=1))
    hearing.step()
    bus.publish(VoiceStopped(turn=1))
    for _ in range(20):                       # inside the tail gate
        hearing.step()

    assert bus.drain(inbox) == [], "its own echo became an utterance"


def test_hearing_listens_again_once_the_tail_has_passed():
    bus = Bus()
    inbox = bus.subscribe(Utterance)
    hearing = Hearing(
        FakeEars(voiced(60) + [("quiet", i) for i in range(60)]),
        is_speech=lambda f: f[0] == "voiced",
        conductor=Conductor(min_speech_s=0.1, hangover_s=0.2, tail_gate_s=0.1),
    )
    hearing.start(bus)

    bus.publish(VoiceStarted(turn=1))
    hearing.step()
    bus.publish(VoiceStopped(turn=1))
    for _ in range(120):
        hearing.step()

    assert bus.drain(inbox), "it should be listening again by now"


def test_the_detector_is_reset_between_utterances():
    """Silero is recurrent: without a reset the last utterance bleeds into the next.

    `vad.py` says so in its own docstring, and nothing was calling it.
    """
    bus = Bus()
    resets = []
    ears = FakeEars(voiced(30) + [("quiet", i) for i in range(60)])
    hearing = Hearing(
        ears,
        is_speech=lambda f: f[0] == "voiced",
        reset_detector=lambda: resets.append(True),
    )
    hearing.start(bus)

    for _ in range(90):
        hearing.step()

    assert resets, "the detector should have been reset after the utterance"


def test_not_passing_a_reset_is_fine():
    """Callers that do not have a resettable detector must still work."""
    bus = Bus()
    hearing = Hearing(FakeEars(voiced(5)), is_speech=lambda f: True)
    hearing.start(bus)

    hearing.step()          # must not raise


# --------------------------------------------- not hammering the robot

# `run` steps every layer at 200 Hz, and the follower returns a yaw on every
# observation even when the rate limiter makes it identical to the last one. So
# vision was sending ~200 body-yaw commands a second down the same WebSocket
# whose death this pet has now suffered three times. Whether or not it is the
# cause, sending 199 redundant commands a second is indefensible.


def test_an_unchanged_aim_is_not_sent_again():
    """The robot is already pointing there. Saying so again is pure noise."""
    eyes = FakeEyes([(True, 0.0, 0.0)] * 50)      # dead centre, never moves
    vision = Vision(eyes)
    vision.start(Bus())
    for i in range(50):
        vision.step(now=i * 0.05)

    assert len(eyes.aimed) <= 1, f"sent {len(eyes.aimed)} commands for one position"


def test_a_real_change_is_still_sent():
    eyes = FakeEyes([(True, 0.9, 0.0)] * 60)
    vision = Vision(eyes)
    vision.start(Bus())
    for i in range(60):
        vision.step(now=i * 0.05)

    assert eyes.aimed, "it should have turned towards them"
    assert abs(eyes.aimed[-1]) > 0.0


def test_the_number_of_commands_is_far_below_the_step_rate():
    """Stepped 200 times; the body cannot meaningfully move 200 times."""
    eyes = FakeEyes([(True, 0.5, 0.0)] * 200)
    vision = Vision(eyes)
    vision.start(Bus())
    for i in range(200):
        vision.step(now=i * 0.005)                # 200 Hz, as `run` really does

    assert len(eyes.aimed) < 40, f"sent {len(eyes.aimed)} commands in one second"


def test_layers_can_ask_to_be_stepped_less_often():
    """Hearing needs every frame; vision does not need 200 looks a second."""
    assert Vision.period > 0.0
    assert Hearing.period == 0.0


# --------------------------------------------------- remembering you at all

# The old cloud implementation kept the conversation in a file and reloaded it
# on start; the rewrite forgot everything on restart. That is a shipped feature
# regressing, so it moves here before that module is deleted.
#
# The store is injected: the layer knows there is a memory, not where it lives.


class FakeMemory:
    """A conversation store that keeps everything in hand."""

    def __init__(self, history=None):
        self.history = list(history or [])
        self.saves = 0

    def load(self):
        return list(self.history)

    def save(self, history):
        self.history = list(history)
        self.saves += 1


def test_the_mind_starts_from_what_it_remembers():
    remembered = [
        {"role": "user", "content": "my cat is called Mim"},
        {"role": "assistant", "content": "Hello Mim."},
    ]
    mind = Mind(think=lambda text, history: (["ok"], ""), memory=FakeMemory(remembered))

    assert mind.history == remembered


def test_the_mind_writes_the_exchange_down():
    bus = Bus()
    memory = FakeMemory()
    mind = Mind(think=lambda text, history: (["Hello Mim."], ""), memory=memory)
    mind.start(bus)

    bus.publish(Heard(text="my cat is called Mim"))
    mind.step()

    assert memory.saves == 1
    assert [m["content"] for m in memory.history] == ["my cat is called Mim", "Hello Mim."]


def test_a_restart_still_knows_you():
    """The whole point: yesterday's conversation is still there."""
    bus = Bus()
    memory = FakeMemory()
    first = Mind(think=lambda text, history: (["Hello Mim."], ""), memory=memory)
    first.start(bus)
    bus.publish(Heard(text="my cat is called Mim"))
    first.step()

    asked = []

    def remember(text, history):
        asked.append(list(history))
        return ["Mim."], ""

    second = Mind(think=remember, memory=memory)
    second.start(Bus())
    second.bus.publish(Heard(text="what is my cat called"))
    second.step()

    assert asked and "my cat is called Mim" in [m["content"] for m in asked[0]]


def test_no_memory_at_all_is_fine():
    """A mind with nowhere to write things down must still work."""
    bus = Bus()
    mind = Mind(think=lambda text, history: (["ok"], ""))
    mind.start(bus)

    bus.publish(Heard(text="hello"))
    mind.step()          # must not raise


def test_a_broken_memory_does_not_stop_it_thinking():
    """A full disk must not make the pet mute."""

    class Broken(FakeMemory):
        def save(self, history):
            raise OSError("disk full")

    bus = Bus()
    inbox = bus.subscribe(Reply)
    mind = Mind(think=lambda text, history: (["ok"], ""), memory=Broken())
    mind.start(bus)

    bus.publish(Heard(text="hello"))
    mind.step()

    assert bus.drain(inbox), "it should still have replied"


def test_the_remembered_conversation_never_opens_on_the_assistant():
    """An assistant-first history is rejected by some model APIs outright."""
    lopsided = [
        {"role": "assistant", "content": "dangling"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    mind = Mind(think=lambda text, history: (["ok"], ""), memory=FakeMemory(lopsided))

    assert mind.history[0]["role"] == "user"
