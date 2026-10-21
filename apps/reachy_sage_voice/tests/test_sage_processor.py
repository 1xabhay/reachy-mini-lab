"""SageProcessor: the body's whole conversation with the kernel, inside a real Pipecat pipeline.

The LLM is not in the pipeline. A final transcript goes to the kernel; reply
pieces stream straight to TTS; every delivered effect gets its result; an
interruption reports what was actually played. The session here is a fake that
speaks protocol v1's shapes without HTTP (test_kernel.py covers the wire).
"""

import asyncio

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.tests.utils import SleepFrame, run_test

from reachy_sage_voice.kernel import KernelClient, KernelUnavailableError, Session
from reachy_sage_voice.sage import Playback, SageProcessor

from conftest import TOKEN

TURN = "<turn>"  # replaced by the turn id the processor chose


class FakeSession:
    """Answers polls from a script, then holds the long poll open, as a quiet kernel does."""

    def __init__(self, polls=(), admit="admitted"):
        self.polls = list(polls)
        self.admit = admit
        self.started: list[tuple[str, str, dict]] = []
        self.results: list[dict] = []
        self.stops: list[str] = []
        self.offsets: list[int] = []

    async def start(self, turn_id, text, timing):
        if self.admit == "outage":
            raise KernelUnavailableError("refused")
        self.started.append((turn_id, text, timing))
        return self.admit

    async def poll(self, offset, wait):
        self.offsets.append(offset)
        if not self.polls:
            await asyncio.Event().wait()
        body = self.polls.pop(0)
        turn_id = self.started[-1][0]
        for effect in body.get("effects", []):
            if effect.get("turn_id") == TURN:
                effect["turn_id"] = turn_id
        if body.get("turn", {}).get("turn_id") == TURN:
            body["turn"]["turn_id"] = turn_id
        return {"sequence": 0, "effects": [], **body}

    async def result(self, sequence, status, text=None, timing=None, reason=None, turn_id=None):
        r = {"sequence": sequence, "status": status, "text": text, "timing": timing,
             "reason": reason, "turn_id": turn_id}
        self.results.append({k: v for k, v in r.items() if v is not None})

    async def stop(self, turn_id):
        self.stops.append(turn_id)


def delta(offset, text):
    return {"kind": "reply_delta", "turn_id": TURN, "offset": offset, "text": text}


def reply(sequence, text, stale=False):
    return {"sequence": sequence, "kind": "reply", "turn_id": TURN, "text": text, "stale": stale}


def emote(sequence, name):
    return {"sequence": sequence, "kind": "emote", "turn_id": TURN, "input": {"name": name},
            "stale": False}


def heard(text):
    return TranscriptionFrame(text=text, user_id="you", timestamp="")


def run(processor, frames):
    down, up = asyncio.run(run_test(processor, frames_to_send=frames))
    return down, up


def spoken(down):
    return [f.text for f in down if isinstance(f, (LLMTextFrame, TTSSpeakFrame))]


def make(session, emotes=None, playback=None, fail_emote=False, sounds=None, move_s=0.0):
    async def play_emote(name, sound):
        if fail_emote:
            raise RuntimeError("motor fault")
        emotes.append(name)
        if sounds is not None:
            sounds.append((name, sound))
        await asyncio.sleep(move_s)
    return SageProcessor(session, play_emote, playback or Playback())


def test_reply_pieces_stream_to_tts_as_they_arrive_then_only_the_rest():
    session = FakeSession([{"effects": [delta(0, "Hel")]},
                           {"effects": [delta(3, "lo there.")]},
                           {"effects": [reply(42, "Hello there. Bye.")]}])
    down, _ = run(make(session), [heard("hi"), SleepFrame(0.1)])
    assert session.started[0][1] == "hi"
    assert session.offsets[:3] == [0, 3, 12]
    assert spoken(down) == ["Hel", "lo there.", " Bye."]
    kinds = [type(f) for f in down if isinstance(
        f, (LLMFullResponseStartFrame, LLMTextFrame, LLMFullResponseEndFrame))]
    assert kinds == [LLMFullResponseStartFrame, LLMTextFrame, LLMTextFrame, LLMTextFrame,
                     LLMFullResponseEndFrame]


def test_a_played_reply_is_reported_done_with_its_timing():
    session = FakeSession([{"effects": [delta(0, "Hello.")]}, {"effects": [reply(42, "Hello.")]}])
    run(make(session), [VADUserStoppedSpeakingFrame(stop_secs=0.2), SleepFrame(0.05),
                        heard("hi"), SleepFrame(0.1), BotStartedSpeakingFrame(),
                        SleepFrame(0.05), BotStoppedSpeakingFrame(), SleepFrame(0.05)])
    stt_ms = session.started[0][2]["stt_ms"]
    assert 200 <= stt_ms < 1000  # end of speech is stop_secs before the VAD said so
    (result,) = session.results
    assert result["sequence"] == 42 and result["status"] == "done"
    timing = result["timing"]
    assert 0 <= timing["ttft_ms"] < 200
    assert timing["first_audio_ms"] >= stt_ms + 80  # 100 ms of sleep, less scheduling jitter


def test_interrupted_while_playing_reports_what_was_actually_played():
    playback = Playback()
    playback.spoken = ["Hello, I was going to say."]
    session = FakeSession([{"effects": [reply(42, "Hello, I was going to say. Then more.")]}])
    run(make(session, playback=playback),
        [heard("hi"), SleepFrame(0.1), BotStartedSpeakingFrame(), SleepFrame(0.05),
         InterruptionFrame(), SleepFrame(0.1)])
    assert session.results == [{"sequence": 42, "status": "interrupted",
                                "text": "Hello, I was going to say.",
                                "timing": session.results[0]["timing"]}]
    assert session.stops == []  # the turn had completed: there is nothing left to stop


def test_interrupted_while_still_streaming_reports_what_was_played_then_stops():
    playback = Playback()
    playback.spoken = ["Once upon a time,"]
    session = FakeSession([{"effects": [delta(0, "Once upon a time, there was")]}])
    run(make(session, playback=playback),
        [heard("tell me a story"), SleepFrame(0.1), BotStartedSpeakingFrame(),
         SleepFrame(0.05), InterruptionFrame(), SleepFrame(0.1)])
    turn_id = session.started[0][0]
    (result,) = session.results
    assert {k: result[k] for k in ("turn_id", "status", "text")} == {
        "turn_id": turn_id, "status": "interrupted", "text": "Once upon a time,"}
    assert "sequence" not in result
    assert session.stops == [turn_id]


def test_a_stop_effect_interrupts_the_body():
    session = FakeSession([{"effects": [reply(42, "A long answer.")]},
                           {"effects": [{"sequence": 43, "kind": "stop", "turn_id": None,
                                         "stale": False}]}])
    down, _ = run(make(session), [heard("hi"), SleepFrame(0.2)])
    assert any(isinstance(f, InterruptionFrame) for f in down)
    assert [(r["sequence"], r["status"]) for r in session.results] == [(42, "interrupted")]


def test_an_emote_effect_plays_on_the_body_and_reports_done():
    emotes = []
    session = FakeSession([{"effects": [emote(41, "cheerful1"), reply(42, "")]}])
    run(make(session, emotes), [heard("hi"), SleepFrame(0.1)])
    assert emotes == ["cheerful1"]
    assert {"sequence": 41, "status": "done"} in session.results


def test_an_emote_the_body_cannot_perform_reports_failed():
    session = FakeSession([{"effects": [emote(41, "cheerful1"), reply(42, "")]}])
    run(make(session, fail_emote=True), [heard("hi"), SleepFrame(0.1)])
    assert {"sequence": 41, "status": "failed"} in session.results


def test_an_emote_waits_for_the_next_audio_boundary():
    emotes = []
    playback = Playback()
    playback.speaking = True  # mid-sentence: the move must not land on a word
    session = FakeSession([{"effects": [emote(41, "cheerful1")]}])
    run(make(session, emotes, playback=playback), [heard("hi"), SleepFrame(0.1)])
    assert emotes == []


def test_an_emote_with_nothing_playing_goes_at_once():
    emotes = []
    session = FakeSession([{"effects": [emote(41, "cheerful1")]}])  # before any reply text
    run(make(session, emotes), [heard("hi"), SleepFrame(0.1)])
    assert emotes == ["cheerful1"]
    assert session.results == [{"sequence": 41, "status": "done"}]


def test_with_nothing_to_say_the_move_plays_its_own_sound():
    sounds = []
    session = FakeSession([{"effects": [emote(41, "cheerful1"), reply(42, "")]}])
    run(make(session, [], sounds=sounds), [heard("hi"), SleepFrame(0.1)])
    assert sounds == [("cheerful1", True)]


def test_with_a_real_answer_to_say_the_move_is_muted():
    sounds = []
    session = FakeSession([{"effects": [emote(41, "cheerful1"),
                                        reply(42, "That is wonderful news, well done.")]}])
    run(make(session, [], sounds=sounds), [heard("hi"), SleepFrame(0.1)])
    assert sounds == [("cheerful1", False)]


def test_a_move_decided_before_the_reply_is_muted():
    sounds = []
    session = FakeSession([{"effects": [emote(41, "cheerful1")]}])  # reply not written yet
    run(make(session, [], sounds=sounds), [heard("hi"), SleepFrame(0.1)])
    assert sounds == [("cheerful1", False)]


def test_a_short_reply_is_said_first_then_the_move_sounds_after_the_voice():
    sounds, playback = [], Playback()
    session = FakeSession([{"effects": [emote(41, "yes1"), reply(42, "Yes!")]}])
    run(make(session, [], playback=playback, sounds=sounds),
        [heard("are you there"), SleepFrame(0.1)])
    assert sounds == []  # "Yes!" is queued for the voice: the sound must wait
    assert playback.busy


def test_the_voice_waits_for_a_move_sound_to_finish():
    playback = Playback()
    playback.hold_voice()  # a move is sounding
    session = FakeSession([{"effects": [delta(0, "Hello there.")]}])
    down, _ = run(make(session, playback=playback), [heard("hi"), SleepFrame(0.1)])
    assert spoken(down) == []
    playback.release_voice()


def test_a_stale_reply_is_not_spoken():
    session = FakeSession([{"effects": [reply(42, "Old news.", stale=True)]}])
    down, _ = run(make(session), [heard("hi"), SleepFrame(0.1)])
    assert spoken(down) == []
    assert session.results == [{"sequence": 42, "status": "failed", "reason": "stale"}]


def test_emoji_in_reply_pieces_are_never_spoken():
    session = FakeSession([{"effects": [delta(0, "Lovely 😊 day")]}])
    down, _ = run(make(session), [heard("hi"), SleepFrame(0.1)])
    assert spoken(down) == ["Lovely  day"]


def test_a_busy_kernel_gets_one_moment():
    down, _ = run(make(FakeSession(admit="busy")), [heard("hi"), SleepFrame(0.05)])
    assert spoken(down) == ["One moment."]


def test_an_unreachable_kernel_says_so_rather_than_nothing():
    down, _ = run(make(FakeSession(admit="outage")), [heard("hi"), SleepFrame(0.05)])
    assert spoken(down) == ["I can't reach my memory right now."]


def test_speech_while_the_kernel_is_thinking_joins_the_turn_it_replaces():
    # The kernel forgets a turn stopped before its reply, so the words go again.
    session = FakeSession()  # thinking: the long poll never answers
    run(make(session), [heard("What's the pin?"), SleepFrame(0.05), heard("Really important."),
                        SleepFrame(0.05), heard("Write it down."), SleepFrame(0.05)])
    assert [text for _, text, _ in session.started] == [
        "What's the pin?", "What's the pin? Really important.",
        "What's the pin? Really important. Write it down."]
    assert session.stops == [turn_id for turn_id, _, _ in session.started[:2]]


def test_speech_after_the_reply_started_is_sent_alone():
    # A streamed reply's turn is in the kernel's memory already, with what was heard.
    session = FakeSession([{"effects": [delta(0, "Sure.")]}])
    run(make(session), [heard("hi"), SleepFrame(0.05), heard("more"), SleepFrame(0.05)])
    assert [text for _, text, _ in session.started] == ["hi", "more"]


def test_a_turn_replaced_mid_reply_clears_its_words_from_the_voice():
    # Live 10:50: "Oh" streamed, a new transcript replaced the turn, and the TTS
    # still held "Oh" when the next reply began: "OhOh, a messy screen!".
    session = FakeSession([{"effects": [delta(0, "Oh")]}])
    down, _ = run(make(session), [heard("hi"), SleepFrame(0.05), heard("more"), SleepFrame(0.05)])
    kinds = [type(f) for f in down if isinstance(f, (LLMTextFrame, InterruptionFrame))]
    assert kinds == [LLMTextFrame, InterruptionFrame]


def test_interim_transcripts_are_not_turns():
    session = FakeSession()
    run(make(session), [InterimTranscriptionFrame(text="hi", user_id="you", timestamp=""),
                        SleepFrame(0.05)])
    assert session.started == []


def test_reachy_hearing_itself_is_not_sent_to_the_kernel():
    session = FakeSession([{"effects": [reply(42, "I think the weather is lovely today.")]}])
    run(make(session), [heard("what is the weather like"), SleepFrame(0.1),
                        BotStartedSpeakingFrame(), BotStoppedSpeakingFrame(), SleepFrame(0.05),
                        heard("the weather is lovely today"), SleepFrame(0.05)])
    assert [text for _, text, _ in session.started] == ["what is the weather like"]


def test_over_http_a_streamed_turn_ends_with_a_done_result(kernel):
    """The real client and session against the scripted kernel, end to end."""
    def turn(fake):
        return fake.events("submitted")[0]["turn_id"]

    kernel.script("POST", "/v1/hello", (200, {"protocol": 1, "sequence": 40}))
    kernel.script("POST", "/v1/events", (200, {"sequence": 41}))
    kernel.script("GET", "/v1/effects",
                  lambda k: (200, {"sequence": 41, "effects": [
                      {"kind": "reply_delta", "turn_id": turn(k), "offset": 0, "text": "Hi"}]}),
                  lambda k: (200, {"sequence": 42, "effects": [
                      {"sequence": 42, "kind": "reply", "turn_id": turn(k), "text": "Hi there.",
                       "stale": False}]}),
                  (200, {"sequence": 42, "effects": []}))
    session = Session(KernelClient(kernel.url, TOKEN), {"senses": ["speech"], "tools": []})
    down, _ = run(SageProcessor(session, lambda name, sound: None, Playback()),
                  [heard("hello"), SleepFrame(0.2), BotStartedSpeakingFrame(),
                   BotStoppedSpeakingFrame(), SleepFrame(0.2)])
    assert spoken(down) == ["Hi", " there."]
    polls = [r["query"] for r in kernel.requests if r["method"] == "GET"]
    assert polls[0]["since"] == ["40"] and polls[1]["offset"] == ["2"]
    (result,) = kernel.events("result")
    assert result["sequence"] == 42 and result["status"] == "done"
    assert set(result["timing"]) == {"ttft_ms"}  # no VAD frame here, so no first_audio_ms


def test_its_own_last_words_are_trimmed_from_the_next_turn():
    session = FakeSession([{"effects": [reply(42, "Yes, I'm here! Let's talk.")]}])
    run(make(session), [heard("you up?"), SleepFrame(0.1),
                        BotStartedSpeakingFrame(), BotStoppedSpeakingFrame(), SleepFrame(0.05),
                        heard("Let's talk. Alright, let's see."), SleepFrame(0.05)])
    assert [text for _, text, _ in session.started] == ["you up?", "Alright, let's see."]


def test_a_turn_that_is_only_its_own_echo_is_dropped():
    session = FakeSession([{"effects": [reply(42, "Yes, I'm here! Let's talk.")]}])
    run(make(session), [heard("you up?"), SleepFrame(0.1),
                        BotStartedSpeakingFrame(), BotStoppedSpeakingFrame(), SleepFrame(0.05),
                        heard("Let's talk."), SleepFrame(0.05)])
    assert [text for _, text, _ in session.started] == ["you up?"]


def test_the_words_actually_played_count_as_its_speech_too():
    playback = Playback()
    playback.spoken = ["Once", "upon", "a", "time,"]  # an interrupted reply
    session = FakeSession()
    run(make(session, playback=playback), [heard("a time, tell me another"), SleepFrame(0.05)])
    assert [text for _, text, _ in session.started] == ["tell me another"]


def test_tool_syntax_split_across_reply_pieces_is_never_spoken():
    session = FakeSession([{"effects": [delta(0, "Great news (emo")]},
                           {"effects": [delta(15, "te: cheerful1) indeed.")]},
                           {"effects": [reply(42, "Great news (emote: cheerful1) indeed.")]}])
    down, _ = run(make(session), [heard("hi"), SleepFrame(0.1)])
    said = "".join(spoken(down))
    assert "emote" not in said and "cheerful1" not in said
    assert said.split() == ["Great", "news", "indeed."]
