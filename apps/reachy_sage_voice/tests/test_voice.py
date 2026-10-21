"""Expressive tools: voice_style and vocal, mapped onto the TTS engine by a table.

The kernel offers them like any MCP-shaped tool; the body decides what each
value means for its engine, and reports `failed, reason unsupported` for a value
its engine cannot do. TTS is faked: no GPU, no Kokoro needed here.
"""

import asyncio

from pipecat.frames.frames import TTSSpeakFrame
from pipecat.tests.utils import SleepFrame, run_test
from test_kernel import MOVES
from test_sage_processor import FakeSession, delta, heard, make, run, spoken
from test_tts import line, ndjson

from reachy_sage_voice.kernel import manifest
from reachy_sage_voice.sage import Playback, SageProcessor
from reachy_sage_voice.tts import KokoroTTSService
from reachy_sage_voice.voice import (
    ENGINES,
    VOCALS,
    VOICE_STYLES,
    VoiceStyleFrame,
    style_settings,
    vocal_action,
)

KOKORO = ENGINES["kokoro"]


def tools():
    return {t["name"]: t for t in manifest(MOVES)["tools"]}


# --- the manifest -------------------------------------------------------------


def test_voice_style_and_vocal_are_offered_as_mcp_tools():
    for name, field, values in (("voice_style", "style", VOICE_STYLES),
                                ("vocal", "sound", VOCALS)):
        tool = tools()[name]
        schema = tool["inputSchema"]
        assert schema["type"] == "object" and schema["required"] == [field]
        prop = schema["properties"][field]
        assert prop["type"] == "string" and prop["enum"] == list(values)
        assert all(f"{v}:" in prop["description"] for v in values)  # one line each
        assert tool["description"] and tool["interruptible"] is True and tool["max_s"] > 0


def test_the_styles_and_sounds_are_the_owners():
    assert VOICE_STYLES == ("neutral", "warm", "excited", "soft", "whisper")
    assert VOCALS == ("laugh", "sigh", "hmm", "gasp")


# --- the Kokoro table -----------------------------------------------------------


def test_kokoro_styles_change_speed_and_nothing_else():
    for style in ("neutral", "warm", "excited", "soft"):
        settings = style_settings(KOKORO, style)
        assert set(settings) == {"speed"} and 0.5 <= settings["speed"] <= 2.0
    assert style_settings(KOKORO, "neutral") == {"speed": 1.0}
    speed = {s: style_settings(KOKORO, s)["speed"] for s in ("excited", "soft")}
    assert speed["excited"] > 1.0 > speed["soft"]


def test_kokoro_cannot_whisper_so_it_speaks_softly():
    assert style_settings(KOKORO, "whisper") == style_settings(KOKORO, "soft")


def test_kokoro_vocals_are_a_moves_own_sound_or_a_written_hmm():
    assert vocal_action(KOKORO, "laugh") == ("sound", "laughing2")
    assert vocal_action(KOKORO, "sigh") == ("sound", "relief1")
    assert vocal_action(KOKORO, "gasp") == ("sound", "surprised1")
    assert vocal_action(KOKORO, "hmm") == ("say", "Hmm,")


def test_every_vocal_sound_is_a_declared_move():
    from reachy_sage_voice.kernel import EMOTIONS

    for kind, value in KOKORO["vocals"].values():
        assert kind == "say" or value in EMOTIONS


def test_a_value_the_engine_lacks_has_no_mapping():
    bare = {"styles": {"neutral": {"speed": 1.0}}, "fallbacks": {}, "vocals": {}}
    assert style_settings(bare, "excited") is None
    assert vocal_action(bare, "laugh") is None


# --- the body acting on them -------------------------------------------------------


def style_effect(sequence, style):
    return {"sequence": sequence, "kind": "voice_style", "turn_id": "<turn>",
            "input": {"style": style}, "stale": False}


def vocal_effect(sequence, sound):
    return {"sequence": sequence, "kind": "vocal", "turn_id": "<turn>",
            "input": {"sound": sound}, "stale": False}


def styles_in(down):
    return [f.settings for f in down if isinstance(f, VoiceStyleFrame)]


def test_a_voice_style_reaches_tts_before_the_words_it_colours():
    session = FakeSession([{"effects": [style_effect(41, "excited"), delta(0, "Wow, great!")]}])
    down, _ = run(make(session), [heard("I won"), SleepFrame(0.1)])
    assert styles_in(down) == [style_settings(KOKORO, "excited")]
    kinds = [type(f).__name__ for f in down if type(f).__name__ in ("VoiceStyleFrame",
                                                                    "LLMTextFrame")]
    assert kinds[0] == "VoiceStyleFrame"
    assert {"sequence": 41, "status": "done"} in session.results


def test_a_whisper_is_done_softly():
    session = FakeSession([{"effects": [style_effect(41, "whisper")]}])
    down, _ = run(make(session), [heard("shh"), SleepFrame(0.1)])
    assert styles_in(down) == [style_settings(KOKORO, "soft")]
    assert {"sequence": 41, "status": "done"} in session.results


def test_an_engine_that_cannot_do_a_style_reports_unsupported():
    bare = {"styles": {"neutral": {"speed": 1.0}}, "fallbacks": {}, "vocals": {}}
    session = FakeSession([{"effects": [style_effect(41, "excited"), vocal_effect(42, "laugh")]}])

    async def no_move(name, sound):
        pass
    down, _ = run(SageProcessor(session, no_move, Playback(), engine=bare),
                  [heard("hi"), SleepFrame(0.1)])
    assert styles_in(down) == []
    assert {"sequence": 41, "status": "failed", "reason": "unsupported"} in session.results
    assert {"sequence": 42, "status": "failed", "reason": "unsupported"} in session.results


def test_a_hmm_is_said():
    session = FakeSession([{"effects": [vocal_effect(41, "hmm"), delta(0, "let me think.")]}])
    down, _ = run(make(session), [heard("why?"), SleepFrame(0.1)])
    assert "".join(spoken(down)).startswith("Hmm,")
    assert {"sequence": 41, "status": "done"} in session.results


def test_a_laugh_is_the_moves_own_sound():
    played = []

    async def no_move(name, sound):
        pass

    async def play_sound(name):
        played.append(name)
    session = FakeSession([{"effects": [vocal_effect(41, "laugh")]}])
    run(SageProcessor(session, no_move, Playback(), play_sound=play_sound),
        [heard("a joke"), SleepFrame(0.1)])
    assert played == ["laughing2"]
    assert {"sequence": 41, "status": "done"} in session.results


# --- Kokoro honouring the style -----------------------------------------------------


def test_kokoro_speaks_at_the_styles_speed_until_the_reply_ends(http_server):
    http_server.script("POST", "/dev/captioned_speech", ndjson(line(0.1, [("Wow.", 0.0, 0.1)])))
    tts = KokoroTTSService(base_url=http_server.url + "/v1")
    asyncio.run(run_test(tts, frames_to_send=[
        VoiceStyleFrame({"speed": 1.1}), TTSSpeakFrame("Wow."), SleepFrame(0.2)]))
    (req,) = [r for r in http_server.requests if r["path"] == "/dev/captioned_speech"]
    assert req["body"]["speed"] == 1.1
