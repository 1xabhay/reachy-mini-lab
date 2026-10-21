"""How the voice sounds: the voice_style and vocal tools, mapped per TTS engine.

The model picks a style or a wordless sound from fixed lists; what each one
means is the body's business, through one table per engine. A value an engine
cannot do maps to nothing, and the body reports it ``unsupported``.
"""

import math
from dataclasses import dataclass

from pipecat.frames.frames import DataFrame

#: Owner's lists. Order is the schema's order.
VOICE_STYLES = ("neutral", "warm", "excited", "soft", "whisper")
VOCALS = ("laugh", "sigh", "hmm", "gasp")

STYLE_LINES = {
    "neutral": "your everyday voice",
    "warm": "gentle and friendly, for comfort or affection",
    "excited": "quick and bright, for good news",
    "soft": "quiet and slow, for calm or a late hour",
    "whisper": "barely spoken, for a secret",
}
VOCAL_LINES = {
    "laugh": "a short laugh, at something funny",
    "sigh": "a sigh, of relief or tiredness",
    "hmm": "a thinking hum, before an answer",
    "gasp": "a gasp, at a surprise",
}

#: One table per engine: style -> TTS settings, unsupported style -> another
#: style, vocal -> ("sound", move) or ("say", text).
ENGINES = {
    # ponytail: speeds are a first guess, to be judged by ear on the robot.
    "kokoro": {
        "styles": {"neutral": {"speed": 1.0}, "warm": {"speed": 0.95},
                   "excited": {"speed": 1.1}, "soft": {"speed": 0.9}},
        "fallbacks": {"whisper": "soft"},  # Kokoro cannot whisper
        "vocals": {"laugh": ("sound", "laughing2"), "sigh": ("sound", "relief1"),
                   "gasp": ("sound", "surprised1"), "hmm": ("say", "Hmm,")},
    },
}


def style_settings(engine: dict, style: str) -> dict | None:
    """Return the TTS settings for a style on this engine, or None if it cannot do it."""
    style = style if style in engine["styles"] else engine["fallbacks"].get(style, style)
    return engine["styles"].get(style)


def vocal_action(engine: dict, sound: str) -> tuple[str, str] | None:
    """Return ("sound", move) or ("say", text) for a vocal, or None if unsupported."""
    return engine["vocals"].get(sound)


def tools(moves) -> list[dict]:
    """Return the voice_style and vocal tools, MCP-shaped; ``moves`` times the vocal sounds."""
    sounds = [moves[m].duration for kind, m in ENGINES["kokoro"]["vocals"].values()
              if kind == "sound" and m in moves]
    return [
        {"name": "voice_style",
         "description": "Set how your next words sound.",
         "inputSchema": {"type": "object", "required": ["style"], "properties": {"style": {
             "type": "string", "enum": list(VOICE_STYLES),
             "description": "\n".join(f"{s}: {STYLE_LINES[s]}" for s in VOICE_STYLES)}}},
         "max_s": 1, "interruptible": True},
        {"name": "vocal",
         "description": "Make one short wordless sound.",
         "inputSchema": {"type": "object", "required": ["sound"], "properties": {"sound": {
             "type": "string", "enum": list(VOCALS),
             "description": "\n".join(f"{v}: {VOCAL_LINES[v]}" for v in VOCALS)}}},
         "max_s": math.ceil(max(sounds, default=1)) + 1, "interruptible": True},
    ]


@dataclass
class VoiceStyleFrame(DataFrame):
    """TTS settings for the rest of the current reply, in order with its text."""

    settings: dict
