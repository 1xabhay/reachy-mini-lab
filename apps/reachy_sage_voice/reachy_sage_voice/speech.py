"""Text rules at the edges of the voice: what is not a user turn, what is never said."""

import re
from collections.abc import Iterable

#: Word-set similarity at which a transcript is the bot hearing itself. Measured
#: in an earlier voice bot; Reachy needs it because its
#: mic hears its own speaker, late, through WebRTC.
SELF_ECHO_SIMILARITY = 0.65

_WORD = re.compile(r"[a-z0-9]+")


def text_similarity(a: str, b: str) -> float:
    """Word-level Jaccard similarity in [0, 1]; 0 if either side has no words."""
    wa, wb = set(_WORD.findall(a.lower())), set(_WORD.findall(b.lower()))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def is_self_echo(transcript: str, recent_bot_speech: Iterable[str]) -> bool:
    """Tell whether this transcript is the bot's own recent speech heard back."""
    return any(text_similarity(transcript, said) >= SELF_ECHO_SIMILARITY
               for said in recent_bot_speech)


#: Emoji a model writes, read as the body language Reachy can show instead.
#: Same table as the pet's ``layers.EMOJI_FEELINGS``, kept here so this app
#: never imports the live pet.
EMOJI_FEELINGS = {
    "😊": "cheerful1", "🙂": "cheerful1", "😄": "cheerful1", "😀": "cheerful1", "😁": "cheerful1",
    "😂": "laughing1", "🤣": "laughing1", "😢": "sad1", "😭": "sad1", "😞": "sad1", "🙁": "sad1",
    "🤔": "thoughtful1", "🧐": "curious1", "😮": "surprised1", "😲": "surprised1", "🤩": "amazed1",
    "❤": "loving1", "😍": "loving1", "🥰": "loving1", "💕": "loving1", "👍": "yes1", "✅": "yes1",
    "👎": "no1", "👋": "welcoming1", "🤗": "welcoming1", "😳": "shy1", "🙈": "shy1", "😴": "tired1",
    "🥱": "tired1", "😨": "scared1", "😱": "scared1", "🙏": "grateful1", "🎉": "success1",
    "😅": "oops1", "😌": "relief1", "😕": "confused1", "😤": "impatient1",
}
_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿️‍⃣]")
#: "EMOTION: thoughtful1", as small models write it (the pet's streaming.py rule).
_EMOTION_LABEL = re.compile(r"emotion[ \t]*:[ \t]*([a-z0-9_-]+)", re.IGNORECASE)


#: The tools this body declares. A model that writes a call into its reply
#: instead of making it ("(emote: cheerful1)") must not be read out.
TOOLS = ("emote",)


def strip_tool_syntax(text: str, names: Iterable[str]) -> str:
    """Remove every (...) or [...] span that names a declared tool or move."""
    pattern = "|".join(re.escape(n) for n in names)
    return re.sub(rf"[(\[][^()\[\]]*\b(?:{pattern})\b[^()\[\]]*[)\]]", " ", text,
                  flags=re.IGNORECASE)


def speakable(words: str, vocabulary: tuple[str, ...],
              tools: tuple[str, ...] = TOOLS) -> tuple[str, str]:
    """Return (speech, feeling): no emoji, emotion label or tool syntax left to say aloud.

    The feeling is the first labelled emotion, else the first emoji, that this
    body can show; empty if none. Labels go whether or not the name is real.
    """
    words = strip_tool_syntax(words, (*tools, *vocabulary))
    label = next((m.group(1).lower() for m in _EMOTION_LABEL.finditer(words)
                  if m.group(1).lower() in vocabulary), "")
    emoji = next((EMOJI_FEELINGS[c] for c in words if EMOJI_FEELINGS.get(c) in vocabulary), "")
    text = _EMOJI.sub(" ", _EMOTION_LABEL.sub(" ", words))
    return re.sub(r"\s+([!?.,;:])", r"\1", " ".join(text.split())), label or emoji


def strip_emoji(piece: str) -> str:
    """Remove emoji from one streamed piece of a reply, keeping its spacing.

    Spacing matters: pieces are joined as they arrive, so normalising here
    would glue words across pieces together.
    """
    return _EMOJI.sub("", piece)


def addressed(transcript: str, looking: bool | None = None,
              names: tuple[str, ...] = ("reachy",)) -> bool:
    """Tell whether speech was meant for the robot: its name, or a look its way.

    A hook, not yet wired: the bench found no local model good enough to gate
    live turns. ``looking`` is the face tracker's gaze, None when unknown.
    ``names`` is the calibration knob for how the STT spells the name.
    """
    return looking is True or not set(names).isdisjoint(_WORD.findall(transcript.lower()))


#: Most words a reply may have for a move to play its own sound beside it.
MOVE_SOUND_MAX_WORDS = 3


def move_sound(reply: str | None, voice_busy: bool) -> bool:
    """Tell whether a move plays its own sound, rather than muted (owner's rule).

    Only when there is little to say: the reply is empty or at most 3 words
    (``None`` while it is still being written, which mutes), and the voice is
    neither speaking nor about to speak, so a sound never overlaps it.
    """
    return (reply is not None and not voice_busy
            and len(_WORD.findall(reply.lower())) <= MOVE_SOUND_MAX_WORDS)


#: Fewest of the robot's own closing words that count as an echo. One shared
#: word ("talk", "yes") is too often just the person's own.
ECHO_MIN_WORDS = 2


def _key(token: str) -> str:
    return "".join(_WORD.findall(token.lower()))


def trim_echo(transcript: str, recent_speech: Iterable[str]) -> str:
    """Strip a leading echo: the robot's own last words heard back before the person's.

    When the transcript starts with ``ECHO_MIN_WORDS`` or more words that end
    something the robot recently said, they are removed (the longest such
    match). Returns what is left, which may be empty.

    ponytail: exact words after normalising case and punctuation; an STT that
    mishears the echo slips through. Allow a near match if live logs show it.
    """
    tokens = [t for t in transcript.split() if _key(t)]
    keys = [_key(t) for t in tokens]
    cut = 0
    for said in recent_speech:
        tail = [k for k in map(_key, said.split()) if k]
        for n in range(min(len(keys), len(tail)), ECHO_MIN_WORDS - 1, -1):
            if keys[:n] == tail[-n:]:
                cut = max(cut, n)
                break
    return " ".join(tokens[cut:]) if cut else transcript


class SpeechStream:
    """Make streamed reply pieces speakable, holding back an unclosed bracket.

    Tool syntax can be split across pieces ("(emo" + "te: cheerful1)"), so a
    piece is released only up to an open bracket until it closes, or until
    ``MAX_HOLD`` characters show it was ordinary text after all.
    """

    MAX_HOLD = 80

    def __init__(self, names: Iterable[str]) -> None:
        """Strip brackets naming any of ``names`` (tools and moves)."""
        self._names, self._held = tuple(names), ""

    def feed(self, piece: str) -> str:
        """Take one piece; return what can be spoken now."""
        text, self._held = self._held + piece, ""
        cut = max(text.rfind("("), text.rfind("["))
        if cut >= 0 and not re.search(r"[)\]]", text[cut:]) and len(text) - cut <= self.MAX_HOLD:
            text, self._held = text[:cut], text[cut:]
        return strip_emoji(strip_tool_syntax(text, self._names))

    def flush(self) -> str:
        """Return whatever was still held, now the reply is over."""
        text, self._held = self._held, ""
        return strip_emoji(strip_tool_syntax(text, self._names))
