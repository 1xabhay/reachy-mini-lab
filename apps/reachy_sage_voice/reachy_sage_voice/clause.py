"""Feed the TTS one clause at a time, and otherwise whole sentences.

An earlier voice bot split only the first clause of a reply (``clause_aggregator.py``): only
the first chunk decides when the voice starts. The owner chose every clause for
Kokoro, which the bench measured batching a whole reply at 0.38 s to first
audio against 0.21 s a clause at a time; shorter pieces also make "what was
played" finer when a reply is interrupted. A clause needs 4+ words, so "Sure,"
waits for more rather than costing its own request; only a reply's first
clause may go at 2 ("Of course,"), which the first-audio replay measured at
p50 313-319 ms, p95 537-570 ms against 326-430 / 640-2559 ms (owner;
sage-kernel bench/LEDGER.md, "Voice first audio").

ponytail: installed on the TTS service's private ``_text_aggregator``; Pipecat
1.12 has no hook for a custom one. Replace when it grows a constructor argument.
"""

from pipecat.utils.text.base_text_aggregator import Aggregation, AggregationType
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator, _LookaheadState

CLAUSE_ENDINGS = (",", ";", ":")
MIN_WORDS = 4
FIRST_CLAUSE_MIN_WORDS = 2
SENTENCE_ENDINGS = (".", "!", "?")
#: A first "sentence" ending in one of these with a period is not finished.
TITLES = frozenset({"mr", "mrs", "ms", "dr", "st", "vs", "prof"})


def _has_words(text: str) -> bool:
    return any(c.isalnum() for c in text)


class ClauseTextAggregator(SimpleTextAggregator):
    """SimpleTextAggregator that also lets each clause go as soon as it completes."""

    def __init__(self, **kwargs):
        """Start a reply expecting its first clause."""
        super().__init__(**kwargs)
        self._first = True

    async def aggregate(self, text: str):
        """Yield sentences, and each clause as soon as it is complete."""
        for char in text:  # by character, like Pipecat's: a token's space starts the next word
            async for sentence in super().aggregate(char):
                if not _has_words(sentence.text):
                    continue  # "..." alone: nothing to say
                self._first = False
                yield sentence
            if self._first and char in SENTENCE_ENDINGS and self._finished_first_sentence(char):
                # Pipecat waits for the next word to confirm a sentence; a reply's
                # first one goes now, or "Sure!" waits on the model (live run 4).
                # ponytail: resets Pipecat's private lookahead state alongside _text.
                sentence, self._text, self._first = self._text.strip(), "", False
                self._lookahead_state = _LookaheadState.IDLE
                yield Aggregation(text=sentence, type=AggregationType.SENTENCE)
                continue
            need = FIRST_CLAUSE_MIN_WORDS if self._first else MIN_WORDS
            # A clause ends at its punctuation *and the space after it* ("1,000" is no clause).
            if (char == " " and self._text[-2:-1] in CLAUSE_ENDINGS
                    and len([w for w in self._text.split() if _has_words(w)]) >= need):
                clause, self._text, self._first = self._text.strip(), "", False
                yield Aggregation(text=clause, type=AggregationType.SENTENCE)

    def _finished_first_sentence(self, char: str) -> bool:
        if char != ".":
            return True
        before = self._text[:-1].rstrip()
        if not _has_words(before):
            return False  # "..": an ellipsis, not a sentence
        return not before[-1].isdigit() and before.split()[-1].lower() not in TITLES

    async def flush(self):
        """End the reply; the next one starts with a first clause again."""
        self._first = True
        rest = await super().flush()
        return rest if rest is not None and _has_words(rest.text) else None  # not a stray ".."

    async def handle_interruption(self):
        """Drop the reply; the next one starts with a first clause again."""
        self._first = True
        await super().handle_interruption()


def install(tts) -> None:
    """Give a TTS service the clause aggregator, keeping its language."""
    tts._text_aggregator = ClauseTextAggregator(
        aggregation_type=AggregationType.SENTENCE, language=tts._text_aggregator.language)
