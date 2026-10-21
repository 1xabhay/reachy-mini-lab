"""Kokoro is fed one clause at a time, and otherwise whole sentences.

A reply's first clause may be 2+ words, later ones need 4+ (owner, after the
bench's first-audio replay: p50 313-319 ms and p95 537-570 ms against 326-430
and 640-2559 ms for 4+ throughout, sage-kernel bench/LEDGER.md).

Ported from an earlier Pipecat voice bot's clause tests, which split only the first
clause. The owner chose every clause: the bench measured Kokoro batching a whole
reply at 0.38 s to first audio against 0.21 s a clause at a time.
"""

import asyncio
import re

from reachy_sage_voice.clause import ClauseTextAggregator


def llm_tokens(reply: str) -> list[str]:
    """How an LLM streams: the space starts the next token, punctuation comes alone."""
    return re.findall(r" ?[\w']+|[^\w\s]", reply)


def chunks(*replies, interrupt_after=None, tokens=llm_tokens):
    async def run():
        agg, out = ClauseTextAggregator(), []
        for n, reply in enumerate(replies):
            spoken = []
            for token in tokens(reply):
                async for a in agg.aggregate(token):
                    spoken.append(a.text)
            if interrupt_after == n:
                await agg.handle_interruption()
            else:
                rest = await agg.flush()
                if rest and rest.text:
                    spoken.append(rest.text)
            out.append(spoken)
        return out
    return asyncio.run(run())


def test_every_clause_goes_as_soon_as_it_is_complete():
    reply = "Well, that depends on the weather, honestly. I would bring a coat, just in case."
    assert chunks(reply) == [
        ["Well, that depends on the weather,", "honestly.",
         "I would bring a coat,", "just in case."]]


def test_whole_words_with_trailing_spaces_work_too():
    assert chunks("Well, that depends on the weather, honestly. Okay.",
                  tokens=lambda r: [w + " " for w in r.split(" ")]) \
        == [["Well, that depends on the weather,", "honestly.", "Okay."]]


def test_a_short_opening_waits_for_more_words():
    assert chunks("Sure, I can help with that. What do you need?") \
        == [["Sure, I can help with that.", "What do you need?"]]


def test_numbers_are_not_clauses():
    assert chunks("It costs about 1,000 dollars a year, give or take.") \
        == [["It costs about 1,000 dollars a year,", "give or take."]]


def test_a_first_clause_of_two_words_goes_at_once():
    assert chunks("Of course, I can help with that.") \
        == [["Of course,", "I can help with that."]]


def test_a_one_word_opening_still_waits_for_more():
    assert chunks("Yes, of course, that works for me. Great.") \
        == [["Yes, of course,", "that works for me.", "Great."]]


def test_later_clauses_still_need_four_words():
    assert chunks("Of course, let me see, I think so.") \
        == [["Of course,", "let me see, I think so."]]


def test_after_the_first_sentence_two_words_are_not_enough():
    assert chunks("Okay. Yes indeed, that works.") == [["Okay.", "Yes indeed, that works."]]


def test_every_reply_and_every_interruption_starts_a_new_first_clause():
    assert chunks("Of course, that is fine.", "Of course, see you soon.") \
        == [["Of course,", "that is fine."], ["Of course,", "see you soon."]]
    out = chunks("I think that the best way, if you ask", "Sure thing, see you soon.",
                 interrupt_after=0)
    assert out[1] == ["Sure thing,", "see you soon."]


def test_a_later_sentence_splits_at_its_clauses_too():
    assert chunks("Okay. When you get home tonight, call me back, please.") \
        == [["Okay.", "When you get home tonight,", "call me back, please."]]


def test_an_interrupted_reply_does_not_carry_over():
    out = chunks("I think that the best way, if you ask", "Okay, so what would you like to do now?",
                 interrupt_after=0)
    assert out[1] == ["Okay, so what would you like to do now?"]


# Live run 4: "Sure!" waited for the next sentence's first word, 1.1 s on
# "Can you recite a joke?". A complete first sentence goes the moment it ends.


async def _released(*tokens):
    agg, out = ClauseTextAggregator(), []
    for token in tokens:
        async for a in agg.aggregate(token):
            out.append(a.text)
    return out


def test_a_one_word_first_sentence_goes_the_moment_it_ends():
    for word, mark in (("Sure", "!"), ("Yes", "."), ("Really", "?")):
        assert asyncio.run(_released(word, mark)) == [word + mark]  # no next word needed


def test_a_one_word_first_clause_ending_in_a_comma_still_waits():
    assert asyncio.run(_released("Yes", ",")) == []
    assert chunks("Yes, I am here.") == [["Yes, I am here."]]


def test_a_number_or_title_is_not_a_finished_first_sentence():
    assert asyncio.run(_released("It", " costs", " 3", ".")) == []  # 3.5 may follow
    assert asyncio.run(_released("Dr", ".")) == []


def test_only_the_first_sentence_skips_the_wait():
    assert chunks("Sure! Why did the robot cross the road? To charge.") \
        == [["Sure!", "Why did the robot cross the road?", "To charge."]]


def _words(chunk):
    return any(c.isalnum() for c in chunk)


def test_an_opening_ellipsis_never_reaches_tts_alone():
    for reply in ("... Well, I think so.", "… Well, I think so."):
        tokens = re.findall(r" ?[\w']+|[^\w\s]", reply)
        out = asyncio.run(_released(*tokens))  # no flush: what goes before the reply ends
        assert all(_words(c) for c in out), out
        assert out and out[-1].endswith("I think so."), out  # the first sentence, at its end


def test_a_trailing_ellipsis_never_reaches_tts_alone():
    for reply in ("Hmm...", "Well... maybe.", "I see… right."):
        (spoken,) = chunks(reply)
        assert spoken and all(_words(c) for c in spoken), spoken
