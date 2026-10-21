"""The pure policy of Reachy's Sage voice: when to yield, what not to answer, what not to say.

Times are plain seconds passed in, so every rule here runs in microseconds
with no clock, no audio and no Pipecat.
"""

from reachy_sage_voice.interruption import InterruptionTracker
from reachy_sage_voice.speech import (
    addressed,
    is_self_echo,
    move_sound,
    speakable,
    text_similarity,
    trim_echo,
)

# --- interruption policy (ported from an earlier voice bot) ---------------


def test_no_interruption_before_the_bot_has_made_a_sound():
    tracker = InterruptionTracker()
    assert tracker.user_onset(0.0, "you") is False


def test_bot_keeps_the_floor_for_its_first_600_ms():
    tracker = InterruptionTracker()
    tracker.bot_started(0.0)
    assert tracker.user_onset(0.599, "you") is False


def test_an_onset_after_the_floor_interrupts():
    tracker = InterruptionTracker()
    tracker.bot_started(0.0)
    assert tracker.user_onset(0.6, "you") is True
    assert tracker.interruptions == 1


def test_only_one_interruption_per_response():
    tracker = InterruptionTracker()
    tracker.bot_started(10.0)
    assert tracker.user_onset(11.0, "you") is True
    assert tracker.user_onset(11.5, "you") is False
    assert tracker.bot_started(12.0) is False  # next sentence cannot re-arm it


def test_the_next_response_can_be_interrupted_again():
    tracker = InterruptionTracker()
    tracker.bot_started(10.0)
    tracker.user_onset(11.0, "you")
    tracker.assistant_turn_stopped(11.2)
    tracker.bot_started(20.0)
    assert tracker.user_onset(21.0, "you") is True


def test_onset_inside_the_floor_is_caught_at_the_next_sentence_start():
    tracker = InterruptionTracker()
    tracker.bot_started(10.0)
    assert tracker.user_onset(10.3, "you") is False
    tracker.bot_stopped(11.0)  # gap between sentences, response still open
    assert tracker.bot_started(11.2) is True


def test_onset_in_a_gap_between_sentences_still_interrupts():
    tracker = InterruptionTracker()
    tracker.bot_started(10.0)
    tracker.bot_stopped(11.0)
    assert tracker.user_onset(11.4, "you") is True


def test_a_speaker_who_stopped_does_not_interrupt_the_next_sentence():
    tracker = InterruptionTracker()
    tracker.bot_started(10.0)
    tracker.user_onset(10.2, "you")
    tracker.user_offset(10.4, "you")
    assert tracker.bot_started(11.0) is False


def test_an_onset_older_than_3_s_is_a_lost_stop_not_a_speaker():
    def yields_at_second_sentence(onset: float) -> bool:
        tracker = InterruptionTracker()
        tracker.user_onset(onset, "you")  # its stop frame never arrives
        tracker.bot_started(3.5)
        tracker.bot_stopped(4.1)
        return tracker.bot_started(4.2)

    assert yields_at_second_sentence(1.5) is True  # 2.7 s ago: still talking
    assert yields_at_second_sentence(1.0) is False  # 3.2 s ago: a lost stop


def test_an_onset_from_the_future_is_ignored():
    tracker = InterruptionTracker()
    tracker.user_onset(50.0, "you")  # a clock we do not share
    tracker.bot_started(10.0)
    assert tracker.bot_started(11.0) is False


def test_talk_over_is_measured_from_onset_to_bot_silence():
    tracker = InterruptionTracker()
    tracker.bot_started(10.0)
    tracker.user_onset(11.0, "you")
    tracker.bot_stopped(11.16)
    assert [round(ms) for ms in tracker.talkovers_ms] == [160]


def test_floor_is_a_calibration_knob():
    tracker = InterruptionTracker(min_bot_speech_ms=0)
    tracker.bot_started(10.0)
    assert tracker.user_onset(10.0, "you") is True


# --- self-echo guard (ported from an earlier voice bot) -----------


def test_similarity_is_word_set_jaccard_ignoring_case_and_punctuation():
    assert text_similarity("Hello, there!", "hello there") == 1.0
    assert text_similarity("a b", "b c") == 1 / 3
    assert text_similarity("", "anything") == 0.0


def test_reachy_hearing_its_own_reply_is_not_a_user_turn():
    said = ["I think the weather is lovely today."]
    assert is_self_echo("the weather is lovely today", said)


def test_a_real_reply_is_a_user_turn():
    said = ["I think the weather is lovely today."]
    assert not is_self_echo("what time is it", said)


def test_echo_threshold_is_0_65():
    # 2 shared of 3 distinct words: 0.667, an echo; 3 of 5: 0.6, a person.
    assert is_self_echo("a b", ["a b c"])
    assert not is_self_echo("a b c", ["a b c d e"])


# --- never spoken: emoji and emotion labels ----------------------------------

VOCAB = ("cheerful1", "thoughtful1")


def test_emoji_become_body_language_not_speech():
    assert speakable("Nice to see you 😊!", VOCAB) == ("Nice to see you!", "cheerful1")


def test_emoji_the_body_cannot_show_are_still_not_spoken():
    assert speakable("Sleep well 😴", VOCAB) == ("Sleep well", "")


def test_an_emotion_label_is_never_spoken():
    assert speakable("Hmm. EMOTION: thoughtful1", VOCAB) == ("Hmm.", "thoughtful1")


def test_an_unknown_emotion_label_is_dropped_too():
    assert speakable("emotion: jubilant7 Hello.", VOCAB) == ("Hello.", "")


# --- "was that meant for me?" (a hook, not wired yet) -------------------------

def test_saying_its_name_addresses_the_robot():
    assert addressed("Hey Reachy, what time is it?")
    assert addressed("what do you think, reachy")


def test_a_name_inside_another_word_does_not():
    assert not addressed("that's out of reachyness")  # whole words only


def test_looking_at_the_robot_addresses_it():
    assert addressed("what time is it", looking=True)
    assert not addressed("what time is it", looking=False)
    assert not addressed("what time is it")  # no gaze known, no name: not for me


def test_the_name_is_a_calibration_knob():
    assert addressed("Richie, are you there?", names=("reachy", "richie"))


# --- a move's own sound, only when there is little to say ---------------------


def test_a_move_sounds_when_there_is_nothing_to_say():
    assert move_sound("", voice_busy=False)


def test_a_move_sounds_beside_up_to_three_words():
    assert move_sound("Yes, of course!", voice_busy=False)


def test_four_words_are_enough_to_mute_it():
    assert not move_sound("Yes, of course, friend.", voice_busy=False)


def test_emoji_are_not_words():
    assert move_sound("Yay 🎉", voice_busy=False)


def test_a_reply_not_yet_written_mutes_it():
    assert not move_sound(None, voice_busy=False)


def test_a_sound_never_overlaps_the_voice():
    assert not move_sound("", voice_busy=True)


# --- a transcript that starts with the robot's own last words -----------------
# Live run 3: the robot said "Yes, I'm here! Let's talk." and the next
# transcript was "Let's talk. Alright, let's see. Improvements."

SAID = ["Yes, I'm here! Let's talk."]


def test_a_leading_echo_of_the_robots_last_words_is_trimmed():
    assert trim_echo("Let's talk. Alright, let's see. Improvements.", SAID) \
        == "Alright, let's see. Improvements."


def test_nothing_but_the_echo_leaves_nothing():
    assert trim_echo("let's talk", SAID) == ""


def test_the_longest_matching_tail_is_trimmed():
    assert trim_echo("I'm here, let's talk. What now?", SAID) == "What now?"


def test_one_shared_word_is_not_an_echo():
    assert trim_echo("Talk to me about the weather.", SAID) == "Talk to me about the weather."


def test_words_from_the_middle_of_its_speech_are_not_trimmed():
    assert trim_echo("Yes, I'm cold.", SAID) == "Yes, I'm cold."


def test_any_recent_speech_can_be_the_echo():
    said = ["Good morning to you.", "Shall we start?"]
    assert trim_echo("to you. Yes please.", said) == "Yes please."


# --- tool syntax is never spoken ------------------------------------------------
# Live run 3: the kernel's reply carried "(emote: )" and the robot read it out.


def test_an_emote_call_written_into_the_reply_is_not_spoken():
    assert speakable("Great news (emote: cheerful1)", VOCAB) == ("Great news", "")
    assert speakable("(emote: )", VOCAB) == ("", "")


def test_bracketed_tool_syntax_is_not_spoken_either():
    assert speakable("[emote cheerful1] Hello there.", VOCAB) == ("Hello there.", "")


def test_any_bracket_naming_a_tool_or_a_move_goes():
    assert speakable("Fine (I'd emote here) thanks.", VOCAB) == ("Fine thanks.", "")
    assert speakable("Hi (cheerful1)!", VOCAB) == ("Hi!", "")


def test_ordinary_brackets_are_spoken():
    assert speakable("It costs (about) five pounds.", VOCAB)[0] == "It costs (about) five pounds."
