"""Checks for the conductor: the pipeline's entire decision-making, pure.

The realtime pipeline is threads moving bytes, which is untestable by nature.
So none of the decisions live there. The conductor is a synchronous state
machine - frames and events in, actions out, with `now` passed in rather than
read from a clock - and it holds every rule about when to listen, when to
speak, and when you have interrupted.

That makes barge-in, which is otherwise a nest of timing races, assertable at
an exact frame index with no threads and no sleeps.
"""

from hypothesis import given, settings
from hypothesis import strategies as st
from reachy_mini_pet.conductor import (
    Conductor,
    Duck,
    Resume,
    Speak,
    StopPlayback,
    Transcribe,
)

FRAME_S = 0.02  # the robot's real frame: 320 samples at 16 kHz


def feed(conductor, voiced, frames, start=0.0):
    """Feed n frames of one kind, returning every action and when it fired."""
    fired = []
    for i in range(frames):
        now = start + i * FRAME_S
        for action in conductor.feed_audio(voiced=voiced, now=now):
            fired.append((i, action))
    return fired


def only(actions, kind):
    """Just the actions of one kind."""
    return [a for _, a in actions if isinstance(a, kind)]


# ------------------------------------------------------------ listening


def test_a_quiet_room_produces_nothing():
    c = Conductor()
    assert feed(c, voiced=False, frames=500) == []


def test_a_full_utterance_is_sent_for_transcription():
    c = Conductor(min_speech_s=0.2, hangover_s=0.4)
    fired = feed(c, voiced=True, frames=20)                       # 0.4s of speech
    fired += feed(c, voiced=False, frames=25, start=0.4)          # 0.5s of silence

    sent = only(fired, Transcribe)
    assert len(sent) == 1
    assert sent[0].turn == 1


def test_the_utterance_is_sent_on_the_exact_frame_the_hangover_expires():
    """The thing wall-clock tests can never pin down."""
    c = Conductor(min_speech_s=0.2, hangover_s=0.4)
    feed(c, voiced=True, frames=20)
    fired = feed(c, voiced=False, frames=40, start=0.4)

    at = [i for i, a in fired if isinstance(a, Transcribe)]
    assert at == [19]  # 20 silent frames = 0.4s; fires on the 20th (index 19)


def test_a_cough_is_not_an_utterance():
    c = Conductor(min_speech_s=0.3, hangover_s=0.4)
    fired = feed(c, voiced=True, frames=5)                        # 0.1s
    fired += feed(c, voiced=False, frames=40, start=0.1)

    assert only(fired, Transcribe) == []


def test_a_pause_shorter_than_the_hangover_does_not_split_a_sentence():
    c = Conductor(min_speech_s=0.2, hangover_s=0.5)
    fired = feed(c, voiced=True, frames=15)
    fired += feed(c, voiced=False, frames=10, start=0.3)          # 0.2s pause
    fired += feed(c, voiced=True, frames=15, start=0.5)
    fired += feed(c, voiced=False, frames=40, start=0.8)

    assert len(only(fired, Transcribe)) == 1


# ------------------------------------------------------------- speaking


def test_a_reply_is_spoken():
    c = Conductor()
    actions = c.on_reply(["Hello there.", "How are you?"], turn=1, now=1.0)

    assert [a.text for a in actions if isinstance(a, Speak)] == [
        "Hello there.",
        "How are you?",
    ]


def test_a_reply_for_an_abandoned_turn_is_never_spoken():
    """The user moved on. Speaking the old answer is the classic bug."""
    c = Conductor()
    c.interrupt(now=1.0)                       # turn advances to 2

    assert c.on_reply(["stale answer"], turn=1, now=1.1) == []


# -------------------------------------------------------------- barge-in


def test_a_brief_noise_while_speaking_only_ducks():
    """Do not stop dead on a single frame - that makes it twitchy."""
    c = Conductor(barge_in_s=0.3, tail_gate_s=0.0)
    c.on_reply(["talking"], turn=1, now=0.0)
    c.on_playback_started(turn=1, now=0.0)

    fired = feed(c, voiced=True, frames=5, start=1.0)             # 0.1s

    assert only(fired, Duck)
    assert only(fired, StopPlayback) == []


def test_sustained_speech_while_speaking_interrupts():
    c = Conductor(barge_in_s=0.3, tail_gate_s=0.0)
    c.on_reply(["talking"], turn=1, now=0.0)
    c.on_playback_started(turn=1, now=0.0)

    fired = feed(c, voiced=True, frames=20, start=1.0)            # 0.4s

    assert only(fired, StopPlayback)
    assert c.turn == 2                                            # a new turn begins


def test_interrupting_twice_does_not_stop_twice():
    c = Conductor(barge_in_s=0.3, tail_gate_s=0.0)
    c.on_reply(["talking"], turn=1, now=0.0)
    c.on_playback_started(turn=1, now=0.0)

    fired = feed(c, voiced=True, frames=60, start=1.0)

    assert len(only(fired, StopPlayback)) == 1


def test_its_own_tail_does_not_count_as_an_interruption():
    """The last moments of its own voice must not read as you talking."""
    c = Conductor(barge_in_s=0.1, tail_gate_s=0.3)
    c.on_reply(["talking"], turn=1, now=0.0)
    c.on_playback_started(turn=1, now=0.0)
    c.on_playback_finished(turn=1, now=1.0)

    fired = feed(c, voiced=True, frames=10, start=1.0)            # inside the gate

    assert only(fired, StopPlayback) == []


def test_speech_after_the_tail_gate_is_you():
    c = Conductor(min_speech_s=0.1, hangover_s=0.3, tail_gate_s=0.3)
    c.on_reply(["talking"], turn=1, now=0.0)
    c.on_playback_started(turn=1, now=0.0)
    c.on_playback_finished(turn=1, now=1.0)

    fired = feed(c, voiced=True, frames=15, start=1.4)            # past the gate
    fired += feed(c, voiced=False, frames=25, start=1.7)

    assert only(fired, Transcribe)


def test_only_what_was_actually_spoken_is_remembered():
    """Interrupted mid-reply, the pet must not believe it said the rest."""
    c = Conductor()
    c.on_reply(["First part.", "Second part.", "Third part."], turn=1, now=0.0)
    c.on_playback_started(turn=1, now=0.0)
    c.on_spoken("First part.", turn=1, now=0.5)
    c.interrupt(now=1.0)

    assert c.spoken_text(turn=1) == "First part."


def test_a_noise_that_stops_lets_it_carry_on_talking():
    """A false alarm must not leave the pet permanently quiet."""
    c = Conductor(barge_in_s=0.5, tail_gate_s=0.0)
    c.on_reply(["talking"], turn=1, now=0.0)
    c.on_playback_started(turn=1, now=0.0)

    ducked = feed(c, voiced=True, frames=3, start=1.0)
    resumed = feed(c, voiced=False, frames=3, start=1.06)

    assert only(ducked, Duck)
    assert only(resumed, Resume)
    assert c.speaking is True


def test_a_monologue_is_cut_rather_than_buffered_for_ever():
    """A detector stuck on must not allocate until the process dies."""
    c = Conductor(min_speech_s=0.1, max_speech_s=1.0, frame_s=FRAME_S)
    fired = feed(c, voiced=True, frames=200)

    assert only(fired, Transcribe)
    assert c.pending_frames <= c.max_speech_frames


def test_playback_starting_on_an_abandoned_turn_is_ignored():
    """The interruption already happened; that audio is never going to play."""
    c = Conductor()
    c.interrupt(now=1.0)
    c.on_playback_started(turn=1, now=1.1)

    assert c.speaking is False


# ------------------------------------------------------------ invariants


@given(
    script=st.lists(st.booleans(), min_size=1, max_size=300),
    interrupts=st.lists(st.integers(min_value=0, max_value=299), max_size=5),
)
@settings(deadline=None, max_examples=100)
def test_the_turn_counter_only_ever_advances(script, interrupts):
    """Everything downstream drops stale work by comparing this integer."""
    c = Conductor()
    seen = c.turn
    for i, voiced in enumerate(script):
        if i in interrupts:
            c.interrupt(now=i * FRAME_S)
        c.feed_audio(voiced=voiced, now=i * FRAME_S)
        assert c.turn >= seen
        seen = c.turn


@given(script=st.lists(st.booleans(), min_size=1, max_size=300))
@settings(deadline=None, max_examples=100)
def test_no_action_ever_carries_a_stale_turn(script):
    c = Conductor()
    for i, voiced in enumerate(script):
        for action in c.feed_audio(voiced=voiced, now=i * FRAME_S):
            assert action.turn <= c.turn


@given(script=st.lists(st.booleans(), min_size=1, max_size=400))
@settings(deadline=None, max_examples=100)
def test_the_pending_utterance_can_never_grow_without_bound(script):
    """It runs for weeks. A stuck detector must not eat the memory."""
    c = Conductor(max_speech_s=2.0)
    for i, voiced in enumerate(script):
        c.feed_audio(voiced=voiced, now=i * FRAME_S)
        assert c.pending_frames <= c.max_speech_frames + 1


# ------------------------------------------------ how much speech, not how long

# The gate originally measured the *span* from first voiced frame to last, which
# is not the same as the amount of speech in it. Two isolated clicks 0.4s apart
# spanned 0.4s and passed, so Whisper was handed a second of near-silence - the
# regime where it hallucinates about half the time. The pet was manufacturing
# its own phantom transcripts, not merely failing to filter them.


def test_two_clicks_far_apart_are_not_an_utterance():
    """A door and a mug on a desk should never reach transcription."""
    c = Conductor(min_speech_s=0.3, hangover_s=0.6)
    fired = feed(c, voiced=True, frames=1)                      # click
    fired += feed(c, voiced=False, frames=19, start=0.02)       # 0.38s of nothing
    fired += feed(c, voiced=True, frames=1, start=0.40)         # click
    fired += feed(c, voiced=False, frames=40, start=0.42)

    assert only(fired, Transcribe) == [], "two clicks are not speech"


def test_a_sprinkle_of_noise_across_a_long_gap_is_not_an_utterance():
    c = Conductor(min_speech_s=0.3, hangover_s=0.5)
    fired = []
    now = 0.0
    for _ in range(6):                                          # 6 lone frames,
        fired += feed(c, voiced=True, frames=1, start=now)      # 0.12s of voice
        fired += feed(c, voiced=False, frames=10, start=now + 0.02)
        now += 0.22

    assert only(fired, Transcribe) == []


def test_continuous_speech_of_the_same_span_still_passes():
    """The gate must not have become deaf - this is real speech."""
    c = Conductor(min_speech_s=0.3, hangover_s=0.6)
    fired = feed(c, voiced=True, frames=20)                     # 0.4s of voice
    fired += feed(c, voiced=False, frames=40, start=0.40)

    assert len(only(fired, Transcribe)) == 1


def test_speech_with_a_natural_pause_in_it_still_passes():
    """Real sentences have gaps; the total voiced amount is what matters."""
    c = Conductor(min_speech_s=0.3, hangover_s=0.6)
    fired = feed(c, voiced=True, frames=10)                     # 0.2s
    fired += feed(c, voiced=False, frames=5, start=0.20)        # 0.1s pause
    fired += feed(c, voiced=True, frames=10, start=0.30)        # 0.2s more
    fired += feed(c, voiced=False, frames=40, start=0.50)

    assert len(only(fired, Transcribe)) == 1


def test_the_utterance_reports_how_much_of_it_was_speech():
    """The transcriber wants to know, so it can refuse a mostly-silent clip."""
    c = Conductor(min_speech_s=0.2, hangover_s=0.4)
    fired = feed(c, voiced=True, frames=15)
    fired += feed(c, voiced=False, frames=25, start=0.30)

    sent = only(fired, Transcribe)[0]
    assert sent.voiced_frames == 15
    assert sent.frames > sent.voiced_frames
