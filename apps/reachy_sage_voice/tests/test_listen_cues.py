"""Listening cues: still while you speak, a glance away while thinking, eyes back to reply.

Copied from Pollen's conversation app. The robot is a fake recording calls; the
moments are Pipecat's own SpeakingObserver events, fed in directly. The cue's
sleeps last until the test ends them: no test races the wall clock.
"""

import asyncio
import functools

import numpy as np
from pipecat.observers.speaking_observer import SpeakingObserver, SpeechEvent
from pipecat.observers.speaking_observer import SpeechEventKind as Kind

from reachy_sage_voice.listen_cues import ListenCues

HOLD, GLANCE = 0.02, 0.05
HERE = np.eye(4)
HERE[0, 3] = 0.01  # wherever the tracker has the head


class Mini:
    def __init__(self):
        self.calls = []

    def get_present_antenna_joint_positions(self):
        return [0.1, -0.1]

    def get_current_head_pose(self):
        return HERE.copy()

    def set_target(self, head=None, antennas=None):
        self.calls.append(("set_target", "head" if head is not None else "antennas",
                           head if head is not None else antennas))

    def goto_target(self, head, duration, body_yaw):
        self.calls.append(("goto", head, body_yaw))

    def start_head_tracking(self, weight):
        self.calls.append(("track", weight))


class Sleeps:
    """Stands in for asyncio.sleep: each sleep is parked until ``elapse`` ends it."""

    def __init__(self):
        self.parked = []

    async def __call__(self, seconds):
        done = asyncio.get_running_loop().create_future()
        self.parked.append((seconds, done))
        await done


def cues(mini):
    return ListenCues(mini, hold_s=HOLD, glance_s=GLANCE, move_s=0.01, sleep=Sleeps())


def pending(c):
    return [(s, done) for s, done in c._sleep.parked if not done.done()]


async def settled(c, within=5.0):
    """Wait until the cue in hand is parked in a sleep, or over (robot calls run on a thread)."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + within
    while not (c._later is None or c._later.done() or pending(c)):
        assert loop.time() < deadline, "the cue never settled"
        await asyncio.sleep(0.001)


async def elapse(c, seconds):
    """End the cue's pending sleep, which must be ``seconds`` long, then let it settle."""
    await settled(c)
    ((asked, done),) = pending(c)
    assert asked == seconds
    done.set_result(None)
    await settled(c)


async def say(c, *kinds):
    for kind in kinds:
        await c.on_speech_event(None, SpeechEvent(kind=kind, timestamp=0.0))


def run(test):
    """Run an async test on its own loop (the suite has no asyncio plugin)."""
    return functools.wraps(test)(lambda: asyncio.run(test()))


def names(mini):
    return [c[0] if c[0] != "track" else c for c in mini.calls]


def test_it_is_a_pipecat_speaking_observer():
    assert isinstance(cues(Mini()).observer(), SpeakingObserver)


@run
async def test_speech_onset_freezes_the_antennas_where_they_are():
    mini = Mini()
    c = cues(mini)
    await say(c, Kind.USER_SPEECH_STARTED)
    assert mini.calls == [("set_target", "antennas", [0.1, -0.1])]
    assert c.listening


@run
async def test_a_reply_within_the_hold_releases_listening_with_no_glance():
    mini = Mini()
    c = cues(mini)
    await say(c, Kind.USER_SPEECH_STARTED, Kind.USER_SPEECH_STOPPED)
    assert c.listening  # held through the short silence
    await say(c, Kind.BOT_SPEECH_STARTED)
    await settled(c)
    assert not c.listening
    assert names(mini) == ["set_target"]  # the antenna freeze, nothing else


@run
async def test_thinking_glances_away_from_the_face_then_back():
    mini = Mini()
    c = cues(mini)
    await say(c, Kind.USER_SPEECH_STARTED, Kind.USER_SPEECH_STOPPED)
    await elapse(c, HOLD)
    assert not c.listening
    # The app target is pinned where the head is, then tracking lets it show
    # through, then the head eases to a small offset: no jump at the weight change.
    assert names(mini)[1:] == ["set_target", ("track", c.weight), "goto"]
    assert np.array_equal(mini.calls[1][2], HERE)
    glance = mini.calls[3]
    assert not np.array_equal(glance[1], HERE) and glance[2] is None  # body yaw untouched
    await elapse(c, GLANCE)
    assert names(mini)[4:] == ["goto", ("track", 1.0)]  # back, and the tracker owns the head
    assert np.array_equal(mini.calls[4][1], HERE)


@run
async def test_reply_audio_looks_back_at_once():
    mini = Mini()
    c = cues(mini)
    await say(c, Kind.USER_SPEECH_STARTED, Kind.USER_SPEECH_STOPPED)
    await elapse(c, HOLD)
    await say(c, Kind.BOT_SPEECH_STARTED)
    assert names(mini)[-2:] == ["goto", ("track", 1.0)]
    await settled(c)
    assert names(mini).count(("track", 1.0)) == 1  # the glance's own return never runs


@run
async def test_speaking_again_mid_glance_looks_back_and_listens():
    mini = Mini()
    c = cues(mini)
    await say(c, Kind.USER_SPEECH_STARTED, Kind.USER_SPEECH_STOPPED)
    await elapse(c, HOLD)
    await say(c, Kind.USER_SPEECH_STARTED)
    assert c.listening
    assert ("track", 1.0) in mini.calls


@run
async def test_an_emote_owns_the_body_while_it_plays():
    mini = Mini()
    c = cues(mini)
    started, over = asyncio.Event(), asyncio.Event()

    async def emote(name, sound):
        started.set()
        await over.wait()

    play = asyncio.ensure_future(c.during(emote)("cheerful1", True))
    await started.wait()
    await say(c, Kind.USER_SPEECH_STARTED, Kind.USER_SPEECH_STOPPED)
    await elapse(c, HOLD)  # the hold runs out mid-move
    over.set()
    await play
    assert mini.calls == []  # no antenna freeze, no glance, mid-move


@run
async def test_an_emote_mid_glance_gets_the_head_back_from_the_cue():
    mini = Mini()
    c = cues(mini)
    await say(c, Kind.USER_SPEECH_STARTED, Kind.USER_SPEECH_STOPPED)
    await elapse(c, HOLD)
    before = len(mini.calls)

    async def emote(name, sound):
        pass

    await c.during(emote)("cheerful1", True)
    # Tracking weight restored, but no head goto: the move is writing the head.
    assert mini.calls[before:] == [("track", 1.0)]
    await settled(c)
    assert len(mini.calls) == before + 1


@run
async def test_with_a_turn_detector_only_the_end_of_the_turn_is_thinking():
    # Smart Turn's VAD stops on every short pause; only its turn end means "your go".
    mini = Mini()
    c = cues(mini)
    await say(c, Kind.USER_SPEECH_STARTED, Kind.USER_TURN_STARTED, Kind.USER_SPEECH_STOPPED)
    await elapse(c, HOLD)
    assert not c.listening and names(mini) == ["set_target"]  # a pause: no glance
    await say(c, Kind.USER_TURN_STOPPED)
    await elapse(c, HOLD)
    assert ("track", c.weight) in mini.calls


@run
async def test_a_robot_that_refuses_a_cue_never_stops_the_voice():
    class Broken(Mini):
        def start_head_tracking(self, weight):
            raise ConnectionError("daemon gone")

    c = cues(Broken())
    await say(c, Kind.USER_SPEECH_STARTED, Kind.USER_SPEECH_STOPPED)
    await elapse(c, HOLD)
    await elapse(c, GLANCE)
    await say(c, Kind.BOT_SPEECH_STARTED)
