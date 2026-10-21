"""Listening cues, after Pollen's conversation app: still while you speak, a glance while thinking.

- Speech onset freezes the antennas where they are. ``listening`` stays True
  until the reply starts, or the user has been quiet for ``hold_s``.
- If the turn has ended and no reply audio has started within ``hold_s``, the
  head glances a little away, then back after ``glance_s``; reply audio or new
  speech brings it back at once. With a turn detector in the pipeline (Smart
  Turn), only its turn end counts, not every pause the VAD hears.
- The daemon's face tracker owns the head (``start_head_tracking(weight=1.0)``
  in ``__main__``), and at weight 1 app head targets are ignored. The glance
  uses the SDK's documented blend instead: pin the app target where the head
  is, lower the weight so it shows through, ease it by a small offset. Then
  ease back and give the tracker the whole head again.
- A playing emote owns the body: no cue starts during one, and one that starts
  mid-glance only gets the tracking weight back, since the move is writing the
  head.

Robot calls run one at a time, in order, off the event loop: a cancelled cue
can never land its weight change after the call that undoes it. A failed call
costs the cue, never the voice. ``make run`` turns this on with
SAGE_VOICE_LISTEN_CUES=1.
"""

import asyncio
import functools
import logging
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor

from pipecat.observers.speaking_observer import SpeakingObserver
from pipecat.observers.speaking_observer import SpeechEventKind as Kind
from reachy_mini.utils import create_head_pose

logger = logging.getLogger(__name__)


class ListenCues:
    """Body cues for the floor changing hands, driven by Pipecat's speaking events."""

    def __init__(self, mini, hold_s: float = 0.4, glance_s: float = 1.0, move_s: float = 0.3,
                 weight: float = 0.5, glance=None, sleep=asyncio.sleep) -> None:
        """``mini`` is a connected ReachyMini; the defaults are first guesses, to tune live.

        ``weight`` is the tracking weight while glancing: the share of the head
        the face keeps. ``glance`` is the offset, in the head's frame, from
        wherever the head is; with no face tracked it shows in full. ``sleep`` is
        for tests.
        """
        self.mini, self.hold_s, self.glance_s, self.move_s = mini, hold_s, glance_s, move_s
        self.weight, self._sleep = weight, sleep
        self.glance = create_head_pose(pitch=-6, yaw=12) if glance is None else glance
        self.listening = False
        self._emotes = 0  # playing now
        self._turns = False  # a turn detector reports turn ends
        self._later: asyncio.Future | None = None
        self._left_from = None  # the head pose a glance left from, while glancing
        self._robot = ThreadPoolExecutor(max_workers=1, thread_name_prefix="listen-cues")

    def observer(self) -> SpeakingObserver:
        """Return a pipeline observer feeding this, for ``PipelineWorker(observers=...)``."""
        observer = SpeakingObserver()
        observer.add_event_handler("on_speech_event", self.on_speech_event)
        return observer

    def during(self, play_emote: Callable[..., Awaitable[None]]) -> Callable[..., Awaitable[None]]:
        """Wrap ``play_emote`` so no cue moves the body while a move plays."""
        @functools.wraps(play_emote)
        async def play(*args, **kwargs):
            self._emotes += 1
            self._cancel()
            try:
                await self._look_back()
                await play_emote(*args, **kwargs)
            finally:
                self._emotes -= 1
        return play

    async def on_speech_event(self, _observer, event) -> None:
        """React to one SpeakingObserver moment. State changes first, then the robot."""
        kind = event.kind
        if kind == Kind.USER_SPEECH_STARTED:
            self._cancel()
            self.listening = True
            await self._look_back()
            if not self._emotes:
                antennas = await self._call("get_present_antenna_joint_positions")
                if antennas is not None:
                    await self._call("set_target", antennas=antennas)
        elif kind == Kind.USER_TURN_STARTED:
            self._turns = True
        elif kind == Kind.USER_SPEECH_STOPPED:
            self._later = asyncio.ensure_future(self._after_hold(glance=not self._turns))
        elif kind == Kind.USER_TURN_STOPPED:
            self._cancel()
            self._later = asyncio.ensure_future(self._after_hold(glance=True))
        elif kind == Kind.BOT_SPEECH_STARTED:
            self._cancel()
            self.listening = False
            await self._look_back()

    async def _after_hold(self, glance: bool) -> None:
        await self._sleep(self.hold_s)
        self.listening = False
        if not glance or self._emotes:
            return
        here = await self._call("get_current_head_pose")
        if here is None:
            return
        self._left_from = here
        await self._call("set_target", head=here)
        await self._call("start_head_tracking", weight=self.weight)
        await self._call("goto_target", head=here @ self.glance, duration=self.move_s,
                         body_yaw=None)
        await self._sleep(self.glance_s)
        await asyncio.shield(self._look_back())  # once begun, a return always finishes

    async def _look_back(self) -> None:
        """End a glance, if one is on: back to the face, the tracker owning the head."""
        here, self._left_from = self._left_from, None
        if here is None:
            return
        if not self._emotes:
            await self._call("goto_target", head=here, duration=self.move_s, body_yaw=None)
        await self._call("start_head_tracking", weight=1.0)

    def _cancel(self) -> None:
        if self._later is not None and self._later is not asyncio.current_task():
            self._later.cancel()
        self._later = None

    async def _call(self, name: str, **kwargs):
        """Run one robot call in order with the others; None if it failed."""
        method = getattr(self.mini, name)
        try:
            return await asyncio.get_running_loop().run_in_executor(
                self._robot, functools.partial(method, **kwargs))
        except Exception as error:  # a cue, not a requirement
            logger.warning("listening cue %s failed (%s)", name, error)
            return None
