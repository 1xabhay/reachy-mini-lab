"""Run Reachy's Sage voice on the robot: ``make run`` (``python -m reachy_sage_voice``).

Everything comes from the environment (see pipeline.py, plus REACHY_MINI_HOST)
and every address must be set: nothing falls back to a network default. The
motors are enabled before the robot wakes; ``wake_up()`` alone leaves the head
limp after a reboot (the pet learned this, live.py ``wake``).
"""

import asyncio
import contextlib
import logging
import os
import sys
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping

logger = logging.getLogger(__name__)

HOST = "REACHY_MINI_HOST"


def unreachable(env: Mapping[str, str], get: Callable = urllib.request.urlopen) -> list[str]:
    """Name the kernel, STT and TTS settings whose URL nothing answers at.

    Any HTTP answer counts: the kernel has no health route and turns a bare GET
    away, and a service still loading its model is up.
    """
    def root(url: str) -> str:
        url = url.rstrip("/")
        return url.removesuffix("/v1")

    urls = {"SAGE_VOICE_KERNEL_URL": env["SAGE_VOICE_KERNEL_URL"].rstrip("/"),
            "SAGE_VOICE_STT_URL": root(env["SAGE_VOICE_STT_URL"]) + "/health",
            "SAGE_VOICE_TTS_URL": root(env["SAGE_VOICE_TTS_URL"]) + "/health"}
    down = []
    for name, url in urls.items():
        try:
            with get(url, timeout=5):
                pass
        except urllib.error.HTTPError:
            pass
        except OSError:  # URLError, refused, timed out
            down.append(name)
    return down


def check(env: Mapping[str, str], required: tuple[str, ...], reach: Callable | None) -> None:
    """Exit with what to fix if a setting is missing or a service is not answering."""
    missing = [name for name in required if not env.get(name)]
    if missing:
        sys.exit("set " + ", ".join(missing) + " (see README.md, 'Settings')")
    down = (reach or unreachable)(env)
    if down:
        sys.exit("cannot reach " + ", ".join(f"{name} ({env[name]})" for name in down)
                 + ": start it (README.md, 'Quick start') or fix the URL")


def main(env: Mapping[str, str] = os.environ, robot: Callable | None = None,
         run: Callable | None = None, moves: Callable | None = None,
         reach: Callable | None = None) -> None:
    """Check the settings, connect to the robot, wake it, and run the voice until stopped."""
    from reachy_sage_voice.pipeline import REQUIRED

    check(env, (*REQUIRED, HOST), reach)
    if robot is None:
        from reachy_mini import ReachyMini

        robot = ReachyMini
    if moves is None:
        from reachy_sage_voice.kernel import load_moves as moves

    declared = moves()  # before connecting: a missing library fails fast
    with robot(host=env[HOST], connection_mode="network") as mini:
        live(mini, declared, env, run)


def app(mini, env: Mapping[str, str], stop_event: threading.Event) -> None:
    """Run the voice on a robot the daemon connected, until ``stop_event`` is set."""
    from reachy_sage_voice import kernel
    from reachy_sage_voice.pipeline import REQUIRED

    check(env, REQUIRED, None)
    live(mini, kernel.load_moves(), env, None, stop_event)


def live(mini, declared: Mapping, env: Mapping[str, str], run: Callable | None,
         stop_event: threading.Event | None = None) -> None:
    """Enable, wake and track, then run the voice until stopped."""
    if run is None:
        from reachy_sage_voice import pipeline

        run = pipeline.run
    mini.enable_motors()  # a requirement: if it fails, nothing else can move
    try:
        mini.wake_up()
    except Exception as error:  # a gesture, not a requirement
        logger.warning("could not wake up (%s) - carrying on", error)
    try:
        # The daemon's tracker is off until asked; without it no face is ever seen.
        mini.start_head_tracking(weight=1.0)
    except Exception as error:  # a reflex, not a requirement
        logger.warning("could not start face tracking (%s) - carrying on", error)

    async def play_emote(name: str, sound: bool) -> None:
        await mini.async_play_move(declared[name], sound=sound)

    async def play_sound(name: str) -> None:  # a vocal: the move's own sound alone
        mini.media.play_sound(str(declared[name].sound_path))
        await asyncio.sleep(declared[name].duration)

    presence = None
    if env.get("SAGE_VOICE_PRESENCE", "0") == "1":  # off until the kernel accepts it
        def presence() -> bool:
            return mini.get_tracked_face(wait=False).detected

    observers = []
    if env.get("SAGE_VOICE_LISTEN_CUES", "0") == "1":  # off until tuned on the robot
        from reachy_sage_voice.listen_cues import ListenCues

        cues = ListenCues(mini)
        play_emote, observers = cues.during(play_emote), [cues.observer()]

    voice = run(mini.media, play_emote, declared, env, play_sound=play_sound,
                presence=presence, observers=observers)
    asyncio.run(voice if stop_event is None else until(stop_event, voice))


async def until(stop_event: threading.Event, voice) -> None:
    """Run ``voice`` until it ends or ``stop_event`` is set, then cancel it."""
    task = asyncio.ensure_future(voice)
    stopped = asyncio.ensure_future(asyncio.to_thread(stop_event.wait))
    await asyncio.wait({task, stopped}, return_when=asyncio.FIRST_COMPLETED)
    stop_event.set()  # frees the waiting thread when the voice ended first
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task  # re-raises the voice's own failure


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
