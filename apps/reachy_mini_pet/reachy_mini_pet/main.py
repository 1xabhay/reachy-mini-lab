"""The app the robot launches: an always-on desk pet, entirely local.

This module is deliberately thin. It is the entry point the Reachy Mini daemon
discovers and starts, and nothing more - the behaviour lives in small pieces
that can be used, tested and replaced one at a time:

    bus.py         how the layers talk without knowing about each other
    events.py      the whole vocabulary they share
    layers.py      vision, hearing, transcription, mind, voice - one job each
    conductor.py   when to listen, when to speak, when you interrupted it
    live.py        the adapters binding those to this robot and these models
    supervisor.py  keeping the pet alive when the robot goes away
    watchdog.py    noticing a sense that has quietly stopped

Every layer runs alone and any combination runs together, so the smallest
useful change stays small. Swapping the language model, the voice or the
detector means writing one adapter in `live.py` and touching nothing else;
adding a sense means writing one `Module` that publishes to the bus.

See CONTRIBUTING.md for the shape of a new layer.
"""

from __future__ import annotations

import threading

from reachy_mini import ReachyMini, ReachyMiniApp

from .live import main as run_from_terminal
from .live import session


class ReachyMiniPet(ReachyMiniApp):
    """An always-on pet: watches you, listens, answers, reacts with its body.

    Nothing leaves the machine - the voice activity detector, the speech
    recognition, the language model and the voice all run here.
    """

    #: The app reads no camera frames of its own (face tracking is the daemon's
    #: job), so a video stream is not needed. Set "gstreamer_no_video" to save
    #: CPU on the wireless unit if your robot's daemon allows it.
    request_media_backend: str | None = None

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        """Hold a conversation until the daemon asks the app to stop.

        The daemon owns the connection and hands it over already open, so this
        runs a single session and lets the daemon handle restarts. Run from a
        terminal instead and the pet supervises its own connection, surviving
        the robot going away and coming back - see `live.serve`.
        """
        session(reachy_mini, stop_event)


def main() -> None:
    """Run the pet from a terminal, supervising its own connection."""
    run_from_terminal()


if __name__ == "__main__":
    main()
