"""The voice as a Reachy Mini app: the daemon launches ``python -m reachy_sage_voice.main``.

Settings come from the environment as for ``make run`` (README.md, "Settings"),
except the robot's address: the daemon connects to the robot itself.
"""

import logging
import os
import threading

from reachy_mini import ReachyMini, ReachyMiniApp


class ReachySageVoice(ReachyMiniApp):
    """Reachy's real-time voice, with the Sage kernel as its mind."""

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        """Run the voice on the robot until the daemon stops the app."""
        from reachy_sage_voice.__main__ import app

        app(reachy_mini, os.environ, stop_event)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = ReachySageVoice()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
