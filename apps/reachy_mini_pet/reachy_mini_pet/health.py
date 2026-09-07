"""Deciding when a session's connection to the robot is beyond saving.

The third distinct way this has broken on real hardware, and the subtlest. A
microphone can stop delivering - the watchdog catches that. The robot's daemon
can hang - the supervisor catches that. But the daemon can also answer HTTP
perfectly while the WebSocket that actually carries commands is dead, and then
every status check is green while nothing works. The pet sat at 28% CPU
throwing a traceback every few milliseconds and no health check noticed,
because they were all watching the wrong channel.

So the signal here is not a status endpoint but the layers themselves. A module
that cannot talk to the robot is the most direct evidence there is that the
session is finished, and it costs nothing to collect.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

#: Exceptions that mean the link to the robot is gone, rather than that one
#: piece of work failed. `ConnectionError` is an `OSError`, but both are listed
#: so the intent survives a refactor.
_CONNECTION_FAILURES = (ConnectionError, OSError)


@dataclass
class SessionHealth:
    """Watches a session's failures and says when the connection has gone."""

    #: Connection failures within the window before the session is declared
    #: dead. One is a blip; tearing down on a blip is just churn.
    tolerance: int = 3
    #: How recent a failure has to be to still count against the link. Judging
    #: on a *window* rather than a consecutive run is deliberate: when the
    #: WebSocket died, the layer that sends commands failed on every step while
    #: the layer reading audio kept succeeding, and any success-resets-the-count
    #: rule would have been held open for ever by the healthy one.
    window_s: float = 15.0
    #: Write at most one report per this many failures. A module failing every
    #: five milliseconds must not produce a traceback every five milliseconds.
    report_every: int = 50

    connection_errors: int = 0
    other_errors: int = 0
    reason: str = ""

    _failures: deque[float] = field(default_factory=deque, repr=False)
    _seen: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        """Reject a tolerance that would tear down the session on one blip."""
        if self.tolerance < 1:
            raise ValueError("tolerance must be at least 1")

    def record_success(self, now: float) -> None:
        """Note that a layer did its work.

        Deliberately does *not* clear the failure record. A microphone that
        still delivers proves nothing about a command channel that has gone.
        """

    def record_error(self, error: BaseException, now: float) -> None:
        """Note that a layer failed, and judge whether the link is the cause."""
        if isinstance(error, _CONNECTION_FAILURES):
            self.connection_errors += 1
            self._failures.append(now)
            self._forget_old(now)
            if len(self._failures) >= self.tolerance and not self.reason:
                self.reason = str(error) or type(error).__name__
        else:
            # Whisper falling over is not the robot going away.
            self.other_errors += 1

    def is_broken(self) -> bool:
        """Whether the connection should be considered gone."""
        return len(self._failures) >= self.tolerance

    def _forget_old(self, now: float) -> None:
        """Drop failures too old to say anything about the link now."""
        cutoff = now - self.window_s
        while self._failures and self._failures[0] < cutoff:
            self._failures.popleft()

    def should_report(self) -> bool:
        """Whether this failure is the one worth writing to the log."""
        report = self._seen % self.report_every == 0
        self._seen += 1
        return report
