"""Noticing when a sense has quietly died.

A microphone that stops delivering looks exactly like a quiet room, and a
camera that stops looks exactly like an empty one. Nothing downstream can tell
the difference: the threads keep running, the queues stay empty, and the pet
sits there deaf while every part of it reports that it is fine. This is the
failure that took it down after forty minutes of working perfectly.

So something has to hold the opinion that silence for *too* long is not
silence, it is a fault. That is all this does - and because it takes `now`
rather than reading a clock, the whole of it is testable in microseconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Watchdog:
    """Watches one sense, and says when it has stopped delivering."""

    #: Silence longer than this is a fault, not a quiet room. Set it well above
    #: the longest legitimate gap - a mic that idles between callbacks is fine.
    timeout_s: float = 5.0
    #: Wait this long between recovery attempts. Restarting a wedged device in
    #: a tight loop makes things worse and floods the log.
    backoff_s: float = 10.0
    #: When the watch began, for a sense that never delivered anything at all.
    now: float = 0.0

    recoveries: int = 0
    _last_fed: float | None = field(default=None, repr=False)
    _last_recovery: float | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Reject a timeout that would call every pause a hardware fault."""
        if self.timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        self._last_fed = self.now

    def feed(self, now: float) -> None:
        """Record that the sense delivered something.

        This also clears the backoff: something arriving proves the device is
        alive, so if it dies again that is a fresh fault and worth acting on
        at once, not something to sit out the old backoff for.
        """
        self._last_fed = now
        self._last_recovery = None

    def silence(self, now: float) -> float:
        """How long it has been since anything arrived."""
        since = self.now if self._last_fed is None else self._last_fed
        return max(0.0, now - since)

    def stalled(self, now: float) -> bool:
        """Whether this sense has been quiet long enough to be considered dead."""
        return self.silence(now) >= self.timeout_s

    def should_recover(self, now: float) -> bool:
        """Whether the caller should restart the device now.

        True at most once per stall, and never more often than the backoff, so
        a device that cannot be revived is not hammered.
        """
        if not self.stalled(now):
            return False
        if self._last_recovery is not None and now - self._last_recovery < self.backoff_s:
            return False
        self._last_recovery = now
        self.recoveries += 1
        return True
