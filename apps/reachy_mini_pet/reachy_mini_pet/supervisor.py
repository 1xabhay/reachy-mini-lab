"""Surviving the robot dying underneath you.

The watchdog handles a sense going quiet. This handles the worse case: the
robot itself stopping. Its daemon hung, every connection timed out, and the pet
had nothing left to talk to - so it was simply gone until a person noticed and
power-cycled the hardware. For something meant to sit in a room for weeks,
"until a person notices" is not a recovery strategy.

The job is to keep trying without making things worse: retry for ever, because
the robot may be plugged back in at any moment, but back off so a dead robot is
not hammered, and forget the backoff the instant it answers.

Like the rest of the decision-making here it takes `now` rather than reading a
clock, so a week of outage is tested in microseconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class State(Enum):
    """Where the robot stands, as far as the pet can tell."""

    UNKNOWN = "unknown"
    CONNECTED = "connected"
    LOST = "lost"


@dataclass
class Supervisor:
    """Decides when to try the robot again, and reports when things change."""

    #: How long to wait after the first failure.
    initial_backoff_s: float = 1.0
    #: The longest it will ever wait. It must keep trying: a robot that has been
    #: dead all night may be switched on while nobody is watching.
    max_backoff_s: float = 30.0
    #: How long a session has to last before it counts as having worked. A
    #: session that dies sooner never really started, so it backs off like any
    #: other failure - otherwise a connect that always fails immediately (a
    #: missing model file, say) spins about once a second for ever, reloading
    #: every model each time.
    min_stable_s: float = 10.0

    state: State = State.UNKNOWN
    backoff: float = field(default=0.0)
    reconnections: int = 0

    _lost_at: float | None = field(default=None, repr=False)
    _connected_at: float | None = field(default=None, repr=False)
    _last_attempt: float | None = field(default=None, repr=False)
    _pending: State | None = field(default=None, repr=False)
    _ever_connected: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        """Start with the first backoff already loaded."""
        self.backoff = self.initial_backoff_s

    def connected(self, now: float) -> None:
        """Record that the robot answered."""
        if self.state is not State.CONNECTED:
            self._pending = State.CONNECTED
            if self._ever_connected:
                self.reconnections += 1
        self.state = State.CONNECTED
        self._ever_connected = True
        self._connected_at = now
        self._lost_at = None
        self._last_attempt = None

    def failed(self, now: float) -> None:
        """Record that the robot did not answer, or that a session died.

        What decides the backoff is whether the last session actually *lasted*,
        not whether the connection succeeded. A session that collapses on
        startup has bought us nothing and must be slowed down exactly like a
        robot that never answered.
        """
        lasted = None if self._connected_at is None else now - self._connected_at
        worked = lasted is not None and lasted >= self.min_stable_s

        if worked:
            # It ran properly and then something went wrong: a fresh fault,
            # worth retrying at once.
            self.backoff = self.initial_backoff_s
        elif self.state is State.LOST or self._connected_at is not None:
            # Either still absent, or it "connected" and fell straight over.
            self.backoff = min(self.backoff * 2.0, self.max_backoff_s)

        if self.state is not State.LOST:
            self._pending = State.LOST
            self._lost_at = now
        self.state = State.LOST
        self._connected_at = None
        self._last_attempt = now

    def should_retry(self, now: float) -> bool:
        """Whether to try the robot again now."""
        if self.state is State.CONNECTED:
            return False
        if self._last_attempt is None:
            return True  # nothing has been tried yet
        if now - self._last_attempt < self.backoff:
            return False
        self._last_attempt = now
        return True

    def lost_for(self, now: float) -> float:
        """How long the robot has been gone, for something worth saying out loud."""
        if self._lost_at is None:
            return 0.0
        return max(0.0, now - self._lost_at)

    def take_transition(self) -> State | None:
        """Return a state change once, so it is logged once and not every loop."""
        pending, self._pending = self._pending, None
        return pending
