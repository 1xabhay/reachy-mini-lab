"""Turning the body, because the head alone cannot reach.

The daemon's face tracker aims the head and only the head - `body_yaw` is
never part of its aim - so the whole system is capped at the head's yaw range.
Once you pass that, it cannot keep you centred, you drift to the edge of the
frame, and the detector loses you. That is one fault, not two.

This supplies the missing joint. The head keeps doing the fast, fine work; the
body swings underneath it when you have gone further than the neck can follow,
which is how a person watches someone cross a room. It also keeps turning for
a moment after losing you, towards wherever you went, so walking out of frame
is recoverable instead of final.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class BodyYawFollower:
    """Decide where the body should point, given where your face is."""

    #: Face offsets smaller than this are the head's job. Without a dead zone
    #: the body hunts continuously around centre, which reads as twitchy.
    deadzone: float = 0.20
    #: Radians of body yaw per unit of face offset beyond the dead zone.
    gain: float = 1.2
    #: Mechanical limit. Kept well inside the real one.
    max_yaw: float = math.radians(60.0)
    #: Radians per second. A body that snaps looks alarming and stresses the motor.
    max_rate: float = 1.2
    #: After losing you, keep turning the way you went for this long.
    reacquire_s: float = 1.2
    #: Which way the motor calls positive - a fact about the robot, not the maths.
    direction: float = 1.0

    yaw: float = 0.0
    _last_seen: float | None = field(default=None, repr=False)
    _last_x: float = field(default=0.0, repr=False)
    _last_now: float | None = field(default=None, repr=False)

    def update(self, detected: bool, x: float | None, now: float) -> float | None:
        """Fold in one observation and return the body yaw to command.

        Returns ``None`` when there is nothing to say - no face, and no recent
        one worth turning towards - so the caller leaves the body alone.
        """
        dt = 0.0 if self._last_now is None else max(0.0, now - self._last_now)
        self._last_now = now

        if detected and x is not None:
            self._last_seen = now
            self._last_x = float(x)
            return self._step(float(x), dt)

        if self.searching(now):
            # Keep going the way they went; they are just out of frame.
            return self._step(self._last_x, dt)
        return None

    def searching(self, now: float) -> bool:
        """Whether it is still worth turning towards where you last were."""
        if self._last_seen is None:
            return False
        return now - self._last_seen <= self.reacquire_s

    def _step(self, x: float, dt: float) -> float:
        """Move one rate-limited step towards the yaw that would centre `x`."""
        target = self._target_for(x)
        delta = target - self.yaw
        limit = self.max_rate * dt
        if abs(delta) > limit:
            delta = math.copysign(limit, delta)
        self.yaw = max(-self.max_yaw, min(self.max_yaw, self.yaw + delta))
        return self.yaw

    def _target_for(self, x: float) -> float:
        """Return the resting body yaw that would bring `x` back towards centre."""
        beyond = abs(x) - self.deadzone
        if beyond <= 0.0:
            return 0.0
        wanted = self.direction * math.copysign(beyond * self.gain, x)
        return max(-self.max_yaw, min(self.max_yaw, wanted))
