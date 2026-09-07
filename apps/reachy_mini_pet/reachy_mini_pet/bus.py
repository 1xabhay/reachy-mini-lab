"""The one thing every layer shares: a bus, so no layer needs any other.

Vision, hearing, voice and mind each have to work alone - the camera tracking
you with nothing else switched on, the voice speaking into an empty room - and
also in any combination, without being rewritten for each one. That is only
true if none of them holds a reference to another. So they publish what they
observe, subscribe to what concerns them, and know nothing else exists.

Two ways to listen, because a body needs both. An inbox is a bounded queue
drained in a module's own time - that is how deliberate work is done. A handler
is called immediately, in the publisher's thread, for the things too fast to
queue: a flinch does not wait its turn.

Every inbox is bounded and drops oldest-first. In something that runs for weeks
the freshest observation is the one worth having, and a module that stops
draining must never be able to grow the bus or stall the modules beside it.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import count

logger = logging.getLogger("reachy_mini.pet.bus")

#: Inbox depth. About a second of observations at sense rates - enough to ride
#: out a pause, few enough that nobody acts on stale news.
DEFAULT_DEPTH = 32

_ids = count(1)


@dataclass
class Event:
    """Something that happened. Subclasses carry the detail."""


@dataclass
class Subscription:
    """A handle on the bus: an inbox, a handler, or both."""

    id: int
    kind: type[Event]
    inbox: deque = field(repr=False)
    handler: Callable[[Event], None] | None = field(default=None, repr=False)


class Bus:
    """Carries events between layers that know nothing about each other."""

    def __init__(self, depth: int = DEFAULT_DEPTH) -> None:
        """Create an empty bus with a fixed inbox depth."""
        self.depth = depth
        self._subscriptions: list[Subscription] = []
        self._lock = threading.Lock()

    def subscribe(
        self, kind: type[Event], handler: Callable[[Event], None] | None = None
    ) -> Subscription:
        """Listen for one kind of event, and every kind that derives from it."""
        subscription = Subscription(
            id=next(_ids), kind=kind, inbox=deque(maxlen=self.depth), handler=handler
        )
        with self._lock:
            self._subscriptions.append(subscription)
        return subscription

    def unsubscribe(self, subscription: Subscription) -> None:
        """Stop listening."""
        with self._lock:
            self._subscriptions = [s for s in self._subscriptions if s.id != subscription.id]

    def publish(self, event: Event) -> None:
        """Offer an event to everyone who cares, and to nobody else."""
        with self._lock:
            interested = [s for s in self._subscriptions if isinstance(event, s.kind)]

        for subscription in interested:
            # deque(maxlen) drops the oldest for us, which is the policy we want.
            subscription.inbox.append(event)
            if subscription.handler is not None:
                try:
                    subscription.handler(event)
                except Exception:
                    # One broken module must never silence the others.
                    logger.exception("Handler for %s failed", type(event).__name__)

    def drain(self, subscription: Subscription) -> list[Event]:
        """Take everything waiting in an inbox, oldest first.

        Popped one at a time rather than copied-then-cleared. `list(inbox)`
        followed by `inbox.clear()` is two atomic operations, not one: anything
        the publisher appended between them was silently thrown away. Each
        `popleft` is atomic on its own, so nothing can fall through the gap -
        and there is no gap.
        """
        inbox = subscription.inbox
        events: list[Event] = []
        while True:
            try:
                events.append(inbox.popleft())
            except IndexError:
                return events

    def pending(self, subscription: Subscription) -> int:
        """How many events are waiting to be drained."""
        return len(subscription.inbox)
