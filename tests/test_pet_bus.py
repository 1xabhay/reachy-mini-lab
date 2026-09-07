"""Checks for the bus that lets the senses run independently.

Vision, hearing, voice and mind each have to work on their own - vision
tracking you with nothing else running, the voice speaking with nothing
listening - and also in any combination. That only stays true if none of them
holds a reference to another. They publish what they observe and subscribe to
what they care about, and know nothing else.

The bus is therefore the one piece every layer depends on, so its rules are
pinned hard: what it delivers, what it drops, and that a slow or broken
subscriber can neither block nor sink the rest.
"""

from dataclasses import dataclass

from hypothesis import given, settings
from hypothesis import strategies as st
from reachy_mini_pet.bus import Bus, Event


class Ping(Event):
    """A test event."""


class Pong(Event):
    """A different test event."""


@dataclass
class Seq(Event):
    """A test event that knows its own place in the sequence."""

    n: int = 0


def test_publishing_with_nobody_listening_is_harmless():
    """Every module must run standalone - usually that means unheard."""
    Bus().publish(Ping())


def test_a_subscriber_receives_what_it_asked_for():
    bus = Bus()
    inbox = bus.subscribe(Ping)
    bus.publish(Ping())

    assert len(bus.drain(inbox)) == 1


def test_a_subscriber_is_not_bothered_with_other_events():
    """Hearing must not wake up for every face the camera sees."""
    bus = Bus()
    inbox = bus.subscribe(Ping)
    bus.publish(Pong())

    assert bus.drain(inbox) == []


def test_every_subscriber_gets_its_own_copy():
    bus = Bus()
    first, second = bus.subscribe(Ping), bus.subscribe(Ping)
    bus.publish(Ping())

    assert len(bus.drain(first)) == 1
    assert len(bus.drain(second)) == 1


def test_draining_empties_the_inbox():
    bus = Bus()
    inbox = bus.subscribe(Ping)
    bus.publish(Ping())
    bus.drain(inbox)

    assert bus.drain(inbox) == []


def test_a_subscriber_can_leave():
    bus = Bus()
    inbox = bus.subscribe(Ping)
    bus.unsubscribe(inbox)
    bus.publish(Ping())

    assert bus.drain(inbox) == []


def test_a_stalled_subscriber_drops_the_oldest_not_the_newest():
    """Realtime: the freshest observation is the one worth having.

    A module that stops draining must never be able to grow the bus, and must
    never leave the others reading stale news.
    """
    bus = Bus(depth=3)
    inbox = bus.subscribe(Ping)
    sent = [Ping() for _ in range(10)]
    for event in sent:
        bus.publish(event)

    kept = bus.drain(inbox)
    assert len(kept) == 3
    assert kept == sent[-3:]


def test_a_stalled_subscriber_cannot_block_a_healthy_one():
    bus = Bus(depth=2)
    stalled, healthy = bus.subscribe(Ping), bus.subscribe(Ping)
    for _ in range(50):
        bus.publish(Ping())
        bus.drain(healthy)

    assert len(bus.drain(stalled)) == 2


def test_a_subscriber_that_raises_does_not_take_the_bus_down():
    """One broken module must not silence the others."""
    bus = Bus()
    bus.subscribe(Ping, handler=lambda _e: 1 / 0)
    inbox = bus.subscribe(Ping)

    bus.publish(Ping())

    assert len(bus.drain(inbox)) == 1


def test_a_handler_is_called_directly_for_reflexes():
    """Some things are too fast to queue - a flinch does not wait its turn."""
    bus = Bus()
    seen = []
    bus.subscribe(Ping, handler=seen.append)
    bus.publish(Ping())

    assert len(seen) == 1


@given(depth=st.integers(min_value=1, max_value=20), count=st.integers(min_value=0, max_value=200))
@settings(deadline=None, max_examples=100)
def test_the_bus_can_never_grow_without_bound(depth, count):
    """It runs for weeks with nobody watching it."""
    bus = Bus(depth=depth)
    inbox = bus.subscribe(Ping)
    for _ in range(count):
        bus.publish(Ping())

    assert bus.pending(inbox) <= depth


def test_subscribing_to_a_base_class_catches_its_kinds():
    """`Attention` wants everything; `Voice` wants only what concerns it."""
    bus = Bus()
    inbox = bus.subscribe(Event)
    bus.publish(Ping())
    bus.publish(Pong())

    assert len(bus.drain(inbox)) == 2


# ------------------------------------------------------ under two threads

# The module's whole docstring is about threads, and every test above ran on
# one. Draining with `list(inbox)` then `inbox.clear()` is two atomic
# operations, not one atomic operation: anything published in the gap between
# them was thrown away silently. Measured at 23.6% loss under contention.
#
# In production the window is about a microsecond against a low publish rate,
# so the practical effect is rare - but the symptom is "the pet just ignored
# me once", with nothing in the log, over weeks.


def test_a_concurrent_drain_loses_nothing():
    """The one property a bus between threads has to have."""
    import threading

    total = 200_000
    bus = Bus(depth=total)          # deep enough that drop-oldest cannot be blamed
    inbox = bus.subscribe(Ping)
    received = []
    done = threading.Event()

    def publisher():
        for _ in range(total):
            bus.publish(Ping())
        done.set()

    worker = threading.Thread(target=publisher, daemon=True)
    worker.start()
    while not done.is_set():
        received.extend(bus.drain(inbox))
    received.extend(bus.drain(inbox))
    worker.join(timeout=10.0)

    assert len(received) == total, f"lost {total - len(received)} events"


def test_draining_while_publishing_keeps_them_in_order():
    import threading

    total = 5_000
    bus = Bus(depth=total)
    inbox = bus.subscribe(Seq)
    received = []
    done = threading.Event()

    def publisher():
        for i in range(total):
            bus.publish(Seq(i))
        done.set()

    threading.Thread(target=publisher, daemon=True).start()
    while not done.is_set():
        received.extend(bus.drain(inbox))
    received.extend(bus.drain(inbox))

    numbers = [e.n for e in received]
    assert numbers == sorted(numbers), "events arrived out of order"
