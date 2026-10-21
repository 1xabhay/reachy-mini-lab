"""The robot, healing a dropped command websocket.

The SDK's WSClient has no reconnect, and its keepalive (websockets' default 20 s
ping, not exposed by the SDK) gives up when this machine stalls. From then on
every command raises ``ConnectionError("Lost connection with the server.")``
while WebRTC audio carries on. So on that error: one fresh WSClient in place of
the dead one (the media session is kept), motors and face tracking back on,
then the call once more. At most once per ``every`` seconds; a reconnect runs
off the event loop, so the voice never waits on it.
"""

import asyncio
import functools
import inspect
import logging
import math
import threading
import time

logger = logging.getLogger(__name__)


def connect(host: str, port: int):
    """Open a fresh SDK command client, as ``ReachyMini`` does."""
    from reachy_mini.io.ws_client import WSClient

    client = WSClient(host, port)
    client.wait_for_connection(timeout=5.0)
    return client


class Reconnecting:
    """Wrap a ``ReachyMini``: every call that loses the connection reconnects and retries once."""

    def __init__(self, mini, connect=connect, now=time.monotonic, every: float = 10.0):
        """Reconnect through ``connect(host, port)``, at most once per ``every`` seconds."""
        self._mini, self._connect, self._now, self._every = mini, connect, now, every
        self._last = -math.inf
        self._lock = threading.Lock()

    def __getattr__(self, name):
        """Pass ``name`` through, its calls guarded."""
        attr = getattr(self._mini, name)
        if not callable(attr):
            return attr
        if inspect.iscoroutinefunction(attr):
            async def call(*args, **kwargs):
                client = getattr(self._mini, "client", None)
                try:
                    return await attr(*args, **kwargs)
                except ConnectionError:
                    if not await asyncio.to_thread(self._reconnect, client):
                        raise
                return await attr(*args, **kwargs)
        else:
            def call(*args, **kwargs):
                client = getattr(self._mini, "client", None)
                try:
                    return attr(*args, **kwargs)
                except ConnectionError:
                    if not self._reconnect(client):
                        raise
                return attr(*args, **kwargs)
        return functools.wraps(attr)(call)

    def _reconnect(self, dead) -> bool:
        """Replace the ``dead`` client; True if the call is worth trying again."""
        with self._lock:
            if self._mini.client is not dead:  # another call already reconnected
                return True
            if self._now() - self._last < self._every:
                return False
            self._last = self._now()
            logger.warning("robot connection lost - reconnecting")
            try:
                fresh = self._connect(dead.host, dead.port)
            except Exception as error:
                logger.warning("could not reconnect to the robot (%s)", error)
                return False
            dead.disconnect()
            self._mini.client = fresh
            for rearm in (self._mini.enable_motors,
                          functools.partial(self._mini.start_head_tracking, weight=1.0)):
                try:
                    rearm()
                except Exception as error:
                    logger.warning("after reconnecting, %s failed (%s)", rearm, error)
            logger.info("robot reconnected")
            return True
