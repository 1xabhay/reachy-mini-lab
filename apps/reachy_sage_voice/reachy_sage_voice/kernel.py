"""Device protocol v1 (sage-kernel docs/PROTOCOL.md) over stdlib HTTP.

``KernelClient`` is the wire: one call per route, every outage a
``KernelUnavailableError``, ``rate_limited`` waited out. ``Session`` is one
body's conversation rules on top: hello before each turn, busy and
``unknown_pending`` handled, the effects cursor kept. Blocking HTTP runs in a
thread so the Pipecat loop never waits on it.
"""

import asyncio
import json
import logging
import math
import re
import time
import urllib.error
import urllib.request
from collections.abc import Awaitable, Callable, Mapping

from reachy_sage_voice import voice

logger = logging.getLogger(__name__)

#: The body language the model may choose, all from the robot's recorded-moves
#: library: the pet's 30 (reachy_mini_pet/live.py), the expressive moves the
#: owner added, and every variant of an offered move. Never a negative one
#: (rage, furious, contempt, disgusted, dying, reprimand, irritated, displeased,
#: frustrated): owner. Copied, not imported: this app never loads the pet.
EMOTIONS = (
    # the pet's 30
    "amazed1", "attentive1", "boredom1", "calming1", "cheerful1", "confused1", "curious1",
    "dance1", "enthusiastic1", "grateful1", "helpful1", "impatient1", "inquiring1",
    "laughing1", "loving1", "no1", "oops1", "proud1", "relief1", "sad1", "scared1",
    "serenity1", "shy1", "success1", "surprised1", "thoughtful1", "tired1",
    "understanding1", "welcoming1", "yes1",
    # expressive moves
    "waiting", "come1", "go_away1", "lonely1", "uncertain1", "electric1", "yes_sad1",
    "no_excited1", "no_sad1",
    # further variants of offered moves
    "attentive2", "boredom2", "dance2", "dance3", "enthusiastic2", "helpful2", "impatient2",
    "inquiring2", "inquiring3", "laughing2", "oops2", "proud2", "proud3", "relief2", "sad2",
    "success2", "surprised2", "thoughtful2", "understanding2", "welcoming2",
)

#: Moves for the coming mood model, not the emote tool: declared so the
#: vocabulary contract keeps them honest, never offered to the model yet.
STATE_MOVES = ("sleep1", "mini-deep-sleep", "wake-mini-up", "toc-toc-toc")

#: Declared moves the library records without a sound; they always play muted.
NO_SOUND = frozenset({"waiting"})

#: One clear line for the moves whose library description is weak: it names
#: another move, or does not say when to use it. The library stays the source
#: for every other move.
DESCRIPTIONS = {
    "attentive2": "Listen closely while someone is talking, to encourage them to go on.",
    "boredom2": "Doze off and snore, when no one has talked to you for a long while.",
    "cheerful1": "A happy little whistle, when someone proposes something that pleases you.",
    "confused1": "Use it when you don't know how to answer a question; it can be funny.",
    "curious1": "Glance around at everyone, in a conversation with several people.",
    "dance3": "Dance energetically and wiggle, when lively music is playing.",
    "enthusiastic2": "Light excitement, for everyday good news.",
    "impatient2": "Show impatience, when you disagree or someone is stalling.",
    "inquiring2": "A light questioning look, when you are thinking and need more details.",
    "laughing2": "A light laugh, for a mildly funny remark.",
    "proud1": "Look around with a satisfied air, when something has gone well.",
    "relief2": "Settle down calmly, when mild annoyance or tension passes.",
    "waiting": "Wait patiently, when you expect someone to answer or act.",
}


def load_moves(vocabulary: tuple[str, ...] = EMOTIONS) -> dict:
    """Load the declared moves from the robot's recorded-moves library, by name.

    The library reads the local Hugging Face cache first (the robot daemon
    preloads it) and downloads only if it is missing.
    """
    from reachy_mini.motion.recorded_move import DEFAULT_EMOTIONS_DATASET, RecordedMoves

    library = RecordedMoves(DEFAULT_EMOTIONS_DATASET)
    return {name: library.get(name) for name in vocabulary}


def describe_move(name: str, move) -> str:
    """One line on what a move means: the override, else the library's first sentence."""
    if name in DESCRIPTIONS:
        return DESCRIPTIONS[name]
    return re.split(r"(?<=[.!?])\s", move.description.strip(), maxsplit=1)[0]


def manifest(moves: Mapping) -> dict:
    """Describe what this body can do, as MCP-shaped tools. Replies are not a tool in v1.

    The tools are emote, then voice_style and vocal (``voice.py``).

    ``moves`` maps each declared name to its recorded move (``.duration``,
    ``.description``). ``max_s`` is the longest of them, rounded up, plus 1 s,
    so the kernel never times out a move still playing. Each name's first
    line (``describe_move``) rides beside the enum, which is what the kernel checks.
    """
    names = list(moves)
    meanings = "\n".join(f"{name}: {describe_move(name, moves[name])}" for name in names)
    return {"senses": ["speech"], "tools": [{
        "name": "emote",
        "description": "Show a feeling with the body, just before speaking.",
        "inputSchema": {"type": "object",
                        "properties": {"name": {"type": "string", "enum": names,
                                                "description": meanings}},
                        "required": ["name"]},
        "max_s": math.ceil(max(m.duration for m in moves.values())) + 1,
        "interruptible": True,
    }, *voice.tools(moves)]}


class KernelUnavailableError(Exception):
    """The kernel could not be reached, or answered nonsense.

    Deliberately not an OSError, as in the pet: a robot session reads OSError
    as the robot itself being gone.
    """


class KernelClient:
    """One method per v1 route. Returns (status, body); raises only on an outage."""

    #: How many times a rate-limited event is resent before giving the 429 back.
    RATE_LIMIT_TRIES = 5

    def __init__(self, url: str, token: str, timeout_s: float = 5.0,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        """Take the kernel's device listener and this body's token."""
        self.url, self.token, self.timeout_s, self.sleep = url.rstrip("/"), token, timeout_s, sleep

    def hello(self, manifest: dict) -> tuple[int, dict]:
        """Declare protocol 1 and what this body can do."""
        return self._call("POST", "/v1/hello", {"protocols": [1], "manifest": manifest})

    def event(self, body: dict) -> tuple[int, dict]:
        """Send one event, waiting out ``rate_limited`` (``stop`` is never limited)."""
        for _ in range(self.RATE_LIMIT_TRIES):
            status, answer = self._call("POST", "/v1/events", body)
            if answer.get("code") != "rate_limited":
                break
            self.sleep(float(answer.get("retry_after", 1)))
        return status, answer

    def effects(self, since: int, wait: float, offset: int = 0) -> tuple[int, dict]:
        """Long-poll for effects after ``since`` and reply text past ``offset``."""
        path = f"/v1/effects?since={since}&wait={wait:g}&offset={offset}"
        return self._call("GET", path, timeout_s=wait + self.timeout_s)

    def _call(self, method: str, path: str, body: dict | None = None,
              timeout_s: float | None = None) -> tuple[int, dict]:
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Authorization": f"Bearer {self.token}"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self.url + path, data, headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout_s or self.timeout_s) as response:
                return response.status, json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as error:
            with error:
                try:
                    return error.code, json.loads(error.read() or b"{}")
                except ValueError:
                    return error.code, {}
        except (OSError, ValueError) as error:  # refused, timeout, garbled
            raise KernelUnavailableError(f"{method} {path}: {error}") from error


class Session:
    """One body's conversation with the kernel: admission, the cursor, results."""

    def __init__(self, client: KernelClient, manifest: dict,
                 retry_s: float = 5.0, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> None:
        """Take the wire and the manifest; ``retry_s`` bounds waiting on another body."""
        self.client, self.manifest = client, manifest
        self.retry_s, self.clock, self.sleep = retry_s, clock, sleep
        self.cursor: int | None = None

    async def _io(self, fn, *args):
        return await asyncio.to_thread(fn, *args)

    async def start(self, turn_id: str, text: str, timing: dict) -> str:
        """Submit a turn: "admitted", "busy" (gave up after ``retry_s``), or "refused"."""
        # Before every turn: the kernel forgets manifests when it restarts.
        status, body = await self._io(self.client.hello, self.manifest)
        if status != 200:
            raise KernelUnavailableError(f"hello refused: {body}")
        if self.cursor is None:
            self.cursor = body["sequence"]  # start at the head: never replay old replies
        event = {"kind": "submitted", "turn_id": turn_id, "text": text}
        if timing:
            event["timing"] = timing
        deadline, acknowledged = self.clock() + self.retry_s, False
        while True:
            status, body = await self._io(self.client.event, event)
            if status == 200:
                return "admitted"
            code = body.get("code")
            if code == "unknown_pending" and not acknowledged:
                acknowledged = await self._acknowledge_own()
                if acknowledged:
                    continue
            if code != "busy":
                logger.warning("kernel refused the turn: %s", body)
                return "refused"
            if self.clock() >= deadline:
                return "busy"
            await self.sleep(0.25)

    async def _acknowledge_own(self) -> bool:
        # The turn field appears only for a turn this body started.
        status, body = await self._io(self.client.effects, self.cursor or 0, 0)
        turn = body.get("turn") or {}
        if status != 200 or turn.get("run_status") != "unknown":
            return False
        logger.info("letting go of interrupted turn %s", turn["turn_id"])
        event = {"kind": "acknowledged", "turn_id": turn["turn_id"]}
        return (await self._io(self.client.event, event))[0] == 200

    async def poll(self, offset: int, wait: float) -> dict:
        """One long poll. Advances the cursor; a ``reset`` restarts it and returns nothing."""
        status, body = await self._io(self.client.effects, self.cursor or 0, wait, offset)
        if status == 409 and body.get("code") == "reset":
            self.cursor = body["sequence"]
            return {"sequence": self.cursor, "effects": []}
        if status != 200:
            raise KernelUnavailableError(f"effects refused: {status} {body}")
        self.cursor = body["sequence"]
        return body

    async def result(self, sequence: int | None, status: str, text: str | None = None,
                     timing: dict | None = None, reason: str | None = None,
                     turn_id: str | None = None, duration_ms: int | None = None) -> None:
        """Report how one delivered effect went: done, failed or interrupted.

        ``duration_ms`` is how long the body spent performing it.
        A reply still streaming has no sequence: pass ``turn_id`` instead.
        ponytail: that shape is assumed; the v1 amendment that lets a streaming
        device report is not written yet. Change it here only.
        """
        event: dict = {"kind": "result"}
        if sequence is not None:
            event["sequence"] = sequence
        else:
            event["turn_id"] = turn_id
        event["status"] = status
        if text is not None:
            event["text"] = text
        if timing:
            event["timing"] = timing
        if reason is not None:
            event["reason"] = reason
        if duration_ms is not None:
            event["duration_ms"] = duration_ms
        await self._io(self.client.event, event)

    async def presence(self, state: str, confidence: float) -> None:
        """Report someone seen or lost (contract draft: {kind: presence, state, confidence})."""
        await self._io(self.client.event,
                       {"kind": "presence", "state": state, "confidence": confidence})

    async def stop(self, turn_id: str) -> None:
        """Cancel this body's running turn."""
        await self._io(self.client.event, {"kind": "stop", "turn_id": turn_id})
