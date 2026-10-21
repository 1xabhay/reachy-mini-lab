"""The sim as a Sage device, protocol v1 (sage-kernel docs/PROTOCOL.md), over stdlib HTTP.

`plan` is the policy: what the sim does with each effect. `turn` is the I/O: one
text turn from hello to the last result, emotes played as recorded moves in the
sim daemon.
"""

import json
import math
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from functools import cache

#: The voice app's emote vocabulary (reachy_sage_voice/kernel.py), copied, not
#: imported: the apps share no venv. A test keeps the two equal.
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
)  # fmt: skip
DATASET = "pollen-robotics/reachy-mini-emotions-library"


@cache
def durations() -> dict[str, float]:
    """Each emote's length in seconds, from the local recorded-moves cache."""
    from reachy_mini.motion.recorded_move import RecordedMoves

    library = RecordedMoves(DATASET)
    return {name: library.get(name).duration for name in EMOTIONS}


def manifest(lengths: Mapping[str, float]) -> dict:
    """One emote tool; max_s is the longest move, rounded up, plus 1 s, as in the voice app."""
    return {
        "senses": ["text"],
        "tools": [
            {
                "name": "emote",
                "description": "Show a feeling with the body, just before speaking.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"name": {"type": "string", "enum": list(EMOTIONS)}},
                    "required": ["name"],
                },
                "max_s": math.ceil(max(lengths.values())) + 1,
                "interruptible": True,
            }
        ],
    }


def plan(effect: dict) -> tuple[str, str | None] | None:
    """Return ("play", move), ("done", None) or ("failed", reason); None for a stop."""
    kind = effect["kind"]
    if kind == "stop":
        return None
    if effect.get("stale"):
        return ("failed", "stale")
    if kind == "emote":
        name = effect.get("input", {}).get("name")
        return ("play", name) if name in EMOTIONS else ("failed", "unknown_move")
    if kind == "reply":
        return ("done", None)  # the sim has no speaker: a reply is done once received
    return ("failed", "unavailable")


class KernelBusyError(Exception):
    """Another turn kept the kernel busy for longer than the retry window."""


class Kernel:
    """The kernel's device listener, as the voice app's KernelClient: (status, body) back."""

    RATE_LIMIT_TRIES = 5

    def __init__(self, url: str, token: str, sleep: Callable[[float], None] = time.sleep):
        """Take the listener's URL and this device's token."""
        self.url, self.token, self.sleep = url, token, sleep

    def call(self, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
        """One request; an HTTP error comes back as its status and coded body."""
        return _http(self.url, method, path, body, self.token)

    def event(self, body: dict) -> tuple[int, dict]:
        """Send one event, resent after `retry_after` while rate-limited."""
        for _ in range(self.RATE_LIMIT_TRIES):
            status, answer = self.call("POST", "/v1/events", body)
            if answer.get("code") != "rate_limited":
                break
            self.sleep(float(answer.get("retry_after", 1)))
        return status, answer

    def ok(self, status_body: tuple[int, dict]) -> dict:
        """Return the body of a 200, or raise with what the kernel said."""
        status, body = status_body
        if status != 200:
            raise RuntimeError(f"kernel refused: {status} {body}")
        return body


def _http(url: str, method: str, path: str, body=None, token: str | None = None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url + path, data, headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=40) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        with error:
            try:
                return error.code, json.loads(error.read() or b"{}")
            except ValueError:
                return error.code, {}


def play_move(sim_api: str, name: str, stopping: threading.Event) -> str:
    """Play a recorded move in the sim daemon: "done", "interrupted" or "failed"."""
    if stopping.is_set():
        return "interrupted"
    status, move = _http(sim_api, "POST", f"/move/play/recorded-move-dataset/{DATASET}/{name}", {})
    if status != 200:
        return "failed"
    while any(m["uuid"] == move["uuid"] for m in _http(sim_api, "GET", "/move/running")[1]):
        if stopping.is_set():
            _http(sim_api, "POST", "/move/stop", move)
            return "interrupted"
        time.sleep(0.05)
    return "done"


class Body:
    """The sim's moves, one at a time, as on the pet: a newer emote supersedes the playing one.

    `report(record, event)` gets each move's outcome and its `result` event.
    """

    def __init__(self, sim_api: str, report: Callable, clock: Callable[[], float] = time.monotonic):
        """Take the sim daemon's API and where to send each move's result."""
        self.sim_api, self.report, self.clock = sim_api, report, clock
        self._lock, self._playing, self._threads = threading.Lock(), None, []

    def emote(self, effect: dict, name: str) -> None:
        """Play `name` for this effect, interrupting whatever is playing."""
        with self._lock:
            previous = self._playing
            if previous:
                self._interrupt(previous[1], "superseded")
            stopping = threading.Event()
            thread = threading.Thread(target=self._play, args=(effect, name, stopping, previous))
            self._playing = (thread, stopping)
            self._threads.append(thread)
            thread.start()

    def stop(self) -> None:
        """Interrupt the playing move, if any (a stop effect)."""
        with self._lock:
            if self._playing:
                self._interrupt(self._playing[1], "stopped")

    def moving(self) -> bool:
        """Whether any move is still playing or reporting."""
        return any(t.is_alive() for t in self._threads)

    def wait(self) -> None:
        """Wait for every move to finish and be reported."""
        for thread in list(self._threads):
            thread.join()

    @staticmethod
    def _interrupt(stopping: threading.Event, reason: str) -> None:
        if not stopping.is_set():
            stopping.reason = reason
            stopping.set()

    def _play(self, effect: dict, name: str, stopping: threading.Event, previous) -> None:
        if previous:
            previous[0].join()  # the old move is stopped before the new one starts
        start = self.clock()
        status = play_move(self.sim_api, name, stopping)
        ms = round((self.clock() - start) * 1000)
        reason = {"failed": "move_failed", "interrupted": getattr(stopping, "reason", None)}
        record = {"kind": "emote", "status": status, "reason": reason.get(status), "name": name}
        record["duration_ms"] = ms
        event = {"kind": "result", "sequence": effect["sequence"], "status": status}
        event["duration_ms"] = ms
        if record["reason"]:
            event["reason"] = record["reason"]
        self.report(record, event)


def turn(
    kernel: str,
    token: str,
    sim_api: str,
    text: str,
    turn_id: str,
    clock: Callable[[], float] = time.monotonic,
    retry_s: float = 5.0,
) -> list[dict]:
    """Say `text` to the kernel as the sim, act on every effect, report each result.

    Moves play beside the poll, as on the robot. Returns one record per result
    sent: kind, status, reason; for a move name, duration_ms and journaled_ms;
    for a reply timing.
    """
    link = Kernel(kernel, token)
    hello = {"protocols": [1], "manifest": manifest(durations())}
    since = link.ok(link.call("POST", "/v1/hello", hello))["sequence"]
    sent, deadline = clock(), clock() + retry_s
    submitted = {"kind": "submitted", "turn_id": turn_id, "text": text}
    while (answer := link.event(submitted))[1].get("code") == "busy":
        if clock() >= deadline:
            raise KernelBusyError(f"kernel busy for {retry_s:g} s; {turn_id} not admitted")
        time.sleep(0.25)
    link.ok(answer)

    offset, first_text, results = 0, None, []

    def report(record: dict, event: dict) -> None:
        answer = link.ok(link.event(event))
        if "duration_ms" in record:
            journaled = answer.get("turn", {}).get("timing", {}).get("durations_ms", {})
            record["journaled_ms"] = journaled.get(str(event["sequence"]))
        results.append(record)

    body = Body(sim_api, report, clock)

    while True:
        polled = link.ok(link.call("GET", f"/v1/effects?since={since}&wait=1&offset={offset}"))
        since = polled["sequence"]
        for effect in polled["effects"]:
            if effect["kind"] == "reply_delta":
                offset += len(effect["text"])
                first_text = first_text or clock()
                continue
            action = plan(effect)
            if action is None:
                body.stop()
                continue
            if action[0] == "play":
                body.emote(effect, action[1])
                continue
            record = {"kind": effect["kind"], "status": action[0], "reason": action[1]}
            event = {"kind": "result", "sequence": effect["sequence"], "status": action[0]}
            if action[1]:
                event["reason"] = action[1]
            if effect["kind"] == "reply":
                first_text = first_text or clock()
                record["timing"] = {"ttft_ms": round((first_text - sent) * 1000)}
                event["timing"] = record["timing"]
            report(record, event)
        status = polled.get("turn", {}).get("run_status")
        if status not in (None, "active") and not body.moving():
            return results
