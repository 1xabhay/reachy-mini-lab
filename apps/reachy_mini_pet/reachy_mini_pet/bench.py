"""Measure a language model against what this pet actually asks of one.

The point is to pick a model for *your* hardware rather than trusting a
leaderboard. A pet needs three things and none of them are what benchmarks
measure: a short spoken reply, a feeling that honestly matches what it heard,
and an answer soon enough that the pause is not awkward.

Grading is deliberately programmatic. The emotion is a label from a closed
set, so an acceptable-set check is exact and free - no judge, no API, no
scoring model whose own reliability would then need establishing. The parts
that need a human ear are printed for you to read, not scored.

    reachy-pet-bench                       # every model Ollama has
    reachy-pet-bench qwen3:1.7b gemma3:1b  # just these
"""

from __future__ import annotations

import json
import statistics
import sys
import time
import urllib.request

from .live import EMOTIONS, local_mind
from .streaming import SPOKEN_BUDGET

#: What a pet actually gets said to, and which feelings would be a fair answer.
#: Several are usually defensible, so each case lists a set - grading against
#: one "correct" emotion would punish a model for a reasonable choice.
CASES: tuple[tuple[str, frozenset[str]], ...] = (
    ("good morning", frozenset({"welcoming1", "cheerful1", "attentive1", "enthusiastic1"})),
    ("I have had an absolutely awful day", frozenset({"sad1", "calming1", "understanding1"})),
    ("that is brilliant news", frozenset({"cheerful1", "enthusiastic1", "amazed1", "proud1"})),
    ("what's two plus two", frozenset({"attentive1", "helpful1", "thoughtful1", "cheerful1"})),
    ("asdkjh sdkjh garbled nonsense", frozenset({"confused1", "inquiring1", "curious1"})),
    ("I'm going to bed, goodnight", frozenset({"serenity1", "tired1", "calming1", "loving1"})),
    ("can you stop doing that", frozenset({"oops1", "understanding1", "sad1", "shy1"})),
    ("I missed you today", frozenset({"loving1", "grateful1", "welcoming1", "shy1"})),
)

#: Anything past this and the pause before it speaks is uncomfortable.
GOOD_LATENCY_S = 1.5


def installed_models() -> list[str]:
    """Ask Ollama what it has, so the bench needs no configuration."""
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5) as response:
            return sorted(m["name"] for m in json.load(response).get("models", []))
    except Exception:
        return []


def resident_bytes(model: str) -> int:
    """How much memory Ollama is holding for this model right now."""
    try:
        with urllib.request.urlopen("http://localhost:11434/api/ps", timeout=5) as response:
            for loaded in json.load(response).get("models", []):
                if loaded["name"] == model:
                    return int(loaded.get("size", 0))
    except Exception:
        pass
    return 0


def unload(model: str) -> None:
    """Ask Ollama to drop a model now.

    The pet keeps its model resident for an hour so it never pays a cold
    start, and the bench borrows the pet's own code - so without this,
    benching four models leaves every one of them in memory. On a 16 GB
    machine that is most of the machine.
    """
    try:
        request = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=json.dumps({"model": model, "keep_alive": 0}).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(request, timeout=15).close()
    except Exception:
        pass


def measure(model: str) -> dict:
    """Run every case against one model and report how it did."""
    think = local_mind(model=model)
    latencies: list[float] = []
    fitting = apt = 0
    lengths: list[int] = []
    transcript: list[tuple[str, str, str, bool]] = []

    for heard, acceptable in CASES:
        started = time.time()
        try:
            clauses, emotion = think(heard, [])
        except Exception as error:
            transcript.append((heard, f"<failed: {error}>", "", False))
            continue
        latencies.append(time.time() - started)

        said = " ".join(clauses)
        lengths.append(len(said))
        suitable = emotion in acceptable
        apt += suitable
        fitting += emotion in EMOTIONS
        transcript.append((heard, said, emotion, suitable))

    resident = resident_bytes(model)
    unload(model)

    return {
        "model": model,
        "cases": len(CASES),
        "median_latency": statistics.median(latencies) if latencies else float("inf"),
        "worst_latency": max(latencies) if latencies else float("inf"),
        "valid_emotions": fitting,
        "apt_emotions": apt,
        "median_length": statistics.median(lengths) if lengths else 0,
        "over_budget": sum(1 for n in lengths if n > SPOKEN_BUDGET),
        "resident": resident,
        "transcript": transcript,
    }


def main() -> None:
    """Bench the models named on the command line, or everything installed."""
    models = sys.argv[1:] or installed_models()
    if not models:
        print("No models found. Is Ollama running? (ollama serve)")
        raise SystemExit(1)

    results = []
    for model in models:
        print(f"\n=== {model} ===", flush=True)
        result = measure(model)
        results.append(result)
        for heard, said, emotion, suitable in result["transcript"]:
            mark = "ok " if suitable else "  ?"
            print(f"  {mark} {heard!r}")
            print(f"      [{emotion}] {said}")

    print(f"\n{'model':22} {'apt':>7} {'valid':>7} {'median':>8} {'worst':>8} "
          f"{'chars':>6} {'long':>5} {'RAM':>8}")
    for r in sorted(results, key=lambda r: (-r["apt_emotions"], r["median_latency"])):
        print(
            f"{r['model']:22} {r['apt_emotions']}/{r['cases']:<5} "
            f"{r['valid_emotions']}/{r['cases']:<5} "
            f"{r['median_latency']:7.2f}s {r['worst_latency']:7.2f}s "
            f"{r['median_length']:6.0f} {r['over_budget']:5} "
            f"{r['resident'] / 1e9:6.1f}GB"
        )
    print(
        f"\napt      = the feeling it chose was a fair answer (graded on a closed set)\n"
        f"valid    = the name exists in the emotions library at all\n"
        f"median   = time to a complete reply; over {GOOD_LATENCY_S}s the pause is awkward\n"
        f"long     = replies over the {SPOKEN_BUDGET}-character spoken budget"
    )


if __name__ == "__main__":
    main()
