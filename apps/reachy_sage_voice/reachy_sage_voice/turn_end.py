"""Calibrate the turn-end silence on recordings of the owner talking to the robot.

Ported from an earlier Pipecat voice bot's pause analysis: drive the same
Silero VAD the pipeline runs over real speech, measure how long the speaker
pauses *inside* a turn, and pick the shortest silence that waits through
almost all of those pauses. Here each recording is one turn spoken to the
robot, so every silence inside it is a pause the robot must not answer.

    uv run sage-voice-turn-end turn-*.wav

Then set ``SAGE_VOICE_TURN_END_MS`` to the value it prints. The recordings are
the owner's voice: analyse them, never commit them.
"""

import asyncio
import itertools
import statistics
import sys
import wave

import numpy as np
from pipecat.audio.vad.vad_analyzer import VADState

#: Thresholds tried, in ms. 450 is an earlier voice bot's measured value and the default.
CANDIDATES_MS = (200, 300, 450, 600, 800, 1000)
DEFAULT_MS = 450
SAMPLE_RATE = 16000
CHUNK = 512  # samples per analyze_audio call at 16 kHz: 32 ms


def speech_runs(states: list[VADState], chunk_s: float) -> list[tuple[float, float]]:
    """Return (start_s, end_s) of each stretch the VAD calls speech, STARTING included."""
    runs, start = [], None
    for i, state in enumerate(states):
        speaking = state in (VADState.STARTING, VADState.SPEAKING)
        if speaking and start is None:
            start = i * chunk_s
        elif not speaking and start is not None:
            runs.append((start, i * chunk_s))
            start = None
    if start is not None:
        runs.append((start, len(states) * chunk_s))
    return runs


def pauses_ms(runs: list[tuple[float, float]]) -> list[float]:
    """Return the silences between consecutive runs of speech, in ms."""
    gaps = [(b[0] - a[1]) * 1000.0 for a, b in itertools.pairwise(runs)]
    return [round(g, 6) for g in gaps if g > 0]


def pause_table(pauses: list[float], candidates=CANDIDATES_MS) -> list[tuple[int, int, float]]:
    """For each threshold: how many pauses it would mistake for the end of a turn, and share."""
    return [(c, n, n / len(pauses)) for c in candidates
            for n in [sum(1 for p in pauses if p >= c)]]


def choose_turn_end_ms(pauses: list[float], tolerance: float = 0.05,
                       candidates=CANDIDATES_MS) -> int:
    """Return the shortest threshold that cuts at most ``tolerance`` of real pauses short.

    ``tolerance`` is the knob: a lower one waits longer and interrupts the owner
    less, at the cost of a slower reply on every turn.
    """
    if not pauses:
        return DEFAULT_MS
    for c, _n, share in pause_table(pauses, candidates):
        if share <= tolerance:
            return c
    return candidates[-1]


def load_pcm16(path: str) -> np.ndarray:
    """Load a WAV as mono int16 at 16 kHz (nearest-neighbour is enough for a VAD)."""
    with wave.open(path, "rb") as w:
        rate, channels = w.getframerate(), w.getnchannels()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.int16)
    if rate != SAMPLE_RATE:
        idx = (np.arange(int(len(pcm) * SAMPLE_RATE / rate)) * rate / SAMPLE_RATE).astype(int)
        pcm = pcm[idx[idx < len(pcm)]]
    return pcm


async def _states(pcm: np.ndarray) -> list[VADState]:
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.audio.vad.vad_analyzer import VADParams

    # A tiny stop_secs reports raw boundaries; the thresholds are judged here instead.
    vad = SileroVADAnalyzer(params=VADParams(stop_secs=0.05, start_secs=0.05))
    vad.set_sample_rate(SAMPLE_RATE)
    return [await vad.analyze_audio(pcm[i:i + CHUNK].tobytes())
            for i in range(0, len(pcm) - CHUNK, CHUNK)]


def main(paths: list[str] | None = None) -> None:
    """Print the pause distribution of the recordings and the turn end it implies."""
    paths = sys.argv[1:] if paths is None else paths
    if not paths:
        sys.exit("usage: sage-voice-turn-end <one-turn-per-file.wav>...")
    pauses: list[float] = []
    for path in paths:
        found = pauses_ms(speech_runs(asyncio.run(_states(load_pcm16(path))), CHUNK / SAMPLE_RATE))
        print(f"{path}: {len(found)} pauses")
        pauses += found
    if len(pauses) > 2:
        q = statistics.quantiles(pauses, n=100)
        print("pauses inside a turn: " + "  ".join(f"p{p}={q[p - 1]:.0f}ms"
                                                    for p in (50, 75, 90, 95, 99)))
    for c, n, share in pause_table(pauses) if pauses else []:
        print(f"  {c:>5} ms would cut {n:>4} pauses short ({share:.0%})")
    print(f"SAGE_VOICE_TURN_END_MS={choose_turn_end_ms(pauses)}")


if __name__ == "__main__":
    main()
