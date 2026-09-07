"""Synthetic microphone audio for the pet's tests.

The mic gate is an energy detector, so what it does depends entirely on the
*shape* of the signal over time, not just its level. White noise at a constant
amplitude would pass every test here while telling us nothing about speech,
which arrives in syllables with short gaps inside a sentence and longer ones
between them. These generators reproduce that structure.
"""

from __future__ import annotations

import numpy as np

SAMPLERATE = 16_000
FRAME = 1600  #: 0.1 s - roughly what the ReSpeaker hands over per call.


def _framed(signal: np.ndarray, channels: int = 1) -> list[np.ndarray]:
    """Chop a signal into mic-sized frames, duplicated across channels."""
    frames = []
    for start in range(0, len(signal) - FRAME + 1, FRAME):
        chunk = signal[start : start + FRAME].astype(np.float32)
        if channels > 1:
            chunk = np.repeat(chunk[:, None], channels, axis=1)
        frames.append(chunk)
    return frames


def _t(seconds: float, samplerate: int = SAMPLERATE) -> np.ndarray:
    """Build a time base."""
    return np.arange(int(seconds * samplerate)) / samplerate


def room_tone(seconds: float, level: float = 0.002, seed: int = 0, channels: int = 1):
    """Generate an empty room: the noise floor of the mic and the building."""
    rng = np.random.default_rng(seed)
    return _framed(rng.normal(0, level, len(_t(seconds))), channels)


def speech(
    seconds: float,
    level: float = 0.25,
    syllable_hz: float = 4.0,
    seed: int = 1,
    channels: int = 1,
):
    """Generate someone talking.

    A voiced fundamental with harmonics, amplitude-modulated at a syllable
    rate. The modulation matters: it puts brief low-energy dips *inside* the
    utterance, which is exactly what the hangover exists to ride over.
    """
    rng = np.random.default_rng(seed)
    t = _t(seconds)
    voiced = sum(np.sin(2 * np.pi * 110 * h * t) / h for h in (1, 2, 3, 4))
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * syllable_hz * t)
    signal = voiced * envelope + rng.normal(0, 0.01, len(t))
    signal *= level / (np.sqrt(np.mean(signal**2)) + 1e-12)
    return _framed(signal, channels)


def clatter(level: float = 0.9, seed: int = 2, channels: int = 1):
    """Generate a cupboard door: one loud frame, gone as fast as it came."""
    rng = np.random.default_rng(seed)
    burst = rng.normal(0, level, FRAME) * np.exp(-np.linspace(0, 6, FRAME))
    return _framed(np.concatenate([burst, np.zeros(FRAME)]), channels)


def tv_chatter(seconds: float, level: float = 0.04, seed: int = 3, channels: int = 1):
    """Generate a television next door: speech-shaped, well below a real voice."""
    return speech(seconds, level=level, syllable_hz=3.0, seed=seed, channels=channels)


def rms(frames: list[np.ndarray]) -> float:
    """Measure the overall level of a run of frames."""
    flat = np.concatenate([np.asarray(f).reshape(-1) for f in frames])
    return float(np.sqrt(np.mean(np.square(flat))))
