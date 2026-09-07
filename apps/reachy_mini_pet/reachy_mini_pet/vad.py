"""Neural voice detection, replacing the energy gate.

An RMS gate cannot tell a voice from a vacuum cleaner - it only knows loud
from quiet, which is why calibrating one against a real room is a losing game.
Silero is a small neural detector that answers the actual question, and it
runs in well under a millisecond on the onnxruntime already installed for the
robot's own face tracking.

It has one awkward requirement: exactly 512 samples at 16 kHz. The robot's mic
delivers 320. Those do not divide, so `Reframer` holds the remainder - in one
preallocated buffer that never grows, because this runs on every frame for as
long as the pet is switched on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as ort

#: Silero v5 accepts exactly this many samples at 16 kHz, and nothing else.
WINDOW = 512
SAMPLERATE = 16_000

#: Silero v5 is not stateless across windows in the way it first appears: it
#: expects the last 64 samples of the *previous* window prepended to this one.
#: Feed it bare 512-sample windows and clear speech scores about 0.006.
CONTEXT = 64


class Reframer:
    """Turn arbitrary mic frames into exactly-sized windows, losing nothing."""

    def __init__(self, window: int = WINDOW) -> None:
        """Allocate the one buffer this will ever use."""
        self.window = window
        self._buffer = np.zeros(window, dtype=np.float32)
        self._held = 0

    @property
    def held(self) -> int:
        """Samples currently waiting for the rest of their window."""
        return self._held

    def push(self, frame: npt.NDArray[np.float32]) -> list[npt.NDArray[np.float32]]:
        """Add one mic frame, returning any complete windows it produced."""
        mono = np.asarray(frame, dtype=np.float32)
        if mono.ndim > 1:  # the mic is an array; the detector wants one channel
            mono = mono.mean(axis=1)
        mono = mono.reshape(-1)

        windows: list[npt.NDArray[np.float32]] = []
        taken = 0
        while taken < mono.size:
            room = self.window - self._held
            take = min(room, mono.size - taken)
            self._buffer[self._held : self._held + take] = mono[taken : taken + take]
            self._held += take
            taken += take
            if self._held == self.window:
                windows.append(self._buffer.copy())
                self._held = 0
        return windows

    def reset(self) -> None:
        """Discard the partial window."""
        self._held = 0


class SileroVAD:
    """Silero v5, asked one window at a time.

    The model is recurrent: its state carries the sense of an utterance in
    progress across windows, so it must be reset between utterances or the
    previous one bleeds into the next.
    """

    def __init__(self, model_path: str | Path, threshold: float = 0.5) -> None:
        """Load the detector onto the CPU provider."""
        options = ort.SessionOptions()
        # One thread: it is sub-millisecond work and it shares this machine
        # with whisper, a language model, and the robot's own vision.
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )
        self.threshold = threshold
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT, dtype=np.float32)

    def reset(self) -> None:
        """Forget the utterance in progress."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT, dtype=np.float32)

    def probability(self, window: npt.NDArray[np.float32]) -> float:
        """Return how likely this window is to be speech, from 0 to 1."""
        if window.size != WINDOW:
            raise ValueError(f"Silero needs exactly {WINDOW} samples, got {window.size}")
        padded = np.concatenate([self._context, window.astype(np.float32)])
        out, self._state = self._session.run(
            None,
            {
                "input": padded.reshape(1, -1),
                "state": self._state,
                "sr": np.array(SAMPLERATE, dtype=np.int64),
            },
        )
        self._context = window[-CONTEXT:].astype(np.float32).copy()
        return float(out[0][0])

    def is_speech(self, window: npt.NDArray[np.float32]) -> bool:
        """Whether this window is speech, at the configured threshold."""
        return self.probability(window) >= self.threshold
