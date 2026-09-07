"""Checks for the re-framer and the neural voice detector.

The mic hands over 320-sample frames; Silero accepts only 512. Those do not
divide, so something has to hold the remainder - and that something runs on
every frame for weeks, which makes "does it ever lose a sample" and "does it
ever grow" the two questions worth asking.

The detector itself is a model, so it is checked against real audio rather
than pinned to exact numbers: speech scores high, silence scores low, and the
threshold sits somewhere sane between them.
"""

import os
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from reachy_mini_pet.vad import SAMPLERATE, WINDOW, Reframer, SileroVAD

DATA = Path(os.environ.get("REACHY_PET_DATA_DIR", "~/.reachy_pet")).expanduser()
MODEL = DATA / "silero_vad.onnx"
VOICES = DATA / "voices"


@pytest.fixture(scope="module")
def real_speech():
    """Synthesise a sentence with the app's own voice, at 16 kHz.

    Real speech rather than a tone, because the whole point of the neural
    detector is that it distinguishes the two - and generated here rather than
    committed, so it works on any machine that has run the install steps
    instead of being silently skipped everywhere but one.
    """
    voice_file = next(VOICES.glob("*.onnx"), None)
    if voice_file is None:
        pytest.skip("no piper voice downloaded - see the app README")

    from piper import PiperVoice
    from scipy.signal import resample_poly

    voice = PiperVoice.load(voice_file)
    chunks = list(voice.synthesize("The quick brown fox jumps over the lazy dog."))
    if not chunks:
        pytest.skip("piper produced no audio")

    rate = chunks[0].sample_rate
    audio = np.concatenate(
        [np.frombuffer(c.audio_int16_bytes, dtype=np.int16) for c in chunks]
    ).astype(np.float32) / 32768.0

    # Silero is trained at 16 kHz and genuinely rate-sensitive: feeding it
    # 22.05 kHz audio scores real speech at 0.04.
    if rate != SAMPLERATE:
        audio = resample_poly(audio, SAMPLERATE, rate).astype(np.float32)
    return audio


# --------------------------------------------------------------- reframer


def test_an_exact_window_passes_straight_through():
    assert len(Reframer().push(np.zeros(WINDOW, dtype=np.float32))) == 1


def test_small_frames_accumulate_into_one_window():
    """The real case: 320-sample frames into 512-sample windows."""
    reframer = Reframer()
    out = [w for _ in range(2) for w in reframer.push(np.zeros(320, dtype=np.float32))]

    assert len(out) == 1                      # 640 samples in -> one window out
    assert len(out[0]) == WINDOW


def test_one_big_frame_yields_several_windows():
    out = Reframer().push(np.zeros(WINDOW * 3, dtype=np.float32))
    assert len(out) == 3


def test_a_frame_smaller_than_a_window_yields_nothing_yet():
    assert Reframer().push(np.zeros(100, dtype=np.float32)) == []


def test_a_stereo_frame_is_mixed_down():
    """The robot's mic is an array; Silero wants one channel."""
    stereo = np.ones((WINDOW, 2), dtype=np.float32)
    out = Reframer().push(stereo)

    assert len(out) == 1
    assert out[0].ndim == 1
    assert out[0] == pytest.approx(np.ones(WINDOW))


@given(sizes=st.lists(st.integers(min_value=1, max_value=2000), min_size=1, max_size=40))
@settings(deadline=None, max_examples=50)
def test_not_one_sample_is_lost_or_reordered(sizes):
    """Every emitted window, concatenated, must be a prefix of what went in."""
    reframer = Reframer()
    fed, out = [], []
    counter = 0
    for size in sizes:
        chunk = np.arange(counter, counter + size, dtype=np.float32)
        counter += size
        fed.append(chunk)
        out.extend(reframer.push(chunk))

    given_ = np.concatenate(fed)
    emitted = np.concatenate(out) if out else np.zeros(0, dtype=np.float32)
    assert emitted.tolist() == given_[: len(emitted)].tolist()


@given(sizes=st.lists(st.integers(min_value=1, max_value=5000), min_size=1, max_size=40))
@settings(deadline=None, max_examples=50)
def test_the_reframer_never_grows(sizes):
    """It runs for weeks - it must hold at most one partial window."""
    reframer = Reframer()
    for size in sizes:
        reframer.push(np.zeros(size, dtype=np.float32))
        assert reframer.held < WINDOW


def test_the_reframer_can_be_emptied():
    reframer = Reframer()
    reframer.push(np.zeros(100, dtype=np.float32))
    reframer.reset()

    assert reframer.held == 0


# ------------------------------------------------------------- detector


@pytest.fixture(scope="module")
def vad():
    if not MODEL.exists():
        pytest.skip("silero model not downloaded")
    return SileroVAD(MODEL)


def test_silence_is_not_speech(vad):
    vad.reset()
    probs = [vad.probability(np.zeros(WINDOW, dtype=np.float32)) for _ in range(20)]

    assert max(probs) < 0.3


def test_real_speech_is_speech(vad, real_speech):
    """Against actual speech, not a synthetic tone."""
    vad.reset()
    reframer = Reframer()
    probs = [vad.probability(w) for w in reframer.push(real_speech)]

    assert probs, "no windows produced"
    assert max(probs) > 0.7
    assert sum(p > 0.5 for p in probs) > len(probs) * 0.2


def test_a_tone_is_not_mistaken_for_a_voice(vad):
    """The advantage over an energy gate: loud is not the same as speech."""
    vad.reset()
    t = np.arange(WINDOW * 20) / 16000.0
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    probs = [vad.probability(w) for w in Reframer().push(tone)]

    assert max(probs) < 0.5, "a sine wave should not read as a voice"


def test_a_wrongly_sized_window_is_refused(vad):
    """Silence would be worse: the model returns nonsense rather than erroring."""
    with pytest.raises(ValueError, match="512"):
        vad.probability(np.zeros(320, dtype=np.float32))


def test_the_threshold_turns_a_probability_into_a_decision(vad):
    vad.reset()
    assert vad.is_speech(np.zeros(WINDOW, dtype=np.float32)) is False
