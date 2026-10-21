"""Calibrating the turn-end silence on the owner's own voice.

Ported from an earlier Pipecat voice bot's pause analysis. Each recording is
one turn spoken to the robot, so every silence inside it is a pause the robot
must wait through. The pure functions carry the decision; the VAD only feeds them.
"""

import wave

import numpy as np
from pipecat.audio.vad.vad_analyzer import VADState

from reachy_sage_voice.turn_end import (
    CANDIDATES_MS,
    choose_turn_end_ms,
    load_pcm16,
    pause_table,
    pauses_ms,
    speech_runs,
)

Q, St, Sp = VADState.QUIET, VADState.STARTING, VADState.SPEAKING
CHUNK_S = 0.032


def test_speech_runs_follow_the_vad_including_its_starting_edge():
    states = [Q, St, Sp, Sp, Q, Q, Sp, Q]
    assert speech_runs(states, CHUNK_S) == [(1 * CHUNK_S, 4 * CHUNK_S), (6 * CHUNK_S, 7 * CHUNK_S)]


def test_speech_running_to_the_end_is_closed_there():
    assert speech_runs([Q, Sp, Sp], 0.5) == [(0.5, 1.5)]


def test_pauses_are_the_silences_between_runs():
    assert pauses_ms([(0.0, 1.0), (1.3, 2.0), (2.45, 3.0)]) == [300.0, 450.0]


def test_the_table_counts_pauses_each_threshold_would_cut_a_turn_at():
    table = pause_table([100.0, 300.0, 500.0, 900.0], (200, 450, 800))
    assert table == [(200, 3, 0.75), (450, 2, 0.5), (800, 1, 0.25)]


def test_the_chosen_turn_end_waits_through_almost_every_real_pause():
    pauses = [120.0] * 50 + [380.0] * 45 + [700.0] * 5  # 5% are long thinking pauses
    assert choose_turn_end_ms(pauses, tolerance=0.05) == 450
    assert choose_turn_end_ms(pauses, tolerance=0.0) == 800


def test_with_no_recordings_the_default_stands():
    assert choose_turn_end_ms([]) == 450


def test_candidates_include_the_default():
    assert 450 in CANDIDATES_MS


def test_recordings_load_as_mono_16k(tmp_path):
    path = tmp_path / "turn.wav"
    stereo = np.tile(np.array([[1000, 3000]], dtype=np.int16), (48000, 1))  # 1 s at 48 kHz
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(stereo.tobytes())
    pcm = load_pcm16(str(path))
    assert pcm.dtype == np.int16 and len(pcm) == 16000 and int(pcm[0]) == 2000


def test_the_script_runs_silero_over_recordings_and_prints_the_setting(tmp_path, capsys):
    from reachy_sage_voice.turn_end import main

    path = tmp_path / "quiet.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.zeros(16000, dtype=np.int16).tobytes())
    main([str(path)])
    assert capsys.readouterr().out.strip().endswith("SAGE_VOICE_TURN_END_MS=450")
