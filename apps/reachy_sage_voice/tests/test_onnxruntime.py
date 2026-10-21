"""Pipecat's two ONNX models we rely on, loaded and run once on reachy-mini's onnxruntime.

Pipecat pins onnxruntime ~=1.24.3; reachy-mini pins ==1.27.0. These are the
evidence that Silero VAD and Smart Turn v3.2 work on 1.27.0, on CPU.
"""

import numpy as np
import onnxruntime
from pipecat.audio.vad.silero import SileroVADAnalyzer

from reachy_sage_voice.pipeline import smart_turn_analyzer

RATE = 16000


def test_onnxruntime_is_the_one_reachy_mini_pins():
    assert onnxruntime.__version__ == "1.27.0"


def test_silero_vad_runs_on_cpu():
    vad = SileroVADAnalyzer(sample_rate=RATE)
    assert vad._model.session.get_providers() == ["CPUExecutionProvider"]
    frame = np.zeros(vad.num_frames_required(), dtype=np.int16).tobytes()
    confidence = vad.voice_confidence(frame)
    assert 0.0 <= confidence < 0.5  # silence is not speech


def test_smart_turn_v3_2_runs_on_cpu():
    analyzer = smart_turn_analyzer(stop_secs=1.0)
    assert analyzer._session.get_providers() == ["CPUExecutionProvider"]
    t = np.arange(2 * RATE) / RATE
    audio = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    result = analyzer._predict_endpoint(audio)
    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["probability"] <= 1.0
