"""Contract checks: the stubs in `conftest` must not drift from the real thing.

Everything else in the pet's suite runs against test doubles, which means the
whole suite can stay green while an SDK upgrade quietly renames the method the
pet actually calls. These tests are the tie-back: they assert, against the
installed packages, that the surface the pet depends on still exists and still
takes the arguments the pet passes.

They introspect only - nothing is instantiated, no network, no robot - so they
run in the same laptop-only way as the rest of the suite.
"""

import inspect

import pytest
from conftest import FakeMoves, FakePetMedia, FakePetMini
from reachy_mini import ReachyMini
from reachy_mini.io.protocol import FaceTarget
from reachy_mini.media.media_manager import MediaManager
from reachy_mini.motion.recorded_move import RecordedMoves


def accepts(func, name: str) -> bool:
    """Whether a callable takes a parameter of that name."""
    return name in inspect.signature(func).parameters


# ------------------------------------------------------------- the robot


@pytest.mark.parametrize(
    "method",
    [
        "wake_up",
        "goto_sleep",
        "enable_wobbling",
        "disable_wobbling",
        "start_head_tracking",
        "stop_head_tracking",
        "get_tracked_face",
        "play_move",
        "media",
    ],
)
def test_the_robot_still_has_what_the_pet_calls(method):
    assert hasattr(ReachyMini, method)


def test_head_tracking_still_takes_a_blend_weight():
    """The pet blends tracking down so emotion moves are not flattened."""
    assert accepts(ReachyMini.start_head_tracking, "weight")


def test_playing_a_move_can_still_suppress_its_canned_sound():
    """Without `sound=False` the recorded chirp talks over the spoken reply."""
    assert accepts(ReachyMini.play_move, "sound")


def test_the_face_target_can_still_be_read_without_blocking():
    assert accepts(ReachyMini.get_tracked_face, "wait")


def test_a_face_target_still_reports_whether_it_saw_anyone():
    assert "detected" in FaceTarget.model_fields


@pytest.mark.parametrize(
    "method",
    [
        "start_recording",
        "stop_recording",
        "get_audio_sample",
        "get_input_audio_samplerate",
        "play_sound",
    ],
)
def test_the_media_manager_still_has_what_the_pet_calls(method):
    assert hasattr(MediaManager, method)


def test_the_microphone_still_hands_back_float32():
    """The gate's thresholds are calibrated for float32 in [-1, 1].

    If this ever became int16 the levels would be four orders of magnitude
    out, the gate would jam permanently open, and every utterance would be a
    15-second slab of noise - silently, with no error anywhere.
    """
    annotation = str(inspect.signature(MediaManager.get_audio_sample).return_annotation)

    assert "float32" in annotation


def test_the_emotions_library_still_looks_the_same():
    assert hasattr(RecordedMoves, "get")
    assert hasattr(RecordedMoves, "list_moves")


# --------------------------------------------------------- the inference APIs


def test_faster_whisper_still_takes_the_options_the_pet_relies_on():
    """These are what stop it inventing words from non-speech."""
    from faster_whisper import WhisperModel

    parameters = inspect.signature(WhisperModel.transcribe).parameters
    for option in (
        "temperature",
        "condition_on_previous_text",
        "vad_filter",
        "vad_parameters",
        "beam_size",
        "language",
    ):
        assert option in parameters, option


def test_a_transcribed_segment_still_reports_its_own_doubt():
    """The pet reads these to refuse a decode it should not believe."""
    import dataclasses

    from faster_whisper.transcribe import Segment, TranscriptionInfo

    segment = {f.name for f in dataclasses.fields(Segment)}
    info = {f.name for f in dataclasses.fields(TranscriptionInfo)}

    assert {"no_speech_prob", "compression_ratio", "avg_logprob"} <= segment
    assert "duration_after_vad" in info


def test_piper_still_streams_audio_in_chunks():
    """Streaming is what lets the pet start speaking before the reply is done."""
    from piper import PiperVoice

    assert hasattr(PiperVoice, "load")
    assert hasattr(PiperVoice, "synthesize")


def test_silero_still_wants_the_window_size_the_pet_feeds_it():
    from reachy_mini_pet.vad import CONTEXT, SAMPLERATE, WINDOW

    assert (WINDOW, CONTEXT, SAMPLERATE) == (512, 64, 16000)


# ------------------------------------------------- the doubles match the real


@pytest.mark.parametrize(
    ("double", "real"),
    [
        (FakePetMini, ReachyMini),
        (FakePetMedia, MediaManager),
        (FakeMoves, RecordedMoves),
    ],
)
def test_every_stubbed_method_exists_on_the_real_class(double, real):
    """A stub with a method the real class lacks is a test that proves nothing."""
    stubbed = {
        name
        for name, value in vars(double).items()
        if callable(value) and not name.startswith("_")
    }
    # Bookkeeping the stubs add for the tests' benefit, not part of any contract.
    stubbed -= {"names"}

    assert stubbed <= set(dir(real)), f"{double.__name__} invents: {stubbed - set(dir(real))}"
