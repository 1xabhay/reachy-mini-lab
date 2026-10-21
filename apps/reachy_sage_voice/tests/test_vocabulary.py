"""The declared vocabularies against the robot's real recorded-moves library.

Reads the library from the local Hugging Face cache only, never the network;
skipped when it has not been downloaded (the robot daemon preloads it).
"""

import re

import pytest
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from reachy_mini.motion.recorded_move import DEFAULT_EMOTIONS_DATASET, RecordedMoves

from reachy_sage_voice.kernel import (
    DESCRIPTIONS,
    EMOTIONS,
    NO_SOUND,
    STATE_MOVES,
    describe_move,
    load_moves,
    manifest,
)

#: Moves the owner keeps out of the model's hands.
NEGATIVE = ("rage", "furious", "contempt", "disgusted", "dying", "reprimand", "irritated",
            "displeased", "frustrated")


@pytest.fixture(scope="module")
def library():
    try:
        snapshot_download(DEFAULT_EMOTIONS_DATASET, repo_type="dataset", local_files_only=True)
    except LocalEntryNotFoundError:
        pytest.skip(f"{DEFAULT_EMOTIONS_DATASET} is not in the local Hugging Face cache; "
                    "download it once (the robot daemon does) to run this contract")
    return RecordedMoves(DEFAULT_EMOTIONS_DATASET)


@pytest.fixture(scope="module")
def moves(library):
    return load_moves()


def test_every_declared_name_is_a_move_with_a_description_and_a_sound(library):
    for name in EMOTIONS + STATE_MOVES:
        move = library.get(name)
        assert move.description.strip(), f"{name} has no description"
        if name in NO_SOUND:
            assert move.sound_path is None, f"{name} has a sound now: take it off NO_SOUND"
        else:
            assert move.sound_path is not None and move.sound_path.exists(), f"{name}: no sound"


def test_every_declared_move_fits_max_s(moves):
    max_s = manifest(moves)["tools"][0]["max_s"]
    too_long = {name: m.duration for name, m in moves.items() if m.duration > max_s - 1}
    assert not too_long, f"moves longer than max_s - 1 s ({max_s} s): {too_long}"


def test_state_moves_are_not_offered_to_the_model(moves):
    enum = manifest(moves)["tools"][0]["inputSchema"]["properties"]["name"]["enum"]
    assert not set(STATE_MOVES) & set(enum)


def test_no_negative_move_is_offered():
    assert not [n for n in EMOTIONS if n.startswith(NEGATIVE)]


def test_every_variant_of_an_offered_move_is_offered_too(library):
    offered = set(EMOTIONS)
    bases = {re.sub(r"\d+$", "", n) for n in offered}
    missing = [n for n in library.list_moves()
               if re.sub(r"\d+$", "", n) in bases and n not in offered
               and not n.startswith(NEGATIVE)]
    assert not missing


def test_every_override_replaces_a_real_offered_move(library):
    for name in DESCRIPTIONS:
        assert name in library.list_moves() and name in EMOTIONS, name


def test_no_description_names_another_move(library, moves):
    names = library.list_moves()
    for name, move in moves.items():
        line = describe_move(name, move)
        others = [o for o in names if o != name and re.search(rf"\b{re.escape(o)}\b", line)]
        assert not others, f"{name}: {line!r} mentions {others}"


def test_no_description_is_a_recording_timestamp(moves):
    for name, move in moves.items():
        assert not describe_move(name, move).startswith("Recorded "), name
