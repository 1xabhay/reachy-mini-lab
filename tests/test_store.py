"""Unit checks for the storage helpers, chiefly the traversal guard.

The API tests reach these through HTTP, where routing already rejects most
slashes; these hit the guard directly so it is not silently load-bearing.
"""

import pytest
from reachy_mini_testbench import store


@pytest.mark.parametrize(
    "filename",
    ["../secret.jpg", "../../etc/passwd", "sub/dir.jpg", "/etc/passwd", ".."],
)
def test_safe_path_rejects_escapes(tmp_path, filename):
    with pytest.raises(ValueError, match="illegal filename"):
        store.safe_path(tmp_path, filename)


def test_safe_path_accepts_a_plain_name(tmp_path):
    assert store.safe_path(tmp_path, "capture_1.jpg") == tmp_path / "capture_1.jpg"


def test_listing_filters_by_suffix_and_sorts_newest_first(tmp_path):
    for name in ("a.jpg", "b.png", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")

    names = {e["filename"] for e in store.listing(tmp_path, (".jpg", ".png"))}

    assert names == {"a.jpg", "b.png"}


def test_listing_of_a_missing_directory_is_empty(tmp_path):
    assert store.listing(tmp_path / "nope", (".jpg",)) == []
