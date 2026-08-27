"""On-disk storage for testbench artefacts (captures, recordings, reports)."""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(os.environ.get("TESTBENCH_DATA_DIR", "/tmp/reachy_mini_testbench"))

CAPTURES = ROOT / "captures"
RECORDINGS = ROOT / "recordings"
REPORTS = ROOT / "reports"


def init() -> None:
    """Create the storage tree if missing."""
    for d in (CAPTURES, RECORDINGS, REPORTS):
        d.mkdir(parents=True, exist_ok=True)


def stamp() -> str:
    """Timestamp suitable for filenames (sorts chronologically).

    Deliberately local time: these names are read by whoever is standing next
    to the robot, and the ordering only ever matters within one machine.
    """
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def safe_path(directory: Path, filename: str) -> Path:
    """Resolve `filename` inside `directory`, rejecting traversal attempts."""
    candidate = (directory / filename).resolve()
    if candidate.parent != directory.resolve():
        raise ValueError(f"illegal filename: {filename!r}")
    return candidate


def listing(directory: Path, suffixes: tuple[str, ...]) -> list[dict[str, object]]:
    """List files in `directory` with the given suffixes, newest first."""
    if not directory.exists():
        return []
    entries = [
        {
            "filename": p.name,
            "size": p.stat().st_size,
            "modified": datetime.fromtimestamp(p.stat().st_mtime)
            .astimezone()
            .isoformat(timespec="seconds"),
        }
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in suffixes
    ]
    return sorted(entries, key=lambda e: str(e["modified"]), reverse=True)


def disk_free_mb() -> float:
    """Free space (MiB) on the volume holding the storage tree."""
    init()
    return shutil.disk_usage(ROOT).free / (1024 * 1024)
