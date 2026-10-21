"""Nothing here may build an SDK client that could find the real robot."""

import ast
from pathlib import Path

APP = Path(__file__).parent.parent


def unsafe_clients(source: str) -> list[int]:
    """Return the lines that construct ReachyMini without connection_mode="localhost_only"."""
    lines = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name != "ReachyMini":
            continue
        mode = next((k.value for k in node.keywords if k.arg == "connection_mode"), None)
        if not (isinstance(mode, ast.Constant) and mode.value == "localhost_only"):
            lines.append(node.lineno)
    return lines


def test_no_code_in_the_app_builds_a_client_that_could_leave_localhost():
    sources = [p for p in APP.rglob("*.py") if ".venv" not in p.parts]
    assert sources
    found = {str(p): unsafe_clients(p.read_text()) for p in sources}
    assert not {p: lines for p, lines in found.items() if lines}


def test_the_check_catches_a_client_without_localhost_only():
    bad = "ReachyMini()\nreachy_mini.ReachyMini(connection_mode='network')"
    assert unsafe_clients(bad) == [1, 2]
    assert unsafe_clients("ReachyMini(port=8742, connection_mode='localhost_only')") == []
