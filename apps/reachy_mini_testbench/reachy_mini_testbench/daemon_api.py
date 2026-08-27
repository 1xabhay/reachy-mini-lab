"""Thin HTTP client for the Reachy Mini daemon.

The SDK needs a *running* backend before it will tell you anything.  These
calls work against the daemon itself, so diagnostics can still report
something useful when the robot is powered but idle - which is exactly the
state you are in when something has gone wrong.
"""

from __future__ import annotations

import os
from typing import Any

import requests

DEFAULT_HOST = os.environ.get("REACHY_MINI_HOST", "reachy-mini.local")
DEFAULT_PORT = int(os.environ.get("REACHY_MINI_PORT", "8000"))


class DaemonClient:
    """Minimal wrapper over the daemon's REST API."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, timeout: float = 5.0):
        """Point the client at a daemon."""
        self.base = f"http://{host}:{port}"
        self.host = host
        self.port = port
        self.timeout = timeout

    def _request(self, method: str, path: str, **kw: Any) -> Any:
        r = requests.request(method, f"{self.base}{path}", timeout=self.timeout, **kw)
        r.raise_for_status()
        return r.json() if r.content else None

    def get(self, path: str, **kw: Any) -> Any:
        """GET `path` and return the decoded JSON body."""
        return self._request("GET", path, **kw)

    def post(self, path: str, **kw: Any) -> Any:
        """POST to `path` and return the decoded JSON body."""
        return self._request("POST", path, **kw)

    def reachable(self) -> bool:
        """Report whether the daemon answers at all."""
        try:
            self.get("/api/daemon/status")
            return True
        except requests.RequestException:
            return False

    def status(self) -> dict[str, Any]:
        """Fetch daemon status: version, hardware id, backend state, IP."""
        return self.get("/api/daemon/status")

    def camera_specs(self) -> dict[str, Any]:
        """Fetch camera intrinsics and the resolutions the daemon can produce."""
        return self.get("/api/camera/specs")

    def backend_running(self) -> bool:
        """Report whether the motor backend is up (the SDK needs this)."""
        try:
            return self.status().get("state") == "running"
        except requests.RequestException:
            return False
