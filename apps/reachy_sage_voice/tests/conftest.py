"""A scripted device-protocol v1 server on localhost, for the HTTP client's tests.

Each route answers from its own queue of (status, body) replies, or of
callables that build one from what was sent so far; the last one repeats.
Every request is recorded as (method, path, query, headers, body).
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest
from pipecat.tests import utils as pipecat_test_utils

TOKEN = "t" * 32

# Pipecat's run_test already waits on the pipeline's started event, but gives up
# after 1 s, which a busy machine misses. Same wait, a generous ceiling: a slow
# start passes, and only a pipeline that never starts fails.
pipecat_test_utils.run_test.__kwdefaults__["start_timeout"] = 10.0


@pytest.fixture(autouse=True)
def no_real_dotenv(monkeypatch):
    """No test reads a real .env: the only settings files are the ones a test makes."""
    import reachy_sage_voice.__main__ as voice

    monkeypatch.setattr(voice, "env_files", lambda: [])


class FakeKernel:
    def __init__(self):
        self.replies: dict[tuple[str, str], list[tuple[int, object]]] = {}
        self.requests: list[dict] = []
        self._lock = threading.Lock()

    def script(self, method: str, path: str, *replies: tuple[int, object]) -> None:
        self.replies[(method, path)] = list(replies)

    def answer(self, method: str, path: str) -> tuple[int, object]:
        with self._lock:
            queue = self.replies.get((method, path)) or [(404, {"code": "not_found"})]
            reply = queue.pop(0) if len(queue) > 1 else queue[0]
        return reply(self) if callable(reply) else reply

    def events(self, kind: str | None = None) -> list[dict]:
        return [r["body"] for r in self.requests
                if r["path"] == "/v1/events" and (kind is None or r["body"]["kind"] == kind)]


@pytest.fixture
def kernel():
    fake = FakeKernel()

    class Handler(BaseHTTPRequestHandler):
        def _serve(self):
            url = urlparse(self.path)
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            fake.requests.append({
                "method": self.command, "path": url.path, "query": parse_qs(url.query),
                "headers": dict(self.headers), "body": json.loads(raw) if raw else None})
            status, body = fake.answer(self.command, url.path)
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        do_GET = do_POST = _serve  # noqa: N815 - http.server names them

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True)
    thread.start()
    fake.url = f"http://127.0.0.1:{server.server_port}"
    yield fake
    server.shutdown()
    server.server_close()


@pytest.fixture
def http_server(kernel):
    """Serve the same scripted routes, for services other than the kernel."""
    return kernel
