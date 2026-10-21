test:
	uv run pytest -q

# The voice against the real Sage kernel on a free loopback port, with its conformance
# fake model. CPU only. The kernel is SAGE_KERNEL_DIR, default the sibling sage-kernel checkout.
integration:
	uv run pytest -q -m integration tests/integration

# The voice on the robot. Reads the environment and .env: see README.md, "Settings".
run:
	uv run python -m reachy_sage_voice

# First audio per clause rule, against a running Kokoro (make -C services up first).
# Results go in sage-kernel bench/LEDGER.md, "Voice first audio".
bench-first-audio:
	SAGE_VOICE_TTS_URL=$${SAGE_VOICE_TTS_URL:-http://127.0.0.1:8782} uv run python bench/first_audio.py
