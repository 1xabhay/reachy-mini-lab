test:
	uv run pytest -q

# The voice on the robot. Reads only the environment: see README.md, "Run it".
run:
	uv run python -m reachy_sage_voice

# First audio per clause rule, against a running Kokoro (make -C services up first).
# Results go in sage-kernel bench/LEDGER.md, "Voice first audio".
bench-first-audio:
	SAGE_VOICE_TTS_URL=$${SAGE_VOICE_TTS_URL:-http://127.0.0.1:8782} uv run python bench/first_audio.py
