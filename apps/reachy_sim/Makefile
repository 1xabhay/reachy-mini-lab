# Pollen's MuJoCo sim, headless, on localhost only. Never the robot.
PORT ?= 8742
SCENE ?= minimal

sim:
	uv run python sim.py $(PORT) $(SCENE)

test:
	uv run pytest -q

# The limits table, as in README.md.
limits:
	uv run pytest -q -s tests/test_limits.py

lint:
	uv run ruff check . && uv run ruff format --check .
