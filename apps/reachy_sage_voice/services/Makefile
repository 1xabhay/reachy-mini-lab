# Native runs keep the Hugging Face cache inside this folder.
export HF_HOME := $(CURDIR)/.local/hf

test:
	uv run python -m unittest discover -s stt -t stt
	uv run python -m unittest discover -s tts -t tts

# Each image installs only its own group, with hashes.
lock:
	uv lock
	uv export --frozen --no-default-groups --group stt --no-emit-project -o stt/requirements.lock
	uv export --frozen --no-default-groups --group tts --no-emit-project -o tts/requirements.lock

# The contract tests again, inside the sealed images.
test-docker:
	docker compose build
	docker compose run --rm --no-deps stt python -m unittest test_app
	docker compose run --rm --no-deps tts python -m unittest test_app

up:
	docker compose up -d --build

native-stt:
	PORT=8791 uv run python stt/app.py

native-tts:
	cd tts && PORT=8792 uv run python app.py
