"""The real robot and the local models, wired to the layers.

Everything here is an adapter: the layers do not know that hearing is Silero,
that the mind is a language model on this machine, or that the voice is Piper.
Swap any one of these for something else and no layer changes.

Nothing leaves the machine.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np
from reachy_mini import ReachyMiniApp
from reachy_mini.motion.recorded_move import DEFAULT_EMOTIONS_DATASET, RecordedMoves

from .bus import Bus, Subscription
from .conductor import Conductor
from .events import Emote
from .following import BodyYawFollower
from .health import SessionHealth
from .layers import ROBOT_GONE, Hearing, Mind, Module, Transcriber, Vision, Voice
from .streaming import clamp_clauses, sentences, take_emotion
from .supervisor import Supervisor
from .vad import SAMPLERATE, WINDOW, Reframer, SileroVAD
from .watchdog import Watchdog

logger = logging.getLogger("reachy_mini.pet.live")

HOST = os.environ.get("REACHY_MINI_HOST", "localhost")
PORT = int(os.environ.get("REACHY_MINI_PORT", "8000"))
DATA = Path(os.environ.get("REACHY_PET_DATA_DIR", "~/.reachy_pet")).expanduser()

VAD_MODEL = DATA / "silero_vad.onnx"
PIPER_VOICE = DATA / "voices" / os.environ.get("REACHY_PET_VOICE", "en_GB-jenny_dioco-medium.onnx")
WHISPER_SIZE = os.environ.get("REACHY_PET_WHISPER", "base.en")
# Chosen by measurement, not by size - see `reachy-pet-bench`. Against the
# pet's own prompts this named a valid emotion 8/8 where qwen2.5:7b managed
# 6/8, replied three times faster (0.23s vs 0.69s) and used 1.9 GB instead of
# 4.7 GB. A pet says one or two short sentences; a 7B is not paying for itself.
OLLAMA_MODEL = os.environ.get("REACHY_PET_LLM", "qwen3:1.7b")
OLLAMA_URL = os.environ.get("REACHY_PET_OLLAMA", "http://localhost:11434/api/chat")

#: Body language the pet can choose from, out of the 85 recorded moves. Curated
#: so it stays good company - the library also holds `dying1` and `rage1`.
EMOTIONS = (
    "amazed1",
    "attentive1",
    "boredom1",
    "calming1",
    "cheerful1",
    "confused1",
    "curious1",
    "dance1",
    "enthusiastic1",
    "grateful1",
    "helpful1",
    "impatient1",
    "inquiring1",
    "laughing1",
    "loving1",
    "no1",
    "oops1",
    "proud1",
    "relief1",
    "sad1",
    "scared1",
    "serenity1",
    "shy1",
    "success1",
    "surprised1",
    "thoughtful1",
    "tired1",
    "understanding1",
    "welcoming1",
    "yes1",
)

SYSTEM = (
    "You are a small desk robot, someone's pet and everyday assistant. You hear "
    "them through your own microphone, so what you receive is speech-to-text and "
    "may be misheard - if it is garbled, say so rather than inventing an answer.\n\n"
    "Answer in exactly this shape:\n"
    "EMOTION: <one name from the list>\n"
    "<what to say out loud>\n\n"
    "The emotion is your body language and it is most of how you come across, so "
    "pick the one that honestly matches what you just heard. Match their mood "
    "rather than performing at them. Choose from: " + ", ".join(EMOTIONS) + ".\n"
    "Never say the emotion name out loud - it belongs on the first line only, "
    "and it is a movement, not a word.\n\n"
    "Then speak the way a fond, unhurried friend talks out loud: one or two short "
    "sentences, no lists, no markdown, no emoji. It is going through a speech "
    "synthesiser, so write only what should be said."
)


class RobotEyes:
    """The camera, through the daemon's own face tracker."""

    def __init__(self, mini) -> None:
        """Take a connected robot."""
        self.mini = mini

    def look(self) -> tuple[bool, float | None, float | None]:
        """Return whether a face is in view, and where."""
        face = self.mini.get_tracked_face(wait=False)
        return bool(face.detected), face.x, face.y

    def turn_body(self, yaw: float) -> None:
        """Turn the body, which the daemon's tracker never does."""
        self.mini.set_target_body_yaw(float(yaw))


class RobotEars:
    """The microphone, re-framed to the windows Silero insists on.

    This is also where the pet defends itself against going quietly deaf. The
    WebRTC audio stream does die - it timed out after forty minutes of working
    perfectly, and because a dead microphone is indistinguishable from a silent
    room, nothing else in the system could notice.
    """

    def __init__(self, mini, audio_timeout_s: float = 8.0) -> None:
        """Take a connected robot and start recording."""
        self.mini = mini
        self.samplerate = mini.media.get_input_audio_samplerate()
        self._reframer = Reframer()
        self._ready: list[np.ndarray] = []
        self._watchdog = Watchdog(timeout_s=audio_timeout_s, now=time.monotonic())
        mini.media.start_recording()

    def listen(self) -> np.ndarray | None:
        """Return one 512-sample window, or None if the mic has nothing yet."""
        if self._ready:
            return self._ready.pop(0)

        now = time.monotonic()
        frame = self.mini.media.get_audio_sample()
        if frame is None:
            if self._watchdog.should_recover(now):
                self._recover()
            return None

        self._watchdog.feed(now)
        self._ready.extend(self._reframer.push(frame))
        return self._ready.pop(0) if self._ready else None

    def _recover(self) -> None:
        """Restart the audio capture, escalating if a gentle nudge did not work."""
        attempt = self._watchdog.recoveries
        logger.warning(
            "microphone silent for %.0fs - recovery attempt %d",
            self._watchdog.timeout_s,
            attempt,
        )
        self._reframer.reset()
        self._ready.clear()
        try:
            if attempt <= 1:
                # Cheap first: the stream may just need re-opening.
                self.mini.media.stop_recording()
                self.mini.media.start_recording()
            else:
                # The pipeline itself is gone; rebuild it.
                self.mini.release_media()
                self.mini.acquire_media()
                self.samplerate = self.mini.media.get_input_audio_samplerate()
                self.mini.media.start_recording()
        except Exception:
            logger.exception("microphone recovery failed")


class RobotMouth:
    """Piper, played through the robot's speaker."""

    def __init__(self, mini, voice_path: Path) -> None:
        """Load the voice once; it takes about half a second."""
        from piper import PiperVoice

        self.mini = mini
        self.voice = PiperVoice.load(voice_path)
        # Per instance: a leaked voice thread from a previous session could
        # otherwise overwrite this one between the write and the play.
        self._path = Path(tempfile.mkdtemp(prefix="reachy_pet_")) / "voice.wav"

    def say(self, text: str) -> None:
        """Synthesise one clause and play it, waiting until it is done."""
        chunks = list(self.voice.synthesize(text))
        if not chunks:
            return
        rate = chunks[0].sample_rate
        audio = np.concatenate([np.frombuffer(c.audio_int16_bytes, dtype=np.int16) for c in chunks])
        with wave.open(str(self._path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            handle.writeframes(audio.tobytes())
        self.mini.media.play_sound(str(self._path))
        time.sleep(len(audio) / rate + 0.2)


class JsonMemory:
    """The conversation, kept in a file so a restart still knows you.

    Deliberately the same path the first implementation used, so a pet that has
    been talked to before does not lose what it already knew.
    """

    def __init__(self, path: Path | None = None) -> None:
        """Take the file to keep the conversation in."""
        self.path = path if path is not None else DATA / "history.json"

    def load(self) -> list[dict]:
        """Read the conversation back, or start fresh if there is not one."""
        try:
            stored = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return []
        if not isinstance(stored, list):
            logger.warning("ignoring a memory file that is not a conversation")
            return []
        return [
            turn
            for turn in stored
            if isinstance(turn, dict) and turn.get("role") in ("user", "assistant")
        ]

    def save(self, history: list[dict]) -> None:
        """Write the conversation down."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(history))


class Body(Module):
    """Plays recorded body language when the mind asks for it."""

    name = "body"
    period = 0.05

    def __init__(self, mini, moves) -> None:
        """Take a connected robot and a library of recorded moves."""
        super().__init__()
        self.mini = mini
        self.moves = moves
        self._inbox: Subscription | None = None

    def subscribe(self, bus: Bus) -> None:
        """Listen for feelings."""
        self._inbox = bus.subscribe(Emote)

    def step(self, now: float = 0.0) -> None:
        """Play whatever was asked for, most recent only."""
        if self.bus is None or self._inbox is None:
            return
        for emote in self.bus.drain(self._inbox)[-1:]:
            try:
                self.mini.play_move(self.moves.get(emote.name), sound=False)
            except ROBOT_GONE:
                raise
            except Exception:
                logger.exception("body: could not play %s", emote.name)


def local_transcriber(size: str = WHISPER_SIZE):
    """Build a speech-to-text function backed by faster-whisper.

    Whisper fabricates text when handed non-speech, and its own thresholds do
    not stop it - they only mark a decode as suspect, re-run it hotter, and
    then return the least-bad attempt anyway. So three things are done here:
    the temperature ladder is switched off (it was what produced the "can't see
    it, can't see it..." loop), its own VAD runs as a second opinion over the
    whole clip, and anything that comes back is checked before it is believed.
    """
    from faster_whisper import WhisperModel

    model = WhisperModel(size, device="cpu", compute_type="int8")

    def transcribe(audio: np.ndarray, samplerate: int) -> str:
        segments, info = model.transcribe(
            np.asarray(audio, dtype=np.float32),
            beam_size=1,
            language="en",
            # A single temperature disables the fallback ladder: re-rolling a
            # bad decode at rising temperature is what invents fluent nonsense.
            temperature=0.0,
            # Free here (each utterance is its own call, well under Whisper's
            # 30s window) but correct, and insurance if that ever changes.
            condition_on_previous_text=False,
            without_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 200},
        )

        # Its VAD runs eagerly while decoding is lazy, so this is known before
        # any transcription work happens. Nothing but silence means nothing
        # was said, whatever the decoder would have invented from it.
        if getattr(info, "duration_after_vad", None) == 0:
            return ""

        kept = []
        for segment in segments:
            if getattr(segment, "no_speech_prob", 0.0) > 0.6:
                continue
            if getattr(segment, "compression_ratio", 0.0) > 2.4:
                continue
            kept.append(segment.text)
        return " ".join(kept).strip()

    return transcribe


def local_mind(model: str = OLLAMA_MODEL, emotions: tuple[str, ...] = EMOTIONS):
    """Build a think function backed by a language model on this machine.

    A pet wants a fast, short, in-character answer, which is close to the
    opposite of what a reasoning model does. Asked to reason, `qwen3:1.7b`
    spent its entire token budget inside `<think>` tags and returned nothing
    at all, so thinking is switched off where the server supports it - which
    also made it five times faster.
    """
    # Not every Ollama build, and not every model, accepts `think`: a model
    # that cannot reason returns 400 for it. Ask once, remember the answer.
    no_thinking = {"supported": True}

    def stream(messages):
        body = {
            "model": model,
            "stream": True,
            "messages": messages,
            "keep_alive": "1h",  # an always-on pet must not pay a cold start
            "options": {"temperature": 0.7, "num_predict": 200},
        }
        if no_thinking["supported"]:
            body["think"] = False

        request = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            response = urllib.request.urlopen(request, timeout=120)
        except urllib.error.HTTPError as error:
            if error.code != 400 or not no_thinking["supported"]:
                raise
            # This model or server does not know about `think`. Drop it and
            # carry on; the reasoning traces get stripped later anyway.
            logger.info("%s does not accept think=false; leaving it on", model)
            no_thinking["supported"] = False
            body.pop("think")
            request = urllib.request.Request(
                OLLAMA_URL,
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            response = urllib.request.urlopen(request, timeout=120)

        with response:
            for line in response:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                piece = chunk.get("message", {}).get("content", "")
                if piece:
                    yield piece
                if chunk.get("done"):
                    return

    def think(text: str, history: list) -> tuple[list[str], str]:
        messages = [
            {"role": "system", "content": SYSTEM},
            *history,
            {"role": "user", "content": text},
        ]
        feeling, speech = take_emotion(stream(messages), emotions or EMOTIONS)
        return clamp_clauses(sentences([speech])), feeling

    return think


def build(mini, bus: Bus | None = None) -> tuple[Bus, list[Module]]:
    """Assemble every layer against a real robot. Start only what you want."""
    bus = bus if bus is not None else Bus()
    vad = SileroVAD(VAD_MODEL)
    ears = RobotEars(mini)

    # Silero and Whisper are both trained at 16 kHz and neither says anything
    # when given another rate - it just scores real speech near zero and
    # transcribes chipmunk noise. Refuse loudly instead of going quietly deaf.
    if ears.samplerate != SAMPLERATE:
        raise RuntimeError(
            f"microphone is {ears.samplerate} Hz; hearing needs {SAMPLERATE} Hz"
        )
    logger.info("mic %d Hz, watchdog %.0fs", ears.samplerate, ears._watchdog.timeout_s)

    moves = RecordedMoves(DEFAULT_EMOTIONS_DATASET)
    available = tuple(e for e in EMOTIONS if e in moves.list_moves())

    modules: list[Module] = [
        Vision(RobotEyes(mini), BodyYawFollower()),
        Hearing(
            ears,
            is_speech=vad.is_speech,
            conductor=Conductor(frame_s=WINDOW / ears.samplerate, hangover_s=0.6),
            reset_detector=vad.reset,
        ),
        Transcriber(local_transcriber()),
        Mind(local_mind(emotions=available), memory=JsonMemory()),
        Voice(RobotMouth(mini, PIPER_VOICE)),
        Body(mini, moves),
    ]
    for module in modules:
        module.start(bus)
    return bus, modules


def run(
    mini,
    modules: list[Module],
    stop: threading.Event,
    rate: float = 0.005,
    health: SessionHealth | None = None,
) -> None:
    """Give each layer a thread that does nothing but call `step`.

    The threads also report their own failures, which is how a dead command
    channel gets noticed. The daemon can answer HTTP perfectly while the
    WebSocket that carries commands is gone, so a layer failing is better
    evidence that the session is over than any status endpoint.
    """
    watch = health if health is not None else SessionHealth()

    def loop(module: Module) -> None:
        while not stop.is_set():
            try:
                module.step(now=time.monotonic())
                watch.record_success(time.monotonic())
            except Exception as error:
                watch.record_error(error, time.monotonic())
                if watch.should_report():
                    logger.exception("%s: step failed", module.name)
                if watch.is_broken():
                    logger.warning("lost the robot (%s) - ending session", watch.reason)
                    stop.set()
                    return
            # Each layer sets its own pace. One global rate meant the fastest
            # need - hearing, which must not miss a frame - also drove vision,
            # which was then sending hundreds of redundant commands a second
            # down the channel this pet keeps losing.
            time.sleep(max(rate, getattr(module, "period", 0.0)))

    threads = [threading.Thread(target=loop, args=(m,), daemon=True, name=m.name) for m in modules]
    for thread in threads:
        thread.start()
    stop.wait()
    for thread in threads:
        thread.join(timeout=2.0)


def daemon_alive(timeout_s: float = 3.0) -> bool:
    """Whether the robot's own daemon is answering.

    This is the thing that actually died: the process stayed up and the port
    stayed open, but requests never returned. So the check has to be a real
    request with a real timeout, not a connection test.
    """
    try:
        url = f"http://{HOST}:{PORT}/api/daemon/status"
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            return response.status == 200
    except Exception:
        return False


def session(mini, stop: threading.Event, heartbeat_s: float = 5.0) -> None:
    """Run every layer until told to stop, or until the robot stops answering."""
    # Waking up and enabling tracking are gestures, not requirements. Either
    # can time out if the robot is still settling from a previous session, and
    # tearing the whole session down over a cosmetic move would mean the pet
    # never gets started on exactly the occasions it most needs to.
    for gesture, action in (
        ("wake up", mini.wake_up),
        ("start tracking", lambda: mini.start_head_tracking(weight=1.0)),
    ):
        try:
            action()
        except Exception as error:
            logger.warning("could not %s (%s) - carrying on", gesture, error)

    _bus, modules = build(mini)
    ended = threading.Event()
    health = SessionHealth()

    worker = threading.Thread(
        target=run, args=(mini, modules, ended), kwargs={"health": health}, daemon=True
    )
    worker.start()
    try:
        waited = 0.0
        while not stop.is_set() and not ended.is_set():
            if waited >= heartbeat_s:
                waited = 0.0
                if not daemon_alive():
                    logger.warning("robot stopped answering - ending this session")
                    return
            stop.wait(0.5)
            waited += 0.5
        if ended.is_set():
            logger.warning("a layer lost the robot - ending this session")
    finally:
        ended.set()
        worker.join(timeout=3.0)
        for module in modules:
            module.stop()


def serve(stop: threading.Event, supervisor: Supervisor | None = None) -> None:
    """Keep a pet alive against a robot that may come and go.

    A robot that dies is not the end of the pet; it is a gap. This waits it
    out, backing off so a dead robot is not hammered, and picks up again the
    moment it answers - including after a power cycle, with nobody watching.
    """
    from reachy_mini import ReachyMini

    boss = supervisor if supervisor is not None else Supervisor()
    while not stop.is_set():
        now = time.monotonic()
        if not boss.should_retry(now):
            stop.wait(0.5)
            continue

        if not daemon_alive():
            boss.failed(time.monotonic())
            if (change := boss.take_transition()) is not None:
                logger.warning("robot is %s", change.value)
            continue

        try:
            with ReachyMini(host=HOST, port=PORT, connection_mode="network") as mini:
                boss.connected(time.monotonic())
                if (change := boss.take_transition()) is not None:
                    logger.info("robot is %s (reconnections: %d)", change.value, boss.reconnections)
                session(mini, stop)
        except Exception:
            logger.exception("session ended badly")
        finally:
            if not stop.is_set():
                boss.failed(time.monotonic())


class LivePetApp(ReachyMiniApp):
    """The pet as a Reachy Mini app, so the robot's own daemon can launch it.

    The registered entry point used to point at the older cloud implementation,
    which needed API keys the local pet no longer wants. This is the one that
    should run.
    """

    def run(self, reachy_mini, stop_event: threading.Event) -> None:
        """Hold a conversation until stopped, reconnecting if the robot dies."""
        session(reachy_mini, stop_event)


def main() -> None:
    """Run the supervised pet outside the robot's app manager."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("reachy_mini", "faster_whisper", "urllib3", "httpx", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # ...but not our own, which is a child of "reachy_mini".
    logging.getLogger("reachy_mini.pet").setLevel(logging.INFO)

    stop = threading.Event()
    try:
        serve(stop)
    except KeyboardInterrupt:
        stop.set()


if __name__ == "__main__":
    main()
