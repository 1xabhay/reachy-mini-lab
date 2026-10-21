"""When the bot must yield to a person who started talking.

Ported from an earlier Pipecat voice bot, where these
rules were measured in pilots (52 talk-overs before, then the floor added after
a group meeting where the bot was cancelled 244 ms into every greeting).

Two windows, deliberately different:

- the audio window (``bot_started`` / ``bot_stopped``) flickers between
  sentences while the next one is synthesised; talk-over is measured on it.
- the response window opens on the first ``bot_started`` and closes on
  ``assistant_turn_stopped``; enforcement is gated on it, so an onset in an
  inter-sentence gap still interrupts.

Every method takes ``t``, monotonic seconds from the caller's one clock.
"""

#: How long an onset with no matching stop still counts as someone talking.
#: Covers the model + TTS window between an onset and the bot's first audio;
#: older than this is a stop frame we never saw, and honouring it would mute
#: the bot for the rest of the session.
STALE_ONSET_S = 3.0


class InterruptionTracker:
    """Pure interruption policy: signals in, "yield now?" out."""

    def __init__(self, min_bot_speech_ms: float = 600) -> None:
        """Start idle. ``min_bot_speech_ms`` is the floor the bot always gets."""
        self.min_bot_speech_ms = min_bot_speech_ms
        self._response_audio_start: float | None = None
        self._bot_audio_on = False
        self._response_open = False
        self._overlap_start: float | None = None
        self._interrupted_this_response = False
        self._speaking: dict[str, float] = {}
        self.interruptions = 0
        self.talkovers_ms: list[float] = []

    def _arm(self, t: float) -> bool:
        """Mark this response interrupted; True the first time only."""
        if not self._response_open or self._interrupted_this_response:
            return False
        self._interrupted_this_response = True
        self._overlap_start = t
        self.interruptions += 1
        return True

    def _someone_talking(self, t: float) -> bool:
        # The lower bound rejects onsets stamped after t: a clock we do not share.
        return any(0.0 <= t - started <= STALE_ONSET_S for started in self._speaking.values())

    def _floor_spent(self, t: float) -> bool:
        if self._response_audio_start is None:
            return False  # nothing has played: cancelling would waste a whole reply
        return (t - self._response_audio_start) * 1000.0 >= self.min_bot_speech_ms

    def bot_started(self, t: float) -> bool:
        """Audio began, for the response or its next sentence. True: yield now."""
        self._bot_audio_on = True
        if not self._response_open:
            self._response_open = True
            self._overlap_start = None
            self._interrupted_this_response = False
        if self._response_audio_start is None:
            self._response_audio_start = t
        return self._floor_spent(t) and self._someone_talking(t) and self._arm(t)

    def bot_stopped(self, t: float) -> None:
        """Audio stopped; may be only the gap before the next sentence."""
        if self._bot_audio_on and self._overlap_start is not None:
            self.talkovers_ms.append(max(0.0, (t - self._overlap_start) * 1000.0))
            self._overlap_start = None
        self._bot_audio_on = False

    def assistant_turn_stopped(self, t: float) -> None:
        """Close the response; the next one may be interrupted afresh."""
        self._response_open = False
        self._overlap_start = None
        self._interrupted_this_response = False
        self._response_audio_start = None

    def user_onset(self, t: float, sid: str) -> bool:
        """Record that a person started talking; True, once per response, means yield."""
        self._speaking[sid] = t
        return self._floor_spent(t) and self._arm(t)

    def user_offset(self, t: float, sid: str) -> None:
        """Record that a person stopped talking."""
        self._speaking.pop(sid, None)
