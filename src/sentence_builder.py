"""
sentence_builder.py -- Word-level sentence accumulation with debouncing.

Replaces the character-level Translator for the ISL word recognition system.
Accumulates recognized ISL WORDS (not letters) into sentences with
intelligent debouncing: a sign must be stable, then the user must return
to neutral before the next word can be added.

State machine:
  IDLE -> RECOGNIZING -> CONFIRMED -> WAITING_NEUTRAL -> IDLE

  IDLE:             No sign detected
  RECOGNIZING:      A sign is being held, waiting for stability
  CONFIRMED:        Sign was stable long enough, word added to sentence
  WAITING_NEUTRAL:  Waiting for user to return hand to neutral/no-sign
"""

import time
from typing import Optional, List


class SentenceBuilder:
    """
    Accumulates recognized ISL words into a sentence.

    Key behaviors:
      - A sign must be stable for `stability_seconds` before being accepted.
      - After a word is accepted, the system waits for the user to return
        to a neutral state (no sign / different sign) before accepting
        the next word.
      - Same word cannot be added twice in a row unless the user
        explicitly returns to neutral between performances.
      - Keyboard controls: space, backspace (word), clear.
    """

    # States
    STATE_IDLE = "IDLE"
    STATE_RECOGNIZING = "RECOGNIZING"
    STATE_CONFIRMED = "CONFIRMED"
    STATE_WAITING_NEUTRAL = "WAITING_NEUTRAL"

    def __init__(
        self,
        stability_seconds: float = 1.0,
        neutral_timeout: float = 0.5,
    ):
        """
        Args:
            stability_seconds: How long a sign must be consistently recognized
                               before it is accepted as a word.
            neutral_timeout: How long no-sign / uncertain must persist before
                             the system returns to IDLE.
        """
        self.stability_seconds = stability_seconds
        self.neutral_timeout = neutral_timeout

        self._words: List[str] = []
        self._state = self.STATE_IDLE

        # Tracking
        self._current_sign = ""
        self._stable_since = 0.0
        self._last_confirmed_sign = ""
        self._neutral_since = 0.0

    @property
    def state(self) -> str:
        """Current state of the sentence builder."""
        return self._state

    @property
    def words(self) -> List[str]:
        """List of accumulated words."""
        return self._words.copy()

    @property
    def text(self) -> str:
        """Get the accumulated sentence as a string."""
        return " ".join(self._words)

    def update(self, sign: str, confidence: float) -> Optional[str]:
        """
        Process a new frame's recognition result.

        Call this every frame with the smoothed prediction from the classifier.

        Args:
            sign: Recognized sign label, or "" / "Uncertain" / "No hand detected"
            confidence: Prediction confidence.

        Returns:
            The word that was added to the sentence, or None.
        """
        now = time.time()
        is_valid_sign = bool(
            sign and sign not in ("", "Uncertain", "No hand detected", "_NEUTRAL")
        )

        # ---- STATE: WAITING_NEUTRAL ----
        if self._state == self.STATE_WAITING_NEUTRAL:
            if not is_valid_sign or sign != self._last_confirmed_sign:
                # User moved away from the confirmed sign
                if not is_valid_sign:
                    # Truly neutral
                    if self._neutral_since == 0:
                        self._neutral_since = now
                    elif now - self._neutral_since >= self.neutral_timeout:
                        self._state = self.STATE_IDLE
                        self._neutral_since = 0.0
                        self._current_sign = ""
                else:
                    # Different sign -- go straight to recognizing it
                    self._state = self.STATE_RECOGNIZING
                    self._current_sign = sign
                    self._stable_since = now
                    self._neutral_since = 0.0
            else:
                # Still showing the same confirmed sign -- keep waiting
                self._neutral_since = 0.0
            return None

        # ---- STATE: IDLE ----
        if self._state == self.STATE_IDLE:
            if is_valid_sign:
                self._state = self.STATE_RECOGNIZING
                self._current_sign = sign
                self._stable_since = now
            return None

        # ---- STATE: RECOGNIZING ----
        if self._state == self.STATE_RECOGNIZING:
            if not is_valid_sign:
                # Lost the sign -- reset
                self._state = self.STATE_IDLE
                self._current_sign = ""
                return None

            if sign != self._current_sign:
                # Sign changed -- restart stability timer
                self._current_sign = sign
                self._stable_since = now
                return None

            # Same sign -- check stability duration
            held_duration = now - self._stable_since
            if held_duration >= self.stability_seconds:
                # Sign is stable! Accept the word.
                self._words.append(sign)
                self._last_confirmed_sign = sign
                self._state = self.STATE_WAITING_NEUTRAL
                self._neutral_since = 0.0
                return sign

            return None

        # ---- STATE: CONFIRMED (transient) ----
        return None

    def backspace(self):
        """Remove the last word from the sentence."""
        if self._words:
            self._words.pop()

    def clear(self):
        """Clear the entire sentence."""
        self._words.clear()
        self._state = self.STATE_IDLE
        self._current_sign = ""
        self._last_confirmed_sign = ""
        self._neutral_since = 0.0

    def get_display_text(self) -> str:
        """Get the sentence formatted for display."""
        if not self._words:
            return "(empty)"
        return " ".join(self._words)

    def get_status(self) -> str:
        """Get a human-readable status string."""
        if self._state == self.STATE_RECOGNIZING:
            elapsed = time.time() - self._stable_since
            remaining = max(0, self.stability_seconds - elapsed)
            if remaining > 0:
                return f"Holding '{self._current_sign}'... ({remaining:.1f}s)"
            return f"Confirming '{self._current_sign}'..."

        if self._state == self.STATE_WAITING_NEUTRAL:
            return f"'{self._last_confirmed_sign}' added! Return to neutral."

        return ""

    def get_word_count(self) -> int:
        """Number of words in the sentence."""
        return len(self._words)


if __name__ == "__main__":
    print("Testing SentenceBuilder...")

    sb = SentenceBuilder(stability_seconds=0.2, neutral_timeout=0.1)

    # Simulate: hold HELLO for 0.5s
    for _ in range(10):
        result = sb.update("HELLO", 0.95)
        time.sleep(0.05)

    assert sb.text == "HELLO", f"Expected 'HELLO', got '{sb.text}'"
    print(f"  After holding HELLO: '{sb.text}' -- PASSED")

    # Should be in WAITING_NEUTRAL, same sign should NOT be added again
    for _ in range(5):
        result = sb.update("HELLO", 0.95)
        time.sleep(0.05)
    assert sb.text == "HELLO", f"Should still be 'HELLO', got '{sb.text}'"
    print(f"  HELLO not repeated: '{sb.text}' -- PASSED")

    # Return to neutral
    for _ in range(5):
        sb.update("", 0.0)
        time.sleep(0.05)

    # Now hold a different sign
    for _ in range(10):
        sb.update("GOOD", 0.90)
        time.sleep(0.05)
    assert sb.text == "HELLO GOOD", f"Expected 'HELLO GOOD', got '{sb.text}'"
    print(f"  After GOOD: '{sb.text}' -- PASSED")

    # Test backspace
    sb.backspace()
    assert sb.text == "HELLO", f"Expected 'HELLO', got '{sb.text}'"
    print(f"  After backspace: '{sb.text}' -- PASSED")

    # Test clear
    sb.clear()
    assert sb.text == "", f"Expected '', got '{sb.text}'"
    print(f"  After clear: '{sb.text}' -- PASSED")

    print("SentenceBuilder test PASSED.")
