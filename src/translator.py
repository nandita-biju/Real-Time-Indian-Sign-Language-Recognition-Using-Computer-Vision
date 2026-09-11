"""
translator.py — Converts recognized signs into text with debouncing.

Handles the accumulation of recognized characters into words/sentences,
with cooldown to prevent repeated characters from a held pose,
and basic editing controls (backspace, clear, space).

Extension point for future word/sentence-level translation.
"""

import time
from typing import Optional


class Translator:
    """
    Translates a stream of recognized signs into accumulated text.

    Features:
      - Cooldown: same sign held continuously only adds one character.
      - Debounce: a sign must be stable for a minimum duration before it's added.
      - Special commands via keyboard: space, backspace, clear.
    """

    def __init__(
        self,
        cooldown_seconds: float = 1.5,
        stability_seconds: float = 0.8,
    ):
        """
        Args:
            cooldown_seconds: Minimum time between adding the same character.
            stability_seconds: How long a sign must be stable before adding it.
        """
        self.cooldown_seconds = cooldown_seconds
        self.stability_seconds = stability_seconds

        self._text = ""
        self._last_added_char = ""
        self._last_added_time = 0.0

        # Stability tracking
        self._current_stable_sign = ""
        self._stable_since = 0.0
        self._sign_already_added = False

    @property
    def text(self) -> str:
        """Get the accumulated text."""
        return self._text

    def update(self, sign: str, confidence: float) -> Optional[str]:
        """
        Process a new recognized sign.

        Call this every frame with the current smoothed prediction.
        Returns the character that was added, or None if nothing was added.

        Args:
            sign: The recognized sign label (e.g., "A", "B").
            confidence: Confidence of the prediction.

        Returns:
            The added character, or None.
        """
        now = time.time()

        # Ignore uncertain or empty predictions
        if not sign or sign == "Uncertain" or sign == "No hand detected":
            self._current_stable_sign = ""
            self._stable_since = now
            self._sign_already_added = False
            return None

        # Check if the sign changed
        if sign != self._current_stable_sign:
            self._current_stable_sign = sign
            self._stable_since = now
            self._sign_already_added = False
            return None

        # Sign is stable — check if it's been held long enough
        held_duration = now - self._stable_since

        if held_duration < self.stability_seconds:
            return None

        # Sign has been stable long enough — check cooldown
        if self._sign_already_added:
            # Already added this sign; wait for pose change
            return None

        # Check cooldown against same character
        if (sign == self._last_added_char and
                now - self._last_added_time < self.cooldown_seconds):
            return None

        # Add the character
        self._text += sign
        self._last_added_char = sign
        self._last_added_time = now
        self._sign_already_added = True

        return sign

    def add_space(self):
        """Add a space to the text."""
        self._text += " "

    def backspace(self):
        """Remove the last character."""
        if self._text:
            self._text = self._text[:-1]

    def clear(self):
        """Clear all accumulated text."""
        self._text = ""
        self._last_added_char = ""
        self._last_added_time = 0.0
        self._current_stable_sign = ""
        self._sign_already_added = False

    def get_display_text(self) -> str:
        """Get text formatted for display."""
        if not self._text:
            return "(empty)"
        return self._text

    def get_status(self) -> str:
        """Get a status string showing what the translator is currently doing."""
        if self._current_stable_sign and not self._sign_already_added:
            elapsed = time.time() - self._stable_since
            remaining = max(0, self.stability_seconds - elapsed)
            if remaining > 0:
                return f"Holding '{self._current_stable_sign}'... ({remaining:.1f}s)"
        return ""


if __name__ == "__main__":
    print("Testing translator...")
    t = Translator(cooldown_seconds=0.5, stability_seconds=0.3)

    # Simulate stable sign
    import time as _time
    for _ in range(5):
        result = t.update("A", 0.95)
        _time.sleep(0.1)

    print(f"Text after holding 'A': '{t.text}'")
    assert t.text == "A", f"Expected 'A', got '{t.text}'"

    t.add_space()
    assert t.text == "A ", f"Expected 'A ', got '{t.text}'"

    t.backspace()
    assert t.text == "A", f"Expected 'A', got '{t.text}'"

    t.clear()
    assert t.text == "", f"Expected '', got '{t.text}'"

    print("Translator test PASSED.")
