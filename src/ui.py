"""
ui.py -- Overlay rendering for ISL sign language recognition.

Draws a polished, professional overlay on top of the camera feed using OpenCV.
Updated for word-level ISL recognition with sentence display.
"""

import cv2
import numpy as np
from typing import Optional, List


class UIOverlay:
    """
    Renders the ISL recognition UI overlay on the camera frame.

    Displays:
      - Title bar (Indian Sign Language Recognition)
      - Detected ISL sign with confidence
      - Sentence builder output
      - Status messages
      - Controls
    """

    # Color scheme (BGR)
    BG_COLOR = (35, 35, 35)
    PANEL_COLOR = (45, 45, 45)
    ACCENT_COLOR = (0, 200, 100)     # Green accent
    ISL_COLOR = (100, 200, 255)      # Warm gold for ISL branding
    TEXT_COLOR = (255, 255, 255)
    DIM_TEXT_COLOR = (170, 170, 170)
    WARNING_COLOR = (0, 140, 255)    # Orange
    HIGHLIGHT_COLOR = (0, 255, 200)  # Bright green
    SENTENCE_COLOR = (255, 220, 150) # Light blue-white for sentence

    def __init__(self, panel_width: int = 300):
        self.panel_width = panel_width

    def render(
        self,
        frame: np.ndarray,
        detected_sign: str = "",
        confidence: float = 0.0,
        sentence_text: str = "",
        status_message: str = "",
        hand_detected: bool = False,
        model_loaded: bool = False,
        builder_status: str = "",
        fps: float = 0.0,
        supported_signs: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Render the full UI overlay.

        Args:
            frame: Camera frame with landmarks drawn.
            detected_sign: Current recognized ISL sign.
            confidence: Prediction confidence (0-1).
            sentence_text: Accumulated sentence from SentenceBuilder.
            status_message: Extra status message.
            hand_detected: Whether a hand is visible.
            model_loaded: Whether the ML model is loaded.
            builder_status: Status from sentence builder.
            fps: Current FPS.
            supported_signs: List of signs the model supports.

        Returns:
            Composite frame with panel.
        """
        h, w, _ = frame.shape

        # Right panel
        panel = np.full((h, self.panel_width, 3), self.PANEL_COLOR, dtype=np.uint8)
        y = 15

        # ---- Panel Title ----
        y = self._draw_text(panel, "ISL Recognition", y, self.ISL_COLOR, scale=0.65, thickness=2)
        y = self._draw_divider(panel, y + 5)

        # ---- Detection Section ----
        y += 12
        if not model_loaded:
            y = self._draw_text(panel, "Model: NOT LOADED", y, self.WARNING_COLOR, scale=0.55)
            y += 3
            y = self._draw_text(panel, "1. python scripts/collect_data.py", y, self.DIM_TEXT_COLOR, scale=0.38)
            y += 2
            y = self._draw_text(panel, "2. python scripts/train_model.py", y, self.DIM_TEXT_COLOR, scale=0.38)
            y += 2
            y = self._draw_text(panel, "3. python app.py", y, self.DIM_TEXT_COLOR, scale=0.38)

        elif not hand_detected:
            y = self._draw_text(panel, "No hand detected", y, self.WARNING_COLOR, scale=0.55)
            y += 3
            y = self._draw_text(panel, "Show your hand to the camera", y, self.DIM_TEXT_COLOR, scale=0.38)

        else:
            y = self._draw_text(panel, "Detected Sign:", y, self.DIM_TEXT_COLOR, scale=0.45)
            y += 3

            if detected_sign and detected_sign not in ("Uncertain", ""):
                # Large sign label
                sign_scale = 1.5 if len(detected_sign) <= 8 else 1.0
                y = self._draw_text(panel, detected_sign, y, self.HIGHLIGHT_COLOR,
                                    scale=sign_scale, thickness=2)
            else:
                y = self._draw_text(panel, "Uncertain", y, self.WARNING_COLOR, scale=0.7)

            # Confidence bar
            y += 8
            y = self._draw_text(panel, "Confidence:", y, self.DIM_TEXT_COLOR, scale=0.4)
            y += 3
            y = self._draw_confidence_bar(panel, y, confidence)
            y = self._draw_text(panel, f"{confidence:.0%}", y + 3, self.TEXT_COLOR, scale=0.4)

        # ---- Divider ----
        y = self._draw_divider(panel, y + 12)

        # ---- Sentence Section ----
        y += 10
        y = self._draw_text(panel, "Sentence:", y, self.DIM_TEXT_COLOR, scale=0.5)
        y += 5

        display_sentence = sentence_text if sentence_text and sentence_text != "(empty)" else "(empty)"
        if display_sentence == "(empty)":
            y = self._draw_text(panel, display_sentence, y, (100, 100, 100), scale=0.45)
        else:
            y = self._draw_wrapped_text(panel, display_sentence, y,
                                        self.SENTENCE_COLOR, scale=0.5)

        # Builder status
        if builder_status:
            y += 8
            y = self._draw_text(panel, builder_status, y, self.ACCENT_COLOR, scale=0.38)

        # ---- Divider ----
        y = self._draw_divider(panel, y + 12)

        # ---- Controls ----
        y += 8
        y = self._draw_text(panel, "Controls:", y, self.DIM_TEXT_COLOR, scale=0.42)
        controls = [
            "BACKSPACE - Remove last word",
            "C - Clear sentence",
            "T - Speak text (TTS)",
            "Q / ESC - Quit",
        ]
        for ctrl in controls:
            y += 2
            y = self._draw_text(panel, ctrl, y, self.DIM_TEXT_COLOR, scale=0.35)

        # ---- FPS ----
        if fps > 0:
            self._draw_text(panel, f"FPS: {fps:.0f}", h - 25,
                            self.DIM_TEXT_COLOR, scale=0.38)

        # ---- Combine ----
        composite = np.hstack([frame, panel])

        # ---- Title bar ----
        bar_h = 38
        cv2.rectangle(composite, (0, 0), (w + self.panel_width, bar_h),
                       self.BG_COLOR, -1)
        cv2.putText(composite, "INDIAN SIGN LANGUAGE RECOGNITION",
                     (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                     self.ISL_COLOR, 2)

        return composite

    def _draw_text(self, panel, text, y, color, scale=0.5, thickness=1):
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        cv2.putText(panel, text, (15, y + th), font, scale, color, thickness)
        return y + th + baseline + 2

    def _draw_divider(self, panel, y):
        pw = panel.shape[1]
        cv2.line(panel, (15, y), (pw - 15, y), (70, 70, 70), 1)
        return y + 5

    def _draw_wrapped_text(self, panel, text, y, color, scale=0.5):
        max_width = panel.shape[1] - 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        words = text.split(" ")
        line = ""
        for word in words:
            test_line = (line + " " + word).strip()
            (tw, _), _ = cv2.getTextSize(test_line, font, scale, 1)
            if tw > max_width and line:
                y = self._draw_text(panel, line, y, color, scale)
                line = word
            else:
                line = test_line
        if line:
            y = self._draw_text(panel, line, y, color, scale)
        return y

    def _draw_confidence_bar(self, panel, y, confidence):
        bar_x, bar_w, bar_h = 15, panel.shape[1] - 30, 12
        cv2.rectangle(panel, (bar_x, y), (bar_x + bar_w, y + bar_h), (70, 70, 70), -1)
        fill_w = int(bar_w * min(max(confidence, 0), 1))
        if confidence > 0.8:
            color = self.ACCENT_COLOR
        elif confidence > 0.5:
            color = (0, 200, 200)
        else:
            color = self.WARNING_COLOR
        if fill_w > 0:
            cv2.rectangle(panel, (bar_x, y), (bar_x + fill_w, y + bar_h), color, -1)
        return y + bar_h


if __name__ == "__main__":
    print("ISL UI overlay module loaded successfully.")
