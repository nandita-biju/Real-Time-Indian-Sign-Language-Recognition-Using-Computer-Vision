"""
hand_tracker.py — Hand detection and landmark tracking using MediaPipe Tasks API.

Uses the modern MediaPipe HandLandmarker (Tasks API) to detect hands in
video frames and return structured landmark data. Supports both left and
right hands. Draws landmarks using OpenCV directly (no legacy mp.solutions).
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, NamedTuple

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandLandmarks(NamedTuple):
    """Structured container for hand landmark data."""
    landmarks: np.ndarray          # Shape (21, 3) — x, y, z for each landmark
    handedness: str                # "Left" or "Right"
    confidence: float              # Detection confidence score
    bounding_box: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max) in pixels


# MediaPipe hand landmark connections for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle finger  -- fixed: 0->9 via 5
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring finger    -- fixed: 0->13 via 9
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),             # Palm connections
]


class HandTracker:
    """
    Detects and tracks hand landmarks using MediaPipe HandLandmarker (Tasks API).

    Supports single or multi-hand detection, drawing landmarks on frames,
    and extracting raw or normalized landmark coordinates.
    """

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        model_path: str = None,
    ):
        """
        Initialize the hand tracker.

        Args:
            max_num_hands: Maximum number of hands to detect.
            min_detection_confidence: Minimum confidence for hand detection.
            min_tracking_confidence: Minimum confidence for landmark tracking.
            model_path: Path to the hand_landmarker.task file.
        """
        if model_path is None:
            # Look for the model in the models/ directory relative to project root
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(project_root, "models", "hand_landmarker.task")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Hand landmarker model not found at: {model_path}\n"
                f"Download it from: https://storage.googleapis.com/mediapipe-models/"
                f"hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def process_frame(self, frame: np.ndarray) -> List[HandLandmarks]:
        """
        Process a BGR frame and return detected hand landmarks.

        Args:
            frame: BGR image from OpenCV.

        Returns:
            List of HandLandmarks, one per detected hand.
        """
        h, w, _ = frame.shape

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect hands
        result = self.landmarker.detect(mp_image)

        detected_hands = []

        if result.hand_landmarks:
            for i, hand_lms in enumerate(result.hand_landmarks):
                # Extract landmark coordinates as numpy array
                landmarks = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_lms],
                    dtype=np.float32,
                )

                # Get handedness
                if result.handedness and i < len(result.handedness):
                    hand_label = result.handedness[i][0].category_name  # "Left" or "Right"
                    hand_conf = result.handedness[i][0].score
                else:
                    hand_label = "Unknown"
                    hand_conf = 0.0

                # Compute bounding box in pixel coordinates
                x_coords = landmarks[:, 0] * w
                y_coords = landmarks[:, 1] * h
                margin = 20
                x_min = max(0, int(x_coords.min()) - margin)
                y_min = max(0, int(y_coords.min()) - margin)
                x_max = min(w, int(x_coords.max()) + margin)
                y_max = min(h, int(y_coords.max()) + margin)

                detected_hands.append(
                    HandLandmarks(
                        landmarks=landmarks,
                        handedness=hand_label,
                        confidence=hand_conf,
                        bounding_box=(x_min, y_min, x_max, y_max),
                    )
                )

        return detected_hands

    def draw_landmarks(
        self,
        frame: np.ndarray,
        hand_landmarks_list: List[HandLandmarks],
        draw_bounding_box: bool = True,
    ) -> np.ndarray:
        """
        Draw hand landmarks and connections on the frame.

        Args:
            frame: BGR image to draw on (modified in-place).
            hand_landmarks_list: List of HandLandmarks from process_frame().
            draw_bounding_box: Whether to draw a bounding box around each hand.

        Returns:
            The annotated frame.
        """
        h, w, _ = frame.shape

        for hand_data in hand_landmarks_list:
            lms = hand_data.landmarks

            # Convert normalized coords to pixel coords
            pts = []
            for lm in lms:
                px = int(lm[0] * w)
                py = int(lm[1] * h)
                pts.append((px, py))

            # Draw connections
            for start_idx, end_idx in HAND_CONNECTIONS:
                cv2.line(frame, pts[start_idx], pts[end_idx], (0, 255, 0), 2)

            # Draw landmark points
            for px, py in pts:
                cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)
                cv2.circle(frame, (px, py), 6, (255, 255, 255), 1)

            # Draw bounding box
            if draw_bounding_box:
                x_min, y_min, x_max, y_max = hand_data.bounding_box
                color = (0, 255, 0)  # Green
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

                # Label with handedness
                label = f"{hand_data.handedness} ({hand_data.confidence:.0%})"
                cv2.putText(
                    frame, label,
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                )

        return frame

    def close(self):
        """Release MediaPipe resources."""
        if self.landmarker:
            self.landmarker.close()
            self.landmarker = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def test_hand_tracker():
    """Quick test: open camera, track hands, show landmarks."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.camera import Camera

    print("Testing hand tracker...")
    cam = Camera()
    if not cam.open():
        return

    tracker = HandTracker(max_num_hands=2)

    print("Show your hand to the camera. Press 'q' to quit.")
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror
        hands = tracker.process_frame(frame)
        tracker.draw_landmarks(frame, hands)

        if hands:
            print(f"\rDetected {len(hands)} hand(s): "
                  f"{', '.join(h.handedness for h in hands)}", end="")

        cv2.imshow("Hand Tracker Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print()
    tracker.close()
    cam.release()
    cv2.destroyAllWindows()
    print("Hand tracker test complete.")


if __name__ == "__main__":
    test_hand_tracker()
