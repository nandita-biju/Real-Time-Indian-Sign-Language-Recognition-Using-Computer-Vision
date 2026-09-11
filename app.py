
"""

app.py -- Real-Time Indian Sign Language Recognition System

Main application entry point. Opens the webcam, detects hand landmarks,

classifies ISL signs (word-level), and displays results with a polished

UI overlay. Accumulates recognized words into sentences.

Usage:

    python app.py

Prerequisites:

    1. Collect ISL training data:  python scripts/collect_data.py

    2. Train the model:             python scripts/train_model.py

Controls:

    BACKSPACE  -- Remove the last word from the sentence

    C           -- Clear the entire sentence

    T           -- Speak the sentence (text-to-speech, if available)

    Q / ESC    -- Quit the application

"""

import os

import sys

import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, PROJECT_ROOT)

import cv2

import numpy as np

from src.camera import Camera

from src.hand_tracker import HandTracker

from src.feature_extractor import extract_features

from src.gesture_classifier import GestureClassifier

from src.sentence_builder import SentenceBuilder

from src.ui import UIOverlay

from src.vocabulary import ISL_SIGNS, get_display_label



def check_dependencies():

    """Verify all required dependencies are available."""

    missing = []

    for pkg, name in [("cv2", "opencv-python"), ("mediapipe", "mediapipe"),

                       ("numpy", "numpy"), ("sklearn", "scikit-learn")]:

        try:

            __import__(pkg)

        except ImportError:

            missing.append(name)

    if missing:

        print("[ERROR] Missing: " + ", ".join(missing))

        print(f"        pip install {' '.join(missing)}")

        sys.exit(1)



def try_speak(text):

    """Attempt text-to-speech using pyttsx3 (optional)."""

    try:

        import pyttsx3

        engine = pyttsx3.init()

        engine.say(text)

        engine.runAndWait()

        return True

    except Exception:

        print("[INFO] TTS not available. Install pyttsx3: pip install pyttsx3")

        return False



def main():

    print()

    print("=" * 60)

    print("  INDIAN SIGN LANGUAGE RECOGNITION")

    print("=" * 60)

    print()

    check_dependencies()

    # ---- Initialize ----

    # Camera

    print("[1/4] Opening camera...")

    cam = Camera()

    if not cam.open():

        print("[ERROR] Cannot open camera.")

        sys.exit(1)

    # Hand tracker

    print("[2/4] Initializing hand tracker...")

    try:

        tracker = HandTracker(max_num_hands=2)

        print("[INFO] Hand tracker ready.")

    except Exception as e:

        print(f"[ERROR] Hand tracker failed: {e}")

        cam.release()

        sys.exit(1)

    # Classifier

    print("[3/4] Loading ISL classifier...")

    classifier = GestureClassifier(

        model_path=os.path.join(PROJECT_ROOT, "models", "static_sign_model.pkl"),

        metadata_path=os.path.join(PROJECT_ROOT, "models", "model_metadata.json"),

        smoothing_window=15,

        confidence_threshold=0.6,

    )

    model_loaded = classifier.load()

    if not model_loaded:

        print()

        print("  +----------------------------------------------------+")

        print("  |  NO TRAINED MODEL FOUND                            |")

        print("  |                                                    |")

        print("  |  The app will run in PREVIEW MODE showing hand     |")

        print("  |  tracking without ISL sign recognition.            |")

        print("  |                                                    |")

        print("  |  To enable recognition:                            |")

        print("  |    1. python scripts/collect_data.py               |")

        print("  |    2. python scripts/train_model.py                |")

        print("  |    3. python app.py                                |")

        print("  +----------------------------------------------------+")

        print()

    # Sentence builder

    print("[4/4] Initializing sentence builder...")

    builder = SentenceBuilder(

        stability_seconds=1.2,

        neutral_timeout=0.6,

    )

    # UI

    ui = UIOverlay(panel_width=300)

    # FPS

    fps = 0.0

    frame_count = 0

    fps_start_time = time.time()

    print()

    print("Application ready! Show ISL signs to the camera.")

    print("Press Q or ESC to quit.")

    print()

    # Create a laptop-friendly resizable window
    cv2.namedWindow("ISL Recognition", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ISL Recognition", 1200, 520)

    # ---- Main Loop ----

    try:

        while True:

            ret, frame = cam.read()

            if not ret:

                print("[ERROR] Lost camera feed.")

                break

            frame = cv2.flip(frame, 1)

            # Detect hands

            hands = tracker.process_frame(frame)

            hand_detected = len(hands) > 0

            tracker.draw_landmarks(frame, hands, draw_bounding_box=True)

            # Classify

            detected_sign = ""

            confidence = 0.0

            if hand_detected and model_loaded:

                hand = hands[0]

                features = extract_features(hand.landmarks)

                if features is not None:

                    detected_sign, confidence = classifier.predict(features)

                    # Get display-friendly label

                    if detected_sign and detected_sign not in ("Uncertain", ""):

                        display_sign = get_display_label(detected_sign)

                    else:

                        display_sign = detected_sign

                    # Update sentence builder with the raw label

                    builder.update(detected_sign, confidence)

                    detected_sign = display_sign

            elif not hand_detected:

                classifier.reset_history()

                # Feed "no hand" to builder so it can detect neutral state

                builder.update("", 0.0)

            # FPS

            frame_count += 1

            elapsed = time.time() - fps_start_time

            if elapsed >= 1.0:

                fps = frame_count / elapsed

                frame_count = 0

                fps_start_time = time.time()

            # Build display sentence with display labels

            sentence_words = []

            for w in builder.words:

                sentence_words.append(get_display_label(w))

            sentence_text = " ".join(sentence_words) if sentence_words else "(empty)"

            # Render UI

            display_frame = ui.render(

                frame=frame,

                detected_sign=detected_sign,

                confidence=confidence,

                sentence_text=sentence_text,

                hand_detected=hand_detected,

                model_loaded=model_loaded,

                builder_status=builder.get_status(),

                fps=fps,

                supported_signs=classifier.get_supported_signs(),

            )

            cv2.imshow("ISL Recognition", display_frame)

            # ---- Input ----

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:

                break

            elif key == 8:  # Backspace

                builder.backspace()

            elif key == ord('c'):

                builder.clear()

            elif key == ord('t'):

                text = " ".join(get_display_label(w) for w in builder.words)

                if text:

                    try_speak(text)

    except KeyboardInterrupt:

        print("\nInterrupted.")

    finally:

        tracker.close()

        cam.release()

        cv2.destroyAllWindows()

    print()

    if builder.words:

        final = " ".join(get_display_label(w) for w in builder.words)

        print(f"Final sentence: {final}")

    print("Application closed.")



if __name__ == "__main__":

    main()

