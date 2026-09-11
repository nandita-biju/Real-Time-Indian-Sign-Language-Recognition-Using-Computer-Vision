"""
collect_data.py -- ISL word-level data collection tool.

Interactive webcam-based tool for collecting hand landmark samples
for ISL (Indian Sign Language) word recognition training.

Supports:
  - Static signs: captures landmark snapshots from individual frames
  - Dynamic signs: captures sequences of landmarks while performing the sign
  - Neutral state collection for better discrimination

Usage:
    python scripts/collect_data.py

Controls:
    LEFT/RIGHT arrows or N/P  = Navigate between signs
    SPACE                     = Toggle recording (for dynamic) / capture (for static)
    S                         = Save static sample (snapshot)
    R                         = Start/stop recording dynamic sequence
    Q                         = Quit and save
"""

import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np

from src.camera import Camera
from src.hand_tracker import HandTracker
from src.feature_extractor import extract_features
from src.vocabulary import (
    ISL_SIGNS, get_all_labels, get_static_labels, get_dynamic_labels,
    NEUTRAL_LABEL, get_sign_info,
)
from src.sequence_buffer import augment_landmarks

# Directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
STATIC_DIR = os.path.join(DATA_DIR, "static")
DYNAMIC_DIR = os.path.join(DATA_DIR, "dynamic")


def ensure_dirs():
    """Create data directories."""
    os.makedirs(STATIC_DIR, exist_ok=True)
    os.makedirs(DYNAMIC_DIR, exist_ok=True)


def count_existing_samples(label, sign_type):
    """Count existing samples for a label."""
    if sign_type == "static":
        path = os.path.join(STATIC_DIR, f"{label}.npy")
        if os.path.exists(path):
            data = np.load(path)
            return len(data)
    else:
        label_dir = os.path.join(DYNAMIC_DIR, label)
        if os.path.exists(label_dir):
            return len([f for f in os.listdir(label_dir) if f.endswith(".npy")])
    return 0


def save_static_sample(label, features):
    """Append a static sample to the label's .npy file."""
    path = os.path.join(STATIC_DIR, f"{label}.npy")
    if os.path.exists(path):
        existing = np.load(path)
        data = np.vstack([existing, features.reshape(1, -1)])
    else:
        data = features.reshape(1, -1)
    np.save(path, data)


def save_dynamic_sample(label, sequence):
    """Save a dynamic sequence as a numbered .npy file."""
    label_dir = os.path.join(DYNAMIC_DIR, label)
    os.makedirs(label_dir, exist_ok=True)
    existing = len([f for f in os.listdir(label_dir) if f.endswith(".npy")])
    path = os.path.join(label_dir, f"seq_{existing:04d}.npy")
    np.save(path, sequence)


def main():
    print()
    print("=" * 60)
    print("  ISL DATA COLLECTION")
    print("  Indian Sign Language -- Word-Level Samples")
    print("=" * 60)
    print()

    ensure_dirs()

    # Build sign list: all vocabulary signs + NEUTRAL
    all_labels = get_all_labels()
    sign_list = list(all_labels) + [NEUTRAL_LABEL]

    # Print summary
    print(f"Total signs to collect: {len(sign_list)}")
    print(f"  Static signs:  {len(get_static_labels())}")
    print(f"  Dynamic signs: {len(get_dynamic_labels())}")
    print(f"  Neutral state: 1")
    print()
    print("Controls:")
    print("  LEFT/RIGHT or N/P = Navigate signs")
    print("  S or SPACE         = Capture static sample")
    print("  R                  = Start/stop recording dynamic sequence")
    print("  A                  = Auto-capture (hold to collect static rapidly)")
    print("  Q                  = Quit and save")
    print()

    # Camera
    cam = Camera()
    if not cam.open():
        print("[ERROR] Cannot open camera.")
        sys.exit(1)

    tracker = HandTracker(max_num_hands=2)

    # State
    current_idx = 0
    auto_capture = False
    auto_capture_interval = 0.08  # seconds between auto captures
    last_auto_capture = 0

    recording_dynamic = False
    dynamic_frames = []

    session_counts = {label: 0 for label in sign_list}

    print("Camera ready. Navigate to a sign and start collecting!")
    print()

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Detect hands
            hands = tracker.process_frame(frame)
            tracker.draw_landmarks(frame, hands)
            hand_detected = len(hands) > 0

            # Current sign info
            current_label = sign_list[current_idx]
            is_neutral = (current_label == NEUTRAL_LABEL)

            if is_neutral:
                sign_type = "static"
                description = "No intentional sign. Hand relaxed or absent."
                num_hands = 1
            else:
                info = get_sign_info(current_label)
                sign_type = info.sign_type
                description = info.description
                num_hands = info.num_hands

            total_samples = count_existing_samples(current_label, sign_type) + session_counts[current_label]

            # ---- Auto-capture for static signs ----
            now = time.time()
            if auto_capture and hand_detected and sign_type == "static":
                if now - last_auto_capture >= auto_capture_interval:
                    hand = hands[0]
                    features = extract_features(hand.landmarks)
                    if features is not None:
                        save_static_sample(current_label, features)
                        session_counts[current_label] += 1
                        last_auto_capture = now

            # ---- Recording dynamic sequences ----
            if recording_dynamic and hand_detected:
                hand = hands[0]
                features = extract_features(hand.landmarks)
                if features is not None:
                    dynamic_frames.append(features)

            # ---- Draw UI ----
            # Top bar
            cv2.rectangle(frame, (0, 0), (w, 45), (40, 40, 40), -1)
            cv2.putText(frame, "ISL DATA COLLECTION", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 100), 2)
            # Sign counter
            counter_text = f"{current_idx + 1}/{len(sign_list)}"
            cv2.putText(frame, counter_text, (w - 100, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Bottom panel
            panel_h = 160
            cv2.rectangle(frame, (0, h - panel_h), (w, h), (40, 40, 40), -1)

            y = h - panel_h + 25
            # Sign label
            display_label = current_label if is_neutral else get_sign_info(current_label).label
            label_color = (0, 255, 200) if not is_neutral else (200, 200, 0)
            cv2.putText(frame, f"Sign: {display_label}", (15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)

            type_text = f"[{sign_type.upper()}] [{num_hands}H]"
            cv2.putText(frame, type_text, (w - 200, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            # Description (truncated)
            y += 28
            desc_short = description[:80] + "..." if len(description) > 80 else description
            cv2.putText(frame, desc_short, (15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

            # Sample count
            y += 28
            samples_text = f"Samples: {total_samples}"
            if total_samples >= 100:
                count_color = (0, 255, 0)
            elif total_samples >= 50:
                count_color = (0, 200, 200)
            else:
                count_color = (0, 140, 255)
            cv2.putText(frame, samples_text, (15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, count_color, 1)

            # Session count
            cv2.putText(frame, f"(+{session_counts[current_label]} this session)",
                        (200, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

            # Status
            y += 28
            if auto_capture:
                cv2.putText(frame, "AUTO-CAPTURE ON (press A to stop)",
                            (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.rectangle(frame, (5, 50), (w - 5, h - panel_h - 5), (0, 255, 0), 3)
            elif recording_dynamic:
                frames_text = f"RECORDING... ({len(dynamic_frames)} frames)"
                cv2.putText(frame, frames_text, (15, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)
                cv2.rectangle(frame, (5, 50), (w - 5, h - panel_h - 5), (0, 100, 255), 3)
            elif not hand_detected:
                cv2.putText(frame, "No hand detected", (15, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1)
            else:
                ctrl_text = "S=Snapshot  A=AutoCapture  R=Record  N/P=Navigate  Q=Quit"
                cv2.putText(frame, ctrl_text, (15, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

            # Controls hint at very bottom
            y += 25
            cv2.putText(frame, "LEFT/RIGHT arrows or N/P to navigate signs",
                        (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)

            cv2.imshow("ISL Data Collection", frame)

            # ---- Handle input ----
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('n') or key == 83 or key == 3:  # N or RIGHT arrow
                current_idx = (current_idx + 1) % len(sign_list)
                auto_capture = False
                recording_dynamic = False
                dynamic_frames = []

            elif key == ord('p') or key == 81 or key == 2:  # P or LEFT arrow
                current_idx = (current_idx - 1) % len(sign_list)
                auto_capture = False
                recording_dynamic = False
                dynamic_frames = []

            elif key == ord('s') or key == ord(' '):
                # Snapshot: save one static sample
                if hand_detected and sign_type == "static":
                    hand = hands[0]
                    features = extract_features(hand.landmarks)
                    if features is not None:
                        save_static_sample(current_label, features)
                        session_counts[current_label] += 1

            elif key == ord('a'):
                # Toggle auto-capture
                if sign_type == "static":
                    auto_capture = not auto_capture
                    if auto_capture:
                        last_auto_capture = 0

            elif key == ord('r'):
                # Toggle dynamic recording
                if sign_type == "dynamic":
                    if recording_dynamic:
                        # Stop and save
                        if len(dynamic_frames) >= 10:
                            sequence = np.array(dynamic_frames, dtype=np.float32)
                            save_dynamic_sample(current_label, sequence)
                            session_counts[current_label] += 1
                            print(f"  Saved dynamic sequence: {current_label} "
                                  f"({len(dynamic_frames)} frames)")
                        else:
                            print(f"  Sequence too short ({len(dynamic_frames)} frames), discarded.")
                        recording_dynamic = False
                        dynamic_frames = []
                    else:
                        recording_dynamic = True
                        dynamic_frames = []

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        tracker.close()
        cam.release()
        cv2.destroyAllWindows()

    # ---- Summary ----
    print()
    print("=" * 60)
    print("  COLLECTION SUMMARY")
    print("=" * 60)
    print()

    total_session = sum(session_counts.values())
    print(f"Samples collected this session: {total_session}")
    print()

    for label in sign_list:
        is_n = (label == NEUTRAL_LABEL)
        stype = "static" if is_n else ISL_SIGNS[label].sign_type if label in ISL_SIGNS else "?"
        total = count_existing_samples(label, stype)
        new = session_counts[label]
        status = "[OK]" if total >= 50 else "[!] need more"
        new_str = f" (+{new})" if new > 0 else ""
        disp = label if is_n else ISL_SIGNS[label].label if label in ISL_SIGNS else label
        print(f"  {disp:15s} [{stype:7s}]: {total:4d}{new_str:12s} {status}")

    print()
    print(f"Data saved to: {DATA_DIR}")
    print()
    print("Next: python scripts/train_model.py")


if __name__ == "__main__":
    main()
