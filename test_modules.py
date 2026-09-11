"""Quick integration test for all ISL recognition modules."""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

print("=" * 50)
print("  ISL MODULE INTEGRATION TESTS")
print("=" * 50)
print()

# Test 1: Vocabulary
print("Testing vocabulary...")
from src.vocabulary import (
    ISL_SIGNS, get_all_labels, get_static_labels, get_dynamic_labels,
    get_mvp_labels, get_extended_labels, get_landmark_only_labels,
    get_sign_info, get_display_label, NEUTRAL_LABEL,
    MVP_VOCABULARY, EXTENDED_VOCABULARY,
)
all_labels = get_all_labels()
assert len(all_labels) >= 15, f"Expected 15+ signs, got {len(all_labels)}"
static = get_static_labels()
dynamic = get_dynamic_labels()
assert len(static) > 0, "No static signs defined"
assert len(dynamic) > 0, "No dynamic signs defined"
assert get_display_label("I_ME") == "I/ME"
mvp = get_mvp_labels()
assert len(mvp) >= 3, f"MVP too small: {len(mvp)}"
landmark_only = get_landmark_only_labels()
assert len(landmark_only) >= 3, f"Too few landmark-only signs: {len(landmark_only)}"
# Verify all MVP signs have landmark_only=True
for label in MVP_VOCABULARY:
    info = get_sign_info(label)
    assert info.landmark_only, f"MVP sign {label} is NOT landmark_only"
print(f"  {len(all_labels)} signs: {len(static)} static, {len(dynamic)} dynamic")
print(f"  MVP: {len(mvp)}, Extended: {len(EXTENDED_VOCABULARY)}, Landmark-only: {len(landmark_only)}")
print("  PASSED")

# Test 2: Feature extractor
print("Testing feature_extractor...")
from src.feature_extractor import extract_features, extract_sequence_features
fake_lm = np.random.rand(21, 3).astype(np.float32)
features = extract_features(fake_lm)
assert features is not None and features.shape == (63,)
print(f"  Static features: {features.shape}")

# Test sequence features
seq = [np.random.rand(21, 3).astype(np.float32) for _ in range(30)]
seq_feat = extract_sequence_features(seq, sequence_length=30)
assert seq_feat is not None and seq_feat.shape == (30, 63)
print(f"  Sequence features: {seq_feat.shape}")
print("  PASSED")

# Test 3: Sequence buffer
print("Testing sequence_buffer...")
from src.sequence_buffer import SequenceBuffer, augment_landmarks
buf = SequenceBuffer(sequence_length=5, feature_dim=63)
assert not buf.is_ready()
for _ in range(5):
    buf.add_frame(np.random.rand(63).astype(np.float32))
assert buf.is_ready()
s = buf.get_sequence()
assert s.shape == (5, 63)

aug = augment_landmarks(fake_lm)
assert aug.shape == (21, 3)
assert not np.array_equal(aug, fake_lm)
print(f"  Buffer + augmentation OK")
print("  PASSED")

# Test 4: Sentence builder
print("Testing sentence_builder...")
from src.sentence_builder import SentenceBuilder
sb = SentenceBuilder(stability_seconds=0.15, neutral_timeout=0.08)

# Hold HELLO
for _ in range(10):
    sb.update("HELLO", 0.95)
    time.sleep(0.03)
assert sb.text == "HELLO", f"Expected 'HELLO', got '{sb.text}'"

# Same sign should NOT repeat
for _ in range(5):
    sb.update("HELLO", 0.95)
    time.sleep(0.03)
assert sb.text == "HELLO", f"Should still be 'HELLO', got '{sb.text}'"

# Go neutral, then new sign
for _ in range(5):
    sb.update("", 0.0)
    time.sleep(0.03)
for _ in range(10):
    sb.update("GOOD", 0.90)
    time.sleep(0.03)
assert sb.text == "HELLO GOOD", f"Expected 'HELLO GOOD', got '{sb.text}'"

sb.backspace()
assert sb.text == "HELLO"
sb.clear()
assert sb.text == ""
print(f"  State machine + debouncing OK")
print("  PASSED")

# Test 5: Gesture classifier (no model expected)
print("Testing gesture_classifier...")
from src.gesture_classifier import GestureClassifier
clf = GestureClassifier()
loaded = clf.load()
print(f"  Model loaded: {loaded} (expected False without trained model)")
print("  PASSED")

# Test 6: UI overlay
print("Testing ui...")
from src.ui import UIOverlay
ui = UIOverlay(panel_width=300)
frame = np.zeros((480, 640, 3), dtype=np.uint8)
result = ui.render(frame, detected_sign="HELLO", confidence=0.95,
                   sentence_text="HELLO GOOD", hand_detected=True, model_loaded=True)
assert result.shape[0] == 480
assert result.shape[1] == 640 + 380
print(f"  Rendered: {result.shape}")
print("  PASSED")

# Test 7: Camera module
print("Testing camera...")
from src.camera import Camera
cam = Camera()
print("  PASSED")

# Test 8: Hand tracker
print("Testing hand_tracker...")
from src.hand_tracker import HandTracker
tracker = HandTracker(max_num_hands=1)
dummy = np.zeros((480, 640, 3), dtype=np.uint8)
hands = tracker.process_frame(dummy)
assert len(hands) == 0
tracker.close()
print(f"  Hands on black frame: {len(hands)}")
print("  PASSED")

print()
print("=" * 50)
print("  ALL ISL MODULE TESTS PASSED")
print("=" * 50)
