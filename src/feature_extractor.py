"""
feature_extractor.py — Extracts normalized landmark features for gesture classification.

Converts raw MediaPipe hand landmarks into a flat, normalized feature vector
suitable for ML classification. Normalization makes recognition independent
of hand position and distance from camera.

Feature vector layout (63 features):
  - 21 landmarks × 3 coordinates (x, y, z)
  - Normalized relative to wrist position and hand scale

Extension point for dynamic signs:
  - extract_sequence_features() can stack multiple frames' features
    for temporal models (LSTM, GRU, Transformer).
"""

import numpy as np
from typing import Optional, List


def extract_features(landmarks: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract a normalized feature vector from hand landmarks.

    Normalization steps:
      1. Translate so wrist (landmark 0) is at origin.
      2. Scale by the maximum distance from wrist to any landmark,
         making features invariant to hand size and distance.
      3. Flatten to a 1D vector.

    Args:
        landmarks: Shape (21, 3) array of (x, y, z) landmark coordinates.

    Returns:
        1D numpy array of shape (63,), or None if input is invalid.
    """
    if landmarks is None or landmarks.shape != (21, 3):
        return None

    # Step 1: Translate — wrist at origin
    wrist = landmarks[0].copy()
    centered = landmarks - wrist

    # Step 2: Scale — normalize by max distance from wrist
    distances = np.linalg.norm(centered, axis=1)
    max_dist = distances.max()

    if max_dist < 1e-6:
        # Degenerate case: all landmarks at same position
        return None

    normalized = centered / max_dist

    # Step 3: Flatten
    features = normalized.flatten()

    return features.astype(np.float32)


def extract_features_batch(landmarks_list: List[np.ndarray]) -> np.ndarray:
    """
    Extract features for a batch of landmark arrays.

    Args:
        landmarks_list: List of (21, 3) landmark arrays.

    Returns:
        2D array of shape (N, 63), excluding invalid entries.
    """
    features = []
    for lm in landmarks_list:
        f = extract_features(lm)
        if f is not None:
            features.append(f)

    if not features:
        return np.empty((0, 63), dtype=np.float32)

    return np.array(features, dtype=np.float32)


def extract_sequence_features(
    landmarks_sequence: List[np.ndarray],
    sequence_length: int = 30,
) -> Optional[np.ndarray]:
    """
    Extract features from a sequence of frames for dynamic sign recognition.

    This is an EXTENSION POINT for future temporal models.

    Args:
        landmarks_sequence: List of (21, 3) arrays, one per frame.
        sequence_length: Expected number of frames in the sequence.

    Returns:
        2D array of shape (sequence_length, 63), or None if insufficient data.
    """
    features = []
    for lm in landmarks_sequence:
        f = extract_features(lm)
        if f is not None:
            features.append(f)

    if len(features) < sequence_length:
        return None

    # Take the last `sequence_length` frames
    features = features[-sequence_length:]
    return np.array(features, dtype=np.float32)


def get_feature_names() -> List[str]:
    """
    Get descriptive names for each feature dimension.

    Useful for debugging and model interpretation.
    """
    landmark_names = [
        "WRIST",
        "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
        "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
        "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
        "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
        "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
    ]
    coords = ["x", "y", "z"]

    names = []
    for lm_name in landmark_names:
        for coord in coords:
            names.append(f"{lm_name}_{coord}")

    return names


if __name__ == "__main__":
    # Quick test with synthetic data
    print("Testing feature extraction...")
    fake_landmarks = np.random.rand(21, 3).astype(np.float32)
    features = extract_features(fake_landmarks)
    if features is not None:
        print(f"Feature vector shape: {features.shape}")
        print(f"Feature names count: {len(get_feature_names())}")
        print(f"Min: {features.min():.4f}, Max: {features.max():.4f}")
        print("Feature extraction test PASSED.")
    else:
        print("Feature extraction test FAILED.")
