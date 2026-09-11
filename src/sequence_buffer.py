"""
sequence_buffer.py -- Frame sequence buffering for dynamic sign recognition.

Maintains a rolling buffer of hand landmark frames for temporal classification.
Used by dynamic sign classifiers (LSTM/GRU) that need a sequence of landmarks
rather than a single frame.

Also provides data augmentation utilities for landmark-based features.
"""

import numpy as np
from typing import Optional, List, Tuple
from collections import deque


class SequenceBuffer:
    """
    Rolling buffer that stores sequences of hand landmark features
    for dynamic sign recognition.
    """

    def __init__(self, sequence_length: int = 30, feature_dim: int = 63):
        """
        Args:
            sequence_length: Number of frames to buffer.
            feature_dim: Dimensionality of each frame's feature vector.
        """
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self._buffer: deque = deque(maxlen=sequence_length)

    def add_frame(self, features: Optional[np.ndarray]):
        """
        Add a frame's features to the buffer.

        If features is None (no hand detected), a zero vector is added
        to maintain temporal continuity.

        Args:
            features: 1D array of shape (feature_dim,) or None.
        """
        if features is not None and features.shape == (self.feature_dim,):
            self._buffer.append(features.copy())
        else:
            self._buffer.append(np.zeros(self.feature_dim, dtype=np.float32))

    def is_ready(self) -> bool:
        """Check if the buffer has enough frames for classification."""
        return len(self._buffer) >= self.sequence_length

    def get_sequence(self) -> Optional[np.ndarray]:
        """
        Get the current sequence as a 2D array.

        Returns:
            Array of shape (sequence_length, feature_dim), or None if not ready.
        """
        if not self.is_ready():
            return None
        return np.array(list(self._buffer), dtype=np.float32)

    def clear(self):
        """Clear the buffer."""
        self._buffer.clear()

    @property
    def length(self) -> int:
        """Current number of frames in the buffer."""
        return len(self._buffer)

    @property
    def fill_ratio(self) -> float:
        """How full the buffer is (0.0 to 1.0)."""
        return len(self._buffer) / self.sequence_length


# ============================================================================
#  DATA AUGMENTATION
# ============================================================================

def augment_landmarks(
    landmarks: np.ndarray,
    noise_std: float = 0.01,
    scale_range: Tuple[float, float] = (0.9, 1.1),
    rotation_max_deg: float = 10.0,
    translate_range: float = 0.05,
) -> np.ndarray:
    """
    Apply random augmentation to a single frame of hand landmarks.

    The input should be raw (unnormalized) landmarks of shape (21, 3).
    Augmentation is applied BEFORE normalization.

    Args:
        landmarks: Shape (21, 3) landmark array.
        noise_std: Std dev of Gaussian noise added to coordinates.
        scale_range: (min, max) random scale factor.
        rotation_max_deg: Maximum random rotation in degrees (around z-axis).
        translate_range: Maximum random translation as fraction of coord range.

    Returns:
        Augmented landmarks of shape (21, 3).
    """
    aug = landmarks.copy()

    # 1. Add Gaussian noise
    noise = np.random.normal(0, noise_std, aug.shape).astype(np.float32)
    aug += noise

    # 2. Random scale
    scale = np.random.uniform(scale_range[0], scale_range[1])
    center = aug.mean(axis=0)
    aug = (aug - center) * scale + center

    # 3. Random rotation around z-axis (in the image plane)
    angle_rad = np.radians(np.random.uniform(-rotation_max_deg, rotation_max_deg))
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rot_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0,      0,     1],
    ], dtype=np.float32)
    center = aug.mean(axis=0)
    aug = (aug - center) @ rot_matrix.T + center

    # 4. Random translation
    dx = np.random.uniform(-translate_range, translate_range)
    dy = np.random.uniform(-translate_range, translate_range)
    aug[:, 0] += dx
    aug[:, 1] += dy

    return aug


def augment_sequence(
    sequence: np.ndarray,
    noise_std: float = 0.005,
    temporal_jitter: int = 2,
) -> np.ndarray:
    """
    Apply augmentation to a sequence of normalized feature vectors.

    Args:
        sequence: Shape (T, 63) array of normalized features.
        noise_std: Std dev of Gaussian noise.
        temporal_jitter: Max frames to shift the sequence temporally.

    Returns:
        Augmented sequence of the same shape.
    """
    aug = sequence.copy()

    # 1. Add small Gaussian noise
    noise = np.random.normal(0, noise_std, aug.shape).astype(np.float32)
    aug += noise

    # 2. Temporal jitter: randomly shift start position
    if temporal_jitter > 0 and len(aug) > temporal_jitter * 2:
        shift = np.random.randint(-temporal_jitter, temporal_jitter + 1)
        if shift > 0:
            aug = np.concatenate([aug[shift:], np.tile(aug[-1:], (shift, 1))])
        elif shift < 0:
            aug = np.concatenate([np.tile(aug[0:1], (-shift, 1)), aug[:shift]])

    return aug


if __name__ == "__main__":
    print("Testing SequenceBuffer...")

    buf = SequenceBuffer(sequence_length=5, feature_dim=63)

    assert not buf.is_ready(), "Should not be ready when empty"

    for i in range(5):
        buf.add_frame(np.random.rand(63).astype(np.float32))

    assert buf.is_ready(), "Should be ready after 5 frames"

    seq = buf.get_sequence()
    assert seq.shape == (5, 63), f"Wrong shape: {seq.shape}"
    print(f"  Sequence shape: {seq.shape} -- PASSED")

    # Test augmentation
    fake_lm = np.random.rand(21, 3).astype(np.float32)
    aug_lm = augment_landmarks(fake_lm)
    assert aug_lm.shape == (21, 3), f"Wrong aug shape: {aug_lm.shape}"
    assert not np.array_equal(fake_lm, aug_lm), "Augmentation should change data"
    print(f"  Landmark augmentation: PASSED")

    fake_seq = np.random.rand(30, 63).astype(np.float32)
    aug_seq = augment_sequence(fake_seq)
    assert aug_seq.shape == (30, 63), f"Wrong aug seq shape: {aug_seq.shape}"
    print(f"  Sequence augmentation: PASSED")

    print("SequenceBuffer test PASSED.")
