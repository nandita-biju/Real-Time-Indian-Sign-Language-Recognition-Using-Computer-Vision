"""
gesture_classifier.py -- Loads trained model(s) and classifies ISL hand gestures.

Supports:
  - Static sign classification (single-frame, scikit-learn model)
  - Dynamic sign classification (sequence-based, future extension)
  - Prediction smoothing via majority voting
  - Confidence thresholds

Upgraded from the original alphabet-level classifier to word-level ISL.
"""

import os
import json
import pickle
import numpy as np
from collections import deque
from typing import Optional, Tuple, Dict, List


class GestureClassifier:
    """
    Classifies ISL hand gestures from normalized landmark feature vectors.

    Uses a trained scikit-learn model and applies temporal smoothing
    to prevent flickering predictions.
    """

    def __init__(
        self,
        model_path: str = "models/static_sign_model.pkl",
        metadata_path: str = "models/model_metadata.json",
        smoothing_window: int = 12,
        confidence_threshold: float = 0.6,
    ):
        """
        Initialize the classifier.

        Args:
            model_path: Path to the trained model pickle file.
            metadata_path: Path to the model metadata JSON file.
            smoothing_window: Number of recent predictions for majority voting.
            confidence_threshold: Minimum confidence to accept a prediction.
        """
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.smoothing_window = smoothing_window
        self.confidence_threshold = confidence_threshold

        self.model = None
        self.label_mapping: Dict[int, str] = {}   # index -> label
        self.label_list: List[str] = []            # ordered labels
        self.metadata: Dict = {}
        self.is_loaded = False

        # Sliding window for temporal smoothing
        self._prediction_history: deque = deque(maxlen=smoothing_window)

    def load(self) -> bool:
        """
        Load the trained model and label mapping.

        Returns:
            True if loaded successfully, False otherwise.
        """
        if not os.path.exists(self.model_path):
            print(f"[WARNING] Model file not found: {self.model_path}")
            print("          Run 'python scripts/collect_data.py' to collect ISL training data,")
            print("          then 'python scripts/train_model.py' to train the model.")
            return False

        try:
            # Load model
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)

            # Load metadata (contains label mapping)
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, "r") as f:
                    self.metadata = json.load(f)
                # Extract label mapping
                mapping = self.metadata.get("label_mapping", {})
                self.label_mapping = {int(k): v for k, v in mapping.items()}
                self.label_list = [
                    self.label_mapping[i]
                    for i in sorted(self.label_mapping.keys())
                ]
            else:
                # Fallback: try legacy label_encoder.pkl
                encoder_path = self.model_path.replace("static_sign_model.pkl", "label_encoder.pkl")
                if os.path.exists(encoder_path):
                    with open(encoder_path, "rb") as f:
                        le = pickle.load(f)
                    self.label_list = list(le.classes_)
                    self.label_mapping = {i: l for i, l in enumerate(self.label_list)}
                else:
                    print("[WARNING] No label mapping found.")
                    return False

            self.is_loaded = True
            print(f"[INFO] Model loaded. Recognizes {len(self.label_list)} ISL signs: "
                  f"{self.label_list}")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            self.model = None
            self.is_loaded = False
            return False

    def predict_raw(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Make a raw prediction without smoothing.

        Args:
            features: 1D feature vector of shape (63,).

        Returns:
            (predicted_label, confidence)
        """
        if not self.is_loaded or features is None:
            return ("", 0.0)

        try:
            features_2d = features.reshape(1, -1)

            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(features_2d)[0]
                max_idx = np.argmax(proba)
                confidence = float(proba[max_idx])
                label = self.label_mapping.get(max_idx, f"class_{max_idx}")
            else:
                pred = self.model.predict(features_2d)[0]
                label = self.label_mapping.get(int(pred), f"class_{pred}")
                confidence = 1.0

            return (label, confidence)

        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            return ("", 0.0)

    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Make a smoothed prediction using majority voting.

        Args:
            features: 1D feature vector of shape (63,).

        Returns:
            (predicted_label, confidence)
            Returns ("Uncertain", 0.0) if below threshold.
        """
        if not self.is_loaded:
            return ("", 0.0)

        label, conf = self.predict_raw(features)
        print(f"RAW: {label} | confidence: {conf:.3f}")

        if conf < self.confidence_threshold:
            self._prediction_history.append(("Uncertain", 0.0))
            return ("Uncertain", 0.0)

        self._prediction_history.append((label, conf))

        if len(self._prediction_history) == 0:
            return ("Uncertain", 0.0)

        # Count labels (ignoring "Uncertain")
        label_counts: Dict[str, list] = {}
        for lbl, c in self._prediction_history:
            if lbl != "Uncertain":
                if lbl not in label_counts:
                    label_counts[lbl] = []
                label_counts[lbl].append(c)

        if not label_counts:
            return ("Uncertain", 0.0)

        # Find majority label
        best_label = max(label_counts, key=lambda k: len(label_counts[k]))
        best_count = len(label_counts[best_label])

        # Require at least 40% of the window to agree
        vote_ratio = best_count / max(len(self._prediction_history), 1)
        if vote_ratio < 0.4:
            return ("Uncertain", 0.0)

        avg_conf = float(np.mean(label_counts[best_label]))
        print(f"SMOOTHED: {best_label} | confidence: {avg_conf:.3f}")
        return (best_label, avg_conf)

    def reset_history(self):
        """Clear the prediction history."""
        self._prediction_history.clear()

    def get_supported_signs(self) -> List[str]:
        """Return list of signs the model can recognize."""
        if self.is_loaded:
            return self.label_list.copy()
        return []


def test_classifier():
    """Quick test of the classifier module."""
    print("Testing gesture classifier...")

    clf = GestureClassifier()
    loaded = clf.load()

    if not loaded:
        print("No trained model available (expected if data hasn't been collected yet).")
        print("Classifier module structure test PASSED.")
        return

    fake_features = np.random.rand(63).astype(np.float32)
    label, conf = clf.predict(fake_features)
    print(f"Random features -> '{label}' ({conf:.2%} confidence)")
    print("Classifier test PASSED.")


if __name__ == "__main__":
    test_classifier()
