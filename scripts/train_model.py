"""
train_model.py -- Train ISL word-level gesture classifiers.

Loads collected static and dynamic ISL sign data, trains classifiers,
evaluates performance, and saves the best model with metadata.

Usage:
    python scripts/train_model.py

Outputs:
    models/static_sign_model.pkl    -- Trained static classifier
    models/model_metadata.json      -- Model metadata and label mapping
"""

import os
import sys
import json
import pickle
import time
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

from src.vocabulary import ISL_SIGNS, get_static_labels, get_all_labels, NEUTRAL_LABEL
from src.sequence_buffer import augment_landmarks
from src.feature_extractor import extract_features

# Paths
STATIC_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "static")
DYNAMIC_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "dynamic")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
STATIC_MODEL_PATH = os.path.join(MODEL_DIR, "static_sign_model.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

NUM_FEATURES = 63


def load_static_data():
    """Load all static sign data from .npy files."""
    if not os.path.exists(STATIC_DIR):
        print(f"[ERROR] Static data directory not found: {STATIC_DIR}")
        print("        Run 'python scripts/collect_data.py' first.")
        sys.exit(1)

    all_features = []
    all_labels = []
    label_counts = {}

    # Load each sign's data
    for fname in sorted(os.listdir(STATIC_DIR)):
        if not fname.endswith(".npy"):
            continue
        label = fname.replace(".npy", "")
        path = os.path.join(STATIC_DIR, fname)

        try:
            data = np.load(path)
            if data.ndim != 2 or data.shape[1] != NUM_FEATURES:
                print(f"  [WARNING] Skipping {fname}: wrong shape {data.shape}")
                continue

            all_features.append(data)
            all_labels.extend([label] * len(data))
            label_counts[label] = len(data)
        except Exception as e:
            print(f"  [WARNING] Error loading {fname}: {e}")

    if not all_features:
        print("[ERROR] No valid static training data found.")
        sys.exit(1)

    X = np.vstack(all_features).astype(np.float32)
    y = np.array(all_labels)

    return X, y, label_counts


def print_data_summary(X, y, label_counts):
    """Print training data statistics."""
    print("=" * 60)
    print("  TRAINING DATA SUMMARY")
    print("=" * 60)
    print()
    print(f"Total samples: {len(y)}")
    print(f"Features per sample: {X.shape[1]}")
    print(f"Classes: {len(label_counts)}")
    print()
    print("Samples per class:")
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        bar = "#" * min(count // 5, 40)
        display = label
        if label in ISL_SIGNS:
            display = ISL_SIGNS[label].label
        elif label == NEUTRAL_LABEL:
            display = "NEUTRAL"
        print(f"  {display:>15s}: {count:>5d}  {bar}")
    print()

    # Warn about imbalance
    counts = list(label_counts.values())
    if counts:
        min_c, max_c = min(counts), max(counts)
        if max_c > 3 * min_c:
            print(f"[WARNING] Class imbalance (min={min_c}, max={max_c}).")
            print()


def train_and_compare(X_train, X_test, y_train, y_test, labels):
    """Train multiple models and return the best one."""
    print("=" * 60)
    print("  MODEL TRAINING & COMPARISON")
    print("=" * 60)
    print()

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=25, random_state=42, n_jobs=-1
        ),
        "KNN (k=7)": KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42, C=10),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(256, 128, 64), max_iter=500,
            random_state=42, early_stopping=True, validation_fraction=0.15,
        ),
    }

    results = {}
    for name, model in models.items():
        print(f"Training {name}...", end=" ", flush=True)
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        results[name] = {
            "model": model,
            "accuracy": acc,
            "f1": f1,
            "time": elapsed,
            "predictions": y_pred,
        }
        print(f"Acc: {acc:.4f} | F1: {f1:.4f} | {elapsed:.2f}s")

    # Select best by F1
    best_name = max(results, key=lambda k: results[k]["f1"])
    best = results[best_name]
    print()
    print(f">>> Best model: {best_name} (F1={best['f1']:.4f})")
    print()
    return best_name, best


def print_metrics(y_test, y_pred, model_name):
    """Print detailed metrics."""
    print("=" * 60)
    print(f"  DETAILED METRICS -- {model_name}")
    print("=" * 60)
    print()
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print()
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    labels_sorted = sorted(set(y_test) | set(y_pred))
    print("Confusion Matrix:")
    header = "     " + "  ".join(f"{l[:4]:>4s}" for l in labels_sorted)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:4d}" for v in row)
        print(f"  {labels_sorted[i][:4]:>4s}  {row_str}")
    print()


def save_model(model, label_mapping, model_name, X_train, X_test, y_test, y_pred):
    """Save model and metadata."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save model
    with open(STATIC_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved: {STATIC_MODEL_PATH}")

    # Build metadata
    metadata = {
        "model_type": model_name,
        "sign_type": "static",
        "feature_dimensions": NUM_FEATURES,
        "vocabulary": sorted(label_mapping.values()),
        "label_mapping": {str(k): v for k, v in label_mapping.items()},
        "num_classes": len(label_mapping),
        "training_date": datetime.now().isoformat(),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "test_precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "test_recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "dataset_source": "User-collected via webcam (scripts/collect_data.py)",
        "notes": "Static ISL sign classifier. Dynamic signs require separate temporal model.",
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved: {METADATA_PATH}")


def main():
    print()

    # Load data
    X, y, label_counts = load_static_data()
    print_data_summary(X, y, label_counts)

    # Create label mapping
    unique_labels = sorted(set(y))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    idx_to_label = {i: l for l, i in label_to_idx.items()}
    y_encoded = np.array([label_to_idx[l] for l in y])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # Augment TRAINING data only (prevents data leakage)
    print("Augmenting training data...", end=" ", flush=True)
    aug_X, aug_y = [], []
    for i in range(len(X_train)):
        features = X_train[i]
        label = y_train[i]
        # Reshape features back to (21, 3) for augmentation
        landmarks = features.reshape(21, 3)
        for _ in range(2):  # 2 augmented copies per sample
            aug_lm = augment_landmarks(landmarks, noise_std=0.01,
                                       scale_range=(0.92, 1.08),
                                       rotation_max_deg=8.0)
            aug_feat = extract_features(aug_lm)
            if aug_feat is not None:
                aug_X.append(aug_feat)
                aug_y.append(label)
    if aug_X:
        X_train = np.vstack([X_train, np.array(aug_X, dtype=np.float32)])
        y_train = np.concatenate([y_train, np.array(aug_y)])
    print(f"Train after augmentation: {len(X_train)}")
    print()

    # Train
    best_name, best = train_and_compare(X_train, X_test, y_train, y_test, unique_labels)

    # Decode for metrics
    y_test_labels = [idx_to_label[i] for i in y_test]
    y_pred_labels = [idx_to_label[i] for i in best["predictions"]]

    print_metrics(y_test_labels, y_pred_labels, best_name)

    # Save
    save_model(best["model"], idx_to_label, best_name,
               X_train, X_test, y_test_labels, y_pred_labels)

    print()
    print("=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print()
    print("Next: python app.py")


if __name__ == "__main__":
    main()
