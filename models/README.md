# Models Directory

This directory stores trained classifier models.

## Files

- `sign_model.pkl` — Trained scikit-learn classifier (created by `scripts/train_model.py`)
- `label_encoder.pkl` — Label encoder mapping class indices to sign labels

## How to Generate

```bash
python scripts/train_model.py
```

Models are generated from data collected via `scripts/collect_data.py`.
