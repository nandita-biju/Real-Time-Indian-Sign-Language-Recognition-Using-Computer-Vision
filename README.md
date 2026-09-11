# Real-Time Indian Sign Language Recognition

A computer-vision application that uses your webcam to detect hand gestures and recognize **Indian Sign Language (ISL) words** in real time. Recognized words are accumulated into sentences.

> **Important:** This is a **Sign Recognition** system, not a full **Sign Language Translator**. It recognizes individual ISL signs and maps them to words. Full ISL translation would require understanding ISL grammar (SOV word order), spatial references, and non-manual markers -- which is significantly more complex.

## Features

- **Real-time hand tracking** using MediaPipe HandLandmarker (Tasks API)
- **ISL word recognition** -- recognizes whole words, not individual letters
- **22 ISL signs** with verified descriptions from ISLRTC/ISL references
- **Static + dynamic sign architecture** (static classifier ready; dynamic extension supported)
- **Neutral state detection** -- prevents accidental repeated recognition
- **Prediction smoothing** via majority voting over sliding window
- **Confidence thresholds** -- displays "Uncertain" when model is unsure
- **Sentence builder** -- accumulates words with intelligent debouncing
- **Word-level debouncing** -- holding a sign adds it only once
- **Text-to-speech** (optional, via pyttsx3)
- **Data augmentation** -- noise, rotation, scale, and translation during collection
- **Professional UI** with ISL branding, confidence bar, and sentence display

## ISL Vocabulary (22 Signs)

### Static Signs (Single Hand Pose)
| Sign | Hindi | Hands | Description |
|------|-------|-------|-------------|
| HELLO | Namaste | 1H | Open palm raised; or palms pressed together |
| GOOD | Accha | 1H | Thumbs-up gesture |
| BAD | Bura | 1H | Thumbs-down or pinky extended |
| I/ME | Main | 1H | Index finger pointing at own chest |
| YOU | Tum/Aap | 1H | Index finger pointing outward |
| YES | Haan | 1H | Fist with wrist nodding (static: fist held) |
| NO | Nahi | 1H | Index finger wagged or extended fingers |
| STOP | Ruko | 1H | Open palm facing outward |
| WAIT | Intezaar | 1H | Palm down pressing gesture |
| HOME | Ghar | 2H | Hands form triangular roof shape |

### Dynamic Signs (Movement Required)
| Sign | Hindi | Hands | Description |
|------|-------|-------|-------------|
| THANK YOU | Dhanyavaad | 1H | Hand from chin moving forward |
| SORRY | Maafi | 1H | Fist circling on chest |
| WATER | Paani | 1H | Drinking motion to mouth |
| FOOD | Khana | 1H | Fingertips to mouth repeatedly |
| COME | Aao | 1H | Beckoning motion toward chest |
| MOTHER | Maa | 1H | Touch side of nose (nose pin reference) |
| FATHER | Papa | 1H | Trace imaginary mustache |
| HAPPY | Khush | 1H | Palm brushing upward on chest |
| HELP | Madad | 2H | Thumbs-up on flat palm, rising |
| PLEASE | Kripaya | 2H | Joined palms moving down |
| FRIEND | Dost | 2H | Index fingers hooked together |
| FAMILY | Parivaar | 2H | Curved hands sweeping outward |
| NAME | Naam | 2H | Fingers tapping across each other |

> **Note:** These signs follow ISL as documented by ISLRTC. ISL is different from ASL (American Sign Language). For example, ISL MOTHER references the nose pin (nath), while ASL MOTHER taps the chin.

## Architecture

```
Webcam
  |
  v
Camera Module (OpenCV + DirectShow fallback)
  |
  v
Hand Tracker (MediaPipe HandLandmarker Tasks API)
  |
  v
21 Hand Landmarks (x, y, z)
  |
  v
Feature Extraction (wrist-relative normalization -> 63-dim vector)
  |
  +--> Static Classifier (RF/SVM/MLP) --> Word prediction
  |
  +--> [Future] Sequence Buffer --> Dynamic Classifier (LSTM/GRU)
  |
  v
Prediction Smoothing (majority voting, confidence filter)
  |
  v
Sentence Builder (state machine: IDLE -> RECOGNIZING -> CONFIRMED -> WAIT_NEUTRAL)
  |
  v
UI Overlay (OpenCV side panel with sign, confidence, sentence)
```

## Project Structure

```
sign_language_recognition/
|-- app.py                          # Main application
|-- requirements.txt
|-- README.md
|-- test_modules.py                 # Integration tests
|
|-- src/
|   |-- __init__.py
|   |-- camera.py                   # Webcam access
|   |-- hand_tracker.py             # MediaPipe hand detection
|   |-- feature_extractor.py        # Landmark normalization
|   |-- vocabulary.py               # ISL sign definitions & metadata
|   |-- gesture_classifier.py       # Model loading + smoothing
|   |-- sentence_builder.py         # Word accumulation + debouncing
|   |-- sequence_buffer.py          # Frame buffering + augmentation
|   |-- ui.py                       # OpenCV overlay rendering
|   |-- translator.py               # Legacy (kept for reference)
|
|-- models/
|   |-- hand_landmarker.task        # MediaPipe model
|   |-- static_sign_model.pkl       # Trained classifier (after training)
|   |-- model_metadata.json         # Model metadata (after training)
|
|-- data/
|   |-- raw/static/                 # Static sign .npy files
|   |-- raw/dynamic/                # Dynamic sign sequences
|   |-- processed/
|   |-- README.md
|
|-- scripts/
|   |-- collect_data.py             # ISL data collection tool
|   |-- train_model.py              # Model training pipeline
|
|-- assets/
    |-- README.md
```

## Installation

### Prerequisites
- Python 3.9+
- Webcam
- Windows (tested), macOS, or Linux

### Install Dependencies
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install opencv-python mediapipe numpy scikit-learn pyttsx3
```

## Usage

### Step 1: Collect ISL Training Data
```bash
python scripts/collect_data.py
```

1. Use LEFT/RIGHT arrows to navigate between ISL signs
2. Read the sign description at the bottom of the screen
3. Perform the ISL sign in front of the camera
4. Press **S** for a single snapshot (static signs)
5. Press **A** to toggle auto-capture (collects rapidly while held)
6. Press **R** to start/stop recording a dynamic sequence
7. Collect **50-150+ samples per sign**
8. Press **Q** to quit and save

### Step 2: Train the Model
```bash
python scripts/train_model.py
```

Trains 4 classifiers (Random Forest, KNN, SVM, MLP), selects the best by F1 score, and saves to `models/`.

### Step 3: Run ISL Recognition
```bash
python app.py
```

**Controls:**
| Key | Action |
|-----|--------|
| BACKSPACE | Remove last word |
| C | Clear sentence |
| T | Speak sentence (TTS) |
| Q / ESC | Quit |

## Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.12 |
| Hand Detection | MediaPipe Tasks API | 0.10.33 |
| Computer Vision | OpenCV | 4.13.0 |
| ML Classification | scikit-learn | 1.8.0 |
| Numerical | NumPy | 2.4.3 |
| Text-to-Speech | pyttsx3 | 2.99 |

## ISL References

- **Indian Sign Language Research and Training Centre (ISLRTC)**
  http://www.islrtc.nic.in
- **Indian Sign Language Dictionary Portal**
  https://indiansignlanguage.org
- **INCLUDE Dataset** (AI4Bharat, ACM MM 2020)
  Sridhar et al., "INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition"
  https://github.com/AI4Bharat/INCLUDE

## Limitations

- **Vocabulary size:** Currently 22 ISL signs (expandable)
- **Static signs only (MVP):** Dynamic signs are architecturally supported but require sequence data collection and temporal model training
- **Single hand (primary):** Two-hand signs are recognized based on the dominant hand's pose; full two-hand feature extraction is a future enhancement
- **User-specific model:** Trained on your own data; may not generalize to other users without retraining
- **Lighting sensitive:** Works best in well-lit environments
- **No ISL grammar:** Words are concatenated left-to-right; ISL uses SOV word order which is not applied
- **Camera angle:** Best results with hand directly facing the camera

## Future Improvements

- **Dynamic sign recognition** using LSTM/GRU on frame sequences
- **Two-hand feature extraction** for bimanual signs
- **Larger ISL vocabulary** (100+ signs)
- **Pre-trained models** using INCLUDE dataset
- **ISL grammar modeling** (SOV word order transformation)
- **Text-to-speech in Hindi**
- **Mobile deployment** (TFLite)
- **Edge AI / Raspberry Pi** deployment

## License

This project is for educational and research purposes.
