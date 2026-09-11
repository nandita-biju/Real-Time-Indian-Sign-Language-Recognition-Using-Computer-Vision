# ISL Dataset & Vocabulary Validation Report

> **Generated:** 2026-09-10
> **Project:** Real-Time Indian Sign Language Recognition
> **Validation method:** Cross-referenced with ISLRTC, indiansignlanguage.org, and INCLUDE dataset documentation

---

## 1. Verified Vocabulary (23 Signs)

| # | Sign | ISL Verified? | Static/Dynamic | Hands | Landmark-Only? | MVP? | Issues |
|---|------|--------------|----------------|-------|---------------|------|--------|
| 1 | HELLO | **VERIFIED** | Static | 1H | **YES** | **MVP** | 1-hand palm wave. Fingers together (vs STOP's spread fingers) |
| 2 | GOOD | **VERIFIED** | Static | 1H | **YES** | **MVP** | Thumbs-up. Very distinct hand shape |
| 3 | BAD | **VERIFIED** | Static | 1H | **YES** | **MVP** | Pinky extended (ISL-specific, not thumbs-down) |
| 4 | STOP | **VERIFIED** | Static | 1H | **YES** | **MVP** | Open flat hand, fingers spread, palm outward |
| 5 | COME | **VERIFIED** | Dynamic | 1H | **YES** | **MVP** | Beckoning curl. Distinct temporal motion |
| 6 | I/ME | **VERIFIED** | Static | 1H | **NO** | Extended | Index pointing at chest -- same hand shape as YOU |
| 7 | YOU | **VERIFIED** | Static | 1H | **NO** | Extended | Index pointing outward -- same hand shape as I/ME |
| 8 | YES | **VERIFIED** | **Dynamic** | 1H | **NO** | Extended | **Reclassified**: was "static". Fist nodding requires motion. Head nod is primary ISL form |
| 9 | NO | **CORRECTED** | **Dynamic** | 1H | **NO** | Extended | **Reclassified**: was "static". Original had ASL "thumb snapping". ISL uses finger wag + head shake |
| 10 | WAIT | **VERIFIED** | **Dynamic** | 1H | **NO** | Extended | **Reclassified**: was "static". Flat hand pressing down = confused with STOP/HELLO if static |
| 11 | HOME | **VERIFIED** | Static | **2H** | YES (shape) | Extended | Distinct triangle shape but needs **2-hand feature extraction** (not implemented) |
| 12 | THANK YOU | **VERIFIED** | Dynamic | 1H | **NO** | Extended | Hand from chin/mouth -- needs face/pose landmarks |
| 13 | SORRY | **CORRECTED** | Dynamic | 1H | **NO** | Extended | **Original desc was ASL** ("fist circling on chest"). ISL: open hand on chest pulling away |
| 14 | WATER | **VERIFIED** | Dynamic | 1H | **NO** | Extended | Drinking motion needs mouth proximity detection |
| 15 | FOOD | **VERIFIED** | Dynamic | 1H | **NO** | Extended | Fingertips to mouth -- needs face landmarks |
| 16 | MOTHER | **VERIFIED** | Dynamic | 1H | **NO** | Extended | Nose touch -- **identical hand shape to FATHER/I_ME/YOU** without face landmarks |
| 17 | FATHER | **VERIFIED** | Dynamic | 1H | **NO** | Extended | Lip trace -- **identical hand shape to MOTHER/I_ME/YOU** without face landmarks |
| 18 | HAPPY | **NEEDS_VERIFICATION** | Dynamic | 1-2H | **NO** | Extended | ISL/ASL overlap unclear. Smiling face is mandatory |
| 19 | HELP | **CORRECTED** | Dynamic | **2H** | YES (shape) | Extended | **Original desc was ASL** ("thumbs-up on palm"). ISL: open palms pushing forward |
| 20 | PLEASE | **VERIFIED** | Dynamic | **2H** | **NO** | Extended | Namaste shape -- identical to formal HELLO without face context |
| 21 | FRIEND | **CORRECTED** | Static | **2H** | YES (shape) | Extended | **Original desc was ASL** ("hooked index fingers"). ISL: handshake/interlocked fingers |
| 22 | FAMILY | **CORRECTED** | Dynamic | **2H** | YES | Extended | **Original desc was ASL** ("curved hands sweeping"). ISL: circular motion for group |
| 23 | NAME | **CORRECTED** | Dynamic | **2H** | **NO** | Extended | **Original desc was ASL** ("H-hands tapping"). ISL: finger/thumb on left palm |

---

## 2. Critical Findings

### 2.1 Signs with INCORRECT (ASL) Descriptions -- NOW CORRECTED

> [!CAUTION]
> 5 signs had descriptions that were **American Sign Language (ASL), not Indian Sign Language (ISL)**. These have been corrected in `vocabulary.py`.

| Sign | Original (ASL) Description | Corrected (ISL) Description |
|------|---------------------------|----------------------------|
| SORRY | "Fist circling on chest" | Open hand on chest pulling away |
| HELP | "Thumbs-up on flat palm rising" | Open palms pushing forward |
| FRIEND | "Index fingers hooked together" | Handshake / interlocked fingers |
| FAMILY | "Curved hands sweeping outward" | Circular motion indicating group |
| NAME | "H-hands tapping across each other" | Finger/thumb sliding on left palm |
| NO | "Extended fingers with thumb snapping" | Index finger wagging (partially ASL) |

### 2.2 Signs Reclassified Static -> Dynamic

| Sign | Reason |
|------|--------|
| YES | A static fist is just a closed hand -- ambiguous. The wrist nod/head nod is what makes it YES |
| NO | Index finger wag or hand twist is inherently temporal |
| WAIT | Flat hand pressing down is identical to STOP/HELLO if captured as static pose |

### 2.3 Signs That CANNOT Work with Hand Landmarks Only

> [!WARNING]
> The current system uses only 21 hand landmark coordinates (x, y, z relative to the hand bounding box). It has **zero knowledge** of where the hand is relative to the face, nose, mouth, chest, or body. This means:

**Indistinguishable sign pairs/groups:**

| Group | Signs | Why |
|-------|-------|-----|
| Pointing signs | **I/ME, YOU** | Both = index finger extended. Only direction differs, which landmarks can't capture |
| Face-touch signs | **MOTHER, FATHER** | Both = index finger. MOTHER=nose, FATHER=lip. Without face landmarks, identical |
| Face-proximity signs | **WATER, FOOD, THANK YOU** | All involve hand near face/mouth. Without pose, just a moving hand shape |
| Flat-hand signs | **STOP, WAIT, HELLO** | All are open flat hands. Only orientation/location differs |

---

## 3. Recommended MVP Vocabulary (5 signs + NEUTRAL)

> [!IMPORTANT]
> With **hand-landmark-only** classification, only signs with **truly unique hand shapes** can be reliably distinguished.

| # | Sign | Type | Hand Shape | Why It Works |
|---|------|------|-----------|-------------|
| 1 | **GOOD** | Static | Thumbs-up (thumb up, 4 fingers curled) | Unique: no other sign uses this shape |
| 2 | **BAD** | Static | Pinky extended (4 fingers + thumb curled) | Unique: ISL-specific pinky gesture |
| 3 | **STOP** | Static | All 5 fingers spread, palm out | Distinguishable: spread fingers vs together |
| 4 | **HELLO** | Static | 5 fingers together, palm wave | Distinguishable: fingers together vs spread |
| 5 | **COME** | Dynamic | Beckoning curl (fingers curling inward) | Unique temporal motion: finger curl sequence |
| -- | **_NEUTRAL** | Static | Relaxed hand / no sign | Baseline "no sign" class |

**Total: 6 classes (5 signs + neutral)**

This is deliberately small. The goal is a **reliable, honest MVP** that actually works, rather than a 23-class model that confuses half the signs.

### How to Expand Beyond MVP

To add the extended vocabulary, the system needs:

1. **MediaPipe Pose landmarks** -- to detect hand-to-body location (which part of the body the hand is near)
2. **Two-hand feature extraction** -- for HOME, HELP, FRIEND, FAMILY, NAME
3. **Face landmarks** -- for MOTHER (nose), FATHER (lip), WATER/FOOD (mouth)
4. **Temporal sequence model** -- for YES, NO, WAIT, and all dynamic signs

---

## 4. Extended Vocabulary (18 signs -- future phases)

### Phase 2: Add Pose Landmarks (enables body-relative detection)
| Sign | Requirement |
|------|------------|
| I/ME, YOU | Pose landmarks to detect pointing direction |
| YES, NO | Temporal model + head/wrist movement tracking |
| WAIT | Temporal model (downward pressing motion) |
| SORRY, THANK YOU | Pose landmarks for chest/chin proximity |
| HAPPY | Pose + face landmarks (chest brush + smile) |

### Phase 3: Add Two-Hand Feature Extraction
| Sign | Requirement |
|------|------------|
| HOME | 2-hand landmark extraction (126 features) |
| HELP, FRIEND, FAMILY, NAME | 2-hand + temporal model |
| PLEASE | 2-hand + facial expression |

### Phase 4: Add Face Landmarks
| Sign | Requirement |
|------|------------|
| MOTHER | Face landmark (nose position) |
| FATHER | Face landmark (upper lip position) |
| WATER, FOOD | Face landmark (mouth position) |

---

## 5. Dataset Options

### Public ISL Datasets Found

| Dataset | Classes | Samples | Format | License | Suitability |
|---------|---------|---------|--------|---------|-------------|
| **INCLUDE** (AI4Bharat) | 263 words | 4,287 videos | Video (MP4) | CC-BY-NC 4.0 | Good for future phases. Requires running MediaPipe over videos. No direct landmark files |
| **ISL MediaPipe Holistic Landmarks** (Kaggle) | ~263 (INCLUDE-derived) | Derived from INCLUDE | Parquet (553 landmarks/frame: hands+face+pose) | CC-BY-NC (derived) | **Best option for landmark-based approach** -- includes pose+face landmarks |
| **CISLR** (Exploration-Lab) | ~4,700 words | Large | Video | Research/Academic | Too large for MVP. Good for future scaling |
| **ISL Alphabets & Digits** (Kaggle) | 36 (A-Z + 0-9) | 36,000-50,000 images | Image | CC0 | Not suitable -- alphabet-level, not word-level |

### Recommendation

> [!NOTE]
> For the MVP (5 signs with unique hand shapes), **webcam self-collection is the best approach**. The signs are simple enough that 50-100 samples per sign with auto-capture will produce reliable results.
>
> For future phases requiring pose/face landmarks, the **ISL MediaPipe Holistic Landmarks** Kaggle dataset would be valuable, but it uses a different feature format (553 dims vs our 63 dims) and would require adapting the feature pipeline.

---

## 6. Data Collection Plan

### MVP Collection Protocol

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Signs to collect | 5 + NEUTRAL = 6 classes | MVP vocabulary |
| Samples per sign | **80-120 real samples** | Enough for train/val/test split with augmentation |
| Recording sessions | **3-5 separate sessions** | Different lighting, angles, distances to prevent overfitting |
| Samples per session | ~20-30 per sign | Spread across sessions |
| Auto-capture interval | 0.08s (~12 samples/sec) | Hold sign, vary position slightly |
| Frame variation | Move hand slightly between captures | Prevents identical-frame overfitting |

### Collection Steps

```
Session 1: Collect ~25 samples per sign (good lighting, centered)
Session 2: Collect ~25 samples per sign (different lighting, slightly off-center)
Session 3: Collect ~25 samples per sign (different distance, hand angle variation)
Optional Session 4-5: More variation
```

### For NEUTRAL class
- Collect with relaxed hand, fist, flat table rest, no hand visible
- Include variety of "non-sign" hand positions

### Train/Validation/Test Split

| Set | Percentage | Purpose |
|-----|-----------|---------|
| Training | 70% | Model fitting |
| Validation | 15% | Hyperparameter tuning (via sklearn's built-in) |
| Test | 15% | Final evaluation (never seen during training) |

Split is done at the **sample level BEFORE augmentation**. Augmentation (noise, rotation, scale) is applied **only to training samples** to prevent data leakage.

---

## 7. Data Leakage Prevention

> [!CAUTION]
> **Critical bug found and fixed:** The original `collect_data.py` saved augmented copies during collection, mixed into the same `.npy` file as originals. This meant `train_test_split()` could place an augmented copy in the test set while the source sample was in training -- **data leakage**.

### Fix Applied:

1. **Removed augmentation from `collect_data.py`** -- only real samples are saved during collection
2. **Added augmentation to `train_model.py`** -- augmentation is applied AFTER `train_test_split()`, only to the training portion
3. Each training sample generates 2 augmented copies (noise + rotation + scale)

### Augmentation Parameters (training only):
- Gaussian noise: std=0.01
- Scale: 0.92x to 1.08x
- Rotation: up to 8 degrees (z-axis)
- Translation: up to 5% of coordinate range

---

## 8. Two-Hand Support Assessment

### Current State

| Component | Two-Hand Support |
|-----------|-----------------|
| `hand_tracker.py` | **YES** -- `max_num_hands=2` in app.py, returns multiple `HandLandmarks` |
| `feature_extractor.py` | **NO** -- only handles single `(21, 3)` input, returns `(63,)` |
| `app.py` | **NO** -- uses only `hands[0]` (first detected hand) |
| `collect_data.py` | **NO** -- uses only `hands[0]` |

### Implication

Two-hand signs (HOME, HELP, FRIEND, FAMILY, NAME, PLEASE) are in the vocabulary but **cannot be properly classified** with the current feature pipeline. They would need a `(42, 3) -> (126,)` or `(2, 21, 3) -> (126,)` feature format.

### Decision for MVP

**Two-hand signs are excluded from the MVP.** All 5 MVP signs are one-hand signs. The tracker detects 2 hands (for drawing), but classification uses hand[0] only.

---

## 9. Facial Expression / Body Movement Dependencies

| Sign | Requires Face | Requires Body/Pose | Can Work Without |
|------|--------------|-------------------|-----------------|
| GOOD | No | No | **YES** |
| BAD | Negative expression helps | No | **YES** (pinky shape is sufficient) |
| STOP | No | No | **YES** |
| HELLO | Smile helps | No | **YES** (palm shape is sufficient) |
| COME | No | No | **YES** (finger motion is sufficient) |
| MOTHER | No | **YES** (nose proximity) | **NO** |
| FATHER | No | **YES** (lip proximity) | **NO** |
| SORRY | Apologetic face | **YES** (chest) | **NO** |
| YES | **YES** (head nod) | No | **NO** |
| NO | **YES** (head shake) | No | **NO** |
| HAPPY | **YES** (smile mandatory) | **YES** (chest) | **NO** |
| WATER | No | **YES** (mouth) | **NO** |
| FOOD | No | **YES** (mouth) | **NO** |
| THANK YOU | No | **YES** (chin) | **NO** |

---

## 10. Known Limitations

1. **Hand landmarks only** -- no body/face reference frame. Cannot determine hand location relative to body parts
2. **One-hand features only** -- (21, 3) -> (63,) feature vector. Two-hand signs are unsupported
3. **No temporal model trained yet** -- dynamic signs (COME) can be collected but the sequence classifier is not yet trained
4. **MediaPipe z-coordinate is unreliable** -- depth estimation from a single camera is noisy
5. **User-specific model** -- trained on one person's data. May not generalize without diverse training data
6. **No ISL grammar** -- words are concatenated left-to-right, not SOV (ISL word order)
7. **Lighting dependent** -- MediaPipe hand detection degrades in poor lighting
8. **Camera angle** -- best results with hand directly facing camera

---

## 11. Code Changes Made During This Validation

| File | Change | Reason |
|------|--------|--------|
| [vocabulary.py](file:///c:/Users/nandi/OneDrive/Desktop/Projects/Misc/camera%20and%20gestures%20project/sign_language_recognition/src/vocabulary.py) | Corrected 5 ASL descriptions to ISL; reclassified YES/NO/WAIT as dynamic; added `verification_status`, `landmark_only`, `notes` fields; created `MVP_VOCABULARY` and `EXTENDED_VOCABULARY` | Accuracy and honesty |
| [collect_data.py](file:///c:/Users/nandi/OneDrive/Desktop/Projects/Misc/camera%20and%20gestures%20project/sign_language_recognition/scripts/collect_data.py) | Removed augmentation from collection | Data leakage prevention |
| [train_model.py](file:///c:/Users/nandi/OneDrive/Desktop/Projects/Misc/camera%20and%20gestures%20project/sign_language_recognition/scripts/train_model.py) | Added post-split augmentation to training data only | Data leakage prevention |
| [test_modules.py](file:///c:/Users/nandi/OneDrive/Desktop/Projects/Misc/camera%20and%20gestures%20project/sign_language_recognition/test_modules.py) | Updated vocabulary tests for new API | Test compatibility |

---

## 12. What You Should Do Next

1. **Collect MVP data** -- Run `python scripts/collect_data.py` and collect **80-120 samples** for each of: `GOOD`, `BAD`, `STOP`, `HELLO`, `_NEUTRAL`. Skip COME for now (it's dynamic and needs the sequence classifier).
2. **Collect across multiple sessions** -- At least 2-3 sessions with different lighting/distance
3. **Train** -- Run `python scripts/train_model.py`
4. **Test** -- Run `python app.py` and verify recognition of the 4 static MVP signs
5. **After MVP works** -- Decide whether to add pose/face landmarks (requires switching to MediaPipe Holistic or combining Hand + Pose models)
