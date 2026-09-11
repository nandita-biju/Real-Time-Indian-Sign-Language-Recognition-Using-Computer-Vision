# Dataset Directory

## Structure

```
data/
├── raw/
│   ├── static/          # Static sign landmark data (.npy files)
│   │   ├── HELLO.npy    # Shape: (N, 63) -- N samples, 63 features each
│   │   ├── GOOD.npy
│   │   ├── _NEUTRAL.npy
│   │   └── ...
│   └── dynamic/         # Dynamic sign sequences
│       ├── THANK_YOU/
│       │   ├── seq_0000.npy   # Shape: (T, 63) -- T frames per sequence
│       │   ├── seq_0001.npy
│       │   └── ...
│       └── ...
└── processed/           # (Reserved for future preprocessing)
```

## Data Format

### Static Signs

Each `.npy` file contains a 2D NumPy array of shape `(N, 63)`:
- **N** = number of collected samples
- **63** = 21 hand landmarks x 3 coordinates (x, y, z), normalized

Normalization:
- Wrist (landmark 0) is translated to origin
- All coordinates scaled by the maximum distance from wrist to any landmark
- Flattened to a 1D vector

### Dynamic Signs

Each sequence `.npy` file contains a 2D array of shape `(T, 63)`:
- **T** = number of frames in the sequence (variable length)
- Each frame is a normalized 63-dim feature vector

## Data Source

**User-collected via webcam** using `scripts/collect_data.py`.

The signs follow Indian Sign Language (ISL) as documented by:
- Indian Sign Language Research and Training Centre (ISLRTC)
  http://www.islrtc.nic.in
- Indian Sign Language Dictionary Portal
  https://indiansignlanguage.org

## Collection Method

1. Run `python scripts/collect_data.py`
2. Navigate to a sign using arrow keys
3. Perform the ISL sign in front of the webcam
4. Press S for snapshot (static) or R to record (dynamic)
5. Press A for auto-capture (rapidly collect static samples)
6. Collect 50-150+ samples per sign for best results

## Data Augmentation

During collection with auto-capture (A key), the system automatically generates
augmented samples with:
- Small Gaussian noise on coordinates
- Random scale variation (0.9x - 1.1x)
- Random rotation (up to 10 degrees)
- Random translation

This helps improve model robustness without requiring thousands of manual samples.

## License

Data collected by the user is their own. No external dataset is redistributed
in this repository.

## Privacy

Only hand landmark coordinates (21 x 3 floats per frame) are stored.
No images or video frames are saved, making the dataset privacy-friendly.
