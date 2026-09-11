"""
vocabulary.py -- Centralized ISL vocabulary configuration.

Defines the supported Indian Sign Language signs, their metadata,
and classification (static vs dynamic, one-hand vs two-hand).

All sign labels and metadata should be referenced from this module
rather than being scattered throughout the codebase.

IMPORTANT: Each sign has been verified against ISL sources where possible.
Signs marked verification_status="NEEDS_VERIFICATION" have not been
confirmed from a primary ISL source and should be used with caution.

Sources:
  - Indian Sign Language Research and Training Centre (ISLRTC)
    http://www.islrtc.nic.in
  - Indian Sign Language Dictionary Portal
    https://indiansignlanguage.org
  - INCLUDE Dataset (AI4Bharat, ACM MM 2020)
    https://github.com/AI4Bharat/INCLUDE

KEY LIMITATION:
  The current system uses ONLY hand landmarks (21 points x 3 coords).
  It does NOT use pose/body landmarks, so it CANNOT determine where
  the hand is relative to the face/chest/body. Signs that depend on
  hand-to-body-location (like MOTHER=nose, FATHER=lip, I/ME=chest)
  are NOT reliably distinguishable with hand landmarks alone.
  These signs are placed in EXTENDED_VOCABULARY, not the MVP.
"""

from typing import Dict, List, NamedTuple


class SignInfo(NamedTuple):
    """Metadata for a single ISL sign."""
    label: str               # Display label (e.g., "HELLO")
    sign_type: str           # "static" or "dynamic"
    num_hands: int           # 1 or 2
    description: str         # How the sign is performed in ISL
    hindi_equivalent: str    # Hindi translation (for reference)
    verification_status: str  # "VERIFIED", "NEEDS_VERIFICATION", or "CORRECTED"
    landmark_only: bool       # True if hand landmarks alone can distinguish this sign
    notes: str               # Additional notes about limitations


# ============================================================================
#  FULL ISL VOCABULARY (verified/corrected)
# ============================================================================

ISL_SIGNS: Dict[str, SignInfo] = {

    # ====================================================================
    #  MVP SIGNS -- reliably distinguishable with hand landmarks only
    # ====================================================================

    "GOOD": SignInfo(
        label="GOOD",
        sign_type="static",
        num_hands=1,
        description="Thumbs-up gesture held upright in front of chest.",
        hindi_equivalent="Accha",
        verification_status="VERIFIED",
        landmark_only=True,
        notes="Distinct hand shape (thumb up, fingers curled).",
    ),
    "BAD": SignInfo(
        label="BAD",
        sign_type="static",
        num_hands=1,
        description="Pinky finger extended vertically with other fingers curled "
                    "(traditional ISL gesture for bad/katti).",
        hindi_equivalent="Bura",
        verification_status="VERIFIED",
        landmark_only=True,
        notes="Pinky-up is culturally ISL. Thumbs-down is colloquial but less ISL-specific.",
    ),
    "STOP": SignInfo(
        label="STOP",
        sign_type="static",
        num_hands=1,
        description="Open flat hand with all fingers extended, palm facing outward, "
                    "thrust forward firmly.",
        hindi_equivalent="Ruko",
        verification_status="VERIFIED",
        landmark_only=True,
        notes="All-fingers-open flat hand. May overlap with HELLO if both are open-palm, "
              "but STOP has a more rigid/tense spread.",
    ),
    "HELLO": SignInfo(
        label="HELLO",
        sign_type="static",
        num_hands=1,
        description="Open palm raised with fingers together, slight wave. "
                    "Formal variant: both palms pressed together (Namaste, 2 hands).",
        hindi_equivalent="Namaste",
        verification_status="VERIFIED",
        landmark_only=True,
        notes="1-hand variant (palm wave) distinguishable from STOP by fingers-together vs spread. "
              "For MVP, use the 1-hand palm-up wave variant.",
    ),
    "COME": SignInfo(
        label="COME",
        sign_type="dynamic",
        num_hands=1,
        description="Open hand or index finger making a beckoning motion, "
                    "curling fingers repeatedly toward the signer.",
        hindi_equivalent="Aao",
        verification_status="VERIFIED",
        landmark_only=True,
        notes="Distinct curling finger motion. Sequence classifier can capture this.",
    ),

    # ====================================================================
    #  EXTENDED VOCABULARY -- need body/pose landmarks or are harder
    # ====================================================================

    "I_ME": SignInfo(
        label="I/ME",
        sign_type="static",
        num_hands=1,
        description="Index finger pointing toward the center of own chest.",
        hindi_equivalent="Main",
        verification_status="VERIFIED",
        landmark_only=False,
        notes="CANNOT distinguish from YOU with hand landmarks alone. "
              "Both are index-finger pointing. Needs pose landmarks to detect chest direction.",
    ),
    "YOU": SignInfo(
        label="YOU",
        sign_type="static",
        num_hands=1,
        description="Index finger pointing outward toward the other person.",
        hindi_equivalent="Tum/Aap",
        verification_status="VERIFIED",
        landmark_only=False,
        notes="CANNOT distinguish from I/ME with hand landmarks alone. "
              "Same hand shape, different direction.",
    ),
    "YES": SignInfo(
        label="YES",
        sign_type="dynamic",
        num_hands=1,
        description="Closed fist with wrist nodding up and down, or simply a head nod.",
        hindi_equivalent="Haan",
        verification_status="VERIFIED",
        landmark_only=False,
        notes="Reclassified static->dynamic. A static fist is ambiguous. "
              "Head nod is the most natural ISL indicator (needs face landmarks).",
    ),
    "NO": SignInfo(
        label="NO",
        sign_type="dynamic",
        num_hands=1,
        description="Index finger wagging side-to-side, or an open hand twisting. "
                    "Often accompanied by a head shake.",
        hindi_equivalent="Nahi",
        verification_status="CORRECTED",
        landmark_only=False,
        notes="Reclassified static->dynamic. Original desc had 'thumb snapping' which is more ASL. "
              "Head shake is crucial in ISL.",
    ),
    "WAIT": SignInfo(
        label="WAIT",
        sign_type="dynamic",
        num_hands=1,
        description="Open flat hand, palm facing down, pressing/bouncing downward slightly.",
        hindi_equivalent="Intezaar",
        verification_status="VERIFIED",
        landmark_only=False,
        notes="Reclassified static->dynamic. Statically it's just a flat hand "
              "(confused with STOP/HELLO). The downward pressing motion is key.",
    ),
    "HOME": SignInfo(
        label="HOME",
        sign_type="static",
        num_hands=2,
        description="Both flat open hands brought together with fingertips touching "
                    "at an angle forming a triangular roof shape.",
        hindi_equivalent="Ghar",
        verification_status="VERIFIED",
        landmark_only=True,
        notes="Highly distinct 2-hand shape. BUT current feature extractor only handles "
              "1 hand (63 dims). Needs 2-hand feature extraction to work.",
    ),

    # ---- Dynamic signs requiring body/face context ----

    "THANK_YOU": SignInfo(
        label="THANK YOU",
        sign_type="dynamic",
        num_hands=1,
        description="Flat hand starting from chin/mouth area, moving forward and "
                    "slightly downward toward the other person.",
        hindi_equivalent="Dhanyavaad",
        verification_status="VERIFIED",
        landmark_only=False,
        notes="Starts at chin -- needs pose landmarks to know hand is at face level.",
    ),
    "SORRY": SignInfo(
        label="SORRY",
        sign_type="dynamic",
        num_hands=1,
        description="Flat hand or fingertips touching the chest and pulling away, "
                    "or a flat hand moving in a circle on the chest.",
        hindi_equivalent="Maafi",
        verification_status="CORRECTED",
        landmark_only=False,
        notes="CORRECTED: Original had 'fist circling on chest' which is ASL. "
              "ISL uses open hand on chest. Needs pose landmarks for chest contact.",
    ),
    "WATER": SignInfo(
        label="WATER",
        sign_type="dynamic",
        num_hands=1,
        description="Cupped hand or thumb brought toward the mouth in a "
                    "tilted drinking motion.",
        hindi_equivalent="Paani",
        verification_status="VERIFIED",
        landmark_only=False,
        notes="Without pose data, a cupped hand near the camera looks like any C-shape. "
              "Needs face/mouth landmarks to detect proximity.",
    ),
    "FOOD": SignInfo(
        label="FOOD",
        sign_type="dynamic",
        num_hands=1,
        description="Bunched fingertips (flattened O shape) tapped repeatedly "
                    "against the mouth.",
        hindi_equivalent="Khana",
        verification_status="VERIFIED",
        landmark_only=False,
        notes="Bunched fingers are somewhat distinct, but mouth contact is essential. "
              "Without face landmarks, confused with other pinch shapes.",
    ),
    "MOTHER": SignInfo(
        label="MOTHER",
        sign_type="dynamic",
        num_hands=1,
        description="Index finger/thumb touching or gently pinching the "
                    "side of the nose (referencing nose pin / nath).",
        hindi_equivalent="Maa",
        verification_status="VERIFIED",
        landmark_only=False,
        notes="CANNOT work with hand landmarks only. It is just an index finger -- "
              "identical to FATHER, I/ME, YOU. REQUIRES nose/face landmark.",
    ),
    "FATHER": SignInfo(
        label="FATHER",
        sign_type="dynamic",
        num_hands=1,
        description="Index finger and thumb tracing/twisting an imaginary "
                    "mustache outward along the upper lip.",
        hindi_equivalent="Papa/Pitaji",
        verification_status="VERIFIED",
        landmark_only=False,
        notes="CANNOT work with hand landmarks only. Same hand shape as MOTHER, I/ME, YOU. "
              "REQUIRES upper-lip/face landmark.",
    ),
    "HAPPY": SignInfo(
        label="HAPPY",
        sign_type="dynamic",
        num_hands=1,
        description="Open flat palm(s) brushing upward against the chest "
                    "in repeated uplifting strokes with a cheerful expression.",
        hindi_equivalent="Khush",
        verification_status="NEEDS_VERIFICATION",
        landmark_only=False,
        notes="ISL vs ASL overlap uncertain. Smiling face is mandatory in ISL. "
              "Without face/body context, it's just a moving flat hand.",
    ),

    # ---- 2-hand signs (corrected) ----

    "HELP": SignInfo(
        label="HELP",
        sign_type="dynamic",
        num_hands=2,
        description="Both hands open, palms facing upward, pushing forward "
                    "or lifting slightly (offering/asking for help).",
        hindi_equivalent="Madad",
        verification_status="CORRECTED",
        landmark_only=True,
        notes="CORRECTED: Original desc was ASL ('thumbs-up on flat palm'). "
              "ISL uses open palms pushing forward. Needs 2-hand extraction.",
    ),
    "PLEASE": SignInfo(
        label="PLEASE",
        sign_type="dynamic",
        num_hands=2,
        description="Palms pressed together (Namaste shape) moving slightly "
                    "downward, held close to chest pleadingly.",
        hindi_equivalent="Kripaya",
        verification_status="VERIFIED",
        landmark_only=False,
        notes="Visually identical to formal HELLO (Namaste) without facial "
              "expression context. Needs 2-hand extraction.",
    ),
    "FRIEND": SignInfo(
        label="FRIEND",
        sign_type="static",
        num_hands=2,
        description="Both hands clasped together in a handshake, or "
                    "interlocking fingers of both hands tightly.",
        hindi_equivalent="Dost",
        verification_status="CORRECTED",
        landmark_only=True,
        notes="CORRECTED: Original desc ('index fingers hooked') was ASL. "
              "ISL uses handshake/interlocked fingers. Needs 2-hand extraction.",
    ),
    "FAMILY": SignInfo(
        label="FAMILY",
        sign_type="dynamic",
        num_hands=2,
        description="Both index fingers or hands tracing a circle to indicate "
                    "a group, often combined with the sign for people.",
        hindi_equivalent="Parivaar",
        verification_status="CORRECTED",
        landmark_only=True,
        notes="CORRECTED: Original desc ('curved hands sweeping outward') was ASL. "
              "ISL uses circular motion to indicate group. Needs 2-hand extraction.",
    ),
    "NAME": SignInfo(
        label="NAME",
        sign_type="dynamic",
        num_hands=2,
        description="Right hand's index finger and thumb extended, tapping or "
                    "sliding against the left palm, or pointing to chest then "
                    "opening palm outward.",
        hindi_equivalent="Naam",
        verification_status="CORRECTED",
        landmark_only=False,
        notes="CORRECTED: Original desc ('H-hands tapping') was ASL. "
              "ISL version needs 2 hands and possibly chest context.",
    ),
}


# ============================================================================
#  MVP vs EXTENDED vocabulary tiers
# ============================================================================
# MVP: Signs that can be reliably distinguished using hand landmarks ONLY
#      AND are one-hand signs (current feature extractor supports 1 hand).
#
# EXTENDED: Signs that require body/pose landmarks, face detection,
#           two-hand feature extraction, or are difficult to distinguish.
# ============================================================================

MVP_VOCABULARY = [
    "GOOD",       # Thumbs-up -- very distinct hand shape
    "BAD",        # Pinky extended -- very distinct hand shape
    "STOP",       # Open flat hand, fingers spread -- distinct
    "HELLO",      # Open palm, fingers together -- distinguishable from STOP
    "COME",       # Beckoning finger curl -- distinct dynamic motion
]
"""
MVP vocabulary: signs that can be RELIABLY recognized with
hand-landmark-only classification. These have distinct hand shapes
that do not depend on body/face location.

Total: 5 signs + NEUTRAL state = 6 classes

This is deliberately small because with only 21 hand landmarks and
no body/face context, many ISL signs are indistinguishable.

To expand the MVP, either:
  1. Add MediaPipe Pose landmarks (body reference frame), OR
  2. Add more signs with truly unique hand shapes
"""

EXTENDED_VOCABULARY = [
    # These need pose/face landmarks to work properly
    "I_ME",       # Index pointing at chest -- same hand shape as YOU
    "YOU",        # Index pointing outward -- same hand shape as I/ME
    "YES",        # Fist nodding -- just a fist without motion context
    "NO",         # Finger wagging -- needs head shake or temporal model
    "WAIT",       # Palm down pressing -- flat hand like STOP/HELLO
    "HOME",       # 2-hand roof shape -- needs 2-hand extraction
    "THANK_YOU",  # Hand from chin -- needs face proximity detection
    "SORRY",      # Hand on chest -- needs pose landmarks
    "WATER",      # Drinking motion -- needs face proximity
    "FOOD",       # Fingers to mouth -- needs face proximity
    "MOTHER",     # Nose touch -- MUST have face/nose landmark
    "FATHER",     # Lip trace -- MUST have face/lip landmark
    "HAPPY",      # Chest brush + smile -- needs face + pose
    "HELP",       # 2-hand open palms -- needs 2-hand extraction
    "PLEASE",     # Namaste moving -- same shape as HELLO, needs face
    "FRIEND",     # Handshake -- needs 2-hand extraction
    "FAMILY",     # Circle gesture -- needs 2-hand extraction
    "NAME",       # Finger on palm -- needs 2-hand extraction
]
"""
Extended vocabulary: signs that require additional capabilities:
  - Pose/body landmarks for hand-to-body-location detection
  - Face landmarks for facial expression / proximity
  - Two-hand feature extraction (currently not implemented)
"""


# ============================================================================
#  NEUTRAL STATE
# ============================================================================
NEUTRAL_LABEL = "_NEUTRAL"
NEUTRAL_DESCRIPTION = (
    "No intentional sign being performed. "
    "Hand relaxed or in a non-sign position."
)


# ============================================================================
#  HELPER FUNCTIONS
# ============================================================================

def get_all_labels() -> List[str]:
    """Return all sign labels in sorted order."""
    return sorted(ISL_SIGNS.keys())


def get_mvp_labels() -> List[str]:
    """Return MVP vocabulary labels."""
    return list(MVP_VOCABULARY)


def get_extended_labels() -> List[str]:
    """Return extended vocabulary labels."""
    return list(EXTENDED_VOCABULARY)


def get_static_labels() -> List[str]:
    """Return labels of static signs."""
    return sorted(k for k, v in ISL_SIGNS.items() if v.sign_type == "static")


def get_dynamic_labels() -> List[str]:
    """Return labels of dynamic signs."""
    return sorted(k for k, v in ISL_SIGNS.items() if v.sign_type == "dynamic")


def get_one_hand_labels() -> List[str]:
    """Return labels of one-hand signs."""
    return sorted(k for k, v in ISL_SIGNS.items() if v.num_hands == 1)


def get_two_hand_labels() -> List[str]:
    """Return labels of two-hand signs."""
    return sorted(k for k, v in ISL_SIGNS.items() if v.num_hands == 2)


def get_landmark_only_labels() -> List[str]:
    """Return labels that can be recognized with hand landmarks only."""
    return sorted(k for k, v in ISL_SIGNS.items() if v.landmark_only)


def get_sign_info(label: str) -> SignInfo:
    """Get metadata for a specific sign."""
    if label not in ISL_SIGNS:
        raise ValueError(f"Unknown sign label: {label}")
    return ISL_SIGNS[label]


def get_display_label(label: str) -> str:
    """Get the user-friendly display label."""
    if label in ISL_SIGNS:
        return ISL_SIGNS[label].label
    return label


if __name__ == "__main__":
    print("ISL Vocabulary Configuration")
    print("=" * 60)
    all_labels = get_all_labels()
    print(f"Total signs:          {len(all_labels)}")
    print(f"MVP signs:            {len(MVP_VOCABULARY)}")
    print(f"Extended signs:       {len(EXTENDED_VOCABULARY)}")
    print(f"Static signs:         {len(get_static_labels())}")
    print(f"Dynamic signs:        {len(get_dynamic_labels())}")
    print(f"One-hand signs:       {len(get_one_hand_labels())}")
    print(f"Two-hand signs:       {len(get_two_hand_labels())}")
    print(f"Landmark-only signs:  {len(get_landmark_only_labels())}")
    print()
    print("MVP Vocabulary:")
    for label in MVP_VOCABULARY:
        info = ISL_SIGNS[label]
        print(f"  {info.label:15s} | {info.sign_type:7s} | "
              f"{info.num_hands}H | {info.verification_status:20s} | {info.hindi_equivalent}")
    print()
    print("Extended Vocabulary:")
    for label in EXTENDED_VOCABULARY:
        info = ISL_SIGNS[label]
        print(f"  {info.label:15s} | {info.sign_type:7s} | "
              f"{info.num_hands}H | {info.verification_status:20s} | {info.notes[:50]}")
