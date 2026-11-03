"""The COCO-17 skeleton ViTPose returns."""

from __future__ import annotations

KPT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

FACE_KPTS = frozenset({0, 1, 2, 3, 4})

SKELETON = [   # (a, b, side)
    (15, 13, "L"), (13, 11, "L"), (16, 14, "R"), (14, 12, "R"),
    (11, 12, "T"), (5, 11, "L"), (6, 12, "R"), (5, 6, "T"),
    (5, 7, "L"), (7, 9, "L"), (6, 8, "R"), (8, 10, "R"),
    (0, 1, "H"), (0, 2, "H"), (1, 3, "H"), (2, 4, "H"), (3, 5, "H"), (4, 6, "H"),
]

_TORSO = frozenset({5, 6, 11, 12})
_PARTS = {
    "left_arm": frozenset({5, 7, 9}), "right_arm": frozenset({6, 8, 10}),
    "left_leg": frozenset({11, 13, 15}), "right_leg": frozenset({12, 14, 16}),
}


def edge_part(a: int, b: int) -> str:
    """Which limb an edge belongs to, for its color: ``"torso"``, ``"left_arm"``, ..."""
    if a in _TORSO and b in _TORSO:
        return "torso"
    return next((name for name, joints in _PARTS.items() if a in joints and b in joints), "torso")
