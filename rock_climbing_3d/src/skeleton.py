"""COCO-17 skeleton definition and per-frame drawing."""

from __future__ import annotations

import cv2

# The 17 keypoints, in the order the model returns them.
KPT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

FACE_KPTS = frozenset({0, 1, 2, 3, 4})   # nose, eyes, ears

# (a, b, part); part picks the colour. Each arm and leg is its own part, named
# after the limb whose point of contact it ends in (shoulder to wrist is the
# left hand's, hip to ankle the left foot's), so the stick figure can be drawn
# in the same four colours the holds and the limb panel use — see `palette`.
SKELETON = [
    (5, 6, "torso"),
    (5, 7, "left_hand"), (7, 9, "left_hand"),
    (6, 8, "right_hand"), (8, 10, "right_hand"),
    (5, 11, "torso"), (6, 12, "torso"),
    (11, 12, "torso"),
    (11, 13, "left_foot"), (13, 15, "left_foot"),
    (12, 14, "right_foot"), (14, 16, "right_foot"),
    (0, 1, "head"), (0, 2, "head"), (1, 3, "head"), (2, 4, "head"),
    (3, 5, "head"), (4, 6, "head"),
]


LIMB_PARTS = {"left_hand": "arms", "right_hand": "arms",
              "left_foot": "legs", "right_foot": "legs"}


def palette(skeleton_colors: dict, limb_colors: dict, *, by_limb: bool) -> dict:
    """The colour for every part: each limb in its own colour, or arms/legs.

    *by_limb* colours each arm and leg the colour of its hand or foot, which is
    what ties the figure to the holds it is using. Off, the two arms share the
    arms colour and the two legs the legs colour, as before.
    """
    colors = dict(skeleton_colors)
    for limb, group in LIMB_PARTS.items():
        colors[limb] = (limb_colors[limb] if by_limb and limb in limb_colors
                        else skeleton_colors.get(group, (255, 255, 255)))
    return colors


def visible_parts(draw_face: bool):
    """The edges and keypoint indices to draw."""
    if draw_face:
        return SKELETON, frozenset(range(len(KPT_NAMES)))
    edges = [(a, b, p) for a, b, p in SKELETON if a not in FACE_KPTS and b not in FACE_KPTS]
    return edges, frozenset(i for i in range(len(KPT_NAMES)) if i not in FACE_KPTS)


def draw_person(img, kpts, valid, *, width: int, height: int, edges, points,
                colors: dict, thickness: int, radius: int,
                bbox=None, bbox_color=(0, 255, 255), bbox_thickness: int = 2):
    """Draw one body. Coordinates arrive normalized to [0, 1].

    They are normalized against the frame the model was shown, but that scaling
    is uniform, so multiplying by this frame's width and height is exact rather
    than approximate — one inference serves any render size.
    """
    pts = [(int(round(x * width)), int(round(y * height))) for x, y in kpts]

    if bbox is not None:
        bx, by, bw, bh = bbox
        cv2.rectangle(img,
                      (int(bx * width), int(by * height)),
                      (int((bx + bw) * width), int((by + bh) * height)),
                      bbox_color, bbox_thickness, cv2.LINE_AA)

    for a, b, part in edges:
        if valid[a] and valid[b]:
            cv2.line(img, pts[a], pts[b], colors[part], thickness, cv2.LINE_AA)

    for i in points:
        if valid[i]:
            cv2.circle(img, pts[i], radius, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, pts[i], radius + 2, (0, 0, 0), 1, cv2.LINE_AA)

    return img
