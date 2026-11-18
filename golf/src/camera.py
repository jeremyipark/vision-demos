"""The phone's geometry, from the clip itself: how it drifts, and how far it's tilted down.

* Drift: a handheld or propped phone creeps a few pixels a frame. Measured on the far
  scene (sky, poles, the range), sub-pixel, against the impact frame, so the ball's
  sightings can be put in one frame's coordinates.
* Tilt: vertical things (a range's net poles) converge toward a point below the frame
  when the phone looks down. With the focal length, that point gives the tilt.
"""

from __future__ import annotations

import cv2
import numpy as np


def drift(frames: dict[int, np.ndarray], ref: int, *, region=(0.5, 0.08, 1.0, 0.37)) -> dict[int, np.ndarray]:
    """``{frame: (dx, dy)}``: how far the far scene moved since frame *ref*, px.

    *region* is (x0, y0, x1, y1) as fractions of the frame: far scene, away from the golfer.
    """
    h, w = frames[ref].shape[:2]
    x0, y0, x1, y1 = int(region[0] * w), int(region[1] * h), int(region[2] * w), int(region[3] * h)

    def gray(img):
        return cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)

    base = gray(frames[ref])
    win = cv2.createHanningWindow(base.shape[::-1], cv2.CV_32F)
    return {f: np.array(cv2.phaseCorrelate(base, gray(img), win)[0]) for f, img in frames.items()}


def vertical_vanishing_y(frame: np.ndarray, *, top_fraction: float = 0.3,
                         max_lean_deg: float = 8.0) -> float | None:
    """Where near-vertical lines in the top of the frame (poles) meet, px; None if too few."""
    h = frame.shape[0]
    gray = cv2.cvtColor(frame[: int(top_fraction * h)], cv2.COLOR_BGR2GRAY)
    found = cv2.HoughLinesP(cv2.Canny(gray, 30, 90), 1, np.pi / 1440, 50,
                            minLineLength=int(0.08 * h), maxLineGap=20)
    if found is None:
        return None
    rows, rhs = [], []
    for x1, y1, x2, y2 in found.reshape(-1, 4):
        lean = np.degrees(np.arctan2(x2 - x1, y2 - y1))
        if min(abs(lean), abs(180 - abs(lean))) > max_lean_deg:
            continue
        a, b = float(y2 - y1), float(x1 - x2)
        norm = np.hypot(a, b)
        rows.append([a / norm, b / norm])
        rhs.append((a * x1 + b * y1) / norm)
    if len(rows) < 6:
        return None
    A, B = np.array(rows), np.array(rhs)
    vp = np.linalg.lstsq(A, B, rcond=None)[0]
    for _ in range(4):   # drop lines that aren't poles (nets, wires)
        r = np.abs(A @ vp - B)
        keep = r < max(2.0, 2.5 * np.median(r))
        if keep.sum() < 6:
            break
        vp = np.linalg.lstsq(A[keep], B[keep], rcond=None)[0]
    return float(vp[1])


def pitch_from_vanishing(vp_y: float, *, cy: float, focal: float) -> float | None:
    """Degrees the phone looks down, from where vertical lines meet below the frame."""
    below = vp_y - cy
    if below <= focal:   # too close to be a downward tilt worth trusting
        return None
    return float(np.degrees(np.arctan2(focal, below)))
