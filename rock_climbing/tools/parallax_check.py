"""Does this clip have parallax? The measurement the whole project rests on.

    python tools/parallax_check.py data/cache/<clip>.mp4

Everything downstream of :mod:`src.camera` assumes one homography can map any
frame of the clip onto any other. That assumption is exactly true for a camera
that only rotates and false for one that translates, and the difference is not a
matter of taste — it decides whether a rotational panorama is the right recovery
or whether the clip has recoverable 3D structure in it.

The test is simple and hard to argue with. Fit one homography between two
frames, then measure how well it reprojects the matches **by depth band**. A
planar map can satisfy surfaces at different depths simultaneously only if the
optical centre did not move; if it did, near structure and far structure
disagree, and they disagree more the further apart in depth they are.

So the output is a table of residual against depth. Bands are approximated by
image height, which on a clip shot up at a wall is a reasonable proxy: ceiling
at the top, the overhang below it, the face below that, the mats at the bottom.

**Reading it.** Every band within a pixel or two of the others means rotation,
which means a homography is exact and there is no baseline to triangulate from —
reaching for COLMAP would be reaching for a degenerate reconstruction. A ceiling
band at tens of pixels while the face stays sharp means real translation, and at
that point lifting the holds into 3D starts to mean something and this project
needs a different backbone.

On IMG_7216 every band comes in under a pixel across most pairs and peaks at
2.4px on the widest baseline in time, which is the first answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

BANDS = [
    ("ceiling / far background", 0.00, 0.25),
    ("the prow / upper wall", 0.25, 0.45),
    ("the main face", 0.45, 0.70),
    ("the lower face", 0.70, 0.85),
    ("the mats / floor", 0.85, 1.00),
]


def frame(capture, index: int):
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, image = capture.read()
    if not ok:
        raise RuntimeError(f"could not read frame {index}")
    return image


def residuals(a, b, *, detector, matcher, ratio=0.72, threshold=3.0):
    """Per-match reprojection error of the best single homography from *a* to *b*."""
    ka, da = detector.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), None)
    kb, db = detector.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None)
    pairs = matcher.knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < ratio * n.distance]
    if len(good) < 30:
        return None, None

    src = np.float32([ka[m.queryIdx].pt for m in good])
    dst = np.float32([kb[m.trainIdx].pt for m in good])
    matrix, _ = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, threshold,
                                   maxIters=8000, confidence=0.9999)
    if matrix is None:
        return None, None
    projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    return src, np.linalg.norm(projected - dst, axis=1)


def main(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        print(f"could not open {path}")
        return 1
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"{path.name}: {total} frames\n")

    detector = cv2.SIFT_create(nfeatures=6000)
    matcher = cv2.BFMatcher()

    # One short baseline, one medium, and the two furthest apart in time — the
    # last is where translation, if there is any, has had the longest to build.
    pairs = [(0, min(60, total - 1)),
             (total // 4, total // 2),
             (total // 2, total - 8),
             (0, total - 8)]

    worst = 0.0
    for first, second in pairs:
        points, error = residuals(frame(capture, first), frame(capture, second),
                                  detector=detector, matcher=matcher)
        print(f"── frame {first} -> {second} "
              f"({'' if points is None else len(points)} matches) " + "─" * 20)
        if points is None:
            print("   too few matches\n")
            continue
        for name, low, high in BANDS:
            in_band = (points[:, 1] >= low * height) & (points[:, 1] < high * height)
            if in_band.sum() < 10:
                print(f"   {name:26} n={int(in_band.sum()):4}   —")
                continue
            band = error[in_band]
            worst = max(worst, float(np.median(band)))
            print(f"   {name:26} n={int(in_band.sum()):4}   "
                  f"median {np.median(band):5.2f}px   p90 {np.percentile(band, 90):6.2f}px")
        print()
    capture.release()

    print("─" * 62)
    if worst < 4.0:
        print(f"Worst band median: {worst:.2f}px — every depth satisfied by one\n"
              "planar map, so the camera rotated and did not translate. A\n"
              "homography is exact here, and there is no baseline to triangulate\n"
              "from: structure-from-motion would be degenerate, not merely slow.")
    else:
        print(f"Worst band median: {worst:.2f}px — the depth bands disagree, which\n"
              "is parallax, which means the camera moved through the scene. The\n"
              "single-homography canvas in src/camera.py is not valid for this\n"
              "clip, and real 3D is both necessary and now possible.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(Path(sys.argv[1])))
