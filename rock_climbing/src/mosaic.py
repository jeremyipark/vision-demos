"""The wall, assembled from every frame that saw it.

This is the right panel's background, and the reason the handheld clip can have
a right panel at all. With a tripod the "static view of the wall" was free — any
frame was it. Handheld, no single frame is: each one is a partial, tilted,
differently-zoomed look, and the climber is in front of the interesting part of
every one of them.

So it is built rather than chosen. Every frame is warped onto the canvas by
:mod:`src.camera`'s homography and the stack is reduced per pixel.

## The reduction is a median, and that is the whole trick

Averaging works and is what a first attempt does, but it leaves the climber on
the wall as a faint smear, because a mean is moved by an outlier in proportion
to how far out it is. A median is not moved by an outlier at all until the
outliers are half the stack.

And the climber is nowhere near half the stack. They are somewhere different in
every frame, so any given wall pixel sees them in a handful of frames out of
hundreds and sees bare wall in the rest. Taking the median over the stack
therefore returns the wall — not wall-with-a-ghost, the wall, as if the climb
had never happened. The same argument disposes of the chalk puffs, the passing
gym-goer at the bottom of the frame and the lens flare.

What it cannot dispose of is something that never moves relative to the wall.
That is fine: on this footage the only such things are the wall and the mats.

## Sharpness

Pixels are weighted by how much wall-resolution the contributing frame had
there, so a tight shot of the top-out contributes its detail instead of being
averaged into the wide shots that also cover it. The weight is the local scale
factor of that frame's homography — literally how many source pixels the frame
spent on this piece of canvas — which is available for nothing from the same
matrix that placed it.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import cv2
import numpy as np

from src.camera import Track


def _local_scale(matrix: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Source pixels per canvas pixel, over the canvas — the frame's local detail.

    The Jacobian determinant of the *inverse* homography, evaluated on a coarse
    grid and resized up. Coarse because it is smooth: a homography's scale does
    not do anything abrupt across a frame, so sampling it at 64 across and
    interpolating is indistinguishable from evaluating it everywhere and much
    cheaper.
    """
    width, height = size
    step = max(1, max(width, height) // 64)
    gw, gh = max(2, width // step), max(2, height // step)
    xs = np.linspace(0, width - 1, gw)
    ys = np.linspace(0, height - 1, gh)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).astype(np.float32)

    inverse = np.linalg.inv(matrix)
    delta = 1.0
    base = cv2.perspectiveTransform(grid.reshape(-1, 1, 2), inverse).reshape(gh, gw, 2)
    dx = cv2.perspectiveTransform((grid + [delta, 0]).reshape(-1, 1, 2),
                                  inverse).reshape(gh, gw, 2) - base
    dy = cv2.perspectiveTransform((grid + [0, delta]).reshape(-1, 1, 2),
                                  inverse).reshape(gh, gw, 2) - base
    jacobian = np.abs(dx[..., 0] * dy[..., 1] - dx[..., 1] * dy[..., 0])
    return cv2.resize(jacobian.astype(np.float32), (width, height),
                      interpolation=cv2.INTER_LINEAR)


def build(video: Path, track: Track, *, stride: int = 4, scale: float = 1.0,
          max_samples: int = 220, sharpness_weight: bool = True,
          console=None) -> tuple[np.ndarray, np.ndarray]:
    """Warp the clip onto the canvas and reduce it. ``(wall_bgr, coverage)``.

    ``coverage`` counts how many frames saw each canvas pixel, and is not
    decoration: it is what tells the renderer which part of the canvas is real
    wall and which is the black triangle outside anything the camera ever
    pointed at.

    Held in memory as a stack so the median is a median and not a running
    approximation of one, which is why ``max_samples`` exists. At the default
    canvas scale that stack is a few GB at 220 frames and considerably more at
    600, and 220 evenly-spaced looks at a static wall is already far past the
    point where another one changes a pixel.
    """
    width, height = track.size
    out_w, out_h = max(1, int(round(width * scale))), max(1, int(round(height * scale)))
    resize = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float64)

    indices = list(range(0, track.n_frames, max(1, stride)))
    if len(indices) > max_samples:
        indices = np.linspace(0, track.n_frames - 1, max_samples).astype(int).tolist()

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video}")

    stack: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    coverage = np.zeros((out_h, out_w), dtype=np.int32)
    ones = None

    for count, index in enumerate(indices):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok:
            continue
        matrix = resize @ track.H[index]
        warped = cv2.warpPerspective(frame, matrix, (out_w, out_h),
                                     flags=cv2.INTER_LINEAR)
        if ones is None:
            ones = np.ones(frame.shape[:2], dtype=np.uint8)
        seen = cv2.warpPerspective(ones, matrix, (out_w, out_h),
                                   flags=cv2.INTER_NEAREST).astype(bool)

        # Outside the frame's own footprint the warp is black, which is not a
        # colour this frame is claiming — it is an absence. NaN says so, and
        # nanmedian then ignores it instead of voting black.
        tile = warped.astype(np.float32)
        tile[~seen] = np.nan
        stack.append(tile)
        coverage += seen

        if sharpness_weight:
            local = _local_scale(matrix, (out_w, out_h))
            local[~seen] = 0.0
            weights.append(local)

        if console and count and count % 50 == 0:
            console.print(f"  [dim]blended {count}/{len(indices)}[/]")
    capture.release()

    if not stack:
        raise RuntimeError("no frame could be warped onto the canvas")

    wall = _reduce(stack, weights if sharpness_weight else None)
    return wall, coverage


def _reduce(stack: list[np.ndarray], weights: list[np.ndarray] | None) -> np.ndarray:
    """The per-pixel median of the warped stack, sharpened toward the closest looks.

    Two passes rather than one. The median alone is robust but soft: it picks a
    middle value across frames that resolved the wall at very different scales,
    so a hold shot tight and a hold shot wide contribute equally and the result
    is the blur of the two. The median is therefore used as the *reference*, and
    the output is a weighted mean of only those samples that agree with it —
    weighted by each frame's local resolution.

    So the median decides what is wall and what is climber, and the weighted
    mean decides how sharply the wall is drawn. Outliers are excluded by the
    first step and cannot come back in the second.
    """
    tiles = np.stack(stack)
    # The canvas is the bounding box of a set of rotated quadrilaterals, so its
    # corners are pixels no frame ever covered: an all-NaN stack there is the
    # expected answer, not an anomaly worth a warning on every run.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nan_to_num(np.nanmedian(tiles, axis=0))

    if weights is None:
        return np.clip(median, 0, 255).astype(np.uint8)

    # "Agrees with the median" is generous on purpose: this rejects the climber,
    # not exposure wobble between frames.
    tolerance = 42.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        close = (np.nanmax(np.abs(tiles - median[None]), axis=-1) <= tolerance)
    close &= ~np.isnan(tiles[..., 0])

    weight = np.stack(weights) * close
    total = weight.sum(axis=0)
    blended = (np.nan_to_num(tiles) * weight[..., None]).sum(axis=0)
    out = np.where(total[..., None] > 1e-6,
                   blended / np.maximum(total, 1e-6)[..., None],
                   median)
    return np.clip(out, 0, 255).astype(np.uint8)


def trim(wall: np.ndarray, coverage: np.ndarray, *, min_frames: int = 1
         ) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Crop away canvas no camera ever pointed at. ``(wall, coverage, box)``.

    The canvas is the bounding box of every frame's footprint, and those
    footprints are rotated quadrilaterals, so its corners are guaranteed empty.
    ``box`` is ``(x, y, w, h)`` in canvas pixels, which is what a caller needs to
    keep hold coordinates pointing at the same wall after the crop.
    """
    seen = coverage >= max(1, min_frames)
    if not seen.any():
        return wall, coverage, (0, 0, wall.shape[1], wall.shape[0])
    ys, xs = np.where(seen)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return wall[y0:y1, x0:x1], coverage[y0:y1, x0:x1], (x0, y0, x1 - x0, y1 - y0)


# ── placing the canvas in a panel ────────────────────────────────────────────

class Fit:
    """Canvas-normalized coordinates, placed inside a panel of a given size.

    The canvas is whatever shape the camera's wanderings made it, and the right
    panel has to be the same size as the left one so the two can sit side by
    side. Those two shapes are close on this clip and will not be on the next,
    so the canvas is scaled to fit and centred, and everything drawn on the
    panel goes through here.

    Expressed as a map from canvas-normalized to *panel*-normalized coordinates
    rather than to pixels, so every drawing helper that already takes normalized
    input keeps working untouched.
    """

    def __init__(self, canvas: tuple[int, int], panel: tuple[int, int],
                 region: tuple[float, float, float, float] | None = None):
        cw, ch = canvas
        pw, ph = panel
        self.panel = (pw, ph)
        self.canvas = (cw, ch)
        # *region* is the slice of the canvas the panel shows, normalized. The
        # default is all of it; framing on the route instead is what keeps the
        # panel from being mostly the ceiling and the mats, since the canvas is
        # as large as the camera's wanderings made it and the boulder is only
        # part of that.
        self.region = region or (0.0, 0.0, 1.0, 1.0)
        rw = max(self.region[2] * cw, 1.0)
        rh = max(self.region[3] * ch, 1.0)
        self.scale = min(pw / rw, ph / rh)
        self.box = (max(1, int(round(rw * self.scale))),
                    max(1, int(round(rh * self.scale))))
        self.origin = ((pw - self.box[0]) // 2, (ph - self.box[1]) // 2)

    @classmethod
    def on_holds(cls, canvas, panel, holds, *, margin: float):
        """Frame the panel on the route's own extent, with *margin* of slack."""
        if not holds:
            return cls(canvas, panel)
        boxes = np.array([h["bbox"] for h in holds], dtype=float)
        x0, y0 = boxes[:, 0].min(), boxes[:, 1].min()
        x1 = (boxes[:, 0] + boxes[:, 2]).max()
        y1 = (boxes[:, 1] + boxes[:, 3]).max()
        pad_x, pad_y = (x1 - x0) * margin, (y1 - y0) * margin
        x0, y0 = max(0.0, x0 - pad_x), max(0.0, y0 - pad_y)
        x1, y1 = min(1.0, x1 + pad_x), min(1.0, y1 + pad_y)
        return cls(canvas, panel, (x0, y0, x1 - x0, y1 - y0))

    def point(self, points: np.ndarray) -> np.ndarray:
        """(N, 2) canvas-normalized -> (N, 2) panel-normalized."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if not len(pts):
            return pts
        pw, ph = self.panel
        rx, ry, rw, rh = self.region
        out = (pts - [rx, ry]) / [max(rw, 1e-9), max(rh, 1e-9)] * self.box
        out[:, 0] = (out[:, 0] + self.origin[0]) / pw
        out[:, 1] = (out[:, 1] + self.origin[1]) / ph
        return out

    def hold(self, hold: dict) -> dict:
        """A copy of a canvas hold with its outline and box in panel coordinates."""
        polygon = hold.get("polygon")
        out = dict(hold)
        if polygon is not None and len(polygon) >= 3:
            moved = self.point(np.asarray(polygon, dtype=np.float64))
            out["polygon"] = moved.tolist()
            lo, hi = moved.min(axis=0), moved.max(axis=0)
        else:
            x, y, w, h = hold["bbox"]
            corners = self.point(np.array([[x, y], [x + w, y + h]]))
            lo, hi = corners[0], corners[1]
        out["bbox"] = [float(lo[0]), float(lo[1]),
                       float(hi[0] - lo[0]), float(hi[1] - lo[1])]
        return out

    def backdrop(self, wall: np.ndarray, *, dim: float) -> np.ndarray:
        """The wall image as a panel-sized frame, darkened to sit behind the route.

        Darkened rather than drawn at full strength because the panel's job is
        the route, not the photograph: at full brightness the lit holds have to
        compete with every other hold in the gym, and the eye loses the line.
        """
        pw, ph = self.panel
        out = np.zeros((ph, pw, 3), dtype=np.uint8)
        height, width = wall.shape[:2]
        rx, ry, rw, rh = self.region
        crop = wall[int(ry * height):max(int((ry + rh) * height), int(ry * height) + 1),
                    int(rx * width):max(int((rx + rw) * width), int(rx * width) + 1)]
        resized = cv2.resize(crop, self.box, interpolation=cv2.INTER_AREA)
        x, y = self.origin
        out[y:y + self.box[1], x:x + self.box[0]] = (
            resized.astype(np.float32) * max(0.0, 1.0 - dim)).astype(np.uint8)
        return out
