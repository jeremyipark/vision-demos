"""Where the camera was pointing, frame by frame.

Off a tripod this module has nothing to do: the wall sits at the same pixels
all clip, every homography below comes out as the identity, and the canvas is
the frame. It exists for the camera somebody is holding, where that assumption
is the first thing to break — a hold is at a different place in every frame, so
"hold 7" stops being a location and has to become a *track*.

What replaces the tripod is a **homography per frame**, mapping that frame onto
one canonical view of the wall — the canvas. Everything downstream then lives in
canvas coordinates, where the wall is nailed down again, and the render warps
back out to the moving frame only at the last step.

## Why a homography is the right model here, and not a compromise

A homography is exact for two cases: a planar scene, or a camera that only
rotates. A bouldering prow is emphatically not planar — the mosaic shows at
least four facets plus the mats and the ceiling trusses — so the model rests
entirely on the second case.

That is worth testing rather than hoping for, because it is falsifiable: if the
camera translated at all, near and far structure would disagree under any single
homography, and by a lot. Over this clip they do not. `tools/parallax_check.py`
fits one homography between a pair of frames and reports its residual *by depth
band*; on the two frames furthest apart in time here — 0 and 1290, the whole 43
seconds — it gives, in pixels:

    ceiling / far background   0.74      the trusses, ~10 m behind the wall
    the prow / upper wall      0.54      the overhang, metres in front of the face
    the main face              0.48
    the mats / floor           2.10      the floor, a different plane again

Four surfaces at wildly different depths, one homography, and the worst of them
two pixels out. Only a camera whose optical centre barely moves can do that. So
the operator panned, tilted and zoomed while standing still, and for that motion
the homography is not an approximation of the geometry — it *is* the geometry.

## Which is also why there is no structure-from-motion in here

The same measurement that licenses the homography rules out 3D. Triangulation
needs a baseline, and a baseline is exactly what a residual like that says is
absent. COLMAP on this clip would not be slow or badly tuned, it would be
degenerate: no parallax, nothing to triangulate, and a planar-degenerate
initialisation on top. The honest recovery from this footage is a rotational
panorama, and a rotational panorama is what this builds.

If a future clip is shot *walking past* the boulder, run the same check first. A
ceiling band that blows up to tens of pixels while the face stays sharp is the
signature of real parallax, and the point at which lifting the holds into 3D
starts to mean something — and becomes possible, which it is not here.

## Getting there without drift

Frame-to-frame homographies chained across 1300 frames would accumulate error
into nonsense. They are not chained. Every frame is matched **directly** to one
reference frame, so each estimate is independent and error cannot compound: the
worst frame in this clip still lands 199 RANSAC inliers on the reference, and
the last frame is as accurate as the second. Chaining exists only as the
fallback for a frame too blurred to match the reference on its own.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

# Bumped when the stored shape changes, so an old cache misses rather than
# loading into a reader that no longer understands it.
CONTRACT = "camera-homography-v1"


# ── the canvas ───────────────────────────────────────────────────────────────

@dataclass
class Track:
    """The camera's path, as one homography per frame onto a shared canvas.

    ``H[i]`` maps frame *i*'s pixels to canvas pixels. ``size`` is the canvas in
    pixels and ``reference`` the frame whose view it was built around — the
    canvas is that frame's projective view of the world, scaled and shifted
    until every other frame fits inside it.

    ``inliers[i]`` is how many correspondences survived RANSAC for that frame,
    and ``source[i]`` how the estimate was come by: ``direct`` against the
    reference, ``chained`` through a neighbour, ``filled`` by interpolation, or
    ``still`` when the probes showed the camera never moved and no frame was
    matched at all. ``motion`` is the most any probe frame's corners moved
    against any other, in pixels — the number that decided ``still``.
    They are kept because a bad frame should be visible as a bad frame rather
    than as a mysteriously smeared hold thirty lines downstream.
    """

    H: np.ndarray                 # (n_frames, 3, 3), frame pixels -> canvas pixels
    size: tuple[int, int]         # (width, height) of the canvas, in pixels
    frame_size: tuple[int, int]   # (width, height) of a source frame
    reference: int
    inliers: np.ndarray           # (n_frames,) int
    source: list[str]
    motion: float | None = None   # px, across the probes; None if not measured

    @property
    def n_frames(self) -> int:
        return len(self.H)

    def to_canvas(self, points: np.ndarray, frame: int) -> np.ndarray:
        """Normalized frame coords -> normalized canvas coords, for one frame.

        Both ends are normalized because that is the currency of every other
        module here: a hold's polygon is fractions of the frame, and on the
        canvas it becomes fractions of the canvas.
        """
        return self._map(points, self.H[frame], self.frame_size, self.size)

    def to_frame(self, points: np.ndarray, frame: int) -> np.ndarray:
        """Normalized canvas coords -> normalized frame coords, for one frame.

        The inverse of :meth:`to_canvas`, and the step the renderer makes on
        every drawn hold: the route is stored once on the canvas and projected
        back into whatever the camera was doing at that instant.
        """
        return self._map(points, np.linalg.inv(self.H[frame]), self.size, self.frame_size)

    @staticmethod
    def _map(points: np.ndarray, matrix: np.ndarray, src: tuple[int, int],
             dst: tuple[int, int]) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2).copy()
        if not len(pts):
            return np.zeros((0, 2))
        pts[:, 0, 0] *= src[0]
        pts[:, 0, 1] *= src[1]
        out = cv2.perspectiveTransform(pts, matrix).reshape(-1, 2)
        out[:, 0] /= dst[0]
        out[:, 1] /= dst[1]
        return out

    def crop(self, box: tuple[int, int, int, int]) -> "Track":
        """The same track against a cropped canvas. ``box`` is ``(x, y, w, h)`` px.

        The mosaic gets trimmed to the part of the canvas some camera actually
        pointed at, and the moment it does, canvas-normalized coordinates stop
        meaning the same thing in the track and in the image. Rather than carry
        an offset around for every consumer to remember, the crop is folded back
        into the homographies — after this the track and the trimmed wall agree
        again, and nothing downstream has to know a crop happened.
        """
        x, y, w, h = box
        shift = np.array([[1.0, 0.0, -x], [0.0, 1.0, -y], [0.0, 0.0, 1.0]])
        return Track(H=np.stack([shift @ m for m in self.H]), size=(int(w), int(h)),
                     frame_size=self.frame_size, reference=self.reference,
                     inliers=self.inliers, source=self.source, motion=self.motion)

    def coverage(self, frame: int) -> np.ndarray:
        """This frame's four corners in normalized canvas coords.

        What part of the wall the camera could see at that moment — the render
        draws it on the canvas so the panel says where the live shot is looking.
        """
        corners = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
        return self.to_canvas(corners, frame)


# ── feature matching ─────────────────────────────────────────────────────────

def _detector(n_features: int):
    """SIFT, because the failure mode matters more than the speed.

    ORB is the faster answer and the wrong one for a wall: it keys on corners,
    and a gym wall is a large low-texture field of matte paint whose corners are
    mostly the holds themselves — the things that get occluded by the climber,
    and the things whose apparent shape changes as the camera swings past a
    prow. SIFT's blob response also fires on the wall's speckle and the ceiling
    ironwork, which is the structure that stays put.
    """
    return cv2.SIFT_create(nfeatures=n_features)


def _match(desc_a, desc_b, ratio: float) -> list:
    """Lowe-ratio nearest-neighbour matches between two descriptor sets."""
    if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
        return []
    matcher = cv2.BFMatcher()
    pairs = matcher.knnMatch(desc_a, desc_b, k=2)
    return [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < ratio * n.distance]


def _homography(pts_a, pts_b, matches, *, threshold: float):
    """RANSAC homography from *a* to *b*. ``(H, n_inliers)``; ``(None, 0)`` if it fails.

    MAGSAC++ rather than plain RANSAC: it marginalises over the inlier threshold
    instead of taking one on faith, which is what keeps the estimate from
    tightening onto the climber's own rigid torso when they fill a third of the
    frame.
    """
    if len(matches) < 20:
        return None, 0
    src = pts_a[[m.queryIdx for m in matches]]
    dst = pts_b[[m.trainIdx for m in matches]]
    matrix, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, threshold,
                                      maxIters=5000, confidence=0.9999)
    if matrix is None:
        return None, 0
    return matrix, int(mask.sum()) if mask is not None else 0


def _features(frame: np.ndarray, detector, mask: np.ndarray | None = None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = detector.detectAndCompute(gray, mask)
    points = np.float32([k.pt for k in keypoints]) if keypoints else np.zeros((0, 2), np.float32)
    return points, descriptors


# ── one wall against another ─────────────────────────────────────────────────

def register(wall: np.ndarray, reference_wall: np.ndarray, *, n_features: int = 6000,
             ratio: float = 0.75, threshold: float = 3.0, min_inliers: int = 60,
             max_side: int = 2400) -> tuple[np.ndarray | None, int]:
    """Homography from one clip's wall mosaic to another's. ``(H, n_inliers)``.

    Each clip builds its canvas around its own reference frame, so hold (0.4,
    0.6) on one canvas and (0.4, 0.6) on another are two unrelated places on the
    wall. This is the one extra step that puts them in the same place: the same
    matcher the track uses, run once between the two mosaics. The mosaics are
    the right images to match — the climber has been medianed out of both, so
    nothing on them moves.

    Matched at no more than *max_side* on the long edge, then scaled back, since
    a canvas can be several frames across and SIFT's cost is in the pixels.
    ``(None, n)`` when fewer than *min_inliers* agree: two takes shot from
    opposite ends of the gym share no wall, and a guess would align holds that
    have nothing to do with each other.
    """
    def shrink(image):
        scale = min(1.0, max_side / max(image.shape[:2]))
        if scale < 1.0:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return image, scale

    small_a, scale_a = shrink(wall)
    small_b, scale_b = shrink(reference_wall)
    detector = _detector(n_features)
    pts_a, desc_a = _features(small_a, detector)
    pts_b, desc_b = _features(small_b, detector)
    matrix, inliers = _homography(pts_a, pts_b, _match(desc_a, desc_b, ratio),
                                  threshold=threshold)
    if matrix is None or inliers < min_inliers:
        return None, inliers
    # Undo the shrink on both ends: full-res a -> small a -> small b -> full-res b.
    return np.diag([1 / scale_b, 1 / scale_b, 1.0]) @ matrix @ np.diag([scale_a, scale_a, 1.0]), inliers


# ── picking the reference ────────────────────────────────────────────────────

def choose_reference(video: Path, *, probes: int, n_features: int, ratio: float,
                     threshold: float) -> tuple[int, dict, float]:
    """The frame every other frame will be matched against.

    Not an arbitrary pick, and not the middle one either. The reference defines
    the canvas's whole projective frame, so a badly chosen one — the most
    zoomed-in shot, or the most oblique — either crops the wall or stretches a
    corner of the canvas into a smear. What is wanted is the most *central*
    view: the frame that sees the most of what every other frame sees.

    That is measurable. Match a spread of candidates against each other and keep
    the one with the best median inlier count; connectivity to the rest of the
    clip is exactly the property the reference needs.

    The same matches answer a second question for free: did the camera move at
    all? The third return value is the furthest any frame corner travels under
    any pair's homography, in pixels. On a water bottle that measured 1-4px
    here, most of it fitting noise magnified at the corners; a phone somebody is
    holding drifted 125px. A pair that cannot be matched makes it infinite, since a camera that
    cannot be shown to be still has to be tracked.
    """
    capture = cv2.VideoCapture(str(video))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    picks = np.linspace(0, max(total - 1, 0), num=min(probes, max(total, 1))).astype(int)

    detector = _detector(n_features)
    feats = {}
    for index in picks:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if ok:
            feats[int(index)] = _features(frame, detector)
    capture.release()

    scores: dict[int, float] = {}
    motion = 0.0 if len(feats) >= 2 else math.inf
    keys = sorted(feats)
    for a in keys:
        counts = []
        for b in keys:
            if a == b:
                continue
            pa, da = feats[a]
            pb, db = feats[b]
            matrix, inliers = _homography(pa, pb, _match(da, db, ratio), threshold=threshold)
            counts.append(inliers)
            motion = max(motion, _corner_shift(matrix, size))
        scores[a] = float(np.median(counts)) if counts else 0.0

    best = max(scores, key=scores.get) if scores else 0
    return best, {int(k): v for k, v in scores.items()}, motion


def _corner_shift(matrix: np.ndarray | None, size: tuple[int, int]) -> float:
    """How far, in pixels, *matrix* moves the furthest-travelling frame corner."""
    if matrix is None:
        return math.inf
    width, height = size
    corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(-1, 1, 2)
    moved = cv2.perspectiveTransform(corners, matrix)
    return float(np.linalg.norm((moved - corners).reshape(-1, 2), axis=1).max())


# ── the track ────────────────────────────────────────────────────────────────

def _corner_smooth(matrices: np.ndarray, frame_size: tuple[int, int],
                   sigma: float) -> np.ndarray:
    """Temporally smooth a sequence of homographies, via their image corners.

    Averaging the nine entries of a homography is meaningless — they are a
    projective equivalence class, not nine independent numbers, and the bottom
    row is a different kind of quantity from the top two. The corners are not:
    four points in canvas pixels, eight numbers that move smoothly because the
    camera does. Smooth those, re-fit the homography that sends the frame's
    corners there, and the result is a valid homography throughout.

    What this removes is the last few tenths of a pixel of independent per-frame
    estimation noise, which the eye reads as the route shimmering against a wall
    that is plainly not moving.
    """
    if sigma <= 0:
        return matrices
    width, height = frame_size
    src = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    warped = np.stack([cv2.perspectiveTransform(src.reshape(-1, 1, 2), m).reshape(4, 2)
                       for m in matrices])
    smoothed = gaussian_filter1d(warped, sigma=sigma, axis=0, mode="nearest")
    out = np.empty_like(matrices)
    for i, quad in enumerate(smoothed):
        out[i] = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    return out


def track(video: Path, *, reference: int | None = None, probes: int = 12,
          n_features: int = 3000, ratio: float = 0.75, threshold: float = 3.0,
          min_inliers: int = 60, canvas_scale: float = 1.0, canvas_limit: float = 3.0,
          smooth_sigma: float = 1.5, still_px: float | None = 5.0,
          console=None) -> Track:
    """Solve the camera's path over the whole clip.

    One pass over the video. Each frame's features are matched straight to the
    reference's, which keeps every estimate independent and therefore drift-free
    — the cost of a frame is the same whether it is the 2nd or the 1298th.

    A frame that cannot make ``min_inliers`` against the reference (motion blur,
    or the climber swinging across most of the wall) falls back to matching the
    previous frame and composing that with the previous frame's answer. Only if
    *that* also fails is the frame left to be interpolated between its
    neighbours, which for a camera moving this smoothly is a better estimate
    than a homography fitted to forty noisy correspondences.

    Before any of that, the probes used to pick the reference are checked for
    motion. If no probe frame's corners move more than *still_px* against any
    other, the camera was on a tripod or a water bottle, every homography would
    come out as the identity anyway, and the per-frame pass is skipped: the
    track is the identity for every frame and the canvas is the frame. That is
    the same result the full pass would reach, in a couple of seconds instead of
    most of a minute. ``still_px=None`` always runs the full pass.
    """
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Run even when the reference is pinned: the probes are also the stillness
    # test, and a couple of seconds is cheap next to the pass it may save.
    capture.release()
    picked, _, motion = choose_reference(video, probes=probes, n_features=n_features,
                                         ratio=ratio, threshold=threshold)
    if reference is None:
        reference = picked

    if still_px is not None and motion <= still_px:
        transform, size = _extent(np.eye(3)[None], (width, height), canvas_scale, 0.0)
        return Track(H=np.repeat(transform[None], total, axis=0), size=size,
                     frame_size=(width, height), reference=int(reference),
                     inliers=np.zeros(total, dtype=int), source=["still"] * total,
                     motion=motion)
    capture = cv2.VideoCapture(str(video))

    detector = _detector(n_features)
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(reference))
    ok, ref_frame = capture.read()
    if not ok:
        raise RuntimeError(f"could not read the reference frame {reference} of {video}")
    ref_points, ref_desc = _features(ref_frame, detector)

    to_ref: list[np.ndarray | None] = [None] * total
    inliers = np.zeros(total, dtype=int)
    source = ["failed"] * total

    previous: tuple[np.ndarray, np.ndarray, int] | None = None   # points, desc, index
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for index in range(total):
        ok, frame = capture.read()
        if not ok:
            break
        points, desc = _features(frame, detector)

        matrix, count = _homography(points, ref_points, _match(desc, ref_desc, ratio),
                                    threshold=threshold)
        how = "direct"

        # Blurred, or the climber is across the wall: borrow the neighbour's
        # answer, which is a far shorter baseline and so a far easier match.
        if (matrix is None or count < min_inliers) and previous is not None:
            prev_points, prev_desc, prev_index = previous
            step, step_count = _homography(points, prev_points,
                                           _match(desc, prev_desc, ratio),
                                           threshold=threshold)
            anchor = to_ref[prev_index]
            if step is not None and anchor is not None and step_count > count:
                matrix, count, how = anchor @ step, step_count, "chained"

        if matrix is not None and count >= min(min_inliers, 20):
            to_ref[index] = matrix
            inliers[index] = count
            source[index] = how
        previous = (points, desc, index)

        if console and index and index % 200 == 0:
            console.print(f"  [dim]frame {index}/{total}[/]")
    capture.release()

    solved = [i for i, m in enumerate(to_ref) if m is not None]
    if not solved:
        raise RuntimeError(f"no frame of {video} could be matched to the reference")
    _interpolate(to_ref, solved, source, (width, height))

    matrices = np.stack(to_ref)
    matrices = _corner_smooth(matrices, (width, height), smooth_sigma)

    transform, size = _extent(matrices, (width, height), canvas_scale, canvas_limit)
    return Track(H=np.stack([transform @ m for m in matrices]), size=size,
                 frame_size=(width, height), reference=int(reference),
                 inliers=inliers, source=source, motion=motion)


def _interpolate(matrices: list, solved: list[int], source: list[str],
                 frame_size: tuple[int, int]) -> None:
    """Fill unsolved frames by interpolating their neighbours' corners, in place.

    Interpolated in corner space for the same reason the smoothing is: it is the
    parameterisation in which "halfway between these two camera poses" means
    something. Frames before the first solution and after the last are held at
    the nearest one rather than extrapolated — a guess that runs off the end of
    the evidence is worse than a stale but valid pose.
    """
    width, height = frame_size
    src = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    corners = {i: cv2.perspectiveTransform(src.reshape(-1, 1, 2), matrices[i]).reshape(4, 2)
               for i in solved}
    first, last = solved[0], solved[-1]

    for index in range(len(matrices)):
        if matrices[index] is not None:
            continue
        if index < first:
            matrices[index] = matrices[first].copy()
        elif index > last:
            matrices[index] = matrices[last].copy()
        else:
            before = max(s for s in solved if s < index)
            after = min(s for s in solved if s > index)
            t = (index - before) / (after - before)
            quad = (1 - t) * corners[before] + t * corners[after]
            matrices[index] = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
        source[index] = "filled"


def _extent(matrices: np.ndarray, frame_size: tuple[int, int], scale: float,
            limit: float) -> tuple[np.ndarray, tuple[int, int]]:
    """The canvas box that holds every frame, and the transform into it.

    Bounded by ``limit`` frame-widths around the reference. Without a bound one
    wild frame — a whip-pan that lands on the ceiling, a fallback pose that came
    out slightly wrong — sets the canvas size for the whole clip, and everything
    real ends up in a postage stamp in the middle of it.
    """
    width, height = frame_size
    corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(-1, 1, 2)
    mapped = np.vstack([cv2.perspectiveTransform(corners, m).reshape(-1, 2) for m in matrices])

    x0 = max(float(mapped[:, 0].min()), -limit * width)
    y0 = max(float(mapped[:, 1].min()), -limit * height)
    x1 = min(float(mapped[:, 0].max()), (1 + limit) * width)
    y1 = min(float(mapped[:, 1].max()), (1 + limit) * height)

    transform = np.array([[scale, 0, -x0 * scale],
                          [0, scale, -y0 * scale],
                          [0, 0, 1]], dtype=np.float64)
    return transform, (max(1, int(round((x1 - x0) * scale))),
                       max(1, int(round((y1 - y0) * scale))))


# ── cache ────────────────────────────────────────────────────────────────────

def cache_path(video: Path, cache_dir: Path, *, settings: dict) -> Path:
    stat = video.stat()
    key = json.dumps({"video": video.name, "size": stat.st_size,
                      "mtime_ns": stat.st_mtime_ns, "settings": settings,
                      "contract": CONTRACT}, sort_keys=True)
    return cache_dir / f"camera.{hashlib.sha256(key.encode()).hexdigest()[:12]}.npz"


def save_cache(path: Path, track_: Track, *, stamp: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, H=track_.H, size=np.array(track_.size), frame_size=np.array(track_.frame_size),
        reference=track_.reference, inliers=track_.inliers,
        source=np.array(track_.source),
        motion=np.nan if track_.motion is None else track_.motion, created=stamp)


def load_cache(path: Path) -> tuple[Track, str] | None:
    if not path.is_file():
        return None
    try:
        blob = np.load(path, allow_pickle=False)
        return Track(
            H=blob["H"], size=tuple(int(v) for v in blob["size"]),
            frame_size=tuple(int(v) for v in blob["frame_size"]),
            reference=int(blob["reference"]), inliers=blob["inliers"],
            source=[str(s) for s in blob["source"]],
            motion=(float(blob["motion"]) if "motion" in blob.files
                    and np.isfinite(blob["motion"]) else None),
        ), str(blob["created"])
    except (OSError, KeyError, ValueError):
        return None
