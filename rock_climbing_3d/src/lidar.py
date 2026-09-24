"""The wall in three dimensions, measured rather than inferred.

Recovering a scene from one ordinary camera means structure from motion: a
baseline to triangulate from, bundle adjustment, and at the end a
reconstruction with no scale. This project does not do that. It requires a
capture with depth — every video frame comes with a metric depth map,
pixel-aligned with it, and the camera's intrinsics — and a clip without one is
the 2D demo's job (``../rock_climbing``). With depth:

* **a feature's 3D position is read, not triangulated.** Depth at the pixel,
  back-projected through K. No baseline needed, so the operator standing still
  and panning is as good as the operator walking.
* **the camera's pose is one PnP per frame** against those points
  (:func:`odometry`). RGB-D odometry is a far better conditioned problem than
  monocular SfM: the 3D side of every correspondence is already known.
* **the holds are lifted by depth** (:func:`lift`). Every depth pixel inside a
  hold's mask is a point on that hold, in meters.
* **everything is in meters.** A hold 1.8 m up the wall is 1.8 m up the wall.

## The capture format

``<capture>/`` holds ``video.mov`` (HEVC, portrait, no rotation tag),
``depth.bin`` (float16 meters, ``frame_count x 320 x 240``, 0 = no reading),
``frames.csv`` (per-frame depth validity) and ``meta.json`` (intrinsics for
both grids). Slab *i* was captured with video frame *i*. See the capture's own
README.txt; :func:`load` reads all of it.

The format is the contract, not the phone. Any depth camera — a RealSense, a
Kinect, another phone app — works if its recording is written in this layout:
metric depth per video frame, and the intrinsics that map one onto the other.

## What depth does not give us

The capture carries no camera pose — the app records AVFoundation depth, not an
ARKit session — so the trajectory still has to be solved, and with it comes
drift. Keyframes are matched against a small local map rather than only the
previous frame, which keeps the drift to centimeters over a clip this length;
:func:`odometry` reports the reprojection error that says so.

There is also no gravity vector. It is recovered instead: the phone was held
upright, which gives a rough "up" (:func:`camera_up`), and the floor — dense and
metric in this cloud — refines that
to the floor's own normal (:func:`ground_plane`). So the 3D panel stands the
wall the right way up, and the wall's angle from vertical is a measurement.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# ── what the rest of the pipeline reads ──────────────────────────────────────

@dataclass
class Reconstruction:
    """Camera poses for the keyframes, and the fused wall they see.

    ``poses[i]`` is ``(R, t)`` taking a world point to keyframe *i*'s camera
    frame, so the camera's centre is ``-R.T @ t``. ``points`` is ``(N, 3)`` in
    the same world, in meters.
    """

    keyframes: list[int]
    poses: dict[int, tuple[np.ndarray, np.ndarray]]
    points: np.ndarray
    point_tracks: list[dict[int, np.ndarray]]   # keyframe -> observed pixel
    K: np.ndarray
    rms_px: float = 0.0

    @property
    def n_points(self) -> int:
        return len(self.points)

    def centre(self, keyframe: int) -> np.ndarray:
        R, t = self.poses[keyframe]
        return (-R.T @ t).ravel()

    def project(self, points3d: np.ndarray, keyframe: int) -> np.ndarray:
        """World points into a keyframe's image, in pixels."""
        R, t = self.poses[keyframe]
        cam = (R @ np.asarray(points3d, dtype=float).reshape(-1, 3).T + t.reshape(3, 1))
        depth = np.where(np.abs(cam[2]) < 1e-9, 1e-9, cam[2])
        pix = (self.K @ (cam / depth))
        return pix[:2].T


@dataclass
class Hold3D:
    """One hold in space: where it is, which way it faces, and its outline."""

    id: int
    centre: np.ndarray            # (3,)
    normal: np.ndarray            # (3,) unit, pointing back toward the cameras
    polygon: np.ndarray           # (N, 3) on the hold's own local plane
    angle_deg: float              # here: how far its sightings spread, in mm
    n_views: int
    source: dict


def _fit_normal(points: np.ndarray, centre: np.ndarray, toward: np.ndarray,
                radius: float) -> np.ndarray:
    """Local surface normal from the cloud around *centre*, facing *toward*.

    The smallest principal direction of the neighbouring cloud is the local
    surface normal. The sign is resolved by pointing it back at the camera,
    since a hold faces the room rather than the rock.
    """
    if len(points):
        near = points[np.linalg.norm(points - centre, axis=1) <= radius]
        if len(near) >= 8:
            centred = near - near.mean(axis=0)
            normal = np.linalg.svd(centred, full_matrices=False)[2][-1]
            if normal @ (toward - centre) < 0:
                normal = -normal
            return normal / np.linalg.norm(normal)
    fallback = toward - centre
    return fallback / max(np.linalg.norm(fallback), 1e-9)


def merge_lifted(holds: dict[int, Hold3D], *, radius: float
                 ) -> tuple[dict[int, Hold3D], list[tuple[int, int]]]:
    """Fuse lifted holds that landed in the same place. ``(holds, merges)``.

    SAM drops a track and picks the same hold up under a new id, and the two
    only become visibly one object once both are placed in the world. "The same
    place" is a distance in meters. The survivor is the one placed from more
    views, since that is the one whose position the data determines better.
    """
    order = sorted(holds, key=lambda h: -holds[h].n_views)
    kept: dict[int, Hold3D] = {}
    merges: list[tuple[int, int]] = []
    for hold_id in order:
        hold = holds[hold_id]
        match = next((k for k in kept
                      if np.linalg.norm(kept[k].centre - hold.centre) <= radius), None)
        if match is None:
            kept[hold_id] = hold
        else:
            merges.append((hold_id, match))
            kept[match].source.setdefault("merged", []).append(hold_id)
    return {k: kept[k] for k in sorted(kept)}, merges


def scene_scale(cloud: np.ndarray) -> float:
    """A robust size for the scene: the median distance from the cloud's centre."""
    if not len(cloud):
        return 1.0
    return float(np.median(np.linalg.norm(cloud - np.median(cloud, axis=0), axis=1))) or 1.0


# ── the capture ──────────────────────────────────────────────────────────────

CAPTURE_FILES = ("meta.json", "depth.bin")
CAPTURE_FORMAT = ("a directory holding video.mov, depth.bin (float16 meters, one "
                  "slab per video frame), meta.json (frame_count, depth width and "
                  "height, intrinsics_video) and optionally frames.csv")


def is_capture(path: Path) -> bool:
    """A directory with depth in it: a recording from the depth-capture app."""
    return path.is_dir() and all((path / name).is_file() for name in CAPTURE_FILES)


@dataclass
class Capture:
    """One LiDAR recording: the video, its depth, and the camera that shot it."""

    root: Path
    video: Path
    depth: np.ndarray            # (N, h, w) float16 meters, memory-mapped
    valid: np.ndarray            # (N,) bool: this frame has a depth map
    K_video: np.ndarray          # (3, 3) in the saved video's pixel grid
    video_size: tuple[int, int]  # (width, height) of video.mov
    depth_size: tuple[int, int]  # (width, height) of each depth slab
    fps: float
    meta: dict

    @property
    def n_frames(self) -> int:
        return len(self.depth)

    @property
    def label(self) -> str:
        return self.root.name

    def K(self, size: tuple[int, int]) -> np.ndarray:
        """The intrinsics rescaled to an image of *size* ``(width, height)``.

        The capture's own grids scale with no half-pixel shift — its depth
        intrinsics are exactly the video's divided by six — so neither does this.
        """
        sx = size[0] / self.video_size[0]
        sy = size[1] / self.video_size[1]
        K = self.K_video.copy()
        K[0] *= sx
        K[1] *= sy
        return K

    def depth_map(self, frame: int, *, edge_tolerance: float = 0.04) -> np.ndarray | None:
        """Frame *frame*'s depth in meters, NaN where it cannot be trusted.

        Two kinds of pixel are thrown away. Zeros are the sensor saying it has
        no reading. Depth *edges* are the subtler one: at a silhouette — the
        climber against the wall, a hold's lip against the panel — a LiDAR pixel
        averages the two surfaces and reports a depth belonging to neither. A
        feature sitting on one of those would be placed in mid-air between them,
        so any pixel whose 3x3 neighbourhood spans more than *edge_tolerance* of
        its own depth is dropped.
        """
        if frame < 0 or frame >= self.n_frames or not self.valid[frame]:
            return None
        depth = np.asarray(self.depth[frame], dtype=np.float32)
        kernel = np.ones((3, 3), np.uint8)
        spread = cv2.dilate(depth, kernel) - cv2.erode(depth, kernel)
        bad = (depth <= 0) | (spread > edge_tolerance * depth + 0.01)
        out = depth.copy()
        out[bad] = np.nan
        return out

    def sample(self, depth: np.ndarray, pixels: np.ndarray, size: tuple[int, int]
               ) -> np.ndarray:
        """Depth at *pixels* (in an image of *size*), nearest-neighbour. NaN if none."""
        pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
        w, h = self.depth_size
        u = np.clip(np.floor(pixels[:, 0] * w / size[0]), 0, w - 1).astype(int)
        v = np.clip(np.floor(pixels[:, 1] * h / size[1]), 0, h - 1).astype(int)
        return depth[v, u]


def load(root: Path) -> Capture:
    """Read a capture directory. Nothing is decoded yet; depth is memory-mapped."""
    meta = json.loads((root / "meta.json").read_text())
    n = int(meta["frame_count"])
    w, h = int(meta["depth"]["width"]), int(meta["depth"]["height"])
    depth_file = root / meta["depth"].get("file", "depth.bin")
    expected = n * w * h * 2
    if depth_file.stat().st_size != expected:
        # A truncated upload or a different sensor's layout: either way, slab i
        # would no longer be frame i, and every hold would be placed wrongly.
        raise ValueError(
            f"{depth_file.name} is {depth_file.stat().st_size:,} bytes; meta.json "
            f"describes {n} frames of {w}x{h} float16, which is {expected:,}")
    depth = np.memmap(depth_file, dtype="<f2",
                      mode="r", shape=(n, h, w))

    valid = np.ones(n, dtype=bool)
    frames_csv = root / "frames.csv"
    if frames_csv.is_file():
        with frames_csv.open() as handle:
            for row in csv.DictReader(handle):
                index = int(row["frame_index"])
                if index < n:
                    valid[index] = row.get("depth_valid", "1").strip() == "1"

    video_meta = meta.get("video", {})
    intr = meta["intrinsics_video"]
    K = np.array([[intr["fx"], 0.0, intr["cx"]],
                  [0.0, intr["fy"], intr["cy"]],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return Capture(
        root=root, video=root / video_meta.get("file", "video.mov"), depth=depth,
        valid=valid, K_video=K,
        video_size=(int(intr["width"]), int(intr["height"])), depth_size=(w, h),
        fps=float(video_meta.get("nominal_fps", 30)), meta=meta)


def backproject(pixels: np.ndarray, depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Pixels with depth into camera coordinates, ``(N, 3)`` meters."""
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    z = np.asarray(depth, dtype=np.float64).reshape(-1)
    x = (pixels[:, 0] - K[0, 2]) * z / K[0, 0]
    y = (pixels[:, 1] - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=1)


# ── the camera's path ────────────────────────────────────────────────────────

@dataclass
class _Keyframe:
    frame: int
    descriptors: np.ndarray      # only for keypoints that have depth
    world: np.ndarray            # (N, 3) those keypoints, placed in the world


@dataclass
class Odometry:
    """Where the camera was for every frame, in meters."""

    poses: dict[int, tuple[np.ndarray, np.ndarray]]   # world -> camera, as (R, t)
    keyframes: list[int]
    inliers: dict[int, int]
    rms_px: float
    path_length_m: float
    lost: list[int]


def _pose_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).ravel()
    return T


def odometry(video: Path, capture: Capture, K: np.ndarray, *, n_features: int = 2500,
             ratio: float = 0.8, max_reproj_px: float = 3.0, local_map: int = 5,
             min_inliers: int = 30, keyframe_inliers: float = 0.55,
             keyframe_every: int = 12, max_step_m: float = 0.25,
             console=None) -> Odometry:
    """RGB-D visual odometry: one PnP per frame against a local map of keyframes.

    For every frame, SIFT features are matched against the last *local_map*
    keyframes. Those keyframes' features already carry world positions — their
    depth was read off the LiDAR the moment they became keyframes — so each
    match is a 2D-3D correspondence and the pose is a single RANSAC PnP, refined
    by Levenberg-Marquardt on its inliers.

    Matching against several keyframes rather than only the last one is what
    holds the drift down: the error in a chain of frame-to-frame poses grows
    with every link, while a frame matched to a keyframe from half a second ago
    is one link from it, and that keyframe is one link from the one before.

    A frame becomes a keyframe when its overlap with the newest one falls below
    *keyframe_inliers* of what that keyframe started with, or every
    *keyframe_every* frames regardless — and only if it has a depth map, since
    a keyframe without depth has no points to offer. A frame *without* depth is
    still localized: PnP needs the map's depth, not its own.

    The climber is the one thing in shot that moves, and their features land in
    keyframes too. They are outliers the next time anyone looks — the wall
    agrees with itself and the climber does not — which is exactly what RANSAC
    is for. *max_step_m* is the backstop for the rare frame where they are the
    majority: a pose that jumps further than a handheld phone moves in one
    frame is dropped rather than believed.
    """
    size = None
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {video}")
    detector = cv2.SIFT_create(nfeatures=n_features)
    matcher = cv2.BFMatcher()

    keyframes: list[_Keyframe] = []
    poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    inliers: dict[int, int] = {}
    residuals: list[float] = []
    lost: list[int] = []
    last_T = None
    since_keyframe = 0
    # Inliers the newest keyframe earned from the first frame matched against
    # it: the overlap it starts with, which later frames are measured against.
    reference = None

    frame = 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        if size is None:
            size = (image.shape[1], image.shape[0])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keys, desc = detector.detectAndCompute(gray, None)
        pixels = np.float32([k.pt for k in keys]) if keys else np.zeros((0, 2), np.float32)
        depth = capture.depth_map(frame)

        T = None
        newest = 0
        if not keyframes:
            T = np.eye(4)
        elif desc is not None and len(desc) >= min_inliers:
            T, count, rms, newest = _localize(
                pixels, desc, keyframes[-local_map:], K, matcher, ratio=ratio,
                max_reproj_px=max_reproj_px, min_inliers=min_inliers, guess=last_T)
            if T is not None and last_T is not None:
                step = np.linalg.norm(
                    (-T[:3, :3].T @ T[:3, 3]) - (-last_T[:3, :3].T @ last_T[:3, 3]))
                if step > max_step_m * max(1, frame - max(poses)):
                    T = None
            if T is not None:
                inliers[frame] = count
                residuals.append(rms)
                if reference is None:
                    reference = max(newest, 1)
        if T is None:
            lost.append(frame)
            frame += 1
            since_keyframe += 1
            continue

        poses[frame] = (T[:3, :3].copy(), T[:3, 3].copy())
        last_T = T
        since_keyframe += 1

        wants = (not keyframes or since_keyframe >= keyframe_every
                 or (reference is not None and newest < keyframe_inliers * reference))
        if wants and depth is not None and desc is not None:
            z = capture.sample(depth, pixels, size)
            has = np.isfinite(z)
            if has.sum() >= min_inliers:
                cam = backproject(pixels[has], z[has], K)
                R, t = T[:3, :3], T[:3, 3]
                world = (cam - t) @ R          # R^T (x - t), row-wise
                keyframes.append(_Keyframe(frame=frame, descriptors=desc[has],
                                           world=world))
                reference = None
                since_keyframe = 0
        frame += 1

    cap.release()
    centres = [(-R.T @ t) for _, (R, t) in sorted(poses.items())]
    path = float(sum(np.linalg.norm(b - a) for a, b in zip(centres, centres[1:])))
    rms = float(np.sqrt(np.mean(np.square(residuals)))) if residuals else 0.0
    if console:
        console.print(f"  [dim]{len(poses)} of {frame} frames placed, "
                      f"{len(keyframes)} keyframes, reprojection RMS {rms:.2f}px, "
                      f"camera travelled {path:.2f} m[/]")
    return Odometry(poses=poses, keyframes=[k.frame for k in keyframes],
                    inliers=inliers, rms_px=rms, path_length_m=path, lost=lost)


def _localize(pixels, desc, keyframes, K, matcher, *, ratio, max_reproj_px,
              min_inliers, guess):
    """PnP of one frame against a handful of keyframes. ``(T, inliers, rms, newest)``.

    ``newest`` is how many of the inliers came from the newest keyframe — the
    number the keyframe policy is decided on.
    """
    object_points, image_points, source = [], [], []
    taken: dict[int, float] = {}
    for rank, keyframe in enumerate(reversed(keyframes)):
        if len(keyframe.descriptors) < 2:
            continue
        for pair in matcher.knnMatch(desc, keyframe.descriptors, k=2):
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance >= ratio * n.distance:
                continue
            # One correspondence per image keypoint: the newest keyframe's claim
            # wins, since it is the one with the least drift between it and here.
            if m.queryIdx in taken:
                continue
            taken[m.queryIdx] = m.distance
            object_points.append(keyframe.world[m.trainIdx])
            image_points.append(pixels[m.queryIdx])
            source.append(rank)
    if len(object_points) < min_inliers:
        return None, 0, 0.0, 0
    object_points = np.asarray(object_points, dtype=np.float64)
    image_points = np.asarray(image_points, dtype=np.float64)
    source = np.asarray(source)

    rvec = tvec = None
    use_guess = guess is not None
    if use_guess:
        rvec = cv2.Rodrigues(guess[:3, :3])[0]
        tvec = guess[:3, 3].reshape(3, 1).copy()
    try:
        ok, rvec, tvec, idx = cv2.solvePnPRansac(
            object_points, image_points, K, None, rvec, tvec,
            useExtrinsicGuess=use_guess, iterationsCount=500,
            reprojectionError=max_reproj_px, confidence=0.999,
            flags=cv2.SOLVEPNP_ITERATIVE if use_guess else cv2.SOLVEPNP_SQPNP)
    except cv2.error:
        return None, 0, 0.0, 0
    if not ok or idx is None or len(idx) < min_inliers:
        return None, 0, 0.0, 0
    idx = idx.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(object_points[idx], image_points[idx], K, None,
                                      rvec, tvec)
    projected = cv2.projectPoints(object_points[idx], rvec, tvec, K, None)[0].reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum((projected - image_points[idx]) ** 2, axis=1))))
    T = _pose_matrix(cv2.Rodrigues(rvec)[0], tvec)
    return T, int(len(idx)), rms, int((source[idx] == 0).sum())


def interpolate(poses: dict[int, tuple[np.ndarray, np.ndarray]], n_frames: int,
                *, max_gap: int = 6) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Fill short gaps in the trajectory. Longer ones are left empty.

    A frame the odometry could not place for a few frames — a motion-blurred
    whip, the climber filling the shot — sits between two good poses a tenth of
    a second apart, and the camera did not go anywhere surprising in between.
    Past *max_gap* frames that stops being true, and a missing pose is honest
    where an invented one is not.
    """
    from scipy.spatial.transform import Rotation, Slerp

    out = dict(poses)
    known = sorted(poses)
    for a, b in zip(known, known[1:]):
        if b - a <= 1 or b - a > max_gap + 1:
            continue
        Ta = np.linalg.inv(_pose_matrix(*poses[a]))    # camera -> world
        Tb = np.linalg.inv(_pose_matrix(*poses[b]))
        slerp = Slerp([a, b], Rotation.from_matrix([Ta[:3, :3], Tb[:3, :3]]))
        for f in range(a + 1, b):
            s = (f - a) / (b - a)
            C = np.eye(4)
            C[:3, :3] = slerp([f]).as_matrix()[0]
            C[:3, 3] = (1 - s) * Ta[:3, 3] + s * Tb[:3, 3]
            W = np.linalg.inv(C)
            out[f] = (W[:3, :3], W[:3, 3])
    return {f: out[f] for f in sorted(out) if 0 <= f < n_frames}


def smooth(poses: dict[int, tuple[np.ndarray, np.ndarray]], *, sigma: float
           ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Gaussian-smooth the camera path over time, rotation and translation both.

    The per-frame PnP carries a millimeter or two of independent noise, and
    projected through three meters of lever arm that is a hold outline
    shimmering by a pixel against a wall that is plainly not moving. Smoothing
    over σ frames removes it without lagging a real pan by anything visible.
    Only runs of consecutive frames are smoothed together; a gap is a boundary.
    """
    if sigma <= 0 or len(poses) < 3:
        return poses
    from scipy.ndimage import gaussian_filter1d
    from scipy.spatial.transform import Rotation

    frames = sorted(poses)
    runs, current = [], [frames[0]]
    for f in frames[1:]:
        if f == current[-1] + 1:
            current.append(f)
        else:
            runs.append(current)
            current = [f]
    runs.append(current)

    out = {}
    for run in runs:
        C = [np.linalg.inv(_pose_matrix(*poses[f])) for f in run]
        if len(run) < 3:
            for f, c in zip(run, C):
                out[f] = poses[f]
            continue
        centres = gaussian_filter1d(np.array([c[:3, 3] for c in C]), sigma, axis=0,
                                    mode="nearest")
        quats = Rotation.from_matrix(np.array([c[:3, :3] for c in C])).as_quat()
        for i in range(1, len(quats)):           # keep the hemisphere continuous
            if quats[i] @ quats[i - 1] < 0:
                quats[i] = -quats[i]
        quats = gaussian_filter1d(quats, sigma, axis=0, mode="nearest")
        rots = Rotation.from_quat(quats / np.linalg.norm(quats, axis=1, keepdims=True))
        for f, centre, rot in zip(run, centres, rots.as_matrix()):
            R = rot.T
            out[f] = (R, -R @ centre)
    return out


# ── the scene ────────────────────────────────────────────────────────────────

def fuse(video: Path, capture: Capture, poses: dict, K: np.ndarray, *,
         frames: list[int], voxel: float = 0.02, max_depth: float = 8.0,
         pixel_stride: int = 2, exclude=None
         ) -> tuple[np.ndarray, np.ndarray]:
    """Every chosen frame's depth map, placed in the world and voxelized.

    ``(points (N, 3) meters, colours (N, 3) BGR uint8)``. The colour is sampled
    from the video frame at each depth pixel, so the cloud is a photograph of
    the wall in three dimensions rather than a grey constellation.

    *exclude* is an optional ``frame -> (h, w) bool mask`` of pixels to leave
    out, in depth-map resolution. The climber is the obvious one: fused over a
    whole clip they become a smear of ghosts up the face of the wall.

    Each voxel keeps the point nearest its own centre-of-mass rather than an
    average, because an average of colours across frames with different exposure
    is a muddier colour than any of them.
    """
    cap = cv2.VideoCapture(str(video))
    w, h = capture.depth_size
    us, vs = np.meshgrid(np.arange(0, w, pixel_stride), np.arange(0, h, pixel_stride))
    wanted = set(int(f) for f in frames)
    chunks, colours = [], []
    frame = 0
    size = None
    while wanted:
        ok, image = cap.read()
        if not ok:
            break
        if size is None:
            size = (image.shape[1], image.shape[0])
            # Depth pixel centres, in video pixels.
            px = np.stack([(us + 0.5) * size[0] / w, (vs + 0.5) * size[1] / h], axis=-1)
        if frame in wanted and frame in poses:
            wanted.discard(frame)
            depth = capture.depth_map(frame)
            if depth is not None:
                z = depth[vs, us]
                keep = np.isfinite(z) & (z < max_depth)
                if exclude is not None:
                    mask = exclude(frame)
                    if mask is not None:
                        keep &= ~mask[vs, us]
                if keep.any():
                    cam = backproject(px[keep], z[keep], K)
                    R, t = poses[frame]
                    chunks.append((cam - t) @ R)
                    xi = np.clip(px[keep][:, 0].astype(int), 0, size[0] - 1)
                    yi = np.clip(px[keep][:, 1].astype(int), 0, size[1] - 1)
                    colours.append(image[yi, xi])
        frame += 1
    cap.release()
    if not chunks:
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
    points = np.vstack(chunks)
    colours = np.vstack(colours)

    keys = np.floor(points / voxel).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    # A voxel seen once is as likely a mixed-pixel stray as a surface.
    sums = np.zeros((counts.size, 3))
    np.add.at(sums, inverse, points)
    means = sums / counts[:, None]
    dist = np.linalg.norm(points - means[inverse], axis=1)
    order = np.lexsort((dist, inverse))
    first = order[np.r_[True, inverse[order][1:] != inverse[order][:-1]]]
    solid = counts[inverse[first]] >= 2
    return points[first][solid], colours[first][solid]


def lift(observations: dict[int, list[tuple[int, np.ndarray]]], poses, capture: Capture,
         K: np.ndarray, frame_size: tuple[int, int], cloud: np.ndarray, *,
         neighbourhood: float = 0.15, max_spread_m: float = 0.25,
         min_views: int = 2, max_outline_m: float = 0.5,
         console=None) -> dict[int, Hold3D]:
    """Place each tracked hold in space, from the depth under its mask.

    Every sighting places the hold on its own: the depth
    pixels inside that frame's outline are points *on the hold*, in meters, and
    their median is where it is. A hold seen from one spot only is placed as
    well as a hold seen from ten.

    The sightings are then reconciled. A hold is bolted to the wall, so all of
    its sightings must land
    in one place; any further than *max_spread_m* from their joint median is a
    SAM identity switch and is dropped, and a hold left with fewer than
    *min_views* is not kept.

    The outline is the clearest single view's, back-projected onto the hold's
    own plane, fitted to the real surface around the hold.
    """
    lifted: dict[int, Hold3D] = {}
    depths_by_frame: dict[int, np.ndarray | None] = {}
    w, h = capture.depth_size
    for hold_id, sightings in sorted(observations.items()):
        placed: dict[int, np.ndarray] = {}
        outlines: dict[int, np.ndarray] = {}
        for frame, polygon in sightings:
            if frame not in poses:
                continue
            if frame not in depths_by_frame:
                depths_by_frame[frame] = capture.depth_map(frame)
            depth = depths_by_frame[frame]
            if depth is None:
                continue
            polygon = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
            if len(polygon) < 3:
                continue
            # The mask, rasterized on the depth grid. Eroded by a pixel, so the
            # lip of the hold — where depth jumps to the wall behind — is out.
            scaled = polygon * [w / frame_size[0], h / frame_size[1]]
            mask = np.zeros((h, w), np.uint8)
            cv2.fillPoly(mask, [np.round(scaled).astype(np.int32)], 1)
            eroded = cv2.erode(mask, np.ones((3, 3), np.uint8))
            if eroded.sum() >= 3:
                mask = eroded
            vs, us = np.nonzero(mask)
            z = depth[vs, us]
            ok = np.isfinite(z)
            if ok.sum() < 3:
                continue
            px = np.stack([(us[ok] + 0.5) * frame_size[0] / w,
                           (vs[ok] + 0.5) * frame_size[1] / h], axis=1)
            cam = backproject(px, z[ok], K)
            R, t = poses[frame]
            placed[frame] = np.median((cam - t) @ R, axis=0)
            outlines[frame] = polygon
        if len(placed) < min_views:
            continue

        # One place per hold. Iterate the median so a cluster of switched
        # sightings cannot drag it far enough to keep themselves in.
        centre = np.median(np.array(list(placed.values())), axis=0)
        for _ in range(3):
            near = {f: p for f, p in placed.items()
                    if np.linalg.norm(p - centre) <= max_spread_m}
            if len(near) < min_views:
                break
            centre = np.median(np.array(list(near.values())), axis=0)
        near = {f: p for f, p in placed.items()
                if np.linalg.norm(p - centre) <= max_spread_m}
        if len(near) < min_views:
            continue
        spread = float(np.median([np.linalg.norm(p - centre) for p in near.values()]))

        eye = np.mean([(-poses[f][0].T @ poses[f][1]).ravel() for f in near], axis=0)
        normal = _fit_normal(cloud, centre, eye, neighbourhood)

        best = max(near, key=lambda f: cv2.contourArea(
            np.asarray(outlines[f], dtype=np.float32).reshape(-1, 1, 2)))
        R, t = poses[best]
        camera_centre = (-R.T @ t).ravel()
        rays = (R.T @ np.linalg.inv(K) @ np.hstack([
            outlines[best], np.ones((len(outlines[best]), 1))]).T).T
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        # A plane seen nearly edge-on sends the outline's rays to meet it far
        # away — one hold's outline ran to tens of meters that way and, as the
        # widest thing in the route, squashed every other hold's flat
        # coordinates to a point. So an oblique plane is swapped for one facing
        # this camera, which is what the outline was traced on anyway.
        plane = normal
        if np.min(np.abs(rays @ plane)) < 0.35:
            plane = camera_centre - centre
            plane /= max(np.linalg.norm(plane), 1e-9)
        denominator = rays @ plane
        safe = np.where(np.abs(denominator) < 1e-9, 1e-9, denominator)
        distances = ((centre - camera_centre) @ plane) / safe
        polygon3d = camera_centre + rays * distances[:, None]
        # And no hold is a meter across: a vertex further out is clamped back.
        offset = polygon3d - centre
        reach = np.linalg.norm(offset, axis=1, keepdims=True)
        polygon3d = centre + offset * np.minimum(1.0, max_outline_m / np.maximum(reach, 1e-9))

        lifted[hold_id] = Hold3D(
            id=hold_id, centre=centre, normal=normal, polygon=polygon3d,
            # No triangulation angle to speak of: depth placed every sighting.
            # The field carries the agreement between sightings instead, in mm,
            # which is what a reader of holds.json wants to know about a place.
            angle_deg=round(spread * 1000.0, 1), n_views=len(near),
            source={"frame": int(best), "views": sorted(int(f) for f in near),
                    "spread_m": round(spread, 4)})
    if console:
        console.print(f"  [dim]{len(lifted)} of {len(observations)} holds placed "
                      f"in 3D from the depth under their masks[/]")
    return lifted


def reconstruction(odo: Odometry, poses: dict, points: np.ndarray, K: np.ndarray
                   ) -> Reconstruction:
    """The odometry and fused cloud, in the shape the rest of the pipeline reads.

    *poses* rather than ``odo.poses``: the caller has filled and smoothed them,
    and the keyframes the rest of the pipeline reads must agree with the frames.
    """
    keyframes = [k for k in odo.keyframes if k in poses]
    return Reconstruction(keyframes=keyframes, poses={k: poses[k] for k in keyframes},
                          points=points, point_tracks=[], K=K, rms_px=odo.rms_px)


# ── the body ─────────────────────────────────────────────────────────────────

def lift_skeleton(capture: Capture, poses, K: np.ndarray, frame_size: tuple[int, int],
                  kpts: dict[int, np.ndarray], valid: dict[int, np.ndarray], *,
                  window: int = 2, max_jump_m: float = 0.35
                  ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """The climber's keypoints in the world, from the depth under each joint.

    ``{frame: (points (17, 3) meters, ok (17,) bool)}``. This is what a single
    camera cannot do without depth: a keypoint is a ray, and a ray has no
    position on it. With LiDAR it does — the depth at the joint's pixel.

    The depth is the *nearest* reading in a small window rather than the value
    at the exact pixel, because a keypoint sits on the body's midline but
    ViTPose's wrist can land a pixel off the arm onto the wall behind, and the
    nearer surface is the body. A joint that then lands more than *max_jump_m*
    from the body's own median is off the body entirely and is dropped.
    """
    out = {}
    w, h = capture.depth_size
    for frame, points in kpts.items():
        if frame not in poses:
            continue
        depth = capture.depth_map(frame, edge_tolerance=1.0)
        if depth is None:
            continue
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2) * frame_size
        ok = np.asarray(valid[frame], dtype=bool).copy()
        z = np.full(len(pts), np.nan)
        for j, (x, y) in enumerate(pts):
            if not ok[j]:
                continue
            u, v = int(x * w / frame_size[0]), int(y * h / frame_size[1])
            patch = depth[max(0, v - window):v + window + 1, max(0, u - window):u + window + 1]
            if patch.size and np.isfinite(patch).any():
                z[j] = np.nanmin(patch)
        ok &= np.isfinite(z)
        if ok.sum() < 4:
            continue
        cam = backproject(pts, np.nan_to_num(z, nan=1.0), K)
        R, t = poses[frame]
        world = (cam - t) @ R
        middle = np.median(world[ok], axis=0)
        ok &= np.linalg.norm(world - middle, axis=1) <= max(max_jump_m * 3, 1.2)
        out[frame] = (world, ok)
    return out


# ── the floor ────────────────────────────────────────────────────────────────

def _local_triplets(points: np.ndarray, count: int, rng, *, radius: float = 0.3):
    """RANSAC samples drawn from one neighbourhood at a time.

    A seed point, then two more within *radius* of it. Three points drawn from
    the whole cloud only share a plane when all three happen to land on the
    same surface — rare for anything but the dominant one — while three points
    from one 30 cm patch almost always do, so the smaller planes get proposed
    as often as the large one.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    produced = 0
    for _ in range(count * 3):
        if produced >= count:
            return
        seed = points[rng.integers(len(points))]
        near = tree.query_ball_point(seed, radius)
        if len(near) < 3:
            continue
        a, b = rng.choice(near, 2, replace=False)
        produced += 1
        yield np.array([seed, points[a], points[b]])


def camera_up(poses: dict) -> np.ndarray:
    """The phone's own "up", averaged over the clip, in the world.

    A person filming in portrait holds the phone upright — the top of the image
    is the sky — so the mean of every frame's image-up direction is a good
    first guess at gravity. Only a guess: the operator tilts to follow the
    climber, and the floor is what refines it (:func:`ground_plane`).
    """
    ups = np.array([R.T @ np.array([0.0, -1.0, 0.0]) for R, _ in poses.values()])
    up = ups.mean(axis=0)
    return up / max(np.linalg.norm(up), 1e-9)


def ground_plane(points: np.ndarray, wall_normal: np.ndarray, up_guess: np.ndarray,
                 *, threshold: float = 0.03, iterations: int = 1500, max_tilt_deg: float = 55.0,
                 seed: int = 0) -> tuple[np.ndarray, float, int] | None:
    """The floor as ``(normal, offset, support)``, ``normal · x = offset``; or None.

    The dense cloud has the mats in it, so the floor can be fitted as what it
    is: the largest plane within
    *max_tilt_deg* of the phone's up (:func:`camera_up`) that lies *below* the
    camera. The ceiling passes the tilt test too, and fails the second.

    It is not the largest plane in the cloud, and must not be looked for as
    one: the wall is, by four times over. Two versions of this got it wrong in
    ways worth remembering. The first took the route's long axis as the tilt
    reference — its sign is SAM's track order — rejected the real floor, and
    rolled the 3D panel by 30°. The second drew three points at random from the
    whole cloud: the floor is under a tenth of it, so a triplet landing on it
    turned up once or twice in 1500 draws, and the sloped edge of the mats won
    at 25° off level. Triplets are now drawn from one neighbourhood
    (:func:`_local_triplets`), so most of them lie on *some* plane, and the
    tilt allowance is wide because an operator following a climber up the wall
    pitches the phone well back from level.

    Its normal is then the best estimate of gravity this capture has, and it
    comes back pointing *up*.
    """
    rng = np.random.default_rng(seed)
    if len(points) < 100:
        return None
    sample_from = points[rng.choice(len(points), min(len(points), 60000), replace=False)]
    best, best_count = None, 0
    for sample in _local_triplets(sample_from, iterations, rng):
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        if normal @ up_guess < 0:
            normal = -normal
        if normal @ up_guess < np.cos(np.radians(max_tilt_deg)):
            continue
        offset = normal @ sample[0]
        # Below the camera (the world origin is frame 0's camera): the floor,
        # not the ceiling.
        if offset >= 0 or abs(normal @ wall_normal) > 0.5:
            continue
        count = int((np.abs(sample_from @ normal - offset) < threshold).sum())
        if count > best_count:
            best, best_count = (normal, offset), count
    if best is None or best_count < 0.03 * len(sample_from):
        return None
    normal, offset = best
    near = points[np.abs(points @ normal - offset) < threshold]
    centroid = near.mean(axis=0)
    refined = np.linalg.svd(near - centroid, full_matrices=False)[2][-1]
    if refined @ normal < 0:
        refined = -refined
    return refined, float(refined @ centroid), int(len(near))


def wall_plane(points: np.ndarray, holds: dict, toward: np.ndarray, *,
               reach: float = 0.6, threshold: float = 0.03, iterations: int = 800,
               seed: int = 0) -> np.ndarray | None:
    """The wall the route is bolted to, as a unit normal facing *toward*.

    Not the holds' own normals: a hold is shaped to be pulled on, so its surface
    faces *up* — that is what makes a jug a jug — and their median tilts the
    "wall" 20° back from the plywood it is on. The panel then looks down on the
    route, and every measurement of the wall's angle comes out that much too
    steep. The plywood between the holds is the wall, so a plane is fitted to
    the cloud within *reach* of the route's holds.
    """
    centres = np.array([h.centre for h in holds.values()])
    lo, hi = centres.min(axis=0) - reach, centres.max(axis=0) + reach
    near = points[np.all((points >= lo) & (points <= hi), axis=1)]
    if len(near) < 200:
        return None
    rng = np.random.default_rng(seed)
    near = near[rng.choice(len(near), min(len(near), 40000), replace=False)]
    best, best_count = None, 0
    for sample in _local_triplets(near, iterations, rng):
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        if np.linalg.norm(normal) < 1e-9:
            continue
        normal /= np.linalg.norm(normal)
        count = int((np.abs((near - sample[0]) @ normal) < threshold).sum())
        if count > best_count:
            best, best_count = (normal, sample[0]), count
    normal, origin = best
    inliers = near[np.abs((near - origin) @ normal) < threshold]
    centroid = inliers.mean(axis=0)
    normal = np.linalg.svd(inliers - centroid, full_matrices=False)[2][-1]
    if normal @ (toward - centroid) < 0:
        normal = -normal
    return normal


def wall_axes(holds: dict, cloud: np.ndarray, eye: np.ndarray, up_guess: np.ndarray):
    """The route's axes, with *up* from the floor rather than from the holds.

    ``(WallFrame, floor or None)``. :func:`src.space.wall_frame` takes up from
    the route's spread, which tilts the whole panel whenever the route
    traverses; it is only the fallback here. The floor is in the cloud, dense
    and flat, and its normal *is* gravity. *out* is the
    holds' own median surface normal with the vertical taken out of it, so the
    default view looks at the wall horizontally, the way someone standing on the
    mat does, however steep the wall is.
    """
    from src.space import WallFrame, wall_frame

    base = wall_frame(holds, cloud, eye)
    centres = np.array([h.centre for h in holds.values()])
    out = wall_plane(cloud, holds, eye)
    if out is None:
        out = np.median(np.array([h.normal for h in holds.values()]), axis=0)
    if np.linalg.norm(out) < 1e-6:
        out = base.out
    out = out / np.linalg.norm(out)
    if out @ (eye - centres.mean(axis=0)) < 0:
        out = -out
    fallback = up_guess - (up_guess @ out) * out
    fallback /= max(np.linalg.norm(fallback), 1e-9)

    floor = ground_plane(cloud, out, up_guess)
    up = floor[0] if floor is not None else fallback
    out = out - (out @ up) * up
    out /= max(np.linalg.norm(out), 1e-9)
    right = np.cross(up, out)
    right /= np.linalg.norm(right)
    frame = WallFrame(origin=centres.mean(axis=0), right=right, up=up, out=out,
                      extent=np.ones(3))
    frame.extent = np.maximum(np.abs(frame.to_wall(centres)).max(axis=0), 1e-6)
    # The wall's angle past vertical, measured against the floor: positive
    # leans out over the climber. Reported, and kept for the panel.
    lean = None
    if floor is not None:
        wall_n = wall_plane(cloud, holds, eye)
        if wall_n is not None:
            lean = float(np.degrees(np.arcsin(np.clip(-(wall_n @ floor[0]), -1, 1))))
    frame.lean_deg = lean
    return frame, floor


# ── contact, in three dimensions ─────────────────────────────────────────────

class ContactGate:
    """Is this limb physically near this hold? Meters, from the depth.

    The 2D test asks whether a keypoint falls inside a hold's outline *in the
    image*, and a hand reaching past a hold — in front of it, on its way to the
    next one — falls inside that outline as surely as a hand gripping it. The
    dwell and graze rules in :mod:`src.climb` exist to paper over exactly that,
    and they can only do it by how long the overlap lasts. With depth the
    question can be asked directly: how far is the wrist from the hold's
    surface, in the room?

    :meth:`distance` answers that, or None when the depth cannot say:

    * no depth map for the frame, or no reading under the joint;
    * **the joint is hidden behind the body.** Filmed from behind, a wrist
      reaching overhead is often behind the head or the back, and the depth at
      its pixel is the climber, not the hand — a body's thickness in front of
      the hold. Treating that as the hand would reject real grips, so a wrist
      inside the torso's own outline in the image is left to the 2D test.

    None is never a rejection. The gate only removes a contact the depth
    positively contradicts.
    """

    def __init__(self, skeleton: dict, holds: dict, poses2d, *, aspect: float,
                 hand_m: float | None, foot_m: float, release_m: float,
                 hand_reach_m: float | None = None):
        self.skeleton = skeleton
        self.poses2d = poses2d
        self.aspect = aspect
        self.hand_m = hand_m
        self.foot_m = foot_m
        self.release_m = release_m
        self.hand_reach_m = hand_reach_m
        self._holds = {}
        for hold_id, hold in holds.items():
            points = np.asarray(hold.polygon, dtype=float)
            centre = points.mean(axis=0)
            basis = np.linalg.svd(points - centre, full_matrices=False)[2]
            flat = (points - centre) @ basis[:2].T
            self._holds[hold_id] = (centre, basis, flat.astype(np.float32).reshape(-1, 1, 2))
        self.log: list[dict] = []

    def _hidden(self, frame: int, kpt: int) -> bool:
        kpts = self.poses2d.kpts.get(frame)
        valid = self.poses2d.valid.get(frame)
        if kpts is None or valid is None or kpt not in (9, 10):
            return False
        quad = [5, 6, 12, 11]                  # shoulders, hips: the torso
        if not all(valid[k] for k in quad):
            return False
        pts = np.asarray([kpts[k] for k in quad], dtype=np.float32) * [self.aspect, 1.0]
        # Grown a little: ViTPose's wrist on an occluded arm sits near the edge.
        middle = pts.mean(axis=0)
        pts = middle + (pts - middle) * 1.15
        x, y = kpts[kpt]
        head = [k for k in (0, 3, 4) if valid[k]]
        if head:
            hx, hy = np.mean([kpts[k] for k in head], axis=0)
            neck = np.mean([kpts[5], kpts[6]], axis=0)
            radius = 0.9 * float(np.hypot((hx - neck[0]) * self.aspect, hy - neck[1]))
            if np.hypot((x - hx) * self.aspect, y - hy) <= radius:
                return True
        return cv2.pointPolygonTest(pts.astype(np.float32).reshape(-1, 1, 2),
                                    (float(x) * self.aspect, float(y)), False) >= 0

    def distance(self, frame: int, kpt: int, hold_id: int) -> float | None:
        """Meters from the joint to the hold's outline patch, or None if unknown."""
        body = self.skeleton.get(frame)
        hold = self._holds.get(hold_id)
        if body is None or hold is None:
            return None
        world, ok = body
        if not ok[kpt] or self._hidden(frame, kpt):
            return None
        centre, basis, contour = hold
        rel = world[kpt] - centre
        off_plane = float(rel @ basis[2])
        in_plane = rel @ basis[:2].T
        inside = cv2.pointPolygonTest(contour, (float(in_plane[0]), float(in_plane[1])), True)
        return float(np.hypot(off_plane, max(0.0, -inside)))

    def reach(self, frame: int, kpt: int, hold_id: int) -> float | None:
        """Meters from the limb's shoulder to the hold's centre, or None.

        The shoulder is a large, visible surface from behind, so its depth is
        the body's even when the hand's is not.
        """
        body = self.skeleton.get(frame)
        hold = self._holds.get(hold_id)
        shoulder = {9: 5, 10: 6}.get(kpt)
        if body is None or hold is None or shoulder is None or not body[1][shoulder]:
            return None
        return float(np.linalg.norm(body[0][shoulder] - hold[0]))

    def allows(self, frame: int, kpt: int, hold_id: int, *, keeping: bool) -> bool:
        """Whether the depth permits this contact. Unknown depth always permits."""
        hand = kpt in (9, 10)
        slack = self.release_m if keeping else 0.0
        if hand and self.hand_reach_m is not None:
            r = self.reach(frame, kpt, hold_id)
            if r is not None and r > self.hand_reach_m + slack:
                self.log.append({"frame": frame, "kpt": kpt, "hold": hold_id,
                                 "reach_m": round(r, 3), "keeping": keeping})
                return False
        # The ankle is not the foot: the toes that touch the hold are a foot's
        # length further on, so feet get the longer reach.
        limit = self.hand_m if hand else self.foot_m
        if limit is None:
            return True
        d = self.distance(frame, kpt, hold_id)
        if d is None:
            return True
        ok = d <= limit + slack
        if not ok:
            self.log.append({"frame": frame, "kpt": kpt, "hold": hold_id,
                             "distance_m": round(d, 3), "keeping": keeping})
        return ok


def pin_hands(skeleton: dict, holding: dict, holds: dict, poses: dict, K: np.ndarray,
              kpts2d: dict, valid2d: dict, frame_size: tuple[int, int], wall_out: np.ndarray,
              *, lift_m: float = 0.03) -> int:
    """Put a gripping wrist where the hold is. Returns how many wrists moved.

    Filmed from behind, the depth under a wrist is the forearm in front of it:
    wrists read a median 38 cm out from the wall on the test capture, where a
    hand on a hold is 5-15 cm out. And because the phone looks *up* at the
    climber, a reading that is too near is also too *low* — the wrists came out
    below the shoulders in 68% of frames, and the highest point of the climb
    was a shoulder rather than the hand on the top hold.

    The depth cannot fix that; the contact can. While a hand is on a hold, the
    hand is at that hold — so its wrist is placed where the camera's ray
    through the wrist pixel meets the hold's own depth from the wall (a plane
    through the hold's centre, parallel to the wall, *lift_m* proud of it).
    The pixel still decides where across and up; only the depth is replaced.
    Frames where the hand is on nothing keep the depth reading.
    """
    moved = 0
    Kinv = np.linalg.inv(K)
    for frame, now in holding.items():
        body = skeleton.get(frame)
        pose = poses.get(frame)
        if body is None or pose is None or frame not in kpts2d:
            continue
        world, ok = body
        R, t = pose
        centre = (-R.T @ t).ravel()
        for limb, kpt in (("left_hand", 9), ("right_hand", 10)):
            hold = holds.get(now.get(limb))
            if hold is None or not valid2d[frame][kpt]:
                continue
            px = np.asarray(kpts2d[frame][kpt], dtype=float) * frame_size
            ray = R.T @ (Kinv @ np.array([px[0], px[1], 1.0]))
            denominator = ray @ wall_out
            if abs(denominator) < 1e-6:
                continue
            plane_point = hold.centre + wall_out * lift_m
            s = ((plane_point - centre) @ wall_out) / denominator
            if s <= 0:
                continue
            world[kpt] = centre + ray * s
            ok[kpt] = True
            moved += 1
    return moved
