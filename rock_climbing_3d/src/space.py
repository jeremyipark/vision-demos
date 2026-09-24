"""The route in 3D: the geometry the climb is read against, and the right panel.

The depth gives the boulder's real shape, and the panel shows it: an overhang
looks like an overhang, the prow has two faces, and a hold a meter proud of the
wall reads as proud of it. A virtual camera looks at the fused wall and the
holds on it, and it moves — chasing the climber, or orbiting the route — which
is the cheapest way to make depth legible on a flat screen. A still 3D view is
just an unfamiliar 2D one.

## The frame the route is described in

The world axes are wherever the first camera pose put them, which is no use for
drawing. :class:`WallFrame` re-expresses the scene in axes the route defines:

* **up** is gravity, the normal of the floor fitted to the cloud
  (:func:`src.lidar.wall_axes`); when no floor is found, :func:`wall_frame`
  takes it from the holds' own spread instead.
* **out** is the wall's surface normal, so the default view faces the wall.
* **right** completes it.

Everything the panel draws is in those axes, which also makes the orbit mean
something: it swings around the route's own vertical.

## Depth, without a depth buffer

Holds are drawn back to front. A painter's algorithm is exact enough here
because holds do not interpenetrate — they are small, separated, convex-ish
patches on a surface — so sorting by distance from the virtual camera gets the
occlusions right at a fraction of the cost of rasterizing with a z-buffer.

The fused wall is drawn first and dimmed, which is what gives the panel its
sense of the boulder: the holds alone floating in black read as a
constellation, and the same holds against the shape of the prow read as a route.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class WallFrame:
    """The route's own axes, and the transform into them."""

    origin: np.ndarray        # (3,) the route's centre
    right: np.ndarray         # (3,) unit
    up: np.ndarray            # (3,) unit, the direction the climb runs
    out: np.ndarray           # (3,) unit, away from the wall toward the cameras
    extent: np.ndarray        # (3,) half-sizes of the route in these axes

    def to_wall(self, points: np.ndarray) -> np.ndarray:
        """World coordinates into route coordinates."""
        pts = np.asarray(points, dtype=float).reshape(-1, 3) - self.origin
        return np.stack([pts @ self.right, pts @ self.up, pts @ self.out], axis=1)

    def from_wall(self, points: np.ndarray) -> np.ndarray:
        """Route coordinates back into world coordinates."""
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        return (self.origin + pts[:, 0:1] * self.right + pts[:, 1:2] * self.up
                + pts[:, 2:3] * self.out)


def wall_frame(holds: dict, cloud: np.ndarray, eye: np.ndarray) -> WallFrame:
    """Axes fitted to the route: up the climb, out of the wall.

    ``up`` comes from the principal direction of the hold centres. A boulder
    problem is long and thin — that is what a route is — so its holds have one
    direction of large variance, and that direction is the climb. This is the
    fallback for when no floor plane was found; with one, gravity replaces it
    (:func:`src.lidar.wall_axes`).

    ``out`` is the dominant normal of the wall the holds sit on, resolved to
    point back at the cameras so the default view is the one a climber would
    take from the mat rather than from inside the rock.
    """
    centres = np.array([h.centre for h in holds.values()]) if holds else cloud
    if len(centres) < 3:
        centres = cloud
    origin = centres.mean(axis=0)
    centred = centres - origin

    # Out: the smallest principal direction of the holds' own spread is the
    # direction they vary in least, which for holds set on a wall is the wall's
    # normal. The cloud is the fallback when there are too few holds to ask.
    basis = cloud if len(cloud) >= len(centres) else centres
    out = np.linalg.svd(basis - basis.mean(axis=0), full_matrices=False)[2][-1]
    if out @ (eye - origin) < 0:
        out = -out
    out = out / np.linalg.norm(out)

    # Up: the holds' largest spread, with any component along `out` removed so
    # the three axes stay orthogonal.
    up = np.linalg.svd(centred, full_matrices=False)[2][0]
    up = up - (up @ out) * out
    if np.linalg.norm(up) < 1e-6:
        up = np.cross(out, [1.0, 0.0, 0.0])
    up = up / np.linalg.norm(up)
    # Point it the way the climb runs: from the first hold toward the last.
    if holds:
        ordered = [holds[k].centre for k in sorted(holds)]
        if (ordered[-1] - ordered[0]) @ up < 0:
            up = -up

    right = np.cross(up, out)
    right = right / np.linalg.norm(right)

    frame = WallFrame(origin=origin, right=right, up=up, out=out,
                      extent=np.ones(3))
    local = frame.to_wall(centres)
    frame.extent = np.maximum(np.abs(local).max(axis=0), 1e-6)
    return frame


@dataclass
class View:
    """A virtual camera looking at the route, in wall coordinates."""

    azimuth: float       # radians, 0 is face-on
    elevation: float     # radians, positive looks down from above
    distance: float
    size: tuple[int, int]
    focal: float
    offset: np.ndarray | None = None   # panel-pixel recentring, set by `fit_view`
    target: np.ndarray | None = None   # what it looks at, in wall coords; None = origin

    def matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """``(R, t)`` taking a wall-coordinate point into this camera."""
        ca, sa = np.cos(self.azimuth), np.sin(self.azimuth)
        ce, se = np.cos(self.elevation), np.sin(self.elevation)
        offset = np.array([self.distance * ce * sa,
                           self.distance * se,
                           self.distance * ce * ca])
        forward = -offset / np.linalg.norm(offset)
        eye = offset if self.target is None else offset + self.target
        world_up = np.array([0.0, 1.0, 0.0])
        # forward x up, not up x forward: image y runs *down*, so the camera's
        # second axis has to be the wall's down. The other order rolls the view
        # 180°, which went unnoticed while `up` came from the holds and its sign
        # was a coin toss anyway — and is plain upside down once it is gravity.
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        R = np.vstack([right, up, forward])
        return R, -R @ eye

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Wall points to panel pixels. ``(pixels, depth)``; depth <= 0 is behind."""
        R, t = self.matrix()
        cam = np.asarray(points, dtype=float).reshape(-1, 3) @ R.T + t
        depth = cam[:, 2]
        safe = np.where(np.abs(depth) < 1e-6, 1e-6, depth)
        width, height = self.size
        pix = np.stack([self.focal * cam[:, 0] / safe + width / 2.0,
                        self.focal * cam[:, 1] / safe + height / 2.0], axis=1)
        if self.offset is not None:
            pix = pix + self.offset
        return pix, depth


def fit_view(frame: WallFrame, size: tuple[int, int], *, azimuth: float,
             elevation: float, margin: float = 1.08, points=None) -> View:
    """A view of the route from this angle, framed to fill the panel.

    Two passes. The first places the camera far enough back that the route's
    bounding *sphere* fits, which is safe and wasteful: a route is a sheet on a
    wall, not a ball, so its sphere is mostly empty and the holds end up small in
    the middle of a large black panel. The second pass projects the route as it
    actually is from this angle, measures the box it really occupies, and pulls
    in until that box fills the panel.

    Doing it per orbit step is what keeps the route the same size throughout the
    sweep — otherwise the rotation reads as a zoom, which is the one thing the
    orbit is supposed not to look like.
    """
    width, height = size
    focal = 0.9 * max(width, height)
    radius = float(np.linalg.norm(frame.extent)) * 1.3
    half_fov = np.arctan(min(width, height) / 2.0 / focal)
    view = View(azimuth=azimuth, elevation=elevation,
                distance=radius / max(np.sin(half_fov), 1e-3),
                size=size, focal=focal)
    if points is None or not len(points):
        return view

    pix, depth = view.project(points)
    pix = pix[depth > 0]
    if len(pix) < 3:
        return view
    extent = np.maximum(pix.max(axis=0) - pix.min(axis=0), 1.0)
    # Scale about the panel centre, then re-centre on what the route occupies:
    # a route seen from the side sits off to one edge, and filling the panel
    # without recentring would just push it off the other one.
    grow = min(width / (extent[0] * margin), height / (extent[1] * margin))
    view.focal = focal * grow
    pix, depth = view.project(points)
    pix = pix[depth > 0]
    if len(pix):
        middle = (pix.min(axis=0) + pix.max(axis=0)) / 2.0
        view.offset = (np.array([width, height]) / 2.0 - middle)
    return view


# ── the route, as the rest of the pipeline needs to see it ───────────────────

class FrameGeometry:
    """Hold geometry per frame, by projecting the 3D route into each image.

    There is no single set of outlines that serves every frame — the camera
    moves, and flattening a prow onto one plane is the thing the 3D route
    exists to avoid. So the test happens where the evidence is: the hold is
    projected into the frame and the keypoint is compared to it there.

    That is not a workaround, it is the exact version. A hand at pixel *p* is on
    hold *h* if and only if *p* falls inside *h*'s projection in that frame, and
    no intermediate surface has to be invented to decide it.

    The margins come along: ``hold_unit`` is recomputed per frame from the
    *projected* holds, so "0.58 hold heights" means the same physical thing when
    the camera is close as when it is far.
    """

    def __init__(self, holds: dict, poses, K, frame_size: tuple[int, int], *,
                 bbox_margin: float, aspect: float):
        self.holds = holds
        self.poses = poses
        self.K = K
        self.frame_size = frame_size
        self.bbox_margin = bbox_margin
        self.aspect = aspect
        self._cache: dict[int, tuple] = {}
        self._ids = {h.id: h for h in holds.values()}

    def project(self, hold, frame: int) -> np.ndarray | None:
        """One hold's outline in frame-normalized coordinates, or None if behind."""
        pose = self.poses.get(frame)
        if pose is None:
            return None
        R, t = pose
        cam = hold.polygon @ R.T + t
        if np.any(cam[:, 2] <= 1e-6):
            return None
        pix = (self.K @ (cam / cam[:, 2:3]).T).T[:, :2]
        return pix / self.frame_size

    def at(self, frame: int):
        """``(HoldGeometry, hold_unit)`` for this frame. Cached; the clip re-reads."""
        if frame in self._cache:
            return self._cache[frame]
        from src.climb import HoldGeometry

        flat = []
        for hold in self.holds.values():
            polygon = self.project(hold, frame)
            if polygon is None or len(polygon) < 3:
                continue
            lo, hi = polygon.min(axis=0), polygon.max(axis=0)
            flat.append({"id": hold.id, "polygon": polygon.tolist(),
                         "bbox": [float(lo[0]), float(lo[1]),
                                  float(hi[0] - lo[0]), float(hi[1] - lo[1])]})
        unit = float(np.median([h["bbox"][3] for h in flat])) if flat else 0.02
        geometry = HoldGeometry(flat, bbox_margin=self.bbox_margin * unit,
                                aspect=self.aspect)
        self._cache[frame] = (geometry, max(unit, 1e-4))
        return self._cache[frame]


class GroundPlane:
    """The wall-floor junction in 3D, projected into whichever frame is asking.

    No segmentation: the junction is geometry. The ground is the floor plane
    fitted to the cloud (or, failing that, a plane across the bottom of it),
    the wall is the route's own plane, and the line where they meet is the
    wall-floor junction. That line is fixed in the world, so it can be
    projected into every frame and the clock stops depending on where the
    operator was standing.

    ``is_clear`` keeps :class:`src.floor.Floor`'s contract exactly, including
    treating an unmeasurable column as clear. It is the same trade: the start
    rule also requires a hand on a hold, so "no floor visible here" cannot start
    the clock on its own.

    The ground height is a low percentile of the cloud rather than its minimum,
    because a sparse reconstruction always keeps a few points triangulated below
    everything else and a minimum would be those points rather than the floor.
    """

    def __init__(self, axes: WallFrame, cloud: np.ndarray, poses, K,
                 frame_size: tuple[int, int], *, percentile: float = 2.0,
                 samples: int = 64):
        self.axes = axes
        self.poses = poses
        self.K = K
        self.frame_size = frame_size
        local = axes.to_wall(cloud)
        # Wide enough that the junction crosses the whole frame rather than
        # stopping under the route and leaving the rest of the mat unmeasured.
        reach = 4.0 * float(max(axes.extent[0], 1e-6))
        self._offsets = np.linspace(-reach, reach, samples)

        # Which end of the up-axis is the floor. `wall_frame` orients `up` from
        # the first hold toward the last, and at this point the holds still
        # carry SAM's track ids rather than the climb's numbering, so that sign
        # is not reliable enough to hang the clock on. The camera is: whichever
        # end projects *lower in the image* is the ground, because the operator
        # filmed the wall the right way up.
        candidates = [float(np.percentile(local[:, 1], percentile)),
                      float(np.percentile(local[:, 1], 100.0 - percentile))]
        self.height = max(candidates, key=lambda h: self._mean_image_y(h))
        self.line = self._line_at(self.height)
        self._cache: dict[int, np.ndarray | None] = {}

    def _line_at(self, height: float) -> np.ndarray:
        return self.axes.from_wall(np.stack(
            [self._offsets, np.full(len(self._offsets), height),
             np.zeros(len(self._offsets))], axis=1))

    def _mean_image_y(self, height: float, probes: int = 12) -> float:
        """Mean image height of that candidate junction, over a few frames."""
        frames = sorted(self.poses)
        if not frames:
            return 0.0
        line = self._line_at(height)
        seen = []
        for frame in frames[::max(1, len(frames) // probes)]:
            R, t = self.poses[frame]
            cam = line @ R.T + t
            front = cam[:, 2] > 1e-6
            if front.sum() < 2:
                continue
            pix = (self.K @ (cam[front] / cam[front][:, 2:3]).T).T[:, :2]
            seen.append(float(np.mean(pix[:, 1] / self.frame_size[1])))
        return float(np.mean(seen)) if seen else 0.0

    def polyline(self, frame: int) -> np.ndarray | None:
        """The junction in this frame, as normalized points, or None."""
        if frame in self._cache:
            return self._cache[frame]
        pose = self.poses.get(frame)
        result = None
        if pose is not None:
            R, t = pose
            cam = self.line @ R.T + t
            front = cam[:, 2] > 1e-6
            if front.sum() >= 2:
                pix = (self.K @ (cam[front] / cam[front][:, 2:3]).T).T[:, :2]
                result = pix / self.frame_size
        self._cache[frame] = result
        return result

    def top_at(self, x: float, frame: int) -> float:
        """The junction's height at normalized *x*. 1.0 where it is unknown.

        Every segment that crosses this column is interpolated and the topmost
        answer wins, rather than sorting the polyline by *x* and interpolating
        once. The projection is not monotonic in general — a junction seen from
        an oblique angle can double back, and near edge-on it is almost
        vertical — and sorting would quietly interpolate across the fold.
        Taking the topmost crossing is reading the first floor pixel down each
        column.
        """
        points = self.polyline(frame)
        if points is None:
            return 1.0
        xs, ys = points[:, 0], points[:, 1]
        best = None
        for i in range(len(points) - 1):
            x0, x1 = xs[i], xs[i + 1]
            if x0 == x1:
                if abs(x0 - x) > 1e-6:
                    continue
                candidate = min(ys[i], ys[i + 1])
            elif min(x0, x1) <= x <= max(x0, x1):
                candidate = float(np.interp(x, [x0, x1] if x0 < x1 else [x1, x0],
                                            [ys[i], ys[i + 1]] if x0 < x1
                                            else [ys[i + 1], ys[i]]))
            else:
                continue
            best = candidate if best is None else min(best, candidate)
        return 1.0 if best is None else float(best)

    def is_clear(self, x: float, y: float, clearance: float,
                 frame: int | None = None) -> bool:
        """True when the point sits above the junction by at least *clearance*."""
        if frame is None:
            return True
        return y < self.top_at(x, frame) - clearance


class MultiModelGround:
    """One :class:`GroundPlane` per model, routed by frame.

    Each model fits its own ground because their scales and origins are
    unrelated; a single plane across all of them would be meaningless.
    """

    def __init__(self, grounds: dict[int, GroundPlane], frame_model: dict[int, int]):
        self.grounds = grounds
        self.frame_model = frame_model

    def at(self, frame: int) -> GroundPlane | None:
        return self.grounds.get(self.frame_model.get(frame, -1))

    def polyline(self, frame: int) -> np.ndarray | None:
        ground = self.at(frame)
        return ground.polyline(frame) if ground is not None else None

    def top_at(self, x: float, frame: int) -> float:
        ground = self.at(frame)
        return ground.top_at(x, frame) if ground is not None else 1.0

    def is_clear(self, x: float, y: float, clearance: float,
                 frame: int | None = None) -> bool:
        if frame is None:
            return True
        ground = self.at(frame)
        return ground.is_clear(x, y, clearance, frame) if ground is not None else True


class MultiModelFrameGeometry:
    """Frame geometry routed to independent, unaligned reconstructions.

    Each entry in ``models`` owns ``recon``, ``poses`` and ``holds`` in one
    coordinate system. ``frame_model`` is the firewall between those systems:
    no pose is ever used to project a hold or cloud from another model.
    """

    def __init__(self, models: list[dict], frame_model: dict[int, int],
                 frame_size: tuple[int, int], *, bbox_margin: float,
                 aspect: float):
        self.models = models
        self.frame_model = frame_model
        self.frame_size = frame_size
        self.bbox_margin = bbox_margin
        self.aspect = aspect
        self._cache: dict[int, tuple] = {}

    def model_at(self, frame: int) -> dict | None:
        index = self.frame_model.get(frame)
        return self.models[index] if index is not None else None

    def holds_at(self, frame: int) -> dict:
        model = self.model_at(frame)
        return model.get("holds", {}) if model is not None else {}

    def project(self, hold, frame: int) -> np.ndarray | None:
        model = self.model_at(frame)
        if model is None or hold not in model.get("holds", {}).values():
            return None
        pose = model["poses"].get(frame)
        if pose is None:
            return None
        R, t = pose
        cam = hold.polygon @ R.T + t
        if np.any(cam[:, 2] <= 1e-6):
            return None
        pix = (model["recon"].K @ (cam / cam[:, 2:3]).T).T[:, :2]
        return pix / self.frame_size

    def at(self, frame: int):
        if frame in self._cache:
            return self._cache[frame]
        from src.climb import HoldGeometry

        flat = []
        for hold in self.holds_at(frame).values():
            polygon = self.project(hold, frame)
            if polygon is None or len(polygon) < 3:
                continue
            lo, hi = polygon.min(axis=0), polygon.max(axis=0)
            flat.append({"id": hold.id, "polygon": polygon.tolist(),
                         "bbox": [float(lo[0]), float(lo[1]),
                                  float(hi[0] - lo[0]), float(hi[1] - lo[1])]})
        unit = float(np.median([h["bbox"][3] for h in flat])) if flat else 0.02
        geometry = HoldGeometry(flat, bbox_margin=self.bbox_margin * unit,
                                aspect=self.aspect)
        self._cache[frame] = (geometry, max(unit, 1e-4))
        return self._cache[frame]


def as_flat_holds(holds: dict, frame: WallFrame) -> list[dict]:
    """The 3D route as flat dicts on the wall plane, for reports and artifacts.

    `holds.json`, the CSVs and the comparison all speak the flat dialect, and the
    natural flattening of a 3D route is its own wall frame seen face-on — which
    is what the panel shows at azimuth zero. The depth is not thrown away, it is
    recorded alongside as ``depth``, so a reader can see which holds stand proud.
    """
    if not holds:
        return []
    corners = np.vstack([frame.to_wall(h.polygon) for h in holds.values()])
    lo, hi = corners[:, :2].min(axis=0), corners[:, :2].max(axis=0)
    span = np.maximum(hi - lo, 1e-9)

    out = []
    for hold_id in sorted(holds):
        hold = holds[hold_id]
        local = frame.to_wall(hold.polygon)
        # y up in the world is y down in an image, so the vertical axis flips.
        flat = np.stack([(local[:, 0] - lo[0]) / span[0],
                         1.0 - (local[:, 1] - lo[1]) / span[1]], axis=1)
        p0, p1 = flat.min(axis=0), flat.max(axis=0)
        centre = frame.to_wall(hold.centre.reshape(1, 3))[0]
        out.append({
            "bbox": [float(p0[0]), float(p0[1]),
                     float(p1[0] - p0[0]), float(p1[1] - p0[1])],
            "polygon": flat.tolist(),
            "score": 1.0,
            "appearances": hold.n_views,
            "appearance_fraction": 1.0,
            "drift_px": 0.0,
            "tracks": [hold_id, *hold.source.get("merged", [])],
            "frames": hold.source.get("views", []),
            "depth": round(float(centre[2]), 5),
            "triangulation_angle_deg": round(hold.angle_deg, 2),
        })
    return out


def on_wall_scorer(geometry: "FrameGeometry"):
    """A ``(bbox, frame) -> overlap`` test for picking the climber, in 3D.

    `pose.pick_track` decides which body is the climber by how much of it sits on
    the route. There is no shared plane to compare them on, so the comparison
    happens in the image: the holds are projected into that frame and the
    question becomes how many of them the box covers.
    """
    def overlap(bbox, frame: int) -> float:
        if hasattr(geometry, "model_at"):
            model = geometry.model_at(frame)
            if model is None:
                return 0.0
            pose = model["poses"].get(frame)
        else:
            pose = geometry.poses.get(frame)
        if pose is None:
            return 0.0
        x, y, w, h = bbox
        hits = 0
        total = 0
        holds = (geometry.holds_at(frame) if hasattr(geometry, "holds_at")
                 else geometry.holds)
        for hold in holds.values():
            polygon = geometry.project(hold, frame)
            if polygon is None:
                continue
            total += 1
            cx, cy = polygon.mean(axis=0)
            if x <= cx <= x + w and y <= cy <= y + h:
                hits += 1
        return hits / total if total else 0.0
    return overlap


# ── the dense wall, when there is one ────────────────────────────────────────

def splat(panel: np.ndarray, local: np.ndarray, colours: np.ndarray, view: View, *,
          size: int = 2) -> np.ndarray:
    """Draw a coloured cloud, far to near, and return the panel's depth buffer.

    A fused LiDAR wall is hundreds of thousands of points, so this is
    vectorized: project everything, sort far to near, and let each square of
    *size* pixels be overwritten by whatever is nearer — a painter's algorithm
    at the resolution of a point, which is enough to make the wall read as a
    solid surface with the volumes standing proud of it.

    The depth buffer comes back so what is drawn next (holds, the climber) can
    be tested against the wall rather than drawn through it.
    """
    height, width = panel.shape[:2]
    zbuffer = np.full((height, width), np.inf, dtype=np.float32)
    if not len(local):
        return zbuffer
    pix, depth = view.project(local)
    x = np.floor(pix[:, 0]).astype(np.int64)
    y = np.floor(pix[:, 1]).astype(np.int64)
    ok = (depth > 0) & (x >= 0) & (y >= 0) & (x < width) & (y < height)
    x, y, depth, cols = x[ok], y[ok], depth[ok], colours[ok]
    order = np.argsort(-depth)
    x, y, depth, cols = x[order], y[order], depth[order], cols[order]
    size = max(1, int(size))
    for dy in range(size):
        for dx in range(size):
            xs = np.minimum(x + dx, width - 1)
            ys = np.minimum(y + dy, height - 1)
            # Fancy assignment keeps the last write per pixel, and the last is
            # the nearest: that is the whole occlusion test.
            panel[ys, xs] = cols
            zbuffer[ys, xs] = depth
    return zbuffer


def draw_dense(panel: np.ndarray, frame: WallFrame, holds: dict, local: np.ndarray,
               colours: np.ndarray, view: View, *, activated: dict, effective: int,
               colors: dict, cfg, point_size: int = 2) -> tuple[dict, np.ndarray]:
    """The wall as a coloured surface, with the route drawn onto it.

    ``(centroids, zbuffer)``. The same contract as :func:`draw`, so the labels,
    pips and midline the shared renderer adds go on top unchanged.

    The cloud is dimmed before it is drawn (the caller passes it dimmed, once):
    the wall is the context, the route is the subject, and at full colour every
    other hold in the gym competes with it. The route's own holds are then
    outlined over the surface in the route colour, lit ones filled, so they sit
    *on* the reconstruction rather than floating in front of it.
    """
    zbuffer = splat(panel, local, colours, view, size=point_size)

    entries = []
    for hold_id, hold in holds.items():
        pix, depth = view.project(frame.to_wall(hold.polygon))
        if np.any(depth <= 0):
            continue
        entries.append((float(depth.mean()), hold_id, pix,
                        frame.to_wall(hold.centre.reshape(1, 3))))
    entries.sort(key=lambda e: -e[0])

    centroids: dict[int, tuple[int, int]] = {}
    for _, hold_id, pix, centre_local in entries:
        poly = np.round(pix).astype(np.int32)
        lit = hold_id in activated and activated[hold_id] <= effective
        colour = colors["active"] if lit else colors["inactive_dense"]
        overlay = panel.copy()
        cv2.fillPoly(overlay, [poly], colour)
        alpha = cfg.HOLD_ACTIVE_FILL_ALPHA if lit else 0.25
        cv2.addWeighted(overlay, alpha, panel, 1.0 - alpha, 0, panel)
        cv2.polylines(panel, [poly], True, colour,
                      max(1, cfg.HOLD_ACTIVE_THICK if lit else cfg.HOLD_INACTIVE_THICK),
                      cv2.LINE_AA)
        centre_pix, _ = view.project(centre_local)
        centroids[hold_id] = (int(centre_pix[0, 0]), int(centre_pix[0, 1]))
    return centroids, zbuffer


def orbit_lidar(cfg, progress: float) -> tuple[float, float]:
    """The LiDAR orbit: a wider swing, from a little above, eased at the ends."""
    span = np.radians(cfg.SPACE_LIDAR_ORBIT_DEGREES)
    eased = 0.5 - 0.5 * np.cos(2 * np.pi * min(max(progress, 0.0), 1.0))
    return (-span / 2 + span * eased, np.radians(cfg.SPACE_LIDAR_ELEVATION_DEGREES))


def chase_path(scene: dict, frames: list[int], cfg, *, hold_from: int | None = None
               ) -> dict[int, tuple[np.ndarray, float, float]]:
    """A virtual camera that follows the climber, steadily. ``{frame: (target, az, el)}``.

    The orbit swings on its own schedule, unrelated to where the operator was,
    so half the time it looks at the wall from the side the phone never went
    to, and it frames the whole route rather than the person on it. This keeps
    the two tied together:

    * **what it looks at** is the climber's torso in 3D (from the depth under the
      joints), so the body stays in the middle of the panel;
    * **where it looks from** is the same side of the wall as the real phone —
      its azimuth around the torso, scaled by ``SPACE_CHASE_AZIMUTH_FOLLOW`` and
      clamped, so a camera that walked left shows the wall from the left;
    * **both are smoothed** over ``SPACE_CHASE_SMOOTH_SECONDS``. A handheld
      phone's small jitter and the body's twitch from move to move are exactly
      what a chase camera should *not* reproduce; it should drift after them.

    Frames with no torso (before the climber is found, or a dropped pose) take
    the nearest one's, so the camera holds rather than jumps. From *hold_from*
    on — the top-out — it stops following altogether: the climber jumps down to
    the mat, and the finished route is what the panel is there to show.
    """
    from scipy.ndimage import gaussian_filter1d

    axes = scene["axes"]
    skeleton = scene.get("skeleton", {})
    torso_idx = list(cfg.TORSO_KP_INDICES)
    frames = list(frames)
    target = np.full((len(frames), 3), np.nan)
    azimuth = np.full(len(frames), np.nan)
    for i, frame in enumerate(frames):
        if hold_from is not None and frame > hold_from:
            continue
        body = skeleton.get(frame)
        if body is not None:
            world, ok = body
            torso = [k for k in torso_idx if ok[k]]
            if len(torso) >= 2:
                target[i] = axes.to_wall(world[torso].mean(axis=0).reshape(1, 3))[0]
        pose = scene["poses"].get(frame)
        if pose is not None and np.isfinite(target[i]).all():
            R, t = pose
            eye = axes.to_wall((-R.T @ t).reshape(1, 3))[0] - target[i]
            azimuth[i] = np.arctan2(eye[0], eye[2])

    def fill(values):
        values = np.asarray(values, dtype=float)
        flat = values.reshape(len(values), -1)
        index = np.arange(len(values))
        for column in range(flat.shape[1]):
            known = np.isfinite(flat[:, column])
            if not known.any():
                flat[:, column] = 0.0
            else:
                flat[:, column] = np.interp(index, index[known], flat[known, column])
        return flat.reshape(values.shape)

    if not np.isfinite(target).any():
        # Nobody to follow: look at the route's middle instead.
        target[:] = 0.0
    target = fill(target)
    # Keep the camera's attention on the wall, not a foot in front of it: the
    # torso is proud of the holds by its own depth, and aiming at it exactly
    # puts the holds the climber is reaching for slightly out of centre.
    target[:, 2] *= 0.5
    azimuth = fill(azimuth)

    sigma = max(1.0, cfg.SPACE_CHASE_SMOOTH_SECONDS * 30.0)
    target = gaussian_filter1d(target, sigma, axis=0, mode="nearest")
    azimuth = gaussian_filter1d(np.unwrap(azimuth), sigma * 1.5, mode="nearest")
    limit = np.radians(cfg.SPACE_CHASE_MAX_AZIMUTH)
    azimuth = np.clip(azimuth * cfg.SPACE_CHASE_AZIMUTH_FOLLOW, -limit, limit)
    elevation = np.radians(cfg.SPACE_CHASE_ELEVATION)
    return {f: (target[i], float(azimuth[i]), elevation) for i, f in enumerate(frames)}


def chase_view(size: tuple[int, int], target: np.ndarray, azimuth: float,
               elevation: float, *, span_m: float) -> View:
    """The chase camera for one frame, framing *span_m* meters of wall top to bottom."""
    width, height = size
    focal = 1.0 * height
    return View(azimuth=azimuth, elevation=elevation, distance=span_m * focal / height,
                size=size, focal=focal, target=np.asarray(target, dtype=float))


def body_travel(path: list, axes: WallFrame, *, fps: float, sigma_s: float = 0.5,
                start: int | None = None, end: int | None = None) -> dict | None:
    """How far the climber's body moved, in meters, along the wall's own axes.

    *path* is ``[(frame, world point), …]`` — the torso centre from the depth.
    Two answers, because "how far did I travel" means two things:

    * **net** — where the torso finished relative to where it started: straight
      up (along gravity) and straight across (along the wall).
    * **travelled** — how much it actually moved in each direction on the way,
      counting every sway and step: the sum of the up/down and left/right
      movement. A route that zig-zags travels much further across than its net.

    The path is smoothed over *sigma_s* first. The torso's depth carries a
    centimeter or two of noise every frame, and summed over 500 frames that
    alone would add meters of "travel" that never happened.
    """
    from scipy.ndimage import gaussian_filter1d

    kept = [(f, p) for f, p in path
            if (start is None or f >= start) and (end is None or f <= end)]
    if len(kept) < 5:
        return None
    local = axes.to_wall(np.array([p for _, p in kept]))
    local = gaussian_filter1d(local, max(1.0, sigma_s * fps), axis=0, mode="nearest")
    step = np.diff(local, axis=0)
    net = local[-1] - local[0]
    return {
        "net_up_m": float(net[1]),
        "net_across_m": float(net[0]),
        "travel_vertical_m": float(np.abs(step[:, 1]).sum()),
        "travel_horizontal_m": float(np.abs(step[:, 0]).sum()),
        "path_m": float(np.linalg.norm(step, axis=1).sum()),
        "frames": len(kept),
        # Where the smoothed torso began and ended, in wall coordinates — the
        # two ends of the rise-and-run triangle the finale draws.
        "start_local": local[0].tolist(),
        "end_local": local[-1].tolist(),
    }


# Joints that count toward the body's reach: everything but the face, whose
# keypoints sit on the head the shoulders already bound.
REACH_JOINTS = tuple(range(5, 17))
JOINT_LIMB = {9: "left_hand", 10: "right_hand", 15: "left_foot", 16: "right_foot"}


def body_extent(skeleton: dict, axes: WallFrame, *, start: int | None, end: int | None,
                floor_height: float = 0.0, despike: int = 7) -> dict | None:
    """How much of the wall the whole body covered: its reach, in meters.

    The torso's own travel undersells a climb: the torso never goes where the
    feet start or where the hands finish. This asks of every tracked joint at
    once — the lowest any of them got (a foot, at the start) and the highest (a
    hand, at the top), the leftmost and the rightmost — over the climb itself,
    from pulling on to topping out.

    One bad depth reading on one frame would otherwise *be* the answer, since
    an extreme is a single sample out of thousands. So each joint's track is
    median-filtered over *despike* frames first: a real reach lasts longer
    than that; a spike does not.

    Heights are also returned above the floor, via *floor_height* — the floor's
    height along the wall frame's up axis.
    """
    from scipy.ndimage import median_filter

    frames = sorted(f for f in skeleton
                    if (start is None or f >= start) and (end is None or f <= end))
    if len(frames) < despike:
        return None
    joints = list(REACH_JOINTS)
    track = np.full((len(frames), len(joints), 3), np.nan)
    for i, f in enumerate(frames):
        world, ok = skeleton[f]
        local = axes.to_wall(world[joints])
        local[~np.asarray(ok)[joints]] = np.nan
        track[i] = local
    # Median over time per joint, ignoring gaps: fill a gap with the joint's own
    # nearest reading for the filter, then put the gap back.
    clean = track.copy()
    for j in range(len(joints)):
        for axis in range(3):
            column = track[:, j, axis]
            known = np.isfinite(column)
            if known.sum() < despike:
                clean[:, j, axis] = np.nan
                continue
            index = np.arange(len(column))
            filled = np.interp(index, index[known], column[known])
            smoothed = median_filter(filled, size=despike, mode="nearest")
            smoothed[~known] = np.nan
            clean[:, j, axis] = smoothed

    def extreme(axis, pick):
        values = clean[:, :, axis]
        if not np.isfinite(values).any():
            return None
        flat = np.nanargmax(values) if pick == "max" else np.nanargmin(values)
        i, j = np.unravel_index(flat, values.shape)
        return {"value": float(values[i, j]), "frame": int(frames[i]),
                "joint": joints[j], "limb": JOINT_LIMB.get(joints[j]),
                "point": clean[i, j].tolist()}

    top, bottom = extreme(1, "max"), extreme(1, "min")
    left, right = extreme(0, "min"), extreme(0, "max")
    if None in (top, bottom, left, right):
        return None
    return {"top": top, "bottom": bottom, "left": left, "right": right,
            "vertical_m": top["value"] - bottom["value"],
            "horizontal_m": right["value"] - left["value"],
            "top_above_floor_m": top["value"] - floor_height,
            "bottom_above_floor_m": bottom["value"] - floor_height}
