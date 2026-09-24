"""The ground, so the clock can start when the climber leaves it.

A boulder starts when both feet are off the floor — not when both feet are on
holds. Those are different rules and only the first one is the gym's: a foot
smeared flat against the wall, on no hold at all, is a legitimate placement and
the climb has still begun.

Finding the floor is the same problem as finding the holds, so it is the same
model with a different prompt. What comes back is not a region but a *line*: the
top edge of the floor, per column, which is the height the feet have to clear.
Per column rather than one number because the wall-floor junction is not level
in a frame — on our test clips it drops 35px from one side to the other, which
a single threshold would get wrong by that much at one end or the other.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .holds import FrameMasks, decode_label_map, instance_mask


class Floor:
    """The floor's top edge, as a height per column. Normalized throughout."""

    def __init__(self, edge: np.ndarray):
        self.edge = np.asarray(edge, dtype=np.float32)   # one y per column

    @property
    def resolution(self) -> int:
        return len(self.edge)

    def top_at(self, x: float) -> float:
        """The floor's height at normalized x. 1.0 (off-frame) where unknown."""
        if not len(self.edge):
            return 1.0
        i = int(round(min(max(x, 0.0), 1.0) * (len(self.edge) - 1)))
        value = float(self.edge[i])
        return value if np.isfinite(value) else 1.0

    def is_clear(self, x: float, y: float, clearance: float,
                 frame: int | None = None) -> bool:
        """True when the point sits above the floor by at least *clearance*.

        *frame* is accepted and ignored: this edge is already in the canvas the
        whole clip shares, so it does not depend on which frame is asking.
        """
        return y < self.top_at(x) - clearance

    def as_list(self) -> list:
        return [None if not np.isfinite(v) else round(float(v), 5) for v in self.edge]

    @classmethod
    def from_list(cls, values: list) -> "Floor":
        return cls(np.array([np.nan if v is None else float(v) for v in values],
                            dtype=np.float32))


def _edge_from_mask(mask: np.ndarray, resolution: int) -> np.ndarray:
    """Top-most floor pixel per column, resampled to *resolution* columns."""
    h, w = mask.shape
    edge = np.full(w, np.nan, dtype=np.float32)
    # argmax on a boolean column gives the first True; any() guards the all-False
    # columns, where argmax would silently return 0 and put the floor at the top
    # of the frame.
    has = mask.any(axis=0)
    first = mask.argmax(axis=0)
    edge[has] = first[has] / h

    if w != resolution:
        xs = np.linspace(0, w - 1, resolution)
        edge = np.interp(xs, np.arange(w), edge)
    return edge


def consensus(per_frame: list[FrameMasks], *, resolution: int, min_score: float,
              min_area: float) -> Floor | None:
    """One floor edge from several frames, by per-column median.

    Median rather than mean, and per column rather than per frame, because the
    climber standing on the mat takes a bite out of it: in that frame those
    columns report the floor starting *below* where it does, or not at all. Over
    a dozen frames the climber is somewhere different each time, so for any given
    column most frames see the real junction and the median lands on it.
    """
    edges = []
    for result in per_frame:
        usable = [i for i in result.items
                  if float(i.get("score") or 0.0) >= min_score
                  and i.get("instance_id") is not None]
        if not usable:
            continue
        # The floor is the big low thing, so the biggest instance wins and a
        # stray patch of grey wall cannot stand in for it. `area` is the mask's
        # own share of the frame, which is the honest measure — a floor seen
        # edge-on fills a wide, shallow box far larger than the mat inside it —
        # with the box as the fallback for a reply that omits it.
        def size(instance):
            if instance.get("area") is not None:
                return float(instance["area"])
            _, _, w, h = instance["bbox_xywh"]
            return float(w * h)

        best = max(usable, key=size)
        if size(best) < min_area:
            continue
        mask = instance_mask(decode_label_map(result.mask), best)
        if mask is not None:
            edges.append(_edge_from_mask(mask, resolution))

    if not edges:
        return None

    stacked = np.vstack(edges)
    with np.errstate(all="ignore"):
        median = np.nanmedian(stacked, axis=0)

    # Columns nobody ever saw floor in: fill from their neighbours, so a foot
    # over a gap is still measured against something rather than treated as
    # airborne by default.
    finite = np.isfinite(median)
    if not finite.any():
        return None
    if not finite.all():
        median = np.interp(np.arange(len(median)), np.flatnonzero(finite), median[finite])
    return Floor(median)


def segment(client, frames, *, model: str, prompt: str, resolution: int,
            min_score: float, min_area: float, workers: int, console=None):
    """Segment the floor on the sampled stills. ``(per-frame results, usages)``.

    Stops short of the reduction, which is the caller's choice now: a fixed
    camera wants :func:`consensus`, a moving one :func:`consensus_canvas`.
    """
    from .holds import segment_frames

    per_frame, usages = segment_frames(
        client, frames, model=model, prompt=prompt, min_score=0.0,
        workers=workers, console=None)
    if console:
        counts = [len(f.items) for f in per_frame]
        console.print(f"  floor: {sum(1 for c in counts if c)}/{len(counts)} frames "
                      f"returned a mask")
    return per_frame, usages


def consensus_canvas(per_frame: list[FrameMasks], indices: list[int], camera, *,
                     resolution: int, min_score: float, min_area: float,
                     min_support: int = 3) -> Floor | None:
    """One floor edge in *canvas* columns, from stills shot at different angles.

    The frame-space version below assumes the floor sits at the same pixels in
    every sample, which a tripod guarantees and a pan does not: averaging those
    edges together handheld would smear the mat line across a third of the
    image. So each frame's edge is turned back into points, carried onto the
    canvas by that frame's homography, and the median is taken per *canvas*
    column instead.

    The result is a ground line attached to the wall rather than to the lens,
    which is what the start rule wanted all along — "both feet clear of the
    floor" is a fact about the gym.
    """
    columns: list[list[float]] = [[] for _ in range(resolution)]
    for index, result in zip(indices, per_frame):
        if index >= camera.n_frames:
            continue
        mask = _best_mask(result, min_score=min_score, min_area=min_area)
        if mask is None:
            continue
        edge = _edge_from_mask(mask, resolution)
        finite = np.isfinite(edge)
        if not finite.any():
            continue
        xs = np.linspace(0.0, 1.0, resolution)[finite]
        points = camera.to_canvas(np.stack([xs, edge[finite]], axis=1), index)
        for cx, cy in points:
            if 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0:
                columns[min(resolution - 1, int(cx * (resolution - 1)))].append(float(cy))

    # A canvas column only counts once a few different frames have put floor in
    # it. The outermost columns of the canvas are at the edge of what any camera
    # saw, so one frame's noisy last pixel would otherwise anchor the whole line
    # and hang a spur off each end of it.
    median = np.array([np.median(c) if len(c) >= min_support else np.nan
                       for c in columns], dtype=np.float32)
    finite = np.flatnonzero(np.isfinite(median))
    if not len(finite):
        return None

    # Gaps *between* observed columns are filled — a foot over one of those is
    # still measured against something. Columns beyond the ends are not: no
    # camera ever saw floor out there, and extending the last known value across
    # the rest of the canvas would draw a confident ground line through a part
    # of the gym nothing in this clip looked at.
    inner = np.arange(finite[0], finite[-1] + 1)
    filled = np.full(resolution, np.nan, dtype=np.float32)
    filled[inner] = np.interp(inner, finite, median[finite])
    return Floor(filled)


def _best_mask(result: FrameMasks, *, min_score: float, min_area: float):
    """The biggest instance in a frame, as a boolean mask, or None.

    The floor is the big low thing, so the biggest instance wins and a stray
    patch of grey wall cannot stand in for it. ``area`` is the mask's own share
    of the frame, which is the honest measure — a floor seen edge-on fills a
    wide, shallow box far larger than the mat inside it — with the box as the
    fallback for a reply that omits it.
    """
    usable = [i for i in result.items
              if float(i.get("score") or 0.0) >= min_score
              and i.get("instance_id") is not None]
    if not usable:
        return None

    def size(instance):
        if instance.get("area") is not None:
            return float(instance["area"])
        _, _, w, h = instance["bbox_xywh"]
        return float(w * h)

    best = max(usable, key=size)
    if size(best) < min_area:
        return None
    return instance_mask(decode_label_map(result.mask), best)


def draw(frame, floor: Floor, *, color, thickness: int, transform=None):
    """The floor line, so the rule the clock uses is visible rather than implied.

    *transform* maps the line's normalized coordinates into the panel's before
    it is drawn — the canvas `Fit` for the right panel, or the camera's own
    projection into this frame for the left.
    """
    if floor is None:
        return frame
    h, w = frame.shape[:2]
    xs = np.linspace(0.0, 1.0, max(2, w))
    ys = np.array([floor.top_at(x) for x in xs])
    keep = ys < 1.0
    if keep.sum() < 2:
        return frame
    pts = np.stack([xs[keep], ys[keep]], axis=1)
    if transform is not None:
        pts = np.asarray(transform(pts))
    points = [(int(round(px * w)), int(round(py * h))) for px, py in pts
              if -0.5 <= px <= 1.5 and -0.5 <= py <= 1.5]
    if len(points) > 1:
        cv2.polylines(frame, [np.array(points, dtype=np.int32)], False, color,
                      thickness, cv2.LINE_AA)
    return frame


# ── cache ────────────────────────────────────────────────────────────────────

def cache_path(video: Path, cache_dir: Path, *, model: str, prompt: str,
               settings: dict) -> Path:
    stat = video.stat()
    key = json.dumps({
        "video": video.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        "model": model, "prompt": prompt, "settings": settings,
    }, sort_keys=True)
    return cache_dir / f"floor.{hashlib.sha256(key.encode()).hexdigest()[:12]}.json"


def save_cache(path: Path, floor: Floor | None, usages: list, *, stamp: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "created": stamp,
        "edge": floor.as_list() if floor is not None else None,
        "usages": usages,
    }))


def load_cache(path: Path):
    """Read a cached floor edge, or None if missing or unreadable.

    A cached *miss* — the model found no floor — is a real result and is
    honoured, so a clip with no visible ground does not re-ask every run.
    """
    if not path.is_file():
        return None
    try:
        blob = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    edge = blob.get("edge")
    floor = Floor.from_list(edge) if edge else None
    return floor, blob.get("usages") or [], blob.get("created", "unknown")
