"""The route: SAM 3.1 tracking the holds through the camera's own motion.

**SAM 3.1's ``track`` method** carries an identity forward through the clip
itself. One call takes the whole video and returns a `track_id` per instance per
sampled frame, so the association is the model's problem rather than ours, and
it is solved with appearance information that segmenting frames independently
would throw away. A clip longer than one call can hold is cut into segments
(:func:`segment_plan`), each tracked on its own and shifted back into the whole
clip's frame and track numbering (:func:`shift`, :func:`concat`).

This module turns the reply into :class:`Observation` s — one outline per track
per sampled frame, in normalized image coordinates — and stops there. Putting
those sightings in one place is :func:`src.lidar.lift`'s job: a hold is bolted
to the wall, so every sighting of one real hold has to land on the same spot in
the measured world, and a track whose sightings do not is SAM quietly handing
its id to a different hold partway through.

Once the holds are placed and flattened onto the wall (:func:`src.space.as_flat_holds`),
:func:`assign_ids` numbers them bottom to top. Each hold comes out as,
everything normalized to [0, 1] on the wall's own face-on view:

    {"bbox": [x, y, w, h], "polygon": [[x, y], ...], "score": float,
     "appearances": int, "tracks": [int, ...], "id": int, ...}
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


CONTENT_OBJECT = "vid.segment.masks"

# Bumped when the request or the response shape this module reads changes, so a
# cache written against the old one is a miss rather than a confusing crash.
CONTRACT = "sam31-track-canvas-v1"

# SAM 3.1's tracker holds one frame of memory per sample and the gateway caps
# the request here; see `_track_sampling.py` in the gateway. A longer clip is
# covered by sampling it more sparsely, not by asking for more frames.
TRACK_MAX_FRAMES = 128


# ── the wire format ──────────────────────────────────────────────────────────

@dataclass
class Observation:
    """One hold, seen in one frame, as a dense outline in that frame's coordinates."""

    frame: int
    track: int
    score: float
    bbox: list[float]                    # normalized frame coords, [x, y, w, h]
    polygon: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))


def request_track(client, *, model: str, video_b64: str, prompt: str,
                  skip_frames: int, max_frames: int, polygons: bool = True
                  ) -> tuple[dict, dict | None]:
    """One call: track every instance of *prompt* through the whole clip.

    ``video_skip_frames`` rather than ``video_fps`` because it says exactly what
    it does — the gateway would otherwise round a target rate into a stride
    anyway, and a stride is the thing that has to divide the clip into at most
    ``TRACK_MAX_FRAMES`` samples.

    The label maps come back as well as the polygons. They are worth the extra
    kilobytes: SAM's own `polys_xy` is simplified to a handful of vertices,
    while the label map is full resolution and yields a contour of hundreds.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "video_url",
                 "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}},
            ],
        }],
        response_format={"type": "json_object"},
        extra_body={"method": "track", "method_params": {
            "prompt": prompt, "video_skip_frames": skip_frames,
            "video_max_frames": max_frames, "mask_format": "png",
            "polygons": polygons}},
    )
    payload = json.loads(response.choices[0].message.content)
    usage = response.usage.model_dump() if response.usage else None
    return payload, usage


def unwrap(payload: dict) -> tuple[list[dict], list[dict]]:
    """``(items, frames)`` out of the envelope, with the payload tag checked.

    Checked rather than assumed: a wrong ``content.object`` means the request
    went somewhere other than video tracking, which is clearer said here than
    surfaced as a KeyError three functions later.
    """
    content = payload.get("content")
    if not isinstance(content, dict):
        raise RuntimeError(f"Unexpected response envelope: {json.dumps(payload)[:400]}")
    got = content.get("object")
    if got != CONTENT_OBJECT:
        raise RuntimeError(f"Expected a {CONTENT_OBJECT} payload, got {got!r}.")
    return content.get("items") or [], content.get("frames") or []


def _decode_label_map(mask: dict | None) -> np.ndarray | None:
    """A frame's label map as uint8: pixel = ``track_id``, 0 = no object."""
    if not isinstance(mask, dict) or mask.get("format") != "png":
        return None
    data = mask.get("data")
    if not isinstance(data, str):
        return None
    raw = base64.b64decode(data.split(",", 1)[-1])
    labels = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if labels is None:
        return None
    if labels.ndim == 3:   # a grayscale PNG some decoders hand back as 3 channels
        labels = labels[:, :, 0]
    return labels.astype(np.uint8)


def _contour(mask: np.ndarray) -> np.ndarray:
    """The largest external contour of a boolean mask, in that mask's pixels.

    Largest rather than all of them: a hold occluded mid-way by a forearm comes
    back as two blobs, and the union of two half-holds is a shape the hold never
    had. One piece of a hold in the right place beats a bridged pair.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((0, 2))
    return max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)


def observations(payload: dict, *, min_score: float) -> list[Observation]:
    """Every sighting, with the densest outline available for each.

    Each frame's label map is decoded once and shared by the tracks in it: one
    PNG carries every hold in that frame, so decoding per track would decode the
    same image as many times as there are holds on the wall.
    """
    items, frames = unwrap(payload)
    maps = {int(f["frame_id"]): _decode_label_map(f.get("mask")) for f in frames}

    out: list[Observation] = []
    for item in items:
        score = float(item.get("score") or 0.0)
        track = item.get("track_id")
        frame = item.get("frame_id")
        box = item.get("bbox_xywh")
        if track is None or frame is None or score < min_score:
            continue
        if not (isinstance(box, (list, tuple)) and len(box) == 4):
            continue

        labels = maps.get(int(frame))
        polygon = np.zeros((0, 2))
        if labels is not None:
            blob = labels == int(track)
            if blob.any():
                pts = _contour(blob)
                if len(pts) >= 3:
                    polygon = pts / [labels.shape[1], labels.shape[0]]

        # No label map: fall back to SAM's own simplified ring, then to the box.
        if len(polygon) < 3:
            rings = item.get("polys_xy") or []
            if rings:
                biggest = max(rings, key=len)
                if len(biggest) >= 3:
                    polygon = np.asarray(biggest, dtype=np.float64)

        out.append(Observation(frame=int(frame), track=int(track), score=score,
                               bbox=[float(v) for v in box], polygon=polygon))
    return out


# ── still-frame segmentation, for the floor ──────────────────────────────────
# The route is tracked, the floor is not: it is one static surface, it is never
# occluded for long, and tracking it would spend a second video call to learn
# something a handful of stills already agree on. These are the pieces
# `src.floor` needs to work that way.

CONTENT_OBJECT_IMAGE = "img.segment.masks"


# ── numbering ────────────────────────────────────────────────────────────────

def route_lean(holds: list[dict]) -> float:
    """``dx/dy`` over the hold centroids. Negative means the route leans right."""
    if len(holds) < 3:
        return 0.0
    xs = np.array([h["bbox"][0] + h["bbox"][2] / 2 for h in holds])
    ys = np.array([h["bbox"][1] + h["bbox"][3] / 2 for h in holds])
    variance = float(np.var(ys))
    if variance < 1e-9:
        return 0.0
    return float(np.cov(xs, ys)[0, 1] / variance)


def assign_ids(holds: list[dict], *, band: float = 0.025,
               direction: str = "auto", deadband: float = 0.05
               ) -> tuple[list[dict], str]:
    """Number the holds bottom-to-top on the wall, ordering a row along the lean.

    On the wall's own face-on view rather than in any one frame, which is the
    point: the numbering is a property of the wall and survives the camera
    moving, where a frame-space ordering would renumber the route every time
    the operator tilted.

    ``deadband`` is what keeps ``auto`` from being a coin toss. This route runs
    almost straight up — its lean measured -0.005 over thirteen holds and +0.018
    over fourteen — so a single hold recovered near the bottom was enough to
    flip the sign and renumber every row in the route. Below the deadband there
    is no lean to read and the fixed default is used instead, which is
    arbitrary but at least stable.
    """
    if direction == "auto":
        lean = route_lean(holds)
        direction = "rtl" if lean > deadband else "ltr"
    if direction not in ("ltr", "rtl"):
        raise ValueError(f"direction must be 'auto', 'ltr' or 'rtl' (got {direction!r})")
    sign = 1.0 if direction == "ltr" else -1.0

    def centre(hold):
        return (hold["bbox"][0] + hold["bbox"][2] / 2,
                hold["bbox"][1] + hold["bbox"][3] / 2)

    remaining = sorted(holds, key=lambda h: -centre(h)[1])
    ordered: list[dict] = []
    while remaining:
        base_y = centre(remaining[0])[1]
        row = [h for h in remaining if base_y - centre(h)[1] <= band]
        row.sort(key=lambda h: sign * centre(h)[0])
        ordered.extend(row)
        remaining = [h for h in remaining if h not in row]

    for i, hold in enumerate(ordered, start=1):
        hold["id"] = i
    holds[:] = ordered
    return holds, direction


# ── the wall, as a region ────────────────────────────────────────────────────

# ── per-frame geometry ───────────────────────────────────────────────────────

# ── cache ────────────────────────────────────────────────────────────────────

def cache_path(video: Path, cache_dir: Path, *, model: str, prompt: str,
               settings: dict) -> Path:
    """Where this exact route lives.

    Keyed on the clip's identity *and* every setting that shapes the result, so
    a cache hit can only mean "same clip, same prompt, same rules".
    """
    stat = video.stat()
    key = json.dumps({
        "video": video.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        "model": model, "prompt": prompt, "settings": settings, "contract": CONTRACT,
    }, sort_keys=True)
    return cache_dir / f"holds.{hashlib.sha256(key.encode()).hexdigest()[:12]}.json"


def save_cache(path: Path, payload: dict, usage: dict | None, *, stamp: str) -> None:
    """Store the gateway's raw reply, not the consolidation built from it.

    Deliberate: the reply costs a hundred seconds and three cents, while every
    threshold downstream of it is free to re-tune. Caching the finished route
    instead would make ``REJECT_RADIUS`` a setting you had to pay to change.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"created": stamp, "payload": payload, "usage": usage}))


def load_cache(path: Path):
    """Read a cached reply, or None if missing or unreadable."""
    if not path.is_file():
        return None
    try:
        blob = json.loads(path.read_text())
        return blob["payload"], blob.get("usage"), blob.get("created", "unknown")
    except (json.JSONDecodeError, AttributeError, KeyError, OSError):
        return None


# ── long clips: tracking in segments ─────────────────────────────────────────

def segment_plan(n_frames: int, stride: int, *, max_frames: int = TRACK_MAX_FRAMES
                 ) -> list[tuple[int, int]]:
    """``[(start, length), ...]`` covering the clip at the requested stride.

    Two different things push toward cutting a clip up, and they push the same
    way.

    The first is the gateway's cap. One `track` call keeps at most 128 samples,
    so a clip longer than ``128 * stride`` frames cannot be covered by one call
    at that stride — and coarsening the stride to fit is the wrong trade when
    what you wanted was a look every third of a second.

    The second is the tracker itself, and it is the reason this exists rather
    than being a nicety. SAM's `track` initialises on what it finds early and
    propagates forward; it does not re-detect. On a clip where the operator
    walks a full circle around the climber, everything leaves the frame around
    the halfway mark and **every track dies there** — on the clip that prompted
    this, all sixteen of them ended by frame 1488 of 2967, and the entire second
    half came back empty. Cutting the clip gives the tracker a fresh start each
    time, which is the only way it sees the far side at all.

    Segments are then stitched back together geometrically, by the same merge
    that already fuses a track SAM dropped and re-acquired: two ids in the same
    place are one hold, and whether they came from one call or two does not
    change that.
    """
    span = max_frames * stride
    if n_frames <= span:
        return [(0, n_frames)]
    count = -(-n_frames // span)
    length = -(-n_frames // count)
    return [(i * length, min(length, n_frames - i * length)) for i in range(count)]


def shift(payload: dict, *, frame_offset: int, track_offset: int) -> dict:
    """Renumber one segment's reply into the whole clip's frame and track space.

    A segment's `frame_id` counts from its own first frame and its `track_id`s
    start again at 1, so without this two segments would claim the same ids for
    different holds and the same frames for different moments.
    """
    items, frames = unwrap(payload)
    for item in items:
        if item.get("frame_id") is not None:
            item["frame_id"] = int(item["frame_id"]) + frame_offset
        if item.get("track_id") is not None:
            item["track_id"] = int(item["track_id"]) + track_offset
    for frame in frames:
        frame["frame_id"] = int(frame["frame_id"]) + frame_offset
    return payload


def concat(payloads: list[dict]) -> dict:
    """Fuse already-shifted segment replies into one, as if a single call made it."""
    items: list[dict] = []
    frames: list[dict] = []
    for payload in payloads:
        part_items, part_frames = unwrap(payload)
        items += part_items
        frames += part_frames
    head = dict(payloads[0])
    head["content"] = {"object": CONTENT_OBJECT, "items": items,
                       "frames": sorted(frames, key=lambda f: f["frame_id"])}
    return head
