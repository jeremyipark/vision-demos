"""The route: SAM 3.1 tracking the holds through the camera's own motion.

The tripod version of this module segmented a dozen unrelated frames and glued
the results together by IoU — two boxes in the same place across two calls were
taken to be the same hold. Handheld, "the same place" does not survive contact
with the footage: the camera pans, tilts and zooms, so a hold is somewhere
different in every frame and position-based matching has nothing to hold onto.

Two things replace it, and they are useful because they fail differently.

**SAM 3.1's ``track`` method** carries an identity forward through the clip
itself. One call takes the whole video and returns a `track_id` per instance per
sampled frame, so the association is the model's problem rather than ours, and
it is solved with the appearance information we threw away by treating frames as
independent. On this clip, 118 sampled frames and 16 stable green holds.

**The camera track** (:mod:`src.camera`) then puts every one of those detections
onto a single canvas, where the wall is nailed down again. And that is what
turns the tracker's output into something checkable, because a hold is bolted to
a wall: every sighting of one real hold has to land on the same canvas pixels.

Measured, most of them do — sub-two-pixel median scatter across all 118 frames.
But a few tracks jump a hundred pixels, which is the tracker quietly handing a
`track_id` to a different hold partway through, and the *only* reason we can see
it is that the canvas gives the wall a fixed frame to be judged against. So the
two mechanisms check each other: SAM supplies an identity through motion the
canvas cannot follow, the canvas supplies a geometry SAM's identity has to obey.

## The consolidation, then, is per track and robust

Each track's sightings are warped to the canvas, the spatial median is taken, and
sightings too far from it are dropped as identity switches before anything is
voted on. What survives is pixel-voted into one canvas polygon per hold, the
same way the tripod version voted — the difference is only that the vote now
happens in canvas space, and that the outliers were thrown out on geometric
grounds first.

## Polygons, warped exactly

A homography maps straight lines to straight lines, so warping a polygon's
*vertices* is exact — there is no resampling and no blur, unlike warping a
raster. Every mask is therefore turned into a dense contour in its own frame and
carried to the canvas as points.

## Why the canonical polygon is also what touch detection uses

Once a hold has a canvas polygon it gets projected *back* into each frame for
drawing and for the touch tests, rather than the tests using SAM's own mask for
that frame. That sounds like a detour and is actually the robust choice: the
frames where SAM loses a hold are overwhelmingly the frames where the climber is
covering it, which are exactly the frames in which a touch is happening. The
canvas polygon is still there when the evidence for it is hidden.

Each hold comes out as, everything normalized to [0, 1] on the *canvas*:

    {"bbox": [x, y, w, h], "polygon": [[x, y], ...], "score": float,
     "appearances": int, "appearance_fraction": float, "drift_px": float,
     "tracks": [int, ...], "frames": [int, ...], "id": int}
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from src.camera import Track

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


@dataclass
class FrameMasks:
    """One still frame's segmentation: the instance rows and their label map."""

    items: list[dict] = field(default_factory=list)
    mask: dict | None = None      # the label map, still PNG-encoded


def decode_label_map(mask: dict | None) -> np.ndarray | None:
    """A label map as uint8: pixel = ``instance_id``, 0 = no object."""
    return _decode_label_map(mask)


def instance_mask(labels: np.ndarray | None, item: dict) -> np.ndarray | None:
    """The boolean mask of one instance, cut out of its frame's label map."""
    if labels is None:
        return None
    instance_id = item.get("instance_id")
    if instance_id is None:
        return None
    mask = labels == int(instance_id)
    return mask if mask.any() else None


def sample_frames(video: Path, n: int) -> list[tuple[int, np.ndarray]]:
    """*n* evenly-spaced frames, as BGR arrays, with their source indices.

    Inset from both ends: the first and last frames of a handheld start and stop
    are the ones most likely to be blurred or half-covered.
    """
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        capture.release()
        raise RuntimeError(f"{video} reports no frames")

    indices = np.linspace(total * 0.05, total * 0.95, num=min(n, total)).astype(int)
    out: list[tuple[int, np.ndarray]] = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if ok:
            out.append((int(index), frame))
    capture.release()
    return out


def encode_jpeg(frame: np.ndarray, quality: int = 92) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed on a sampled frame")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def segment_frames(client, frames, *, model: str, prompt: str, min_score: float,
                   workers: int, console=None) -> tuple[list[FrameMasks], list[dict]]:
    """Segment every sampled still, in parallel. ``(per-frame results, usages)``.

    The calls are independent, so they go out together. A frame whose call fails
    contributes nothing rather than failing the run: the consensus downstream is
    built to work from however many samples actually came back.
    """
    from concurrent.futures import ThreadPoolExecutor

    def one(entry):
        index, frame = entry
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": [{
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encode_jpeg(frame)}"},
            }]}],
            response_format={"type": "json_object"},
            extra_body={"method": "segment",
                        "method_params": {"prompt": prompt, "mask_format": "png"}},
        )
        payload = json.loads(response.choices[0].message.content)
        content = payload.get("content") or {}
        if content.get("object") != CONTENT_OBJECT_IMAGE:
            raise RuntimeError(f"Expected {CONTENT_OBJECT_IMAGE}, got {content.get('object')!r}.")
        result = FrameMasks(
            items=[i for i in (content.get("items") or [])
                   if float(i.get("score") or 0.0) >= min_score],
            mask=content.get("mask"))
        return index, result, (response.usage.model_dump() if response.usage else None)

    results: list[tuple[int, FrameMasks, dict | None]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for index, result, usage in pool.map(one, frames):
            results.append((index, result, usage))
            if console:
                console.print(f"  frame [dim]{index}[/]: {len(result.items)} instances")

    results.sort(key=lambda r: r[0])
    return [r[1] for r in results], [r[2] for r in results if r[2]]


# ── onto the canvas ──────────────────────────────────────────────────────────

def _bbox(points: np.ndarray) -> list[float]:
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    return [float(lo[0]), float(lo[1]), float(hi[0] - lo[0]), float(hi[1] - lo[1])]


def project(obs: list[Observation], camera: Track) -> dict[int, list[tuple[Observation, np.ndarray]]]:
    """Warp every sighting onto the canvas, grouped by ``track_id``.

    A sighting with no usable outline is carried by its box corners instead of
    being dropped — under a homography the box is no longer a box, so all four
    corners go through rather than two.
    """
    grouped: dict[int, list[tuple[Observation, np.ndarray]]] = {}
    for item in obs:
        if item.frame >= camera.n_frames:
            continue
        points = item.polygon
        if len(points) < 3:
            x, y, w, h = item.bbox
            points = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
        grouped.setdefault(item.track, []).append(
            (item, camera.to_canvas(points, item.frame)))
    return grouped


def _vote(polygons: list[np.ndarray], *, raster_size: int, vote_fraction: float,
          min_area_px: int, fill_holes: bool):
    """Pixel-vote a hold's canvas outlines into one shape. ``(polygon, bbox)``.

    The vote happens in a grid over the hold's own extent rather than over the
    whole canvas: a hold is a couple of percent of the wall, so a canvas-sized
    grid would resolve it into a dozen pixels, while the same raster budget
    spent locally gives the shape a few hundred across.
    """
    if not polygons:
        return None
    stacked = np.vstack(polygons)
    x0, y0 = stacked.min(axis=0)
    x1, y1 = stacked.max(axis=0)

    pad = 0.005
    x0, y0 = max(0.0, x0 - pad), max(0.0, y0 - pad)
    x1, y1 = min(1.0, x1 + pad), min(1.0, y1 + pad)
    w, h = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    long_side = max(w, h)
    grid_w = max(8, int(round(raster_size * w / long_side)))
    grid_h = max(8, int(round(raster_size * h / long_side)))

    accum = np.zeros((grid_h, grid_w), dtype=np.int32)
    for polygon in polygons:
        local = (polygon - [x0, y0]) / [w, h] * [grid_w - 1, grid_h - 1]
        one = np.zeros((grid_h, grid_w), dtype=np.uint8)
        cv2.fillPoly(one, [np.round(local).astype(np.int32)], 1)
        accum += one

    threshold = max(1, int(np.ceil(len(polygons) * vote_fraction)))
    binary = (accum >= threshold).astype(np.uint8)
    if not binary.any():
        return None

    labeled, count = ndimage.label(binary)
    if count == 0:
        return None
    areas = ndimage.sum(binary, labeled, index=range(1, count + 1)).astype(int)
    largest = int(np.argmax(areas)) + 1
    if int(areas[largest - 1]) < min_area_px:
        return None

    cleaned = labeled == largest
    if fill_holes:
        cleaned = ndimage.binary_fill_holes(cleaned)

    contours, _ = cv2.findContours((cleaned.astype(np.uint8) * 255), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    eps = max(0.5, 0.004 * cv2.arcLength(contour, True))
    approx = cv2.approxPolyDP(contour, eps, True).reshape(-1, 2)
    if len(approx) < 3:
        approx = contour.reshape(-1, 2)
    if len(approx) < 3:
        return None

    polygon = approx / [grid_w - 1, grid_h - 1] * [w, h] + [x0, y0]
    return polygon.tolist(), _bbox(polygon)


def visible_frames(centroid: np.ndarray, camera: Track, frames: list[int], *,
                   inset: float = 0.02) -> list[int]:
    """Which sampled frames could have seen a point at *centroid* on the canvas.

    The denominator for "how often was this hold detected". The tripod version
    could divide by the number of samples because every sample saw the whole
    wall; handheld, a hold that is off-screen for half the clip would look like
    a hold the model kept losing. The camera track knows where each frame was
    pointing, so the question can be asked properly.

    Inset from the frame edge because a hold half out of shot is one SAM has
    little chance on and should not be marked down for.
    """
    seen = []
    for frame in frames:
        if frame >= camera.n_frames:
            continue
        point = camera.to_frame(centroid.reshape(1, 2), frame)[0]
        if inset <= point[0] <= 1 - inset and inset <= point[1] <= 1 - inset:
            seen.append(frame)
    return seen


@dataclass
class Cluster:
    """One hold's surviving sightings, before they are reduced to a shape.

    Kept as a stage of its own because merging two tracks has to happen *before*
    the vote, not after it. Voting first and merging the results would leave the
    loser's outline on the floor: two ids on one hold each carry their own view
    of it, and the hold is the agreement between them, not whichever of the two
    happened to be seen more often.
    """

    tracks: list[int]
    polygons: list[np.ndarray]        # canvas coords, one per surviving sighting
    frames: list[int]
    scores: list[float]

    @property
    def centre(self) -> np.ndarray:
        return np.median(np.array([p.mean(axis=0) for p in self.polygons]), axis=0)

    @property
    def bbox(self) -> list[float]:
        return _bbox(np.vstack(self.polygons))


def consolidate(obs: list[Observation], camera: Track, sampled: list[int], *,
                reject_radius: float, min_appearance: float, min_sightings: int
                ) -> tuple[list[Cluster], list[dict]]:
    """One cluster per track, identity switches rejected. ``(clusters, rejected)``.

    ``reject_radius`` is in multiples of the hold's own median size, not in
    pixels: a big volume's centroid legitimately wanders further than a crimp's,
    because SAM's mask covers a different part of it as the view swings round,
    and a fixed pixel budget would either wave through a switch on a small hold
    or condemn an honest wobble on a large one.
    """
    clusters: list[Cluster] = []
    rejected: list[dict] = []

    for track, sightings in sorted(project(obs, camera).items()):
        polygons = [p for _, p in sightings]
        centroids = np.array([p.mean(axis=0) for p in polygons])
        median = np.median(centroids, axis=0)

        sizes = np.array([max(_bbox(p)[2], _bbox(p)[3]) for p in polygons])
        scale = float(np.median(sizes)) or 1e-3
        distance = np.linalg.norm(centroids - median, axis=1)
        keep = distance <= reject_radius * scale

        if (~keep).any():
            rejected.append({"track": track, "dropped": int((~keep).sum()),
                             "of": len(sightings), "reason": "identity switch",
                             "worst_px": round(float(distance.max() * max(camera.size)), 1)})
        if keep.sum() < max(min_sightings, 1):
            continue

        cluster = Cluster(
            tracks=[int(track)],
            polygons=[p for p, ok in zip(polygons, keep) if ok],
            frames=[int(o.frame) for (o, _), ok in zip(sightings, keep) if ok],
            scores=[float(o.score) for (o, _), ok in zip(sightings, keep) if ok])

        could_see = visible_frames(cluster.centre, camera, sampled)
        fraction = len(cluster.frames) / max(len(could_see), 1)
        if fraction < min_appearance:
            rejected.append({"track": track, "dropped": len(sightings),
                             "of": len(sightings), "reason": "rarely seen",
                             "fraction": round(fraction, 3)})
            continue
        clusters.append(cluster)
    return clusters, rejected


def merge(clusters: list[Cluster], *, iou_threshold: float
          ) -> tuple[list[Cluster], list[tuple[int, int, float]]]:
    """Fuse tracks that turned out to be the same hold. ``(clusters, merges)``.

    SAM sometimes drops a track and picks the hold back up under a new id — a
    long occlusion, or the hold leaving shot and coming back. On the canvas those
    two ids are two outlines in the same place, which is a question only the
    canvas can answer, since in the video they never coexist.

    The threshold wants to be high. The genuine re-identifications on this clip
    overlap at 0.93 and above, because they are literally the same hold seen
    twice; the pair that overlaps at 0.30 is the two long rails set parallel a
    hand's width apart, and fusing those loses a hold and invents a shape
    spanning both. A merge is a claim that two outlines are one object, and
    should be made only when they nearly coincide.
    """
    if not clusters:
        return [], []

    order = sorted(range(len(clusters)), key=lambda i: -len(clusters[i].frames))
    merged: list[Cluster] = []
    events: list[tuple[int, int, float]] = []
    for index in order:
        cluster = clusters[index]
        best, best_iou = -1, iou_threshold
        for position, kept in enumerate(merged):
            score = _iou(cluster.bbox, kept.bbox)
            if score > best_iou:
                best, best_iou = position, score
        if best < 0:
            merged.append(Cluster(list(cluster.tracks), list(cluster.polygons),
                                  list(cluster.frames), list(cluster.scores)))
            continue
        into = merged[best]
        events.append((cluster.tracks[0], into.tracks[0], round(best_iou, 3)))
        into.tracks = sorted(set(into.tracks) | set(cluster.tracks))
        into.polygons += cluster.polygons
        into.frames += cluster.frames
        into.scores += cluster.scores
    return merged, events


def shape(clusters: list[Cluster], camera: Track, sampled: list[int], *,
          raster_size: int, vote_fraction: float, min_area_px: int,
          fill_holes: bool) -> list[dict]:
    """Reduce each cluster to one canvas hold: the shape its sightings agree on."""
    holds: list[dict] = []
    for cluster in clusters:
        voted = _vote(cluster.polygons, raster_size=raster_size,
                      vote_fraction=vote_fraction, min_area_px=min_area_px,
                      fill_holes=fill_holes)
        if voted is None:
            # No agreement worth a shape; keep the hold on its mean box so it is
            # not silently lost. Touch tests degrade to the box plus its margin.
            polygon = []
            bbox = [float(v) for v in np.mean([_bbox(p) for p in cluster.polygons],
                                              axis=0)]
        else:
            polygon, bbox = voted

        centre = cluster.centre
        scatter = np.linalg.norm(
            np.array([p.mean(axis=0) for p in cluster.polygons]) - centre, axis=1)
        frames = sorted(set(cluster.frames))
        could_see = visible_frames(centre, camera, sampled)
        holds.append({
            "bbox": bbox,
            "polygon": polygon,
            "score": round(float(np.mean(cluster.scores)), 4),
            "appearances": len(frames),
            "appearance_fraction": round(min(len(frames) / max(len(could_see), 1), 1.0), 4),
            "drift_px": round(float(np.median(scatter) * max(camera.size)), 2),
            "tracks": sorted(cluster.tracks),
            "frames": frames,
        })
    return holds


# ── geometry shared with the tripod version ──────────────────────────────────

def _iou(a, b) -> float:
    """Intersection over union for two normalized ``[x, y, w, h]`` boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 1e-9 else 0.0


def _area(box) -> float:
    return float(box[2] * box[3])


def _containment(inner, outer) -> float:
    """Fraction of *inner*'s area that lies inside *outer*. Normalized xywh."""
    ax, ay, aw, ah = inner
    bx, by, bw, bh = outer
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area = aw * ah
    return float(inter / area) if area > 1e-9 else 0.0


def suppress_contained(holds: list[dict], *, max_containment: float | None
                       ) -> tuple[list[dict], list[tuple[dict, dict, float]]]:
    """Drop a hold that sits mostly inside a better one. ``(kept, dropped)``.

    IoU is symmetric and therefore blind to a small box nested in a large one: a
    blob covering a fifth of a hold's area scores about 0.28 against it, under
    any sane merge threshold, and survives as a hold in its own right.
    Containment is the asymmetric test that catches it — how much of the
    *smaller* box is inside the larger, ~95% for a knob segmented off its own
    volume and near zero for two holds merely set close together.

    Ranked by score, then area: the survivor is the detection SAM was surest of
    and, failing that, the whole hold rather than the piece of it.
    """
    if not holds or max_containment is None:
        return list(holds), []

    order = sorted(range(len(holds)),
                   key=lambda i: (-float(holds[i].get("score") or 0.0),
                                  -_area(holds[i]["bbox"])))
    kept: list[int] = []
    dropped: list[tuple[dict, dict, float]] = []
    for i in order:
        box = holds[i]["bbox"]
        covered = max(((j, _containment(box, holds[j]["bbox"])) for j in kept),
                      key=lambda p: p[1], default=None)
        if covered is not None and covered[1] >= max_containment:
            dropped.append((holds[i], holds[covered[0]], covered[1]))
        else:
            kept.append(i)
    return [holds[i] for i in sorted(kept)], dropped


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
    """Number the holds bottom-to-top on the canvas, ordering a row along the lean.

    On the canvas rather than in any one frame, which is the point: the
    numbering is now a property of the wall and survives the camera moving,
    where a frame-space ordering would renumber the route every time the
    operator tilted.

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

def wall_mask(holds: list[dict], *, res: int, dilate: float) -> np.ndarray | None:
    """A binary canvas mask of the route's footprint, dilated.

    A rectangle around the holds is the wrong shape: a route runs on a diagonal,
    so its bounding box is mostly the empty triangle beside it, and somebody
    standing in that triangle would score as "on the wall" exactly like the
    climber. The real hold pixels do not have that problem. The dilation is what
    lets a torso spanning the gap between two holds still register.
    """
    if not holds:
        return None

    mask = np.zeros((res, res), dtype=np.uint8)
    for hold in holds:
        polygon = hold.get("polygon")
        if polygon is not None and len(polygon) >= 3:
            pts = np.asarray([[p[0] * (res - 1), p[1] * (res - 1)] for p in polygon],
                             dtype=np.int32)
            cv2.fillPoly(mask, [pts], 1)
        else:
            bx, by, bw, bh = hold["bbox"]
            cv2.rectangle(mask, (int(bx * (res - 1)), int(by * (res - 1))),
                          (int((bx + bw) * (res - 1)), int((by + bh) * (res - 1))),
                          1, thickness=-1)

    px = max(0, int(round(dilate * res)))
    if px:
        mask = cv2.dilate(mask, np.ones((2 * px + 1, 2 * px + 1), np.uint8))
    return mask if mask.any() else None


def wall_overlap(bbox_xywh, mask: np.ndarray) -> float:
    """Fraction of a normalized ``[x, y, w, h]`` box that sits on wall pixels."""
    if mask is None:
        return 0.0
    res = mask.shape[0]
    x, y, w, h = bbox_xywh
    x0 = max(0, int(round(x * (res - 1))))
    y0 = max(0, int(round(y * (res - 1))))
    x1 = min(res, int(round((x + w) * (res - 1))) + 1)
    y1 = min(res, int(round((y + h) * (res - 1))) + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    region = mask[y0:y1, x0:x1]
    return float(region.sum()) / float(region.size) if region.size else 0.0


def filter_by_pose_region(holds: list[dict], points: np.ndarray, *,
                          margin: float) -> tuple[list[dict], int]:
    """Drop canvas holds outside the convex hull of where the climber's body went.

    The prompt asks for a colour, and a colour is not a route: the same green
    appears on the wall round the corner and above the finish. What separates
    this route from the rest of the green is that the climber's body passed over
    it, so the body's own trajectory is the filter — and the trajectory has to be
    in canvas coordinates too, or the hull is the union of everywhere the
    *camera* pointed rather than everywhere the climber went.
    """
    if len(points) < 3:
        return holds, 0

    from scipy.spatial import ConvexHull, QhullError

    try:
        hull = ConvexHull(points)
    except (QhullError, ValueError):
        return holds, 0

    # Rasterize the hull and dilate it, rather than scaling the vertices about
    # the centroid: a scale moves a far vertex further than a near one, so the
    # margin would mean a different distance on every edge.
    res = 512
    verts = np.round(np.clip(points[hull.vertices], 0, 1) * (res - 1)).astype(np.int32)
    grid = np.zeros((res, res), dtype=np.uint8)
    cv2.fillPoly(grid, [verts.reshape(-1, 1, 2)], 1)
    px = max(1, int(round(margin * res)))
    grid = cv2.dilate(grid, np.ones((2 * px + 1, 2 * px + 1), np.uint8))

    kept, dropped = [], 0
    for hold in holds:
        x, y, w, h = hold["bbox"]
        cx = min(res - 1, max(0, int(round((x + w / 2) * (res - 1)))))
        cy = min(res - 1, max(0, int(round((y + h / 2) * (res - 1)))))
        if grid[cy, cx]:
            kept.append(hold)
        else:
            dropped += 1

    # Dropping everything means the filter, not the route, is wrong.
    return (kept, dropped) if kept else (holds, 0)


# ── per-frame geometry ───────────────────────────────────────────────────────

def in_frame(hold: dict, camera: Track, frame: int) -> np.ndarray:
    """A canvas hold's outline projected into one frame, normalized to that frame.

    The renderer's last step and the touch test's first. Exact, because the
    homography maps the polygon's vertices and the edges between them stay
    straight.
    """
    polygon = hold.get("polygon")
    if polygon is not None and len(polygon) >= 3:
        points = np.asarray(polygon, dtype=np.float64)
    else:
        x, y, w, h = hold["bbox"]
        points = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
    return camera.to_frame(points, frame)


def frame_bbox(hold: dict, camera: Track, frame: int) -> list[float]:
    """The axis-aligned box of :func:`in_frame`, normalized to the frame."""
    return _bbox(in_frame(hold, camera, frame))


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
