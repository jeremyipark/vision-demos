"""The climber: the ViTPose request, and picking which body is the climber.

The gateway returns one flat list of person records, each tagged with the frame
it came from and the track it belongs to. Two things happen here that the model
does not do for us:

  * **which track is the climber.** The choice is made once for the whole clip
    rather than per frame, by scoring each track against the route's own
    footprint. Somebody walking past the wall scores near zero against it and so
    cannot take the overlay from the climber halfway up.
  * **smoothing.** Keypoints jitter a pixel or two between frames; a hold's
    touch test is a threshold on position, so the jitter shows up as a hold
    flickering on and off. Smoothed per track, across gaps, never across them.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .skeleton import KPT_NAMES

CONTENT_OBJECT_KPTS = "vid.pose.kpts"

# Bumped when the request or the response shape this module reads changes, so a
# cache written against the old one is a miss rather than a confusing crash.
CONTRACT = "flat-envelope-v1"

N_KPTS = 17


@dataclass
class PoseResult:
    """One track's keypoints, resolved per source-frame index."""

    kpts: dict[int, np.ndarray]       # frame index -> (17, 2) normalized
    valid: dict[int, np.ndarray]      # frame index -> (17,) bool
    bboxes: dict[int, list[float]]    # frame index -> [x, y, w, h] normalized
    track_id: int | None
    n_frames_returned: int
    track_ids: list[int]

    @property
    def n_posed(self) -> int:
        return len(self.kpts)

    def frames(self) -> list[int]:
        return sorted(self.kpts)


def encode_video(video: Path) -> str:
    """The clip as base64, for a request that uploads the whole thing.

    Both video calls need it — ViTPose for the pose and SAM 3.1 for the hold
    tracks — so it lives in one place rather than being spelled twice.
    """
    return base64.b64encode(video.read_bytes()).decode("ascii")


def build_request(video: Path, *, every_frame: bool, fps: float, n_frames: int,
                  video_fps: float, video_max_frames, precision: int) -> tuple[str, dict]:
    """Return ``(base64_video, extra_body)``.

    ``video_fps`` is the *detector* cadence, not a sampling rate; pose runs on
    every decoded frame regardless. It reaches stride 1 as soon as it is >= the
    decoded rate. ``video_max_frames`` is what controls how many frames are
    decoded, posed and billed.
    """
    if every_frame:
        video_fps, video_max_frames = fps, n_frames

    extra_body = {"method": "pose", "video_fps": video_fps, "precision": precision}
    if video_max_frames is not None:
        extra_body["video_max_frames"] = video_max_frames

    return encode_video(video), extra_body


def request_poses(client, *, model: str, video_b64: str, extra_body: dict):
    """One blocking chat-completions call, returning the parsed JSON payload."""
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [{
                "type": "video_url",
                "video_url": {"url": f"data:video/mp4;base64,{video_b64}"},
            }],
        }],
        response_format={"type": "json_object"},
        extra_body=extra_body,
    )
    payload = json.loads(response.choices[0].message.content)
    usage = response.usage.model_dump() if response.usage else None
    return payload, usage


def unwrap(payload: dict) -> tuple[list[dict], list[int]]:
    """Return ``(person_records, sampled_frame_indices)``.

    The envelope is flat — the model, the clip's details, then `content` — so the
    payload is read straight off the reply. Each record carries the `frame_id` it
    came from and the `track_id` it belongs to; `frames` lists every sampled
    frame, including the ones nobody was found in.

    The joint order is checked rather than assumed. `kpts_labels` names the
    joints the model actually returned, and every index in `skeleton.py` — which
    wrist, which ankle — is a position in that list. A model that returned a
    different layout would still draw, silently, as a tangle of limbs.
    """
    content = payload.get("content")
    if not isinstance(content, dict):
        raise RuntimeError(f"Unexpected response envelope: {json.dumps(payload)[:400]}")

    got = content.get("object")
    if got != CONTENT_OBJECT_KPTS:
        raise RuntimeError(
            f"Expected a video pose payload ({CONTENT_OBJECT_KPTS}), got {got!r}. "
            "Did the request send an image_url instead of a video_url?"
        )

    labels = content.get("kpts_labels")
    if labels is not None and list(labels) != KPT_NAMES:
        raise RuntimeError(
            f"The model returned {len(labels)} joints in an order this demo does not "
            f"know: {list(labels)}. src/skeleton.py is written against COCO-17."
        )

    items = content.get("items") or []
    frames = [int(f["frame_id"]) for f in content.get("frames") or []]
    return items, frames


def _area(box) -> float:
    return float(box[2] * box[3])


def pick_track(items: list[dict], *, mask, overlap_fn, min_overlap: float,
               lock_to_wall: bool
               ) -> tuple[int | None, dict]:
    """Choose the climber's track, once, for the whole clip.

    Scored on how much of each track sits on the route rather than on size or
    centrality: the climber is by definition the person on the holds, and that
    is the only one of those three a passerby cannot accidentally win. Coverage
    is a tiebreak so a two-frame detection cannot outrank the actual climb on a
    lucky frame. With no wall to score against, it falls back to the track that
    is present longest.
    """
    by_track: dict[int | None, list[dict]] = {}
    for item in items:
        by_track.setdefault(item.get("track_id"), []).append(item)

    if not by_track:
        return None, {}

    stats = {}
    total_frames = len({i.get("frame_id") for i in items}) or 1
    for track, records in by_track.items():
        if mask is not None:
            overlaps = [overlap_fn(r["bbox_xywh"], mask) for r in records]
        else:
            overlaps = []
        stats[track] = {
            "frames": len(records),
            "coverage": len(records) / total_frames,
            "mean_overlap": float(np.mean(overlaps)) if overlaps else 0.0,
            "mean_area": float(np.mean([_area(r["bbox_xywh"]) for r in records])),
        }

    if lock_to_wall and mask is not None:
        eligible = {t: s for t, s in stats.items() if s["mean_overlap"] >= min_overlap}
        if eligible:
            best = max(eligible, key=lambda t: (eligible[t]["mean_overlap"] * eligible[t]["coverage"]))
            return best, stats

    return max(stats, key=lambda t: stats[t]["coverage"]), stats


def resolve(items: list[dict], frames: list[int], track_id, *,
            min_kpt_score: float = 0.0) -> PoseResult:
    """Collapse the flat record list into per-frame arrays for one track.

    Two things mark a joint invisible. ``(0, 0)`` is the model's "not visible"
    sentinel, not a real corner point, so it is recorded as invalid rather than
    drawn at the origin. And `kpts_score` now carries one confidence per joint,
    so a joint the model placed but is unsure of — a wrist behind the body, an
    ankle out of frame — can be dropped on its own merits rather than only when
    the model gives up on it entirely.

    `min_kpt_score` of 0 keeps every placed joint, which is the sentinel-only
    behaviour the dwell thresholds in config.py were tuned against.
    """
    kpts: dict[int, np.ndarray] = {}
    valid: dict[int, np.ndarray] = {}
    bboxes: dict[int, list[float]] = {}

    for item in items:
        if item.get("track_id") != track_id:
            continue
        index = int(item.get("frame_id", -1))
        if index < 0:
            continue
        raw = item.get("kpts_xy") or []
        scores = item.get("kpts_score") or []
        arr = np.zeros((N_KPTS, 2), dtype=np.float32)
        ok = np.zeros(N_KPTS, dtype=bool)
        for i, xy in enumerate(raw[:N_KPTS]):
            x, y = float(xy[0]), float(xy[1])
            arr[i] = (x, y)
            confident = (float(scores[i]) >= min_kpt_score
                         if i < len(scores) and scores[i] is not None else True)
            ok[i] = confident and not (x == 0.0 and y == 0.0)
        # One person per track per frame; a duplicate keeps the larger box.
        if index in bboxes and _area(bboxes[index]) >= _area(item["bbox_xywh"]):
            continue
        kpts[index] = arr
        valid[index] = ok
        bboxes[index] = [float(v) for v in item["bbox_xywh"]]

    track_ids = sorted({i["track_id"] for i in items if i.get("track_id") is not None})
    return PoseResult(kpts=kpts, valid=valid, bboxes=bboxes, track_id=track_id,
                      n_frames_returned=len(frames), track_ids=track_ids)


def smooth(result: PoseResult, *, sigma: float) -> PoseResult:
    """Gaussian-smooth each keypoint's path in time, across contiguous runs only.

    Run-wise rather than over the whole clip: a keypoint that disappears for
    thirty frames and comes back somewhere else has not travelled between those
    two positions, and smoothing across the gap would draw exactly that journey.
    """
    if sigma <= 0 or len(result.kpts) < 3:
        return result

    from scipy.ndimage import gaussian_filter1d

    frames = result.frames()
    # Contiguous stretches of returned frames; a break starts a new run.
    runs: list[list[int]] = [[frames[0]]]
    for prev, curr in zip(frames, frames[1:]):
        if curr - prev == 1:
            runs[-1].append(curr)
        else:
            runs.append([curr])

    for kpt in range(N_KPTS):
        for run in runs:
            present = [f for f in run if result.valid[f][kpt]]
            if len(present) < 3:
                continue
            xs = np.array([result.kpts[f][kpt, 0] for f in present])
            ys = np.array([result.kpts[f][kpt, 1] for f in present])
            xs = gaussian_filter1d(xs, sigma=sigma, mode="nearest")
            ys = gaussian_filter1d(ys, sigma=sigma, mode="nearest")
            for f, x, y in zip(present, xs, ys):
                result.kpts[f][kpt] = (x, y)
    return result


def confident_points(result: PoseResult) -> np.ndarray:
    """Every visible keypoint, as an (N, 2) array. The climber's whole trajectory."""
    pts = [result.kpts[f][result.valid[f]] for f in result.frames()]
    return np.concatenate(pts, axis=0) if pts else np.zeros((0, 2), dtype=np.float32)


def project(result: PoseResult, camera) -> PoseResult:
    """The same keypoints, moved onto the canvas.

    Everything after this point — the touch tests, the route line, the body's
    trail on the right panel — wants the climber in the wall's frame of
    reference rather than the camera's, and for three separate reasons.

    **A margin stops meaning two different things.** ``HOLD_MASK_MARGIN`` is a
    fraction of the frame. With a fixed camera that is a fixed distance on the
    wall, so "within 1.8% of the frame of this hold" is one rule. Under a zoom
    it is not: the same fraction is a hand's width on the wide shots and a
    fingertip on the tight ones, so a hold would get easier to touch exactly as
    the operator pushed in on the crux. On the canvas the wall has one scale and
    the margin is one distance.

    **A still hand becomes still.** In frame coordinates a hand locked onto a
    hold still travels across the image whenever the camera pans, so the touch
    test sees motion where there is none, and — worse — smoothing a keypoint's
    path in time would smooth the *camera's* motion into the body's. Which is
    why :func:`smooth` should run after this and not before.

    **The right panel has something to draw on.** The trail and the route line
    are statements about the wall, and the wall only holds still here.
    """
    kpts, valid, bboxes = {}, {}, {}
    for frame in result.frames():
        if frame >= camera.n_frames:
            continue
        kpts[frame] = camera.to_canvas(result.kpts[frame], frame).astype(np.float32)
        valid[frame] = result.valid[frame]
        box = result.bboxes.get(frame)
        if box is not None:
            x, y, w, h = box
            corners = camera.to_canvas(
                np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]), frame)
            lo, hi = corners.min(axis=0), corners.max(axis=0)
            bboxes[frame] = [float(lo[0]), float(lo[1]),
                             float(hi[0] - lo[0]), float(hi[1] - lo[1])]
    return PoseResult(kpts=kpts, valid=valid, bboxes=bboxes, track_id=result.track_id,
                      n_frames_returned=result.n_frames_returned,
                      track_ids=result.track_ids)


def project_boxes(items: list[dict], camera) -> list[dict]:
    """Copies of the raw pose records with their boxes on the canvas.

    :func:`pick_track` scores each track against the route's footprint, and the
    route now lives on the canvas, so the boxes it scores have to as well. Done
    before a track is chosen rather than after, because choosing is the thing
    that needs the comparison.
    """
    out = []
    for item in items:
        frame = item.get("frame_id")
        box = item.get("bbox_xywh")
        if frame is None or frame >= camera.n_frames or not box:
            continue
        x, y, w, h = box
        corners = camera.to_canvas(
            np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]), int(frame))
        lo, hi = corners.min(axis=0), corners.max(axis=0)
        out.append({**item, "bbox_xywh": [float(lo[0]), float(lo[1]),
                                          float(hi[0] - lo[0]), float(hi[1] - lo[1])]})
    return out


# ── cache ────────────────────────────────────────────────────────────────────

def cache_path(video: Path, cache_dir: Path, *, model: str, extra_body: dict) -> Path:
    """Where this exact request's poses live."""
    stat = video.stat()
    key = json.dumps({
        "video": video.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        "model": model, "extra_body": extra_body, "contract": CONTRACT,
    }, sort_keys=True)
    return cache_dir / f"poses.{hashlib.sha256(key.encode()).hexdigest()[:12]}.json"


def save_cache(path: Path, payload: dict, usage: dict | None, timing_dict: dict,
               *, stamp: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "created": stamp, "payload": payload, "usage": usage, "timing": timing_dict}))


def load_cache(path: Path):
    """Read a cached response, or None if it is missing or unreadable."""
    if not path.is_file():
        return None
    try:
        blob = json.loads(path.read_text())
        return blob["payload"], blob.get("usage"), blob.get("timing") or {}, \
            blob.get("created", "unknown")
    except (json.JSONDecodeError, KeyError, OSError):
        return None
