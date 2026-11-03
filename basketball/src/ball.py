"""The ball: SAM 3.1 tracking it through a window around each shot.

In the hands and the net SAM's box grows past the ball, so boxes off the ball's
size are dropped, and frames with no ball stay empty rather than guessed.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CONTENT_OBJECT = "vid.segment.masks"
CONTRACT = "sam31-ball-track-v1"

TRACK_MAX_FRAMES = 128   # SAM 3.1's per-call limit


def request_track(client, *, model: str, video_b64: str, prompt: str, stride: int,
                  attempts: int = 4) -> tuple[dict, dict | None]:
    """Every instance of *prompt*, tracked through the clip. Retries 5xx (the gateway
    sometimes answers 502 under load)."""
    from openai import APIStatusError

    for attempt in range(attempts):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}},
                ]}],
                response_format={"type": "json_object"},
                extra_body={"method": "track", "method_params": {
                    "prompt": prompt, "video_skip_frames": stride,
                    "video_max_frames": TRACK_MAX_FRAMES, "mask_format": "png",
                    "polygons": False}},
            )
            break
        except APIStatusError as exc:
            if exc.status_code < 500 or attempt == attempts - 1:
                raise
            time.sleep(2 * (attempt + 1))
    payload = json.loads(response.choices[0].message.content)
    usage = response.usage.model_dump() if response.usage else None
    return payload, usage


def detections(payload: dict, *, frame_offset: int, width: int, height: int,
               min_score: float) -> dict[int, list[tuple[float, float, float, float, float]]]:
    """``{clip frame: [(cx, cy, w, h, score), ...]}`` in pixels of the inference frame."""
    content = payload.get("content")
    if not isinstance(content, dict) or content.get("object") != CONTENT_OBJECT:
        raise RuntimeError(f"Expected a {CONTENT_OBJECT} payload, got "
                           f"{json.dumps(payload)[:300]}")
    out: dict[int, list] = {}
    for item in content.get("items") or []:
        box = item.get("bbox_xywh")
        if item.get("frame_id") is None or not box or float(item.get("score") or 0) < min_score:
            continue
        x, y, w, h = box
        out.setdefault(int(item["frame_id"]) + frame_offset, []).append(
            ((x + w / 2) * width, (y + h / 2) * height, w * width, h * height,
             float(item.get("score") or 0)))
    return out


def cache_path(video: Path, cache_dir: Path, *, model: str, prompt: str, start: int,
               length: int, stride: int) -> Path:
    stat = video.stat()
    key = json.dumps({"video": video.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                      "model": model, "prompt": prompt, "start": start, "length": length,
                      "stride": stride, "contract": CONTRACT}, sort_keys=True)
    return cache_dir / f"ball.{hashlib.sha256(key.encode()).hexdigest()[:12]}.json"


@dataclass
class BallTrack:
    """The ball's center and diameter on every frame of the clip; NaN where unseen."""

    x: np.ndarray
    y: np.ndarray
    d: np.ndarray
    covered: np.ndarray                      # bool: SAM looked at this frame
    windows: list[tuple[int, int]] = field(default_factory=list)
    diameter: float = float("nan")           # the ball's typical size, px
    shown: BallTrack | None = None           # a looser pick for drawing only, held in the hands too

    @property
    def found(self) -> np.ndarray:
        return np.isfinite(self.x)


def typical_diameter(raw: dict[int, list], frames=None) -> float:
    """The ball's size: the median of the round boxes (one grown into a hand isn't round),
    over *frames* if given."""
    sizes = [max(w, h) for f, boxes in raw.items() if frames is None or f in frames
             for _, _, w, h, _ in boxes if abs(w - h) <= 0.2 * max(w, h)]
    return float(np.median(sizes)) if sizes else float("nan")


def select(raw: dict[int, list], n_frames: int, windows: list[tuple[int, int]], *,
           size_tolerance: float, local_size: bool = False) -> BallTrack:
    """One ball per frame: ball-sized, round, and on a smooth path.

    *local_size* sizes the ball per window, since it looks bigger near the camera.
    """
    diameter = typical_diameter(raw)
    x = np.full(n_frames, np.nan)
    y = np.full(n_frames, np.nan)
    d = np.full(n_frames, np.nan)
    covered = np.zeros(n_frames, bool)
    for start, length in windows:
        covered[start:start + length] = True
    if not np.isfinite(diameter):
        return BallTrack(x, y, d, covered, windows, diameter)

    local = np.full(n_frames, diameter)
    for start, length in windows if local_size else []:
        size = typical_diameter(raw, range(start, start + length))
        if np.isfinite(size):
            local[start:start + length] = size

    previous = None
    for frame in sorted(raw):
        if not 0 <= frame < n_frames:
            continue
        candidates = [b for b in raw[frame]
                      if abs(max(b[2], b[3]) / local[frame] - 1) <= size_tolerance
                      and abs(b[2] - b[3]) <= 0.35 * max(b[2], b[3])]
        if not candidates:
            continue
        if previous is None:
            best = max(candidates, key=lambda b: b[4])
        else:
            best = min(candidates, key=lambda b: np.hypot(b[0] - previous[0], b[1] - previous[1]))
        x[frame], y[frame], d[frame] = best[0], best[1], max(best[2], best[3])
        previous = best

    # Drop lone points far off the path their neighbours draw (a head, a shoe).
    # Twice, since dropping one can expose the next.
    for _ in range(2):
        kept = np.nonzero(np.isfinite(x))[0]
        drop = []
        for k in range(1, len(kept) - 1):
            a, i, b = kept[k - 1], kept[k], kept[k + 1]
            if b - a > 8:
                continue
            t = (i - a) / (b - a)
            px, py = x[a] + t * (x[b] - x[a]), y[a] + t * (y[b] - y[a])
            if np.hypot(x[i] - px, y[i] - py) > 2.5 * local[i]:
                drop.append(i)
        x[drop] = y[drop] = d[drop] = np.nan
    return BallTrack(x, y, d, covered, windows, diameter)


def smoothed(track: BallTrack, *, max_gap: int, window: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """The ball's ``(x, y)`` for drawing: short gaps filled along the path, then lightly
    smoothed, so the ring glides instead of blinking. Judging uses the strict track."""
    track = track.shown or track
    x, y = track.x.copy(), track.y.copy()
    seen = np.nonzero(np.isfinite(x))[0]
    for a, b in zip(seen, seen[1:]):
        if 1 < b - a <= max_gap + 1:
            t = (np.arange(a + 1, b) - a) / (b - a)
            x[a + 1:b] = x[a] + t * (x[b] - x[a])
            y[a + 1:b] = y[a] + t * (y[b] - y[a])
    half = window // 2
    out = []
    for v in (x, y):
        s = v.copy()
        for i in np.nonzero(np.isfinite(v))[0]:
            seg = v[max(0, i - half):i + half + 1]
            s[i] = np.nanmean(seg)
        out.append(s)
    return out[0], out[1]
