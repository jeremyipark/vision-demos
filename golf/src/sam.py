"""SAM 3.1 tracking through the Gateway: one prompt, the whole clip, cut into calls.

SAM 3.1's ``track`` method returns every instance of the prompt on every sampled
frame, each with a ``track_id``, plus one label-map PNG per frame (pixel =
track id, 0 = nothing). The gateway caps a call at 128 frames, so a longer clip
goes in segments, renumbered back into the clip's own frame and track space.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

CONTENT_OBJECT = "vid.segment.masks"
CONTRACT = "sam31-golf-track-v1"   # bump when the request or reply shape changes
TRACK_MAX_FRAMES = 128             # SAM 3.1's per-call limit


def request_track(client, *, model: str, video_b64: str, prompt: str, stride: int,
                  max_frames: int, attempts: int = 4) -> tuple[dict, dict | None]:
    """Every instance of *prompt*, tracked through one segment. Retries 5xx."""
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
                    "video_max_frames": max_frames, "mask_format": "png",
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


def segment_plan(n_frames: int, stride: int, *, max_frames: int = TRACK_MAX_FRAMES
                 ) -> list[tuple[int, int]]:
    """``[(start, length), ...]`` covering the clip at *stride*, evenly, within the cap."""
    span = max_frames * stride
    if n_frames <= span:
        return [(0, n_frames)]
    count = -(-n_frames // span)
    length = -(-n_frames // count)
    return [(i * length, min(length, n_frames - i * length)) for i in range(count)]


def unwrap(payload: dict) -> tuple[list[dict], list[dict]]:
    content = payload.get("content")
    if not isinstance(content, dict) or content.get("object") != CONTENT_OBJECT:
        raise RuntimeError(f"Expected a {CONTENT_OBJECT} payload, got {json.dumps(payload)[:300]}")
    return content.get("items") or [], content.get("frames") or []


def decode_label_map(mask) -> np.ndarray | None:
    """A frame's label map: pixel = SAM's id for that segment, 0 = nothing."""
    if not isinstance(mask, dict) or mask.get("format") != "png":
        return None
    data = mask.get("data")
    if not isinstance(data, str):
        return None
    raw = base64.b64decode(data.split(",", 1)[-1])
    labels = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if labels is None:
        return None
    return labels[:, :, 0] if labels.ndim == 3 else labels


def detections(segments: list[tuple[int, dict]], *, width: int, height: int,
               min_score: float) -> dict[int, list[dict]]:
    """``{clip frame: [det, ...]}`` across every segment.

    Each det: ``track`` (unique across segments), ``box`` (x0, y0, x1, y1 px),
    ``score``, and ``mask`` (a bool array the size of the inference frame, or None).
    """
    out: dict[int, list[dict]] = {}
    for seg_no, (offset, payload) in enumerate(segments):
        items, frames = unwrap(payload)
        labels = {int(f["frame_id"]): f.get("mask") for f in frames}
        decoded: dict[int, np.ndarray | None] = {}
        for item in items:
            box, fid = item.get("bbox_xywh"), item.get("frame_id")
            score = float(item.get("score") or 0)
            if fid is None or not box or score < min_score:
                continue
            fid = int(fid)
            if fid not in decoded:
                decoded[fid] = decode_label_map(labels.get(fid))
            mask = None
            lm = decoded[fid]
            tid = item.get("track_id")
            if lm is not None and tid is not None:
                m = lm == int(tid)
                if m.shape != (height, width):
                    m = cv2.resize(m.astype(np.uint8), (width, height),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
                mask = m if m.any() else None
            x, y, w, h = box
            out.setdefault(fid + offset, []).append({
                "track": f"{seg_no}:{tid}", "score": score, "mask": mask,
                "box": (x * width, y * height, (x + w) * width, (y + h) * height)})
    return out


def cache_path(video: Path, cache_dir: Path, *, kind: str, model: str, prompt: str,
               start: int, length: int, stride: int) -> Path:
    stat = video.stat()
    key = json.dumps({"video": video.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                      "model": model, "prompt": prompt, "start": start, "length": length,
                      "stride": stride, "contract": CONTRACT}, sort_keys=True)
    return cache_dir / f"{kind}.{hashlib.sha256(key.encode()).hexdigest()[:12]}.json"
