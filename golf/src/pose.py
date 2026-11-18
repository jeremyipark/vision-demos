"""ViTPose through the Gateway: the request, the reply, and its cache."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from .skeleton import KPT_NAMES

CONTENT_OBJECT_KPTS = "vid.pose.kpts"
CONTRACT = "flat-envelope-v1"   # bump when the request or reply shape changes


def build_request(video: Path, *, video_fps: float, max_frames: int, precision: int
                  ) -> tuple[str, dict]:
    """``(base64 video, extra_body)``.

    ``video_fps`` is the detector cadence, not a sampling rate. Pose runs on
    every decoded frame. ``video_max_frames`` is what caps how many are billed.
    """
    extra_body = {"method": "pose", "video_fps": video_fps, "precision": precision,
                  "video_max_frames": max_frames}
    return base64.b64encode(video.read_bytes()).decode("ascii"), extra_body


def unwrap(payload: dict) -> tuple[list[dict], dict[int, list[dict]]]:
    """Person records grouped by frame; frames with nobody stay as empty lists."""
    content = payload.get("content")
    if not isinstance(content, dict):
        raise RuntimeError(f"Unexpected response envelope: {json.dumps(payload)[:400]}")
    if content.get("object") != CONTENT_OBJECT_KPTS:
        raise RuntimeError(f"Expected a video pose payload ({CONTENT_OBJECT_KPTS}), "
                           f"got {content.get('object')!r}.")
    labels = content.get("kpts_labels")
    if labels is not None and list(labels) != KPT_NAMES:
        raise RuntimeError(f"Unexpected joint order {list(labels)}; this demo expects COCO-17.")

    by_index: dict[int, list[dict]] = {int(f["frame_id"]): [] for f in content.get("frames") or []}
    for person in content.get("items") or []:
        if person.get("frame_id") is not None:
            by_index.setdefault(int(person["frame_id"]), []).append(person)
    frames = [{"index": i, "persons": by_index[i]} for i in sorted(by_index)]
    return frames, by_index


def request_poses(client, *, model: str, video_b64: str, extra_body: dict) -> tuple[dict, dict | None]:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [
            {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}},
        ]}],
        response_format={"type": "json_object"},
        extra_body=extra_body,
    )
    payload = json.loads(response.choices[0].message.content)
    return payload, response.usage.model_dump() if response.usage else None


def cache_path(video: Path, cache_dir: Path, *, model: str, extra_body: dict) -> Path:
    stat = video.stat()
    key = json.dumps({"video": video.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                      "model": model, "extra_body": extra_body, "contract": CONTRACT},
                     sort_keys=True)
    return cache_dir / f"poses.{hashlib.sha256(key.encode()).hexdigest()[:12]}.json"


def save_cache(path: Path, payload: dict, usage: dict | None, *, stamp: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"created": stamp, "payload": payload, "usage": usage}))


def load_cache(path: Path) -> tuple[dict, dict | None, str] | None:
    if not path.is_file():
        return None
    try:
        blob = json.loads(path.read_text())
        return blob["payload"], blob.get("usage"), blob.get("created", "unknown")
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def _iou(a, b) -> float:
    """Intersection over union of two normalized ``[x, y, w, h]`` boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    inter = (max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
             * max(0.0, min(ay + ah, by + bh) - max(ay, by)))
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def dedupe(frames: list[dict], *, max_persons: int | None, iou: float) -> int:
    """Drop duplicate boxes per frame, keeping the best match to the last frame. Returns the count."""
    previous = None
    dropped = 0
    for frame in frames:
        persons = frame.get("persons") or []
        if not persons:
            continue
        order = sorted(persons, key=lambda p: (
            -(_iou(p["bbox_xywh"], previous) if previous is not None else 0.0),
            -float(p["bbox_xywh"][2] * p["bbox_xywh"][3])))
        kept: list[dict] = []
        for person in order:
            if all(_iou(person["bbox_xywh"], k["bbox_xywh"]) < iou for k in kept):
                kept.append(person)
        if max_persons:
            kept = kept[:max_persons]
        dropped += len(persons) - len(kept)
        frame["persons"] = kept
        if kept:
            previous = kept[0]["bbox_xywh"]
    return dropped
