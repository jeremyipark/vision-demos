"""The gateway calls, each cached: ViTPose over the clip, and SAM 3.1 tracking a prompt."""

from __future__ import annotations

import base64
import json
import time
from datetime import datetime
from pathlib import Path

from . import pose, sam, video


def stamp_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def cost_of(usage: dict | None) -> float:
    return float((usage or {}).get("cost") or 0)


def client(api_key: str, cfg):
    from openai import OpenAI

    return OpenAI(base_url=cfg.GATEWAY_BASE_URL, api_key=api_key,
                  timeout=cfg.REQUEST_TIMEOUT, max_retries=1)


def pose_clip(api_key: str, mp4: Path, info: video.VideoInfo, cfg, console) -> dict:
    """ViTPose over the whole clip, cached."""
    video_b64, extra_body = pose.build_request(mp4, video_fps=cfg.POSE_FPS,
                                               max_frames=info.n_frames, precision=cfg.PRECISION)
    cache = pose.cache_path(mp4, cfg.CACHE_DIR, model=cfg.POSE_MODEL, extra_body=extra_body)
    cached = pose.load_cache(cache) if cfg.REUSE_CACHE else None
    seconds = None
    if cached is not None:
        payload, usage, created = cached
        console.print(f"  [green]cache hit[/]: poses from [dim]{created}[/]; no gateway call")
    else:
        t0 = time.perf_counter()
        with console.status("[cyan]waiting on the gateway[/]; ViTPose on every frame…",
                            spinner="dots"):
            payload, usage = pose.request_poses(client(api_key, cfg), model=cfg.POSE_MODEL,
                                                video_b64=video_b64, extra_body=extra_body)
        seconds = time.perf_counter() - t0
        pose.save_cache(cache, payload, usage, stamp=stamp_now())
        console.print(f"  [green]done in {seconds:.1f}s[/]")

    frames, _ = pose.unwrap(payload)
    dropped = pose.dedupe(frames, max_persons=cfg.MAX_PERSONS, iou=cfg.PERSON_IOU_THRESHOLD)
    return {"by_index": {f["index"]: f["persons"] for f in frames}, "dropped": dropped,
            "seconds": seconds, "extra_body": extra_body,
            "cost": 0.0 if cached is not None else cost_of(usage)}


def track(api_key: str, mp4: Path, info: video.VideoInfo, *, kind: str, prompt: str,
          min_score: float, cfg, console) -> tuple[dict[int, list[dict]], float, int]:
    """SAM 3.1 tracking *prompt* through the whole clip in <=128-frame segments, cached.

    Returns ``(detections by clip frame, cost, gateway calls made)``.
    """
    segments: list[tuple[int, dict]] = []
    cost, calls = 0.0, 0
    plan = sam.segment_plan(info.n_frames, cfg.SAM_STRIDE)
    with console.status(f"[cyan]waiting on the gateway[/]; SAM is tracking the {kind}…",
                        spinner="dots") as status:
        for n, (start, length) in enumerate(plan, 1):
            cache = sam.cache_path(mp4, cfg.CACHE_DIR, kind=kind, model=cfg.SAM_MODEL,
                                   prompt=prompt, start=start, length=length,
                                   stride=cfg.SAM_STRIDE)
            blob = json.loads(cache.read_text()) if cfg.REUSE_CACHE and cache.is_file() else None
            if blob is None:
                status.update(f"[cyan]waiting on the gateway[/]; SAM is tracking the {kind} "
                              f"(part {n}/{len(plan)})…")
                seg = cfg.CACHE_DIR / f"{mp4.stem}.{kind}{start}_{length}.mp4"
                if not seg.is_file():
                    video.segment(mp4, seg, start_frame=start, n_frames=length, fps=info.fps,
                                  crf=cfg.CONVERT_CRF)
                payload, usage = sam.request_track(
                    client(api_key, cfg), model=cfg.SAM_MODEL,
                    video_b64=base64.b64encode(seg.read_bytes()).decode("ascii"),
                    prompt=prompt, stride=cfg.SAM_STRIDE,
                    max_frames=sam.TRACK_MAX_FRAMES)
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps({"created": stamp_now(), "payload": payload,
                                             "usage": usage}))
                seg.unlink(missing_ok=True)
                cost += cost_of(usage)
                calls += 1
            else:
                payload = blob["payload"]
            segments.append((start, payload))
    dets = sam.detections(segments, width=info.width, height=info.height, min_score=min_score)
    return dets, cost, calls
