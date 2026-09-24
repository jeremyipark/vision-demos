"""Segment a boulder problem with SAM 3.1, pose the climber with ViTPose, and
read the route off the pair — in three dimensions, from a capture with depth.

    conda activate rock_climbing_3d
    python main.py

Both models run on the VLM Run Gateway, so there are no weights to download.

Every input is a depth capture: a directory with the video, a metric depth map
per frame and the camera's intrinsics (see :mod:`src.lidar`). A plain video has
no depth to place the route with, and is the 2D demo's job — ``../rock_climbing``.

With ``BATCH_MODE`` on, every clip in ``INPUT_DIR`` is read as another attempt at
the same route and each render ends on a panel comparing its sequence of holds
against the others'. The pipeline therefore runs in two passes: every clip is
analyzed first, because no clip can be rendered until all of them are known.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import config as cfg
from src import (climb, compare, holds as holds_mod, lidar, pose, render,
                 space as space_mod, timing, video)
from src.env import load_api_key

console = Console()


def rule(step: int, title: str) -> None:
    console.rule(f"[bold cyan]{step}[/] {title}", align="left")


def kv(pairs: dict, title: str | None = None) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", justify="right")
    table.add_column()
    for key, value in pairs.items():
        table.add_row(key, str(value))
    console.print(Panel(table, title=title, title_align="left", expand=False) if title else table)


def rel(path: Path) -> Path:
    """Project-relative path, for console output."""
    try:
        return path.relative_to(cfg.PROJECT_DIR)
    except ValueError:
        return path


def _renumber_space(lifted: dict, route: list[dict]) -> dict:
    """Re-key the 3D holds onto the ids `assign_ids` just gave their flat copies.

    Each flat hold records the track it came from, which is the key the lifted
    dict is stored under, so the mapping is a lookup rather than a guess.
    """
    out = {}
    for flat in route:
        for track in flat.get("tracks", []):
            hold = lifted.get(track)
            if hold is not None:
                hold.id = flat["id"]
                out[flat["id"]] = hold
                break
    return out


def route_metadata(prompt: str, color: str | None = None) -> dict:
    """What problem this is. Recorded into every artifact the run writes."""
    return {"color": color or cfg.HOLD_COLOR, "grade": cfg.ROUTE_GRADE,
            "name": cfg.ROUTE_NAME, "prompt": prompt}


def discover() -> list[Path]:
    """The captures to process, in a stable order.

    Batch mode takes every depth capture in INPUT_DIR — each is another attempt
    at the same route, which is the assumption the comparison rests on. Sorted
    by name so the reference clip and the panel's reading order do not move
    between runs.

    Anything else in the folder is named and skipped rather than quietly
    ignored: a plain video dropped in here has no depth, and the person who put
    it there should hear that it went nowhere and where it should go instead.
    """
    if not cfg.BATCH_MODE:
        return [cfg.INPUT_CAPTURE]
    if not cfg.INPUT_DIR.is_dir():
        return []
    entries = [p for p in sorted(cfg.INPUT_DIR.iterdir()) if not p.name.startswith(".")]
    captures = [p for p in entries if lidar.is_capture(p)]
    # A capture's zip sitting beside the unpacked capture is expected, not a mistake.
    unpacked = {p.name for p in captures}
    for p in entries:
        if p in captures or (p.suffix.lower() == ".zip" and p.stem in unpacked):
            continue
        console.print(f"  [yellow]skipping {p.name}[/] — not a depth capture. "
                      + ("A video on its own has no depth; run it through "
                         "../rock_climbing (the 2D demo) instead."
                         if p.is_file() else f"Expected {lidar.CAPTURE_FORMAT}."))
    return captures


# ── one clip ─────────────────────────────────────────────────────────────────

@dataclass
class Run:
    """Everything one clip produced, up to but not including its render.

    Held whole because in batch mode the render cannot start until every clip
    has been analyzed: the panel a clip ends on is made of the other clips.
    """

    video: Path
    label: str
    run_dir: Path
    src_info: dict
    mp4: Path
    info: object
    export_mp4: Path
    export_info: object
    prompt: str
    hold_settings: dict
    colour: str
    scene: object
    route: list
    numbering: str
    ground: object
    poses: object
    frame_poses: object
    analysis: object
    util: object
    rows: list
    warmup: int
    window: tuple
    extra_body: dict
    track_id: object
    track_stats: dict
    request_timing: object
    pose_usage: dict | None
    holds_cached: bool
    poses_cached: bool
    tonemap: str | None
    tonemap_reason: str
    convert_seconds: float = 0.0
    convert_cached: bool = True
    hold_seconds: float = 0.0
    hold_cost: float = 0.0
    upload_mb: float = 0.0
    started: float = 0.0      # perf counter at the top of `analyze`, so the
                              # per-clip metrics cover the analysis too

    @property
    def aspect(self) -> float:
        """Frame width over frame height: what puts x and y into one unit."""
        return self.info.width / self.info.height

    def attempt(self) -> compare.Attempt:
        """This clip as a row in the comparison."""
        return compare.Attempt(
            label=self.label,
            video=self.video.name,
            holds=self.route,
            sequence=climb.activation_sequence(self.analysis),
            limb_sequence=climb.limb_sequence(self.analysis),
            elapsed=(round(self.analysis.elapsed, 3)
                     if self.analysis.elapsed is not None else None),
            topped_out=self.analysis.completion_frame is not None,
            n_detected=len(self.route),
            n_used=self.analysis.n_used)


def _lidar_scene(capture, mp4: Path, info) -> dict | None:
    """The wall in 3D: odometry, a fused wall, one metric model.

    The scene is a list of models with one entry, because depth gives every
    frame the same metric world and there is nothing to split; the geometry
    classes in :mod:`src.space` read it in that shape.
    """
    import hashlib

    size = (info.width, info.height)
    K = capture.K(size)
    settings = {"features": cfg.LIDAR_FEATURES, "ratio": cfg.LIDAR_MATCH_RATIO,
                "reproj": cfg.LIDAR_MAX_REPROJ_PX, "local_map": cfg.LIDAR_LOCAL_MAP,
                "every": cfg.LIDAR_KEYFRAME_EVERY, "min_inliers": cfg.LIDAR_MIN_INLIERS}
    stat = mp4.stat()
    key = json.dumps({"video": mp4.name, "size": stat.st_size, "settings": settings,
                      "depth": str(capture.root), "contract": "lidar-odometry-v1"},
                     sort_keys=True)
    cache = cfg.CACHE_DIR / f"lidar.{hashlib.sha256(key.encode()).hexdigest()[:12]}.npz"

    odo = None
    if cfg.REUSE_LIDAR and cache.is_file():
        blob = np.load(cache)
        frames = [int(f) for f in blob["frames"]]
        odo = lidar.Odometry(
            poses={f: (R, t) for f, R, t in zip(frames, blob["R"], blob["t"])},
            keyframes=[int(k) for k in blob["keyframes"]], inliers={},
            rms_px=float(blob["rms"]), path_length_m=float(blob["path"]),
            lost=[int(f) for f in blob["lost"]])
        console.print(f"  [green]cache hit[/]: camera path [dim]{rel(cache)}[/]")
    else:
        t0 = time.perf_counter()
        with console.status("[cyan]RGB-D odometry: placing every frame against the "
                            "depth[/]…", spinner="dots"):
            odo = lidar.odometry(
                mp4, capture, K, n_features=cfg.LIDAR_FEATURES,
                ratio=cfg.LIDAR_MATCH_RATIO, max_reproj_px=cfg.LIDAR_MAX_REPROJ_PX,
                local_map=cfg.LIDAR_LOCAL_MAP, keyframe_every=cfg.LIDAR_KEYFRAME_EVERY,
                min_inliers=cfg.LIDAR_MIN_INLIERS, console=console)
        frames = sorted(odo.poses)
        cfg.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache, frames=np.array(frames), R=np.array([odo.poses[f][0] for f in frames]),
            t=np.array([odo.poses[f][1] for f in frames]),
            keyframes=np.array(odo.keyframes), rms=odo.rms_px, path=odo.path_length_m,
            lost=np.array(odo.lost, dtype=int))
        console.print(f"  [green]done in {time.perf_counter() - t0:.1f}s[/]")
    if not odo.poses:
        console.print("[red]The odometry could not place a single frame.[/]")
        return None

    poses = lidar.interpolate(odo.poses, info.n_frames, max_gap=cfg.LIDAR_FILL_GAP)
    poses = lidar.smooth(poses, sigma=cfg.LIDAR_POSE_SMOOTH_SIGMA)
    points, colours = _fuse_wall(capture, mp4, poses, K)
    recon = lidar.reconstruction(odo, poses, points, K)
    centres = np.array([(-R.T @ t) for R, t in poses.values()])
    kv({
        "frames placed": f"[bold]{len(odo.poses)}[/] of {info.n_frames}"
                         + (f" (+{len(poses) - len(odo.poses)} interpolated)"
                            if len(poses) > len(odo.poses) else ""),
        "keyframes": len(odo.keyframes),
        "reprojection RMS": f"{odo.rms_px:.2f}px",
        "camera moved": f"{np.ptp(centres, axis=0).max():.2f} m across, "
                        f"{odo.path_length_m:.2f} m of path",
        "wall": f"{len(points):,} points at {cfg.LIDAR_VOXEL_M * 100:.1f} cm "
                f"[dim](fused from every {cfg.LIDAR_FUSE_STRIDE}th depth map)[/]",
    }, title="reconstruction  [dim](meters)[/]")
    return {"recon": recon,
            "models": [{"recon": recon, "poses": poses, "holds": {}}],
            "poses": poses, "frame_model": {f: 0 for f in poses}, "K": K,
            "lidar": True, "capture": capture, "colours": colours}


def _fuse_wall(capture, mp4: Path, poses: dict, K, bboxes: dict | None = None):
    """The fused cloud; with *bboxes*, the climber's box is cut out of every frame."""
    w, h = capture.depth_size
    exclude = None
    if bboxes:
        margin = cfg.LIDAR_BODY_MARGIN

        def exclude(frame):
            box = bboxes.get(frame)
            if box is None:
                return None
            x, y, bw, bh = box
            mask = np.zeros((h, w), bool)
            x0, x1 = int(max(0, (x - margin) * w)), int(min(w, (x + bw + margin) * w))
            y0, y1 = int(max(0, (y - margin) * h)), int(min(h, (y + bh + margin) * h))
            mask[y0:y1, x0:x1] = True
            return mask

    return lidar.fuse(mp4, capture, poses, K,
                      frames=list(range(0, capture.n_frames, cfg.LIDAR_FUSE_STRIDE)),
                      voxel=cfg.LIDAR_VOXEL_M, max_depth=cfg.LIDAR_MAX_DEPTH_M,
                      exclude=exclude)


def _gate_contacts(flat, route, poses, scene, info, aspect, ground, per_frame):
    """Re-read the climb with the 3D contact gate, and say what it changed.

    The 2D reading is passed in rather than thrown away, because the difference
    between the two *is* the result: every contact the depth removed was a limb
    the image put on a hold that it was not physically touching.
    """
    gate = lidar.ContactGate(
        scene["skeleton"], scene["holds"], poses, aspect=aspect,
        hand_m=cfg.CONTACT_3D_HAND_M, foot_m=cfg.CONTACT_3D_FOOT_M,
        release_m=cfg.CONTACT_3D_RELEASE_M, hand_reach_m=cfg.CONTACT_3D_HAND_REACH_M)

    # Calibration, printed: how far from the hold were the limbs the 2D test
    # called on it? Real grips cluster near zero; reaches-past are the tail.
    by_kind = {"feet to hold": [], "hand shoulders to hold": []}
    unknown = 0
    for contact in flat.contacts:
        kpt = climb.LIMB_BY_KEY[contact.limb].kpt
        for frame in range(contact.start, contact.end + 1):
            if kpt in (15, 16):
                d = gate.distance(frame, kpt, contact.hold_id)
                kind = "feet to hold"
            else:
                d = gate.reach(frame, kpt, contact.hold_id)
                kind = "hand shoulders to hold"
            if d is None:
                unknown += 1
            else:
                by_kind[kind].append(d)

    per_frame.gate = gate
    gated = climb.analyze(route, poses, fps=info.fps, aspect=aspect, cfg=cfg,
                          floor=ground, per_frame=per_frame)
    scene["contact_gate"] = gate

    def pairs(analysis):
        return {(c.limb, c.hold_id) for c in analysis.contacts}

    removed = sorted(pairs(flat) - pairs(gated))
    added = sorted(pairs(gated) - pairs(flat))
    rows = {"measured": ", ".join(
        f"{kind} median {np.median(v) * 100:.0f} cm, p90 {np.percentile(v, 90) * 100:.0f} cm"
        for kind, v in by_kind.items() if v)
        + f" [dim]({unknown} contact-frames hidden or without depth → 2D decides)[/]"}
    rows["limits"] = (f"feet within {cfg.CONTACT_3D_FOOT_M * 100:.0f} cm of the hold; "
                      + (f"hands within {cfg.CONTACT_3D_HAND_M * 100:.0f} cm; "
                         if cfg.CONTACT_3D_HAND_M is not None else "")
                      + (f"hands within {cfg.CONTACT_3D_HAND_REACH_M:.2f} m reach of "
                         f"the shoulder; " if cfg.CONTACT_3D_HAND_REACH_M else "")
                      + f"+{cfg.CONTACT_3D_RELEASE_M * 100:.0f} cm to stay on")
    rows["frames rejected"] = (f"{len(gate.log)} limb-frames where the image put a limb "
                               f"on a hold and the depth said it was not"
                               + (f" [dim](feet {sum(e['kpt'] in (15, 16) for e in gate.log)}, "
                                  f"hands {sum(e['kpt'] in (9, 10) for e in gate.log)})[/]"
                                  if gate.log else ""))
    rows["contacts removed"] = (", ".join(f"{l.replace('_', ' ')} on "
                                          f"{render.hold_label(h, cfg)}" for l, h in removed)
                                or "none")
    if added:
        rows["contacts added"] = ", ".join(f"{l.replace('_', ' ')} on "
                                           f"{render.hold_label(h, cfg)}" for l, h in added)
    rows["sequence"] = (" ".join(render.hold_label(h, cfg)
                                 for h in climb.activation_sequence(flat)) + "  →  "
                        + " ".join(render.hold_label(h, cfg)
                                   for h in climb.activation_sequence(gated)))
    kv(rows, title="contact in 3D  [dim](2D reading → with depth)[/]")
    return gated


def analyze(source: Path, *, api_key: str, run_dir: Path, batch: bool) -> Run | None:
    """Steps 2-7 for one capture: convert, reconstruct, segment, pose, read the climb.

    Everything up to the render, which in batch mode has to wait for the others.
    """
    from openai import OpenAI

    t_run = time.perf_counter()
    label = source.stem

    # A capture directory carries its own video, depth and intrinsics, and
    # without the depth there is nothing here to place the route with. So a
    # plain video stops at the door rather than half-running: the 2D demo is
    # the one built for it.
    if not lidar.is_capture(source):
        console.print(f"[red]{source.name} is not a depth capture.[/] This demo "
                      f"places the route in 3D from per-frame depth, so it needs "
                      f"{lidar.CAPTURE_FORMAT}. For a video on its own, use "
                      "../rock_climbing (the 2D demo).")
        return None
    try:
        capture = lidar.load(source)
    except (KeyError, ValueError, OSError) as exc:
        console.print(f"[red]{source.name} looks like a depth capture but could "
                      f"not be read:[/] {exc}")
        return None
    label = source.name
    source = capture.video

    # ── 2. Source ────────────────────────────────────────────────────────────
    rule(2, f"Source video{f' — [bold]{label}[/]' if batch else ''}")
    if not source.is_file():
        console.print(f"[red]Not found:[/] {source}")
        return None

    src_info = video.probe_source(source)
    kv({
        "path": rel(source),
        "size": f"{source.stat().st_size / 1e6:.1f} MB",
        "codec": f"{src_info['codec']} ({src_info['pix_fmt']})",
        "stored": f"{src_info['width']}x{src_info['height']}",
        "rotation": f"{src_info['rotation']}°"
                    + ("  [dim](applied on convert; OpenCV would ignore it)[/]"
                       if src_info["rotation"] else ""),
        "colour": f"{src_info['color_primaries'] or '?'} / {src_info['color_transfer'] or '?'}"
                  + ("  [yellow](HDR)[/]" if src_info["is_hdr"] else "  [dim](SDR)[/]"),
        "frames": f"{src_info['n_frames']} ({src_info['duration']:.1f}s)",
        "depth": f"[bold green]LiDAR[/] {capture.depth_size[0]}x{capture.depth_size[1]}, "
                 f"{int(capture.valid.sum())} of {capture.n_frames} frames "
                 f"[dim]({capture.meta.get('device', '?')}, meters)[/]",
    })

    # ── 3. Convert (cached) ──────────────────────────────────────────────────
    rule(3, "Convert to MP4")
    # Decided before the cache key is built, so switching tone-mapping on or off
    # produces a different MP4 rather than silently reusing the other one.
    tonemap, tonemap_reason = video.tonemap_filter(
        src_info, mode=cfg.TONEMAP, algorithm=cfg.TONEMAP_ALGORITHM,
    )
    console.print(f"  tone-map: {'[green]on[/] — ' if tonemap else ''}{tonemap_reason}")

    def prepare(height, tag):
        """Convert to *height* (None = source), reusing a cached copy if present."""
        path = video.cache_path(source, cfg.CACHE_DIR, target_height=height,
                                trim_seconds=cfg.TRIM_SECONDS, crf=cfg.CONVERT_CRF,
                                tonemap=tonemap, passthrough=True)
        seconds, cached = 0.0, True
        if path.is_file() and not cfg.FORCE_RECONVERT:
            console.print(f"  [green]cache hit[/] ({tag}): [dim]{rel(path)}[/]")
        else:
            cached = False
            reason = "FORCE_RECONVERT" if path.is_file() else "no cached copy"
            console.print(f"  [yellow]converting[/] ({tag}, {reason})")
            t0 = time.perf_counter()
            video.convert(source, path, target_height=height,
                          trim_seconds=cfg.TRIM_SECONDS, crf=cfg.CONVERT_CRF,
                          total_frames=src_info["n_frames"], console=console,
                          tonemap=tonemap, passthrough=True)
            seconds = time.perf_counter() - t0
            console.print(f"  done in {seconds:.1f}s -> [dim]{rel(path)}[/]")
        return path, video.inspect(path), seconds, cached

    # Two renditions, because they answer to different limits: the models cap at
    # a 2048px long edge internally, so uploading more is pure wait, while the
    # export answers only to what you want to watch. Every coordinate on both
    # sides is normalized, so one inference serves any render size.
    mp4, info, convert_seconds, convert_cached = prepare(cfg.INFERENCE_HEIGHT, "inference")

    if cfg.EXPORT_HEIGHT == cfg.INFERENCE_HEIGHT:
        export_mp4, export_info = mp4, info
    else:
        export_mp4, export_info, extra_s, extra_cached = prepare(cfg.EXPORT_HEIGHT, "export")
        convert_seconds += extra_s
        convert_cached = convert_cached and extra_cached

    console.print(f"  inference [bold]{info}[/]  ({mp4.stat().st_size / 1e6:.1f} MB)")
    if not cfg.TRIM_SECONDS and info.n_frames != capture.n_frames:
        # Depth slab i belongs to video frame i. Off by one frame anywhere and
        # every hold after it is placed from the wrong depth map.
        console.print(f"[red]The converted clip has {info.n_frames} frames and the "
                      f"capture has {capture.n_frames} depth maps.[/] They must pair "
                      "one to one; set FORCE_RECONVERT = True and try again.")
        return None
    if export_mp4 is not mp4:
        console.print(f"  export    [bold]{export_info}[/]")

    # ── 4. The wall, in three dimensions ─────────────────────────────────────
    rule(4, "The wall, in three dimensions (LiDAR)")
    scene = _lidar_scene(capture, mp4, info)
    if scene is None:
        return None
    K = scene["K"]

    # ── 6. The route ─────────────────────────────────────────────────────────
    rule(5, "The route (SAM 3.1, tracked)")
    colour = cfg.HOLD_COLOR
    prompt = cfg.HOLD_PROMPT.format(color=colour)
    # One `track` call keeps at most 128 samples and initialises once, so a long
    # clip is cut into segments rather than having its stride coarsened. See
    # `holds.segment_plan` — on a clip where the camera walks all the way round,
    # segmenting is not an optimisation, it is the only way the second half gets
    # tracked at all.
    plan = holds_mod.segment_plan(info.n_frames, cfg.HOLD_TRACK_STRIDE,
                                  max_frames=cfg.HOLD_TRACK_MAX_FRAMES)
    hold_settings = {"stride": cfg.HOLD_TRACK_STRIDE,
                     "max_frames": cfg.HOLD_TRACK_MAX_FRAMES,
                     "min_score": cfg.HOLD_MIN_SCORE, "segments": len(plan)}
    holds_cache = holds_mod.cache_path(mp4, cfg.CACHE_DIR, model=cfg.HOLD_MODEL,
                                       prompt=prompt, settings=hold_settings)
    cached_holds = holds_mod.load_cache(holds_cache) if cfg.REUSE_HOLDS else None

    n_samples = sum(-(-length // cfg.HOLD_TRACK_STRIDE) for _, length in plan)
    kv({"model": cfg.HOLD_MODEL,
        "method": f"track [dim]({len(plan)} call{'s' if len(plan) > 1 else ''}, "
                  f"{'segmented' if len(plan) > 1 else 'whole clip'})[/]",
        "prompt": f"[bold]{prompt}[/]",
        "sampled": f"every {cfg.HOLD_TRACK_STRIDE} frames — {n_samples} looks, "
                   f"one every {cfg.HOLD_TRACK_STRIDE / info.fps:.2f}s"},
       title="request")

    hold_seconds = 0.0
    hold_usage = None
    if cached_holds is not None:
        payload, hold_usage, created = cached_holds
        console.print(f"  [green]cache hit[/]: tracks from [dim]{created}[/] "
                      f"([dim]{holds_cache.name}[/]); no gateway call")
        console.print("  [dim]set REUSE_HOLDS = False in config.py to re-track[/]")
    else:
        if cfg.REUSE_HOLDS:
            console.print("  [yellow]no cached tracks for these settings[/], "
                          "calling the gateway")
        client = OpenAI(base_url=cfg.GATEWAY_BASE_URL, api_key=api_key,
                        timeout=cfg.REQUEST_TIMEOUT, max_retries=1)
        parts, usages = [], []
        t0 = time.perf_counter()
        for n, (start, length) in enumerate(plan):
            if len(plan) == 1:
                clip = mp4
            else:
                clip = cfg.CACHE_DIR / f"{mp4.stem}.seg{n}_{start}_{length}.mp4"
                if not clip.is_file():
                    video.segment(mp4, clip, start_frame=start, n_frames=length,
                                  fps=info.fps)
            with console.status(f"[cyan]waiting on the gateway[/]; SAM is tracking "
                                f"segment {n + 1}/{len(plan)}…", spinner="dots"):
                part, usage = holds_mod.request_track(
                    client, model=cfg.HOLD_MODEL,
                    video_b64=pose.encode_video(clip), prompt=prompt,
                    skip_frames=cfg.HOLD_TRACK_STRIDE,
                    max_frames=cfg.HOLD_TRACK_MAX_FRAMES)
            # Each segment numbers its frames and tracks from scratch, so both
            # are shifted into the whole clip's space before they are fused.
            parts.append(holds_mod.shift(part, frame_offset=start,
                                         track_offset=n * 1000))
            if usage:
                usages.append(usage)
            if len(plan) > 1:
                items, _ = holds_mod.unwrap(parts[-1])
                console.print(f"  segment {n + 1}/{len(plan)} "
                              f"[dim](frames {start}-{start + length - 1})[/]: "
                              f"{len({i['track_id'] for i in items})} tracks")
        hold_seconds = time.perf_counter() - t0
        payload = holds_mod.concat(parts)
        hold_usage = {"cost": sum(u.get("cost") or 0.0 for u in usages)}
        holds_mod.save_cache(holds_cache, payload, hold_usage,
                             stamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        console.print(f"  [green]done in {hold_seconds:.1f}s[/]")

    # Everything below the cache is free to re-tune: the reply is what cost a
    # hundred seconds, and a threshold should not be something you pay to change.
    sightings = holds_mod.observations(payload, min_score=cfg.HOLD_MIN_SCORE)
    _, sampled_frames = holds_mod.unwrap(payload)
    sampled = [int(f["frame_id"]) for f in sampled_frames]
    n_tracks = len({o.track for o in sightings})

    # SAM's job is to carry an identity through the clip, and ours is to put
    # every sighting of a hold in one place and see whether they agree. The
    # place is the world the depth measured, and "agree" is a distance in meters.
    width, height = info.width, info.height
    by_track: dict[int, list] = {}
    for item in sightings:
        points = item.polygon
        if len(points) < 3:
            x, y, w, h = item.bbox
            points = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
        by_track.setdefault(item.track, []).append(
            (item.frame, np.asarray(points) * [width, height]))

    model_scene = scene["models"][0]
    model = model_scene["recon"]
    camera_poses = model_scene["poses"]
    observations = {
        track_id: [(f, p) for f, p in sighted if f in camera_poses]
        for track_id, sighted in by_track.items()
    }
    observations = {k: v for k, v in observations.items() if len(v) >= 2}
    span = lidar.scene_scale(model.points)
    # Placed, not triangulated: the depth under each sighting's mask is the
    # hold, in meters, from one view as well as from ten.
    lifted = lidar.lift(
        observations, camera_poses, capture, K, (width, height), model.points,
        neighbourhood=cfg.LIDAR_HOLD_NORMAL_RADIUS_M,
        max_spread_m=cfg.LIDAR_HOLD_SPREAD_M,
        min_views=cfg.LIDAR_HOLD_MIN_VIEWS, console=console)
    if len(lifted) >= 4:
        centres = np.array([h.centre for h in lifted.values()])
        middle = np.median(centres, axis=0)
        reach = cfg.LIDAR_HOLD_MAX_REACH * span
        for key in [k for k, h in lifted.items()
                    if np.linalg.norm(h.centre - middle) > reach]:
            lifted.pop(key)
    lifted, merges3d = lidar.merge_lifted(lifted, radius=cfg.LIDAR_HOLD_MERGE_M)
    for a, b in merges3d:
        console.print(f"  [dim]tracks {a} and {b} land in the same place[/]")
    model_scene.update({"holds": lifted, "span": span})
    if not lifted:
        console.print("[red]No hold could be placed from the depth.[/] Check that "
                      "the route is inside the sensor's range "
                      f"(LIDAR_MAX_DEPTH_M = {cfg.LIDAR_MAX_DEPTH_M}) and that "
                      "SAM found it — see the tracks above.")
        return None

    eye = model.centre(model.keyframes[len(model.keyframes) // 2])
    # Up from the floor, which the dense cloud has in it: gravity.
    axes, floor_fit = lidar.wall_axes(lifted, model.points, eye,
                                      lidar.camera_up(camera_poses))
    model_scene["floor"] = floor_fit
    if floor_fit is not None:
        lean = getattr(axes, "lean_deg", None)
        console.print(
            f"  floor: [bold]{floor_fit[2]:,}[/] points on one plane, "
            f"{-floor_fit[1]:.2f} m below the phone at the first frame"
            + (f"; the wall is [bold]{abs(lean):.0f}° "
               f"{'overhanging' if lean > 0 else 'slabby'}[/] "
               f"[dim](plywood plane against the floor)[/]"
               if lean is not None and abs(lean) >= 1 else
               "; the wall is vertical" if lean is not None else ""))
    else:
        console.print("  [yellow]no floor plane found[/] in the cloud — "
                      "up is taken from the route's spread instead")
    model_scene["axes"] = axes
    route = space_mod.as_flat_holds(lifted, axes)
    for flat in route:
        flat["model"] = 0
    spreads = [h.angle_deg for h in lifted.values()]   # mm, see lidar.lift

    scene.update({"primary_model": 0, "recon": model, "holds": lifted,
                  "axes": axes, "span": span})
    hold_cost = float((hold_usage or {}).get("cost") or 0.0)
    kv({
        "tracks returned": f"{n_tracks} over {len(sampled)} sampled frames "
                           f"({len(sightings)} sightings)",
        "holds placed": f"[bold]{len(route)}[/] of {len(by_track)} tracks",
        "sightings agree": f"median {np.median(spreads):.0f} mm, worst "
                           f"{max(spreads):.0f} mm from each hold's place "
                           f"[dim](dropped past "
                           f"{cfg.LIDAR_HOLD_SPREAD_M * 100:.0f} cm)[/]",
        "cost": f"${hold_cost:.4f}",
    }, title="response")

    if not route:
        console.print(f"[red]No holds found for {prompt!r}.[/] Try another HOLD_COLOR, "
                      "or lower HOLD_MIN_SCORE in config.py.")
        return None

    # ── 5b. The floor ────────────────────────────────────────────────────────
    # No segmentation call here, and none needed: the floor is already in the
    # cloud, fitted above as the plane `up` is taken from. Where it meets the
    # route's own wall plane is the wall-floor junction the start rule and the
    # clock measure against. See space.GroundPlane.
    ground = None
    if cfg.DETECT_FLOOR and len(model.points):
        plane = space_mod.GroundPlane(
            axes, model.points, camera_poses, model.K, (width, height),
            percentile=cfg.LIDAR_GROUND_PERCENTILE)
        if floor_fit is not None:
            # The measured floor, not a percentile of the cloud: `up` is its
            # normal, so its height along `up` is simply where it sits.
            normal, offset, _ = floor_fit
            plane.height = float((normal * offset - axes.origin) @ axes.up)
            plane.line = plane._line_at(plane.height)
            plane._cache = {}
        ground = space_mod.MultiModelGround({0: plane}, scene["frame_model"])
        console.print("  [dim]floor plane fitted to the LiDAR cloud (it is also "
                      "the panel's up); the clock measures against the "
                      "wall-floor junction[/]" if floor_fit is not None else
                      f"  [dim]no floor plane in the cloud, so the ground is the "
                      f"bottom {cfg.LIDAR_GROUND_PERCENTILE:.0f}% of it[/]")

    # ── 6. Pose ──────────────────────────────────────────────────────────────
    rule(6, "The climber (ViTPose)")
    video_b64, extra_body = pose.build_request(
        mp4, every_frame=cfg.EVERY_FRAME, fps=info.fps, n_frames=info.n_frames,
        video_fps=cfg.VIDEO_FPS, video_max_frames=cfg.VIDEO_MAX_FRAMES,
        precision=cfg.PRECISION)
    upload_mb = len(video_b64) / 1e6
    kv({
        "model": cfg.POSE_MODEL,
        "mode": ("every frame, detector stride 1, nothing skipped"
                 if cfg.EVERY_FRAME else f"video_fps={cfg.VIDEO_FPS} (detector cadence)"),
        "billed units": f"~{extra_body.get('video_max_frames') or min(info.n_frames, 900)} frames",
        "upload": f"{upload_mb:.1f} MB base64",
    }, title="request")

    poses_cache = pose.cache_path(mp4, cfg.CACHE_DIR, model=cfg.POSE_MODEL,
                                  extra_body=extra_body)
    cached_poses = pose.load_cache(poses_cache) if cfg.REUSE_POSES else None

    if cached_poses is not None:
        payload, pose_usage, cached_timing, created = cached_poses
        request_timing = timing.RequestTiming(**cached_timing) if cached_timing \
            else timing.RequestTiming()
        elapsed = request_timing.round_trip
        console.print(f"  [green]cache hit[/]: poses from [dim]{created}[/] "
                      f"([dim]{poses_cache.name}[/]); no gateway call")
        console.print("  [dim]set REUSE_POSES = False in config.py to re-run detection[/]")
    else:
        if cfg.REUSE_POSES:
            console.print("  [yellow]no cached poses for these settings[/], calling the gateway")
        marks: dict = {}
        http_client = timing.timed_http_client(cfg.REQUEST_TIMEOUT, marks)
        client = OpenAI(base_url=cfg.GATEWAY_BASE_URL, api_key=api_key,
                        timeout=cfg.REQUEST_TIMEOUT, max_retries=1,
                        **({"http_client": http_client} if http_client else {}))
        t0 = time.perf_counter()
        with console.status("[cyan]waiting on the gateway[/]; pose runs on every decoded frame…",
                            spinner="dots"):
            payload, pose_usage = pose.request_poses(
                client, model=cfg.POSE_MODEL, video_b64=video_b64, extra_body=extra_body)
        elapsed = time.perf_counter() - t0
        request_timing = timing.summarize(marks, elapsed)
        pose.save_cache(poses_cache, payload, pose_usage, asdict(request_timing),
                        stamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        if request_timing.measured_split:
            console.print(f"  [green]done in {elapsed:.1f}s[/]: "
                          f"upload {request_timing.upload:.1f}s, "
                          f"[bold]server {request_timing.server:.1f}s[/], "
                          f"download {request_timing.download:.1f}s")
        else:
            console.print(f"  [green]done in {elapsed:.1f}s[/]")
    del video_b64   # tens of MB per clip, and a batch holds every clip at once

    items, returned_frames = pose.unwrap(payload)

    # Which body is the climber, decided once against the route: each frame's
    # boxes are scored against the holds projected into that same frame.
    probe = space_mod.MultiModelFrameGeometry(
        scene["models"], scene["frame_model"], (width, height),
        bbox_margin=cfg.HOLD_BBOX_MARGIN, aspect=width / height)
    track_id, track_stats = pose.pick_track(
        items, mask=None, per_frame=True,
        overlap_fn=space_mod.on_wall_scorer(probe),
        min_overlap=cfg.MIN_WALL_OVERLAP, lock_to_wall=cfg.LOCK_TO_WALL)
    frame_poses = pose.resolve(items, returned_frames, track_id,
                               min_kpt_score=cfg.POSE_MIN_KPT_SCORE)
    # The holds come to the keypoints, projected into each frame, so smoothing
    # happens in frame coordinates. That mixes a little camera motion into the
    # body's path, which is small at σ = 2 frames.
    poses = pose.smooth(frame_poses, sigma=cfg.POSE_SMOOTH_SIGMA)

    coverage = poses.n_frames_returned / info.n_frames if info.n_frames else 0
    kv({
        "frames returned": f"{poses.n_frames_returned} of {info.n_frames} source ({coverage:.0%})",
        "tracks seen": (lambda ranked: ", ".join(
            f"{t}{' [bold](climber)[/]' if t == track_id else ''} "
            f"[dim]({st['coverage']:.0%} of frames, {st['mean_overlap']:.2f} "
            f"on-wall)[/]" for t, st in ranked[:6])
            + (f" [dim]… and {len(ranked) - 6} more[/]" if len(ranked) > 6 else "")
        )(sorted(track_stats.items(), key=lambda kv: -kv[1]["coverage"])) or "none",
        "frames with the climber": poses.n_posed,
        "joint confidence": (f"≥ {cfg.POSE_MIN_KPT_SCORE}" if cfg.POSE_MIN_KPT_SCORE
                             else "off (the (0, 0) sentinel alone)"),
        "smoothing": (f"σ = {cfg.POSE_SMOOTH_SIGMA} frames, in frame space"
                      if cfg.POSE_SMOOTH_SIGMA else "off"),
        "usage": pose_usage or "—",
    }, title="response")

    if not poses.n_posed:
        console.print("[red]No climber found in any frame.[/]")
        return None

    # Now that the climber has a box in every frame, the wall is fused again
    # without them — the first fusion, used to orient the holds, had a whole
    # climb's worth of them smeared up its face. And with depth, the body itself
    # has a place: each joint is read off the depth under it.
    points, colours = _fuse_wall(capture, mp4, scene["poses"], K,
                                 bboxes=frame_poses.bboxes)
    scene["recon"].points = points
    scene["colours"] = colours
    scene["skeleton"] = lidar.lift_skeleton(
        capture, scene["poses"], K, (width, height), frame_poses.kpts, frame_poses.valid)
    console.print(f"  [dim]wall re-fused without the climber: {len(points):,} "
                  f"points; climber placed in 3D on "
                  f"{len(scene['skeleton'])} frames from the depth under each "
                  f"joint[/]")

    # Written now rather than with the rest of the artifacts, so the raw
    # response can be dropped before the next clip is loaded: a batch otherwise
    # holds every clip's frame-by-frame keypoints in memory at once.
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "poses.json").write_text(json.dumps(payload))
    del payload, items

    # The colour prompt finds every hold of that colour in shot, which is more
    # wall than this route. The body's own trajectory is what narrows it.
    if cfg.HOLD_SPATIAL_FILTER:
        kept = {}
        # Not `track_id`: that name is the climber's pose track, recorded below.
        for hold_track, hold in model_scene["holds"].items():
            near_body = False
            for frame in model_scene["poses"]:
                box = poses.bboxes.get(frame)
                polygon = probe.project(hold, frame)
                if box is None or polygon is None:
                    continue
                x, y, w, h = box
                margin = cfg.HOLD_SPATIAL_MARGIN
                cx, cy = polygon.mean(axis=0)
                if (x - margin <= cx <= x + w + margin
                        and y - margin <= cy <= y + h + margin):
                    near_body = True
                    break
            if near_body:
                kept[hold_track] = hold
        dropped = len(model_scene["holds"]) - len(kept)
        model_scene["holds"] = kept
        scene["holds"] = kept
        before = len(route)
        if dropped:
            console.print(f"  [dim]{dropped} holds dropped: never near the climber "
                          f"(HOLD_SPATIAL_MARGIN = {cfg.HOLD_SPATIAL_MARGIN})[/]")
        if not kept:
            console.print("[red]No reconstructed hold lies near the climber.[/]")
            return None
        # Redrawn over the holds that are left rather than filtered: numbered
        # off the old set, a hold across the gym that was just dropped would
        # still be setting the scale — and "bottom to top" is measured along
        # real gravity.
        route = space_mod.as_flat_holds(kept, axes)
        for flat in route:
            flat["model"] = 0
        console.print(f"  [dim]{len(route)} of {before} reconstructed holds remain "
                      "in the climber's route region[/]")
    route, numbering = holds_mod.assign_ids(
        route, band=cfg.HOLD_NUMBER_BAND, direction=cfg.HOLD_NUMBERING,
        deadband=cfg.HOLD_LEAN_DEADBAND)
    # The flat copies have just been renumbered 1..n; the 3D holds they were
    # made from still carry SAM's track ids, and those are what the geometry the
    # climb is measured against reports. Without this the panel says "hold 4"
    # and the sequence says "hold 2013" — the same hold, two names.
    model_scene["holds"] = _renumber_space(model_scene["holds"], route)
    scene["holds"] = model_scene["holds"]

    # ── 7. The climb ─────────────────────────────────────────────────────────
    rule(7, "The climb")
    # Frame coordinates throughout: the holds are projected into each frame to
    # meet the keypoints there. The aspect is the frame's, which is what makes
    # HOLD_MASK_MARGIN one distance in x and y.
    aspect = width / height
    per_frame = space_mod.MultiModelFrameGeometry(
        scene["models"], scene["frame_model"], (width, height),
        bbox_margin=cfg.HOLD_BBOX_MARGIN, aspect=aspect)
    scene["geometry"] = per_frame
    per_frame.gate = None
    analysis = climb.analyze(route, poses, fps=info.fps, aspect=aspect, cfg=cfg,
                             floor=ground, per_frame=per_frame)
    if cfg.CONTACT_3D:
        analysis = _gate_contacts(analysis, route, poses, scene, info, aspect,
                                  ground, per_frame)
    if scene.get("skeleton"):
        # Now the climb says which hand was on which hold, the hands can be put
        # where the holds are rather than where the forearm's depth says — see
        # `lidar.pin_hands`. Everything drawn and measured in 3D after this
        # (the skeleton, the body's reach) uses the corrected wrists.
        moved = lidar.pin_hands(
            scene["skeleton"], analysis.holding, scene["holds"], scene["poses"], K,
            frame_poses.kpts, frame_poses.valid, (width, height), axes.out)
        console.print(f"  [dim]{moved} gripping-wrist positions placed at their hold "
                      f"rather than at the forearm's depth[/]")

    covered = sorted(scene["poses"])
    console.print(f"  [dim]the climb is read over frames {covered[0]}-{covered[-1]}, "
                  f"where the camera could be placed; outside that the route has "
                  f"no position and no contact is recorded[/]")

    lean = holds_mod.route_lean(route)
    console.print(f"  [dim]numbered bottom to top, "
                  f"{'left to right' if numbering == 'ltr' else 'right to left'} "
                  f"within a row (lean {lean:+.3f}"
                  + (", forced" if cfg.HOLD_NUMBERING != "auto"
                     else f", |lean| < {cfg.HOLD_LEAN_DEADBAND} so the default"
                     if abs(lean) <= cfg.HOLD_LEAN_DEADBAND else "") + ")[/]")

    # The utilization window is the climb itself, so warm-up touches before the
    # clock started are not counted into the split.
    frames = poses.frames()
    window = (analysis.start_frame if analysis.start_frame is not None else frames[0],
              analysis.completion_frame if analysis.completion_frame is not None
              else frames[-1])
    # One window governs the panel, the pips and every reported figure, so they
    # cannot disagree about whether a warm-up touch counts.
    warmup = climb.restrict(analysis, window)
    util = climb.Utilization(analysis, window=window)
    rows = climb.hold_times(analysis, cfg=cfg)

    table = Table.grid(padding=(0, 3))
    table.add_column(style="dim", justify="right")
    table.add_column()
    table.add_row("holds on route", f"[bold]{len(route)}[/]")
    table.add_row("holds used", f"[bold]{analysis.n_used}[/] of {len(route)}")
    table.add_row("start", f"frame {analysis.start_frame} "
                           f"({analysis.start_frame / info.fps:.2f}s)"
                  if analysis.start_frame is not None else "[yellow]not detected[/]")
    table.add_row("topped out", f"frame {analysis.completion_frame} "
                                f"(hold {render.hold_label(analysis.final_hold_id, cfg)})"
                  if analysis.completion_frame is not None else "[yellow]no[/]")
    if analysis.elapsed is not None:
        table.add_row("time on the wall", f"[bold green]{analysis.elapsed:.2f}s[/]")
    table.add_row("sequence", "[bold]"
                  + " ".join(render.hold_label(h, cfg)
                             for h in climb.activation_sequence(analysis))
                  + "[/]" if analysis.n_used else "—")
    console.print(Panel(table, title="[bold]climb[/]", title_align="left", expand=False))

    limb_table = Table.grid(padding=(0, 3))
    limb_table.add_column(style="dim", justify="right")
    limb_table.add_column(justify="right")
    limb_table.add_column(justify="right")
    limb_table.add_column(justify="right")
    limb_table.add_column()
    limb_table.add_row("", "contact", "on hold", "holds", "in order (the beta)")
    for limb in climb.LIMBS:
        limb_table.add_row(
            limb.name,
            f"{util.seconds(limb.key):.2f}s",
            f"[bold]{util.coverage(limb.key) * 100:.0f}%[/]",
            f"{len(util.holds_by_limb[limb.key])}",
            " ".join(render.hold_label(h, cfg) for h in util.sequence(limb.key)) or "—")
    console.print(Panel(
        limb_table,
        title="[bold]limb utilization[/]  [dim]on hold = this limb's contact time "
              "over the climb; the four do not add to 100%[/]",
        title_align="left", expand=False))

    for problem in climb.check(analysis, util):
        console.print(f"  [red]inconsistent:[/] {problem}")

    if analysis.start_frame is None:
        console.print("  [yellow]warning[/] no start detected: both feet never dwelt on a "
                      "hold, so the clock and the route line are off. Lower "
                      "HOLD_DWELL_SECONDS or raise ANKLE_TO_TOE_OFFSET.")
    if analysis.completion_frame is None:
        console.print("  [yellow]warning[/] no top-out: both wrists never dwelt on hold "
                      f"{analysis.final_hold_id} together. Lower "
                      "FINAL_HOLD_DWELL_SECONDS, or check that the top hold was found.")

    return Run(
        video=source, label=label, run_dir=run_dir, src_info=src_info, mp4=mp4,
        info=info, export_mp4=export_mp4, export_info=export_info, prompt=prompt,
        hold_settings=hold_settings, colour=colour, scene=scene,
        route=route, numbering=numbering, ground=ground,
        poses=poses, frame_poses=frame_poses, analysis=analysis, util=util,
        rows=rows, warmup=warmup,
        window=window, extra_body=extra_body, track_id=track_id,
        track_stats=track_stats, request_timing=request_timing, pose_usage=pose_usage,
        holds_cached=cached_holds is not None, poses_cached=cached_poses is not None,
        tonemap=tonemap, tonemap_reason=tonemap_reason,
        convert_seconds=convert_seconds, convert_cached=convert_cached,
        hold_seconds=hold_seconds, hold_cost=hold_cost,
        upload_mb=upload_mb, started=t_run)


# ── the comparison ───────────────────────────────────────────────────────────

def compare_runs(runs: list[Run]) -> compare.Comparison | None:
    """Put every attempt into one numbering, and say where the walls disagree.

    This is the step that makes "1 2 5 8" from one clip mean the same thing as
    "1 2 5 8" from another. Without it the numbers are four private languages
    that happen to share an alphabet.

    **Not yet valid**, and it stops rather than pretending otherwise. The
    alignment matches holds between clips by position, which needs every take
    in one coordinate system. Each capture's world is anchored to wherever its
    own camera started, so two captures are not comparable until their walls
    have been registered to each other — a tractable job (both are metric, so
    it is one rigid transform between two fused clouds) that is not done. Until
    it is, comparing the positions would silently align holds that are nothing
    to do with each other, which is worse than not comparing them.
    """
    if len(runs) < 2:
        return None

    rule(8, "The same route, several ways")
    if not cfg.COMPARE_ATTEMPTS:
        console.print(
            "  [yellow]off:[/] each capture's world is anchored to where its own "
            "camera started, so hold positions are not comparable between clips "
            "until the two walls have been registered to each other. The clips "
            "are analyzed separately instead. Set COMPARE_ATTEMPTS = True to run "
            "the alignment anyway — it will not error, it will quietly match holds "
            "that have nothing to do with each other.")
        return None

    comparison = compare.build(
        [run.attempt() for run in runs],
        aspect=runs[0].aspect,
        max_distance=cfg.HOLD_MATCH_DISTANCE,
        drift_warn=cfg.HOLD_DRIFT_WARN,
        align_ids=cfg.ALIGN_HOLD_IDS,
        route=route_metadata(runs[0].prompt))

    reference_run = next(r for r in runs if r.label == comparison.reference)
    counts = sorted({len(run.route) for run in runs})
    console.print(f"  reference wall: [bold]{comparison.reference}[/] "
                  f"([dim]{len(reference_run.route)} holds — the count the clips "
                  f"agree on{'' if len(counts) == 1 else f'; the clips found {counts}'}; "
                  f"every other clip's holds are matched onto it by position[/])")

    table = Table.grid(padding=(0, 3))
    for column, justify in (("attempt", "right"), ("holds", "right"),
                            ("matched", "right"), ("time", "right"), ("sequence", "left")):
        table.add_column(justify=justify, style="dim" if column == "attempt" else None)
    table.add_row("", "found", "matched", "time", "sequence (reference numbering)")
    shortest = comparison.shortest()
    for attempt in comparison.attempts:
        stats = attempt.alignment
        sequence = " ".join(str(h) for h in attempt.aligned_sequence) or "—"
        tag = "  [bold yellow]shortest[/]" if shortest and \
            attempt.label == shortest.label else ""
        table.add_row(
            attempt.label,
            str(attempt.n_detected),
            f"{stats['matched']}/{stats['reference_holds']}",
            f"{attempt.elapsed:.2f}s" if attempt.elapsed is not None else "—",
            f"{sequence}{tag}")
    console.print(Panel(table, title="[bold]attempts[/]", title_align="left", expand=False))

    if cfg.CHECK_HOLD_CONSISTENCY:
        if comparison.problems:
            for problem in comparison.problems:
                console.print(f"  [yellow]hold check:[/] {problem}")
            if cfg.HOLD_CONSISTENCY_STRICT:
                console.print("  [red]HOLD_CONSISTENCY_STRICT is on[/] — stopping rather "
                              "than comparing sequences that may not line up")
                raise SystemExit(1)
            console.print("  [dim]sequences are compared in the reference clip's "
                          "numbering, so the matched holds still line up[/]")
        else:
            offsets = [a.alignment["max_offset"] for a in comparison.attempts
                       if a.alignment.get("max_offset") is not None]
            console.print(f"  [green]walls agree[/]: every clip found the same "
                          f"{comparison.attempts[0].n_detected} holds, within "
                          f"{max(offsets):.4f} of a frame height of each other")
    return comparison


# ── one clip's output ────────────────────────────────────────────────────────

def adopt_numbering(runs: list[Run], comparison: compare.Comparison) -> None:
    """Renumber every clip's holds into the run's one numbering.

    Up to here the alignment only reaches the comparison: each clip still draws
    its own SAM ids on the wall, so a clip whose wall segmented differently
    shows one number in the video and another on the card under it — the video
    says you matched hold 15, the card says 14. They are the same hold. Nothing
    but the numbering disagrees, and the numbering is ours to choose.

    So the clip adopts it. From here the ids in the overlay, route3d.png,
    summary.txt, the CSVs and the JSON are all the reference clip's, and the
    only artifact that still speaks a clip's private numbering is its cache.
    """
    if not comparison.aligned:
        return

    renumbered = []
    for run in runs:
        attempt = comparison.by_label(run.label)
        if attempt is None or not attempt.numbering:
            continue
        # `run.route` is the same list of dicts as `run.analysis.holds`, so this
        # renumbers the wall and the climb together.
        climb.renumber(run.analysis, attempt.numbering)
        # Both read hold ids at construction and keep their own copies.
        run.util = climb.Utilization(run.analysis, window=run.window)
        run.rows = climb.hold_times(run.analysis, cfg=cfg)
        # The attempt's own-id sequence is now the aligned one; leaving it as it
        # was would put two numberings back into comparison.json.
        attempt.sequence = list(attempt.aligned_sequence)
        if attempt.alignment.get("renumbered") or attempt.alignment.get("unmatched_in_other"):
            renumbered.append(run.label)

    if renumbered:
        console.print(f"  [dim]{', '.join(renumbered)} renumbered onto "
                      f"{comparison.reference}'s wall — every artifact and the "
                      f"ids drawn on the video now use that numbering[/]")


def _write_scene(run: Run) -> None:
    """The metric scene, for looking at outside the video.

    ``wall.ply`` is the fused wall with the route's holds painted in, which opens
    in any point-cloud viewer (MeshLab, CloudCompare, Blender) — in meters, with
    gravity along +y. ``scene3d.npz`` is everything the panel was drawn from.
    """
    scene = run.scene
    axes = scene["axes"]
    points = axes.to_wall(scene["recon"].points)
    colours = scene["colours"].copy()
    hold_points, hold_colours = [], []
    colour = cfg.HOLD_RENDER_COLOR.get(run.colour, cfg.HOLD_COLOR_FALLBACK)
    for hold in scene["holds"].values():
        outline = axes.to_wall(hold.polygon)
        closed = np.vstack([outline, outline[:1]])
        for a, b in zip(closed, closed[1:]):
            steps = max(2, int(np.linalg.norm(b - a) / 0.004))
            hold_points.append(np.linspace(a, b, steps))
            hold_colours.append(np.tile(colour, (steps, 1)))
    if hold_points:
        points = np.vstack([points, *hold_points])
        colours = np.vstack([colours, *hold_colours]).astype(np.uint8)
    ply = run.run_dir / "wall.ply"
    with ply.open("wb") as handle:
        handle.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\n"
                      "property float x\nproperty float y\nproperty float z\n"
                      "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                      "end_header\n").encode())
        record = np.zeros(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                              ("r", "u1"), ("g", "u1"), ("b", "u1")])
        record["x"], record["y"], record["z"] = points.T
        record["r"], record["g"], record["b"] = colours[:, 2], colours[:, 1], colours[:, 0]
        handle.write(record.tobytes())
    np.savez_compressed(
        run.run_dir / "scene3d.npz", points=scene["recon"].points, colours=scene["colours"],
        origin=axes.origin, right=axes.right, up=axes.up, out=axes.out,
        hold_ids=np.array(list(scene["holds"])),
        hold_centres=np.array([h.centre for h in scene["holds"].values()]),
        floor=np.r_[scene["models"][0]["floor"][:2]]
        if scene["models"][0].get("floor") is not None else np.zeros(0),
        skeleton_frames=np.array(sorted(scene.get("skeleton", {}))),
        skeleton_points=np.array([scene["skeleton"][f][0] for f in sorted(scene.get("skeleton", {}))]),
        skeleton_ok=np.array([scene["skeleton"][f][1] for f in sorted(scene.get("skeleton", {}))]))
    console.print(f"  wall.ply -> [dim]{rel(ply)}[/] [dim]({len(points):,} points, meters, "
                  f"+y up; open in any point-cloud viewer)[/]")


def deliver(run: Run, comparison: compare.Comparison | None, *, step: int,
            batch: bool) -> Path:
    """The last steps for one clip: render it, then write everything it produced."""
    rule(step, f"Render{f' — [bold]{run.label}[/]' if batch else ''}")
    run_dir = run.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    analysis, info, export_info = run.analysis, run.info, run.export_info

    # Written here rather than at the end of `analyze`: the ids are not settled
    # until every clip has been read and the numbering chosen.
    for hold in run.route:
        hold["label"] = render.hold_label(hold["id"], cfg)   # what the video shows
    (run_dir / "holds.json").write_text(json.dumps(run.route, indent=2))

    if cfg.SAVE_HOLDS_PREVIEW:
        # The route rendered face-on over the fused wall: the image to look at
        # first when the reconstruction seems wrong. A route that comes out as a
        # smear of holds at random depths is visible here in a way it is not in
        # any single frame.
        scene = run.scene
        outline = np.vstack([scene["axes"].to_wall(h.polygon)
                             for h in scene["holds"].values()])
        panel = np.zeros((export_info.height, export_info.width, 3), np.uint8)
        view = space_mod.fit_view(scene["axes"], (export_info.width, export_info.height),
                                  azimuth=np.radians(cfg.SPACE_PREVIEW_AZIMUTH),
                                  elevation=np.radians(cfg.SPACE_LIDAR_ELEVATION_DEGREES),
                                  margin=1.25, points=outline)
        colour = cfg.HOLD_RENDER_COLOR.get(run.colour, cfg.HOLD_COLOR_FALLBACK)
        centres, _ = space_mod.draw_dense(
            panel, scene["axes"], scene["holds"],
            scene["axes"].to_wall(scene["recon"].points),
            (scene["colours"] * (1.0 - cfg.SPACE_CLOUD_DIM)).astype(np.uint8),
            view, activated={h: 0 for h in scene["holds"]}, effective=10 ** 9,
            colors={"active": colour, "inactive_dense": (200, 200, 200)}, cfg=cfg,
            point_size=max(1, round(cfg.SPACE_POINT_SIZE * export_info.width / 608)))
        for hid, (cx, cy) in centres.items():
            cv2.putText(panel, render.hold_label(hid, cfg), (cx + 8, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, colour, 1, cv2.LINE_AA)
        preview = run_dir / "route3d.png"
        cv2.imwrite(str(preview), panel)
        console.print(f"  route3d -> [dim]{rel(preview)}[/] "
                      f"[dim](the reconstruction, face-on)[/]")

    _write_scene(run)


    raw_pair = run_dir / "_pair.mp4"
    raw_solo = run_dir / "_solo.mp4" if cfg.SAVE_RIGHT_PANEL else None
    t0 = time.perf_counter()
    frame_range = None
    if cfg.SPACE_RENDER_COVERED_ONLY:
        covered = sorted(run.scene["poses"])
        frame_range = (covered[0], covered[-1])
        if frame_range != (0, export_info.n_frames - 1):
            console.print(f"  [dim]rendering frames {frame_range[0]}-{frame_range[1]} "
                          f"only — outside the reconstructed span there is no camera "
                          f"pose, so there is no route to draw[/]")

    stats = render.render(export_info, run.poses, analysis, raw_pair,
                          frame_range=frame_range, cfg=cfg, console=console,
                          frame_poses=run.frame_poses, scene=run.scene,
                          colour=run.colour,
                          right_path=raw_solo, ground=run.ground,
                          comparison=comparison, label=run.label)
    render_seconds = time.perf_counter() - t0
    console.print(f"  drew the climber on [bold]{stats['frames_with_pose']}/"
                  f"{stats['frames_written']}[/] frames")
    if stats["tail_frames"]:
        console.print(f"  [dim]completion panel held for {stats['tail_frames']} frames "
                      f"({stats['tail_frames'] / export_info.fps:.1f}s) past the end "
                      f"of the clip[/]")

    # The render is longer than its source once the panel is held, so the length
    # is stated rather than left to `-shortest`, which would trim the tail away.
    duration = (stats["frames_written"] / export_info.fps
                if stats["tail_frames"] else None)
    out_video = run_dir / f"{run.label}_climb.mp4"
    t0 = time.perf_counter()
    # A depth capture records no sound, so there is no audio to carry over.
    video.encode_h264(raw_pair, out_video, crf=cfg.OUTPUT_CRF, duration=duration)
    encode_seconds = time.perf_counter() - t0
    raw_pair.unlink(missing_ok=True)
    console.print(f"  video  -> [dim]{rel(out_video)}[/] "
                  f"({out_video.stat().st_size / 1e6:.1f} MB)")

    solo_video = None
    if raw_solo is not None:
        solo_video = run_dir / f"{run.label}_route.mp4"
        t0 = time.perf_counter()
        video.encode_h264(raw_solo, solo_video, crf=cfg.OUTPUT_CRF, duration=duration)
        encode_seconds += time.perf_counter() - t0
        raw_solo.unlink(missing_ok=True)
        console.print(f"  route  -> [dim]{rel(solo_video)}[/]")

    # ── artifacts ────────────────────────────────────────────────────────────
    rule(step + 1, "Artifacts")
    util, rows = run.util, run.rows

    with (run_dir / "hold_times.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "hold_id", "order", "limbs", "time_start", "time_end", "contact_seconds"])
        writer.writeheader()
        writer.writerows(rows)

    # One row per (limb, hold) that actually happened — the join key for
    # comparing two climbers on the same route is (hold_id, limb).
    with (run_dir / "limb_usage.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["limb", "hold_id", "seconds", "frames", "share_of_contact"])
        writer.writeheader()
        writer.writerows(util.rows())

    sequence = climb.activation_sequence(analysis)
    moves = climb.limb_sequence(analysis)
    attempt = comparison.by_label(run.label) if comparison else None

    (run_dir / "climb.json").write_text(json.dumps({
        "route": route_metadata(run.prompt),
        "route_color": run.colour, "prompt": run.prompt, "fps": info.fps,
        "hold_numbering": {"order": "bottom_to_top", "within_row": run.numbering,
                           "band": cfg.HOLD_NUMBER_BAND,
                           "lean": round(holds_mod.route_lean(run.route), 4)},
        **analysis.summary(),
        "sequence": sequence,
        "moves": moves,
        "utilization": util.as_dict(),
        "holds": rows,
    }, indent=2))

    # The sequence, on its own and in full. Two of them, because they answer
    # different questions: `sequence` is the path up the wall and is what two
    # attempts are compared on, `moves` is which appendage went where and is
    # what tells you *how* the path was climbed.
    (run_dir / "sequence.json").write_text(json.dumps({
        "video": run.video.name,
        "label": run.label,
        "route": route_metadata(run.prompt),
        "topped_out": analysis.completion_frame is not None,
        "elapsed_seconds": round(analysis.elapsed, 3)
        if analysis.elapsed is not None else None,
        "holds_detected": len(run.route),
        "holds_used": analysis.n_used,
        "sequence": sequence,
        "sequence_aligned": attempt.aligned_sequence if attempt else sequence,
        "first_limb_by_hold": {str(k): v for k, v in
                               sorted(climb.first_limb_by_hold(analysis).items())},
        "moves": moves,
        "per_limb_sequence": {limb.key: util.sequence(limb.key) for limb in climb.LIMBS},
        "comparison": None if comparison is None else {
            "reference": comparison.reference,
            "ids_aligned": comparison.aligned,
            "shortest": comparison.shortest().label if comparison.shortest() else None,
            "fastest": comparison.fastest().label if comparison.fastest() else None,
            "shared_holds": sorted(comparison.shared_holds()),
            "others": [{"label": other.label,
                        "sequence": other.aligned_sequence,
                        "holds": len(other.aligned_sequence),
                        "elapsed_seconds": other.elapsed,
                        "topped_out": other.topped_out}
                       for other in comparison.others(run.label)],
        },
    }, indent=2))

    (run_dir / "summary.txt").write_text(climb.summary_text(
        analysis, rows, util, source=run.video.name, color=run.colour,
        numbering=run.numbering, warmup=run.warmup))
    console.print(f"  holds.json, poses.json, climb.json, sequence.json, "
                  f"hold_times.csv, limb_usage.csv, summary.txt -> [dim]{rel(run_dir)}/[/]")

    # ── performance ──────────────────────────────────────────────────────────
    rule(step + 2, "Performance")
    total = time.perf_counter() - run.started
    pose_cost = (run.pose_usage or {}).get("cost")
    metrics = timing.Metrics(
        frames=run.poses.n_frames_returned, video_seconds=info.duration,
        convert_seconds=run.convert_seconds, convert_cached=run.convert_cached,
        encode_seconds=encode_seconds, request=run.request_timing,
        render_seconds=render_seconds, total_seconds=total,
        upload_mb=run.upload_mb, cost_usd=pose_cost,
        poses_cached=run.poses_cached)

    (run_dir / "metrics.json").write_text(json.dumps(metrics.as_dict(), indent=2))
    (run_dir / "metrics.txt").write_text(metrics.as_text())

    m = metrics.as_dict()
    total_cost = (pose_cost or 0.0) + run.hold_cost
    kv({
        "ViTPose throughput": f"[bold green]{m['pose']['fps']:.1f} fps[/]  "
                              f"({m['pose']['ms_per_frame']:.0f} ms/frame, "
                              f"{m['pose']['realtime_factor']:.2f}x realtime)",
        "SAM 3.1": f"{run.hold_seconds:.1f}s tracking the holds"
                   if run.hold_seconds else "cached",
        "gateway round trip": f"{m['gateway_round_trip']['seconds']:.2f}s",
        "render": f"{m['local']['render_seconds']:.2f}s ({m['local']['render_fps']:.1f} fps)",
        "cost": f"${total_cost:.4f}  [dim](pose ${pose_cost or 0:.4f} + "
                f"holds ${run.hold_cost:.4f})[/]"
                if total_cost else "—",
    }, title="metrics")

    (run_dir / "run.json").write_text(json.dumps({
        "stamp": run_dir.name,
        "total_seconds": round(total, 2),
        "batch_mode": cfg.BATCH_MODE,
        "reconstruction": {
            "method": "lidar",
            "keyframes": run.scene["recon"].keyframes,
            "keyframe_count": len(run.scene["recon"].keyframes),
            "points": run.scene["recon"].n_points,
            "rms_px": round(run.scene["recon"].rms_px, 4),
            "localized_frames": len(run.scene["poses"]),
            "frame_span": [min(run.scene["poses"]), max(run.scene["poses"])],
            "coverage": round(len(run.scene["poses"]) / max(info.n_frames, 1), 4),
            "holds": len(run.scene["holds"]),
        },
        "route": route_metadata(run.prompt),
        "input": {"path": str(run.video), **run.src_info},
        "converted": {"path": str(run.mp4), "width": info.width, "height": info.height,
                      "fps": info.fps, "n_frames": info.n_frames},
        "tonemap": {"mode": cfg.TONEMAP, "algorithm": cfg.TONEMAP_ALGORITHM,
                    "filter": run.tonemap, "reason": run.tonemap_reason},
        "holds": {"model": cfg.HOLD_MODEL, "prompt": run.prompt,
                  "settings": run.hold_settings, "detected": len(run.route),
                  "from_cache": run.holds_cached, "cost_usd": run.hold_cost},
        "pose": {"model": cfg.POSE_MODEL, "extra_body": run.extra_body,
                 "min_kpt_score": cfg.POSE_MIN_KPT_SCORE,
                 "smooth_sigma": cfg.POSE_SMOOTH_SIGMA,
                 "track_id": run.track_id, "tracks": run.track_stats,
                 "from_cache": run.poses_cached, "usage": run.pose_usage},
        "climb": {**analysis.summary(), "numbering": run.numbering,
                  "sequence": sequence},
        "floor": {"method": "plane fitted to the depth",
                  "found": run.ground is not None,
                  "clearance": cfg.FLOOR_CLEARANCE}
                 if cfg.DETECT_FLOOR else None,
        "utilization": util.as_dict(),
        "comparison": None if attempt is None else {
            "reference": comparison.reference,
            "alignment": attempt.alignment,
            "sequence_aligned": attempt.aligned_sequence},
        "render": {**stats, "width": export_info.width, "height": export_info.height},
        "metrics": metrics.as_dict(),
        "outputs": {"video": out_video.name,
                    "route": solo_video.name if solo_video else None,
                    "holds": "holds.json", "climb": "climb.json",
                    "sequence": "sequence.json",
                    "limb_usage": "limb_usage.csv",
                    "summary": "summary.txt", "metrics": "metrics.json"},
    }, indent=2))
    return run_dir


def to_render(runs: list[Run], console) -> list[tuple[int, Run]]:
    """The attempts to render, as (attempt number, run). Empty if none matched.

    RENDER_ONLY narrows the render, never the analysis: the card a clip ends on
    is built from every clip, so all of them are analyzed either way. The number
    is the one the card shows, because both come from this same list.
    """
    numbered = list(enumerate(runs, start=1))
    want = cfg.RENDER_ONLY
    if want is None:
        return numbered
    if isinstance(want, int):
        chosen = [pair for pair in numbered if pair[0] == want]
    else:
        stem = Path(want).stem.lower()
        chosen = [pair for pair in numbered if pair[1].label.lower() == stem]
    if not chosen:
        console.print(f"[red]RENDER_ONLY = {want!r} matches no attempt.[/] Available: "
                      + ", ".join(f"{i} = {r.label}" for i, r in numbered))
    return chosen


def main() -> int:
    t_start = time.perf_counter()
    console.print()
    console.rule("[bold]SAM 3.1 + ViTPose: rock climbing[/]", align="center")

    # ── 1. Key ───────────────────────────────────────────────────────────────
    rule(1, "API key")
    api_key, source = load_api_key(cfg.PROJECT_DIR)
    console.print(f"  loaded from [green]{source}[/] ([dim]…{api_key[-4:]}[/])")

    sources = discover()
    if not sources:
        where = cfg.INPUT_DIR if cfg.BATCH_MODE else cfg.INPUT_CAPTURE
        console.print(f"[red]No depth captures found:[/] {where}\n  Each one is "
                      f"{lidar.CAPTURE_FORMAT}. A video without depth belongs "
                      "in ../rock_climbing (the 2D demo).")
        return 1

    stamp = datetime.now().strftime(cfg.RUN_STAMP_FORMAT)
    batch = cfg.BATCH_MODE and len(sources) > 1
    batch_dir = cfg.OUTPUT_DIR / stamp
    if batch:
        console.print(f"  [bold]batch mode[/]: {len(sources)} attempts at the "
                      f"[bold]{cfg.HOLD_COLOR}[/] route"
                      + (f" ({cfg.ROUTE_GRADE})" if cfg.ROUTE_GRADE else "")
                      + f" — [dim]{rel(cfg.INPUT_DIR)}/[/]")
        for path in sources:
            console.print(f"    [dim]{path.name}[/]")

    # ── every clip, before any of them is rendered ───────────────────────────
    runs: list[Run] = []
    for i, path in enumerate(sources, start=1):
        if batch:
            console.print()
            console.rule(f"[bold magenta]attempt {i}/{len(sources)}[/] {path.name}",
                         align="left")
        run_dir = batch_dir / path.stem if batch else batch_dir
        run = analyze(path, api_key=api_key, run_dir=run_dir, batch=batch)
        if run is None:
            if not batch:
                return 1
            console.print(f"  [yellow]skipping {path.name}[/]; it produced nothing "
                          "to compare")
            continue
        runs.append(run)

    if not runs:
        console.print("[red]Nothing analyzed.[/]")
        return 1

    # ── the comparison, then the renders it feeds ────────────────────────────
    console.print()
    comparison = compare_runs(runs)
    if comparison is not None:
        adopt_numbering(runs, comparison)
    # `analyze` ends on step 7, and `compare_runs` takes 8 when it runs at all.
    step = 9 if comparison is not None else 8

    selected = to_render(runs, console)
    if not selected:
        return 1
    if cfg.RENDER_ONLY is not None:
        only = ", ".join(f"attempt {i} ({r.label})" for i, r in selected)
        console.print(f"  [bold]rendering {only}[/] only; the rest were analyzed, "
                      "not drawn")

    for i, run in selected:
        if batch:
            console.print()
            console.rule(f"[bold magenta]{i}/{len(runs)}[/] {run.label}", align="left")
        deliver(run, comparison, step=step, batch=batch)

    if comparison is not None:
        batch_dir.mkdir(parents=True, exist_ok=True)
        (batch_dir / "comparison.json").write_text(
            json.dumps(comparison.as_dict(), indent=2))
        (batch_dir / "sequences.txt").write_text(compare.summary_text(comparison))
        console.print()
        console.print(f"  comparison.json, sequences.txt -> [dim]{rel(batch_dir)}/[/]")

    total = time.perf_counter() - t_start
    console.print()
    console.rule(f"[bold green]done[/] in {total:.1f}s", align="left")
    console.print(f"  [bold]{rel(batch_dir)}/[/]\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted[/]")
        sys.exit(130)
    except Exception as exc:
        console.print(f"\n[red]error:[/] {exc}")
        raise
