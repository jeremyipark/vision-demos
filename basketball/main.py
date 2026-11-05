"""Track a shooting session: every shot, make or miss, a shot chart, and the elbow.

    python main.py                         # every clip in data/input/
    python main.py data/input/session.mov  # one clip
    python main.py -h                      # every flag

ViTPose finds the shots; SAM 3.1 finds the hoop, then tracks the ball around each shot.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import config as cfg
from src import ball as ball_mod
from src import court, export, hoop as hoop_mod, pose, render, shots as shots_mod, video
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
    """*path* relative to the project, for printing."""
    try:
        return path.relative_to(cfg.PROJECT_DIR)
    except ValueError:
        return path


def stamp_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def client(api_key: str):
    from openai import OpenAI

    return OpenAI(base_url=cfg.GATEWAY_BASE_URL, api_key=api_key,
                  timeout=cfg.REQUEST_TIMEOUT, max_retries=1)


def new_run_dir(kind: str) -> Path:
    """A fresh ``data/output/<timestamp>-<kind>/``: e.g. ``20261105-143012-shot-chart``.
    A second run in the same second gets ``-2``, ``-3``, ..."""
    stamp = datetime.now().strftime(cfg.RUN_STAMP_FORMAT)
    run_dir = cfg.OUTPUT_DIR / f"{stamp}-{kind}"
    n = 2
    while run_dir.exists():
        run_dir = cfg.OUTPUT_DIR / f"{stamp}-{kind}-{n}"
        n += 1
    run_dir.mkdir(parents=True)
    return run_dir


def cost_of(usage: dict | None) -> float:
    return float((usage or {}).get("cost") or 0)


# ── models ───────────────────────────────────────────────────────────────────

def pose_clip(api_key: str, mp4: Path, info: video.VideoInfo) -> dict:
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
            payload, usage = pose.request_poses(client(api_key), model=cfg.POSE_MODEL,
                                                video_b64=video_b64, extra_body=extra_body)
        seconds = time.perf_counter() - t0
        pose.save_cache(cache, payload, usage, stamp=stamp_now())
        console.print(f"  [green]done in {seconds:.1f}s[/]")

    frames, _ = pose.unwrap(payload)
    dropped = pose.dedupe(frames, max_persons=cfg.MAX_PERSONS, iou=cfg.PERSON_IOU_THRESHOLD)
    return {"by_index": {f["index"]: f["persons"] for f in frames}, "dropped": dropped,
            "seconds": seconds, "extra_body": extra_body,
            "cost": 0.0 if cached is not None else cost_of(usage)}


def find_hoop(api_key: str, mp4: Path, info: video.VideoInfo) -> tuple[hoop_mod.Hoop, float]:
    """SAM 3.1 on one frame, cached. The phone stays still, so it holds for the clip."""
    seconds = cfg.HOOP_FRAME_SECONDS
    frame = info.n_frames // 2 if seconds is None else min(info.n_frames - 1, int(seconds * info.fps))
    cache = hoop_mod.cache_path(mp4, cfg.CACHE_DIR, model=cfg.SAM_MODEL, prompt=cfg.HOOP_PROMPT,
                                frame=frame)
    blob = json.loads(cache.read_text()) if cfg.REUSE_CACHE and cache.is_file() else None
    cost = 0.0
    if blob is None:
        ok, png = cv2.imencode(".png", hoop_mod.read_frame(mp4, frame))
        with console.status("[cyan]waiting on the gateway[/]; SAM is finding the hoop…",
                            spinner="dots"):
            payload, usage = hoop_mod.request_segment(client(api_key), model=cfg.SAM_MODEL,
                                                      image_png=png.tobytes(), prompt=cfg.HOOP_PROMPT)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"created": stamp_now(), "payload": payload, "usage": usage}))
        cost = cost_of(usage)
    else:
        payload = blob["payload"]
        console.print(f"  [green]cache hit[/]: hoop from [dim]{blob.get('created')}[/]")
    return hoop_mod.locate(payload, frame=frame, width=info.width, height=info.height), cost


def track_window(api_key: str, mp4: Path, info: video.VideoInfo, start: int, length: int
                 ) -> tuple[dict, float]:
    """SAM 3.1 on frames ``[start, start + length)``, cached; detections in clip frames."""
    cache = ball_mod.cache_path(mp4, cfg.CACHE_DIR, model=cfg.SAM_MODEL, prompt=cfg.BALL_PROMPT,
                                start=start, length=length, stride=cfg.BALL_STRIDE)
    blob = json.loads(cache.read_text()) if cfg.REUSE_CACHE and cache.is_file() else None
    cost = 0.0
    if blob is None:
        seg = cfg.CACHE_DIR / f"{mp4.stem}.win{start}_{length}.mp4"
        if not seg.is_file():
            video.segment(mp4, seg, start_frame=start, n_frames=length, fps=info.fps)
        payload, usage = ball_mod.request_track(
            client(api_key), model=cfg.SAM_MODEL,
            video_b64=base64.b64encode(seg.read_bytes()).decode("ascii"),
            prompt=cfg.BALL_PROMPT, stride=cfg.BALL_STRIDE)
        cache.write_text(json.dumps({"created": stamp_now(), "payload": payload, "usage": usage}))
        seg.unlink(missing_ok=True)
        cost = cost_of(usage)
    else:
        payload = blob["payload"]
    return ball_mod.detections(payload, frame_offset=start, width=info.width, height=info.height,
                               min_score=cfg.BALL_MIN_SCORE), cost


def track_ball(api_key: str, mp4: Path, info: video.VideoInfo, spans, hoop) -> tuple[
        ball_mod.BallTrack, float, int]:
    """The ball around every shot, extending a window while the ball is still at the rim."""
    fps = info.fps
    cap = ball_mod.TRACK_MAX_FRAMES * cfg.BALL_STRIDE
    raw: dict[int, list] = {}
    windows: list[tuple[int, int]] = []
    cost, calls = 0.0, 0
    with console.status("[cyan]waiting on the gateway[/]; SAM is tracking the ball…",
                        spinner="dots") as status:
        for n, (a, _) in enumerate(spans, 1):
            start = max(0, a - int(cfg.BALL_BEFORE_SECONDS * fps))
            end = min(info.n_frames, a + int(cfg.BALL_AFTER_SECONDS * fps))
            if windows and start < sum(windows[-1]):
                start = sum(windows[-1])   # never pay twice for a frame
            for extension in range(cfg.BALL_MAX_EXTENSIONS + 1):
                length = min(end - start, cap)
                if length <= 0:
                    break
                status.update(f"[cyan]waiting on the gateway[/]; SAM is tracking the ball "
                              f"(shot {n}/{len(spans)}{', more' if extension else ''})…")
                found, c = track_window(api_key, mp4, info, start, length)
                raw.update(found)
                windows.append((start, length))
                cost += c
                calls += 1
                track = ball_mod.select(raw, info.n_frames, windows,
                                        size_tolerance=cfg.BALL_SIZE_TOLERANCE)
                if not shots_mod.ball_unresolved(track.x, track.y, start, start + length, hoop):
                    break
                start, end = start + length, min(info.n_frames,
                                                 start + length + int(cfg.BALL_EXTEND_SECONDS * fps))
    track = ball_mod.select(raw, info.n_frames, windows, size_tolerance=cfg.BALL_SIZE_TOLERANCE)
    track.shown = ball_mod.select(raw, info.n_frames, windows, local_size=True,
                                  size_tolerance=cfg.BALL_DRAW_SIZE_TOLERANCE)
    return track, cost, calls


# ── the pipeline ─────────────────────────────────────────────────────────────

def analyze(kp: np.ndarray, info: video.VideoInfo, spans, hoop, track, side: str
            ) -> list[shots_mod.Shot]:
    fps = info.fps
    angle = shots_mod.elbow_angle(kp, side, min_projected=cfg.ELBOW_MIN_PROJECTED)
    wrist_y = kp[:, shots_mod.J[f"{side}_wrist"], 1]
    ratio = shots_mod.shoulder_ratio(kp)
    # The ball as drawn, for where the shooting motion starts.
    bx, by = ball_mod.smoothed(track, max_gap=int(cfg.BALL_FILL_SECONDS * fps))
    shown = track.shown or track
    size = np.where(np.isfinite(shown.d), shown.d, track.diameter)
    wrists = kp[:, [shots_mod.J["left_wrist"], shots_mod.J["right_wrist"]], :2]
    shots = []
    for n, (a, b) in enumerate(spans, 1):
        nxt = spans[n][0] if n < len(spans) else info.n_frames
        release = shots_mod.find_release(a, b, track.y, wrist_y, track.diameter, fps)
        lo = max(0, a - int(cfg.SET_WINDOW_SECONDS * fps))
        before = angle[lo:release + 1]
        set_frame = lo + int(np.nanargmin(before)) if np.isfinite(before).any() else a
        after = angle[release:min(len(angle), release + int(0.2 * fps)) + 1]
        side_on = bool(np.nanmedian(ratio[lo:release + 1]) < cfg.SIDE_ON_MAX_SHOULDER_RATIO)
        nan = float("nan")
        shot = shots_mod.Shot(
            number=n, start=a, end=b, release=release, set_frame=set_frame, side_on=side_on,
            elbow_set=float(angle[set_frame]) if side_on else nan,
            elbow_release=float(np.nanmax(after)) if side_on and np.isfinite(after).any() else nan)
        shot.motion = shots_mod.motion_start(a, bx, by, size, wrists, fps,
                                             lookback=cfg.BALL_BEFORE_SECONDS, reach=cfg.HELD_REACH)
        # The ball is judged up to the next shot, or the end of what SAM saw.
        covered_until = release
        while covered_until < info.n_frames and track.covered[covered_until]:
            covered_until += 1
        shots_mod.judge(shot, track.x, track.y, track.d, hoop, fps, until=min(nxt, covered_until),
                        inner_fraction=cfg.RIM_INNER_FRACTION, net_min_seconds=cfg.NET_MIN_SECONDS,
                        net_side_tolerance=cfg.NET_SIDE_TOLERANCE,
                        rebound_margin=cfg.REBOUND_MARGIN_RIMS,
                        max_gap=int(cfg.RIM_GAP_SECONDS * fps),
                        ball_to_rim=cfg.BALL_DIAMETER_M / cfg.RIM_DIAMETER_M,
                        front_size=cfg.FRONT_OF_RIM_SIZE)
        where = shots_mod.stance(kp, a, fps)
        if where is not None:
            shot.feet_px, shot.body_px = where
            shot.court_xy = court.project(
                shot.feet_px[0], shot.body_px, hoop, frame_width=info.width,
                hfov_degrees=cfg.CAMERA_HFOV_DEGREES, player_height_m=cfg.PLAYER_HEIGHT_M,
                rim_diameter_m=cfg.RIM_DIAMETER_M, yaw_degrees=cfg.COURT_YAW_DEGREES)
            shot.distance_m = float(np.hypot(*shot.court_xy))
        shots.append(shot)
    return shots


def run_clip(api_key: str) -> dict:
    """The whole pipeline for ``cfg.INPUT_VIDEO``. Returns what the batch summary prints."""
    t_start = time.perf_counter()

    # ── 2. Source ────────────────────────────────────────────────────────────
    rule(2, "Source video")
    src_info = video.probe_source(cfg.INPUT_VIDEO)
    kv({
        "path": rel(cfg.INPUT_VIDEO),
        "size": f"{cfg.INPUT_VIDEO.stat().st_size / 1e6:.1f} MB",
        "codec": f"{src_info['codec']} ({src_info['pix_fmt']})",
        "stored": f"{src_info['width']}x{src_info['height']}",
        "color": f"{src_info['color_primaries'] or '?'} / {src_info['color_transfer'] or '?'}"
                  + ("  [yellow](HDR)[/]" if src_info["is_hdr"] else "  [dim](SDR)[/]"),
        "frames": f"{src_info['n_frames']} ({src_info['duration']:.1f}s)",
    })

    # ── 3. Convert (cached) ──────────────────────────────────────────────────
    rule(3, "Convert to MP4")
    tonemap, reason = video.tonemap_filter(src_info, mode=cfg.TONEMAP,
                                           algorithm=cfg.TONEMAP_ALGORITHM)
    console.print(f"  tone-map: {'[green]on[/] — ' if tonemap else ''}{reason}")
    mp4 = video.cache_path(cfg.INPUT_VIDEO, cfg.CACHE_DIR, target_height=cfg.INFERENCE_HEIGHT,
                           trim_seconds=cfg.TRIM_SECONDS, crf=cfg.CONVERT_CRF, tonemap=tonemap)
    if mp4.is_file() and not cfg.FORCE_RECONVERT:
        console.print(f"  [green]cache hit[/]: [dim]{rel(mp4)}[/]")
    else:
        video.convert(cfg.INPUT_VIDEO, mp4, target_height=cfg.INFERENCE_HEIGHT,
                      trim_seconds=cfg.TRIM_SECONDS, crf=cfg.CONVERT_CRF,
                      total_frames=src_info["n_frames"], console=console, tonemap=tonemap)
    info = video.inspect(mp4)
    console.print(f"  [bold]{info}[/]  ({mp4.stat().st_size / 1e6:.1f} MB)")

    # ── 4. Pose: ViTPose ─────────────────────────────────────────────────────
    rule(4, "Pose (ViTPose)")
    posed = pose_clip(api_key, mp4, info)
    kp = shots_mod.keypoints(posed["by_index"], info.n_frames, info.width, info.height)
    n_posed = int(np.isfinite(kp[:, 5, 0]).sum())
    spans = shots_mod.candidates(kp, info.fps, armed_below=cfg.SHOT_ARMED_BELOW,
                                 min_gap_seconds=cfg.SHOT_MIN_GAP_SECONDS)
    side = (cfg.SHOOTING_HAND if cfg.SHOOTING_HAND in ("left", "right")
            else shots_mod.shooting_side(kp, spans, info.fps))
    kv({"model": cfg.POSE_MODEL, "frames with a person": f"{n_posed} of {info.n_frames}",
        "shots found": f"{len(spans)}, at " + ", ".join(f"{a / info.fps:.1f}s" for a, _ in spans),
        "shooting hand": side + ("" if cfg.SHOOTING_HAND in ("left", "right") else " (auto)"),
        "cost": f"${posed['cost']:.4f}"}, title="pose")

    # ── 5. Hoop and ball: SAM 3.1 ────────────────────────────────────────────
    rule(5, "Hoop and ball (SAM 3.1)")
    hoop, hoop_cost = find_hoop(api_key, mp4, info)
    track, ball_cost, calls = track_ball(api_key, mp4, info, spans, hoop)
    tracked = int(track.covered.sum())
    kv({"hoop": f"rim at ({hoop.cx:.0f}, {hoop.cy:.0f}) px, {hoop.width:.0f} px wide; "
                f"net down to {hoop.net_bottom:.0f} px (score {hoop.score:.2f})",
        "ball windows": f"{calls} calls, {tracked} of {info.n_frames} frames "
                        f"({tracked / info.n_frames:.0%} of the clip)",
        "ball found": f"{int(track.found.sum())} frames, ~{track.diameter:.0f} px across",
        "cost": f"${hoop_cost + ball_cost:.4f}"}, title="SAM 3.1")

    # ── 6. Shots ─────────────────────────────────────────────────────────────
    rule(6, "Shots: make or miss")
    shots = analyze(kp, info, spans, hoop, track, side)
    view = court.CourtView(10, 10, three_radius=cfg.THREE_POINT_RADIUS_M,
                           ft_distance=cfg.FREE_THROW_DISTANCE_M)
    made = sum(1 for s in shots if s.made)

    def deg(v):
        return f"{v:.0f}°" if np.isfinite(v) else "[dim]—[/]"

    table = Table(box=None, pad_edge=False)
    for col in ("shot", "at", "result", "why", "elbow set", "release", "spot (m)", "distance"):
        table.add_column(col, justify="left" if col in ("result", "why") else "right",
                         style="dim" if col == "shot" else None)
    for s in shots:
        spot = (f"{s.court_xy[0]:+.1f}, {s.court_xy[1]:.1f}" if s.court_xy else "—")
        dist = (f"{s.distance_m * 3.2808:.0f} ft" + (" (3)" if view.is_three(*s.court_xy) else "")
                if s.distance_m is not None else "—")
        table.add_row(str(s.number), f"{s.release / info.fps:.1f}s",
                      "[green]MAKE[/]" if s.made else "[red]MISS[/]" if s.made is False else "[dim]?[/]",
                      s.reason, deg(s.elbow_set), deg(s.elbow_release), spot, dist)
    console.print(Panel(table, title=f"[bold]{made}/{len(shots)}[/] "
                                     f"({made / len(shots):.0%} FG)" if shots else "no shots",
                        title_align="left", expand=False))

    # ── 7. Output ────────────────────────────────────────────────────────────
    rule(7, "Render")
    run_dir = new_run_dir("shot-chart")
    stamp = run_dir.name

    angle = shots_mod.elbow_angle(kp, side, min_projected=cfg.ELBOW_MIN_PROJECTED)
    side_on = shots_mod.shoulder_ratio(kp) < cfg.SIDE_ON_MAX_SHOULDER_RATIO
    # An export taller than the analyzed clip is drawn on a taller copy of the source
    # (never past the source's own height), so it isn't upscaled.
    canvas = info
    clip_h = min(int(cfg.EXPORT_HEIGHT) // 2, src_info["height"]) // 2 * 2
    if clip_h > info.height:
        big = video.cache_path(cfg.INPUT_VIDEO, cfg.CACHE_DIR, target_height=clip_h,
                               trim_seconds=cfg.TRIM_SECONDS, crf=cfg.CONVERT_CRF,
                               tonemap=tonemap)
        if not big.is_file() or cfg.FORCE_RECONVERT:
            video.convert(cfg.INPUT_VIDEO, big, target_height=clip_h,
                          trim_seconds=cfg.TRIM_SECONDS, crf=cfg.CONVERT_CRF,
                          total_frames=src_info["n_frames"], console=console, tonemap=tonemap)
        canvas = video.inspect(big)
    raw = run_dir / "_raw.mp4"
    t0 = time.perf_counter()
    stats = render.render(info, canvas=canvas, kp=kp, hoop=hoop, track=track, shots=shots,
                          dst=raw, cfg=cfg,
                          console=console)
    render_seconds = time.perf_counter() - t0
    out_video = run_dir / f"{cfg.INPUT_VIDEO.stem}_basketball.mp4"
    video.encode_h264(raw, out_video, crf=cfg.OUTPUT_CRF)
    raw.unlink(missing_ok=True)
    console.print(f"  video  -> [dim]{rel(out_video)}[/] "
                  f"({out_video.stat().st_size / 1e6:.1f} MB, {render_seconds:.1f}s)")
    plots = run_dir / "plots.png"
    render.plot_summary(shots, info.fps, hoop, hoop_mod.read_frame(mp4, hoop.frame), plots,
                        cfg=cfg)
    console.print(f"  graph  -> [dim]{rel(plots)}[/]")
    elbow_png = run_dir / "elbow.png"
    render.plot_elbow(shots, angle, info.fps, mp4, kp, side, elbow_png, cfg=cfg)
    console.print(f"  graph  -> [dim]{rel(elbow_png)}[/]")
    for path in export.write(shots, angle, side_on, track, info.fps, run_dir, cfg.INPUT_VIDEO.stem):
        console.print(f"  data   -> [dim]{rel(path)}[/]")

    total = time.perf_counter() - t_start
    cost = posed["cost"] + hoop_cost + ball_cost
    (run_dir / "run.json").write_text(json.dumps({
        "stamp": stamp,
        "total_seconds": round(total, 2),
        "input": {"path": str(cfg.INPUT_VIDEO), **src_info},
        "converted": {"path": str(mp4), "width": info.width, "height": info.height,
                      "fps": info.fps, "n_frames": info.n_frames, "tonemap": tonemap},
        "pose": {"model": cfg.POSE_MODEL, "extra_body": posed["extra_body"], "frames_posed": n_posed},
        "hoop": {"model": cfg.SAM_MODEL, "prompt": cfg.HOOP_PROMPT, **hoop.to_dict()},
        "ball": {"model": cfg.SAM_MODEL, "prompt": cfg.BALL_PROMPT, "stride": cfg.BALL_STRIDE,
                 "windows": track.windows, "frames_tracked": tracked,
                 "frames_found": int(track.found.sum()), "diameter_px": round(track.diameter, 1)},
        "shooting_hand": side,
        "court": {"hfov_degrees": cfg.CAMERA_HFOV_DEGREES, "player_height_m": cfg.PLAYER_HEIGHT_M,
                  "yaw_degrees": cfg.COURT_YAW_DEGREES, "three_point_radius_m": cfg.THREE_POINT_RADIUS_M},
        "shots": [s.summary(info.fps) for s in shots],
        "made": made, "attempts": len(shots),
        "cost_usd": {"pose": round(posed["cost"], 6), "hoop": round(hoop_cost, 6),
                     "ball": round(ball_cost, 6), "total": round(cost, 6)},
        "render": {**stats, "seconds": round(render_seconds, 2), "credit": cfg.CREDIT},
    }, indent=2))

    console.print()
    kv({"pose": f"${posed['cost']:.4f}", "hoop": f"${hoop_cost:.4f}", "ball": f"${ball_cost:.4f}",
        "total": f"[bold]${cost:.4f}[/]"}, title="cost")
    console.rule(f"[bold green]done[/] in {total:.1f}s", align="left")
    console.print(f"  [bold]{rel(run_dir)}/[/]\n")
    return {"made": made, "attempts": len(shots), "cost": cost, "seconds": total, "output": run_dir}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The per-run settings, as flags. Anything left out keeps its config.py value."""
    parser = argparse.ArgumentParser(
        description="Track a shooting session: every shot, make or miss, the elbow, and a shot "
                    "chart from above. Flags override config.py for this run only.")
    parser.add_argument("inputs", nargs="*", type=Path, metavar="CLIP|DIR",
                        help=f"clips, or folders of clips, to run (config: {rel(cfg.INPUT)})")
    parser.add_argument("--height", type=int, metavar="PX",
                        help=f"the output's total height: the clip on top, the court below, each half "
                             f"(config: {cfg.EXPORT_HEIGHT})")
    parser.add_argument("--trim", type=float, metavar="SECONDS",
                        help="use only the first SECONDS of each clip, for a quick test")
    parser.add_argument("--hand", choices=("auto", "left", "right"),
                        help=f"the shooting hand (config: {cfg.SHOOTING_HAND})")
    parser.add_argument("--player-height", type=float, metavar="M",
                        help=f"the shooter's height in meters, which scales every distance "
                             f"on the court (config: {cfg.PLAYER_HEIGHT_M:g})")
    parser.add_argument("--fov", type=float, metavar="DEG",
                        help=f"the camera's horizontal field of view (config: {cfg.CAMERA_HFOV_DEGREES:g})")
    parser.add_argument("--yaw", type=float, metavar="DEG",
                        help=f"turn the court if the camera isn't square to the backboard "
                             f"(config: {cfg.COURT_YAW_DEGREES:g})")
    parser.add_argument("--three", type=float, metavar="M",
                        help=f"the three-point radius: 6.75 FIBA, 7.24 NBA, 6.02 US high school "
                             f"(config: {cfg.THREE_POINT_RADIUS_M:g})")
    parser.add_argument("--fresh", action="store_true",
                        help="ignore every cached gateway reply and call the models again")
    credit = parser.add_argument_group(
        "credit", "the lines in the panel's corner; any of these replaces config.py's CREDIT")
    credit.add_argument("--li", metavar="TEXT", help='LinkedIn, e.g. "Jeremy Park, PhD"')
    credit.add_argument("--x", metavar="HANDLE", help="X handle, e.g. @jeremyparkphd")
    credit.add_argument("--ig", metavar="HANDLE", help="Instagram handle")
    credit.add_argument("--no-credit", action="store_true", help="no credit at all")
    return parser.parse_args(argv)


def apply_args(args: argparse.Namespace) -> None:
    """Write the flags over the config, before anything reads it."""
    if args.height:
        cfg.EXPORT_HEIGHT = args.height
    if args.trim:
        cfg.TRIM_SECONDS = args.trim
    if args.hand:
        cfg.SHOOTING_HAND = args.hand
    if args.player_height:
        cfg.PLAYER_HEIGHT_M = args.player_height
    if args.fov:
        cfg.CAMERA_HFOV_DEGREES = args.fov
    if args.yaw is not None:
        cfg.COURT_YAW_DEGREES = args.yaw
    if args.three:
        cfg.THREE_POINT_RADIUS_M = args.three
    if args.fresh:
        cfg.REUSE_CACHE = False

    def at(handle):
        return handle if handle.startswith("@") else f"@{handle}"

    handles = [(label, value) for label, value in
               (("LI", args.li), ("X", args.x and at(args.x)),
                ("IG", args.ig and at(args.ig))) if value]
    if args.no_credit:
        cfg.CREDIT = []
    elif handles:
        cfg.CREDIT = handles


def find_clips(inputs: list[Path]) -> list[Path]:
    """Every clip named, and every video directly inside every folder named, in order."""
    clips: list[Path] = []
    for item in inputs:
        path = item.expanduser().resolve()
        if path.is_dir():
            clips += sorted(p for p in path.iterdir()
                            if p.is_file() and not p.name.startswith(".")
                            and p.suffix.lower() in cfg.VIDEO_EXTENSIONS)
        elif path.is_file():
            clips.append(path)
        else:
            console.print(f"[red]Not found:[/] {item}")
    return list(dict.fromkeys(clips))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    apply_args(args)
    clips = find_clips(args.inputs or [cfg.INPUT])
    console.print()
    console.rule("[bold]Basketball: makes, misses, elbow and shot chart[/]", align="center")
    if not clips:
        console.print(f"[red]No clips to run.[/] Put a shooting video in "
                      f"{rel(cfg.INPUT)}/, or pass one: python main.py path/to/clip.mov")
        return 1

    # ── 1. Key ───────────────────────────────────────────────────────────────
    rule(1, "API key")
    api_key, source = load_api_key(cfg.PROJECT_DIR)
    console.print(f"  key loaded from [green]{source}[/]")

    results: dict[str, dict | str] = {}
    for n, clip in enumerate(clips, 1):
        if len(clips) > 1:
            console.print()
            console.rule(f"[bold magenta]clip {n}/{len(clips)}[/] {clip.name}", align="left")
        cfg.INPUT_VIDEO = clip
        try:
            results[clip.name] = run_clip(api_key)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            console.print(f"\n[red]error on {clip.name}:[/] {exc}")
            results[clip.name] = str(exc).splitlines()[0][:100]

    if len(clips) > 1:
        table = Table(title="batch", box=None, pad_edge=False, title_justify="left")
        for col in ("clip", "made", "FG", "cost", "output"):
            table.add_column(col)
        for name, r in results.items():
            if isinstance(r, str):
                table.add_row(name, "[red]failed[/]", r, "", "")
            else:
                fg = f"{r['made'] / r['attempts']:.0%}" if r["attempts"] else "—"
                table.add_row(name, f"{r['made']}/{r['attempts']}", fg, f"${r['cost']:.3f}",
                              str(rel(r["output"])))
        console.print(table)
    return 0 if all(not isinstance(r, str) for r in results.values()) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted[/]")
        sys.exit(130)
    except Exception as exc:
        console.print(f"\n[red]error:[/] {exc}")
        sys.exit(1)
