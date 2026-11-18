"""Break down a golf swing: pose, the club segmented, the ball's launch, the tempo.

    python main.py                          # every clip in data/input/
    python main.py data/input/swing.mov     # one clip
    python main.py -h                       # every flag

ViTPose tracks the golfer; SAM 3.1 tracks the club and the ball. The ball is seen on
the tee and for the few frames after impact before it's too small and fast, so its
flight past that is projected, not measured.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import config as cfg
from src import camera, faint, flight, models, render, swing as swing_mod, video
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
    try:
        return path.relative_to(cfg.PROJECT_DIR)
    except ValueError:
        return path


def new_run_dir(kind: str) -> Path:
    stamp = datetime.now().strftime(cfg.RUN_STAMP_FORMAT)
    run_dir = cfg.OUTPUT_DIR / f"{stamp}-{kind}"
    n = 2
    while run_dir.exists():
        run_dir = cfg.OUTPUT_DIR / f"{stamp}-{kind}-{n}"
        n += 1
    run_dir.mkdir(parents=True)
    return run_dir


def _num(value, digits: int = 2):
    return "" if value is None or not np.isfinite(value) else round(float(value), digits)


def write_frames_csv(path: Path, *, kp, club, ball, swing, fps: float) -> None:
    grip = swing_mod.hands(kp)
    with path.open("w", newline="") as fh:
        out = csv.writer(fh)
        out.writerow(["frame", "t_s", "phase", "hands_x_px", "hands_y_px", "club_found",
                      "clubhead_x_px", "clubhead_y_px", "ball_x_px", "ball_y_px", "ball_d_px",
                      "ball_state"])
        for f in range(len(kp)):
            state = ("rest" if f <= ball.rest_until else "flight" if f in ball.flight else "")
            out.writerow([f, _num(f / fps, 3), swing.phase(f), _num(grip[f, 0], 1),
                          _num(grip[f, 1], 1), bool(club.found[f]), _num(club.head[f, 0], 1),
                          _num(club.head[f, 1], 1), _num(ball.x[f], 1), _num(ball.y[f], 1),
                          _num(ball.d[f], 1), state])


def run_clip(api_key: str) -> dict:
    t_start = time.perf_counter()

    # ── 2. Source ────────────────────────────────────────────────────────────
    rule(2, "Source video")
    src_info = video.probe_source(cfg.INPUT_VIDEO)
    kv({"path": rel(cfg.INPUT_VIDEO),
        "size": f"{cfg.INPUT_VIDEO.stat().st_size / 1e6:.1f} MB",
        "codec": f"{src_info['codec']} ({src_info['pix_fmt']})",
        "stored": f"{src_info['width']}x{src_info['height']}",
        "color": f"{src_info['color_primaries'] or '?'} / {src_info['color_transfer'] or '?'}"
                 + ("  [yellow](HDR)[/]" if src_info["is_hdr"] else "  [dim](SDR)[/]"),
        "frames": f"{src_info['n_frames']} ({src_info['duration']:.1f}s)"})

    # ── 3. Convert (cached) ──────────────────────────────────────────────────
    rule(3, "Convert to MP4")
    tonemap, reason = video.tonemap_filter(src_info, mode=cfg.TONEMAP, algorithm=cfg.TONEMAP_ALGORITHM)
    console.print(f"  tone-map: {'[green]on[/] — ' if tonemap else ''}{reason}")
    mp4 = video.cache_path(cfg.INPUT_VIDEO, cfg.CACHE_DIR, long_side=cfg.INFERENCE_LONG_SIDE,
                           trim_seconds=cfg.TRIM_SECONDS, crf=cfg.CONVERT_CRF, tonemap=tonemap)
    if mp4.is_file() and not cfg.FORCE_RECONVERT:
        console.print(f"  [green]cache hit[/]: [dim]{rel(mp4)}[/]")
    else:
        video.convert(cfg.INPUT_VIDEO, mp4, long_side=cfg.INFERENCE_LONG_SIDE,
                      trim_seconds=cfg.TRIM_SECONDS, crf=cfg.CONVERT_CRF,
                      total_frames=src_info["n_frames"], console=console, tonemap=tonemap)
    info = video.inspect(mp4)
    console.print(f"  [bold]{info}[/]  ({mp4.stat().st_size / 1e6:.1f} MB)")

    # ── 4. Models, in parallel: ViTPose, SAM on the club, SAM on the ball ────
    rule(4, "Pose (ViTPose), club and ball (SAM 3.1)")
    t0 = time.perf_counter()
    with ThreadPoolExecutor(3) as pool:
        f_pose = pool.submit(models.pose_clip, api_key, mp4, info, cfg, console)
        f_club = pool.submit(models.track, api_key, mp4, info, kind="club", prompt=cfg.CLUB_PROMPT,
                             min_score=cfg.CLUB_MIN_SCORE, cfg=cfg, console=console)
        f_ball = pool.submit(models.track, api_key, mp4, info, kind="ball", prompt=cfg.BALL_PROMPT,
                             min_score=cfg.BALL_MIN_SCORE, cfg=cfg, console=console)
        posed, (club_dets, club_cost, club_calls), (ball_dets, ball_cost, ball_calls) = (
            f_pose.result(), f_club.result(), f_ball.result())
    model_seconds = time.perf_counter() - t0

    n = info.n_frames
    kp = swing_mod.keypoints(posed["by_index"], n, info.width, info.height)
    torso = swing_mod.torso_length(kp)
    grip = swing_mod.hands(kp)
    ball = swing_mod.track_ball(ball_dets, club_dets, n, torso, rest_move=cfg.BALL_REST_MOVE,
                                flight_max_step=cfg.BALL_FLIGHT_MAX_STEP, max_size=cfg.BALL_MAX_SIZE)
    sam_flight = len(ball.flight)
    faint_frames = (faint.extend(ball, mp4, n, info.fps, max_seconds=cfg.FAINT_MAX_SECONDS)
                    if cfg.FAINT_SEARCH else [])
    # The phone's drift, against the impact frame, so the ball's path stays put on the scene.
    frames_all = faint.read_frames(mp4, 0, n)
    ref = max(0, ball.rest_until)
    moved = camera.drift(frames_all, ref)
    ball.drift = np.array([moved.get(f, np.zeros(2)) for f in range(n)])
    swing_mod.fit_flight(ball, width=info.width, height=info.height, fps=info.fps,
                         extend_seconds=cfg.TRAJECTORY_EXTEND_SECONDS)
    est, focal, pitch, pitch_from = None, None, None, None
    if cfg.FLIGHT_ESTIMATE and ball.flight:
        focal = cfg.CAMERA_FOCAL_PX or (min(info.width, info.height) / 2
                                        / np.tan(np.radians(cfg.CAMERA_SHORT_SIDE_FOV_DEG) / 2))
        pitch, pitch_from = cfg.CAMERA_PITCH_DEG, "config"
        if pitch is None:
            vp = camera.vertical_vanishing_y(frames_all[ref])
            pitch = camera.pitch_from_vanishing(vp, cy=info.height / 2, focal=focal) if vp else None
            pitch_from = "the poles" if pitch is not None else "fallback"
            if pitch is None:
                pitch = cfg.CAMERA_PITCH_FALLBACK_DEG
        with console.status("[cyan]fitting the ball's flight[/]…", spinner="dots"):
            est = flight.estimate(ball, width=info.width, height=info.height, fps=info.fps,
                                  focal=focal, pitch_deg=pitch, drift=moved,
                                  sam_frames=ball.flight[:sam_flight])
        if est is not None:
            swing_mod.use_flight_model(ball, est, fps=info.fps,
                                       extend_seconds=cfg.TRAJECTORY_EXTEND_SECONDS)
    del frames_all
    club = swing_mod.track_club(club_dets, ball, grip, n, torso, max_hand_gap=cfg.CLUB_MAX_HAND_GAP)
    n_posed = int(np.isfinite(kp[:, 5, 0]).sum())
    kv({"pose": f"{cfg.POSE_MODEL}: golfer on {n_posed} of {n} frames",
        "club": f"{cfg.SAM_MODEL} \"{cfg.CLUB_PROMPT}\": on {int(club.found.sum())} of {n} frames "
                f"({club_calls} calls)",
        "ball": f"{cfg.SAM_MODEL} \"{cfg.BALL_PROMPT}\": on the tee through frame {ball.rest_until}, "
                f"in flight on {sam_flight} frame(s) ({ball_calls} calls)",
        "faint ball": f"followed by motion for {len(faint_frames)} more frame(s)"
                      + (f", to frame {faint_frames[-1]}" if faint_frames else ""),
        "time": f"{model_seconds:.1f}s",
        "cost": f"${posed['cost'] + club_cost + ball_cost:.4f}"}, title="models")

    # ── 5. The swing ─────────────────────────────────────────────────────────
    rule(5, "Swing: phases and tempo")
    if not ball.struck:
        console.print("  [yellow]the ball never left the tee; impact is taken from the hands[/]")
    impact = ball.impact if np.isfinite(ball.impact) else float("nan")
    swing = swing_mod.phases(grip, club, impact, info.fps, torso)
    fps = info.fps
    table = Table(box=None, pad_edge=False)
    for col in ("moment", "frame", "time"):
        table.add_column(col, justify="right" if col != "moment" else "left")
    for name, f in (("takeaway", swing.takeaway), ("top", swing.top), ("impact", swing.impact),
                    ("finish", swing.finish)):
        table.add_row(name, f"{f:.1f}" if isinstance(f, float) else str(f), f"{f / fps:.2f}s")
    console.print(Panel(table, title=f"[bold]tempo {swing.tempo:.1f} : 1[/]  "
                                     f"(back {swing.backswing_s:.2f}s, down {swing.downswing_s:.2f}s)",
                        title_align="left", expand=False))
    if np.isfinite(ball.launch_deg):
        side = "right" if ball.launch_deg > 0 else "left"
        console.print(f"  ball starts {abs(ball.launch_deg):.1f}° {side} of straight up the frame")
    if est is not None:
        b, r = est["best"], est["range"]
        curve = "draw (curves left)" if b.axis_tilt_deg < -3 else "fade (curves right)" \
            if b.axis_tilt_deg > 3 else "fairly straight"
        kv({"camera": f"focal {focal:.0f} px, tilted {pitch:.1f}° down (from {pitch_from}); "
                      f"drift {np.hypot(*ball.drift[ball.flight[-1]]):.0f} px over the flight",
            "ball speed": f"[bold]{b.speed * 2.237:.0f} mph[/]  (range {r['speed_mph'][0]:.0f}–{r['speed_mph'][1]:.0f})",
            "launch": f"[bold]{b.launch_deg:.0f}°[/]  (range {r['launch_deg'][0]:.0f}–{r['launch_deg'][1]:.0f})",
            "carry": f"[bold]{b.carry_m * 1.0936:.0f} yd[/]  (range {r['carry_yd'][0]:.0f}–{r['carry_yd'][1]:.0f}), "
                     f"{b.hang_s:.1f}s in the air",
            "apex": f"{b.apex_m:.0f} m  (range {r['apex_m'][0]:.0f}–{r['apex_m'][1]:.0f})",
            "shape": curve,
            "fit": f"{b.rms_px:.1f} px rms over {len(ball.flight)} sightings; {est['fits']} fits for the range"},
           title="ball flight (estimate)")
    elif cfg.FLIGHT_ESTIMATE:
        console.print("  [yellow]too few ball sightings for a flight estimate[/]")

    # ── 6. Output ────────────────────────────────────────────────────────────
    rule(6, "Render")
    run_dir = new_run_dir("swing")
    raw = run_dir / "_raw.mp4"
    t0 = time.perf_counter()
    stats = render.render(mp4, info, kp=kp, club=club, ball=ball, swing=swing, dst=raw,
                          cfg=cfg, console=console)
    render_seconds = time.perf_counter() - t0
    out_video = run_dir / f"{cfg.INPUT_VIDEO.stem}_golf.mp4"
    video.encode_h264(raw, out_video, crf=cfg.OUTPUT_CRF)
    raw.unlink(missing_ok=True)
    console.print(f"  video  -> [dim]{rel(out_video)}[/] "
                  f"({out_video.stat().st_size / 1e6:.1f} MB, {render_seconds:.1f}s)")
    still = run_dir / "swing.png"
    render.plot_swing(mp4, info, kp=kp, club=club, ball=ball, swing=swing, dst=still,
                      cfg=cfg)
    console.print(f"  image  -> [dim]{rel(still)}[/]")
    frames_csv = run_dir / f"{cfg.INPUT_VIDEO.stem}_frames.csv"
    write_frames_csv(frames_csv, kp=kp, club=club, ball=ball, swing=swing, fps=fps)
    console.print(f"  data   -> [dim]{rel(frames_csv)}[/]")

    total = time.perf_counter() - t_start
    cost = posed["cost"] + club_cost + ball_cost
    (run_dir / "run.json").write_text(json.dumps({
        "stamp": run_dir.name,
        "total_seconds": round(total, 2),
        "input": {"path": str(rel(cfg.INPUT_VIDEO)), **src_info},
        "converted": {"path": str(rel(mp4)), "width": info.width, "height": info.height,
                      "fps": fps, "n_frames": n, "tonemap": tonemap},
        "pose": {"model": cfg.POSE_MODEL, "extra_body": posed["extra_body"], "frames_posed": n_posed},
        "club": {"model": cfg.SAM_MODEL, "prompt": cfg.CLUB_PROMPT,
                 "frames_found": int(club.found.sum())},
        "ball": {"model": cfg.SAM_MODEL, "prompt": cfg.BALL_PROMPT,
                 "diameter_px": _num(ball.diameter, 1), "rest_until_frame": ball.rest_until,
                 "flight_frames": ball.flight, "flight_frames_by_motion": faint_frames,
                 "flight_px": [[f, _num(ball.x[f], 1), _num(ball.y[f], 1), _num(ball.d[f], 1)]
                               for f in ball.flight],
                 "impact_frame": _num(ball.impact, 2),
                 "start_line_deg_right_of_vertical": _num(ball.launch_deg, 1),
                 "path": "the sightings, smoothed; past the last one, projected from a quadratic "
                         "through the last dozen"},
        "flight_estimate": None if est is None else {
            "note": "a ball-flight model (gravity, drag, spin lift) fitted to the sightings; "
                    "estimated from one phone, not measured",
            "focal_px": round(focal, 1), "pitch_deg": round(pitch, 2), "pitch_from": pitch_from,
            "ball_speed_mph": round(est["best"].speed * 2.237, 1),
            "launch_deg": round(est["best"].launch_deg, 1),
            "side_deg_right_of_camera": round(est["best"].side_deg, 1),
            "carry_yd": round(est["best"].carry_m * 1.0936, 1),
            "apex_m": round(est["best"].apex_m, 1), "hang_s": round(est["best"].hang_s, 2),
            "spin_axis_tilt_deg": round(est["best"].axis_tilt_deg, 1),
            "backspin_rpm_unreliable": round(est["best"].spin_rpm),
            "impact_frame": round(est["best"].impact, 2), "fit_rms_px": round(est["best"].rms_px, 2),
            "range": est["range"]},
        "swing": swing.summary(),
        "cost_usd": {"pose": round(posed["cost"], 6), "club": round(club_cost, 6),
                     "ball": round(ball_cost, 6), "total": round(cost, 6)},
        "render": {**stats, "seconds": round(render_seconds, 2),
                   "credit": {"golfer": cfg.CREDIT_GOLFER, "handles": cfg.CREDIT}},
    }, indent=2))
    console.print()
    kv({"pose": f"${posed['cost']:.4f}", "club": f"${club_cost:.4f}", "ball": f"${ball_cost:.4f}",
        "total": f"[bold]${cost:.4f}[/]"}, title="cost")
    console.rule(f"[bold green]done[/] in {total:.1f}s", align="left")
    console.print(f"  [bold]{rel(run_dir)}/[/]\n")
    return {"tempo": swing.tempo, "cost": cost, "seconds": total, "output": run_dir}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Break down a golf swing: pose, the club segmented, the ball's launch and "
                    "the tempo. Flags override config.py for this run only.")
    parser.add_argument("inputs", nargs="*", type=Path, metavar="CLIP|DIR",
                        help=f"clips, or folders of clips (config: {rel(cfg.INPUT)})")
    parser.add_argument("--trim", type=float, metavar="SECONDS", help="only the first SECONDS")
    parser.add_argument("--slowmo", type=int, metavar="X",
                        help=f"replay the whole swing X times slower; 1 for no replay (config: {cfg.SLOWMO_FACTOR})")
    parser.add_argument("--fresh", action="store_true", help="ignore every cached gateway reply")
    credit = parser.add_argument_group(
        "credit", "a strip along the bottom: the golfer on the left, your handles on the right; "
                  "these replace config.py's CREDIT_GOLFER and CREDIT")
    credit.add_argument("--golfer", metavar="NAME", help='who is swinging, e.g. "Your Name"')
    credit.add_argument("--li", metavar="TEXT", help='LinkedIn, e.g. "Jeremy Park, PhD"')
    credit.add_argument("--x", metavar="HANDLE", help="X handle, e.g. @jeremyparkphd")
    credit.add_argument("--ig", metavar="HANDLE", help="Instagram handle")
    credit.add_argument("--no-credit", action="store_true", help="no credit at all")
    return parser.parse_args(argv)


def find_clips(inputs: list[Path]) -> list[Path]:
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
    if args.trim:
        cfg.TRIM_SECONDS = args.trim
    if args.slowmo:
        cfg.SLOWMO_FACTOR = args.slowmo
    if args.fresh:
        cfg.REUSE_CACHE = False

    def at(handle):
        return handle if handle.startswith("@") else f"@{handle}"

    handles = [(label, value) for label, value in
               (("X", args.x and at(args.x)), ("LI", args.li), ("IG", args.ig and at(args.ig))) if value]
    if handles:
        cfg.CREDIT = handles
    if args.golfer:
        cfg.CREDIT_GOLFER = args.golfer
    if args.no_credit:
        cfg.CREDIT, cfg.CREDIT_GOLFER = [], None
    clips = find_clips(args.inputs or [cfg.INPUT])
    console.print()
    console.rule("[bold]Golf: club, ball, and swing tempo[/]", align="center")
    if not clips:
        console.print(f"[red]No clips to run.[/] Put a swing in {rel(cfg.INPUT)}/, "
                      "or pass one: python main.py path/to/swing.mov")
        return 1

    rule(1, "API key")
    api_key, source = load_api_key(cfg.PROJECT_DIR)
    console.print(f"  key loaded from [green]{source}[/]")

    failed = 0
    for n, clip in enumerate(clips, 1):
        if len(clips) > 1:
            console.print()
            console.rule(f"[bold magenta]clip {n}/{len(clips)}[/] {clip.name}", align="left")
        cfg.INPUT_VIDEO = clip
        try:
            run_clip(api_key)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            console.print(f"\n[red]error on {clip.name}:[/] {exc}")
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted[/]")
        sys.exit(130)
