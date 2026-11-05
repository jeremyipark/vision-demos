"""Run ViTPose on a local video, measure the gait, and render both back onto it.

    conda activate running
    python main.py
    python main.py --height full --li "Your Name" --x @yourhandle
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import config as cfg
from src import gait, pose, render, timing, video
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
    return path.relative_to(cfg.PROJECT_DIR)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Every setting lives in config.py; these override a few per run.")

    def height(text):
        return "full" if text.lower() in ("full", "source") else int(text)

    parser.add_argument("--height", type=height, metavar="PX|full",
                        help=f"output video height in px, or 'full' for the source's own; "
                             f"only ever scales down, and the model still sees "
                             f"{cfg.INFERENCE_HEIGHT}px, so no new gateway call "
                             f"(config: {cfg.EXPORT_HEIGHT})")
    parser.add_argument("--trim", type=float, metavar="SECONDS",
                        help="use only the first SECONDS of the clip, for a quick test")
    credit = parser.add_argument_group(
        "credit", "the lines in the corner; any of these replaces config.py's CREDIT")
    credit.add_argument("--li", metavar="TEXT", help='LinkedIn, e.g. "Jeremy Park, PhD"')
    credit.add_argument("--x", metavar="HANDLE", help="X handle, e.g. @jeremyparkphd")
    credit.add_argument("--ig", metavar="HANDLE", help="Instagram handle")
    credit.add_argument("--no-credit", action="store_true", help="no credit at all")
    return parser.parse_args(argv)


def apply_args(args: argparse.Namespace) -> None:
    """Write the flags over the config, before anything reads it."""
    if args.height:
        cfg.EXPORT_HEIGHT = None if args.height == "full" else args.height
    if args.trim:
        cfg.TRIM_SECONDS = args.trim

    def at(handle):
        return handle if handle.startswith("@") else f"@{handle}"

    handles = [(label, value) for label, value in
               (("LI", args.li), ("X", args.x and at(args.x)),
                ("IG", args.ig and at(args.ig))) if value]
    if args.no_credit:
        cfg.CREDIT = []
    elif handles:
        cfg.CREDIT = handles


def main(argv: list[str] | None = None) -> int:
    apply_args(parse_args(argv))
    t_start = time.perf_counter()
    console.print()
    console.rule("[bold]ViTPose: running[/]", align="center")

    # ── 1. Key ───────────────────────────────────────────────────────────────
    rule(1, "API key")
    api_key, source = load_api_key(cfg.PROJECT_DIR)
    console.print(f"  loaded from [green]{source}[/]")

    # ── 2. Source ────────────────────────────────────────────────────────────
    rule(2, "Source video")
    if not cfg.INPUT_VIDEO.is_file():
        console.print(f"[red]Not found:[/] {cfg.INPUT_VIDEO}")
        return 1

    src_info = video.probe_source(cfg.INPUT_VIDEO)
    kv({
        "path": rel(cfg.INPUT_VIDEO),
        "size": f"{cfg.INPUT_VIDEO.stat().st_size / 1e6:.1f} MB",
        "codec": f"{src_info['codec']} ({src_info['pix_fmt']})",
        "stored": f"{src_info['width']}x{src_info['height']}",
        "rotation": f"{src_info['rotation']}°"
                    + ("  [dim](applied on convert; OpenCV would ignore it)[/]"
                       if src_info["rotation"] else ""),
        "frames": f"{src_info['n_frames']} ({src_info['duration']:.1f}s)",
    })

    # ── 3. Convert (cached) ──────────────────────────────────────────────────
    rule(3, "Convert to MP4")

    def prepare(height, label):
        """Convert to *height* (None = source), reusing a cached copy if present."""
        path = video.cache_path(
            cfg.INPUT_VIDEO, cfg.CACHE_DIR,
            target_height=height, trim_seconds=cfg.TRIM_SECONDS, crf=cfg.CONVERT_CRF,
        )
        seconds, cached = 0.0, True
        if path.is_file() and not cfg.FORCE_RECONVERT:
            console.print(f"  [green]cache hit[/] ({label}): [dim]{rel(path)}[/]")
        else:
            cached = False
            reason = "FORCE_RECONVERT" if path.is_file() else "no cached copy"
            console.print(f"  [yellow]converting[/] ({label}, {reason})")
            t0 = time.perf_counter()
            video.convert(
                cfg.INPUT_VIDEO, path,
                target_height=height, trim_seconds=cfg.TRIM_SECONDS,
                crf=cfg.CONVERT_CRF, total_frames=src_info["n_frames"], console=console,
            )
            seconds = time.perf_counter() - t0
            console.print(f"  done in {seconds:.1f}s -> [dim]{rel(path)}[/]")
        return path, video.inspect(path), seconds, cached

    # Two renditions, because they answer to different limits: the model caps at
    # a 2048px long edge internally, so uploading more is pure wait, while the
    # export answers only to what you want to watch. Pose coordinates are
    # normalized, so one inference serves any render size.
    mp4, info, convert_seconds, convert_cached = prepare(cfg.INFERENCE_HEIGHT, "inference")

    if cfg.EXPORT_HEIGHT == cfg.INFERENCE_HEIGHT:
        export_mp4, export_info = mp4, info
    else:
        export_mp4, export_info, extra_s, extra_cached = prepare(cfg.EXPORT_HEIGHT, "export")
        convert_seconds += extra_s
        convert_cached = convert_cached and extra_cached
        if export_info.n_frames != info.n_frames:
            console.print(
                f"  [yellow]warning[/] frame counts differ "
                f"({info.n_frames} inference vs {export_info.n_frames} export); "
                "poses are matched by frame index, so the overlay may drift."
            )

    console.print(f"  inference [bold]{info}[/]  ({mp4.stat().st_size / 1e6:.1f} MB)")
    if export_mp4 is not mp4:
        console.print(f"  export    [bold]{export_info}[/]  "
                      f"({export_mp4.stat().st_size / 1e6:.1f} MB)")

    # ── 4. Request ───────────────────────────────────────────────────────────
    rule(4, "Pose estimation")
    video_b64, extra_body = pose.build_request(
        mp4,
        every_frame=cfg.EVERY_FRAME, fps=info.fps, n_frames=info.n_frames,
        video_fps=cfg.VIDEO_FPS, video_max_frames=cfg.VIDEO_MAX_FRAMES,
        precision=cfg.PRECISION,
    )
    kv({
        "model": cfg.MODEL,
        "mode": ("every frame, detector stride 1, nothing skipped"
                 if cfg.EVERY_FRAME else f"video_fps={cfg.VIDEO_FPS} (detector cadence)"),
        "video_fps": extra_body["video_fps"],
        "video_max_frames": extra_body.get("video_max_frames", "not set (model default: 900)"),
        "billed units": f"~{extra_body.get('video_max_frames') or min(info.n_frames, 900)} frames",
        "upload": f"{len(video_b64) / 1e6:.1f} MB base64",
    }, title="request")

    from openai import OpenAI

    poses_cache = pose.cache_path(mp4, cfg.CACHE_DIR, model=cfg.MODEL,
                                  extra_body=extra_body)
    cached = pose.load_cache(poses_cache) if cfg.REUSE_POSES else None

    if cached is not None:
        # Reuse is deliberate and loud: rendering changes should not cost a model
        # call, but a silent cache hit would make a stale result look fresh.
        payload, usage, cached_timing, created = cached
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
        client = OpenAI(
            base_url=cfg.GATEWAY_BASE_URL, api_key=api_key,
            timeout=cfg.REQUEST_TIMEOUT, max_retries=1,
            **({"http_client": http_client} if http_client else {}),
        )

        t0 = time.perf_counter()
        with console.status("[cyan]waiting on the gateway[/]; pose runs on every decoded frame…",
                            spinner="dots"):
            payload, usage = pose.request_poses(
                client, model=cfg.MODEL, video_b64=video_b64, extra_body=extra_body
            )
        elapsed = time.perf_counter() - t0
        request_timing = timing.summarize(marks, elapsed)

        pose.save_cache(poses_cache, payload, usage, asdict(request_timing),
                        stamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    frames, by_index = pose.unwrap(payload)

    # Before anything reads the poses: one body should be one subject. Done here
    # rather than in the renderer so the overlay and the gait signal agree: the
    # signal reads persons[0], and a duplicate could otherwise swap which body
    # the ankles come from mid-stride.
    dropped = pose.dedupe(frames, max_persons=cfg.MAX_PERSONS,
                          iou=cfg.PERSON_IOU_THRESHOLD)
    if dropped:
        by_index = {f["index"]: f["persons"] for f in frames}
        console.print(f"  [dim]{dropped} duplicate detections suppressed "
                      f"(IoU > {cfg.PERSON_IOU_THRESHOLD}), tracker identity handoff[/]")
    result = pose.PoseResult(payload, frames, by_index, elapsed, usage)

    if cached is None:
        if request_timing.measured_split:
            console.print(
                f"  [green]done in {elapsed:.1f}s[/]: "
                f"upload {request_timing.upload:.1f}s, "
                f"[bold]server {request_timing.server:.1f}s[/], "
                f"download {request_timing.download:.1f}s"
            )
        else:
            console.print(f"  [green]done in {elapsed:.1f}s[/]")

    coverage = len(by_index) / info.n_frames if info.n_frames else 0
    kv({
        "frames returned": f"{len(frames)} of {info.n_frames} source ({coverage:.0%})",
        "frames with a person": result.n_posed,
        "track ids": result.track_ids or "none",
        "usage": usage or "—",
    }, title="response")

    if cfg.EVERY_FRAME and len(by_index) < info.n_frames:
        console.print(
            f"  [yellow]note[/] {info.n_frames - len(by_index)} frames were not returned "
            "despite EVERY_FRAME; the decoder's frame count can differ slightly from OpenCV's."
        )

    # ── 5. Gait ──────────────────────────────────────────────────────────────
    rule(5, "Gait")
    stamp = datetime.now().strftime(cfg.RUN_STAMP_FORMAT)
    run_dir = cfg.OUTPUT_DIR / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    poses_path = run_dir / "poses.json"
    poses_path.write_text(json.dumps(payload, indent=2))
    console.print(f"  poses  -> [dim]{rel(poses_path)}[/] "
                  f"({poses_path.stat().st_size / 1e6:.1f} MB)")

    analysis = None
    if cfg.ANALYZE_GAIT:
        # The aspect ratio of the frame the *model* was shown, since that is
        # what its coordinates are normalized against. Both renditions share
        # it -- they differ only in height -- but this is the honest source.
        analysis = gait.analyze(frames, info.fps, export_info.n_frames, cfg=cfg,
                                aspect=info.width / info.height)
        left, right = analysis.balance()
        if analysis.count >= 2:
            table = Table.grid(padding=(0, 3))
            table.add_column(style="dim", justify="right")
            table.add_column()
            for key, value in {
                "cadence": f"[bold green]{analysis.mean_cadence:.1f} steps/min[/]",
                "steps": f"{analysis.count}  "
                         f"({len(analysis.steps_for('left'))} left, "
                         f"{len(analysis.steps_for('right'))} right)",
                "step time": f"{np.mean(analysis.intervals):.3f}s",
                "stride time": f"{analysis.mean_stride:.3f}s",
                # The real per-foot rate, printed for both feet precisely
                # because it has to come out equal: the feet alternate, so a
                # gap here would mean the detection is wrong, not the runner.
                "stride rate": f"{analysis.foot_stride_rate('left'):.2f} spm left / "
                               f"{analysis.foot_stride_rate('right'):.2f} spm right"
                               f"  [dim](equal by construction)[/]",
                "phase split": f"{analysis.phase_split()[0]:.1f}% / "
                               f"{analysis.phase_split()[1]:.1f}%  "
                               f"[dim](one stride, two halves)[/]",
                "stride var": f"[bold]{analysis.stride_spread() * 1000:.1f} ms[/]  "
                              f"[dim](stride to stride; the clean one)[/]",
                "symmetry": f"{analysis.symmetry():.1f}%  "
                            f"[dim](phase; NOT measurable from one side view)[/]",
                "contact": f"{analysis.contact_mean('left'):.3f}s left / "
                           f"{analysis.contact_mean('right'):.3f}s right  "
                           f"[dim]({left:.1f}% / {right:.1f}%)[/]",
                "airborne": f"{analysis.flight_ratio * 100:.0f}% of the clip",
                "vertical osc.": f"{analysis.vertical_oscillation():.3f} leg lengths",
            }.items():
                table.add_row(key, value)
            console.print(Panel(table, title="[bold]gait[/]", title_align="left",
                                expand=False))
        else:
            console.print("  [yellow]too few foot strikes detected[/]. "
                          "Check clearance.png")

        # The failure modes that would otherwise pass silently.
        w = analysis.warnings
        if w.get("consecutive_same_foot_strikes"):
            console.print(f"  [yellow]warning[/] "
                          f"{w['consecutive_same_foot_strikes']} consecutive "
                          "same-foot strikes: a strike on the other foot was "
                          "missed, and those strides span two steps")
        if w.get("untimed_strikes"):
            console.print(f"  [dim]{w['untimed_strikes']} strike(s) not timed "
                          "(stance cut off by the clip end); counted, not "
                          "measured against[/]")
        if w.get("knee_landmark_on_window_edge"):
            console.print(f"  [dim]{w['knee_landmark_on_window_edge']} knee "
                          "landmark(s) hit the search-window edge; widen "
                          "KNEE_SEARCH_SECONDS if that grows[/]")
        if w.get("implausible_stance_flexion_frames"):
            console.print(
                f"  [yellow]warning[/] "
                f"{w['implausible_stance_flexion_frames']} frame(s) have a knee "
                f"bent past {cfg.KNEE_STANCE_LIMIT_DEG:.0f}° while the foot is "
                "down, which no runner does. Usually a left/right keypoint swap "
                "as the legs cross; check knee.png")
        interp = sum((w.get("frames_interpolated") or {}).values())
        if interp:
            console.print(f"  [dim]{interp} limb frame(s) interpolated across "
                          "gaps; excluded as knee landmarks[/]")

    # ── 6. Render ────────────────────────────────────────────────────────────
    rule(6, "Render")
    raw = run_dir / "_raw.mp4"
    t0 = time.perf_counter()
    stats = render.render(export_info, by_index, len(frames), raw, cfg=cfg, console=console,
                          analysis=analysis)
    render_seconds = time.perf_counter() - t0
    console.print(f"  drew poses on [bold]{stats['frames_with_pose']}/{stats['frames_written']}[/] "
                  f"frames" + (f" ({stats['frames_held']} held)" if stats["frames_held"] else ""))

    out_video = run_dir / f"{cfg.INPUT_VIDEO.stem}_pose.mp4"
    t0 = time.perf_counter()
    video.encode_h264(raw, out_video, crf=cfg.OUTPUT_CRF)
    encode_seconds = time.perf_counter() - t0
    raw.unlink(missing_ok=True)   # MPEG-4 Part 2 intermediate; no player wants it
    console.print(f"  video  -> [dim]{rel(out_video)}[/] "
                  f"({out_video.stat().st_size / 1e6:.1f} MB)")

    clearance_path = steps_path = knee_path = None
    if analysis is not None:
        if cfg.SAVE_CLEARANCE_PLOT:
            clearance_path = run_dir / "clearance.png"
            render.plot_clearance(
                analysis, clearance_path, cfg=cfg,
                title=f"Ankle height: {analysis.count} strikes, "
                      f"{analysis.mean_cadence:.1f} steps/min",
            )
            console.print(f"  graph  -> [dim]{rel(clearance_path)}[/]")

        if cfg.SAVE_KNEE_PLOT and analysis.steps:
            knee_path = run_dir / "knee.png"
            left, right = (analysis.segment_ratio(f) for f in ("left", "right"))
            if render.plot_knee(
                analysis, knee_path, cfg=cfg,
                title=f"Knee at foot strike — {analysis.count} strikes  "
                      f"(shank/thigh {left:.2f} left, {right:.2f} right; "
                      f"should be ~1.0)"):
                console.print(f"  graph  -> [dim]{rel(knee_path)}[/]")
            else:
                knee_path = None

        if cfg.SAVE_STEPS_PLOT and analysis.steps:
            steps_path = run_dir / "steps.png"
            render.plot_steps(analysis, steps_path, cfg=cfg,
                              title="Per step: timing and contact, by foot")
            console.print(f"  graph  -> [dim]{rel(steps_path)}[/]")

        gait_path = run_dir / "gait.json"
        gait_path.write_text(json.dumps({
            "fps": info.fps,
            "signal": gait.signal_config(cfg),
            "leg_length": round(analysis.leg_length, 5),
            # The frame's width/height, which every angle and length in here was
            # measured after correcting for. Recorded because a wrong value
            # would skew all of them silently. See gait.analyze.
            "frame_aspect": round(analysis.frame_aspect, 5),
            "ground": {foot: round(v, 5)
                       for foot, v in analysis.ground.items()},
            "contact_level": round(analysis.contact_level, 5),
            **analysis.summary(),
            "series": analysis.series(),
        }, indent=2))
        console.print(f"  gait   -> [dim]{rel(gait_path)}[/]")

        summary_path = run_dir / "summary.txt"
        summary_path.write_text(gait.summary_text(
            analysis, video_seconds=info.duration, source=cfg.INPUT_VIDEO.name))
        console.print(f"  summary-> [dim]{rel(summary_path)}[/]")

    plot_path = None
    if cfg.SAVE_JOINT_PLOT:
        plot_path = run_dir / f"{cfg.PLOT_JOINT}.png"
        if render.plot_joint(frames, info.fps, cfg.PLOT_JOINT, plot_path):
            console.print(f"  plot   -> [dim]{rel(plot_path)}[/]")
        else:
            console.print(f"  [yellow]plot skipped[/]: {cfg.PLOT_JOINT} never visible")
            plot_path = None

    # ── 7. Metrics ───────────────────────────────────────────────────────────
    rule(7, "Performance")
    total = time.perf_counter() - t_start
    metrics = timing.Metrics(
        frames=len(frames),
        video_seconds=info.duration,
        convert_seconds=convert_seconds,
        convert_cached=convert_cached,
        encode_seconds=encode_seconds,
        request=request_timing,
        render_seconds=render_seconds,
        total_seconds=total,
        upload_mb=len(video_b64) / 1e6,
        cost_usd=(usage or {}).get("cost"),
        poses_cached=cached is not None,
    )

    (run_dir / "metrics.json").write_text(json.dumps(metrics.as_dict(), indent=2))
    (run_dir / "metrics.txt").write_text(metrics.as_text())

    m = metrics.as_dict()
    kv({
        "ViTPose throughput": f"[bold green]{m['pose']['fps']:.1f} fps[/]  "
                              f"({m['pose']['ms_per_frame']:.0f} ms/frame, "
                              f"{m['pose']['realtime_factor']:.2f}x realtime)",
        "server time": f"{m['pose']['seconds']:.2f}s for {m['frames']} frames",
        "gateway round trip": f"{m['gateway_round_trip']['seconds']:.2f}s "
                              f"({m['gateway_round_trip']['fps']:.1f} fps end-to-end)",
        "  upload": f"{m['gateway_round_trip']['upload_seconds']:.2f}s "
                    f"({m['gateway_round_trip']['upload_mb']:.1f} MB)"
                    if m["gateway_round_trip"]["upload_seconds"] is not None else "not separable",
        "render": f"{m['local']['render_seconds']:.2f}s ({m['local']['render_fps']:.1f} fps)",
        "cost": f"${m['cost']['usd']:.6f}  "
                f"(${m['cost']['usd_per_1000_frames']:.4f} / 1000 frames)"
                if "cost" in m else "—",
    }, title="metrics")
    console.print(f"  metrics -> [dim]{rel(run_dir / 'metrics.txt')}[/] + metrics.json")

    (run_dir / "run.json").write_text(json.dumps({
        "stamp": stamp,
        "total_seconds": round(total, 2),
        "input": {"path": str(cfg.INPUT_VIDEO), **src_info},
        "converted": {"path": str(mp4), "width": info.width, "height": info.height,
                      "fps": info.fps, "n_frames": info.n_frames},
        "export": {"path": str(export_mp4), "width": export_info.width,
                   "height": export_info.height, "n_frames": export_info.n_frames},
        "request": {"model": cfg.MODEL, "extra_body": extra_body,
                    "upload_bytes_b64": len(video_b64)},
        "response": {"elapsed_seconds": round(elapsed, 2), "frames_returned": len(frames),
                     "frames_with_person": result.n_posed, "track_ids": result.track_ids,
                     "usage": usage},
        "render": {**stats, "draw_face": cfg.DRAW_FACE, "credit": cfg.CREDIT},
        "metrics": metrics.as_dict(),
        "gait": {**analysis.summary(), "signal": gait.signal_config(cfg)}
                if analysis else None,
        "quality": analysis.warnings if analysis else None,
        "outputs": {"video": out_video.name, "poses": poses_path.name,
                    "plot": plot_path.name if plot_path else None,
                    "clearance": clearance_path.name if clearance_path else None,
                    "steps": steps_path.name if steps_path else None,
                    "knee": knee_path.name if knee_path else None,
                    "gait": "gait.json" if analysis else None,
                    "summary": "summary.txt" if analysis else None,
                    "metrics": "metrics.json"},
    }, indent=2))

    console.print()
    console.rule(f"[bold green]done[/] in {total:.1f}s", align="left")
    console.print(f"  [bold]{rel(run_dir)}/[/]\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted[/]")
        sys.exit(130)
    except Exception as exc:
        console.print(f"\n[red]error:[/] {exc}")
        sys.exit(1)
