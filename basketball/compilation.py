"""A compilation of shots from several clips, in the main video's layout (clip on top, court below).

    python compilation.py                  # every clip in data/input/: 4 swishes, 2 rim-then-in, 2 misses
    python compilation.py --swish 3 --rim 1 --miss 1
    python compilation.py --cached-only    # never call the gateway; skip a clip that isn't cached

Walks the clips in the order highlights.py plays them (recorded first, then by path) and takes each
shot while its kind (swish, rim then in, miss) still has room, until every kind is full. So the
compilation is the session in order, minus the shots that would overfill a kind.

    swish         a make with no rim contact       (rim.py)
    rim then in   a make that touched the rim first
    miss          any miss

Only shots that cross the frame from the side are used, makes and misses alike. A shot toward or away
from the camera is left out: it doesn't render well, and rim.py can't see the rim's front or back from there.

The picked shots are a selection, not a session, so the panel shows no field goals or FG%, only the
spots on the court. Nor does a hot streak light up: three makes in a row here were picked, not made.

Each shot runs from --before seconds ahead of the hands going up to --after seconds past the verdict.
Writes data/output/compilation/compilation.mp4. Everything else (--height, --li, --x, ...) is
main.py's: `python main.py -h`.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
import types
import warnings
from pathlib import Path

import cv2

import config as cfg
import history as hist
import main
import rim
from src import render, video
from src.env import load_api_key
from src.panel import Panel
from src.render import frame_size

console = main.console
OUT = cfg.OUTPUT_DIR / "compilation" / "compilation.mp4"
DEFAULT_INPUTS = [cfg.INPUT]
FPS = 30.0


def kind_of(shot, hoop, fps: float) -> str | None:
    """``swish``, ``rim`` (a make that touched the rim first), ``miss``; None if it can't be called."""
    if shot.made is None:
        return None
    r = rim.read_shot(shot, hoop, fps)
    if r["view"] != "across":   # toward or away from the camera: doesn't render well
        return None
    if not shot.made:
        return "miss"
    return "swish" if r["outcome"] == "swish" else "rim"


def pick(clips: list[dict], want: dict[str, int]) -> list[dict]:
    """The session in order, a shot taken while its kind has room, until every kind is full."""
    left = dict(want)
    picked = []
    for c in clips:
        for shot in c["shots"]:
            kind = c["kinds"][shot.number]
            if left.get(kind, 0) > 0:
                left[kind] -= 1
                picked.append({"clip": c, "shot": shot, "kind": kind})
                if not any(left.values()):
                    return picked
    short = ", ".join(f"{n} more {k}" for k, n in left.items() if n)
    raise SystemExit(f"Not enough shots: need {short}.")


def window(c: dict, shot, before: float, after: float) -> range:
    """The clip's frames for a shot: from just ahead of the motion to past the verdict."""
    fps = c["info"].fps
    first = shot.start if shot.motion is None else shot.motion
    last = (shot.decided if shot.decided is not None else shot.end) + int(cfg.VERDICT_SECONDS * fps)
    return range(max(0, first - int(before * fps)),
                 min(c["info"].n_frames, last + int(after * fps)))


def main_(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0], add_help=False)
    parser.add_argument("-h", "--help", action="store_true")
    parser.add_argument("--swish", type=int, default=4, help="swishes (default 4)")
    parser.add_argument("--rim", type=int, default=2, help="makes that touched the rim first (default 2)")
    parser.add_argument("--miss", type=int, default=2, help="misses (default 2)")
    parser.add_argument("--before", type=float, default=1.0, help="seconds ahead of the shot")
    parser.add_argument("--after", type=float, default=0.8, help="seconds past the verdict")
    parser.add_argument("--cached-only", action="store_true",
                        help="never call the gateway; skip a clip whose replies aren't cached")
    parser.add_argument("--out", type=Path, default=OUT)
    mine, rest = parser.parse_known_args(argv)
    if mine.help:
        parser.print_help()
        console.print("\nand main.py's: --height, --trim, --fresh, --li, --x, --ig, --no-credit, "
                      "and the clips or folders to use.")
        return 0
    args = main.parse_args(rest)
    main.apply_args(args)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    clips = hist.find_clips(args.inputs or DEFAULT_INPUTS)
    if not clips:
        console.print("[red]No clips.[/]")
        return 1
    api_key = ""
    if mine.cached_only:
        main.client = hist._refuse
    else:
        try:
            api_key, _ = load_api_key(cfg.PROJECT_DIR)
        except Exception:
            main.client = hist._refuse

    infos = {clip: hist.capture_info(clip) for clip in clips}
    clips = sorted(clips, key=lambda c: (infos[c]["when"], hist.rel(c)))
    loaded = []
    for clip in clips:
        console.rule(f"[bold magenta]{hist.session_label(infos[clip]['when'])}[/]  {hist.rel(clip)}",
                     align="left")
        try:
            c = hist.load_clip(clip, api_key)
        except hist.NotCached as exc:
            console.print(f"  [yellow]skipped[/]: not cached ({exc}); run without --cached-only")
            continue
        c["path"] = clip
        c["kinds"] = {s.number: kind_of(s, c["hoop"], c["info"].fps) for s in c["shots"]}
        loaded.append(c)

    picked = pick(loaded, {"swish": mine.swish, "rim": mine.rim, "miss": mine.miss})

    # One panel for the whole video, on a timeline of its own: each shot's segment follows
    # the last, and the shots (renumbered 1..n) are decided where they fall on it.
    sizes = {frame_size(c["info"], None, cfg) for c in loaded}
    if len(sizes) != 1:
        console.print("[red]The clips differ in size; can't share a panel.[/]")
        return 1
    width, out_h, video_h = sizes.pop()
    segments, virtual, cursor = [], [], 0
    for n, p in enumerate(picked, 1):
        c, shot = p["clip"], p["shot"]
        frames = window(c, shot, mine.before, mine.after)
        shift = cursor - frames.start
        virtual.append(dataclasses.replace(
            shot, number=n, start=shot.start + shift, end=shot.end + shift,
            release=shot.release + shift, decided=shot.decided + shift,
            motion=None if shot.motion is None else shot.motion + shift))
        segments.append((c, frames, cursor))
        cursor += len(frames)

    # No streak: three makes in a row in a compilation were chosen, not made.
    quiet = types.SimpleNamespace(**{k: getattr(cfg, k) for k in dir(cfg) if k.isupper()})
    quiet.HOT_STREAK = 10 ** 9
    footer = int(round(cfg.CREDIT_FOOTER * width / 1280 / 2)) * 2 if render.credit_lines(cfg) else 0
    panel = Panel(virtual, FPS, width, out_h - video_h, cfg=quiet, footer=footer, show_fg=False)

    mine.out.parent.mkdir(parents=True, exist_ok=True)
    raw = mine.out.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (width, out_h))
    t0 = time.perf_counter()
    for n, ((c, frames, offset), p) in enumerate(zip(segments, picked), 1):
        console.print(f"  {n}/{len(picked)}  {p['kind']:<5}  {hist.rel(c['path'])}  "
                      f"shot {p['shot'].number}  {frames.start / c['info'].fps:.1f}s-"
                      f"{frames.stop / c['info'].fps:.1f}s")
        render.render(c["info"], kp=c["kp"], hoop=c["hoop"], track=c["track"], shots=c["shots"],
                      dst=None, cfg=quiet, console=console, frames=frames, panel=panel,
                      offset=offset, writer=writer)
    writer.release()
    video.encode_h264(raw, mine.out, crf=cfg.OUTPUT_CRF)
    raw.unlink(missing_ok=True)
    kinds = [p["kind"] for p in picked]
    console.print(f"\n  {kinds.count('swish')} swishes, {kinds.count('rim')} rim then in, "
                  f"{kinds.count('miss')} misses, {cursor / FPS:.0f}s")
    console.print(f"  -> [bold]{main.rel(mine.out)}[/] "
                  f"({mine.out.stat().st_size / 1e6:.1f} MB, {time.perf_counter() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main_())
