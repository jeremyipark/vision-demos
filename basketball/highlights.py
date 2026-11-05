"""A highlight reel for checking form by eye: every side-on shot back to back, misses first,
with the shooting arm and leg drawn on and their angles printed.

    python highlights.py                  # side-on shots from every clip under data/input/
    python highlights.py --all            # every shot, side-on or not
    python highlights.py --rim            # rim.py's shots: misses, then rim-then-in, then
                                          # the swishes, with the ball and its rim touch
    python highlights.py --cached-only    # never call the gateway; skip a clip that isn't cached

Each shot runs from --before seconds ahead of the release to --after seconds past it,
long enough to see the ball reach the hoop and come down; shots play straight on, one
after another. On screen:

    top      MISS 1 of 3, the session, the shot number and its time in the clip, and the
             shot's measures (as form.py reads them)
    body     the shooting arm (blue) and leg (orange); the elbow and knee angle beside each
    corner   the elbow angle through the shot, with a cursor at this frame

The elbow on screen and in the corner graph is the angle of the arm as detected, every
frame: no frames are left out, so it always matches what is drawn on the body.

With --rim the reel checks rim.py by eye: every shot it charts (seen from the side), the
misses first, then the makes that touched the rim, then the swishes. The ball is
circled with a short trail, and the moment rim.py calls a rim touch is marked where it
happened and stays marked; a make is labeled at the hoop as it goes through.

Uses the same cache as main.py, so nothing is called again. Writes
data/output/highlights/highlights.mp4 (rim_highlights.mp4 with --rim).
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config as cfg
import form
import history as hist
import main
import rim
from src import video
from src.env import load_api_key
from src.shots import J
from src.text import resolve_font

console = main.console
OUT = cfg.OUTPUT_DIR / "highlights" / "highlights.mp4"
RIM_OUT = cfg.OUTPUT_DIR / "highlights" / "rim_highlights.mp4"

MAKE = (81, 207, 102)      # RGB
MISS = (255, 107, 107)
ARM = (28, 126, 214)
LEG = (255, 146, 43)
WHITE = (255, 255, 255)
GRAY = (170, 170, 170)
RIM_IN = (47, 158, 68)     # a make that touched the rim: a deeper green
BALL = (255, 146, 43)


class Fonts:
    def __init__(self, height: int):
        found = resolve_font("auto")
        s = height / 1080

        def face(px):
            px = max(10, int(px * s))
            if found is None:
                return ImageFont.load_default()
            return ImageFont.truetype(found[0], px, index=found[1])

        self.big, self.mid, self.small, self.tiny = face(54), face(32), face(26), face(20)
        self.scale = s


# ── what goes in the reel ────────────────────────────────────────────────────

def collect(clips: list[Path], api_key: str, side_on_only: bool, max_ratio: float,
            rim_mode: bool = False) -> list[dict]:
    infos = {clip: hist.capture_info(clip) for clip in clips}
    clips = sorted(clips, key=lambda c: (infos[c]["when"], hist.rel(c)))
    shared = {c for c in clips for d in clips if d != c and infos[d]["when"] == infos[c]["when"]}
    items = []
    for clip in clips:
        when = infos[clip]["when"]
        tag = (clip.parent.name if clip.parent != cfg.INPUT else clip.stem) if clip in shared else ""
        label = hist.session_label(when) + (f" ({tag})" if tag else "")
        console.rule(f"[bold magenta]{label}[/]  {hist.rel(clip)}", align="left")
        try:
            c = hist.load_clip(clip, api_key)
        except hist.NotCached as exc:
            console.print(f"  [yellow]skipped[/]: not cached ({exc}); run without --cached-only")
            continue
        c["knee"] = form.joint_angle(c["kp"], f"{c['side']}_hip", f"{c['side']}_knee",
                                     f"{c['side']}_ankle")
        c["raw_elbow"] = form.joint_angle(c["kp"], f"{c['side']}_shoulder", f"{c['side']}_elbow",
                                          f"{c['side']}_wrist")
        for shot in c["shots"]:
            if shot.made is None:
                continue
            r = None
            if rim_mode:   # rim.py's shots: the ones it can read, seen from the side
                r = rim.read_shot(shot, c["hoop"], c["info"].fps)
                if r["view"] != "across":
                    continue
            m = form.measure(c, shot, max_ratio)
            if m is None:
                if side_on_only:
                    continue
                m = form.measure(c, shot, float("inf"))   # measured anyway, flagged below
                m["not_side_on"] = True
            items.append({"clip": c, "shot": shot, "m": m, "label": label, "rim": r})
        console.print(f"  {sum(1 for i in items if i['clip'] is c)} shots")
    # Misses first, then makes; each in the order they were taken.
    items.sort(key=lambda i: (bool(i["shot"].made), i["label"], i["shot"].number))
    return items


# ── drawing ──────────────────────────────────────────────────────────────────

def _deg(v) -> str:
    return f"{v:.0f}°" if v is not None and np.isfinite(v) else "—"


def text_box(draw, xy, lines, fonts, pad=14):
    """Lines of (text, font, color) on a dark box at *xy*."""
    x, y = xy
    sizes = [draw.textbbox((0, 0), t, font=f) for t, f, _ in lines]
    w = max(b[2] - b[0] for b in sizes) + 2 * pad
    h = sum(b[3] - b[1] for b in sizes) + pad * (len(lines) + 1)
    draw.rounded_rectangle([x, y, x + w, y + h], radius=12, fill=(0, 0, 0, 165))
    cy = y + pad
    for (t, f, color), b in zip(lines, sizes):
        draw.text((x + pad, cy - b[1]), t, font=f, fill=color)
        cy += b[3] - b[1] + pad


def draw_frame(img_bgr: np.ndarray, f: int, item: dict, k: int, n: int, fonts: Fonts
               ) -> np.ndarray:
    c, shot, m = item["clip"], item["shot"], item["m"]
    kp, side, fps = c["kp"], c["side"], c["info"].fps
    s = fonts.scale
    img = Image.fromarray(img_bgr[..., ::-1]).convert("RGBA")
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    # The shooting arm and leg, with their angles.
    def limb(names, color, value, label):
        pts = kp[f, [J[f"{side}_{j}"] for j in names]]
        if not np.isfinite(pts).all():
            return
        p = [tuple(map(float, q)) for q in pts]
        d.line(p, fill=color + (255,), width=max(3, int(7 * s)), joint="curve")
        for q in p:
            r = 7 * s
            d.ellipse([q[0] - r, q[1] - r, q[0] + r, q[1] + r], fill=WHITE + (255,),
                      outline=color + (255,), width=max(2, int(3 * s)))
        ex, ey = p[1]
        text_box(d, (ex + 18 * s, ey - 26 * s),
                 [(f"{label} {_deg(value)}", fonts.small, WHITE)], fonts, pad=int(8 * s))

    limb(("hip", "knee", "ankle"), LEG, c["knee"][f], "knee")
    limb(("shoulder", "elbow", "wrist"), ARM, c["raw_elbow"][f], "elbow")

    # Top: what this shot is.
    made = bool(shot.made)
    t = shot.release / fps
    head = f"{item.get('head') or ('MAKE' if made else 'MISS')} {k} of {n}"
    where = f"{item['label']}  ·  shot {shot.number}  ·  {int(t // 60)}:{t % 60:04.1f} in the clip"
    stats = (f"set {_deg(m['elbow_set_deg'])}   release {_deg(m['elbow_release_deg'])}   "
             f"follow-through {_deg(m['follow_through_deg'])}   knee {_deg(m['knee_bend_deg'])}"
             f"   side-on {m['side_on_ratio']:.2f}"
             + ("  (not side-on)" if m.get("not_side_on") else ""))
    text_box(d, (int(28 * s), int(28 * s)),
             [(head, fonts.big, item.get("color") or (MAKE if made else MISS)),
              (where, fonts.small, WHITE),
              (stats, fonts.tiny, (220, 220, 220))], fonts, pad=int(14 * s))

    if item.get("rim") is not None:
        draw_ball(d, f, item, fonts)

    # Corner: the elbow through the shot, with a cursor.
    W, H = img.size
    gw, gh = int(460 * s), int(200 * s)
    gx, gy = W - gw - int(28 * s), H - gh - int(28 * s)
    d.rounded_rectangle([gx, gy, gx + gw, gy + gh], radius=12, fill=(0, 0, 0, 165))
    f0, f1 = item["span"]
    ang = c["raw_elbow"][f0:f1].copy()
    ok = np.isfinite(ang)
    if ok.any() and not ok.all():   # a frame with no arm at all: bridge it, don't break the line
        idx = np.arange(len(ang))
        ang[~ok] = np.interp(idx[~ok], idx[ok], ang[ok])
    xs = np.linspace(gx + 16 * s, gx + gw - 16 * s, len(ang))
    top, bottom = gy + 34 * s, gy + gh - 14 * s

    def y_of(v):
        return bottom - (v / 180) * (bottom - top)

    for v in (90, 180):
        d.line([(xs[0], y_of(v)), (xs[-1], y_of(v))], fill=(255, 255, 255, 50), width=1)
    line = [(x, y_of(v)) for x, v in zip(xs, ang) if np.isfinite(v)]
    if len(line) > 1:
        d.line(line, fill=ARM + (255,), width=max(2, int(3 * s)), joint="curve")
    rel = m["release_frame"] - f0
    if 0 <= rel < len(xs):
        d.line([(xs[rel], top), (xs[rel], bottom)], fill=(255, 255, 255, 120), width=1)
    cur = min(max(f - f0, 0), len(xs) - 1)
    d.line([(xs[cur], top), (xs[cur], bottom)], fill=(255, 212, 59, 255), width=max(2, int(3 * s)))
    d.text((gx + 14 * s, gy + 8 * s), "elbow through the shot  (0–180°)", font=fonts.tiny,
           fill=(220, 220, 220))

    out = Image.alpha_composite(img, layer).convert("RGB")
    return np.asarray(out)[..., ::-1].copy()


def draw_ball(d, f: int, item: dict, fonts: Fonts) -> None:
    """The ball with a short trail; rim.py's touch, marked where and once it happened; and at
    the hoop, once a make goes through, how it went in."""
    c, shot, r = item["clip"], item["shot"], item["rim"]
    hoop, s = c["hoop"], fonts.scale
    radius = hoop.width * rim.BALL_R
    path = {int(pf): (px, py) for pf, px, py in shot.path}
    trail = [path[g] for g in range(f - 10, f + 1) if g in path]
    for j, (x, y) in enumerate(trail[:-1]):
        a = int(60 + 140 * j / max(1, len(trail) - 1))
        d.ellipse([x - 4 * s, y - 4 * s, x + 4 * s, y + 4 * s], fill=BALL + (a,))
    if f in path:
        x, y = path[f]
        d.ellipse([x - radius, y - radius, x + radius, y + radius], outline=BALL + (255,),
                  width=max(2, int(3 * s)))
    if r["frame"] is not None and f >= r["frame"]:
        x, y = r["px"]
        rr = radius * 1.25
        d.ellipse([x - rr, y - rr, x + rr, y + rr], outline=MISS + (255,), width=max(3, int(5 * s)))
        text_box(d, (x + rr + 10 * s, y - rr - 30 * s),
                 [(f"rim touch: {('back' if r['u'] > rim.FRONT_BACK else 'front' if r['u'] < -rim.FRONT_BACK else 'side (toward/away from camera)')}",
                   fonts.small, WHITE)], fonts, pad=int(8 * s))
    if shot.made and shot.crossing is not None and f >= shot.crossing:
        clean = r["outcome"] == "swish"
        text_box(d, (hoop.right + 30 * s, hoop.cy + 20 * s),
                 [("swish" if clean else "in, off the rim", fonts.mid,
                   MAKE if clean else RIM_IN)], fonts, pad=int(10 * s))


# ── main ─────────────────────────────────────────────────────────────────────

def main_(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("inputs", nargs="*", type=Path, metavar="CLIP|DIR",
                        help=f"clips or folders (searched with subfolders); default {main.rel(cfg.INPUT)}")
    parser.add_argument("--all", action="store_true", help="every shot, not only side-on ones")
    parser.add_argument("--rim", action="store_true",
                        help="rim.py's shots: misses, then rim-then-in, then the swishes")
    parser.add_argument("--max-ratio", type=float, default=form.FORM_MAX_RATIO,
                        help=f"side-on test, as in form.py (default {form.FORM_MAX_RATIO})")
    parser.add_argument("--before", type=float, default=1.2, help="seconds before the release")
    parser.add_argument("--after", type=float, default=2.6,
                        help="seconds after the release: through the hoop and down to the floor")
    parser.add_argument("--cached-only", action="store_true",
                        help="never call the gateway; skip a clip whose replies aren't cached")
    parser.add_argument("--out", type=Path, default=None,
                        help=f"the MP4 to write (default {main.rel(OUT)}, or {main.rel(RIM_OUT)} with --rim)")
    args = parser.parse_args(argv)
    args.out = args.out or (RIM_OUT if args.rim else OUT)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    clips = hist.find_clips(args.inputs or [cfg.INPUT])
    if not clips:
        console.print(f"[red]No clips.[/] Put shooting videos in {main.rel(cfg.INPUT)}/")
        return 1
    api_key = ""
    if args.cached_only:
        main.client = hist._refuse
    else:
        try:
            api_key, _ = load_api_key(cfg.PROJECT_DIR)
        except Exception:
            main.client = hist._refuse

    items = collect(clips, api_key, not (args.all or args.rim), args.max_ratio, rim_mode=args.rim)
    if not items:
        console.print("[red]No shots for the reel.[/]")
        return 1
    misses = [i for i in items if not i["shot"].made]
    makes = [i for i in items if i["shot"].made]
    if args.rim:
        rimmed = [i for i in makes if i["rim"]["outcome"] != "swish"]
        clean = [i for i in makes if i["rim"]["outcome"] == "swish"]
        groups = [(misses, "miss", MISS), (rimmed, "rim, then in", RIM_IN),
                  (clean, "swish", MAKE)]
        for i in misses:
            i["head"] = f"MISS · {i['rim']['outcome'].upper()}"
        for group, title, color in groups[1:]:
            for i in group:
                i["head"] = title.upper()
        for group, _, color in groups:
            for i in group:
                i["color"] = color
    else:
        groups = [(misses, "miss", MISS), (makes, "make", MAKE)]

    first = items[0]["clip"]["info"]
    W, H = first.width, first.height
    fps = 30.0
    fonts = Fonts(H)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    raw = args.out.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    t0 = time.perf_counter()
    for group, title, color in groups:
        if not group:
            continue
        for k, item in enumerate(group, 1):
            c = item["clip"]
            info = c["info"]
            rf = item["m"]["release_frame"]
            f0 = max(0, rf - int(args.before * info.fps))
            f1 = min(info.n_frames, rf + int(args.after * info.fps))
            item["span"] = (f0, f1)
            cap = cv2.VideoCapture(str(c["mp4"]))
            cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
            for f in range(f0, f1):
                ok, img = cap.read()
                if not ok:
                    break
                if img.shape[1] != W or img.shape[0] != H:
                    img = cv2.resize(img, (W, H))
                writer.write(draw_frame(img, f, item, k, len(group), fonts))
            cap.release()
            console.print(f"  {title} {k}/{len(group)}: {item['label']}, shot {item['shot'].number}")
    writer.release()
    video.encode_h264(raw, args.out, crf=cfg.OUTPUT_CRF)
    raw.unlink(missing_ok=True)
    console.print(f"\n  -> [bold]{main.rel(args.out)}[/] "
                  f"({args.out.stat().st_size / 1e6:.1f} MB, {time.perf_counter() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main_())
