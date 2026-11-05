"""A video for social: a session's shots, with the pose and where the ball met the rim.

    python social.py                    # each clip under data/input/, one video each
    python social.py data/input/session # just this clip
    python social.py --side-only        # only shots seen from the side
    python social.py --uncut            # the clip as is, not trimmed between shots
    python social.py --height 1920      # taller (the clip and the panel each take half)
    python social.py data/input/a data/input/b --pick 4,2,2
                                        # one video mixing clips: 4 swishes, 2 makes with rim, 2 misses
    python social.py --pick 4,2,2 --end-on-swish
                                        # ...with the last swish moved to the end

Top half: the clip's shots back to back, with the pose, the ball's trail, and the rim lit
in the shot's color; where the ball first touched the rim (rim.py's read) is ringed.

Bottom half, filling in shot by shot: on the left, the hoop straight on with each ball path
and where it met the rim; on the right, FG and FG%, swish / make with rim / miss as bars, and
how far off center each shot was.

Rim contact can only be read on shots seen from the side. The others still count in FG and
the bars but get no mark; --side-only drops them, and --pick always does. A --pick video is
a compilation, so it shows how the shots ended but no FG or FG%.

Uses the same cache as main.py, so nothing is called again. Writes
data/output/<timestamp>-swish-detection/<clip>_swish_detection.mp4 (one folder a run).
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
from src.panel import BG, FG, MAKE, MISS, MUTED, RIM as RIM_COLOR, BALL, _rgb, credit_lines
from src.render import ClipOverlay
from src.text import resolve_font

console = main.console

RIM_IN = "#2f9e44"       # a make that touched the rim first: still a make, so a deeper green
BEFORE_S, AFTER_S = 1.2, 2.6   # each shot: from before the release to the ball on the floor
TRAJ_U, TRAJ_TOP = 1.3, 1.1    # the view around the rim the hoop drawing is framed on (rim widths);
                               # a path runs on past it, cut off only at the drawing's edges
TRAJ_SMOOTH = 15               # frames in the smoothing window (a local quadratic, ~0.5 s); in
                               # the air a ball's path IS a quadratic in time, so a long window costs
                               # the arc nothing and only rounds a bounce off the rim a little
TRAJ_STRAY = 0.12              # a detection this far off the smoothed path (rim widths) is dropped
TRAJ_WIDTH = 4                 # px at 1280 wide, every path alike
TRAJ_SS = 3                    # paths drawn this many times larger, then shrunk: antialiased
TRAJ_SLOPE_PTS = 3             # tracked points each side of a gap that set its ends' speed


def _end_speed(f: np.ndarray, x: np.ndarray) -> float:
    """Speed per frame at the end of a few points: a straight-line fit's slope."""
    return float(np.polyfit(f, x, 1)[0]) if len(f) >= 2 else 0.0


def _fill(run):
    """Every frame from a run's first to its last: a lost stretch filled with the curve that
    leaves the last point seen, and meets the next, at the ball's own position and speed
    (a cubic Hermite), so the line flows through it instead of kinking."""
    f = np.array([q[0] for q in run], float)
    xy = np.array([[q[1], q[2]] for q in run], float)
    ff = np.arange(f[0], f[-1] + 1)
    out = np.empty((len(ff), 2))
    k = TRAJ_SLOPE_PTS
    for i in range(len(f) - 1):
        a, b = int(f[i] - f[0]), int(f[i + 1] - f[0])
        out[a] = xy[i]
        n = b - a
        if n == 1:
            continue
        lo, hi = slice(max(0, i - k + 1), i + 1), slice(i + 1, i + 1 + k)
        m0 = np.array([_end_speed(f[lo], xy[lo, d]) for d in (0, 1)]) * n
        m1 = np.array([_end_speed(f[hi], xy[hi, d]) for d in (0, 1)]) * n
        t = (np.arange(1, n) / n)[:, None]
        h00, h10 = 2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t
        h01, h11 = -2 * t**3 + 3 * t**2, t**3 - t**2
        out[a + 1:b] = h00 * xy[i] + h10 * m0 + h01 * xy[i + 1] + h11 * m1
    out[-1] = xy[-1]
    return ff, out[:, 0], out[:, 1]


def smooth_path(traj, corners=(), window=TRAJ_SMOOTH):
    """The tracked path with SAM's frame-to-frame jitter taken out. A real ball's flight is
    smooth, so each frame is replaced by a local quadratic fit (Savitzky-Golay): in the air the
    path IS a quadratic in time, so the fit keeps the arc. Frames where the ball was lost
    (hidden by the rim or net, or just missed) are filled in first, however many.

    *corners*, each ``(frame, u, v)``: where the ball hit the rim (rim.py's read), every bounce
    of a rattle. A bounce is a corner, not a curve, and smoothing across it would round it off:
    turn the ball early, short of the rim, and melt two quick bounces into one. So each flight
    between two bounces is smoothed on its own, and pinned to the bounces at its ends."""
    if len(traj) < 2:
        return traj
    corners = sorted({int(c[0]): (int(c[0]), float(c[1]), float(c[2])) for c in corners}.values())
    cuts = [None] + corners + [None]
    out = []
    for a, b in zip(cuts, cuts[1:]):
        lo, hi = (a[0] if a else -np.inf), (b[0] if b else np.inf)
        part = ([a] if a else []) + [q for q in traj if lo < q[0] < hi] + ([b] if b else [])
        if len(part) < 2:
            continue
        ff, uu, vv = _smoothed(part, [p for p in (a, b) if p], window)
        if out and a and out[-1][0] == ff[0]:   # this corner already ends the part before
            ff, uu, vv = ff[1:], uu[1:], vv[1:]
        out += [(int(x), float(y), float(z)) for x, y, z in zip(ff, uu, vv)]
    return out or traj


def _smoothed(run, pins, window):
    """``(frames, u, v)`` for a run, gaps filled and smoothed. Stray detections (a hand, the net)
    pull a kink into the line: what sits far off a first smoothing is dropped and the hole filled
    like any lost frame. *pins*, points of the run (its ends, at a bounce), are never dropped,
    and the smoothed line is eased back onto each so it passes through exactly."""
    from scipy.signal import savgol_filter

    def smooth(points):
        ff, uu, vv = _fill(points)
        w = min(window, len(ff) if len(ff) % 2 else len(ff) - 1)
        if w >= 5:
            uu, vv = savgol_filter(uu, w, 2, mode="interp"), savgol_filter(vv, w, 2, mode="interp")
        return ff, uu, vv

    ff, uu, vv = smooth(run)
    at = {int(f): i for i, f in enumerate(ff)}
    kept = [q for q in run if any(q is p for p in pins)
            or np.hypot(q[1] - uu[at[q[0]]], q[2] - vv[at[q[0]]]) <= TRAJ_STRAY]
    if len(kept) >= 2 and len(kept) < len(run):
        ff, uu, vv = smooth(kept)
    # Each pin's correction fades out before it reaches the run's other end, so a short flight
    # between two bounces is pinned to both.
    reach = max(1, min(window // 2 + 1, len(ff) - 1))   # frames a correction fades out over
    du, dv = np.zeros(len(ff)), np.zeros(len(ff))
    for p in pins:
        i = int(p[0] - ff[0])
        ease = 0.5 + 0.5 * np.cos(np.pi * np.clip(np.abs(np.arange(len(ff)) - i) / reach, 0, 1))
        du += (p[1] - uu[i]) * ease
        dv += (p[2] - vv[i]) * ease
    return ff, uu + du, vv + dv


def rgb(c, a=255):
    return _rgb(c) + (a,)


class Faces:
    def __init__(self, s: float):
        found = resolve_font("auto")

        def face(px):
            px = max(9, int(px * s))
            return ImageFont.truetype(found[0], px, index=found[1]) if found else ImageFont.load_default()
        self.cache, self.face = {}, face

    def __call__(self, px):
        if px not in self.cache:
            self.cache[px] = self.face(px)
        return self.cache[px]


def pill(d, xy, text, font, fill, fg=(255, 255, 255, 255), pad=10, anchor="la"):
    x, y = xy
    b = d.textbbox((0, 0), text, font=font)
    w, h = b[2] - b[0] + 2 * pad, b[3] - b[1] + 2 * pad
    if anchor[0] == "r":
        x -= w
    d.rounded_rectangle([x, y, x + w, y + h], radius=h // 3, fill=fill)
    d.text((x + pad - b[0], y + pad - b[1]), text, font=font, fill=fg)
    return w, h


# ── what each shot is ────────────────────────────────────────────────────────

def miss_depth(r) -> str:
    """A miss, as the shooter sees it: SHORT (front rim, or came down before it), LONG (back rim,
    or past it), or "" (the side of the rim, or not seen from the side: can't say)."""
    if r is None or r["view"] != "across":
        return ""
    return {"front rim": "SHORT", "short, no rim": "SHORT",
            "back rim": "LONG", "long, no rim": "LONG"}.get(r["outcome"], "")


def outcome(shot, r) -> tuple[str, str]:
    """``(label, color)`` for a decided shot: SWISH, MAKE WITH RIM, or MISS, short or long."""
    if not shot.made:
        depth = miss_depth(r)
        return "MISS" + (f" · {depth}" if depth else ""), MISS
    if r is not None and r["view"] == "across":
        return ("SWISH", MAKE) if r["outcome"] == "swish" else ("MAKE WITH RIM", RIM_IN)
    return "MAKE", MAKE


# ── the bottom half ──────────────────────────────────────────────────────────

class RimPanel:
    def __init__(self, items, hoop, W: int, H: int, left_out: int = 0, board=None,
                 show_fg: bool = True):
        self.items, self.W, self.H, self.left_out = items, W, H, left_out
        self.show_fg = show_fg   # off for a compilation, whose FG% isn't a session's
        self.board = board if board is not None else rim.default_board()
        s = self.s = W / 1280
        self.f = Faces(s)
        self.rim_h = (hoop.cy - hoop.top) / hoop.width
        self.net = (hoop.net_bottom - hoop.cy) / hoop.width
        # The hoop on the left two thirds, whole, as the camera sees it: rim widths to pixels,
        # as big as fits across and down (the board's top, or the highest path, to below the net).
        span_u = max(float(np.abs(self.board[:, 0]).max()), TRAJ_U)
        top_v = min(float(self.board[:, 1].min()), -TRAJ_TOP)
        bottom_v = self.net + 0.3
        top_px, bottom_px = 96 * s, 24 * s
        self.k = min((W * 2 / 3 - 50 * s) / (2 * span_u),
                     (H - top_px - bottom_px) / (bottom_v - top_v))
        self.c = (int(W / 3), int(top_px - top_v * self.k))
        self.x_right = W * 2 / 3 + 20 * s     # the bars in the right third
        base = Image.new("RGB", (W, H), _rgb(BG))
        d = ImageDraw.Draw(base)
        self._static(d)
        self.base = base
        self._heat_key, self._heat = None, None

    def P(self, u, v):
        return self.c[0] + u * self.k, self.c[1] + v * self.k

    def _static(self, d):
        s, f = self.s, self.f
        mut = _rgb(MUTED)
        # Left two thirds: the whole hoop straight on, at the rim's height: the backboard as traced
        # from the clip, the pole under it, the rim edge-on as a flat line, the net's outline
        # hanging straight down. Left and right as in the video; nothing claims front or back.
        d.polygon([self.P(u, v) for u, v in self.board], fill=(40, 42, 46), outline=(92, 95, 101),
                  width=max(2, int(3 * s)))
        foot = float(self.board[:, 1].max())
        px0, py0 = self.P(-0.07, foot)
        px1, _ = self.P(0.07, foot)
        d.rectangle([px0, py0, px1, self.H], fill=(52, 54, 58))
        net = self.net
        outline = [(-0.5, 0), (-0.38, net), (0.38, net), (0.5, 0)]
        for a, b in zip(outline, outline[1:]):
            self._dashed(d, self.P(*a), self.P(*b), (150, 152, 158), max(1, int(2 * s)))
        d.text((40 * s, 36 * s), "SHOT PATHWAY", font=f(32), fill=mut)
        # Bottom-right: the credit (config.py's CREDIT, or --li / --x / --ig), one handle a line.
        y = self.H - 32 * s
        for label, value in reversed(credit_lines(cfg)):
            vx = self.W - 40 * s
            d.text((vx, y), value, font=f(26), fill=_rgb(FG), anchor="rs")
            if label:   # the same size as the handle, only muted
                vw = d.textlength(value, font=f(26))
                d.text((vx - vw - 10 * s, y), label, font=f(26), fill=mut, anchor="rs")
            y -= 38 * s
        # Right third: the headline labels.
        if self.show_fg:
            d.text((self.x_right, 36 * s), "FIELD GOALS", font=f(24), fill=mut)
            d.text((self.W - 40 * s, 36 * s), "FG%", font=f(24), fill=mut, anchor="ra")
            d.text((self.x_right, 262 * s), "SHOT OUTCOME", font=f(24), fill=mut)
        else:
            d.text((self.x_right, 36 * s), "SHOT OUTCOME", font=f(32), fill=mut)

    @staticmethod
    def _dashed(d, a, b, fill, width, dash=10, gap=7):
        (x0, y0), (x1, y1) = a, b
        n = np.hypot(x1 - x0, y1 - y0)
        t = 0.0
        while t < n:
            e = min(t + dash, n)
            d.line([(x0 + (x1 - x0) * t / n, y0 + (y1 - y0) * t / n),
                    (x0 + (x1 - x0) * e / n, y0 + (y1 - y0) * e / n)], fill=fill, width=width)
            t = e + gap

    def render(self, state: dict) -> np.ndarray:
        """*state*: per item index, which parts are shown: 'decided' and 'mark'."""
        s, f = self.s, self.f
        img = self.base.copy().convert("RGBA")
        decided = [it for j, it in enumerate(self.items) if state.get(j, {}).get("decided")]
        marked = [(j, it) for j, it in enumerate(self.items) if state.get(j, {}).get("mark")]

        # Marks on the rim's line, by left-to-right position as in the video: a touch where
        # the ball met the rim, a swish where it came through.
        spots = {}
        for j, it in marked:
            r = it["rim"]
            u = (r["u"] if r["frame"] is not None else r["cross_u"]) * r["mirror"]
            if np.isfinite(u) and (r["frame"] is not None or it["shot"].made):
                spots[j] = (u, 0.0)
        latest = max(spots, default=None)   # the most recent shot's mark gets a ring
        d = ImageDraw.Draw(img)
        # The ball's paths, smoothed: earlier shots faint in their color, the current one growing
        # frame by frame (in the ball's orange until it's called). Lost frames are filled in, so
        # the line never breaks; it runs over the board's edge and is cut off only at the drawing's.
        cur, frame = state.get("current"), state.get("frame")
        # Drawn TRAJ_SS times larger and shrunk, so the lines are antialiased and even, and only
        # inside the hoop drawing's edges: under the title, short of the bars.
        x1, y0 = int(self.x_right - 30 * s), int(78 * s)
        ss = TRAJ_SS
        paths = Image.new("RGBA", (x1 * ss, (self.H - y0) * ss), (0, 0, 0, 0))
        d = ImageDraw.Draw(paths)

        def Q(u, v):
            x, y = self.P(u, v)
            return x * ss, (y - y0) * ss

        width = max(2, round(TRAJ_WIDTH * s * ss))
        for j, it in enumerate(self.items):
            if cur is None or j > cur or not it["traj"]:
                continue
            pts = it["traj"] if j < cur else [q for q in it["traj"] if q[0] <= frame]
            if not pts:
                continue
            done = j < cur or state.get(j, {}).get("decided")
            color = {"swish": MAKE, "rim": RIM_IN}.get(it["kind"], MISS if not it["shot"].made else MAKE)
            if j < cur:   # faint: mixed into the board's color (drawing here doesn't blend)
                fill = tuple(int(0.45 * x + 0.55 * y) for x, y in zip(_rgb(color), (40, 42, 46))) + (255,)
            else:
                fill = rgb(color if done else BALL, 255)
            # One polyline, round-jointed and round-capped, so its width holds through every bend.
            line = [Q(u, v) for _, u, v in pts]
            if len(line) >= 2:
                d.line(line, fill=fill, width=width, joint="curve")
            for end in (line[0], line[-1]):
                d.ellipse([end[0] - width / 2, end[1] - width / 2,
                           end[0] + width / 2, end[1] + width / 2], fill=fill)
            if j == cur and pts[-1][0] == frame:   # the ball itself, this frame
                bx, by = Q(pts[-1][1], pts[-1][2])
                br = rim.BALL_R * self.k * ss
                d.ellipse([bx - br, by - br, bx + br, by + br], outline=rgb(BALL, 255),
                          width=max(ss, round(2 * s * ss)))   # to scale: a ball is half the rim's width
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        layer.paste(paths.resize((x1, self.H - y0), Image.LANCZOS), (0, y0))
        img = Image.alpha_composite(img, layer)
        d = ImageDraw.Draw(img)

        # The rim edge-on, a flat line.
        lx0, ly = self.P(-0.5, 0)
        lx1, _ = self.P(0.5, 0)
        d.line([(lx0, ly), (lx1, ly)], fill=rgb(RIM_COLOR), width=max(4, int(9 * s)))

        # The marks: a swish where it came through, a touch where it met the rim.
        for j, it in marked:
            r, shot = it["rim"], it["shot"]
            if j not in spots:
                continue
            px, py = self.P(*spots[j])
            if r["frame"] is not None:
                color = (MISS if not shot.made else RIM_IN)
                rr = 12 * s
                if shot.made:
                    d.ellipse([px - rr, py - rr, px + rr, py + rr], fill=rgb(color),
                              outline=rgb(BG), width=max(1, int(2 * s)))
                else:
                    w = max(3, int(6 * s))
                    d.line([(px - rr, py - rr), (px + rr, py + rr)], fill=rgb(color), width=w)
                    d.line([(px - rr, py + rr), (px + rr, py - rr)], fill=rgb(color), width=w)
            elif shot.made:
                rr = 15 * s
                d.ellipse([px - rr, py - rr, px + rr, py + rr], outline=rgb(MAKE, 230),
                          width=max(2, int(4 * s)))
            else:
                continue
            if j == latest:
                rr = 24 * s
                d.ellipse([px - rr, py - rr, px + rr, py + rr], outline=(255, 255, 255, 200),
                          width=max(2, int(3 * s)))

        # Right: the headline (field goals and FG%), then the bars.
        xr = self.x_right
        right = self.W - 40 * s                 # the right third's usable right edge
        made = sum(1 for it in decided if it["shot"].made)
        n = len(decided)
        if self.show_fg:
            frac, pct = (f"{made}/{n}", f"{made / n:.0%}") if n else ("0/0", "—")
            size = 92
            while size > 40 and (d.textlength(frac, font=f(size)) + d.textlength(pct, font=f(size))
                                 + 24 * s > right - xr):   # shrink until the two don't touch
                size -= 4
            d.text((xr, 70 * s), frac, font=f(size), fill=_rgb(FG))
            d.text((right, 70 * s), pct, font=f(size), fill=_rgb(FG), anchor="ra")

        rows = [("SWISH", MAKE, lambda it: it["kind"] == "swish"),
                ("MAKE WITH RIM", RIM_IN, lambda it: it["kind"] == "rim"),
                ("MISS", MISS, lambda it: it["kind"] == "miss")]
        if any(it["kind"] == "make" for it in self.items):
            rows.insert(2, ("MAKE, RIM NOT SEEN", MUTED, lambda it: it["kind"] == "make"))
        total = max(1, len(self.items))
        if self.show_fg:
            top, gap, bh = 312 * s, (84 if len(rows) <= 3 else 68) * s, 30 * s
            label_px, count_px, count_w = 19, 28, 50 * s
            # Keep the rows, and the note under them, clear of the credit: squeeze the gaps.
            n_credit = len(credit_lines(cfg))
            if n_credit and len(rows) > 1:
                ends = top + (len(rows) - 1) * gap + (label_px + 12) * s + bh + 16 * s + 24 * s
                room = self.H - (32 + 38 * n_credit + 10) * s
                if ends > room:
                    gap -= (ends - room) / (len(rows) - 1)
        else:
            # No headline: big bars, close together, the block centered in the column between
            # the title and the credit.
            label_px, count_px, count_w = 28, 40, 64 * s
            bh = (76 if len(rows) <= 3 else 56) * s
            row_h = (label_px + 12) * s + bh
            gap = row_h + 34 * s
            y_top = 100 * s                       # under the title
            n_credit = len(credit_lines(cfg))
            y_bottom = self.H - (32 + 38 * n_credit + 10) * s if n_credit else self.H - 30 * s
            block = (len(rows) - 1) * gap + row_h
            top = y_top + max(0.0, (y_bottom - y_top - block) / 2)
        bar_w = right - xr - count_w
        for i, (label, color, pick) in enumerate(rows):
            y = top + i * gap
            cnt = sum(1 for it in decided if pick(it))
            d.text((xr, y), label, font=f(label_px), fill=_rgb(MUTED))
            yb = y + (label_px + 12) * s
            d.rounded_rectangle([xr, yb, xr + bar_w, yb + bh], radius=int(bh / 6),
                                fill=(36, 38, 42, 255))
            if cnt:
                w = bar_w * cnt / total
                d.rounded_rectangle([xr, yb, xr + w, yb + bh], radius=int(bh / 6),
                                    fill=rgb(color))
            d.text((right, yb + bh / 2), str(cnt), font=f(count_px), fill=_rgb(FG), anchor="rm")

        hidden = sum(1 for it in self.items if it["rim"]["view"] != "across")
        note = (f"{hidden} facing camera: counted, not marked"
                if hidden else
                f"shots seen from the side; {self.left_out} toward the camera left out"
                if self.left_out else "")
        if note:
            import textwrap
            y = top + (len(rows) - 1) * gap + (label_px + 12) * s + bh + 16 * s
            for line in textwrap.wrap(note, 38):
                d.text((xr, y), line, font=f(17), fill=_rgb(MUTED))
                y += 24 * s
        return np.asarray(img.convert("RGB"))[..., ::-1].copy()


# ── the top half ─────────────────────────────────────────────────────────────

def draw_top(frame: np.ndarray, f: int, it: dict, c: dict, sx: float, faces: Faces, k: int,
             n: int, fps: float) -> np.ndarray:
    shot, r, hoop = it["shot"], it["rim"], c["hoop"]
    # The clip drawn exactly as the shot chart video draws it: pose, rim, ball and comet,
    # rim lit on the verdict. No hot streak: a compilation's shots aren't in a row. Only the
    # shot count, the outcome and the touch ring below are this video's.
    if c.get("overlay") is None or c["overlay"].sx != sx:
        c["overlay"] = ClipOverlay(kp=c["kp"], hoop=hoop, track=c["track"], shots=c["shots"],
                                   n_frames=c["info"].n_frames, fps=fps, sx=sx,
                                   width=frame.shape[1], cfg=cfg, hot=False)
    c["overlay"].draw(frame, f)
    decided = shot.decided is not None and f >= shot.decided
    label, color = outcome(shot, r)

    img = Image.fromarray(frame[..., ::-1]).convert("RGBA")
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    sc = img.size[0] / 1280

    # Where it first met the rim, from the moment it does.
    if r["view"] == "across" and r["frame"] is not None and f >= r["frame"]:
        x, y = r["px"][0] * sx, r["px"][1] * sx
        br = hoop.width * sx * rim.BALL_R * 1.3
        tc = MISS if not shot.made else RIM_IN
        d.ellipse([x - br, y - br, x + br, y + br], outline=rgb(tc), width=max(3, int(4 * sc)))

    # Top-left: which shot. Top-right, once decided: how it ended.
    m = int(26 * sc)
    pill(d, (m, m), f"SHOT {k} OF {n}", faces(34), (0, 0, 0, 170), pad=int(14 * sc))
    if decided:
        pill(d, (img.size[0] - m, m), label, faces(38), rgb(color, 235),
             fg=_rgb(BG) + (255,), pad=int(15 * sc), anchor="ra")
    return np.asarray(Image.alpha_composite(img, layer).convert("RGB"))[..., ::-1].copy()

# ── main ─────────────────────────────────────────────────────────────────────

def collect(clip: Path, api_key: str, side_only: bool) -> tuple[dict, list[dict], int]:
    """A clip's shots, each with what the video needs; and how many were left out."""
    c = hist.load_clip(clip, api_key)
    info, hoop, fps = c["info"], c["hoop"], c["info"].fps
    c["clip"] = clip

    items, left_out = [], 0
    for shot in c["shots"]:
        if shot.made is None:
            continue
        r = rim.read_shot(shot, hoop, fps)
        if side_only and r["view"] != "across":
            left_out += 1   # toward or away from the camera: its rim can't be read
            continue
        m = form.measure(c, shot, float("inf"))
        kind = ("miss" if not shot.made else
                "make" if r["view"] != "across" else
                "swish" if r["outcome"] == "swish" else "rim")
        rf = m["release_frame"]
        f0, f1 = max(0, rf - int(BEFORE_S * fps)), min(info.n_frames, rf + int(AFTER_S * fps))
        if shot.decided is not None:   # a long rattle: run on until it's called, and a beat after
            f1 = min(info.n_frames, max(f1, shot.decided + int(0.6 * fps)))
        # When it shows below: the verdict when decided, the mark when it touches or goes through.
        decided = min(shot.decided if shot.decided is not None else f1 - 1, f1 - 1)
        mark = (r["frame"] if r["view"] == "across" and r["frame"] is not None else
                int(shot.crossing) if shot.crossing is not None else decided)
        x, y = shot.court_xy if shot.court_xy is not None else (1.0, 0.0)
        # The ball's path around the hoop, as the camera sees it, in rim widths from the rim's
        # center: what the hoop drawing shows. Every detection, over the board or not, smoothed,
        # with lost frames filled in, so it never breaks. Each bounce off the rim is a sharp
        # corner where rim.py says the ball hit, so the ball meets the rim on that frame.
        touches = []
        if r["view"] == "across" and r["frame"] is not None:
            touches = [(fr, (tx - hoop.cx) / hoop.width, (ty - hoop.cy) / hoop.width)
                       for fr, tx, ty in r["touches"]]
        traj = smooth_path([(int(pf), (px - hoop.cx) / hoop.width, (py - hoop.cy) / hoop.width)
                            for pf, px, py in shot.path], corners=touches)
        items.append({"shot": shot, "rim": r, "kind": kind, "span": (f0, f1), "c": c, "traj": traj,
                      "decided_at": decided, "mark_at": min(mark, f1 - 1),
                      # how far the shot line is off the picture's plane: lower, surer rim read
                      "angle": float(np.degrees(np.arctan2(abs(y), abs(x))))})
    # Windows can't overlap: a shot's clip ends where the next one's starts.
    for a, b in zip(items, items[1:]):
        if a["span"][1] > b["span"][0]:
            a["span"] = (a["span"][0], b["span"][0])
    return c, items, left_out


def pick(items: list[dict], counts: dict[str, int], end_on_swish: bool = False) -> list[dict]:
    """*counts* of each kind (swish, rim, miss), the shots taken most squarely from the side
    first (their rim is read surest). They play in the order taken; with *end_on_swish* the
    last swish moves to the end, so the video finishes on one."""
    chosen = []
    for kind, n in counts.items():
        mine = [it for it in items if it["kind"] == kind]
        if len(mine) < n:
            console.print(f"  [yellow]only {len(mine)} {kind} shot(s), asked for {n}[/]")
        chosen += sorted(mine, key=lambda it: it["angle"])[:n]
    out = [it for it in items if any(it is c for c in chosen)]
    last = next((it for it in reversed(out) if it["kind"] == "swish"), None)
    if end_on_swish and last is not None:
        out.remove(last)
        out.append(last)
    return out


def render(items: list[dict], dst: Path, total_h: int, left_out: int = 0,
           continuous: bool = False, show_fg: bool = True) -> Path:
    """One video, the panel filling in below: each item's shot back to back, or with
    *continuous* (one clip's items) the clip itself, uncut, from the first shot to the last."""
    first = items[0]["c"]
    info, fps = first["info"], first["info"].fps
    video_h = total_h // 2 // 2 * 2
    W = int(round(info.width * video_h / info.height / 2)) * 2
    board = rim.board_outline(first)
    panel = RimPanel(items, first["hoop"], W, total_h - video_h, left_out=left_out, board=board,
                     show_fg=show_fg)
    faces = Faces(W / 1280)
    faces.s = W / 1280

    dst.parent.mkdir(parents=True, exist_ok=True)
    raw = dst.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, total_h))
    # (frame, item index) in playing order. Continuous: every frame from the first shot's
    # start to the last's end, each belonging to the shot most recently started.
    if continuous:
        starts = [it["span"][0] for it in items]
        seq = [(f, int(np.searchsorted(starts, f, side="right")) - 1)
               for f in range(starts[0], items[-1]["span"][1])]
    else:
        seq = [(f, i) for i, it in enumerate(items) for f in range(*it["span"])]
    caps, at = {}, {}
    for n, (f, i) in enumerate(seq):
        it = items[i]
        c = it["c"]
        sx = video_h / c["info"].height
        cap = caps.setdefault(id(c), cv2.VideoCapture(str(c["mp4"])))
        if at.get(id(c)) != f:                 # a cut: seek; otherwise just read on
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, src = cap.read()
        if not ok:
            break
        at[id(c)] = f + 1
        k = i + 1
        top = cv2.resize(src, (W, video_h), interpolation=cv2.INTER_AREA)
        top = draw_top(top, f, it, c, sx, faces, k, len(items), fps)
        state = {"current": i, "frame": f}
        for j, other in enumerate(items):
            if j < i:
                state[j] = {"decided": True, "mark": True}
            elif j == i:
                state[j] = {"decided": f >= other["decided_at"], "mark": f >= other["mark_at"]}
        writer.write(np.vstack([top, panel.render(state)]))
        if n + 1 == len(seq) or seq[n + 1][1] != i:
            label, _ = outcome(it["shot"], it["rim"])
            console.print(f"  shot {k}/{len(items)} ({hist.rel(c['clip'])} #{it['shot'].number}): "
                          f"{label.lower()}")
    for cap in caps.values():
        cap.release()
    writer.release()
    video.encode_h264(raw, dst, crf=cfg.OUTPUT_CRF)
    raw.unlink(missing_ok=True)
    return dst


def clip_name(clip: Path) -> str:
    return clip.stem if clip.parent == cfg.INPUT else f"{clip.parent.name}_{clip.stem}"


def main_(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("inputs", nargs="*", type=Path, metavar="CLIP|DIR",
                        help=f"clips or folders (searched with subfolders); default {main.rel(cfg.INPUT)}")
    parser.add_argument("--side-only", action="store_true",
                        help="only shots seen from the side, whose rim can be read; by default every "
                             "shot, so FG is the session's")
    parser.add_argument("--uncut", action="store_true",
                        help="play the clip as is, from its first shot to its last; by default it's "
                             "trimmed to the shots, back to back")
    parser.add_argument("--pick", metavar="SWISH,RIM,MISS",
                        help="one video from all the clips: this many swishes, makes with rim and "
                             "misses, the shots taken most squarely from the side first, in the order taken "
                             "(e.g. 4,2,2)")
    parser.add_argument("--end-on-swish", action="store_true",
                        help="with --pick: move the last swish to the end, so the video finishes on one")
    parser.add_argument("--height", type=int, default=1440,
                        help="the video's total height; the clip and the panel each take half (default 1440)")
    parser.add_argument("--cached-only", action="store_true",
                        help="never call the gateway; skip a clip whose replies aren't cached")
    parser.add_argument("--out", type=Path, default=None,
                        help="the folder to write to (default: a new data/output/<timestamp>-swish-detection/)")
    credit = parser.add_argument_group(
        "credit", "the handles in the panel's bottom-right corner; any of these replaces config.py's CREDIT")
    credit.add_argument("--li", metavar="TEXT", help='LinkedIn, e.g. "Jeremy Park, PhD"')
    credit.add_argument("--x", metavar="HANDLE", help="X handle, e.g. @jeremyparkphd")
    credit.add_argument("--ig", metavar="HANDLE", help="Instagram handle")
    credit.add_argument("--no-credit", action="store_true", help="no credit at all")
    args = parser.parse_args(argv)
    handles = [(label, value if label == "LI" or value.startswith("@") else f"@{value}")
               for label, value in (("LI", args.li), ("X", args.x), ("IG", args.ig)) if value]
    if args.no_credit:
        cfg.CREDIT = []
    elif handles:
        cfg.CREDIT = handles
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
    t0 = time.perf_counter()

    def out_dir() -> Path:   # made on first use, so a run with nothing to write leaves no folder
        if args.out is None:
            args.out = main.new_run_dir("swish-detection")
        return args.out

    loaded = []
    for clip in clips:
        console.rule(f"[bold magenta]{hist.rel(clip)}[/]", align="left")
        try:
            loaded.append(collect(clip, api_key, side_only=args.side_only or bool(args.pick)))
        except hist.NotCached as exc:
            console.print(f"  [yellow]skipped[/]: not cached ({exc}); run without --cached-only")
    if args.pick:
        n = [int(x) for x in args.pick.split(",")]
        items = pick([it for _, its, _ in loaded for it in its],
                     dict(zip(("swish", "rim", "miss"), n + [0] * (3 - len(n)))),
                     end_on_swish=args.end_on_swish)
        if not items:
            console.print("[red]No shots to pick.[/]")
            return 1
        # A compilation: its FG% isn't a session's, so it isn't shown.
        dst = render(items, out_dir() / f"picks_{'_'.join(map(str, n))}_swish_detection.mp4", args.height,
                     show_fg=False)
        console.print(f"  -> [bold]{main.rel(dst)}[/] ({dst.stat().st_size / 1e6:.1f} MB, "
                      f"{time.perf_counter() - t0:.0f}s)")
        return 0
    for c, items, left_out in loaded:
        if not items:
            console.print(f"  [yellow]{hist.rel(c['clip'])}: no shots[/]")
            continue
        dst = render(items, out_dir() / f"{clip_name(c['clip'])}_swish_detection.mp4", args.height, left_out,
                     continuous=args.uncut)
        console.print(f"  -> [bold]{main.rel(dst)}[/] ({dst.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main_())
