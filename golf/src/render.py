"""The export video (overlay, a live timeline of the swing along the bottom), and swing.png."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from .skeleton import FACE_KPTS, SKELETON, edge_part
from .text import resolve_font

FG = "#f2f4f7"
MUTED = "#8b9199"
CLUB = "#2f6bff"        # the club's mask
PATH = "#3ee66b"        # the clubhead's path
BALL = "#ffffff"
TRACER = "#ffffff"      # the projected flight (dashed)
BALL_RING = "#ff8a1f"   # the ring on the ball, on the tee and in flight


def _rgb(color: str) -> tuple[int, int, int]:
    return tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))


def _bgr(color: str) -> tuple[int, int, int]:
    return _rgb(color)[::-1]


def _mix(color: str, toward: str, t: float) -> str:
    """*color* moved a fraction *t* of the way to *toward*."""
    a, b = np.array(_rgb(color)), np.array(_rgb(toward))
    return "#" + "".join(f"{int(round(v)):02x}" for v in a + (b - a) * t)


@lru_cache(maxsize=32)
def _face(path: str, index: int, px: int):
    from PIL import ImageFont

    return ImageFont.truetype(path, px, index=index)


def _text(img: np.ndarray, items, font) -> None:
    """Draw ``(text, (x, y), px, "#rrggbb", anchor)`` items; ``anchor`` is PIL's (e.g. "lm")."""
    if not items:
        return
    if font is None:
        for text, (x, y), px, color, anchor in items:
            fs = px / 34
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
            x -= {"l": 0, "m": tw // 2, "r": tw}[anchor[0]]
            y += {"t": th, "m": th // 2, "b": 0, "s": 0}[anchor[1]]
            cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        _bgr(color), 2, cv2.LINE_AA)
        return
    from PIL import Image, ImageDraw

    pil = Image.fromarray(img[..., ::-1])
    draw = ImageDraw.Draw(pil)
    for text, xy, px, color, anchor in items:
        draw.text(xy, text, font=_face(font[0], font[1], int(px)), fill=_rgb(color), anchor=anchor)
    img[:] = np.array(pil)[..., ::-1]


def _p(v) -> tuple[int, int]:
    return int(round(v[0])), int(round(v[1]))


def _draw_skeleton(frame, pts: np.ndarray, *, colors, thick: int, radius: int) -> None:
    ok = np.isfinite(pts).all(axis=1)
    for a, b, _ in SKELETON:
        if a in FACE_KPTS or b in FACE_KPTS or not (ok[a] and ok[b]):
            continue
        cv2.line(frame, _p(pts[a]), _p(pts[b]), colors[edge_part(a, b)], thick, cv2.LINE_AA)
    for i in range(len(pts)):
        if ok[i] and i not in FACE_KPTS:
            cv2.circle(frame, _p(pts[i]), radius, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, _p(pts[i]), radius + 1, (20, 22, 26), 1, cv2.LINE_AA)


def _blend(frame, layer, alpha_mask: np.ndarray) -> None:
    """``frame = frame * (1 - a) + layer * a``, with *alpha_mask* in [0, 1] per pixel."""
    a = alpha_mask[..., None]
    frame[:] = (frame * (1 - a) + layer * a).astype(np.uint8)


def _draw_club(frame, mask: np.ndarray, *, alpha: float, thick: int) -> None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub, m = frame[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    layer = np.broadcast_to(np.array(_bgr(CLUB), np.float32), sub.shape)
    _blend(sub, layer, m.astype(np.float32) * alpha)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, _bgr(CLUB), max(1, thick // 2), cv2.LINE_AA)


def _draw_ball_mask(frame, x0: int, y0: int, m: np.ndarray, *, thick: int) -> None:
    """The ball's mask: a translucent fill and an outline, in the ring's color."""
    h, w = m.shape
    pad = thick + 2
    X0, Y0 = max(0, x0 - pad), max(0, y0 - pad)
    X1, Y1 = min(frame.shape[1], x0 + w + pad), min(frame.shape[0], y0 + h + pad)
    if X1 <= X0 or Y1 <= Y0:
        return
    sub = frame[Y0:Y1, X0:X1]
    full = np.zeros(sub.shape[:2], np.uint8)
    full[y0 - Y0:y0 - Y0 + h, x0 - X0:x0 - X0 + w] = m.astype(np.uint8) * 255
    layer = np.empty_like(sub)
    layer[:] = _bgr(BALL_RING)
    _blend(sub, layer, full.astype(np.float32) / 255 * 0.5)
    contours, _ = cv2.findContours(full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(sub, contours, -1, _bgr(BALL_RING), thick, cv2.LINE_AA)


def _glow_line(frame, pts: np.ndarray, color: str, thick: int, *, alpha: float = 1.0,
               dashed: bool = False, fade_tail: bool = False, glow: float = 0.9) -> None:
    """A soft-glow polyline (a halo, then the line), optionally dashed or fading at its start.

    Drawn on a crop around the points only: blurring the whole frame per line is slow.
    """
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 2 or alpha <= 0:
        return
    H, W = frame.shape[:2]
    pad = int(thick * 6) + 4
    x0 = int(max(0, np.floor(pts[:, 0].min()) - pad))
    y0 = int(max(0, np.floor(pts[:, 1].min()) - pad))
    x1 = int(min(W, np.ceil(pts[:, 0].max()) + pad))
    y1 = int(min(H, np.ceil(pts[:, 1].max()) + pad))
    if x1 <= x0 or y1 <= y0:
        return
    _glow_line_crop(frame[y0:y1, x0:x1], pts - [x0, y0], color, thick, alpha=alpha,
                    dashed=dashed, fade_tail=fade_tail, glow=glow)


def _glow_line_crop(frame, pts, color, thick, *, alpha, dashed, fade_tail, glow) -> None:
    halo = np.zeros_like(frame)
    core = np.zeros_like(frame)
    weight = np.zeros(frame.shape[:2], np.float32)
    bgr = _bgr(color)
    n = len(pts)
    for i in range(n - 1):
        if dashed and (i // 3) % 2:
            continue
        a = (i + 1) / n if fade_tail else 1.0
        p, q = _p(pts[i]), _p(pts[i + 1])
        cv2.line(halo, p, q, bgr, thick * 3, cv2.LINE_AA)
        cv2.line(core, p, q, (255, 255, 255), max(1, thick // 2), cv2.LINE_AA)
        cv2.line(weight, p, q, a, thick * 3, cv2.LINE_AA)
    halo = cv2.GaussianBlur(halo, (0, 0), thick * 1.2)
    wblur = cv2.GaussianBlur(weight, (0, 0), thick * 1.2)
    out = frame.astype(np.float32) + halo.astype(np.float32) * glow * alpha * np.clip(wblur, 0, 1)[..., None]
    # the colored line itself
    line = np.zeros_like(frame)
    for i in range(n - 1):
        if dashed and (i // 3) % 2:
            continue
        cv2.line(line, _p(pts[i]), _p(pts[i + 1]), bgr, thick, cv2.LINE_AA)
    # Coverage from the anti-aliased intensity, so the soft edge pixels blend in instead
    # of being painted at full strength as a dark fringe; the colour itself is solid.
    solid = np.array(bgr, np.float32)
    lmask = line.max(axis=2).astype(np.float32) / max(1, max(bgr)) * alpha
    if fade_tail:
        lmask *= np.clip(cv2.blur(weight, (3, 3)) / max(1e-6, weight.max()), 0, 1)
    out = out * (1 - lmask[..., None]) + solid * lmask[..., None]
    cmask = core.max(axis=2).astype(np.float32) / 255.0 * 0.6 * alpha
    out = out * (1 - cmask[..., None]) + 255.0 * cmask[..., None]
    frame[:] = np.clip(out, 0, 255).astype(np.uint8)


def _trim_end(pts: np.ndarray, gap: float) -> np.ndarray:
    """``pts`` with its last ``gap`` px of length cut off (empty if it's shorter)."""
    if len(pts) < 2 or gap <= 0:
        return pts
    seg = np.hypot(*np.diff(pts, axis=0).T)
    back = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])   # length from each point to the end
    keep = back >= gap
    if keep.sum() < 1:
        return pts[:0]
    i = int(np.nonzero(keep)[0][-1])            # last point at least gap from the end
    over = back[i] - gap                         # how far past it the cut lies
    cut = pts[i] + (pts[i + 1] - pts[i]) * (over / seg[i]) if seg[i] > 0 else pts[i]
    return np.vstack([pts[:i + 1], cut])


def _ring(frame, center, r: float, color: str, thick: int) -> None:
    cv2.circle(frame, _p(center), int(r), (20, 22, 26), thick + 2, cv2.LINE_AA)
    cv2.circle(frame, _p(center), int(r), _bgr(color), thick, cv2.LINE_AA)


def _rounded_card(frame, x0, y0, x1, y1, radius: int, alpha: float) -> None:
    mask = np.zeros(frame.shape[:2], np.uint8)
    cv2.rectangle(mask, (x0 + radius, y0), (x1 - radius, y1), 255, -1)
    cv2.rectangle(mask, (x0, y0 + radius), (x1, y1 - radius), 255, -1)
    for cx, cy in ((x0 + radius, y0 + radius), (x1 - radius, y0 + radius),
                   (x0 + radius, y1 - radius), (x1 - radius, y1 - radius)):
        cv2.circle(mask, (cx, cy), radius, 255, -1, cv2.LINE_AA)
    _blend(frame, np.zeros_like(frame), mask.astype(np.float32) / 255 * alpha)


def _text_width(text: str, px: float, font) -> int:
    if font is None:
        return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, px / 34, 2)[0][0]
    return int(_face(font[0], font[1], int(px)).getlength(text))


def _pairs(entries) -> list[tuple[str, str]]:
    """``("LABEL", "value")`` pairs or plain strings, blank values skipped."""
    out = []
    for entry in entries or []:
        label, value = (entry[0], entry[1]) if isinstance(entry, (tuple, list)) else ("", entry)
        label, value = str(label or "").strip(), str(value or "").strip()
        if value:
            out.append((label, value))
    return out


class Credit:
    """A strip along the bottom on every frame: the golfer, then the handles, spread evenly.

    Each entry is a muted label then its value, with equal space between entries and at
    both ends. Nothing is drawn when there are no entries.
    """

    HEIGHT = 64     # px at 1080 wide
    GAP = 16        # between the strip and the timeline panel above it

    def __init__(self, width: int, height: int, cfg, font):
        self.s = width / 1080
        self.font = font
        self.left = _pairs([("GOLFER", cfg.CREDIT_GOLFER)] if cfg.CREDIT_GOLFER else [])
        self.right = _pairs(cfg.CREDIT)
        s = self.s
        self.x0, self.x1 = int(36 * s), width - int(36 * s)
        self.y1 = height - int(40 * s)
        self.y0 = self.y1 - int(self.HEIGHT * s)

    @property
    def on(self) -> bool:
        return bool(self.left or self.right)

    @property
    def room(self) -> int:
        """How far the timeline panel moves up to clear the strip."""
        return int((self.HEIGHT + self.GAP) * self.s) if self.on else 0

    def draw(self, frame) -> None:
        if not self.on:
            return
        s = self.s
        _rounded_card(frame, self.x0, self.y0, self.x1, self.y1, int(22 * s), 0.72)
        y = (self.y0 + self.y1) // 2
        label_px = value_px = 28 * s     # the labels as large as the names
        space = int(10 * s)

        entries = self.left + self.right
        widths = [(_text_width(label, label_px, self.font) + space if label else 0)
                  + _text_width(value, value_px, self.font) for label, value in entries]
        gap = (self.x1 - self.x0 - sum(widths)) / (len(entries) + 1)
        items, x = [], self.x0 + gap
        for (label, value), w in zip(entries, widths):
            if label:
                items.append((label, (int(x), y), label_px, MUTED, "lm"))
            items.append((value, (int(x + w - _text_width(value, value_px, self.font)), y),
                          value_px, FG, "lm"))
            x += w + gap
        _text(frame, items, self.font)


class Timeline:
    """A live panel along the bottom: the swing's phase over one continuous timeline, and its tempo.

    Four phases, split at the swing's moments: ADDRESS until takeaway, BACKSWING until the
    top, DOWNSWING until impact, then FOLLOW-THROUGH. The track is split into those four,
    each its own color (dim until played), with impact marked under it; the heading names
    the current phase in its color. A playhead moves along it with the video
    (slowly, through the slow-mo replay). The track runs at one scale from just before
    takeaway to just after the finish, so each phase's length on it is its real duration.
    Under it, the backswing and downswing times count up as they happen and the tempo
    appears at impact.
    """

    HEIGHT = 300    # px at 1080 wide
    PANEL = "#16181c"   # roughly the panel's color over the video, to dim unplayed sections toward
    COLORS = {"ADDRESS": "#facc15", "BACKSWING": "#22d3ee", "DOWNSWING": "#ff922b",
              "FOLLOW-THROUGH": PATH}
    MARGIN_S = 0.5  # the track's span past takeaway (before) and the finish (after)

    def __init__(self, width: int, height: int, swing, n_frames: int, font, bottom: int = 0):
        self.s = width / 1080
        self.swing, self.n, self.font = swing, n_frames, font
        s = self.s
        self.x0, self.x1 = int(36 * s), width - int(36 * s)
        self.y1 = height - int(40 * s) - bottom
        self.y0 = self.y1 - int(self.HEIGHT * s)
        pad = int(40 * s)
        self.lx, self.rx = self.x0 + pad, self.x1 - pad
        sw, last = swing, n_frames - 1
        moments = (sw.takeaway, sw.top, sw.impact, sw.finish)
        if all(np.isfinite(m) for m in moments):
            margin = self.MARGIN_S * sw.fps
            self.t0, self.t1 = max(0.0, sw.takeaway - margin), min(float(last), sw.finish + margin)
        else:                       # a moment missing: the whole clip
            self.t0, self.t1 = 0.0, float(last)
        bounds = [self.t0, sw.takeaway, sw.top, sw.impact, self.t1]
        names = ("ADDRESS", "BACKSWING", "DOWNSWING", "FOLLOW-THROUGH")
        self.segments = [(name, a, b) for name, a, b in zip(names, bounds, bounds[1:])
                         if np.isfinite(a) and np.isfinite(b)]

    def phase(self, f: float) -> str:
        sw = self.swing
        for name, until in (("ADDRESS", sw.takeaway), ("BACKSWING", sw.top), ("DOWNSWING", sw.impact)):
            if not np.isfinite(until) or f < until:
                return name
        return "FOLLOW-THROUGH"

    def _x(self, f: float) -> int:
        u = (f - self.t0) / max(1e-6, self.t1 - self.t0)
        return int(round(self.lx + (self.rx - self.lx) * np.clip(u, 0, 1)))

    def _tempo(self, f: float) -> list[tuple[str, str]]:
        """(label, value) for tempo, backswing and downswing as of frame *f*; "—" until known."""
        sw, fps, dash = self.swing, self.swing.fps, "—"

        def span(a, b):     # seconds from a to min(f, b): counting up, then held
            if not (np.isfinite(a) and np.isfinite(b)) or f < a:
                return dash
            return f"{(min(f, b) - a) / fps:.2f} s"

        tempo = f"{sw.tempo:.1f} : 1" if f >= sw.impact and np.isfinite(sw.tempo) else dash
        return [("TEMPO", tempo), ("BACKSWING", span(sw.takeaway, sw.top)),
                ("DOWNSWING", span(sw.top, sw.impact))]

    def draw(self, frame, f: float, *, slowmo: int | None = None) -> None:
        s, sw = self.s, self.swing
        _rounded_card(frame, self.x0, self.y0, self.x1, self.y1, int(28 * s), 0.72)
        lx, rx, y0 = self.lx, self.rx, self.y0

        # The phase, and where in the clip we are.
        name_y = y0 + int(56 * s)
        clock = f"{f / sw.fps:.2f} s" + (f"  ·  {slowmo}× slow-mo" if slowmo else "")
        now = self.phase(f)
        items = [(now, (lx, name_y), 46 * s, self.COLORS[now], "lm"),
                 (clock, (rx, name_y), 28 * s, MUTED, "rm")]

        # The timeline: one colored section per phase (dim until played), impact marked
        # under it, a playhead. No words on the track: the heading names the phase in its color.
        ty = y0 + int(122 * s)
        h = max(4, int(12 * s))
        head = self._x(f)
        strip = frame[ty - int(24 * s):ty + int(24 * s)]
        cy = int(24 * s)
        gap = max(1, int(3 * s))
        for name, a, b in self.segments:
            color = self.COLORS[name]
            xa, xb = self._x(a) + gap, self._x(b) - gap
            cv2.line(strip, (xa, cy), (xb, cy), _bgr(_mix(color, self.PANEL, 0.7)), h, cv2.LINE_AA)
            if f > a:
                cv2.line(strip, (xa, cy), (max(xa, min(xb, head)), cy), _bgr(color), h, cv2.LINE_AA)
        label_px = 24 * s
        if np.isfinite(sw.impact):    # impact: a tick, labelled under the track
            xi, hit = self._x(sw.impact), f >= sw.impact
            cv2.line(strip, (xi, cy - int(20 * s)), (xi, cy + int(20 * s)),
                     _bgr(FG if hit else MUTED), max(2, int(4 * s)), cv2.LINE_AA)
            items.append(("Impact", (xi, ty + int(30 * s)), label_px, FG if hit else MUTED, "mt"))
        cv2.circle(strip, (head, cy), int(13 * s), (20, 22, 26), -1, cv2.LINE_AA)
        cv2.circle(strip, (head, cy), int(10 * s), _bgr(FG), -1, cv2.LINE_AA)

        # The tempo, live.
        col = (rx - lx) / 3
        for k, (label, value) in enumerate(self._tempo(f)):
            x = int(lx + k * col)
            items += [(label, (x, y0 + int(206 * s)), 24 * s, self.COLORS.get(label, MUTED), "lm"),
                      (value, (x, y0 + int(254 * s)), 46 * s, FG, "lm")]
        _text(frame, items, self.font)


def _catmull_rom(times: np.ndarray, pts: np.ndarray, per_frame: int = 12
                 ) -> tuple[np.ndarray, np.ndarray]:
    """A smooth curve through *pts* (2D or 3D, seen at frame *times*): ``(times, points)``, dense.

    The clubhead moves hundreds of px between frames, so straight lines between them read
    as a jagged polygon. Each point is first pulled a little toward its neighbours (light
    smoothing, ends kept), then a centripetal Catmull-Rom spline goes through them: it
    can't loop or overshoot at the sharp turns (the top, the frame edge).
    """
    if len(pts) < 3:
        return times, pts
    q = pts.copy()
    q[1:-1] = 0.25 * pts[:-2] + 0.5 * pts[1:-1] + 0.25 * pts[2:]
    ext = np.vstack([2 * q[0] - q[1], q, 2 * q[-1] - q[-2]])
    out_t, out_p = [], []
    for i in range(len(q) - 1):
        p0, p1, p2, p3 = ext[i:i + 4]
        k0 = 0.0
        k1 = k0 + max(np.linalg.norm(p1 - p0), 1e-3) ** 0.5
        k2 = k1 + max(np.linalg.norm(p2 - p1), 1e-3) ** 0.5
        k3 = k2 + max(np.linalg.norm(p3 - p2), 1e-3) ** 0.5
        steps = max(2, int(per_frame * (times[i + 1] - times[i])))
        u = np.linspace(0, 1, steps, endpoint=False)[:, None]
        k = k1 + (k2 - k1) * u
        a1 = (k1 - k) / (k1 - k0) * p0 + (k - k0) / (k1 - k0) * p1
        a2 = (k2 - k) / (k2 - k1) * p1 + (k - k1) / (k2 - k1) * p2
        a3 = (k3 - k) / (k3 - k2) * p2 + (k - k2) / (k3 - k2) * p3
        b1 = (k2 - k) / (k2 - k0) * a1 + (k - k0) / (k2 - k0) * a2
        b2 = (k3 - k) / (k3 - k1) * a2 + (k - k1) / (k3 - k1) * a3
        out_p.append((k2 - k) / (k2 - k1) * b1 + (k - k1) / (k2 - k1) * b2)
        out_t.append(times[i] + (times[i + 1] - times[i]) * u[:, 0])
    out_t.append(times[-1:])
    out_p.append(q[-1:])
    return np.concatenate(out_t), np.vstack(out_p)


def _head_path(head: np.ndarray, a: int, b: int, t: float, start: float | None = None
               ) -> np.ndarray:
    """The clubhead's smoothed path over frames ``[a, b]``, from *start* to time *t*.

    The curve is always built over the whole of ``[a, b]`` and then cut, so a trail
    showing only its last stretch keeps the same shape as it moves.
    """
    frames = np.arange(max(0, a), min(b, len(head) - 1) + 1)
    frames = frames[np.isfinite(head[frames]).all(axis=1)]
    if len(frames) < 2:
        return np.zeros((0, 2))
    ts, pts = _catmull_rom(frames.astype(float), head[frames].astype(float))
    lo, hi = (a if start is None else start), min(b, t)
    keep = (ts >= lo) & (ts <= hi)
    out = pts[keep]
    if ts[0] < hi < ts[-1]:   # end exactly at time t, between samples
        out = np.vstack([out, [np.interp(hi, ts, pts[:, 0]), np.interp(hi, ts, pts[:, 1])]])
    return out


def draw_overlay(frame, f: float, *, kp, club, ball, swing, cfg, scale: float) -> None:
    """Everything over the clip at (possibly fractional, in slow-mo) frame *f*."""
    fi = int(round(min(f, len(kp) - 1)))
    thick = max(2, int(cfg.LINE_THICKNESS * scale))
    radius = max(2, int(cfg.POINT_RADIUS * scale))

    # The club, under everything else.
    if club.masks[fi] is not None:
        _draw_club(frame, club.masks[fi], alpha=cfg.CLUB_FILL_ALPHA, thick=thick)

    # The clubhead's path: the whole swing faint, the last moments bright.
    trail = int(cfg.CLUB_TRAIL_SECONDS * swing.fps)
    for a, b, col in ((swing.takeaway, swing.top, PATH), (swing.top, int(np.ceil(swing.impact)), PATH)):
        if f >= a:
            path = _head_path(club.head, a, b, f)
            _glow_line(frame, path, col, max(1, thick // 2), alpha=0.6)
            recent = _head_path(club.head, a, b, f, start=max(a, f - trail))
            if len(recent) >= 2 and f <= b + trail:
                fade = 1.0 if f <= b else max(0.0, 1 - (f - b) / trail)
                _glow_line(frame, recent, col, thick, alpha=fade, fade_tail=True)

    _draw_skeleton(frame, kp[fi], colors=cfg.SKELETON_COLORS, thick=thick, radius=radius)

    # The ball: SAM's mask of it on every frame SAM sees it (on the tee and the first frames
    # of flight), a ring on the tee, then its flight.
    if fi in ball.masks:
        x0, y0, m = ball.masks[fi]
        _draw_ball_mask(frame, x0, y0, m, thick=max(1, thick // 2))
    if ball.rest_until >= 0 and f <= ball.impact and np.isfinite(ball.x[fi]):
        _ring(frame, (ball.x[fi], ball.y[fi]), 0.5 * ball.diameter + 6 * scale, BALL_RING, thick)
    if ball.path is not None and f >= ball.impact:
        path = ball.path.copy()
        if ball.drift is not None:   # glued to the scene as the phone drifts
            path[:, 1:] += ball.drift[fi]
        seen_until = ball.flight[-1] if ball.flight else ball.impact
        upto = path[path[:, 0] <= f]
        solid = upto[upto[:, 0] <= seen_until][:, 1:]
        proj = upto[upto[:, 0] >= seen_until][:, 1:]
        # A ring on the ball where it is now (it's a few px across by the apex). While it's
        # being seen, the trace stops short of the ring so the ball itself shows inside it,
        # untouched, for a human to check the tracking by eye.
        now = [g for g in ball.flight if f - 2 <= g <= f]
        if now and not len(proj):
            solid = _trim_end(solid, cfg.BALL_TRACE_GAP * scale)
        _glow_line(frame, solid, BALL, thick + 1, glow=0.25)
        _glow_line(frame, proj, TRACER, thick, dashed=True, alpha=0.9, glow=0.25)
        if now:
            g = now[-1]
            r = max(9 * scale, 0.5 * ball.d[g] + 6 * scale)
            _ring(frame, (ball.x[g], ball.y[g]), r, BALL_RING, thick)


def _progress(console):
    return Progress(TextColumn("[cyan]rendering[/]"), BarColumn(), TaskProgressColumn(),
                    TextColumn("{task.completed}/{task.total} frames"), TimeElapsedColumn(),
                    console=console, transient=True)


def render(mp4: Path, info, *, kp, club, ball, swing, dst: Path, cfg, console) -> dict:
    """The clip with its overlay, then (if set) a slow-mo replay of the whole swing."""
    frames = []
    cap = cv2.VideoCapture(str(mp4))
    while True:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(img)
    cap.release()
    n = min(len(frames), info.n_frames)
    h, w = frames[0].shape[:2]
    scale = w / 720
    font = resolve_font(cfg.PANEL_FONT, cfg.PANEL_FONT_INDEX)
    credit = Credit(w, h, cfg, font)
    bar = Timeline(w, h, swing, n, font, bottom=credit.room)

    # The schedule: every frame once, then the whole swing again (takeaway to finish), slowed.
    schedule: list[tuple[float, int | None]] = [(float(f), None) for f in range(n)]
    k = int(cfg.SLOWMO_FACTOR)
    if k > 1 and np.isfinite(swing.impact):
        start = swing.takeaway if np.isfinite(swing.takeaway) else swing.impact
        end = swing.finish if np.isfinite(swing.finish) else swing.impact
        a = max(0, int(start - cfg.SLOWMO_BEFORE_SECONDS * info.fps))
        after = cfg.SLOWMO_AFTER_SECONDS
        b = n - 1 if after is None else min(n - 1, int(end + after * info.fps))
        schedule += [(a + i / k, k) for i in range((b - a) * k + 1)]

    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (w, h))
    with _progress(console) as progress:
        task = progress.add_task("render", total=len(schedule))
        for f, slow in schedule:
            img = frames[int(round(min(f, n - 1)))].copy()
            draw_overlay(img, f, kp=kp, club=club, ball=ball, swing=swing, cfg=cfg, scale=scale)
            bar.draw(img, f, slowmo=slow)
            credit.draw(img)
            writer.write(img)
            progress.advance(task)
    writer.release()
    return {"frames_written": len(schedule), "width": w, "height": h,
            "slowmo_factor": k if k > 1 else None}


def plot_swing(mp4: Path, info, *, kp, club, ball, swing, dst: Path, cfg) -> None:
    """One still: the address frame with the whole clubhead path and the ball's flight."""
    cap = cv2.VideoCapture(str(mp4))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, swing.takeaway - 1))
    ok, base = cap.read()
    cap.release()
    if not ok:
        return
    img = base.copy()
    scale = img.shape[1] / 720
    thick = max(2, int(cfg.LINE_THICKNESS * scale))
    end = len(kp) - 1
    _glow_line(img, _head_path(club.head, swing.takeaway, swing.top, end), PATH, thick)
    _glow_line(img, _head_path(club.head, swing.top, int(np.ceil(swing.impact)), end), PATH, thick)
    _draw_skeleton(img, kp[max(0, swing.takeaway - 1)], colors=cfg.SKELETON_COLORS,
                   thick=thick, radius=max(2, int(cfg.POINT_RADIUS * scale)))
    if ball.path is not None:
        seen = ball.flight[-1] if ball.flight else ball.impact
        path = ball.path.copy()
        if ball.drift is not None:
            path[:, 1:] += ball.drift[max(0, swing.takeaway - 1)]
        _glow_line(img, path[path[:, 0] <= seen][:, 1:], BALL, thick + 1, glow=0.25)
        _glow_line(img, path[path[:, 0] >= seen][:, 1:], TRACER, thick, dashed=True, glow=0.25)
    font = resolve_font(cfg.PANEL_FONT, cfg.PANEL_FONT_INDEX)
    Credit(img.shape[1], img.shape[0], cfg, font).draw(img)
    cv2.imwrite(str(dst), img)
