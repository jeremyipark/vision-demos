"""The panel under the video (field goals, FG% and the shot chart), and the hot streak drawn over the video.

The static parts are drawn once; each shot's dot appears when it is decided.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from .text import resolve_font

# The palette shared with the other demos.
BG = "#000000"
FG = "#f2f4f7"
MUTED = "#8b9199"
MAKE = "#51cf66"
MISS = "#ff6b6b"
BALL = "#ff922b"      # the ball and its path
RIM = "#d2622c"       # the rim: a red-brown orange, like the real one
STREAK = "#ff922b"    # the hot streak's count


def _rgb(color: str) -> tuple[int, int, int]:
    """``"#rrggbb"`` -> ``(r, g, b)``."""
    return tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))


def _bgr(color: str) -> tuple[int, int, int]:
    return _rgb(color)[::-1]


@lru_cache(maxsize=32)
def _face(path: str, index: int, px: int):
    from PIL import ImageFont

    return ImageFont.truetype(path, px, index=index)


def _text(img: np.ndarray, items, font) -> np.ndarray:
    """Draw ``(text, (x, y), px, rgb, anchor)`` items; ``anchor`` is PIL's (e.g. "lm", "rm")."""
    if not items:
        return img
    if font is None:
        for text, (x, y), px, rgb, anchor in items:
            fs = px / 34
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
            x -= {"l": 0, "m": tw // 2, "r": tw}[anchor[0]]
            y += {"t": th, "m": th // 2, "b": 0, "s": 0}[anchor[1]]
            cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        rgb[::-1], 2, cv2.LINE_AA)
        return img

    from PIL import Image, ImageDraw

    pil = Image.fromarray(img[..., ::-1])
    draw = ImageDraw.Draw(pil)
    for text, xy, px, rgb, anchor in items:
        draw.text(xy, text, font=_face(font[0], font[1], px), fill=rgb, anchor=anchor)
    img[:] = np.array(pil)[..., ::-1]
    return img


def credit_lines(cfg) -> list[tuple[str, str]]:
    """``cfg.CREDIT`` as ``[(label, value), ...]``, blank entries skipped.

    Each entry is a ``(label, value)`` pair or a plain string.
    """
    lines = []
    for entry in getattr(cfg, "CREDIT", None) or []:
        if isinstance(entry, (tuple, list)):
            label, value = (entry[0], entry[1]) if len(entry) >= 2 else ("", entry[0])
        else:
            label, value = "", entry
        label, value = str(label or "").strip(), str(value or "").strip()
        if value:
            lines.append((label, value))
    return lines


def _draw_credit(panel, font, *, cfg, scale: float, footer: int) -> None:
    """The credit on one line, right-aligned in the footer strip."""
    lines = credit_lines(cfg)
    if not lines or not footer:
        return
    h, w = panel.shape[:2]
    px = max(9, int(cfg.CREDIT_SIZE * scale))
    right = w - int(cfg.CREDIT_MARGIN * scale)
    y = h - footer // 2 + px // 3          # baseline, so the text sits mid-strip
    sep = int(cfg.CREDIT_SEPARATION * scale)
    value_rgb, label_rgb = cfg.CREDIT_COLOR[::-1], cfg.CREDIT_LABEL_COLOR[::-1]

    if font is None:
        for label, value in reversed(lines):
            text = f"{label}: {value}" if label else value
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, px / 34, 1)
            cv2.putText(panel, text, (right - tw, y), cv2.FONT_HERSHEY_SIMPLEX, px / 34,
                        cfg.CREDIT_COLOR, 1, cv2.LINE_AA)
            right -= tw + sep
        return

    from PIL import Image, ImageDraw

    pil = Image.fromarray(panel[..., ::-1])
    draw = ImageDraw.Draw(pil)
    face = _face(font[0], font[1], px)
    for label, value in reversed(lines):
        # Baseline anchors ("rs"), so the label and value sit on one line.
        draw.text((right, y), value, font=face, fill=value_rgb, anchor="rs")
        vw = draw.textlength(value, font=face)
        lw = 0
        if label:
            lw = draw.textlength(f"{label}: ", font=face)
            draw.text((right - vw, y), f"{label}: ", font=face, fill=label_rgb, anchor="rs")
        right -= int(vw + lw + sep)
    panel[:] = np.array(pil)[..., ::-1]


def _spread(points, gap: float):
    """Move dots that would overlap onto a small ring around their spot.

    In court meters, so the dots fall the same way at any export size.
    """
    out = []
    for p in points:
        if p is None:
            out.append(None)
            continue
        placed = [q for q in out if q is not None]
        if not any(np.hypot(p[0] - q[0], p[1] - q[1]) < gap for q in placed):
            out.append(p)
            continue
        for ring in range(1, 6):
            options = [(p[0] + ring * gap * np.cos(a), p[1] + ring * gap * np.sin(a))
                       for a in np.linspace(0, 2 * np.pi, 6 * ring, endpoint=False) + np.pi / 2]
            free = [o for o in options if all(np.hypot(o[0] - q[0], o[1] - q[1]) >= gap
                                              for q in placed)]
            if free:
                out.append(free[0])
                break
        else:
            out.append(p)
    return out


def _fire_lut() -> np.ndarray:
    """256 BGR colors from dark red through orange and yellow to white-hot."""
    stops = np.array([(10, 10, 90), (20, 30, 200), (0, 100, 255), (0, 190, 255),
                      (150, 245, 255), (245, 255, 255)], np.float32)
    t = np.linspace(0, len(stops) - 1, 256)
    i = np.minimum(t.astype(int), len(stops) - 2)
    u = (t - i)[:, None]
    return (stops[i] * (1 - u) + stops[i + 1] * u).astype(np.uint8)


class MagmaText:
    """A word in moving magma with a flickering glow and rising embers, drawn from the frame number alone."""

    def __init__(self, text: str, px: int, font):
        from PIL import Image, ImageDraw

        pad = px // 2
        face = _face(font[0], font[1], px) if font else None
        probe = ImageDraw.Draw(Image.new("L", (1, 1)))
        x0, y0, x1, y1 = probe.textbbox((0, 0), text, font=face) if face else (0, 0, px * len(text) // 2, px)
        self.w, self.h, self.pad = x1 - x0 + 2 * pad, y1 - y0 + 2 * pad, pad
        img = Image.new("L", (self.w, self.h))
        ImageDraw.Draw(img).text((pad - x0, pad - y0), text, font=face, fill=255)
        self.mask = np.asarray(img, np.float32)[..., None] / 255
        self.halo = cv2.GaussianBlur(self.mask, (0, 0), px * 0.18)[..., None]
        self.halo = self.halo / max(1e-6, float(self.halo.max()))
        rng = np.random.default_rng(7)
        # Soft noise twice the text's height, scrolled upward through the letters.
        noise = cv2.GaussianBlur(rng.random((self.h * 2, self.w)).astype(np.float32), (0, 0), px * 0.12)
        self.noise = (noise - noise.min()) / (noise.max() - noise.min())
        self.ramp = np.linspace(0.35, 1.0, self.h, dtype=np.float32)[:, None]   # hotter at the bottom
        self.embers = rng.random((24, 3))   # x, phase, speed
        self.lut = _fire_lut()

    def draw(self, panel, k: int, level: int, *, right: int, bottom: int) -> None:
        """Draw at frame *k*, right-aligned at *right*, text bottom at *bottom*; hotter with *level*."""
        x0, y0 = right - self.w + self.pad, bottom - self.h + self.pad
        H, W = panel.shape[:2]
        if x0 < 0 or y0 < 0 or x0 + self.w > W or y0 + self.h > H:
            return
        rows = (np.arange(self.h) + int(k * (1.5 + 0.5 * level))) % self.noise.shape[0]
        v = np.clip(0.55 * self.ramp + 0.6 * self.noise[rows] + 0.05 * level - 0.1, 0, 1)
        fill = self.lut[(v * 255).astype(np.uint8)].astype(np.float32)

        flicker = 0.85 + 0.15 * np.sin(k * 0.9) * np.cos(k * 0.37)
        glow = self.halo * np.array((0, 90, 255), np.float32) * (0.5 + 0.2 * level) * flicker

        roi = panel[y0:y0 + self.h, x0:x0 + self.w].astype(np.float32)
        roi = np.clip(roi * (1 - 0.6 * self.halo) + glow, 0, 255)   # a dark halo, so it reads on a bright video
        roi = roi * (1 - self.mask) + fill * self.mask

        # Embers rise off the letters, added as light so they never darken them.
        sparks = np.zeros_like(roi)
        n = min(len(self.embers), 6 + 4 * level)
        for ex, phase, speed in self.embers[:n]:
            life = (phase + k * (0.012 + 0.02 * speed)) % 1.0
            px_ = int(ex * (self.w - 2 * self.pad) + self.pad)
            py_ = int((self.h - self.pad) * 0.6 * (1 - life))
            color = self.lut[int(255 * (1 - 0.5 * life))].astype(np.float32) * (1 - life)
            cv2.circle(sparks, (px_, py_), 1 + int(2 * (1 - life)), color.tolist(), -1, cv2.LINE_AA)
        panel[y0:y0 + self.h, x0:x0 + self.w] = np.clip(roi + sparks, 0, 255).astype(np.uint8)


class Panel:
    """The court filling the panel, with field goals and FG% in its corner; and the hot streak."""

    def __init__(self, shots, fps: float, width: int, height: int, *, cfg, footer: int = 0,
                 show_fg: bool = True):
        from .court import CourtView

        self.show_fg = show_fg   # off for a compilation: its field goals are picked, not earned

        self.shots, self.fps, self.hot = shots, fps, cfg.HOT_STREAK
        s = self.s = width / 1280
        self.font = resolve_font(cfg.PANEL_FONT, cfg.PANEL_FONT_INDEX)
        m = int(28 * s)

        base = np.zeros((height, width, 3), np.uint8)
        base[:] = _bgr(BG)

        top = int(20 * s)
        cw, ch = width - 2 * m, height - footer - top - (0 if footer else int(20 * s))
        self.court = CourtView(cw, ch, three_radius=cfg.THREE_POINT_RADIUS_M,
                               ft_distance=cfg.FREE_THROW_DISTANCE_M,
                               min_depth=cfg.COURT_MIN_DEPTH_M)
        self.origin = (m, top)
        base[top:top + ch, m:m + cw] = self.court.base
        self.r = max(5, int(17 * s))
        spots = _spread([self.court.clamp(*shot.court_xy) if shot.court_xy is not None else None
                         for shot in shots], (2 * self.r + int(4 * s)) / self.court.scale)
        self.spots = [self.court.px(*p) if p is not None else None for p in spots]

        # The tallies, in the court's corner outside the arc, sized to the court.
        t = self.t = min(s, self.court.scale / 83)   # court px per meter, so it holds at any size
        cx, cy = self.court.corner()
        inset = int(24 * t)
        self.value_x = m + cx - inset
        self.label_x = self.value_x - int(220 * t)
        self.rows_y = [top + cy - inset - int(106 * t), top + cy - inset - int(30 * t)]
        if show_fg:
            _text(base, [(label, (self.label_x, y), int(23 * t), _rgb(MUTED), "rm")
                         for label, y in zip(("FIELD GOALS", "FG%"), self.rows_y)], self.font)

        # The hot streak goes on the video, where the eye already is.
        self.magma = MagmaText("HOT STREAK!", int(56 * s), self.font)

        _draw_credit(base, self.font, cfg=cfg, scale=s, footer=footer)
        self.base = base

    def _done(self, k: int) -> list:
        return [shot for shot in self.shots if shot.decided is not None and shot.decided <= k]

    def streak_box(self, width: int) -> tuple[int, int, int, int]:
        """``(x0, y0, x1, y1)``: where the streak text sits on a video *width* wide."""
        s, m = self.s, self.magma
        self.streak_right = width - max(m.pad, int(30 * s))   # room for its glow
        self.streak_bottom = m.h - m.pad   # its glow box flush with the top; the letters sit a pad lower
        return (self.streak_right - m.w + 2 * m.pad, 0, self.streak_right,
                self.streak_bottom + int(14 * s) + int(28 * s))

    def streak(self, frame: np.ndarray, k: int, alpha: float = 1.0) -> None:
        """On a streak: HOT STREAK! and the count in the video's top-right corner, hotter with each make.

        *alpha* below 1 fades it, so the ball stays visible behind it.
        """
        run = 0
        for shot in reversed(self._done(k)):
            if not shot.made:
                break
            run += 1
        if run < self.hot:
            return
        s = self.s
        level = run - self.hot + 1
        heat = min(1.0, (level - 1) / 3)
        color = tuple(int(a + (b - a) * heat) for a, b in zip(_rgb(STREAK), _rgb(MISS)))
        x0, y0, x1, y1 = self.streak_box(frame.shape[1])
        x0, x1, y1 = max(0, x0 - self.magma.pad), frame.shape[1], min(frame.shape[0], y1 + int(12 * s))
        before = frame[y0:y1, x0:x1].copy()

        self.magma.draw(frame, k, level, right=self.streak_right, bottom=self.streak_bottom)
        count = [(f"{run} IN A ROW", (self.streak_right - x0, self.streak_bottom + int(14 * s) - y0),
                  int(28 * s), color, "rt")]
        # A soft dark shadow under the count, like the halo under HOT STREAK!, so it reads on a busy background.
        roi = frame[y0:y1, x0:x1]
        shade = _text(np.zeros_like(roi), [(t, xy, px, (255, 255, 255), a) for t, xy, px, _, a in count], self.font)
        shade = cv2.GaussianBlur(shade[..., :1].astype(np.float32) / 255, (0, 0), max(1.0, 3 * s))[..., None]
        roi[:] = (roi * (1 - 0.7 * np.clip(shade * 2, 0, 1))).astype(np.uint8)
        _text(roi, count, self.font)

        if alpha < 1:
            frame[y0:y1, x0:x1] = (before * (1 - alpha) + frame[y0:y1, x0:x1] * alpha).astype(np.uint8)

    def render(self, k: int) -> np.ndarray:
        panel = self.base.copy()
        s, r = self.s, self.r
        done = self._done(k)
        made = sum(1 for shot in done if shot.made)

        t = self.t
        big = int(64 * t)
        values = (f"{made}/{len(done)}", f"{made / len(done):.0%}" if done else "—")
        items = ([(v, (self.value_x, y), big, _rgb(FG), "rm") for v, y in zip(values, self.rows_y)]
                 if self.show_fg else [])

        # A dot per decided shot, numbered, the newest ringed for a moment.
        ox, oy = self.origin
        for shot in done:
            spot = self.spots[self.shots.index(shot)]
            if spot is None:
                continue
            p = (ox + spot[0], oy + spot[1])
            color = _bgr(MAKE if shot.made else MISS)
            cv2.circle(panel, p, r + max(1, int(3 * s)), _bgr(BG), -1, cv2.LINE_AA)
            cv2.circle(panel, p, r, color, -1, cv2.LINE_AA)
            items.append((str(shot.number), (p[0], p[1] + 1), int(17 * s), _rgb(BG), "mm"))
            if shot is done[-1] and k - shot.decided < 1.5 * self.fps:
                cv2.circle(panel, p, r + int(9 * s), color, max(2, int(3 * s)), cv2.LINE_AA)
        return _text(panel, items, self.font)
