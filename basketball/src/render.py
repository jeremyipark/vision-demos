"""The export video (clip on top, panel below), its overlay, plots.png and elbow.png."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from .ball import smoothed
from .panel import BALL, MAKE, MISS, RIM, Panel, _bgr, _spread, credit_lines
from .shots import J, streaks
from .skeleton import SKELETON, FACE_KPTS, edge_part

_PROGRESS_COLUMNS = (
    TextColumn("[cyan]rendering[/]"),
    BarColumn(),
    TaskProgressColumn(),
    TextColumn("{task.completed}/{task.total} frames"),
    TimeElapsedColumn(),
)



def _p(v) -> tuple[int, int]:
    return int(round(v[0])), int(round(v[1]))


def _draw_skeleton(frame, pts: np.ndarray, *, colors, thick: int, radius: int) -> None:
    """The whole body in the demos' per-limb colors."""
    ok = np.isfinite(pts).all(axis=1)
    for a, b, _ in SKELETON:
        if a in FACE_KPTS or b in FACE_KPTS or not (ok[a] and ok[b]):
            continue
        cv2.line(frame, _p(pts[a]), _p(pts[b]), colors[edge_part(a, b)], thick, cv2.LINE_AA)
    for i in range(len(pts)):
        if ok[i] and i not in FACE_KPTS:
            cv2.circle(frame, _p(pts[i]), radius, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, _p(pts[i]), radius + 1, (20, 22, 26), 1, cv2.LINE_AA)


def _glow(frame, draw, box, sigma: float, gain: float) -> None:
    """Blurred light added onto *box*: *draw* paints the layer, which is then blurred."""
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = (max(0, int(box[0])), max(0, int(box[1])),
                      min(w, int(box[2])), min(h, int(box[3])))
    if x1 <= x0 or y1 <= y0:
        return
    layer = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
    draw(layer, x0, y0)
    layer = cv2.GaussianBlur(layer, (0, 0), sigma)
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = np.clip(roi + gain * layer, 0, 255).astype(np.uint8)


# Flame colors, BGR, from the tail of the trail to the ball.
_FIRE = np.array([(20, 30, 200), (0, 90, 255), (0, 170, 255), (120, 240, 255)], np.float32)


def _fire_color(t: float) -> np.ndarray:
    """Red at 0, through orange, to yellow-white at 1."""
    i = min(len(_FIRE) - 2, int(t * (len(_FIRE) - 1)))
    u = t * (len(_FIRE) - 1) - i
    return _FIRE[i] * (1 - u) + _FIRE[i + 1] * u


def _ball_fire(frame, path, r: float, level: int, fade: float, rng) -> None:
    """A short flame behind the ball along *path* (oldest first), hotter with *level*, dimmed by *fade*."""
    pts = np.array([(x, y) for _, x, y in path], float)
    pad = 3 * r
    box = (pts[:, 0].min() - pad, pts[:, 1].min() - pad, pts[:, 0].max() + pad,
           pts[:, 1].max() + pad)

    def draw(layer, x0, y0):
        n = len(pts)
        for i, (x, y) in enumerate(pts):
            t = (i + 1) / n
            size = r * (0.35 + 0.75 * t) * (1 + 0.1 * level) * rng.uniform(0.95, 1.05)
            jitter = rng.normal(0, 0.06 * r, 2)
            cv2.circle(layer, (int(x - x0 + jitter[0]), int(y - y0 + jitter[1])), int(size),
                       (_fire_color(t) * t ** 1.2).tolist(), -1, cv2.LINE_AA)

    _glow(frame, draw, box, sigma=0.4 * r, gain=(0.8 + 0.15 * level) * fade)


def _rim_heat(frame, center, axes, thick: int, heat: float, rng) -> None:
    """The rim glowing: *heat* 1 is the flare as a hot make drops through, ~0.25 a warm idle."""
    pad = 3 * thick + axes[0] // 3
    box = (center[0] - axes[0] - pad, center[1] - axes[1] - pad,
           center[0] + axes[0] + pad, center[1] + axes[1] + pad)

    def draw(layer, x0, y0):
        cv2.ellipse(layer, (center[0] - x0, center[1] - y0), axes, 0, 0, 360,
                    _fire_color(0.4 + 0.4 * heat).tolist(), max(1, int(thick * (1 + 1.5 * heat))),
                    cv2.LINE_AA)

    _glow(frame, draw, box, sigma=thick * (0.8 + 1.2 * heat),
          gain=(0.4 + 1.1 * heat) * rng.uniform(0.96, 1.04))


def _rim_heats(shots, streak: np.ndarray, fps: float, cfg) -> np.ndarray:
    """The rim's heat per frame: 1 as a streak make drops through, cooling to RIM_WARM."""
    heat = np.where(streak >= cfg.HOT_STREAK, cfg.RIM_WARM, 0.0)
    t = np.arange(len(streak))
    for shot in shots:
        if not shot.made or shot.crossing is None or shot.decided is None:
            continue
        if streak[min(shot.decided, len(streak) - 1)] < cfg.HOT_STREAK:
            continue
        f = int(shot.crossing)
        alive = (t >= f) & ((streak >= cfg.HOT_STREAK) | (t < shot.decided))
        flare = np.exp(-(t - f) / (cfg.RIM_COOL_SECONDS * fps / 3))
        heat = np.where(alive, np.maximum(heat, cfg.RIM_WARM + (1 - cfg.RIM_WARM) * flare), heat)
    return heat


def render(info, *, kp: np.ndarray, hoop, track, shots, dst: Path, cfg, console) -> dict:
    """Write the overlaid clip with the panel beneath it."""
    video_h = min(int(cfg.EXPORT_HEIGHT) // 2, info.height) // 2 * 2
    sx = video_h / info.height
    width = int(round(info.width * sx / 2)) * 2
    out_h = 2 * video_h   # the panel is as tall as the clip
    footer = int(round(cfg.CREDIT_FOOTER * width / 1280 / 2)) * 2 if credit_lines(cfg) else 0
    panel = Panel(shots, info.fps, width, out_h - video_h, cfg=cfg, footer=footer)

    scale = width / 720
    thick = max(1, round(cfg.LINE_THICKNESS * scale))
    radius = max(1, round(cfg.POINT_RADIUS * scale))
    fps = info.fps
    flash = int(cfg.VERDICT_SECONDS * fps)
    trail = int(cfg.BALL_TRAIL_SECONDS * fps)

    # The rim as an ellipse, in output pixels.
    rim_c = (int(hoop.cx * sx), int(hoop.cy * sx))
    rim_ax = (max(2, int(hoop.width * sx / 2)), max(2, int((hoop.cy - hoop.top) * sx)))
    streak = streaks(shots, info.n_frames)
    bx, by = smoothed(track, max_gap=int(cfg.BALL_FILL_SECONDS * fps))
    ball_r = max(3, int(track.diameter * sx / 2)) if np.isfinite(track.diameter) else 6
    # The streak text fades while the ball passes behind it, easing over a few frames.
    x0, y0, x1, y1 = panel.streak_box(width)
    with np.errstate(invalid="ignore"):
        under = np.flatnonzero((bx * sx >= x0 - ball_r) & (bx * sx <= x1 + ball_r)
                               & (by * sx >= y0 - ball_r) & (by * sx <= y1 + ball_r))
    near = (np.abs(np.arange(info.n_frames)[:, None] - under[None]).min(axis=1)
            if len(under) else np.full(info.n_frames, 99))
    streak_alpha = cfg.STREAK_FADE + (1 - cfg.STREAK_FADE) * np.clip((near - 1) / 4, 0, 1)
    rim_heat = _rim_heats(shots, streak, fps, cfg)
    rng = np.random.default_rng(0)   # the flames' flicker, the same every run

    cap = cv2.VideoCapture(str(info.path))
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, out_h))
    ball_frames = 0
    with Progress(*_PROGRESS_COLUMNS, console=console, transient=True) as progress:
        task = progress.add_task("render", total=info.n_frames)
        for k in range(info.n_frames):
            ok, src = cap.read()
            if not ok:
                break
            frame = cv2.resize(src, (width, video_h), interpolation=cv2.INTER_AREA)
            active = next((s for s in shots if s.start - fps <= k <= (s.decided or s.end) + flash),
                          None)

            pts = kp[k] * sx
            if np.isfinite(pts).any():
                _draw_skeleton(frame, pts, colors=cfg.SKELETON_COLORS, thick=thick, radius=radius)

            # On a streak the rim flares on each make, then cools to a warm glow.
            if rim_heat[k] > 0:
                _rim_heat(frame, rim_c, rim_ax, thick, float(rim_heat[k]), rng)

            # The rim, lit green or red once a shot is decided, for VERDICT_SECONDS.
            verdict = next((s for s in shots if s.decided is not None
                            and s.decided <= k < s.decided + flash), None)
            rim_color = (_bgr(MAKE if verdict.made else MISS) if verdict is not None else _bgr(RIM))
            cv2.ellipse(frame, rim_c, rim_ax, 0, 0, 360, rim_color,
                        max(2, thick if verdict is not None else thick // 2 + 1), cv2.LINE_AA)

            # The ball and its path through the whole shooting motion; on fire when shot on a streak.
            first = None if active is None else active.start if active.motion is None else active.motion
            if active is not None and k >= first and np.isfinite(bx[k]):
                ball_frames += 1
                hot = streak[max(0, active.start - 1)] - cfg.HOT_STREAK + 1
                lo = max(active.start - int(fps), k - trail)
                path = [(f, bx[f] * sx, by[f] * sx) for f in range(lo, k + 1)
                        if np.isfinite(bx[f]) and f >= first]
                r = ball_r
                if hot > 0:
                    # On fire from the release, fading out once it drops through the rim.
                    out = active.crossing if active.crossing is not None else active.decided
                    fade = 1.0 if out is None or k <= out else max(
                        0.0, 1 - (k - out) / (cfg.BALL_FIRE_FADE_SECONDS * fps))
                    flame = [p for p in path if p[0] >= max(active.release, k - int(cfg.BALL_FIRE_SECONDS * fps))]
                    if fade > 0 and len(flame) >= 2:
                        _ball_fire(frame, flame, r, hot, fade, rng)
                    path = []
                for (f0, x0, y0), (f1, x1, y1) in zip(path, path[1:]):
                    if f1 - f0 > 6:
                        continue
                    fade = 0.25 + 0.75 * (f1 - lo) / max(1, k - lo)
                    color = tuple(int(c * fade) for c in _bgr(BALL))
                    cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), color,
                             max(1, int(thick * fade)), cv2.LINE_AA)
                c = (int(bx[k] * sx), int(by[k] * sx))
                cv2.circle(frame, c, r + 2, (20, 22, 26), 1, cv2.LINE_AA)
                cv2.circle(frame, c, r, _bgr(BALL), max(2, thick - 1), cv2.LINE_AA)

            panel.streak(frame, k, streak_alpha[k])
            out = np.vstack([frame, panel.render(k)])
            writer.write(out)
            progress.update(task, completed=k + 1)
    cap.release()
    writer.release()
    return {"width": width, "height": out_h, "video_height": video_h,
            "ball_frames_drawn": ball_frames}


# ── plots.png and elbow.png ──────────────────────────────────────────────────

def plot_summary(shots, fps: float, hoop, frame_bgr: np.ndarray, dst: Path, *, cfg) -> None:
    """Two views: every ball path at the hoop, and the shot chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .court import CourtView

    good, bad = "#2f9e44", "#e03131"
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 7.5), dpi=110,
                                  gridspec_kw={"width_ratios": [1.1, 1]})

    # 1. The ball against the rim: what each verdict was read from.
    w = hoop.width
    x0, x1 = int(hoop.cx - 4 * w), int(hoop.cx + 4 * w)
    y0, y1 = int(hoop.top - 3.5 * w), int(hoop.net_bottom + 2.5 * w)
    x0, y0 = max(0, x0), max(0, y0)
    crop = frame_bgr[y0:y1, x0:x1, ::-1]
    ax.imshow(crop, extent=(x0, x0 + crop.shape[1], y0 + crop.shape[0], y0), alpha=0.55)
    for shot in shots:
        if not shot.path:
            continue
        f, x, y = np.array(shot.path).T
        color = good if shot.made else bad
        ax.plot(x, y, "-", color=color, lw=1.6, alpha=0.9)
        ax.plot(x, y, ".", color=color, ms=3)
        if shot.crossing is not None:
            i = int(np.clip(np.searchsorted(f, shot.crossing), 0, len(f) - 1))
        else:
            i = int(np.argmin(np.hypot(x - hoop.cx, y - hoop.cy)))
        ax.annotate(str(shot.number), (x[i], y[i]), color=color, fontsize=9, weight="bold",
                    xytext=(6, -6), textcoords="offset points")
    inner = cfg.RIM_INNER_FRACTION * w / 2
    ax.plot([hoop.left, hoop.right], [hoop.cy, hoop.cy], color="#f08c00", lw=2.5, label="rim")
    ax.plot([hoop.cx - inner, hoop.cx + inner], [hoop.cy, hoop.cy], color="k", lw=1, ls=":",
            label="inside the rim")
    ax.axhline(hoop.net_bottom, color="#868e96", lw=1, ls="--", label="bottom of the net")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_title("Ball path at the hoop (green make, red miss)", loc="left")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])

    # 2. The shot chart.
    view = CourtView(700, 760, three_radius=cfg.THREE_POINT_RADIUS_M,
                     ft_distance=cfg.FREE_THROW_DISTANCE_M)
    img = view.base.copy()
    spots = _spread([view.px(*view.clamp(*s.court_xy)) if s.court_xy is not None else None
                     for s in shots], 32)
    for shot, p in zip(shots, spots):
        if p is None:
            continue
        cv2.circle(img, p, 14, _bgr(MAKE if shot.made else MISS), -1, cv2.LINE_AA)
        cv2.putText(img, str(shot.number), (p[0] - 6 * len(str(shot.number)), p[1] + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 22, 26), 2, cv2.LINE_AA)
    ax2.imshow(img[..., ::-1])
    made = sum(1 for s in shots if s.made)
    ax2.set_title(f"Shot chart: {made}/{len(shots)}"
                  + (f" ({made / len(shots):.0%})" if shots else "")
                  + "  ·  spots estimated from one camera", loc="left")
    ax2.axis("off")

    fig.tight_layout()
    fig.savefig(dst)
    plt.close(fig)


def plot_elbow(shots, angle: np.ndarray, fps: float, video_path: Path, kp: np.ndarray,
               side: str, dst: Path, *, cfg) -> None:
    """Left: each side-on shot's elbow, lined up on its release. Right: a strip of
    frames per shot with the arm and its angle drawn on."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    good, bad = "#2f9e44", "#e03131"
    measured = [s for s in shots if s.side_on]
    rows = max(1, len(measured))
    columns = cfg.ELBOW_STRIP
    fig = plt.figure(figsize=(7 + 2.1 * len(columns), max(6.0, 2.3 * rows)), dpi=110)
    outer = fig.add_gridspec(1, 2, width_ratios=[7, 2.1 * len(columns)], wspace=0.08)
    ax = fig.add_subplot(outer[0])

    # 1. Through the shot: the set point, the extension, the release.
    lo, hi = int(-cfg.ELBOW_BEFORE_SECONDS * fps), int(cfg.ELBOW_AFTER_SECONDS * fps)
    offsets = np.arange(lo, hi + 1)
    # Labels at each curve's end, nudged apart where they'd collide.
    ends = {}
    for shot in sorted(measured, key=lambda s: -s.elbow_release):
        idx = min(len(angle) - 1, shot.release + offsets[-1])
        y = angle[idx] if np.isfinite(angle[idx]) else shot.elbow_release
        for other in ends.values():
            if abs(y - other) < 6:
                y = other - 6
        ends[shot.number] = y
    for shot in measured:
        idx = shot.release + offsets
        ok = (idx >= 0) & (idx < len(angle))
        curve = np.full(len(idx), np.nan)
        curve[ok] = angle[idx[ok]]
        color = good if shot.made else bad
        ax.plot(offsets / fps, curve, color=color, lw=1.6, alpha=0.85)
        ax.plot((shot.set_frame - shot.release) / fps, shot.elbow_set, "o", mfc="white",
                mec=color, ms=6)
        ax.plot(0, shot.elbow_release, "o", color=color, ms=7)
        ax.annotate(f"{shot.number}: {shot.elbow_set:.0f}° – {shot.elbow_release:.0f}°",
                    (offsets[-1] / fps, ends[shot.number]), xytext=(4, 0),
                    textcoords="offset points", fontsize=8, color=color, va="center")
    ax.axvline(0, color="#868e96", lw=1)
    ax.axhline(90, color="#adb5bd", lw=1, ls="--")
    # Mark the moments the strip shows.
    for c in cfg.ELBOW_STRIP:
        if c not in ("set", "0"):
            ax.axvline(float(c), color="#dee2e6", lw=1, ls=":", zorder=0)
    ax.set_xlim(offsets[0] / fps, offsets[-1] / fps + 0.3)
    ax.set_ylim(0, 190)
    ax.set_xlabel("seconds from release")
    ax.set_ylabel("degrees")
    skipped = len(shots) - len(measured)
    ax.set_title("Elbow through the shot, side-on shots"
                 + (f" ({skipped} not side-on, left out)" if skipped else ""), loc="left")

    # 2. The strip: the same moments of every shot, side by side.
    strip = outer[1].subgridspec(rows, len(columns), wspace=0.04, hspace=0.12)
    arm = [J[f"{side}_{j}"] for j in ("shoulder", "elbow", "wrist")]
    upper = [J[n] for n in ("nose", "left_shoulder", "right_shoulder", "left_elbow",
                             "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip")]
    cap = cv2.VideoCapture(str(video_path))
    for r, shot in enumerate(measured):
        # "release" is the arm's fullest extension just after the ball leaves (the plot's dot).
        after = angle[shot.release:shot.release + int(0.2 * fps) + 1]
        peak = shot.release + (int(np.nanargmax(after)) if np.isfinite(after).any() else 0)
        frames = [shot.set_frame if c == "set" else peak if c == "0"
                  else shot.release + int(round(float(c) * fps)) for c in columns]
        frames = [int(np.clip(f, 0, len(angle) - 1)) for f in frames]
        # One crop per shot around the upper body, so only the body moves along the strip.
        pts = kp[frames][:, upper].reshape(-1, 2)
        pts = pts[np.isfinite(pts).all(axis=1)]
        (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
        y0 -= 0.25 * (y1 - y0)   # headroom for the ball over the hands
        half = 0.6 * max(x1 - x0, y1 - y0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        color = good if shot.made else bad
        for c, (label, f) in enumerate(zip(columns, frames)):
            cell = fig.add_subplot(strip[r, c])
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if ok:
                h, w = img.shape[:2]
                a0, a1 = int(max(0, cx - half)), int(min(w, cx + half))
                b0, b1 = int(max(0, cy - half)), int(min(h, cy + half))
                cell.imshow(img[b0:b1, a0:a1, ::-1])
                p = kp[f, arm] - [a0, b0]
                if np.isfinite(p).all():
                    cell.plot(p[:, 0], p[:, 1], "-", color="#1c7ed6", lw=2.5)
                    cell.plot(p[:, 0], p[:, 1], "o", color="white", mec="#1c7ed6", ms=4)
                text = f"{angle[f]:.0f}°" if np.isfinite(angle[f]) else "—"
                cell.text(0.04, 0.05, text, transform=cell.transAxes, color="white",
                          fontsize=11, weight="bold", va="bottom",
                          bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6, lw=0))
            cell.set_xticks([])
            cell.set_yticks([])
            for spine in cell.spines.values():
                spine.set_color(color if label in ("set", "0") else "#ced4da")
                spine.set_linewidth(2 if label in ("set", "0") else 0.8)
            if r == 0:
                title = ("set point" if label == "set" else "release" if label == "0"
                         else f"{float(label):+.1f}s")
                cell.set_title(title, fontsize=10)
            if c == 0:
                cell.set_ylabel(f"shot {shot.number}\n{shot.label}", color=color, fontsize=10,
                                weight="bold", rotation=0, ha="right", va="center")
    cap.release()

    fig.savefig(dst, bbox_inches="tight")
    plt.close(fig)
