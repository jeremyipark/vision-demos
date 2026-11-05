"""Every session at once: a shot chart, a heat map of where the shots come from, FG% over time,
and CSVs to download.

    python history.py                  # every clip under data/input/, subfolders included
    python history.py --cached-only    # never call the gateway; skip a clip that isn't cached
    python history.py a.mov b.mov      # just these

Re-uses main.py's cached model replies (data/cache/), so a clip that has been run
before costs nothing and renders no video. Sessions are named by when the phone
recorded them (the clip's capture date), not by file name. Clips cut from the same
recording share its date, so they also carry their folder: "2:53 PM (session)".

Writes to data/output/history/, replacing the last run:

    report.jpg      one page: FG%, the shot chart, hot and cold spots, a frame from each session
    shot_chart.jpg  every shot on the court, make or miss
    heat_map.jpg    hot and cold spots: FG% around each spot against the overall average,
                    faded where there are few shots
    volume_map.jpg  where the shots come from: red where most were taken
    trends.jpg      FG% per session, and by distance
    shots.csv       one row per shot
    sessions.csv    one row per session
    zones.csv       one row per court zone: attempts, makes, FG%
"""

from __future__ import annotations

import argparse
import warnings
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

import config as cfg
import main
from src import court, hoop as hoop_mod, shots as shots_mod, video
from src.env import load_api_key

console = main.console
HISTORY_DIR = cfg.OUTPUT_DIR / "history"
FT_PER_M = 3.2808
TITLE_SIZE = 20

# Colors (light, print-friendly). Make/miss also differ by marker: filled dot vs x.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8b8a85"
MAKE = "#0ca30c"
MISS = "#e34948"
BAR = "#2a78d6"
RIM = "#d2622c"
# The heat map, few shots -> many: pale yellow through orange to deep red.
HEAT = ["#fff1a8", "#fed16b", "#fd9a44", "#f0582f", "#d42a20", "#9e0f14"]
HEAT_SIGMA_M = 0.9        # how far one shot's warmth spreads
# Hot and cold: FG% around a spot against the overall average. Diverging, gray at the average.
COLD_HOT = ["#1c5cab", "#86b6ef", "#f0efec", "#f39a7c", "#c62a22"]
HOT_RANGE = 0.35          # this far above (below) the average is full red (blue)
HOT_PRIOR = 0.5           # shots' worth of the average mixed in, so one shot isn't 0% or 100%

DISTANCE_BINS_FT = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 99)]


class NotCached(Exception):
    pass


# ── clips and when they were recorded ────────────────────────────────────────

def find_clips(inputs: list[Path]) -> list[Path]:
    clips: list[Path] = []
    for item in inputs:
        path = item.expanduser().resolve()
        if path.is_dir():
            clips += sorted(p for p in path.rglob("*")
                            if p.is_file() and not p.name.startswith(".")
                            and p.suffix.lower() in cfg.VIDEO_EXTENSIONS)
        elif path.is_file():
            clips.append(path)
        else:
            console.print(f"[red]Not found:[/] {item}")
    return list(dict.fromkeys(clips))


def _parse_time(value: str) -> datetime | None:
    for parse in (lambda v: datetime.strptime(v, "%Y-%m-%dT%H:%M:%S%z"),
                  lambda v: datetime.fromisoformat(v.replace("Z", "+00:00"))):
        try:
            return parse(value)
        except ValueError:
            pass
    return None


def capture_info(clip: Path) -> dict:
    """When and on what the clip was recorded, from its metadata.

    iPhones write ``com.apple.quicktime.creationdate``: the moment recording started,
    in the phone's local time. ``creation_time`` is a fallback, but editing or
    AirDropping a clip can rewrite it; the file's own date is the last resort.
    """
    proc = subprocess.run([video.tool("ffprobe"), "-v", "error", "-show_entries", "format=duration"
                           ":format_tags", "-of", "json", str(clip)], capture_output=True, text=True)
    fmt = json.loads(proc.stdout or "{}").get("format", {})
    tags = fmt.get("tags", {})
    when, source = None, ""
    for key, label in (("com.apple.quicktime.creationdate", "recorded"),
                       ("creation_time", "file metadata")):
        if tags.get(key) and (when := _parse_time(tags[key])):
            source = label
            break
    if when is None:
        when, source = datetime.fromtimestamp(clip.stat().st_mtime).astimezone(), "file date"
    # Show it in the time zone it was recorded in (creation_time is UTC).
    if source != "recorded":
        when = when.astimezone()
    camera = " ".join(filter(None, (tags.get("com.apple.quicktime.make"),
                                    tags.get("com.apple.quicktime.model"))))
    return {"when": when, "source": source, "camera": camera,
            "duration": float(fmt.get("duration") or 0)}


def session_label(when: datetime) -> str:
    """``Sat Nov 1, 2:51 PM``"""
    return f"{when:%a %b} {when.day}, {when.strftime('%I:%M %p').lstrip('0')}"


# ── analysis, from main.py's cache ───────────────────────────────────────────

def analyze_clip(clip: Path, api_key: str) -> tuple[video.VideoInfo, list[shots_mod.Shot], tuple]:
    """main.py's steps 2-6 (no render): shots judged make or miss, with court spots,
    and a thumbnail of the session."""
    c = load_clip(clip, api_key)
    return c["info"], c["shots"], thumbnail(c["mp4"], c["shots"], c["angle"], c["kp"],
                                            c["hoop"], c["info"].fps)


def load_clip(clip: Path, api_key: str) -> dict:
    """Everything one clip's analysis leaves: ``info, shots, kp, side, angle, mp4, hoop``."""
    cfg.INPUT_VIDEO = clip
    src = video.probe_source(clip)
    tonemap, _ = video.tonemap_filter(src, mode=cfg.TONEMAP, algorithm=cfg.TONEMAP_ALGORITHM)
    mp4 = video.cache_path(clip, cfg.CACHE_DIR, target_height=cfg.INFERENCE_HEIGHT,
                           trim_seconds=cfg.TRIM_SECONDS, crf=cfg.CONVERT_CRF, tonemap=tonemap)
    if not mp4.is_file():
        if main.client is _refuse:
            raise NotCached("never converted")
        video.convert(clip, mp4, target_height=cfg.INFERENCE_HEIGHT,
                      trim_seconds=cfg.TRIM_SECONDS, crf=cfg.CONVERT_CRF,
                      total_frames=src["n_frames"], console=console, tonemap=tonemap)
    info = video.inspect(mp4)
    try:
        posed = main.pose_clip(api_key, mp4, info)
        kp = shots_mod.keypoints(posed["by_index"], info.n_frames, info.width, info.height)
        spans = shots_mod.candidates(kp, info.fps, armed_below=cfg.SHOT_ARMED_BELOW,
                                     min_gap_seconds=cfg.SHOT_MIN_GAP_SECONDS)
        side = (cfg.SHOOTING_HAND if cfg.SHOOTING_HAND in ("left", "right")
                else shots_mod.shooting_side(kp, spans, info.fps))
        hoop, _ = main.find_hoop(api_key, mp4, info)
        track, _, calls = main.track_ball(api_key, mp4, info, spans, hoop)
    except NotCached:
        for seg in cfg.CACHE_DIR.glob(f"{mp4.stem}.win*.mp4"):   # cut for a call never made
            seg.unlink(missing_ok=True)
        raise
    console.print(f"  [green]cached[/]: poses, hoop and {calls} ball windows")
    shots = main.analyze(kp, info, spans, hoop, track, side)
    angle = shots_mod.elbow_angle(kp, side, min_projected=cfg.ELBOW_MIN_PROJECTED)
    return {"info": info, "shots": shots, "kp": kp, "side": side, "angle": angle, "mp4": mp4,
            "hoop": hoop, "track": track}


def thumbnail(mp4: Path, shots, angle: np.ndarray, kp: np.ndarray, hoop, fps: float,
              width: int = 720, aspect: float = 3 / 2):
    """``(RGB frame, shot)``: a made shot at its release, as elbow.png's "release" column
    picks it: the arm's fullest extension just after the ball leaves. Side-on shots first,
    since the arm reads best from the side. Cropped to the shooter and the hoop."""
    pick = (next((s for s in shots if s.made and s.side_on), None)
            or next((s for s in shots if s.made), None) or (shots[0] if shots else None))
    if pick is None:
        return None, None
    after = angle[pick.release:pick.release + int(0.2 * fps) + 1]
    frame = pick.release + (int(np.nanargmax(after)) if np.isfinite(after).any() else 0)
    frame = min(frame, len(angle) - 1)
    img = hoop_mod.read_frame(mp4, frame)
    h, w = img.shape[:2]
    body = kp[frame][:, :2]
    body = body[np.isfinite(body).all(axis=1)]
    if len(body):
        xs = [*body[:, 0], hoop.left, hoop.right]
        ys = [*body[:, 1], hoop.top, hoop.net_bottom]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        pad = 0.15 * max(x1 - x0, y1 - y0)
        x0, x1, y0, y1 = x0 - pad, x1 + pad, y0 - pad, y1 + pad
        # Grow the short side to *aspect*, then slide the box back inside the frame.
        bw, bh = x1 - x0, y1 - y0
        bw, bh = max(bw, bh * aspect), max(bh, bw / aspect)
        bw, bh = min(bw, w), min(bh, h)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        a0 = int(np.clip(cx - bw / 2, 0, w - bw))
        b0 = int(np.clip(cy - bh / 2, 0, h - bh))
        img = img[b0:b0 + int(bh), a0:a0 + int(bw)]
    img = cv2.resize(img, (width, round(img.shape[0] * width / img.shape[1])),
                     interpolation=cv2.INTER_AREA)
    return img[..., ::-1], pick


def _refuse(api_key):
    raise NotCached("needs a gateway call")


# ── court zones ──────────────────────────────────────────────────────────────

ZONES = ["Restricted area", "Paint", "Mid-range left", "Mid-range center", "Mid-range right",
         "Corner 3 left", "Wing 3 left", "Top of key 3", "Wing 3 right", "Corner 3 right"]


def zone_index(x, y) -> np.ndarray:
    """Zone of each spot, meters from the hoop (x is the shooter's right as they face it)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    r3 = cfg.THREE_POINT_RADIUS_M
    corner_y = math.sqrt(max(0.0, r3 ** 2 - court.CORNER_THREE_X ** 2))
    angle = np.degrees(np.arctan2(np.maximum(y, 0), x))   # 0 right baseline, 90 straight on
    side3 = np.where(angle < 60, 2, np.where(angle > 120, 0, 1))   # left, center, right
    corner = (np.abs(x) >= court.CORNER_THREE_X) & (y <= corner_y)
    three = corner | (np.hypot(x, y) > r3)
    out = np.where(three, np.where(corner, np.where(x > 0, 9, 5), 6 + side3), 2 + side3)
    paint = (np.abs(x) < court.LANE_HALF_WIDTH) & (y < cfg.FREE_THROW_DISTANCE_M) & ~three
    out = np.where(paint, 1, out)
    out = np.where(np.hypot(x, y) < 1.25, 0, out)
    return out


def zone_of(x: float, y: float) -> str:
    return ZONES[int(zone_index([x], [y])[0])]


# ── rows ─────────────────────────────────────────────────────────────────────

@dataclass
class Session:
    clip: Path
    when: datetime
    source: str
    camera: str
    duration: float
    fps: float
    shots: list
    tag: str = ""        # set when another clip shares this one's recording time
    thumb: np.ndarray | None = None   # RGB, a frame from the shot below
    thumb_shot: object = None

    @property
    def name(self) -> str:
        """For the CSVs: ``2026-11-01 14:53``, or ``2026-11-01 14:53 (session)``."""
        return self.when.strftime("%Y-%m-%d %H:%M") + (f" ({self.tag})" if self.tag else "")

    @property
    def label(self) -> str:
        return session_label(self.when) + (f" ({self.tag})" if self.tag else "")


def shot_rows(sessions: list[Session]) -> list[dict]:
    view = court.CourtView(10, 10, three_radius=cfg.THREE_POINT_RADIUS_M,
                           ft_distance=cfg.FREE_THROW_DISTANCE_M)
    rows = []
    for s in sessions:
        streak = 0
        for shot in s.shots:
            streak = streak + 1 if shot.made else 0
            t = shot.release / s.fps
            xy = shot.court_xy
            rows.append({
                "session": s.name,
                "date": s.when.strftime("%Y-%m-%d"),
                "time_of_day": (s.when + timedelta(seconds=t)).strftime("%H:%M:%S"),
                "clip": rel(s.clip),
                "shot": shot.number,
                "video_time": f"{int(t // 60)}:{t % 60:04.1f}",
                "result": {True: "make", False: "miss"}.get(shot.made, "unknown"),
                "points": (3 if xy and view.is_three(*xy) else 2) if shot.made else 0,
                "shot_type": ("3PT" if view.is_three(*xy) else "2PT") if xy else "",
                "zone": zone_of(*xy) if xy else "",
                "distance_ft": _r(shot.distance_m and shot.distance_m * FT_PER_M, 1),
                "court_x_ft": _r(xy and xy[0] * FT_PER_M, 1),
                "court_y_ft": _r(xy and xy[1] * FT_PER_M, 1),
                "makes_in_a_row": streak,
                "elbow_set_deg": _r(shot.elbow_set, 0),
                "elbow_release_deg": _r(shot.elbow_release, 0),
                "how_it_was_called": shot.reason,
            })
    return rows


def _r(value, digits: int):
    if value is None or not np.isfinite(value):
        return ""
    return round(float(value), digits) if digits else int(round(float(value)))


def _pct(made: int, att: int):
    """FG% as a number (62.5), so spreadsheets can sort and chart it."""
    return round(100 * made / att, 1) if att else ""


def _show(pct) -> str:
    return "" if pct == "" else f"{pct:.0f}%"


def tally(rows: list[dict]) -> dict:
    made = sum(r["result"] == "make" for r in rows)
    out = {"attempts": len(rows), "makes": made, "fg_pct": _pct(made, len(rows))}
    for kind in ("2PT", "3PT"):
        sub = [r for r in rows if r["shot_type"] == kind]
        m = sum(r["result"] == "make" for r in sub)
        out[f"{kind.lower()}_made"], out[f"{kind.lower()}_att"] = m, len(sub)
        out[f"{kind.lower()}_pct"] = _pct(m, len(sub))
    out["points"] = sum(r["points"] for r in rows)
    out["longest_make_streak"] = max((r["makes_in_a_row"] for r in rows), default=0)
    dist = [r["distance_ft"] for r in rows if r["distance_ft"] != ""]
    out["avg_distance_ft"] = round(float(np.mean(dist)), 1) if dist else ""
    return out


def session_rows(sessions: list[Session], rows: list[dict]) -> list[dict]:
    out = []
    for s in sessions:
        mine = [r for r in rows if r["clip"] == rel(s.clip)]
        elbows = [r["elbow_release_deg"] for r in mine if r["elbow_release_deg"] != ""]
        out.append({
            "session": s.name, "date": s.when.strftime("%Y-%m-%d"),
            "day": s.when.strftime("%A"), "start_time": s.when.strftime("%H:%M:%S"),
            "length_min": round(s.duration / 60, 1), "clip": rel(s.clip), "camera": s.camera,
            "date_from": s.source, **tally(mine),
            "avg_elbow_at_release_deg": round(float(np.mean(elbows))) if elbows else "",
        })
    return out


def zone_rows(rows: list[dict]) -> list[dict]:
    out = []
    for z in ZONES:
        mine = [r for r in rows if r["zone"] == z]
        made = sum(r["result"] == "make" for r in mine)
        out.append({"zone": z, "shot_type": "3PT" if " 3" in z else "2PT",
                    "attempts": len(mine), "makes": made, "fg_pct": _pct(made, len(mine))})
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        out = csv.DictWriter(fh, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(cfg.INPUT))
    except ValueError:
        return str(main.rel(path))


# ── charts ───────────────────────────────────────────────────────────────────

def _hex_bgr(color: str) -> np.ndarray:
    return np.array([int(color[i:i + 2], 16) for i in (5, 3, 1)], np.float32)


def _interp(colors: list[str], t: np.ndarray) -> np.ndarray:
    """Values 0-1 to BGR along *colors*, interpolated."""
    stops = np.array([_hex_bgr(c) for c in colors])
    t = np.clip(t, 0, 1) * (len(colors) - 1)
    i = np.minimum(t.astype(int), len(colors) - 2)
    f = (t - i)[..., None]
    return stops[i] * (1 - f) + stops[i + 1] * f


def _view(**colors) -> court.CourtView:
    """The court drawn to just past the three-point arc: where the shots are."""
    return court.CourtView(700, 500, three_radius=cfg.THREE_POINT_RADIUS_M,
                           ft_distance=cfg.FREE_THROW_DISTANCE_M, **colors)


def _court(fill: np.ndarray | None = None) -> tuple[court.CourtView, np.ndarray]:
    """A light court; *fill* (BGR per pixel) is painted under the lines."""
    view = _view(line=(255, 255, 255), floor=(0, 0, 0), paint=(0, 0, 0))
    alpha = view.base.max(axis=2, keepdims=True).astype(np.float32) / 255   # where the lines are
    under = np.empty_like(view.base, np.float32)
    under[:] = _hex_bgr(SURFACE)
    if fill is not None:
        under = fill.astype(np.float32)
    img = under * (1 - alpha) + _hex_bgr(INK_2) * alpha
    cv2.circle(img, view.px(0, 0), max(3, int(court.BACKBOARD_HALF_WIDTH / 3 * view.scale)),
               tuple(float(v) for v in _hex_bgr(RIM)), 3, cv2.LINE_AA)
    return view, img.astype(np.uint8)[..., ::-1]   # RGB, for matplotlib


def _pixel_meters(view: court.CourtView) -> tuple[np.ndarray, np.ndarray]:
    v, u = np.mgrid[0:view.h, 0:view.w]
    return (u - view.ox) / view.scale, (v - view.oy) / view.scale


def draw_shot_chart(ax, rows: list[dict]) -> None:
    view, img = _court()
    ax.imshow(img)
    # Misses drawn last: a thin x stays visible on top of a dot at the same spot.
    for result, color, marker, size in (("make", MAKE, "o", 190), ("miss", MISS, "x", 170)):
        pts = [view.px(*view.clamp(r["court_x_ft"] / FT_PER_M, r["court_y_ft"] / FT_PER_M))
               for r in rows if r["result"] == result and r["court_x_ft"] != ""]
        if pts:
            u, v = np.array(pts).T
            ax.scatter(u, v, s=size, marker=marker, c=color, alpha=0.85,
                       linewidths=3.2 if marker == "x" else 2,
                       edgecolors=SURFACE if marker == "o" else None,
                       label=f"{result} ({len(pts)})", zorder=3)
    ax.set_title("Shot chart",
                 loc="left", color=INK, fontsize=TITLE_SIZE, weight="bold", pad=14)
    ax.legend(loc="lower right", bbox_to_anchor=(0.9, 0.1), frameon=False, fontsize=13,
              labelcolor=INK_2, markerscale=0.8)
    ax.axis("off")


def _spots_m(rows: list[dict], view: court.CourtView) -> np.ndarray:
    """Every shot's spot in meters, kept on the drawn court."""
    return np.array([view.clamp(r["court_x_ft"] / FT_PER_M, r["court_y_ft"] / FT_PER_M)
                     for r in rows if r["court_x_ft"] != ""]).reshape(-1, 2)


def _shot_field(rows: list[dict]):
    """Each shot spread over the floor: ``(view grid x, y, volume, makes, on_court)``."""
    probe = _view()
    mx, my = _pixel_meters(probe)
    with_spot = [r for r in rows if r["court_x_ft"] != ""]
    volume = np.zeros(mx.shape, np.float32)
    makes = np.zeros(mx.shape, np.float32)
    for (x, y), r in zip(_spots_m(with_spot, probe), with_spot):
        g = np.exp(-((mx - x) ** 2 + (my - y) ** 2) / (2 * HEAT_SIGMA_M ** 2))
        volume += g
        makes += g * (r["result"] == "make")
    on_court = ((np.abs(mx) <= court.COURT_HALF_WIDTH) & (my >= court.BASELINE_Y)
                & (my <= probe.depth_m))
    return volume, makes, on_court


def _fade(volume: np.ndarray) -> np.ndarray:
    """Opacity from shot volume: a lone shot is a soft glow, a busy spot is solid."""
    level = volume / volume.max() if volume.max() > 0 else volume
    return (np.clip((level - 0.03) / 0.35, 0, 1) * 0.92)[..., None]


def _color_key(ax, colors: list[str], ticks: list[tuple[float, str]], note: str = "") -> None:
    """A thin color bar under the court, right-aligned, with *note* on the left."""
    from matplotlib.colors import LinearSegmentedColormap
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    cax = inset_axes(ax, width="40%", height="100%", loc="center right", borderpad=0,
                     bbox_to_anchor=(0, -0.075, 1, 0.03), bbox_transform=ax.transAxes)
    cax.imshow(np.linspace(0, 1, 256)[None], aspect="auto",
               cmap=LinearSegmentedColormap.from_list("key", colors))
    cax.set_xticks([t * 255 for t, _ in ticks], [label for _, label in ticks], fontsize=11,
                   color=INK_2)
    cax.set_yticks([])
    for spine in cax.spines.values():
        spine.set_visible(False)
    cax.tick_params(length=0)
    if note:
        ax.text(0.01, -0.06, note, transform=ax.transAxes, fontsize=11, color=INK_2, va="center")


def draw_heat_map(ax, rows: list[dict]) -> None:
    """Hot and cold spots: FG% around each spot against the overall average, red above and
    blue below, faded where few shots were taken."""
    volume, makes, on_court = _shot_field(rows)
    shots = [r for r in rows if r["court_x_ft"] != ""]
    avg = sum(r["result"] == "make" for r in shots) / len(shots) if shots else 0.0
    fg = (makes + HOT_PRIOR * avg) / (volume + HOT_PRIOR)
    t = np.clip(0.5 + (fg - avg) / (2 * HOT_RANGE), 0, 1)
    alpha = _fade(volume)
    fill = _hex_bgr(SURFACE) * (1 - alpha) + _interp(COLD_HOT, t) * alpha
    fill[~on_court] = _hex_bgr(SURFACE)
    _, img = _court(fill)
    ax.imshow(img)
    ax.set_title("Hot and cold spots", loc="left", color=INK, fontsize=TITLE_SIZE,
                 weight="bold", pad=14)
    ax.axis("off")
    _color_key(ax, COLD_HOT, [(0, "colder"), (0.5, f"avg {avg:.0%}"), (1, "hotter")],
               note="faint where few shots were taken")


def draw_volume_map(ax, rows: list[dict]) -> None:
    """Where the shots come from: each shot warms the floor around it, red where most were taken."""
    volume, _, on_court = _shot_field(rows)
    level = volume / volume.max() if volume.max() > 0 else volume
    alpha = _fade(volume)
    fill = _hex_bgr(SURFACE) * (1 - alpha) + _interp(HEAT, level) * alpha
    fill[~on_court] = _hex_bgr(SURFACE)
    _, img = _court(fill)
    ax.imshow(img)
    ax.set_title("Where the shots come from", loc="left", color=INK,
                 fontsize=TITLE_SIZE, weight="bold", pad=14)
    ax.axis("off")
    _color_key(ax, [SURFACE, *HEAT], [(0, "fewer shots"), (1, "more shots")])


def _style(ax) -> None:
    for name in ("top", "right", "left"):
        ax.spines[name].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c3")
    ax.tick_params(colors=INK_2, length=0)
    ax.grid(axis="y", color="#e6e5e1", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])


def _bars(ax, labels, made, att) -> None:
    pct = [m / a if a else 0 for m, a in zip(made, att)]
    x = np.arange(len(labels))
    ax.bar(x, pct, width=0.6, color=BAR)
    ax.set_xlim(-0.6, max(len(labels), 4) - 0.4)   # a few sessions don't make giant bars
    for xi, p, m, a in zip(x, pct, made, att):
        if not a:
            ax.text(xi, 0.03, "no shots", ha="center", va="bottom", color=MUTED, fontsize=8)
            continue
        ax.text(xi, p + 0.03, f"{p:.0%}", ha="center", va="bottom",
                color=INK, fontsize=10, weight="bold")
        ax.text(xi, 0.03, f"{m}/{a}", ha="center", va="bottom",
                color="white" if p > 0.12 else INK_2, fontsize=9)
    ax.set_xticks(x, labels, fontsize=9)
    _style(ax)


def draw_sessions(ax, sessions: list[dict], labels: list[str]) -> None:
    labels = [label.replace(", ", "\n").replace(" (", "\n(") for label in labels]
    _bars(ax, labels, [s["makes"] for s in sessions], [s["attempts"] for s in sessions])
    ax.set_title("FG% by session", loc="left", color=INK, fontsize=TITLE_SIZE, weight="bold", pad=14)


def draw_distance(ax, rows: list[dict]) -> None:
    labels, made, att = [], [], []
    for lo, hi in DISTANCE_BINS_FT:
        mine = [r for r in rows if r["distance_ft"] != "" and lo <= r["distance_ft"] < hi]
        labels.append(f"{lo}+ ft" if hi > 50 else f"{lo}–{hi} ft")
        made.append(sum(r["result"] == "make" for r in mine))
        att.append(len(mine))
    _bars(ax, labels, made, att)
    ax.set_title("FG% by distance", loc="left", color=INK, fontsize=TITLE_SIZE, weight="bold", pad=14)


def headline(rows: list[dict], sessions: list[dict]) -> str:
    t = tally(rows)
    parts = [f"{t['makes']} of {t['attempts']} made ({_show(t['fg_pct'])})"]
    if t["3pt_att"]:
        parts.append(f"3PT {t['3pt_made']}/{t['3pt_att']}")
    parts.append(f"{len(sessions)} session{'s' * (len(sessions) != 1)}")
    return "   ·   ".join(parts)


def draw_thumbnails(fig, spec, sessions: list) -> None:
    """One frame per session, mid-shot, with its date and FG under the title."""
    cells = spec.subgridspec(len(sessions), 1, hspace=0.45)
    for k, sess in enumerate(sessions):
        ax = fig.add_subplot(cells[k])
        made = sum(1 for x in sess.shots if x.made)
        att = len(sess.shots)
        ax.set_title(sess.label, loc="left", color=INK, fontsize=14, weight="bold", pad=24)
        ax.text(0, 1.03, f"{made} of {att} made ({_show(_pct(made, att))})",
                transform=ax.transAxes, color=INK_2, fontsize=12, va="bottom")
        if sess.thumb is not None:
            ax.imshow(sess.thumb)
            shot = sess.thumb_shot
            ax.text(0.025, 0.05, f"shot {shot.number}, {shot.label.lower()}",
                    transform=ax.transAxes, color="white", fontsize=10, va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.55, lw=0))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)


def render_charts(rows: list[dict], sessions: list[dict], clips: list[Session], out: Path
                  ) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                         "savefig.facecolor": SURFACE, "font.size": 10})
    note = ("Court spots are estimated from one camera.  Generated "
            + datetime.now().strftime("%Y-%m-%d %H:%M"))
    written = []

    def save(fig, name):
        path = out / name
        fig.savefig(path, dpi=130, pil_kwargs={"quality": 92})
        plt.close(fig)
        written.append(path)

    labels = [c.label for c in clips]
    # One row: the two court views, then a frame from each session down the right.
    fig = plt.figure(figsize=(22, 8.8))
    grid = fig.add_gridspec(1, 2, wspace=0.06, left=0.03, right=0.71, top=0.7, bottom=0.1)
    draw_shot_chart(fig.add_subplot(grid[0]), rows)
    draw_heat_map(fig.add_subplot(grid[1]), rows)
    # The sessions run the full height on the right.
    draw_thumbnails(fig, fig.add_gridspec(1, 1, left=0.75, right=0.98, top=0.93,
                                          bottom=0.05)[0], clips)
    # The header: the overall FG% is the biggest thing on the page.
    t = tally(rows)
    days = sorted({x["date"] for x in sessions})
    span = days[0] if len(days) == 1 else f"{days[0]} to {days[-1]}"
    head = fig.add_axes([0.03, 0.80, 0.68, 0.18])
    head.axis("off")
    head.text(0, 1, f"Shooting history  ·  {len(sessions)} session"
              f"{'s' * (len(sessions) != 1)}, {span}", fontsize=18, color=INK_2,
              weight="bold", va="top", transform=head.transAxes)
    big = head.text(0, 0, _show(t["fg_pct"]), fontsize=72, weight="bold", color=INK,
                    va="bottom", transform=head.transAxes)
    head.annotate(f"{t['makes']} of {t['attempts']} made", xy=(1, 0), xycoords=big,
                  xytext=(18, 10), textcoords="offset points", fontsize=28, color=INK,
                  va="bottom")
    if t["3pt_att"]:
        head.annotate(f"3PT {t['3pt_made']} of {t['3pt_att']}", xy=(1, 0), xycoords=big,
                      xytext=(18, 52), textcoords="offset points", fontsize=16, color=INK_2,
                      va="bottom")
    fig.text(0.03, 0.02, note, fontsize=9, color=MUTED)
    save(fig, "report.jpg")

    for name, draw in (("shot_chart.jpg", draw_shot_chart), ("heat_map.jpg", draw_heat_map),
                       ("volume_map.jpg", draw_volume_map)):
        fig, ax = plt.subplots(figsize=(8, 6.6))
        fig.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.12)
        draw(ax, rows)
        fig.text(0.03, 0.015, note, fontsize=7, color=MUTED)
        save(fig, name)

    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5))
    fig.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.16, wspace=0.12)
    draw_sessions(a, sessions, labels)
    draw_distance(b, rows)
    save(fig, "trends.jpg")
    return written


# ── main ─────────────────────────────────────────────────────────────────────

def main_(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("inputs", nargs="*", type=Path, metavar="CLIP|DIR",
                        help=f"clips or folders (searched with subfolders); default {main.rel(cfg.INPUT)}")
    parser.add_argument("--cached-only", action="store_true",
                        help="never call the gateway; skip a clip whose replies aren't cached")
    parser.add_argument("--out", type=Path, default=HISTORY_DIR, help="where to write")
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore", category=RuntimeWarning)   # numpy on frames with no person

    clips = find_clips(args.inputs or [cfg.INPUT])
    if not clips:
        console.print(f"[red]No clips.[/] Put shooting videos in {main.rel(cfg.INPUT)}/")
        return 1

    api_key = ""
    if args.cached_only:
        main.client = _refuse
    else:
        try:
            api_key, _ = load_api_key(cfg.PROJECT_DIR)
        except Exception:   # fine while everything is cached
            main.client = _refuse

    # Clips cut from one recording share its time; their folder tells them apart.
    infos = {clip: capture_info(clip) for clip in clips}
    keep = sorted(clips, key=lambda c: (infos[c]["when"], rel(c)))
    shared = {c for c in keep for d in keep if d != c and infos[d]["when"] == infos[c]["when"]}

    sessions: list[Session] = []
    for clip in keep:
        meta = infos[clip]
        console.rule(f"[bold magenta]{session_label(meta['when'])}[/]  {rel(clip)}", align="left")
        try:
            info, shots, (thumb, thumb_shot) = analyze_clip(clip, api_key)
        except NotCached as exc:
            console.print(f"  [yellow]skipped[/]: not cached ({exc}); run without --cached-only")
            continue
        except Exception as exc:
            console.print(f"  [red]failed[/]: {str(exc).splitlines()[0][:120]}")
            continue
        made = sum(1 for s in shots if s.made)
        console.print(f"  {made}/{len(shots)} made")
        sessions.append(Session(clip, meta["when"], meta["source"], meta["camera"],
                                meta["duration"], info.fps, shots,
                                tag=(clip.parent.name if clip.parent != cfg.INPUT else clip.stem)
                                if clip in shared else "", thumb=thumb, thumb_shot=thumb_shot))

    rows = shot_rows(sessions)
    if not rows:
        console.print("[red]No shots to report.[/]")
        return 1
    srows = session_rows(sessions, rows)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("shots.csv", rows), ("sessions.csv", srows), ("zones.csv", zone_rows(rows))):
        write_csv(out / name, data)
    charts = render_charts(rows, srows, sessions, out)

    console.print()
    console.print(f"[bold]{headline(rows, srows)}[/]")
    for path in [*charts, out / "shots.csv", out / "sessions.csv", out / "zones.csv"]:
        console.print(f"  -> [dim]{main.rel(path)}[/]")
    return 0


if __name__ == "__main__":
    sys.exit(main_())
