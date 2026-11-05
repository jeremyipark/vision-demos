"""Does the ball's flight differ between makes and misses? Side-on shots only.

    python trajectory.py                  # every clip under data/input/, subfolders included
    python trajectory.py --cached-only    # never call the gateway; skip a clip that isn't cached

Only shots that cross the picture are used (rim.py's "across" view, the ones social.py and
the rim reel show): there the camera sees the arc's true shape. A shot toward or away from
the camera comes at the rim along the line of sight, and its arc is squashed.

For each shot, the tracked ball from the release to just before it reaches the rim (before
the rim, the backboard or the net can touch it) is fitted with a parabola: across the
picture at a steady speed, up and down with a constant pull. A few stray detections are
dropped from the fit. Then every arc is put on one scale: the release at the left, the
rim's center at (1, 0), so a shot from the right is mirrored and a long shot is shrunk to
the same width as a short one (the same factor both ways, so the angles are the real ones).

From each fit: the launch angle at release, the entry angle as it comes down to the rim's
height, how high the arc peaks over the rim, and where it reaches the rim's height, short
of the center (-) or past it (+), in rim radii. Makes and misses are compared as form.py
does: the median of each, and how often a random split of the same shots would differ as
much (a permutation test). A few misses make any difference a lead, not a finding.

Writes to data/output/trajectory/, replacing the last run:

    trajectory.jpg          every arc, makes against misses, and each measure
    trajectory_shots.csv    one row per side-on shot, with every measure
    trajectory_summary.csv  one row per measure: makes vs misses, the difference, the p-value
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path

import numpy as np

import config as cfg
import form
import history as hist
import main
import rim
from src.env import load_api_key
from src.shots import J

console = main.console
TRAJ_DIR = cfg.OUTPUT_DIR / "trajectory"
MIN_POINTS = 6          # tracked frames a fit needs
MIN_STEADY = 0.9        # the ball's pace across the picture, late over early, at least this
STRAY = 0.06            # a point further than this off the arc (rim widths) is dropped
X_GRID = np.linspace(0, 1, 101)   # the common scale the typical arcs are taken on

MEASURES = [
    # key, title, unit, what higher means
    ("launch_deg", "Launch angle", "°", "steeper off the hand"),
    ("entry_deg", "Entry angle", "°", "steeper into the rim"),
    ("apex_ft", "Arc peak over the rim", " ft", "a higher arc"),
    ("crossing_r", "Where it reached the rim", " r", "long (+), short (-)"),
]


# ── the fit ──────────────────────────────────────────────────────────────────

def flight(shot, hoop, r: dict) -> np.ndarray | None:
    """The ball's free flight in pixels, rows (frame, x, y): from the release until the rim
    could touch it. That's the first touch rim.py found, else the ball coming down to a
    ball's height over the rim, else the first rise after the peak (a bounce)."""
    p = np.array(shot.path, float)
    if len(p) < MIN_POINTS:
        return None
    ball_r = rim.BALL_R * hoop.width
    stop = r["frame"] if r.get("frame") is not None else math.inf
    apex = int(np.argmin(p[:, 2]))
    keep = []
    for i, (f, x, y) in enumerate(p):
        if f >= stop:
            break
        if i > apex and (y > hoop.top - ball_r or y < p[i - 1, 2] - 0.02 * hoop.width):
            break
        keep.append(i)
    return p[keep] if len(keep) >= MIN_POINTS else None


def steady(p: np.ndarray) -> float:
    """The ball's pace across the picture over the last third of its flight, over the first
    third. Side-on to the camera it holds (about 1); a shot that also travels away from the
    camera slows down in the picture as it goes, and its arc is squashed by perspective."""
    f, x = p[:, 0], p[:, 1]
    k = max(4, len(f) // 3)
    early, late = abs(np.polyfit(f[:k], x[:k], 1)[0]), abs(np.polyfit(f[-k:], x[-k:], 1)[0])
    return float(late / early) if early > 0 else math.nan


def fit(p: np.ndarray, hoop, fps: float) -> dict | None:
    """x(t) a line and y(t) a parabola. A stray SAM box (the hand, a second ball, the rim) is
    dropped: the point furthest off the fit, refitted, until every point is within
    STRAY of it. None when the arc doesn't hold together, or its pull isn't gravity's."""
    t, x, y = p[:, 0] - p[0, 0], p[:, 1], p[:, 2]
    use = np.ones(len(t), bool)
    while True:
        cx, cy = np.polyfit(t[use], x[use], 1), np.polyfit(t[use], y[use], 2)
        res = np.hypot(x - np.polyval(cx, t), y - np.polyval(cy, t))
        worst = int(np.argmax(np.where(use, res, -1)))
        if res[worst] <= STRAY * hoop.width:
            break
        use[worst] = False
        if use.sum() < MIN_POINTS:
            return None
    g = 9.81 * hoop.width / cfg.RIM_DIAMETER_M / fps ** 2      # gravity, px per frame²
    pull = 2 * cy[0] / g
    if not 0.7 < pull < 1.3:     # not a ball in free flight at the rim's distance
        return None
    rms = float(np.sqrt(np.mean(res[use] ** 2)) / hoop.width)
    return {"cx": cx, "cy": cy, "t0": p[0, 0], "rms": rms, "dropped": int((~use).sum()),
            "n": int(use.sum()), "gravity": float(pull)}


def hand_x(c: dict, shot) -> float:
    """Where the shooting hand is across the picture as the ball leaves it."""
    f = shot.release
    wrist = c["kp"][max(0, f - 3):f + 2, J[f"{c['side']}_wrist"], 0]
    return float(np.nanmedian(wrist)) if np.isfinite(wrist).any() else math.nan


def measures(fitted: dict, hoop, fps: float, hand: float) -> dict | None:
    """Angles, peak and where it reaches the rim's height, plus the arc on the common scale.
    The arc starts where the fitted flight passes the shooting hand: the release as the hand
    places it (the detected release frame can come a few frames late, the ball already up)."""
    cx, cy = fitted["cx"], fitted["cy"]
    t0 = (hand - cx[1]) / cx[0] if np.isfinite(hand) and cx[0] != 0 else 0.0
    if not -0.4 * fps < t0 < 0.4 * fps:   # the hand nowhere near the flight's start
        t0 = 0.0
    x0, y0 = np.polyval(cx, t0), np.polyval(cy, t0)
    side = 1.0 if hoop.cx >= x0 else -1.0          # +1: shot from the left of the picture
    span = abs(hoop.cx - x0)
    if span < 0.5 * hoop.width:
        return None
    # Coming down to the rim's height (y = hoop.cy): the later root.
    roots = np.roots([cy[0], cy[1], cy[2] - hoop.cy])
    roots = [float(z.real) for z in roots if abs(z.imag) < 1e-9 and z.real > t0]
    if not roots:
        return None
    tc = max(roots)
    vx = cx[0] * side                              # toward the rim, px per frame
    if vx <= 0:
        return None
    vy0 = -(2 * cy[0] * t0 + cy[1])                # upward is +
    vyc = -(2 * cy[0] * tc + cy[1])
    t_apex = -cy[1] / (2 * cy[0])
    m_per_px = cfg.RIM_DIAMETER_M / hoop.width
    xc = np.polyval(cx, tc)
    tt = np.linspace(t0, tc, 120)
    arc = np.column_stack([(np.polyval(cx, tt) - x0) * side / span,
                           (hoop.cy - np.polyval(cy, tt)) / span])
    return {
        "launch_deg": math.degrees(math.atan2(vy0, vx)),
        "entry_deg": math.degrees(math.atan2(-vyc, vx)),
        "apex_ft": (hoop.cy - np.polyval(cy, t_apex)) * m_per_px * hist.FT_PER_M,
        "crossing_r": (xc - hoop.cx) * side / hoop.width * rim.RIM_RADII,
        "release_below_rim_ft": (y0 - hoop.cy) * m_per_px * hist.FT_PER_M,
        "across_ft": span * m_per_px * hist.FT_PER_M,
        "flight_s": (tc - t0) / fps,
        "arc": arc,
        "rim_half": 0.5 * hoop.width / span,        # the rim's half width on the common scale
    }


def typical(arcs: list[np.ndarray]) -> np.ndarray:
    """The median arc: each arc's height at the same points across."""
    ys = np.array([np.interp(X_GRID, a[:, 0], a[:, 1], left=np.nan, right=np.nan) for a in arcs])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(ys, axis=0)


def compare(rows: list[dict]) -> list[dict]:
    out = []
    for key, title, unit, meaning in MEASURES:
        made = np.array([r[key] for r in rows if r["result"] == "make" and r[key] != ""], float)
        miss = np.array([r[key] for r in rows if r["result"] == "miss" and r[key] != ""], float)
        row = {"measure": title, "unit": unit.strip(), "higher_means": meaning,
               "makes_n": len(made), "makes_median": "", "misses_n": len(miss),
               "misses_median": "", "difference": "", "p_value": "", "read": "not enough shots"}
        if len(made) >= 2 and len(miss) >= 2:
            p = form.permutation_p(made, miss)
            digits = 2 if key in ("crossing_r",) else 1
            row.update(makes_median=round(float(np.median(made)), digits),
                       misses_median=round(float(np.median(miss)), digits),
                       difference=round(float(np.median(miss) - np.median(made)), digits),
                       p_value=round(p, 3),
                       read=(("misses differ" if len(miss) >= form.MIN_MISSES else
                              f"a lead, only {len(miss)} misses") if p < 0.05 else
                             "a hint, not proven" if p < 0.15 else "no clear difference"))
        out.append(row)
    return out


# ── chart ────────────────────────────────────────────────────────────────────

def render(rows: list[dict], arcs: list[tuple[str, np.ndarray]], summary: list[dict],
           n_all: int, rim_half: float, dst: Path, min_steady: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"figure.facecolor": hist.SURFACE, "axes.facecolor": hist.SURFACE,
                         "savefig.facecolor": hist.SURFACE, "font.size": 12})
    makes = sum(r["result"] == "make" for r in rows)
    misses = len(rows) - makes
    fig = plt.figure(figsize=(22, 12.5))
    fig.text(0.03, 0.965, "Trajectory: makes vs misses", fontsize=32, weight="bold",
             color=hist.INK, va="top")
    fig.text(0.03, 0.91, f"{len(rows)} side-on shots of {n_all}  ·  {makes} makes, {misses} misses"
             "  ·  each arc a parabola fitted to the tracked ball, release at left, rim's center at (1, 0)",
             fontsize=14, color=hist.INK_2, va="top")
    grid = fig.add_gridspec(2, 3, left=0.04, right=0.98, top=0.84, bottom=0.13,
                            width_ratios=[2.1, 1, 1], wspace=0.28, hspace=0.45)

    # Left: every arc on the common scale, and the typical make and miss.
    ax = fig.add_subplot(grid[:, 0])
    for result, arc in arcs:
        color = hist.MAKE if result == "make" else hist.MISS
        ax.plot(arc[:, 0], arc[:, 1], color=color, lw=1.3, alpha=0.35, zorder=2)
    for result, color, label in (("make", hist.MAKE, "typical make"),
                                 ("miss", hist.MISS, "typical miss")):
        group = [a for r, a in arcs if r == result]
        if len(group) >= 2:
            y = typical(group)
            ax.plot(X_GRID, y, color=hist.SURFACE, lw=7, zorder=3)
            ax.plot(X_GRID, y, color=color, lw=3.5, zorder=4,
                    ls="-" if result == "make" else (0, (6, 3)))
    ax.plot([1 - rim_half, 1 + rim_half], [0, 0], color=hist.RIM, lw=5,
            solid_capstyle="round", zorder=5)
    ax.text(1, -0.05, "rim", color=hist.RIM, ha="center", va="top", fontsize=13, weight="bold")
    ax.scatter([0], [np.median([a[0, 1] for _, a in arcs])], s=70, color=hist.INK_2, zorder=5)
    ax.text(0.01, np.median([a[0, 1] for _, a in arcs]) - 0.05, "release", color=hist.INK_2,
            va="top", fontsize=13)
    ax.set_aspect("equal")
    ax.set_anchor("N")
    ax.set_xlim(-0.05, 1.15)
    ax.grid(color="#e6e5e1", lw=0.8)
    form._plain(ax)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["release", "", "halfway", "", "rim"])
    ax.set_ylabel("height over the rim, in shot lengths", color=hist.INK_2, fontsize=12)
    ax.set_title("Every arc, shooter on the left", loc="left", fontsize=hist.TITLE_SIZE,
                 weight="bold", color=hist.INK, pad=12)
    ax.legend(handles=[Line2D([], [], color=hist.MAKE, lw=3.5, label=f"make ({makes}), typical bold"),
                       Line2D([], [], color=hist.MISS, lw=3.5, ls=(0, (6, 3)),
                              label=f"miss ({misses}), typical bold")],
              loc="upper right", frameon=False, fontsize=13)

    # Right: each measure, every shot a dot, the medians as bars.
    rng = np.random.default_rng(1)
    for i, ((key, title, unit, meaning), s) in enumerate(zip(MEASURES, summary)):
        bx = fig.add_subplot(grid[i // 2, 1 + i % 2])
        for k, (result, color, marker) in enumerate((("make", hist.MAKE, "o"),
                                                     ("miss", hist.MISS, "X"))):
            vals = np.array([r[key] for r in rows if r["result"] == result and r[key] != ""], float)
            if not len(vals):
                continue
            xs = k + rng.uniform(-0.13, 0.13, len(vals))
            bx.scatter(xs, vals, s=90, color=color, marker=marker, edgecolors=hist.SURFACE,
                       linewidths=1.2, zorder=3)
            med = float(np.median(vals))
            bx.plot([k - 0.3, k + 0.3], [med, med], color=hist.INK, lw=2.5, zorder=4)
            bx.text(k + 0.33, med, f"{med:.2f}" if key == "crossing_r" else f"{med:.1f}",
                    va="center", fontsize=12, color=hist.INK)
        if key == "crossing_r":
            bx.axhspan(-rim.CLEAN_R, rim.CLEAN_R, color=hist.MAKE, alpha=0.08, lw=0)
            bx.axhline(0, color="#9b9a95", lw=1)
        bx.set_xticks([0, 1], ["makes", "misses"], fontsize=13)
        bx.set_xlim(-0.5, 1.6)
        bx.grid(axis="y", color="#e6e5e1", lw=0.8)
        form._plain(bx)
        bx.set_title(f"{title}" + (f" ({unit.strip()})" if unit.strip() else ""), loc="left",
                     fontsize=16, weight="bold", color=hist.INK, pad=26)
        p = s["p_value"]
        bx.text(0, 1.03, f"{meaning}  ·  " + (f"p = {p}, {s['read']}" if p != "" else s["read"]),
                transform=bx.transAxes, fontsize=11.5, color=hist.INK_2)

    note = (
             f"Side-on: the ball crosses the picture and keeps its pace across it (late over early at least {min_steady:g}), "
             "so the arc's shape is seen as it is; a shot also heading away from the camera slows in the picture and is left out. The fit stops before the rim can touch the ball. Shot length: release to the rim's center, "
             "across the picture.\nA permutation test asks how often a random split of the same shots differs as much; "
             f"with fewer than {form.MIN_MISSES} misses a low p is a lead to watch, not a finding. "
             "Rim radii: 0 the center, ±1 the rim's inside edge; inside the green band the ball clears both edges.")
    import textwrap
    fig.text(0.03, 0.025, "\n".join(textwrap.wrap(note, 250)), fontsize=11.5, color=hist.INK_2,
             va="bottom")
    fig.savefig(dst, dpi=110, pil_kwargs={"quality": 92})
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main_(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("inputs", nargs="*", type=Path, metavar="CLIP|DIR",
                        help=f"clips or folders (searched with subfolders); default {main.rel(cfg.INPUT)}")
    parser.add_argument("--cached-only", action="store_true",
                        help="never call the gateway; skip a clip whose replies aren't cached")
    parser.add_argument("--min-steady", type=float, default=MIN_STEADY,
                        help="keep a shot whose pace across the picture, late over early, is at "
                             f"least this (default {MIN_STEADY}; lower lets in shots angled away)")
    parser.add_argument("--out", type=Path, default=TRAJ_DIR, help="where to write")
    args = parser.parse_args(argv)
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

    infos = {clip: hist.capture_info(clip) for clip in clips}
    clips.sort(key=lambda c: (infos[c]["when"], hist.rel(c)))
    shared = {c for c in clips for d in clips if d != c and infos[d]["when"] == infos[c]["when"]}

    rows, arcs, halves, n_all = [], [], [], 0
    for clip in clips:
        when = infos[clip]["when"]
        tag = (clip.parent.name if clip.parent != cfg.INPUT else clip.stem) if clip in shared else ""
        session = when.strftime("%Y-%m-%d %H:%M") + (f" ({tag})" if tag else "")
        console.rule(f"[bold magenta]{hist.session_label(when)}[/]  {hist.rel(clip)}", align="left")
        try:
            c = hist.load_clip(clip, api_key)
        except hist.NotCached as exc:
            console.print(f"  [yellow]skipped[/]: not cached ({exc}); run without --cached-only")
            continue
        fps, hoop = c["info"].fps, c["hoop"]
        kept = 0
        for shot in c["shots"]:
            n_all += 1
            if shot.made is None:
                continue
            r = rim.read_shot(shot, hoop, fps)
            if r["view"] != "across":
                continue
            p = flight(shot, hoop, r)
            pace = steady(p) if p is not None else math.nan
            if p is not None and not (args.min_steady <= pace <= 1 / args.min_steady):
                console.print(f"  shot {shot.number}: slows to {pace:.2f} of its pace across the picture "
                              "(heading away from the camera): left out")
                continue
            fitted = fit(p, hoop, fps) if p is not None else None
            m = measures(fitted, hoop, fps, hand_x(c, shot)) if fitted is not None else None
            if m is None:
                console.print(f"  shot {shot.number}: side-on, but too little of the flight tracked")
                continue
            kept += 1
            result = "make" if shot.made else "miss"
            arcs.append((result, m["arc"]))
            halves.append(m["rim_half"])
            rows.append({"session": session, "clip": hist.rel(clip), "shot": shot.number,
                         "video_time_s": round(shot.release / fps, 1), "result": result,
                         "outcome": r["outcome"],
                         **{k: form._cell(m[k], 2 if k == "crossing_r" else 1) for k, *_ in MEASURES},
                         "release_below_rim_ft": form._cell(m["release_below_rim_ft"], 1),
                         "across_ft": form._cell(m["across_ft"], 1),
                         "flight_s": form._cell(m["flight_s"], 2),
                         "steady_pace": round(pace, 2),
                         "fit_points": fitted["n"], "fit_dropped": fitted["dropped"],
                         "fit_rms_rim_widths": round(fitted["rms"], 3),
                         "fit_gravity": round(fitted["gravity"], 2)})
        console.print(f"  {kept} of {len(c['shots'])} shots side-on and fitted")

    if not rows:
        console.print("[red]No side-on shots to compare.[/]")
        return 1
    summary = compare(rows)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    hist.write_csv(out / "trajectory_shots.csv", [{k: v for k, v in r.items()} for r in rows])
    hist.write_csv(out / "trajectory_summary.csv", summary)
    render(rows, arcs, summary, n_all, float(np.median(halves)), out / "trajectory.jpg",
           args.min_steady)

    console.print()
    for s in summary:
        console.print(f"  {s['measure']:<28} misses {s['misses_median']!s:>6} vs makes "
                      f"{s['makes_median']!s:>6}   p={s['p_value']!s:<6} {s['read']}")
    for name in ("trajectory.jpg", "trajectory_shots.csv", "trajectory_summary.csv"):
        console.print(f"  -> {hist.rel(out / name)}")
    return 0


if __name__ == "__main__":
    sys.exit(main_())
