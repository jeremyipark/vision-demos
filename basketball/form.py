"""Does shooting form differ between makes and misses? Side-on shots only.

    python form.py                  # every clip under data/input/, subfolders included
    python form.py --cached-only    # never call the gateway; skip a clip that isn't cached
    python form.py --max-ratio 0.45 # stricter about what counts as side-on

Joint angles from one camera are only real when the body is side-on to it: facing
the camera, or facing away, the arm and legs bend toward the lens and their angles
in the picture shrink. So, like elbow.png, only shots taken side-on are used: the
shoulders look narrow next to the torso (shoulder width / torso length under
--max-ratio over the set and release, the same test elbow.png uses).

For each of those shots it measures the elbow at the set point and at release, the
knee at its deepest bend, the time from the dip to the release, how high the release
is, and whether the arm holds its follow-through, then compares makes with misses: the median of each, and how often a
random split of the same shots would differ as much (a permutation test). The
distance is shown too, as context: a miss may just be a longer shot.

Writes to data/output/form/, replacing the last run:

    form.jpg          the elbow through every shot, and each measure, makes against misses
    form_shots.csv    one row per side-on shot, with every measure
    form_summary.csv  one row per measure: makes vs misses, the difference, and the p-value
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

import config as cfg
import history as hist
import main
from src import shots as shots_mod
from src.env import load_api_key
from src.shots import J

console = main.console
FORM_DIR = cfg.OUTPUT_DIR / "form"
# Side-on cutoff for form. Lower than config's 0.55 (elbow.png): on these clips, shots with
# the back to the camera scored 0.50-0.55 and slipped through it; side-on ones score
# 0.05-0.2, angled ones up to ~0.46.
FORM_MAX_RATIO = 0.48
# The elbow plot's window around the release: wide enough for the set and the whole follow-through.
CURVE_BEFORE_S = 1.0
CURVE_AFTER_S = 1.2
CURVE_T = np.round(np.arange(-CURVE_BEFORE_S, CURVE_AFTER_S + 1e-9, 1 / 30), 4)   # common grid
MIN_MISSES = 5   # fewer, and even a low p is only a lead: one odd shot moves the median

# (key, title, unit, what a higher value means)
MEASURES = [
    ("elbow_set_deg", "Elbow at the set point", "°", "higher = less bent"),
    ("elbow_release_deg", "Elbow at release", "°", "180 = arm straight"),
    ("follow_through_deg", "Follow-through: elbow 0.3 s after release", "°",
     "higher = arm still up and straight"),
    ("knee_bend_deg", "Knee at its deepest bend", "°", "lower = deeper bend"),
    ("dip_to_release_s", "Dip to release", "s", "how long the shot takes"),
    ("release_height", "Release height", "", "wrist over the nose, in torso lengths"),
    ("distance_ft", "Distance (context, not form)", " ft", ""),
]


# ── measures ─────────────────────────────────────────────────────────────────

def joint_angle(kp: np.ndarray, a: str, b: str, c: str) -> np.ndarray:
    """The angle at joint *b* per frame, degrees (180 = straight)."""
    u, v = kp[:, J[a]] - kp[:, J[b]], kp[:, J[c]] - kp[:, J[b]]
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (u * v).sum(axis=1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
        return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def measure(c: dict, shot, max_ratio: float) -> dict | None:
    """One shot's form, or None when it wasn't side-on."""
    kp, side, angle, fps = c["kp"], c["side"], c["angle"], c["info"].fps
    n = len(angle)
    lo = max(0, shot.start - int(cfg.SET_WINDOW_SECONDS * fps))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ratio = float(np.nanmedian(shots_mod.shoulder_ratio(kp)[lo:shot.release + 1]))
    if not np.isfinite(ratio) or ratio >= max_ratio:
        return None
    # Release as elbow.png draws it: the arm's fullest extension just after the ball leaves.
    after = angle[shot.release:min(n, shot.release + int(0.2 * fps) + 1)]
    peak = shot.release + (int(np.nanargmax(after)) if np.isfinite(after).any() else 0)
    set_lo = max(0, shot.start - int(cfg.SET_WINDOW_SECONDS * fps))
    before = angle[set_lo:shot.release + 1]
    elbow_set = float(np.nanmin(before)) if np.isfinite(before).any() else math.nan
    set_frame = set_lo + int(np.nanargmin(before)) if np.isfinite(before).any() else None
    elbow_release = float(np.nanmax(after)) if np.isfinite(after).any() else math.nan
    # The knee over the gather: from the dip (or 0.6 s before the hands go up) to the release.
    start = shot.motion if shot.motion is not None else shot.start - int(0.6 * fps)
    knee = joint_angle(kp, f"{side}_hip", f"{side}_knee", f"{side}_ankle")[max(0, start):shot.release + 1]
    knee_bend = float(np.nanmin(knee)) if np.isfinite(knee).any() else math.nan
    # Follow-through: is the arm still extended 0.3 s on? (median over 0.25-0.35 s, for gaps)
    hold = angle[min(n, peak + int(0.25 * fps)):min(n, peak + int(0.35 * fps) + 1)]
    follow = float(np.nanmedian(hold)) if np.isfinite(hold).any() else math.nan
    # Release height: the wrist over the nose, in torso lengths, so it doesn't change with distance.
    f = min(peak, n - 1)
    torso = np.linalg.norm((kp[f, J["left_shoulder"]] + kp[f, J["right_shoulder"]]) / 2
                           - (kp[f, J["left_hip"]] + kp[f, J["right_hip"]]) / 2)
    height = (kp[f, J["nose"], 1] - kp[f, J[f"{side}_wrist"], 1]) / torso if torso else math.nan
    return {
        "release_frame": peak, "set_frame": set_frame,   # for highlights.py
        "side_on_ratio": round(ratio, 2),
        "elbow_set_deg": elbow_set,
        "elbow_release_deg": elbow_release,
        "knee_bend_deg": knee_bend,
        "dip_to_release_s": (shot.release - shot.motion) / fps if shot.motion is not None else math.nan,
        "release_height": float(height),
        "follow_through_deg": follow,
        "distance_ft": shot.distance_m * hist.FT_PER_M if shot.distance_m is not None else math.nan,
    }


def curve(raw: np.ndarray, release: int, fps: float) -> np.ndarray:
    """The elbow on CURVE_T (seconds from release), from a clip at *fps*; NaN off the clip's ends."""
    lo = max(0, release - int(CURVE_BEFORE_S * fps) - 1)
    hi = min(len(raw), release + int(CURVE_AFTER_S * fps) + 2)
    t = (np.arange(lo, hi) - release) / fps
    y = raw[lo:hi]
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return np.full(len(CURVE_T), np.nan)
    out = np.interp(CURVE_T, t[ok], y[ok])
    out[(CURVE_T < t[0]) | (CURVE_T > t[-1])] = np.nan
    return out


# ── makes vs misses ──────────────────────────────────────────────────────────

def permutation_p(makes: np.ndarray, misses: np.ndarray, *, draws: int = 20000,
                  seed: int = 0) -> float:
    """Two-sided: how often relabeling the same shots at random gives a median gap at least
    this big. Exact (every relabeling) when there are few enough."""
    pooled = np.concatenate([makes, misses])
    k, n = len(misses), len(pooled)
    observed = abs(np.median(makes) - np.median(misses))
    if math.comb(n, k) <= draws:
        splits = (np.array(c) for c in itertools.combinations(range(n), k))
    else:
        rng = np.random.default_rng(seed)
        splits = (rng.choice(n, k, replace=False) for _ in range(draws))
    hits = total = 0
    for idx in splits:
        mask = np.zeros(n, bool)
        mask[idx] = True
        gap = abs(np.median(pooled[~mask]) - np.median(pooled[mask]))
        hits += gap >= observed - 1e-9
        total += 1
    return hits / total


def compare(rows: list[dict]) -> list[dict]:
    out = []
    for key, title, unit, meaning in MEASURES:
        made = np.array([r[key] for r in rows if r["result"] == "make" and r[key] != ""], float)
        miss = np.array([r[key] for r in rows if r["result"] == "miss" and r[key] != ""], float)
        row = {"measure": title, "unit": unit.strip(), "higher_means": meaning,
               "makes_n": len(made), "makes_median": "", "misses_n": len(miss),
               "misses_median": "", "difference": "", "p_value": "", "read": "not enough shots"}
        if len(made) >= 2 and len(miss) >= 2:
            p = permutation_p(made, miss)
            digits = 2 if key in ("dip_to_release_s", "release_height") else 1
            row.update(makes_median=round(float(np.median(made)), digits),
                       misses_median=round(float(np.median(miss)), digits),
                       difference=round(float(np.median(miss) - np.median(made)), digits),
                       p_value=round(p, 3),
                       read=(("misses differ" if len(miss) >= MIN_MISSES else
                              f"a lead, only {len(miss)} misses") if p < 0.05 else
                             "a hint, not proven" if p < 0.15 else "no clear difference"))
        out.append(row)
    return out


# ── chart ────────────────────────────────────────────────────────────────────

def render(rows: list[dict], summary: list[dict], curves: list[tuple], n_all: int, dst: Path,
           max_ratio: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.facecolor": hist.SURFACE, "axes.facecolor": hist.SURFACE,
                         "savefig.facecolor": hist.SURFACE, "font.size": 11})
    makes = sum(r["result"] == "make" for r in rows)
    misses = sum(r["result"] == "miss" for r in rows)
    fig = plt.figure(figsize=(22, 13.5))
    fig.text(0.03, 0.97, "Form: makes vs misses", fontsize=32, weight="bold",
             color=hist.INK, va="top")
    fig.text(0.03, 0.915, f"{len(rows)} side-on shots of {n_all}  ·  {makes} makes, "
             f"{misses} misses  ·  shots facing toward or away from the camera left out",
             fontsize=17, color=hist.INK_2, va="top")

    # Left: the elbow through every shot, lined up on the release.
    ax = fig.add_axes([0.04, 0.1, 0.38, 0.72])
    # Each shot faint; over them, each group's median with a band of one standard deviation.
    for result, color in (("make", hist.MAKE), ("miss", hist.MISS)):
        mine = np.array([y for r, _, y in curves if r == result])
        if not len(mine):
            continue
        for y in mine:
            ax.plot(CURVE_T, y, color=color, lw=1, alpha=0.2)
        med = np.nanmedian(mine, axis=0)
        sd = np.nanstd(mine, axis=0)
        ax.fill_between(CURVE_T, med - sd, med + sd, color=color, alpha=0.16, lw=0)
        n = len(mine)
        rough = ", rough" if result == "miss" and n < MIN_MISSES else ""
        ax.plot(CURVE_T, med, color=color, lw=3.2,
                label=f"{'makes' if result == 'make' else 'misses'}: median ± 1 SD "
                      f"({n} shot{'s' * (n != 1)}{rough})")
    ax.axvline(0, color=hist.MUTED, lw=1)
    ax.text(0.01, 4, "release", color=hist.INK_2, fontsize=11)
    ax.set_ylim(0, 190)
    ax.set_xlim(CURVE_T[0], CURVE_T[-1])
    ax.set_xlabel("seconds from release", color=hist.INK_2, fontsize=12)
    ax.set_ylabel("elbow angle (180 = straight)", color=hist.INK_2, fontsize=12)
    ax.set_title("Elbow through the shot: set, release, follow-through", loc="left", fontsize=hist.TITLE_SIZE, weight="bold",
                 color=hist.INK, pad=14)
    ax.legend(loc="lower left", frameon=False, fontsize=13, labelcolor=hist.INK_2)
    _plain(ax)
    ax.grid(axis="y", color="#e6e5e1", lw=0.8)

    # Right: each measure, every shot as a dot, makes on top and misses below.
    grid = fig.add_gridspec(4, 2, left=0.49, right=0.98, top=0.8, bottom=0.08,
                            hspace=1.05, wspace=0.18)
    rng = np.random.default_rng(1)
    for k, ((key, title, unit, meaning), s) in enumerate(zip(MEASURES, summary)):
        a = fig.add_subplot(grid[k // 2, k % 2])
        for y, result, color, marker in ((1, "make", hist.MAKE, "o"), (0, "miss", hist.MISS, "X")):
            vals = np.array([r[key] for r in rows if r["result"] == result and r[key] != ""], float)
            if len(vals):
                a.scatter(vals, y + rng.uniform(-0.12, 0.12, len(vals)), s=90, c=color,
                          marker=marker, alpha=0.85, linewidths=1.5, edgecolors=hist.SURFACE,
                          zorder=3)
                med = np.median(vals)
                a.plot([med, med], [y - 0.3, y + 0.3], color=hist.INK, lw=2.5, zorder=4)
        a.set_ylim(-0.6, 1.6)
        a.set_yticks([0, 1], ["misses", "makes"], fontsize=12)
        a.set_title(title, loc="left", fontsize=16, weight="bold", color=hist.INK, pad=26)
        line = (f"misses {s['misses_median']}{unit} vs makes {s['makes_median']}{unit}"
                f"  ·  {s['read']} (p = {s['p_value']})" if s["p_value"] != "" else s["read"])
        a.text(0, 1.04, line, transform=a.transAxes, fontsize=11.5, color=hist.INK_2,
               va="bottom")
        if meaning:
            a.set_xlabel(meaning, fontsize=10.5, color=hist.MUTED)
        _plain(a)
        a.grid(axis="x", color="#e6e5e1", lw=0.8)

    fig.text(0.03, 0.02,
             f"Black ticks are medians. p is how often a random relabeling of the same shots differs "
             f"as much (low = less likely chance). With {misses} misses, treat any gap as a lead "
             f"to watch, not a cause. Side-on: shoulder width under {max_ratio:g} of the torso.  "
             f"Generated {datetime.now():%Y-%m-%d %H:%M}",
             fontsize=10.5, color=hist.MUTED)
    fig.savefig(dst, dpi=130, pil_kwargs={"quality": 92})
    plt.close(fig)


def _plain(ax) -> None:
    for name in ("top", "right", "left"):
        ax.spines[name].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c3")
    ax.tick_params(colors=hist.INK_2, length=0)
    ax.set_axisbelow(True)


# ── main ─────────────────────────────────────────────────────────────────────

def _cell(value, digits: int = 1):
    return "" if value is None or not np.isfinite(value) else round(float(value), digits)


def main_(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("inputs", nargs="*", type=Path, metavar="CLIP|DIR",
                        help=f"clips or folders (searched with subfolders); default {main.rel(cfg.INPUT)}")
    parser.add_argument("--cached-only", action="store_true",
                        help="never call the gateway; skip a clip whose replies aren't cached")
    parser.add_argument("--max-ratio", type=float, default=FORM_MAX_RATIO,
                        help="shoulder width / torso under this counts as side-on "
                             f"(default {FORM_MAX_RATIO}; lower is stricter)")
    parser.add_argument("--out", type=Path, default=FORM_DIR, help="where to write")
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

    rows, curves, n_all = [], [], 0
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
        fps = c["info"].fps
        # The plot uses the arm's angle as detected on every frame (as highlights.py shows it),
        # so no curve has gaps; a frame with no arm at all is bridged.
        raw = joint_angle(c["kp"], f"{c['side']}_shoulder", f"{c['side']}_elbow",
                          f"{c['side']}_wrist")
        kept = 0
        for shot in c["shots"]:
            n_all += 1
            if shot.made is None:
                continue
            m = measure(c, shot, args.max_ratio)
            if m is None:
                continue
            kept += 1
            result = "make" if shot.made else "miss"
            rows.append({"session": session, "clip": hist.rel(clip), "shot": shot.number,
                         "video_time_s": round(shot.release / fps, 1), "result": result,
                         "side_on_ratio": m["side_on_ratio"],
                         **{k: _cell(m[k], 2 if k in ("dip_to_release_s", "release_height") else 1)
                            for k, *_ in MEASURES}})
            curves.append((result, CURVE_T, curve(raw, shot.release, fps)))
        console.print(f"  {kept} of {len(c['shots'])} shots side-on")

    if not rows:
        console.print("[red]No side-on shots to compare.[/]")
        return 1
    summary = compare(rows)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    hist.write_csv(out / "form_shots.csv", rows)
    hist.write_csv(out / "form_summary.csv", summary)
    render(rows, summary, curves, n_all, out / "form.jpg", args.max_ratio)

    console.print()
    for s in summary:
        console.print(f"  {s['measure']:<32} misses {s['misses_median']!s:>6} vs makes "
                      f"{s['makes_median']!s:>6}   p={s['p_value']!s:<6} {s['read']}")
    for name in ("form.jpg", "form_shots.csv", "form_summary.csv"):
        console.print(f"  -> [dim]{main.rel(out / name)}[/]")
    return 0


if __name__ == "__main__":
    sys.exit(main_())
