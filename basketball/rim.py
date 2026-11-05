"""Where the ball meets the rim: swish, or front rim, back rim, side rim.

    python rim.py                  # every clip under data/input/, subfolders included
    python rim.py --cached-only    # never call the gateway; skip a clip that isn't cached

The camera faces the backboard, and the shots come across the frame from the side, so
left and right in the picture are the rim's front and back as the shooter sees them:
short or long. (Its left and right, toward and
away from the lens, can't be seen.) Every shot is turned so the shooter is on the left,
and the ball's path is measured in rim widths from the rim's center.

    rim contact    the first frame the ball leaves its free flight at the rim. In the air
                   only gravity acts on it, so a parabola fitted to its fall from the top
                   of the arc down to the rim says where it would be untouched; the first
                   frame it strays from that path by OFF_PATH (OFF_PATH_LOW once its
                   center is below the rim line, where the net slows it too), while the
                   top of the ball is still above the rim, is the first touch: dated to
                   when the stray began, and placed where the untouched ball met the rim
    swish          a make with no rim contact
    crossing       where the ball came down through the rim's height

A shot taken toward or away from the camera comes at the rim along the line of sight, so
its front and back can't be told apart; those are counted, not charted.

Writes to data/output/rim/, replacing the last run:

    rim.jpg        a close-up of the hoop face-on, every shot turned so the shooter is on
                   the left: where the ball touched it, and
                   where the clean makes went through
    rim_shots.csv  one row per shot: what it touched, where, and how it was called
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

import config as cfg
import history as hist
import main
from src import court
from src.env import load_api_key

console = main.console
RIM_DIR = cfg.OUTPUT_DIR / "rim"

BALL_R = cfg.BALL_DIAMETER_M / cfg.RIM_DIAMETER_M / 2   # ball radius in rim widths (~0.24)
ACROSS_MIN_SLOPE = 0.3     # sideways travel per unit of drop, coming in: below this, toward the camera
OFF_PATH = 0.15            # rim widths off its free flight: a touch (the fit itself holds to ~0.01)
OFF_PATH_ONSET = 0.06      # the touch is dated back a frame if the stray had passed this
NET_BELOW = 0.08           # center this far below the rim line: the net slows it too ...
OFF_PATH_LOW = 0.30        # ... so there a touch must stray this far (a swish strays ~0.1-0.17)
FRONT_BACK = 0.12          # a contact this far from center is front or back rim; nearer, a side
SIDE = "side rim"          # the rim's left or right, toward or away from the camera: which, unseen
RIM_IN = "#7fc97f"         # a make that touched the rim first: a lighter green
# Long or short is relative to the rim, in rim radii: 0 its center, 1 its inside edge.
INNER_R_M = 0.457 / 2                                   # the rim's inside radius
RIM_RADII = cfg.RIM_DIAMETER_M / INNER_R_M              # one rim width (the mask's) in rim radii
CLEAN_R = 1 - cfg.BALL_DIAMETER_M / 0.457               # the ball's center within this (~0.48) clears
                                                        # the rim's near and far edges
HIT_R = 1 + cfg.BALL_DIAMETER_M / 0.457                 # ...and within this (~1.52) some of the
                                                        # ball still meets the rim: past it, none


def off_word(r: float) -> str:
    """How far off center the ball's center was, in words, whichever way: on line (the clean
    window), off center (some of the ball meets the rim), way off (clear of the rim)."""
    if not np.isfinite(r):
        return ""
    return "on line" if abs(r) <= CLEAN_R else "off center" if abs(r) <= HIT_R else "way off"


def depth_word(r: float) -> str:
    """Long or short in words, relative to the rim: on line (the clean window), long or short
    (still inside the rim), way long or way short (past its edge)."""
    if not np.isfinite(r):
        return ""
    if abs(r) <= CLEAN_R:
        return "on line"
    return ("way " if abs(r) > 1 else "") + ("long" if r > 0 else "short")


# ── the ball at the rim ──────────────────────────────────────────────────────

def read_shot(shot, hoop, fps: float) -> dict:
    """What the ball did at the rim, in rim widths: u front (-) to back (+), v down (+)."""
    out = {"view": "", "outcome": "", "u": math.nan, "v": math.nan, "off_path": math.nan,
           "cross_u": math.nan, "frame": None, "px": None,
           "mirror": 1.0,   # -1: the shot came from the right and u, cross_u were flipped
           "aim_u": math.nan, "long": math.nan,   # where the free flight crossed the rim's height,
                                                  # in rim radii past (+) or before (-) its center
           "touches": []}   # every bounce off the rim, the first included: (frame, x, y) in pixels
    p = np.array(shot.path, float)
    if len(p) < 4:
        out["view"] = "ball not tracked"
        return out
    f, u, v = p[:, 0], (p[:, 1] - hoop.cx) / hoop.width, (p[:, 2] - hoop.cy) / hoop.width
    apex = int(np.argmin(v))

    # Coming down from the apex to just over the rim: how far it travels sideways per unit dropped.
    down = []
    for i in range(apex, len(f)):
        # Stop at the rim, or as soon as it rises again (a bounce), before anything turns it.
        if v[i] > -0.3 or (i > apex and v[i] < v[i - 1] - 0.02):
            break
        if v[i] >= -1.2 and abs(u[i]) < 2.5:
            down.append(i)
    if len(down) >= 2 and v[down[-1]] > v[down[0]]:
        slope = abs(u[down[-1]] - u[down[0]]) / (v[down[-1]] - v[down[0]])
        side = u[down[-1]] - u[down[0]]
    else:
        slope, side = 0.0, 0.0
    if slope < ACROSS_MIN_SLOPE:
        out["view"] = "toward the camera"
        return out
    out["view"] = "across"
    mirror = -1.0 if side < 0 else 1.0   # it came from the right: mirror, so the shooter is on the left
    out["mirror"] = mirror
    u = u * mirror

    # Free flight, from the top of the arc down to the rim (the ball's bottom about to reach
    # the rim's top), fitted with gravity fixed; then the first frame it strays from that path.
    rim_h = (hoop.cy - hoop.top) / hoop.width          # the rim ellipse's half height
    clear = -(rim_h + 2 * BALL_R)
    arrive = next((i for i in range(1, len(f)) if v[i] >= clear and v[i] > v[i - 1]), len(f))
    top = int(np.argmin(v[:arrive])) if arrive else 0
    fit = [i for i in range(top, arrive) if i == top or f[i] - f[i - 1] <= 2][-6:]
    if len(fit) >= 3:
        g = 9.81 / cfg.RIM_DIAMETER_M / fps ** 2       # gravity, rim widths per frame²
        t = f[fit]
        cu = np.polyfit(t, u[fit], 1)
        cv = np.polyfit(t, v[fit] - 0.5 * g * t ** 2, 1)
        dev = np.hypot(u - np.polyval(cu, f), v - (np.polyval(cv, f) + 0.5 * g * f ** 2))
        def flight(frame):                              # the untouched ball, at a frame
            return np.polyval(cu, frame), np.polyval(cv, frame) + 0.5 * g * frame ** 2

        # Where it was headed: the free flight carried down to the rim's height, whatever the
        # rim then did to it. Past the rim's center is long (+), before it short (-), as the
        # shooter sees it, from whichever side they shot.
        roots = np.roots([0.5 * g, cv[0], cv[1]])
        roots = [float(x.real) for x in roots if abs(x.imag) < 1e-9 and x.real > f[fit[-1]] - 2]
        if roots:
            aim = float(np.polyval(cu, min(roots)))
            out.update(aim_u=aim, long=aim * RIM_RADII)

        for i in range(fit[-1] + 1, len(f)):
            # Look on while the top of the ball is still above the rim: it can hit the rim
            # until then. Below the rim line the net slows it too, so a touch must stray further.
            if v[i] > BALL_R:
                break                                   # through the rim's height untouched
            if dev[i] < (OFF_PATH if v[i] <= NET_BELOW else OFF_PATH_LOW):
                continue
            # When it touched: the frame before if it had already begun to stray; the first
            # untracked frame if the ball was lost right there (hidden by the rim as it hit).
            if f[i] - f[i - 1] > 1 and dev[i - 1] < OFF_PATH_ONSET:
                k_frame = f[i - 1] + 1
            elif i - 1 > fit[-1] and dev[i - 1] >= OFF_PATH_ONSET:
                k_frame = f[i - 1]
            else:
                k_frame = f[i]
            # Where: the free flight's ball at that moment, the spot where it met the rim.
            ku, kv = flight(k_frame)
            out.update(u=float(ku), v=float(kv), off_path=float(dev[i]), frame=int(k_frame),
                       px=(float(hoop.cx + ku * mirror * hoop.width), float(hoop.cy + kv * hoop.width)))
            # And every bounce after it: off the back rim and onto the front, say.
            hits = [(int(k_frame), float(ku), float(kv))] + later_touches(f, u, v, (k_frame, ku, kv), g, rim_h)
            out["touches"] = [(fr, float(hoop.cx + tu * mirror * hoop.width), float(hoop.cy + tv * hoop.width))
                              for fr, tu, tv in hits]
            break

    # The first time down through the rim's height; not after a touch, where it's no entry.
    for i in range(apex, len(f) - 1 if not np.isfinite(out["u"]) else apex):
        if v[i] < 0 <= v[i + 1]:
            t = -v[i] / (v[i + 1] - v[i])
            out["cross_u"] = float(u[i] + t * (u[i + 1] - u[i]))
            break

    touched = np.isfinite(out["u"])
    if shot.made:
        out["outcome"] = "rim, then in" if touched else "swish"
    elif touched:
        out["outcome"] = ("front rim" if out["u"] < -FRONT_BACK else
                          "back rim" if out["u"] > FRONT_BACK else SIDE)
    else:
        cu = out["cross_u"]
        out["outcome"] = ("short, no rim" if np.isfinite(cu) and cu < 0 else
                          "long, no rim" if np.isfinite(cu) else "no rim contact seen")
    return out


def later_touches(f, u, v, start, g: float, rim_h: float, most: int = 4) -> list[tuple[int, float, float]]:
    """The bounces after the first touch, each ``(frame, u, v)``. Between two touches the ball
    is in free flight again, so the same test as the first one holds: fit the flight with
    gravity fixed, from the last touch on, and the first frame it strays from it is the next.
    It's placed where the ball was last seen on that flight. Only while the ball's top is above the rim, and only a stray where the ball could reach
    the rim's metal; and a touch needs a couple of frames of flight after the last, so a ball
    rolling on the rim isn't a zigzag of them."""
    reach_u = 0.5 + BALL_R + 0.1                       # rim's edge, plus the ball, plus slack
    reach_v = rim_h + BALL_R + 0.1
    out = []
    while len(out) < most:
        cf, cu0, cv0 = start
        seg = [(float(cf), float(cu0), float(cv0))]   # the flight since the last touch
        later = [i for i in range(len(f)) if f[i] > cf]
        found = None
        for n, i in enumerate(later):
            if v[i] > BALL_R:
                return out                             # its top under the rim: done with it
            if len(seg) >= 3:
                t = np.array([q[0] for q in seg])
                cu = np.polyfit(t, [q[1] for q in seg], 1)
                cv = np.polyfit(t, np.array([q[2] for q in seg]) - 0.5 * g * t ** 2, 1)
                def flight(fr, cu=cu, cv=cv):
                    return float(np.polyval(cu, fr)), float(np.polyval(cv, fr) + 0.5 * g * fr ** 2)
                pu, pv = flight(f[i])
                dev = math.hypot(u[i] - pu, v[i] - pv)
                if dev >= OFF_PATH_ONSET:
                    # Begun to stray: a touch if it goes on to OFF_PATH within a few frames.
                    ahead = later[n:n + 3]
                    if any(v[j] <= BALL_R and math.hypot(u[j] - flight(f[j])[0], v[j] - flight(f[j])[1])
                           >= (OFF_PATH if v[j] <= NET_BELOW else OFF_PATH_LOW) for j in ahead):
                        # Where: the ball as last seen on its path, the frame before it strayed
                        # (it met the rim between that frame and this). Lost in between: where
                        # the flight had it a frame later.
                        prev = later[n - 1]
                        if f[i] - f[prev] > 1:
                            k = int(f[prev]) + 1
                            ku, kv = flight(k)
                        else:
                            k, ku, kv = int(f[prev]), float(u[prev]), float(v[prev])
                        if abs(ku) <= reach_u and abs(kv) <= reach_v and k > cf:
                            found = (k, ku, kv)
                        break                          # strayed, near the rim or not: the flight is over
            seg.append((float(f[i]), float(u[i]), float(v[i])))
        if found is None:
            return out
        out.append(found)
        start = found
    return out


def on_rim(u: float, v: float, rim_h: float) -> tuple[float, float]:
    """The point on the rim (as the camera sees it: an ellipse 1 wide, 2 * rim_h tall)
    nearest the ball's center: where the ball met it."""
    t = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    eu, ev = 0.5 * np.cos(t), rim_h * np.sin(t)
    k = int(np.argmin((eu - u) ** 2 + (ev - v) ** 2))
    return float(eu[k]), float(ev[k])


def board_outline(c: dict) -> np.ndarray | None:
    """The backboard's outline from the clip, in rim widths from the rim's center (u right, v
    down): the light board against what's behind it, around the rim, its convex hull (the
    rim and net in front of it don't dent it). None when no board-sized shape is found."""
    import cv2
    from src import hoop as hoop_mod

    h = c["hoop"]
    w = h.width
    img = hoop_mod.read_frame(c["mp4"], h.frame)
    x0, y0 = max(0, int(h.cx - 2.6 * w)), max(0, int(h.cy - 2.8 * w))
    crop = img[y0:int(h.cy + 1.2 * w), x0:int(h.cx + 2.6 * w)]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 2] > 165) & (hsv[..., 1] < 70)).astype(np.uint8)
    k = max(3, int(w * 0.07)) | 1                      # wide enough to cut the net's strings off
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    _, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    sx, sy = int(h.cx - x0), int(h.cy - 0.7 * w - y0)   # a point on the board, just above the rim
    if not (0 <= sy < lab.shape[0] and 0 <= sx < lab.shape[1]) or lab[sy, sx] == 0:
        return None
    comp = lab[sy, sx]
    area = stats[comp, 4] / w ** 2
    if not 1.0 < area < 8.0:                            # rim widths²: not a backboard
        return None
    cnts, _ = cv2.findContours((lab == comp).astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))[:, 0, :].astype(float)
    for _ in range(4):   # round off the hull's corners (Chaikin): the board's curve, not facets
        nxt = np.roll(hull, -1, axis=0)
        hull = np.stack([0.75 * hull + 0.25 * nxt, 0.25 * hull + 0.75 * nxt], axis=1).reshape(-1, 2)
    return tidy_board(np.column_stack([(hull[:, 0] + x0 - h.cx) / w, (hull[:, 1] + y0 - h.cy) / w]))


def tidy_board(outline: np.ndarray) -> np.ndarray:
    """The traced outline with its bottom edge level. The trace's bottom runs a little diagonal
    (the net, the pole and shadows meet the board there, and the phone leans a touch), where a
    board's bottom is flat: it's cut level at the bottom edge's highest point across the middle,
    and the rest of the shape is left as traced."""
    u, v = outline[:, 0], outline[:, 1]
    edges = [v[(u >= x) & (u < x + 0.1)].max() for x in np.arange(-0.6, 0.6, 0.1)
             if ((u >= x) & (u < x + 0.1)).any()]
    if not edges:
        return outline
    return np.column_stack([u, np.minimum(v, min(edges))])


def default_board() -> np.ndarray:
    """A fan-shaped board, for when none is traced."""
    t = np.linspace(0, np.pi, 60)
    return np.array([(1.3, 0.45)] + [(1.3 * np.cos(a), 0.45 - 1.55 * np.sin(a)) for a in t]
                    + [(-1.3, 0.45)])


# ── chart ────────────────────────────────────────────────────────────────────

def render(rows: list[dict], rim_h: float, net_depth: float, dst: Path, board=None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    plt.rcParams.update({"figure.facecolor": hist.SURFACE, "axes.facecolor": hist.SURFACE,
                         "savefig.facecolor": hist.SURFACE, "font.size": 12})
    across = [r for r in rows if r["view"] == "across"]
    makes = [r for r in across if r["result"] == "make"]
    swish = [r for r in makes if r["outcome"] == "swish"]
    rimmed = [r for r in across if r["rim_u"] != ""]
    hidden = len(rows) - len(across)

    fig = plt.figure(figsize=(19, 12))
    fig.text(0.04, 0.965, "At the rim", fontsize=32, weight="bold", color=hist.INK, va="top")
    pct = f"{len(swish) / len(makes):.0%}" if makes else "—"
    big = fig.text(0.04, 0.89, pct, fontsize=64, weight="bold", color=hist.INK, va="top")
    fig.canvas.draw()
    bb = big.get_window_extent().transformed(fig.transFigure.inverted())
    fig.text(bb.x1 + 0.015, 0.875, f"swish: {len(swish)} of {len(makes)} makes",
             fontsize=24, color=hist.INK, va="top")
    fig.text(bb.x1 + 0.015, 0.825,
             f"{len(rimmed)} of {len(across)} shots touched the rim  ·  "
             f"{len(across)} shots seen from the side"
             + (f", {hidden} toward the camera left out" if hidden else ""),
             fontsize=15, color=hist.INK_2, va="top")

    # The close-up: the hoop face-on, as the camera sees it, but with every shot turned so
    # the shooter is on the left: shots from both sides, front and back on the same sides.
    ax = fig.add_axes([0.03, 0.07, 0.5, 0.64])
    ax.set_aspect("equal")
    lim_u, top_v, bot_v = 1.5, -1.3, net_depth + 0.45     # the whole backboard
    ax.set_xlim(-lim_u, lim_u)
    ax.set_ylim(bot_v, top_v)

    # Marks on the rim's line, by left-to-right position only: that's what the camera
    # measures. A touch where the ball met the rim; a swish where it came through.
    spots = ([(r["rim_u"], 0.0, "miss") for r in rimmed if r["result"] == "miss"]
             + [(r["rim_u"], 0.0, "rim") for r in rimmed if r["result"] == "make"]
             + [(r["cross_u"], 0.0, "swish") for r in swish if r["cross_u"] != ""])

    # The whole hoop straight on at the rim's height: the backboard as traced from a clip, the
    # pole under it, the rim edge-on as a flat line, the net's outline hanging straight down,
    # all to scale.
    board = board if board is not None else default_board()
    foot = float(board[:, 1].max())
    ax.add_patch(Polygon(board, closed=True, fc="#ecebe7", ec="#b9b8b2", lw=1.5, zorder=0.5))
    ax.add_patch(Polygon([(-0.07, foot), (0.07, foot), (0.07, bot_v), (-0.07, bot_v)],
                         closed=True, fc="#c9c8c3", ec="none", zorder=0.6))
    ax.plot([-0.5, -0.38, 0.38, 0.5], [0, net_depth, net_depth, 0], color="#b9b8b2", lw=1.2,
            ls=(0, (4, 3)), zorder=2)
    ax.plot([-0.5, 0.5], [0, 0], color=hist.RIM, lw=7, solid_capstyle="round", zorder=3)
    ax.text(-0.5, 0.1, "shooter's side", color=hist.INK_2, fontsize=12, ha="right", va="top")
    ax.text(0.5, 0.1, "far side", color=hist.INK_2, fontsize=12, ha="left", va="top")
    ax.annotate("", xy=(-0.75, -0.4), xytext=(-1.2, -0.4),
                arrowprops=dict(arrowstyle="-|>", color=hist.INK_2, lw=1.5))
    ax.text(-1.2, -0.46, "shot from", color=hist.INK_2, fontsize=11, va="bottom")

    style = {"swish": dict(s=230, marker="o", facecolors="none", edgecolors=hist.MAKE, linewidths=2.5),
             "rim": dict(s=150, marker="o", c=RIM_IN, edgecolors=hist.SURFACE, linewidths=1.5),
             "miss": dict(s=190, marker="X", c=hist.MISS, edgecolors=hist.SURFACE, linewidths=1.5)}
    for u, v, kind in spots:
        ax.scatter(u, v, zorder=6, **style[kind])
    ax.axis("off")
    ax.set_title("The rim, every shot turned so the shooter is on the left", loc="left",
                 fontsize=hist.TITLE_SIZE, weight="bold", color=hist.INK, pad=40)
    where = [("back" if r["rim_u"] > FRONT_BACK else "front" if r["rim_u"] < -FRONT_BACK
              else "side") for r in rimmed]
    parts = [f"{where.count(k)} {name}" for k, name in
             (("back", "far side"), ("front", "shooter's side"), ("side", "middle")) if where.count(k)]
    line = ("no rim touches" if not rimmed else "first touches: " + ", ".join(parts)
            + "  ·  placed left to right only, as the camera measures it")
    ax.text(0, 1.02, line, transform=ax.transAxes, fontsize=14, color=hist.INK_2, va="bottom")
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", ms=15, mfc="none", mec=hist.MAKE, mew=2.5,
               label="swish: where it came through"),
        Line2D([], [], marker="o", ls="", ms=12, mfc=RIM_IN, mec=hist.SURFACE,
               label="make: where it first touched the rim"),
        Line2D([], [], marker="X", ls="", ms=13, mfc=hist.MISS, mec=hist.SURFACE,
               label="miss: where it hit the rim"),
    ], loc="upper left", bbox_to_anchor=(0.0, 1.0), frameon=False, fontsize=11,
        labelcolor=hist.INK_2)

    # Right, top: how every side-on shot ended up.
    order = ["swish", "rim, then in", "front rim", SIDE, "back rim",
             "short, no rim", "long, no rim", "no rim contact seen"]
    counts = [(o, sum(r["outcome"] == o for r in across)) for o in order]
    counts = [(o, n) for o, n in counts if n]
    bx = fig.add_axes([0.66, 0.56, 0.31, 0.15])
    y = np.arange(len(counts))[::-1]
    colors = [hist.MAKE if o == "swish" else RIM_IN if o == "rim, then in" else hist.MISS
              for o, _ in counts]
    bx.barh(y, [n for _, n in counts], color=colors, height=0.6)
    for yi, (o, n) in zip(y, counts):
        bx.text(n + 0.15, yi, str(n), va="center", fontsize=14, weight="bold", color=hist.INK)
    shown = {"back rim": "miss, far side", "front rim": "miss, shooter's side", SIDE: "miss, middle"}
    bx.set_yticks(y, [shown.get(o, o) for o, _ in counts], fontsize=13)
    bx.set_xticks([])
    for name in ("top", "right", "bottom"):
        bx.spines[name].set_visible(False)
    bx.spines["left"].set_color("#c9c8c3")
    bx.tick_params(length=0, colors=hist.INK_2)
    bx.set_title("How each shot ended", loc="left", fontsize=hist.TITLE_SIZE, weight="bold",
                 color=hist.INK, pad=14)

    # Right, bottom: long or short, shot by shot, in the order taken. Where each shot was
    # headed at the rim's height (its free flight, before the rim touched it).
    tx = fig.add_axes([0.66, 0.09, 0.31, 0.33])
    seq = [r for r in across if r["long_short"] != ""]
    xs = np.arange(len(seq))
    ys = np.array([r["long_short"] for r in seq], float)
    lim = max(1.6, float(np.abs(ys).max()) + 0.25 if len(ys) else 1.6)
    tx.axhspan(-CLEAN_R, CLEAN_R, color=hist.MAKE, alpha=0.1, lw=0)
    for e in (-1, 1):   # the rim's inside edges
        tx.axhline(e, color=hist.RIM, lw=1.5, alpha=0.8)
    tx.axhline(0, color="#9b9a95", lw=1)
    sessions = [r["session"] for r in seq]
    for i in range(1, len(seq)):
        if sessions[i] != sessions[i - 1]:
            tx.axvline(i - 0.5, color="#d9d8d3", lw=1)
    for name in dict.fromkeys(sessions):   # a running median within each session: drifting?
        idx = [i for i, sn in enumerate(sessions) if sn == name]
        if len(idx) >= 3:
            med = [float(np.median(ys[idx][max(0, j - 2):j + 1])) for j in range(len(idx))]
            tx.plot(xs[idx], med, color=hist.INK_2, lw=1.5, ls="--", alpha=0.7)
        tx.text(idx[0] - 0.3, lim * 0.97, name.split(" ", 1)[1], fontsize=9, color=hist.MUTED,
                va="top")
    kind = {"swish": hist.MAKE, "rim, then in": RIM_IN}
    for x, yv, r in zip(xs, ys, seq):
        color = hist.MISS if r["result"] == "miss" else kind.get(r["outcome"], hist.MAKE)
        tx.scatter(x, yv, s=90, marker="^" if r["shot_type"] == "3PT" else "o", c=color,
                   edgecolors=hist.SURFACE, linewidths=1.2, zorder=3)
    tx.set_ylim(-lim, lim)
    tx.set_xlim(-0.8, len(seq) - 0.2)
    tx.set_xticks([])
    tx.set_ylabel("short  ·  long", color=hist.INK_2, fontsize=12)
    tx.set_yticks([-1, -CLEAN_R, 0, CLEAN_R, 1], ["rim edge", "", "center", "", "rim edge"])
    for name in ("top", "right", "bottom"):
        tx.spines[name].set_visible(False)
    tx.spines["left"].set_color("#c9c8c3")
    tx.tick_params(colors=hist.INK_2, length=0)
    med_all = float(np.median(ys)) if len(ys) else 0.0
    tx.set_title("Long or short, shot by shot", loc="left", fontsize=hist.TITLE_SIZE,
                 weight="bold", color=hist.INK, pad=30)
    miss_ys = [r["long_short"] for r in seq if r["result"] == "miss"]
    tx.text(0, 1.02, f"typical shot: {depth_word(med_all)}"
            + (f"  ·  typical miss: {depth_word(float(np.median(miss_ys)))}" if miss_ys else "")
            + "  ·  dashed: last 3 shots  ·  ▲ = three",
            transform=tx.transAxes, fontsize=11, color=hist.INK_2, va="bottom")

    fig.text(0.04, 0.025,
             "Left and right along the rim as the camera sees it, turned so the shooter is on the left; "
             "reading that as the shooter's front and back (and long or short) assumes the shot crosses the picture.\nA touch: the first frame the ball strays from its free flight while "
             "its top is still above the rim. Long or short: where that free flight crossed the rim's height, relative to the rim.  "
             f"Generated {datetime.now():%Y-%m-%d %H:%M}",
             fontsize=10, color=hist.MUTED)
    fig.savefig(dst, dpi=130, pil_kwargs={"quality": 92})
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def _cell(v, digits=2):
    return "" if v is None or not np.isfinite(v) else round(float(v), digits)


def main_(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("inputs", nargs="*", type=Path, metavar="CLIP|DIR",
                        help=f"clips or folders (searched with subfolders); default {main.rel(cfg.INPUT)}")
    parser.add_argument("--cached-only", action="store_true",
                        help="never call the gateway; skip a clip whose replies aren't cached")
    parser.add_argument("--out", type=Path, default=RIM_DIR, help="where to write")
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
    rows, rim_hs, nets, board = [], [], [], None
    view3 = court.CourtView(10, 10, three_radius=cfg.THREE_POINT_RADIUS_M,
                            ft_distance=cfg.FREE_THROW_DISTANCE_M)
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
        hoop, fps = c["hoop"], c["info"].fps
        rim_hs.append((hoop.cy - hoop.top) / hoop.width)
        if board is None:
            board = board_outline(c)
        nets.append((hoop.net_bottom - hoop.cy) / hoop.width)
        for shot in c["shots"]:
            if shot.made is None:
                continue
            r = read_shot(shot, hoop, fps)
            rows.append({"session": session, "clip": hist.rel(clip), "shot": shot.number,
                         "video_time_s": round(shot.release / fps, 1),
                         "result": "make" if shot.made else "miss", "view": r["view"],
                         "outcome": r["outcome"], "rim_u": _cell(r["u"]), "rim_v": _cell(r["v"]),
                         "off_path": _cell(r["off_path"]), "cross_u": _cell(r["cross_u"]),
                         "long_short": _cell(r["long"]) if r["view"] == "across" else "",
                         "long_short_word": depth_word(r["long"]) if r["view"] == "across" else "",
                         "shot_type": ("3PT" if view3.is_three(*shot.court_xy) else "2PT")
                                      if shot.court_xy else "",
                         "distance_ft": _cell(shot.distance_m * hist.FT_PER_M, 1)
                                        if shot.distance_m is not None else "",
                         "how_it_was_called": shot.reason})
        across = [r for r in rows if r["clip"] == hist.rel(clip) and r["view"] == "across"]
        console.print(f"  {len(across)} of {len(c['shots'])} shots seen from the side: "
                      + ", ".join(f"{r['shot']} {r['outcome']}" for r in across))

    if not rows:
        console.print("[red]No shots.[/]")
        return 1
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    hist.write_csv(out / "rim_shots.csv", rows)
    render(rows, float(np.median(rim_hs)), float(np.median(nets)), out / "rim.jpg", board=board)
    for name in ("rim.jpg", "rim_shots.csv"):
        console.print(f"  -> [dim]{main.rel(out / name)}[/]")
    return 0


if __name__ == "__main__":
    sys.exit(main_())
