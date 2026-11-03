"""Shots: found in the pose, judged by the ball against the rim, plus the elbow and stance."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .hoop import Hoop
from .skeleton import KPT_NAMES

J = {name: i for i, name in enumerate(KPT_NAMES)}
HEAD = [J[n] for n in ("nose", "left_eye", "right_eye", "left_ear", "right_ear")]


# ── pose, as arrays ──────────────────────────────────────────────────────────

def keypoints(by_index: dict, n_frames: int, width: int, height: int) -> np.ndarray:
    """``(n_frames, 17, 2)`` in pixels; NaN where nobody is posed or a joint is missing."""
    out = np.full((n_frames, len(KPT_NAMES), 2), np.nan)
    for i in range(n_frames):
        persons = by_index.get(i)
        if not persons:
            continue
        kpts = np.asarray(persons[0].get("kpts_xy") or [], float)
        if kpts.shape != (len(KPT_NAMES), 2):
            continue
        missing = (kpts == 0).all(axis=1)   # (0, 0) is the model's "not visible"
        kpts = kpts * [width, height]
        kpts[missing] = np.nan
        out[i] = kpts
    return out


def elbow_angle(kp: np.ndarray, side: str, *, min_projected: float = 0.0) -> np.ndarray:
    """The elbow angle per frame in the image, degrees (180 = straight).

    NaN where the upper arm or forearm shows under *min_projected* of its usual
    length: pointing at the camera, its angle means nothing.
    """
    s, e, w = (kp[:, J[f"{side}_{j}"]] for j in ("shoulder", "elbow", "wrist"))
    a, b = s - e, w - e
    la, lb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        angle = np.degrees(np.arccos(np.clip((a * b).sum(axis=1) / (la * lb), -1, 1)))
        if min_projected > 0:
            short = (la < min_projected * np.nanpercentile(la, 90)) | (
                lb < min_projected * np.nanpercentile(lb, 90))
            angle[short] = np.nan
    return angle


def shoulder_ratio(kp: np.ndarray) -> np.ndarray:
    """Shoulder width / torso length per frame: ~0.1-0.45 side-on, ~0.6 facing or facing away."""
    shoulders = np.linalg.norm(kp[:, J["left_shoulder"]] - kp[:, J["right_shoulder"]], axis=1)
    mid_s = (kp[:, J["left_shoulder"]] + kp[:, J["right_shoulder"]]) / 2
    mid_h = (kp[:, J["left_hip"]] + kp[:, J["right_hip"]]) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        return _median(shoulders / np.linalg.norm(mid_s - mid_h, axis=1), 9)


def _median(values: np.ndarray, frames: int) -> np.ndarray:
    """Running median over *frames*, ignoring NaN."""
    half = frames // 2
    out = np.full(len(values), np.nan)
    for i in range(len(values)):
        window = values[max(0, i - half):i + half + 1]
        if np.isfinite(window).any():
            out[i] = np.nanmedian(window)
    return out


# ── finding shots ────────────────────────────────────────────────────────────

def candidates(kp: np.ndarray, fps: float, *, armed_below: str, min_gap_seconds: float
               ) -> list[tuple[int, int]]:
    """``[(first, last frame with both wrists over the head), ...]``, one per shot.

    The wrists must drop below *armed_below* between shots, so a follow-through can't count twice.
    """
    head_top = np.nanmin(kp[:, HEAD, 1], axis=1)
    wrists = np.nanmax(kp[:, [J["left_wrist"], J["right_wrist"]], 1], axis=1)   # the lower wrist
    arm_line = np.nanmax(kp[:, [J[f"left_{armed_below}"], J[f"right_{armed_below}"]], 1], axis=1)
    with np.errstate(invalid="ignore"):
        up = _median((wrists < head_top).astype(float), 5) > 0.5
        down = _median((wrists > arm_line).astype(float), 5) > 0.5

    spans: list[list[int]] = []
    armed = True
    i = 0
    while i < len(up):
        if armed and up[i]:
            j = i
            while j + 1 < len(up) and up[j + 1]:
                j += 1
            if spans and i - spans[-1][1] < min_gap_seconds * fps:
                spans[-1][1] = j
            else:
                spans.append([i, j])
            armed = False
            i = j + 1
            continue
        if down[i]:
            armed = True
        i += 1
    return [(a, b) for a, b in spans]


def shooting_side(kp: np.ndarray, spans: list[tuple[int, int]], fps: float) -> str:
    """The arm that straightens most through the shots; the guide hand barely does."""
    swing = {}
    for side in ("left", "right"):
        angle = elbow_angle(kp, side)
        moves = []
        for a, _ in spans:
            before = angle[max(0, a - int(0.4 * fps)):a + 1]
            after = angle[a:a + int(0.6 * fps)]
            if np.isfinite(before).any() and np.isfinite(after).any():
                moves.append(np.nanmax(after) - np.nanmin(before))
        swing[side] = np.median(moves) if moves else 0.0
    return max(swing, key=swing.get)


# ── one shot ─────────────────────────────────────────────────────────────────

@dataclass
class Shot:
    number: int
    start: int                 # first frame with both hands overhead
    end: int                   # last frame overhead
    release: int               # the ball leaves the hand
    set_frame: int             # the deepest elbow bend before the release
    elbow_set: float           # degrees, as seen by the camera
    elbow_release: float
    side_on: bool = True       # elbow values are NaN when not side-on
    made: bool | None = None   # None: the ball was never tracked
    reason: str = ""
    crossing: float | None = None   # frame the ball crossed the rim's height going in
    decided: int | None = None      # frame the verdict shows on the video
    feet_px: tuple[float, float] | None = None
    body_px: float | None = None    # shoulder to ankle, px
    court_xy: tuple[float, float] | None = None   # meters from the hoop
    distance_m: float | None = None
    path: list[tuple[int, float, float]] = field(default_factory=list)   # ball (frame, x, y)
    motion: int | None = None       # the shooting motion's start: the ball's low point in the hands

    @property
    def label(self) -> str:
        return "MAKE" if self.made else "MISS" if self.made is False else "?"

    def summary(self, fps: float) -> dict:
        return {
            "shot": self.number, "result": self.label.lower(), "reason": self.reason,
            "motion_s": round(self.motion / fps, 2) if self.motion is not None else None,
            "release_s": round(self.release / fps, 2), "side_on": self.side_on,
            "elbow_set_deg": _round(self.elbow_set), "elbow_release_deg": _round(self.elbow_release),
            "court_xy_m": [round(v, 2) for v in self.court_xy] if self.court_xy else None,
            "distance_m": round(self.distance_m, 2) if self.distance_m is not None else None,
            "distance_ft": round(self.distance_m * 3.2808, 1) if self.distance_m is not None else None,
        }


def _round(value: float) -> float | None:
    return round(float(value), 1) if np.isfinite(value) else None


def motion_start(start: int, x: np.ndarray, y: np.ndarray, size: np.ndarray, wrists: np.ndarray,
                 fps: float, *, lookback: float, reach: float, max_gap: int = 3) -> int:
    """Where the shooting motion begins: back from the hands going up, the ball's lowest point
    while it stays in the hands. The pocket at the hip for most shots; the shoulder for a catch-and-shoot.

    *wrists* is ``(n_frames, 2, 2)``; the ball counts as held within *reach* ball sizes of a wrist.
    """
    best, gap = None, 0
    for f in range(start, max(-1, start - int(lookback * fps)), -1):
        if not np.isfinite(x[f]):
            gap += 1
            if gap > max_gap:
                break
            continue
        gap = 0
        near = np.hypot(wrists[f, :, 0] - x[f], wrists[f, :, 1] - y[f])
        if not np.isfinite(near).any() or np.nanmin(near) > reach * size[f]:
            break
        if best is None or y[f] > y[best]:
            best = f
    return start if best is None else best


def find_release(start: int, end: int, ball_y: np.ndarray, wrist_y: np.ndarray,
                 diameter: float, fps: float) -> int:
    """The first frame the ball is 1.5 diameters over the shooting wrist and rising.

    (Resting in the hand it sits about one diameter up.) Without a ball, the wrist's peak.
    """
    lo, hi = max(0, start - int(0.3 * fps)), min(len(ball_y) - 3, end + int(0.5 * fps))
    for f in range(lo, hi):
        if not (np.isfinite(ball_y[f]) and np.isfinite(wrist_y[f])):
            continue
        nxt = ball_y[f + 1:f + 3]
        rising = np.isfinite(nxt).all() and nxt[0] < ball_y[f] and nxt[1] < nxt[0]
        if ball_y[f] < wrist_y[f] - 1.5 * diameter and rising:
            return f
    window = wrist_y[start:min(len(wrist_y), start + int(0.7 * fps))]
    return start + int(np.nanargmin(window)) if np.isfinite(window).any() else start


def judge(shot: Shot, x: np.ndarray, y: np.ndarray, size: np.ndarray, hoop: Hoop, fps: float, *,
          until: int, inner_fraction: float, net_min_seconds: float, net_side_tolerance: float,
          rebound_margin: float, max_gap: int, ball_to_rim: float, front_size: float) -> None:
    """Make or miss. A make comes down from above the rim, crosses the rim line inside it
    at the rim's distance (not in front of it), leaves the net slowed by it, and never
    rises back over the rim. Every downward crossing is tried, so a roll-in is a make.
    """
    w = hoop.width
    frames = [f for f in range(shot.release, min(until, len(x))) if np.isfinite(x[f])]
    shot.path = [(f, float(x[f]), float(y[f])) for f in frames]
    if not frames:
        shot.made, shot.reason = None, "ball not tracked"
        return

    above = [f for f in frames if y[f] < hoop.cy - 0.5 * w]
    if not above:
        shot.made, shot.reason, shot.decided = False, "never above the rim", frames[-1]
        return

    reason = "never came down through the rim"
    for a, b in zip(frames, frames[1:]):
        # The ball can hide behind the rim and net, so a crossing may span a gap.
        if a < above[0] or b - a > max_gap or not (y[a] < hoop.cy <= y[b]):
            continue
        t = (hoop.cy - y[a]) / (y[b] - y[a])
        cross = a + t * (b - a)
        cross_x = x[a] + t * (x[b] - x[a])
        if abs(cross_x - hoop.cx) > inner_fraction * w / 2:
            reason = "outside the rim"
            continue

        after = [f for f in frames if f > cross]
        out = None
        rebound = None
        for c, d in zip([a] + after, after):
            if y[d] < hoop.cy - rebound_margin * w:
                rebound = d
                break
            if y[d] >= hoop.net_bottom:
                s = (hoop.net_bottom - y[c]) / (y[d] - y[c]) if y[d] != y[c] else 1.0
                # Where it left the net, not where it was next seen (maybe on the floor).
                out = (max(cross, c + s * (d - c)), x[c] + s * (x[d] - x[c]), d)
                break
        if rebound is not None:
            reason = "rimmed out"
            continue
        # At the rim's distance a ball is ball_to_rim rim widths across; much bigger is
        # closer to the camera, i.e. falling in front of the rim.
        end = out[2] if out is not None else cross + 0.3 * fps
        sizes = [size[f] for f in frames if cross - 0.2 * fps <= f <= end and np.isfinite(size[f])]
        if sizes and np.median(sizes) / w > front_size * ball_to_rim:
            reason = "in front of the rim"
            continue
        if out is None:
            # Lost in the net and never seen again.
            shot.made, shot.reason, shot.crossing = True, "into the net", cross
            shot.decided = int(np.ceil(cross + net_min_seconds * fps))
            return
        out_frame, out_x, seen = out
        if out_frame - cross < net_min_seconds * fps:
            reason = "fell past the net"
            continue
        if abs(out_x - hoop.cx) > w / 2 + net_side_tolerance * w:
            reason = "wide of the net"
            continue
        shot.made, shot.reason, shot.crossing, shot.decided = True, "through the net", cross, seen
        return

    # A miss shows once the ball leaves the rim for good; "off the rim" if it bounced up there.
    near = [f for f in frames if abs(x[f] - hoop.cx) < w and abs(y[f] - hoop.cy) < w]
    bounced = any(y[f] - min((y[g] for g in frames if g > f), default=y[f]) > 0.4 * w
                  for f in near)
    later = [f for f in frames if near and f > near[-1]]
    shot.made, shot.decided = False, (later[0] if later else frames[-1])
    shot.reason = "off the rim" if bounced else reason


def ball_unresolved(x: np.ndarray, y: np.ndarray, start: int, end: int, hoop: Hoop) -> bool:
    """Whether the ball was last seen at the rim, so its window should run longer."""
    seen = [f for f in range(start, min(end, len(x))) if np.isfinite(x[f])]
    if not seen:
        return False
    f = seen[-1]
    return y[f] < hoop.net_bottom and abs(x[f] - hoop.cx) < 1.5 * hoop.width


def stance(kp: np.ndarray, start: int, fps: float) -> tuple[tuple[float, float], float] | None:
    """``((feet x, y), shoulder-to-ankle px)`` for a shot.

    The feet: median over the 0.5 s before the hands go up. The size: standing tall
    (90th percentile over 2 s), since bent knees in the gather would read as farther away.
    """
    def legs(lo: int, hi: int):
        seg = kp[max(0, lo):hi]
        ankles = np.nanmean(seg[:, [J["left_ankle"], J["right_ankle"]]], axis=1)
        shoulders = np.nanmean(seg[:, [J["left_shoulder"], J["right_shoulder"]]], axis=1)
        return ankles, np.linalg.norm(ankles - shoulders, axis=1)

    ankles, _ = legs(start - int(0.5 * fps), start + 1)
    _, size = legs(start - int(1.5 * fps), start + int(0.5 * fps))
    if not (np.isfinite(size).any() and np.isfinite(ankles).any()):
        return None
    return (float(np.nanmedian(ankles[:, 0])), float(np.nanmedian(ankles[:, 1]))), float(
        np.nanpercentile(size[np.isfinite(size)], 90))


def streaks(shots: list[Shot], n_frames: int) -> np.ndarray:
    """Makes in a row as of each frame, counting a shot from the frame it is decided."""
    out = np.zeros(n_frames, int)
    run = 0
    for shot in sorted((s for s in shots if s.decided is not None), key=lambda s: s.decided):
        run = run + 1 if shot.made else 0
        out[min(shot.decided, n_frames):] = run
    return out
