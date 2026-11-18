"""From the models' replies to the swing: the club, the ball, the phases and the tempo.

SAM's track ids are not trusted. On this kind of clip the club's track hands
over to the ball partway through, and the club's mask often carries the ball
as a second blob. So both are picked per frame from mask blobs instead: the
ball is the small round blob where the ball should be, and the club is
whatever is left of the biggest mask held near the hands.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .skeleton import KPT_NAMES

J = {name: i for i, name in enumerate(KPT_NAMES)}
MIN_KPT_SCORE = 0.3


def keypoints(by_index: dict[int, list[dict]], n_frames: int, width: int, height: int
              ) -> np.ndarray:
    """``(n_frames, 17, 2)`` in pixels; NaN where unseen or unsure."""
    kp = np.full((n_frames, 17, 2), np.nan)
    for f, persons in by_index.items():
        if not persons or not 0 <= f < n_frames:
            continue
        xy = np.asarray(persons[0]["kpts_xy"], float) * [width, height]
        score = np.asarray(persons[0].get("kpts_score") or np.ones(17), float)
        xy[score < MIN_KPT_SCORE] = np.nan
        kp[f] = xy
    return kp


def _fill(values: np.ndarray) -> np.ndarray:
    """Linear fill across NaNs, per column, holding the ends."""
    out = values.copy()
    idx = np.arange(len(out))
    for c in range(out.shape[1]):
        ok = np.isfinite(out[:, c])
        if ok.any():
            out[:, c] = np.interp(idx, idx[ok], out[ok, c])
    return out


def hands(kp: np.ndarray) -> np.ndarray:
    """``(n, 2)``: the grip, as the midpoint of the wrists, gaps filled."""
    w = kp[:, [J["left_wrist"], J["right_wrist"]], :]
    mid = np.nanmean(w, axis=1) if np.isfinite(w).any() else np.full((len(kp), 2), np.nan)
    return _fill(mid)


def torso_length(kp: np.ndarray) -> float:
    """Median shoulder-to-hip distance, px: the scale every threshold is in."""
    sh = np.nanmean(kp[:, [J["left_shoulder"], J["right_shoulder"]]], axis=1)
    hp = np.nanmean(kp[:, [J["left_hip"], J["right_hip"]]], axis=1)
    d = np.hypot(*(sh - hp).T)
    return float(np.nanmedian(d)) if np.isfinite(d).any() else 300.0


# ── blobs ────────────────────────────────────────────────────────────────────

@dataclass
class Blob:
    cx: float
    cy: float
    w: int
    h: int
    area: int
    label: int          # index in its mask's component labelling
    labels: np.ndarray  # that labelling (shared by every blob of the mask)

    @property
    def d(self) -> float:
        return float(max(self.w, self.h))

    @property
    def round(self) -> bool:
        return 0.6 <= self.w / max(1, self.h) <= 1.65

    def mask(self) -> np.ndarray:
        return self.labels == self.label

    def crop(self) -> tuple[int, int, np.ndarray]:
        """``(x0, y0, mask)``: just the blob's bounding box, cheap to keep per frame."""
        ys, xs = np.nonzero(self.labels == self.label)
        x0, y0 = int(xs.min()), int(ys.min())
        return x0, y0, self.labels[y0:ys.max() + 1, x0:xs.max() + 1] == self.label


def blobs(mask: np.ndarray, min_area: int = 12) -> list[Blob]:
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    return [Blob(float(cents[i][0]), float(cents[i][1]), int(stats[i][2]), int(stats[i][3]),
                 int(stats[i][4]), i, labels)
            for i in range(1, n) if stats[i][4] >= min_area]


# ── the ball ─────────────────────────────────────────────────────────────────

@dataclass
class BallTrack:
    """The ball on every frame: at rest on the tee, then the few frames it's seen flying."""

    x: np.ndarray
    y: np.ndarray
    d: np.ndarray
    rest_until: int            # last frame on the tee
    flight: list[int]          # frames it was seen in flight
    diameter: float            # at rest, px
    impact: float = float("nan")   # fractional frame the ball was struck
    path: np.ndarray | None = None  # (m, 3): t, x, y of the projected flight
    launch_deg: float = float("nan")  # start direction in frame: 0 straight up, + right
    drift: np.ndarray | None = None   # (n, 2): the phone's drift since impact, px; path is in impact coords
    diameter_spread: tuple[float, float] | None = None   # its size at rest, 10th–90th percentile, px
    masks: dict[int, tuple[int, int, np.ndarray]] = field(default_factory=dict)  # SAM's mask, per frame seen

    @property
    def struck(self) -> bool:
        return bool(self.flight) or self.rest_until < len(self.x) - 1


def _ball_candidates(frame_dets: list[dict]) -> list[Blob]:
    out = []
    for det in frame_dets:
        if det["mask"] is not None:
            out += [b for b in blobs(det["mask"]) if b.round]
    return out


def track_ball(ball_dets: dict[int, list[dict]], club_dets: dict[int, list[dict]],
               n_frames: int, torso: float, *, rest_move: float, flight_max_step: float,
               max_size: float) -> BallTrack:
    """The ball off both prompts' masks (SAM's "golf club" track often catches the ball)."""
    x, y, d = (np.full(n_frames, np.nan) for _ in range(3))
    cands = {f: _ball_candidates(ball_dets.get(f, []) + club_dets.get(f, []))
             for f in range(n_frames)}
    cands = {f: [b for b in c if b.d <= max_size * torso] for f, c in cands.items()}

    # The ball at address: the most common small round blob over the opening frames.
    opening = [b for f in range(min(n_frames, 15)) for b in cands[f]]
    if not opening:
        return BallTrack(x, y, d, -1, [], float("nan"))
    seed = min(opening, key=lambda b: sum(np.hypot(b.cx - o.cx, b.cy - o.cy) for o in opening))
    diameter = float(np.median([b.d for b in opening
                                if np.hypot(b.cx - seed.cx, b.cy - seed.cy) < seed.d]))

    # At rest: within a fraction of a ball of where it last sat (the phone drifts a little).
    px, py = seed.cx, seed.cy
    rest_until = -1
    misses = 0
    masks: dict[int, tuple[int, int, np.ndarray]] = {}
    for f in range(n_frames):
        near = [b for b in cands[f] if np.hypot(b.cx - px, b.cy - py) <= rest_move * diameter
                and 0.6 <= b.d / diameter <= 1.5]
        if near:
            b = min(near, key=lambda b: np.hypot(b.cx - px, b.cy - py))
            x[f], y[f], d[f] = b.cx, b.cy, b.d
            masks[f] = b.crop()
            px, py = b.cx, b.cy
            rest_until = f
            misses = 0
        else:
            misses += 1
            # A frame or two hidden by the club is fine; longer and it's gone.
            if misses > 2 and rest_until >= 0:
                break
    # Its size: the median over every frame it was seen at rest, not just the opening ones.
    # It wobbles a px or two frame to frame, and the flight estimate scales with it, so
    # its spread (10th to 90th percentile) is kept for the estimate's range.
    seen = d[:rest_until + 1][np.isfinite(d[:rest_until + 1])]
    if seen.size:
        diameter = float(np.median(seen))
        diameter_spread = (float(np.percentile(seen, 10)), float(np.percentile(seen, 90)))
    else:
        diameter_spread = (diameter, diameter)
    # Frames between sightings at rest: hold the last position.
    for f in range(rest_until + 1):
        if not np.isfinite(x[f]):
            prev = np.nonzero(np.isfinite(x[:f]))[0]
            if prev.size:
                x[f], y[f], d[f] = x[prev[-1]], y[prev[-1]], d[prev[-1]]

    # In flight: away from the tee, no bigger than at rest, on a smooth path.
    flight: list[int] = []
    last = np.array([x[rest_until], y[rest_until]]) if rest_until >= 0 else None
    vel = None
    misses = 0
    for f in range(rest_until + 1, n_frames):
        if last is None:
            break
        pool = [b for b in cands[f] if b.d <= 1.15 * diameter and b.d >= 0.25 * diameter
                and np.hypot(b.cx - x[rest_until], b.cy - y[rest_until]) > 1.5 * diameter]
        if vel is None:
            pool = [b for b in pool if np.hypot(b.cx - last[0], b.cy - last[1])
                    <= flight_max_step * torso * (1 + misses) and b.cy < last[1]]
            pick = max(pool, key=lambda b: b.cy, default=None)   # the one nearest the tee
        else:
            guess = last + vel * (1 + misses)
            pool = [b for b in pool if np.hypot(b.cx - guess[0], b.cy - guess[1])
                    <= 0.5 * np.hypot(*vel) + 2 * diameter]
            pick = min(pool, key=lambda b: np.hypot(b.cx - guess[0], b.cy - guess[1]),
                       default=None)
        if pick is None:
            misses += 1
            if misses > 1:
                break
            continue
        now = np.array([pick.cx, pick.cy])
        # The first step starts mid-frame (at impact), so it undercounts the speed a little.
        vel = (now - last) / (f - flight[-1]) if flight else (now - last) * 0.75
        x[f], y[f], d[f] = pick.cx, pick.cy, pick.d
        masks[f] = pick.crop()
        flight.append(f)
        last = now
        misses = 0
    return BallTrack(x, y, d, rest_until, flight, diameter, diameter_spread=diameter_spread,
                     masks=masks)


def fit_flight(ball: BallTrack, *, width: int, height: int, fps: float,
               extend_seconds: float) -> None:
    """The ball's flight in frame, assuming it flies straight at constant speed for a while.

    For a straight 3D path at constant speed, ``(x - cx) * Z`` and ``Z`` are both linear
    in time, and the ball's size gives ``Z`` up to a scale (``Z ~ 1 / size``). Fitting
    those through the tee and the flight sightings gives the moment of impact and a
    path to extend past the last sighting. Projected, not measured: the ball is gone.
    """
    if ball.rest_until < 0 or not ball.flight:
        return
    cx, cy = width / 2, height / 2
    drift = ball.drift if ball.drift is not None else np.zeros((len(ball.x), 2))
    tee = np.array([ball.x[ball.rest_until], ball.y[ball.rest_until]]) - drift[ball.rest_until]
    fl = np.array(ball.flight, float)
    inv = 1.0 / ball.d[ball.flight]   # ~ depth
    inv0 = 1.0 / ball.diameter

    # Impact: where depth, extrapolated back, is the tee's.
    impact = ball.rest_until + 0.5
    if len(fl) >= 2:
        slope, icpt = np.polyfit(fl, inv, 1)
        if slope > 0:
            impact = float(np.clip((inv0 - icpt) / slope, ball.rest_until, fl[0]))
    ball.impact = impact

    # Start direction: from the tee to the first sighting. 0 = straight up the frame.
    first = np.array([ball.x[ball.flight[0]], ball.y[ball.flight[0]]]) - drift[ball.flight[0]]
    ball.launch_deg = float(np.degrees(np.arctan2(first[0] - tee[0], tee[1] - first[1])))

    t = np.concatenate([[impact], fl])
    z = np.concatenate([[inv0], inv]) / inv0                       # depth, tee = 1
    xs = np.concatenate([[tee[0]], ball.x[ball.flight] - drift[ball.flight, 0]])
    ys = np.concatenate([[tee[1]], ball.y[ball.flight] - drift[ball.flight, 1]])
    t_end = fl[-1] + extend_seconds * fps
    ts = np.linspace(impact, t_end, 120)

    zfit = np.polyfit(t, z, 1) if len(t) >= 3 else None
    if len(t) >= 6:
        # Followed long enough to need no model: the path is the sightings themselves,
        # lightly smoothed, and only the stretch past the last one is projected, from a
        # quadratic through the last dozen.
        ts_seen = np.linspace(impact, fl[-1], max(2, int(8 * (fl[-1] - impact))))
        k = 3
        sx = np.array([np.mean(xs[max(0, i - k // 2):i + k // 2 + 1]) for i in range(len(xs))])
        sy = np.array([np.mean(ys[max(0, i - k // 2):i + k // 2 + 1]) for i in range(len(ys))])
        sx[0], sy[0], sx[-1], sy[-1] = xs[0], ys[0], xs[-1], ys[-1]
        seen = np.stack([ts_seen, np.interp(ts_seen, t, sx), np.interp(ts_seen, t, sy)], axis=1)
        tail = slice(max(0, len(t) - 12), len(t))
        qx, qy = np.polyfit(t[tail], xs[tail], 2), np.polyfit(t[tail], ys[tail], 2)
        ts_proj = np.linspace(fl[-1], t_end, 40)
        proj = np.stack([ts_proj, np.polyval(qx, ts_proj), np.polyval(qy, ts_proj)], axis=1)
        proj[:, 1:] += np.array([xs[-1], ys[-1]]) - proj[0, 1:]   # start where the ball was last
        path = np.concatenate([seen, proj[1:]])
        keep = (path[:, 1] > -width) & (path[:, 1] < 2 * width) & (path[:, 2] > -height) \
            & (path[:, 2] < 2 * height)
        ball.path = path[keep]
        return
    if zfit is not None and zfit[0] > 0:
        ufit = np.polyfit(t, (xs - cx) * z, 1)
        vfit = np.polyfit(t, (ys - cy) * z, 1)
        zs = np.polyval(zfit, ts)
        px = cx + np.polyval(ufit, ts) / zs
        py = cy + np.polyval(vfit, ts) / zs
    else:
        # Too few sightings for depth: a straight line, slowing as it would recede.
        step = (np.array([xs[-1], ys[-1]]) - [xs[0], ys[0]]) / max(1e-6, t[-1] - t[0])
        u = ts - impact
        span = t[-1] - impact
        k = np.where(u <= span, u, span + (u - span) / (1 + 0.15 * (u - span)))
        px, py = xs[0] + step[0] * k, ys[0] + step[1] * k
    keep = (px > -width) & (px < 2 * width) & (py > -height)
    ball.path = np.stack([ts[keep], px[keep], py[keep]], axis=1)


# ── the club ─────────────────────────────────────────────────────────────────

@dataclass
class ClubTrack:
    masks: list[np.ndarray | None]
    head: np.ndarray              # (n, 2), NaN where unseen
    found: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))


def _without_ball(mask: np.ndarray, ball: BallTrack, f: int) -> np.ndarray:
    """*mask* with the ball and its tee cut out while the ball sits on the tee.

    SAM's "golf club" track often carries the ball and its tee, sometimes joined to the
    clubhead by the tee; cutting both out leaves only the club.
    """
    if not (0 <= f <= ball.rest_until and np.isfinite(ball.x[f])):
        return mask
    out = mask.copy()
    bx, by, dm = int(round(ball.x[f])), int(round(ball.y[f])), ball.diameter
    cv2.circle(out, (bx, by), int(np.ceil(0.65 * dm)), 0, -1)
    # The tee under it: thin and upright, so an opening with a kernel wider than the tee
    # removes it from that patch while the (much wider) clubhead survives.
    x0, x1 = max(0, bx - int(dm)), min(out.shape[1], bx + int(dm) + 1)
    y0, y1 = max(0, by), min(out.shape[0], by + int(2 * dm) + 1)
    if x1 > x0 and y1 > y0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(0.45 * dm)), 1))
        patch = out[y0:y1, x0:x1].astype(np.uint8)
        out[y0:y1, x0:x1] = cv2.morphologyEx(patch, cv2.MORPH_OPEN, k).astype(out.dtype)
    return out


def _on_tee(b: Blob, ball: BallTrack, f: int) -> bool:
    """A blob that's the ball's tee (or what's left of the ball): small and just under it."""
    if not (0 <= f <= ball.rest_until and np.isfinite(ball.x[f])):
        return False
    dm = ball.diameter
    return (b.area <= 2 * dm * dm and abs(b.cx - ball.x[f]) <= dm
            and ball.y[f] - dm <= b.cy <= ball.y[f] + 2 * dm)


def track_club(club_dets: dict[int, list[dict]], ball: BallTrack, grip: np.ndarray,
               n_frames: int, torso: float, *, max_hand_gap: float) -> ClubTrack:
    masks: list[np.ndarray | None] = [None] * n_frames
    head = np.full((n_frames, 2), np.nan)
    for f in range(n_frames):
        best, best_area = None, 0
        for det in club_dets.get(f, []):
            if det["mask"] is None:
                continue
            keep = np.zeros_like(det["mask"])
            for b in blobs(_without_ball(det["mask"], ball, f)):
                is_ball = (np.isfinite(ball.x[f]) and b.round and b.d <= 1.6 * ball.diameter
                           and np.hypot(b.cx - ball.x[f], b.cy - ball.y[f]) <= ball.diameter)
                if not is_ball and not _on_tee(b, ball, f):
                    keep |= b.mask()
            area = int(keep.sum())
            if area < 150:
                continue
            ys, xs = np.nonzero(keep)
            gap = np.min(np.hypot(xs - grip[f, 0], ys - grip[f, 1]))
            if gap > max_hand_gap * torso:
                continue
            if area > best_area:
                best, best_area = keep, area
        if best is None:
            continue
        masks[f] = best
        ys, xs = np.nonzero(best)
        dist = np.hypot(xs - grip[f, 0], ys - grip[f, 1])
        far = dist >= 0.85 * dist.max()
        head[f] = xs[far].mean(), ys[far].mean()
    return ClubTrack(masks, head, np.isfinite(head[:, 0]))


# ── phases and tempo ─────────────────────────────────────────────────────────

@dataclass
class Swing:
    takeaway: int
    top: int
    impact: float
    finish: int
    fps: float

    @property
    def backswing_s(self) -> float:
        return (self.top - self.takeaway) / self.fps

    @property
    def downswing_s(self) -> float:
        return (self.impact - self.top) / self.fps

    @property
    def tempo(self) -> float:
        return self.backswing_s / self.downswing_s if self.downswing_s > 0 else float("nan")

    def phase(self, f: float) -> str:
        if f < self.takeaway:
            return "Address"
        if f < self.top - 1:
            return "Backswing"
        if f <= self.top + 1:
            return "Top"
        if f < self.impact - 0.5:
            return "Downswing"
        if f < self.impact + 1.5:
            return "Impact"
        return "Follow-through" if f < self.finish else "Finish"

    def summary(self) -> dict:
        r = lambda v: round(float(v), 3)
        return {"takeaway_s": r(self.takeaway / self.fps), "top_s": r(self.top / self.fps),
                "impact_s": r(self.impact / self.fps), "finish_s": r(self.finish / self.fps),
                "backswing_s": r(self.backswing_s), "downswing_s": r(self.downswing_s),
                "tempo": r(self.tempo)}


def phases(grip: np.ndarray, club: ClubTrack, impact: float, fps: float, torso: float) -> Swing:
    """Takeaway, top and finish off the hands; impact off the ball."""
    n = len(grip)
    imp = int(np.clip(round(impact), 1, n - 1)) if np.isfinite(impact) else n // 2
    # Address: where the hands rest at the start.
    rest = np.nanmedian(grip[:max(3, int(0.3 * fps))], axis=0)
    moved = np.hypot(*(grip - rest).T) > 0.12 * torso
    # Top: the hands at their highest before impact.
    window = grip[:imp, 1]
    top = int(np.nanargmin(window)) if np.isfinite(window).any() else imp // 2
    # Takeaway: the hands start moving and keep moving, before the top.
    takeaway = 0
    for f in range(top - 1, -1, -1):
        if not moved[f]:
            takeaway = f + 1
            break
    # Then back up to where the clubhead itself first left the ball (it moves first). The
    # club can be missing for a few blurred frames around here, so only its resting place
    # at address has to be known; frames without it are skipped on the way back.
    head = club.head
    address = head[:max(3, takeaway // 2)]
    if np.isfinite(address).all(axis=1).any():
        ref = np.nanmedian(address, axis=0)
        for f in range(takeaway, max(-1, takeaway - int(0.6 * fps)), -1):
            if np.isfinite(head[f]).all() and np.hypot(*(head[f] - ref)) < 0.08 * torso:
                takeaway = f + 1 if f < takeaway else takeaway
                break
    # Finish: the hands settle after impact (speed under a tenth of a torso per second).
    speed = np.hypot(*np.gradient(grip, axis=0).T) * fps
    finish = n - 1
    for f in range(imp + int(0.3 * fps), n - 3):
        if (speed[f:f + 3] < 0.35 * torso).all():
            finish = f
            break
    return Swing(takeaway, top, impact if np.isfinite(impact) else float(imp), finish, fps)


def use_flight_model(ball: BallTrack, est: dict, *, fps: float, extend_seconds: float) -> None:
    """Past the last sighting, draw the fitted flight (projected) instead of the quadratic."""
    if ball.path is None or not ball.flight:
        return
    b, cam = est["best"], est["camera"]
    last = ball.flight[-1]
    path = b.path_3d
    t_frames = b.impact + path[:, 0] * fps
    tail = path[t_frames >= last]
    if len(tail) < 2:
        return
    uv, z = cam.project(tail[:, 1:])
    keep = z > 0.5
    proj = np.column_stack([b.impact + tail[keep, 0] * fps, uv[keep]])
    # Start the projection where the ball was last seen (the fit's off by a px or two there).
    proj[:, 1:] += ball.path[ball.path[:, 0] <= last][-1, 1:] - proj[0, 1:]
    seen = ball.path[ball.path[:, 0] <= last]
    ball.path = np.concatenate([seen, proj[1:]])
