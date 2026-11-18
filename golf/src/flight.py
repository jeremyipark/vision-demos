"""Ball speed, launch, carry and apex: a ball-flight model fitted to where the ball was seen.

The ball's size on the tee (a golf ball is 42.7 mm) and the focal length put the tee
in 3D. A launch (speed, angle up, angle sideways, backspin, spin-axis tilt, moment of
impact) is flown forward with gravity, drag and spin lift, projected through the
(tilted) phone, and fitted to every sighting. Sightings are first put in the impact
frame's coordinates, so the phone's drift isn't mistaken for curve. The fitted flight
is then flown to landing for carry and apex.

An estimate from one phone: it leans on the ball's size on the tee (which sets the
scale), the focal length, the tilt and the drag and lift model, so it is re-fitted
across a spread of each, one at a time, and reported as a range. Not a confidence
interval: the fit itself can't tell the ends of the range apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DIAMETER = 0.04267     # m
MASS = 0.04593         # kg
AREA = np.pi * (DIAMETER / 2) ** 2
RHO = 1.2              # air, kg/m^3
G = 9.81


@dataclass
class Launch:
    speed: float           # m/s
    launch_deg: float      # up from level
    side_deg: float        # right of the phone's forward direction
    spin_rpm: float        # backspin; poorly pinned down, kept for the record
    axis_tilt_deg: float   # spin axis tilt; negative curves left (a draw, for a right-hander)
    impact: float          # fractional frame
    carry_m: float
    apex_m: float
    hang_s: float
    rms_px: float
    path_3d: np.ndarray = field(repr=False, default=None)   # (m, 4): t, x, y, z (world)


class Camera:
    """Pinhole at the origin, tilted down by *pitch_deg*. World: x right, y up, z forward."""

    def __init__(self, focal: float, cx: float, cy: float, pitch_deg: float):
        self.f, self.cx, self.cy = focal, cx, cy
        th = np.radians(pitch_deg)
        self.R = np.array([[1.0, 0, 0],
                           [0, -np.cos(th), -np.sin(th)],   # image down
                           [0, -np.sin(th), np.cos(th)]])   # optical axis

    def project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c = pts @ self.R.T
        return np.stack([self.cx + self.f * c[:, 0] / c[:, 2],
                         self.cy + self.f * c[:, 1] / c[:, 2]], axis=1), c[:, 2]

    def back_project(self, uv, depth: float) -> np.ndarray:
        ray = np.array([(uv[0] - self.cx) / self.f, (uv[1] - self.cy) / self.f, 1.0]) * depth
        return self.R.T @ ray


def fly(start: np.ndarray, speed, launch_deg, side_deg, spin_rpm, tilt_deg, *, t_end: float,
        cd0: float = 0.22, lift_scale: float = 1.0, dt: float = 1 / 300,
        ground: float | None = None) -> np.ndarray:
    """``(m, 4)``: t, x, y, z. Stops at *t_end*, or on coming back down to *ground*."""
    la, sd = np.radians(launch_deg), np.radians(side_deg)
    vel = speed * np.array([np.cos(la) * np.sin(sd), np.sin(la), np.cos(la) * np.cos(sd)])
    pos = start.astype(float).copy()
    omega = spin_rpm * 2 * np.pi / 60
    tl = np.radians(tilt_deg)
    out = [(0.0, *pos)]
    t = 0.0
    while t < t_end:
        sp = np.linalg.norm(vel)
        s = omega * DIAMETER / 2 / max(sp, 1e-6)          # spin factor
        cd = cd0 + 0.2 * s
        cl = lift_scale * min(0.35, 1.2 * s / (1 + 2 * s))
        right = np.cross([0.0, 1.0, 0.0], vel)
        right /= np.linalg.norm(right) + 1e-9
        up = np.cross(vel, right) / max(sp, 1e-6)           # perpendicular to the flight, upward
        side = np.cross(up, vel) / max(sp, 1e-6)
        lift = np.cos(tl) * up + np.sin(tl) * side
        acc = (np.array([0.0, -G, 0.0]) - 0.5 * RHO * AREA * cd * sp * vel / MASS
               + 0.5 * RHO * AREA * cl * sp * sp * lift / MASS)
        vel = vel + acc * dt
        pos = pos + vel * dt
        t += dt
        out.append((t, *pos))
        if ground is not None and t > 0.3 and pos[1] < ground:
            break
    return np.array(out)


def _fit_once(cam: Camera, tee_world: np.ndarray, frames: np.ndarray, obs: np.ndarray,
              sizes: dict[int, float], fps: float, rest_frame: int, *, cd0: float,
              lift_scale: float, starts) -> Launch | None:
    from scipy.optimize import least_squares

    def residuals(p):
        speed, la, sd, rpm, tilt, impact = p
        ts = (frames - impact) / fps
        path = fly(tee_world, speed, la, sd, rpm, tilt, t_end=ts.max() + 0.01, cd0=cd0,
                   lift_scale=lift_scale)
        pts = np.stack([np.interp(ts, path[:, 0], path[:, k]) for k in (1, 2, 3)], axis=1)
        uv, z = cam.project(pts)
        r = list(((uv - obs) / 3.0).ravel())
        for k, f in enumerate(frames):    # SAM's sizes, the frames it saw the ball: depth
            if f in sizes:
                r.append((cam.f * DIAMETER / z[k] - sizes[f]) / 1.5)
        return np.array(r)

    lo = [10, 0, -30, 0, -45, rest_frame]
    hi = [90, 50, 30, 7000, 45, rest_frame + 1]
    best = None
    for speed, la in starts:
        try:
            res = least_squares(residuals, [speed, la, 0, 2500, 0, rest_frame + 0.5], bounds=(lo, hi),
                                x_scale=[10, 5, 5, 1000, 10, 0.3])
        except ValueError:
            continue
        if best is None or res.cost < best.cost:
            best = res
    if best is None:
        return None
    speed, la, sd, rpm, tilt, impact = best.x
    path = fly(tee_world, speed, la, sd, rpm, tilt, t_end=15, cd0=cd0, lift_scale=lift_scale,
               ground=tee_world[1])
    land = path[-1, 1:]
    rms = float(np.sqrt(np.mean(best.fun[: 2 * len(frames)] ** 2)) * 3.0)
    return Launch(speed, la, sd, rpm, tilt, impact,
                  carry_m=float(np.hypot(land[0] - tee_world[0], land[2] - tee_world[2])),
                  apex_m=float(path[:, 2].max() - tee_world[1]), hang_s=float(path[-1, 0]),
                  rms_px=rms, path_3d=path)


def estimate(ball, *, width: int, height: int, fps: float, focal: float, pitch_deg: float,
             drift: dict[int, np.ndarray], sam_frames: list[int], min_sightings: int = 8,
             focal_spread: float = 0.1, pitch_spread: float = 4.0,
             drag_spread: float = 0.04) -> dict | None:
    """The best fit, plus the range its numbers span over the assumptions' spread."""
    if ball.rest_until < 0 or len(ball.flight) < min_sightings:
        return None
    frames = np.array(ball.flight)
    shift = np.array([drift.get(int(f), np.zeros(2)) for f in frames])
    obs = np.stack([ball.x[frames], ball.y[frames]], axis=1) - shift
    tee_uv = (ball.x[ball.rest_until], ball.y[ball.rest_until])
    sizes = {int(f): float(ball.d[f]) for f in sam_frames}

    def fit(f, pitch, cd0, lift, starts, diameter=ball.diameter):
        cam = Camera(f, width / 2, height / 2, pitch)
        tee = cam.back_project(tee_uv, f * DIAMETER / diameter)
        out = _fit_once(cam, tee, frames, obs, sizes, fps, ball.rest_until, cd0=cd0,
                        lift_scale=lift, starts=starts)
        return out, cam

    grid = [(s, la) for s in (35, 55, 75) for la in (12, 20, 30)]
    best, cam = fit(focal, pitch_deg, 0.22, 1.0, grid)
    if best is None:
        return None
    seed = [(best.speed, best.launch_deg)]
    spread = []
    # The ball's size on the tee sets the scale (its depth, so every metre), and it
    # wobbles a px or two between frames: the biggest of these, so it's swept too.
    d_lo, d_hi = getattr(ball, "diameter_spread", None) or (ball.diameter, ball.diameter)
    d0 = ball.diameter
    for f, pitch, cd0, dm in ((focal * (1 - focal_spread), pitch_deg, 0.22, d0),
                              (focal * (1 + focal_spread), pitch_deg, 0.22, d0),
                              (focal, pitch_deg - pitch_spread, 0.22, d0),
                              (focal, pitch_deg + pitch_spread, 0.22, d0),
                              (focal, pitch_deg, 0.22 - drag_spread, d0),
                              (focal, pitch_deg, 0.22 + drag_spread, d0),
                              (focal, pitch_deg, 0.22, d_lo),
                              (focal, pitch_deg, 0.22, d_hi)):
        if dm != d0 and abs(dm - d0) < 0.25:
            continue
        alt, _ = fit(f, pitch, cd0, 1.0, seed, diameter=dm)
        if alt is not None and alt.rms_px < 2 * best.rms_px + 1:
            spread.append(alt)
    every = [best] + spread

    def span(get):
        vals = [get(x) for x in every]
        return [round(float(min(vals)), 1), round(float(max(vals)), 1)]

    return {"best": best, "camera": cam,
            "range": {"speed_mph": span(lambda x: x.speed * 2.237),
                      "launch_deg": span(lambda x: x.launch_deg),
                      "carry_yd": span(lambda x: x.carry_m * 1.0936),
                      "apex_m": span(lambda x: x.apex_m)},
            "fits": len(every)}
