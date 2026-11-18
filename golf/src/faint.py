"""The ball after SAM loses it: a few pixels across, so found by motion instead.

Once the ball is a few pixels wide (and here, flying alongside a net pole) SAM 3.1
finds nothing, even on an upscaled crop. The phone only creeps a few px a frame, so the ball is
the one small thing that moves: each frame is compared with the median of frames a
few steps either side (where the ball isn't), only in a window around where the
ball's last steps say it should be next.
"""

from __future__ import annotations

import cv2
import numpy as np


def read_frames(mp4, start: int, stop: int) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(str(mp4))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start))
    out = {}
    for f in range(max(0, start), stop):
        ok, img = cap.read()
        if not ok:
            break
        out[f] = img
    cap.release()
    return out


def _blobs(frames, f: int, guess: np.ndarray, radius: float, max_d: float, gap: int):
    """Moving blobs near *guess* on frame *f*: ``[(x, y, d), ...]``."""
    h, w = frames[f].shape[:2]
    r = int(np.ceil(radius))
    x0, y0 = int(max(0, guess[0] - r)), int(max(0, guess[1] - r))
    x1, y1 = int(min(w, guess[0] + r + 1)), int(min(h, guess[1] + r + 1))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return []
    # The phone jitters a pixel or so, which lights up every edge (a pole, the net) in a
    # plain difference. So each reference is first aligned to this frame, sub-pixel, on a
    # wider patch around the window.
    m = 48
    X0, Y0, X1, Y1 = max(0, x0 - m), max(0, y0 - m), min(w, x1 + m), min(h, y1 + m)
    cur = frames[f][Y0:Y1, X0:X1].astype(np.float32)
    cur_g = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    win = cv2.createHanningWindow(cur_g.shape[::-1], cv2.CV_32F)
    refs = []
    for g in (f - gap - 1, f - gap, f + gap, f + gap + 1):
        if g not in frames:
            continue
        ref = frames[g][Y0:Y1, X0:X1].astype(np.float32)
        (dx, dy), _ = cv2.phaseCorrelate(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), cur_g, win)
        shift = np.float32([[1, 0, dx], [0, 1, dy]])
        ref = cv2.warpAffine(ref, shift, ref.shape[1::-1], flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
        refs.append(ref[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0])
    if len(refs) < 2:
        return []
    bg = np.median(np.stack(refs), axis=0)
    diff = np.abs(cur[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] - bg).max(axis=2)
    thresh = max(14.0, float(np.median(diff) + 6 * np.median(np.abs(diff - np.median(diff)))))
    n, labels, stats, cents = cv2.connectedComponentsWithStats((diff > thresh).astype(np.uint8))
    out = []
    for i in range(1, n):
        bw, bh, area = stats[i][2], stats[i][3], stats[i][4]
        if area < 2 or max(bw, bh) > max_d or max(bw, bh) > 3 * min(bw, bh) + 2:
            continue
        out.append((x0 + cents[i][0], y0 + cents[i][1], float(max(bw, bh))))
    return out


def extend(ball, mp4, n_frames: int, fps: float, *, max_seconds: float, gap: int = 3,
           ratio: float = 0.72, corridor: float = 70.0, max_skip: int = 2,
           max_turn_deg: float = 40.0) -> list[int]:
    """Follow the ball past SAM's last sighting; adds to ``ball.flight``. Returns the frames added.

    Every moving blob in a corridor ahead of the ball is a candidate on each frame; the
    chain through them that best keeps the ball's motion (each step about ``ratio`` of the
    last, as it recedes, and in about the same direction) wins, found over the whole
    sequence at once so one wrong pick can't drag the track off. Frames may be skipped
    (the ball crossing a pole, say), at a cost.
    """
    if len(ball.flight) < 2 or ball.rest_until < 0:
        return []
    f0, f1 = ball.flight[-2], ball.flight[-1]
    p1 = np.array([ball.x[f1], ball.y[f1]])
    v1 = (p1 - [ball.x[f0], ball.y[f0]]) / (f1 - f0)
    stop = min(n_frames, f1 + int(max_seconds * fps) + 1)
    frames = read_frames(mp4, f1 - gap - 2, stop + gap + 2)
    h, w = frames[f1].shape[:2]
    max_d = 1.3 * float(ball.d[f1]) + 4

    def advance(pos, vel, steps):
        for _ in range(steps):
            vel = vel * ratio
            pos = pos + vel
        return pos, vel

    # Candidates: around where a ball slowing at `ratio` would be, with room to curve.
    cands: dict[int, list] = {}
    pos, vel = p1, v1
    for f in range(f1 + 1, stop):
        pos, vel = advance(pos, vel, 1)
        if not (0 <= pos[0] < w and 0 <= pos[1] < h):
            break
        cands[f] = _blobs(frames, f, pos, corridor + 0.5 * np.hypot(*(pos - p1)) ** 0.5 * 4,
                          max_d=max_d, gap=gap)

    # Viterbi-style: each state keeps its best chain's cost, position and velocity.
    start = {"f": f1, "pos": p1, "vel": v1, "cost": 0.0, "back": None}
    states = [start]
    by_frame: dict[int, list] = {f1: [start]}
    for f in sorted(cands):
        here = []
        for cx, cy, size in cands[f]:
            c = np.array([cx, cy])
            best = None
            for g in range(f - 1 - max_skip, f):
                for s_ in by_frame.get(g, []):
                    dt = f - s_["f"]
                    pred, _ = advance(s_["pos"], s_["vel"], dt)
                    move = (c - s_["pos"]) / dt
                    speed, prev = np.hypot(*move), np.hypot(*s_["vel"])
                    if speed > 1.2 * prev + 3:
                        continue        # it doesn't speed up
                    if prev > 8:
                        cos = move @ s_["vel"] / (speed * prev + 1e-9)
                        if np.degrees(np.arccos(np.clip(cos, -1, 1))) > max_turn_deg:
                            continue
                    sigma = 0.25 * np.hypot(*(pred - s_["pos"])) + 4
                    err = np.hypot(*(c - pred)) / sigma
                    cost = s_["cost"] + err ** 2 - (2.0 + 0.3 * min(size, 6)) + 1.5 * (dt - 1)
                    if best is None or cost < best["cost"]:
                        best = {"f": f, "pos": c, "vel": move, "cost": cost, "back": s_,
                                "size": size}
            if best is not None:
                here.append(best)
        by_frame[f] = here
        states += here

    end = min(states, key=lambda s_: s_["cost"])
    chain = []
    while end is not None and end["f"] != f1:
        chain.append(end)
        end = end["back"]
    added = []
    for s_ in reversed(chain):
        f = s_["f"]
        ball.x[f], ball.y[f], ball.d[f] = s_["pos"][0], s_["pos"][1], s_["size"]
        ball.flight.append(f)
        added.append(f)
    return added
