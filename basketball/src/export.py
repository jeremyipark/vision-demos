"""CSV and JSON exports: one row per shot, and one row per frame."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def _num(value, digits: int = 2):
    return "" if value is None or not np.isfinite(value) else round(float(value), digits)


def write(shots, angle: np.ndarray, side_on: np.ndarray, track, fps: float, run_dir: Path,
          stem: str) -> list[Path]:
    shots_csv = run_dir / f"{stem}_shots.csv"
    with shots_csv.open("w", newline="") as fh:
        out = csv.writer(fh)
        out.writerow(["shot", "result", "reason", "release_s", "side_on", "elbow_set_deg",
                      "elbow_release_deg", "court_x_m", "court_y_m", "distance_ft"])
        for s in shots:
            x, y = s.court_xy or (None, None)
            out.writerow([s.number, s.label.lower(), s.reason, _num(s.release / fps),
                          s.side_on, _num(s.elbow_set, 1), _num(s.elbow_release, 1), _num(x),
                          _num(y), _num(s.distance_m * 3.2808 if s.distance_m else None, 1)])

    frames_csv = run_dir / f"{stem}_frames.csv"
    shot_of = np.zeros(len(angle), int)
    for s in shots:
        shot_of[s.start:(s.decided or s.end) + 1] = s.number
    with frames_csv.open("w", newline="") as fh:
        out = csv.writer(fh)
        out.writerow(["frame", "t_s", "elbow_deg", "side_on", "ball_x_px", "ball_y_px", "shot"])
        for k in range(len(angle)):
            out.writerow([k, _num(k / fps, 3), _num(angle[k], 1), bool(side_on[k]),
                          _num(track.x[k], 1), _num(track.y[k], 1), shot_of[k] or ""])

    shots_json = run_dir / "shots.json"
    made = sum(1 for s in shots if s.made)
    shots_json.write_text(json.dumps({
        "fps": fps, "made": made, "attempts": len(shots),
        "fg_pct": round(made / len(shots), 3) if shots else None,
        "shots": [{**s.summary(fps),
                   "ball_path": [[f, round(x, 1), round(y, 1)] for f, x, y in s.path]}
                  for s in shots],
    }, indent=2))
    return [shots_csv, frames_csv, shots_json]
