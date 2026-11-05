"""What a finished run hands back to a person: a few headline numbers and two CSVs.

The pipelines write JSON for the next program and text for the terminal. Someone
who uploaded a clip wants neither: they want the answer at a glance, and a table
they can open in a spreadsheet and plot. Every demo gets the same two files:

    <clip>_summary.csv   one row per rep, step, attempt or dancer pair
    <clip>_frames.csv    one row per video frame, every measurement the run took
                         (climbing: <clip>_holds.csv, one row per hold contact)

Only reads what a run already wrote; nothing here changes a pipeline. A demo
whose files are missing or malformed yields an empty Export carrying the reason,
not an exception, because the run itself succeeded and its video is still worth
showing.
"""

from __future__ import annotations

import csv
import json
import math
import zipfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Export:
    highlights: list[tuple[str, str]] = field(default_factory=list)   # (label, value)
    summary: list[dict] = field(default_factory=list)
    frames: list[dict] = field(default_factory=list)
    summary_label: str = "Summary"      # what one row of the summary is
    frames_label: str = "Frame by frame"
    frames_file: str = "frames"         # <clip>_<this>.csv
    problem: str = ""                   # why the export is empty, for the run log


# ── helpers ──────────────────────────────────────────────────────────────────

def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _find(run_dir: Path, name: str) -> list[Path]:
    return sorted(run_dir.rglob(name))


def _num(value, digits: int = 3):
    """A number for a CSV cell: rounded, and blank rather than NaN or None."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, float):
        return "" if math.isnan(value) else round(value, digits)
    return value


def _s(value, unit: str = "s", digits: int = 2) -> str:
    return "–" if value is None else f"{value:.{digits}f}{unit}"


def _series_rows(series: dict, *, time_key: str = "time", fps: float | None = None) -> list[dict]:
    """A dict of equal-length lists (nested one level for left/right) as rows.

    `{"time": [...], "clearance": {"left": [...], "right": [...]}}` becomes
    rows with `time_s, clearance_left, clearance_right`.
    """
    columns: dict[str, list] = {}
    for key, values in series.items():
        if isinstance(values, dict):
            for sub, inner in values.items():
                if isinstance(inner, list):
                    columns[f"{key}_{sub}"] = inner
        elif isinstance(values, list) and values and not isinstance(values[0], (list, dict)):
            columns[key] = values
        elif isinstance(values, list) and not values:
            continue
    if not columns:
        return []

    n = max(len(v) for v in columns.values())
    times = columns.pop(time_key, None)
    rows = []
    for i in range(n):
        row = {"frame": i}
        if times is not None and i < len(times):
            row["time_s"] = _num(times[i], 4)
        elif fps:
            row["time_s"] = round(i / fps, 4)
        for name, values in columns.items():
            row[name] = _num(values[i], 5) if i < len(values) else ""
        rows.append(row)
    return rows


def _tag_reps(rows: list[dict], reps: list[dict], start: str, end: str) -> list[dict]:
    """*rows* with a `rep` column, which rep each frame falls in (blank between
    reps), placed after the time, where someone filtering by rep looks for it."""
    spans = [(r["number"], r["frames"][start], r["frames"][end])
             for r in reps
             if r.get("frames", {}).get(start) is not None
             and r["frames"].get(end) is not None]
    tagged = []
    for row in rows:
        frame, time_s = row.pop("frame"), row.pop("time_s", "")
        rep = next((n for n, a, b in spans if a <= frame <= b), "")
        tagged.append({"frame": frame, "time_s": time_s, "rep": rep, **row})
    return tagged


# ── per demo ─────────────────────────────────────────────────────────────────

def _pull_ups(run_dir: Path) -> Export:
    data = next((d for p in _find(run_dir, "reps.json") if (d := _load(p))), None)
    if not data:
        return Export()
    reps = data.get("reps_detail") or []
    summary = []
    for r in reps:
        rom = r.get("rom") or {}
        summary.append({
            "rep": r["number"],
            "full_range_of_motion": "yes" if rom.get("complete") else "no",
            "note": "" if rom.get("complete") else rom.get("reason", ""),
            "start_s": _num(r.get("valley_time")),
            "pull_starts_s": _num(r.get("onset_time")),
            "top_s": _num(r.get("peak_time")),
            "end_s": _num(r.get("end_time")),
            "ascent_s": _num(r.get("ascent_seconds")),
            "pulling_s": _num(r.get("moving_seconds")),
            "descent_s": _num(r.get("descent_seconds")),
            "rep_duration_s": _num(r.get("total_seconds")),
            "hang_elbow_deg": _num(rom.get("hang_degrees"), 1),
            "top_elbow_deg": _num(rom.get("peak_degrees"), 1),
            "head_clearance_torso_lengths": _num(rom.get("clearance_torso_lengths")),
        })

    frames = _tag_reps(_series_rows(data.get("series") or {}, fps=data.get("fps")),
                       reps, "valley", "end")

    n, full = data.get("reps", len(reps)), data.get("full_rom_reps")
    highlights = [("Reps", str(n))]
    if full is not None:
        highlights.append(("Full range of motion", f"{full} of {data.get('graded_reps', n)}"))
    highlights += [("Mean rep", _s(data.get("mean_rep_seconds"))),
                   ("Mean ascent", _s(data.get("mean_ascent_seconds")))]
    return Export(highlights, summary, frames, summary_label="Per rep")


def _chin_ups(run_dir: Path) -> Export:
    data = next((d for p in _find(run_dir, "reps.json") if (d := _load(p))), None)
    if not data:
        return Export()
    reps = data.get("reps_detail") or []
    summary = [{
        "rep": r["number"],
        "start_s": _num(r.get("valley_time")),
        "pull_starts_s": _num(r.get("onset_time")),
        "top_s": _num(r.get("peak_time")),
        "end_s": _num(r.get("end_time")),
        "ascent_s": _num(r.get("ascent_seconds")),
        "pulling_s": _num(r.get("moving_seconds")),
        "descent_s": _num(r.get("descent_seconds")),
        "rep_duration_s": _num(r.get("total_seconds")),
        "height": _num(r.get("height"), 4),
    } for r in reps]

    frames = _tag_reps(_series_rows(data.get("series") or {}, fps=data.get("fps")),
                       reps, "valley", "end")

    return Export([
        ("Reps", str(data.get("reps", len(reps)))),
        ("Mean rep", _s(data.get("mean_rep_seconds"))),
        ("Mean ascent", _s(data.get("mean_ascent_seconds"))),
        ("Mean descent", _s(data.get("mean_descent_seconds"))),
    ], summary, frames, summary_label="Per rep")


def _deadlift(run_dir: Path) -> Export:
    data = next((d for p in _find(run_dir, "reps.json") if (d := _load(p))), None)
    if not data:
        return Export()
    reps = data.get("reps_detail") or []
    summary = []
    for r in reps:
        back = r.get("back") or {}
        summary.append({
            "rep": r["number"],
            "back": back.get("label", ""),
            "rounded_probability": _num(back.get("p")),
            "start_s": _num(r.get("onset_time")),
            "lockout_s": _num(r.get("peak_time")),
            "end_s": _num(r.get("end_time")),
            "lift_s": _num(r.get("lift_seconds")),
            "lower_s": _num(r.get("lower_seconds")),
            "bar_travel_cm": _num(r.get("height_cm"), 1),
            "max_hip_angle_deg": _num(r.get("hip_deg_max"), 1),
        })

    series = dict(data.get("series") or {})
    reads = next((d for p in _find(run_dir, "reads.json") if (d := _load(p))), None)
    if reads and isinstance(reads.get("series"), dict):
        p = reads["series"].get("p_rounded")
        if isinstance(p, list):
            series["back_rounded_probability"] = p
    frames = _tag_reps(_series_rows(series, fps=data.get("fps")), reps, "onset", "end")

    rounded = sum(1 for r in summary if r["back"] == "rounded")
    highlights = [("Reps", str(data.get("reps", len(reps))))]
    if any(r["back"] for r in summary):
        highlights.append(("Rounded back", f"{rounded} of {len(summary)}"))
    highlights += [("Mean lift", _s(data.get("mean_lift_seconds"))),
                   ("Bar travel", _s(data.get("typical_rep_size"), " cm", 1))]
    return Export(highlights, summary, frames, summary_label="Per rep")


def _running(run_dir: Path) -> Export:
    data = next((d for p in _find(run_dir, "gait.json") if (d := _load(p))), None)
    if not data:
        return Export()
    summary = [{
        "step": s["number"],
        "foot": s.get("foot", ""),
        "strike_s": _num(s.get("time"), 4),
        "contact_s": _num(s.get("contact_seconds"), 4),
        "knee_flexion_at_strike_deg": _num(s.get("knee_flexion_deg"), 1),
        "peak_clearance_leg_lengths": _num(s.get("peak_clearance"), 4),
        "step_interval_s": _num(s.get("step_interval"), 4),
        "stride_s": _num(s.get("stride_seconds"), 4),
    } for s in data.get("steps_detail") or []]

    frames = _series_rows(data.get("series") or {}, fps=data.get("fps"))

    per_foot = data.get("steps_per_foot") or {}
    contact = data.get("contact_seconds") or {}
    knee = data.get("knee_flexion_at_strike_deg") or {}

    def lr(d: dict, fmt) -> str:
        if "left" not in d or "right" not in d:
            return "–"
        return f"{fmt(d['left'])} L · {fmt(d['right'])} R"

    return Export([
        ("Cadence", _s(data.get("cadence_spm"), " spm", 1)),
        ("Foot strikes", f"{data.get('steps', len(summary))}"
                         + (f" ({per_foot['left']} L · {per_foot['right']} R)"
                            if "left" in per_foot and "right" in per_foot else "")),
        ("Contact time", lr(contact, lambda v: f"{v:.2f}s")),
        ("Knee at strike", lr(knee, lambda v: f"{v['mean']:.0f}°")),
    ], summary, frames, summary_label="Per step")


def _rock_climbing(run_dir: Path) -> Export:
    attempts = [(p.parent, d) for p in _find(run_dir, "sequence.json") if (d := _load(p))]
    if not attempts:
        return Export()
    summary = [{
        "attempt": d.get("label") or folder.name,
        "video": d.get("video", ""),
        "topped_out": "yes" if d.get("topped_out") else "no",
        "time_on_wall_s": _num(d.get("elapsed_seconds")),
        "holds_detected": d.get("holds_detected"),
        "holds_used": d.get("holds_used"),
        "sequence": " ".join(str(h) for h in d.get("sequence_aligned") or d.get("sequence") or []),
    } for folder, d in attempts]

    # No per-frame measurement here worth a spreadsheet: the detail is per hold.
    frames = []
    for folder, d in attempts:
        label = d.get("label") or folder.name
        try:
            with (folder / "hold_times.csv").open(newline="") as handle:
                frames += [{"attempt": label, **row} for row in csv.DictReader(handle)]
        except OSError:
            pass

    sends = [r for r in summary if r["topped_out"] == "yes"]
    if len(summary) == 1:
        r = summary[0]
        highlights = [("Topped out", r["topped_out"]),
                      ("Time on wall", _s(r["time_on_wall_s"] or None)),
                      ("Holds used", f"{r['holds_used']} of {r['holds_detected']}")]
    else:
        fastest = min((r for r in sends if r["time_on_wall_s"] not in ("", None)),
                      key=lambda r: r["time_on_wall_s"], default=None)
        fewest = min(sends, key=lambda r: r["holds_used"] or 1e9, default=None)
        highlights = [("Attempts", str(len(summary))),
                      ("Topped out", f"{len(sends)} of {len(summary)}")]
        if fastest:
            highlights.append(("Fastest", f"{fastest['attempt']} ({_s(fastest['time_on_wall_s'])})"))
        if fewest:
            highlights.append(("Fewest holds", f"{fewest['attempt']} ({fewest['holds_used']})"))
    return Export(highlights, summary, frames, summary_label="Per attempt",
                  frames_label="Per hold", frames_file="holds")


def _dance_sync(run_dir: Path) -> Export:
    data = next((d for p in _find(run_dir, "similarity.json") if (d := _load(p))), None)
    if not data:
        return Export()
    if not data.get("scored_frames"):
        return Export([("Frames scored", "0")], summary_label="Per pair")

    pairs = ["-".join(map(str, p)) for p in data.get("pair_labels") or []]
    stats = data.get("pairs") or {}
    summary = [{"pair": name,
                "mean_sync_percent": _num(stats.get(name, {}).get("mean"), 1),
                "median_sync_percent": _num(stats.get(name, {}).get("median"), 1)}
               for name in pairs]

    series = data.get("series") or {}
    slots = data.get("slots")
    n_dancers = len(series.get("per_dancer", [[]])[0]) if series.get("per_dancer") else 0
    dancers = ([str(s) for s in slots] if isinstance(slots, list) and len(slots) == n_dancers
               else [str(i + 1) for i in range(n_dancers)])

    frames = []
    times = series.get("time") or []
    for i, frame in enumerate(series.get("frame") or []):
        row = {"frame": frame, "time_s": _num(times[i], 4) if i < len(times) else "",
               "sync_percent": series["score"][i] if i < len(series.get("score", [])) else ""}
        for key, name in (("posture", "posture_percent"), ("timing", "timing_percent")):
            if i < len(series.get(key) or []):
                row[name] = series[key][i]
        for j, name in enumerate(pairs):
            pw = series.get("pairwise") or []
            if i < len(pw) and j < len(pw[i]):
                row[f"pair_{name}_percent"] = pw[i][j]
        for j, name in enumerate(dancers):
            pd = series.get("per_dancer") or []
            if i < len(pd) and j < len(pd[i]):
                row[f"dancer_{name}_percent"] = pd[i][j]
        frames.append(row)

    score = data.get("score_percent") or {}
    return Export([
        ("Sync score", _s(score.get("median"), "%", 0)),
        ("Mean", _s(score.get("mean"), "%", 0)),
        ("Frames scored", str(data.get("scored_frames"))),
        ("Coverage", _s((data.get("coverage") or 0) * 100, "%", 0)),
    ], summary, frames, summary_label="Per pair")


def _basketball(run_dir: Path) -> Export:
    data = next((d for p in _find(run_dir, "shots.json") if (d := _load(p))), None)
    if not data:
        return Export()
    shots = data.get("shots") or []
    summary = []
    for s in shots:
        x, y = s.get("court_xy_m") or (None, None)
        summary.append({
            "shot": s.get("shot"),
            "result": s.get("result", ""),
            "reason": s.get("reason", ""),
            "motion_starts_s": _num(s.get("motion_s")),
            "release_s": _num(s.get("release_s")),
            "side_on": "yes" if s.get("side_on") else "no",
            "elbow_set_deg": _num(s.get("elbow_set_deg"), 1),
            "elbow_release_deg": _num(s.get("elbow_release_deg"), 1),
            "court_x_m": _num(x, 2),
            "court_y_m": _num(y, 2),
            "distance_ft": _num(s.get("distance_ft"), 1),
        })

    # The pipeline already writes the per-frame table; this only re-reads it.
    frames = []
    for path in _find(run_dir, "*_frames.csv")[:1]:
        with path.open(newline="") as handle:
            frames = list(csv.DictReader(handle))

    made, attempts = data.get("made", 0), data.get("attempts", len(shots))
    highlights = [("Made", f"{made} of {attempts}")]
    if attempts:
        highlights.append(("FG", f"{made / attempts:.0%}"))
    longest = max((s["distance_ft"] for s in shots
                   if s.get("result") == "make" and s.get("distance_ft") is not None),
                  default=None)
    if longest is not None:
        highlights.append(("Longest make", f"{longest:.0f} ft"))
    elbows = [s["elbow_release_deg"] for s in shots if s.get("elbow_release_deg") is not None]
    if elbows:
        highlights.append(("Elbow at release",
                           f"{sum(elbows) / len(elbows):.0f}° ({len(elbows)} side-on)"))
    return Export(highlights, summary, frames, summary_label="Per shot")


BUILDERS = {
    "basketball": _basketball,
    "pull_ups": _pull_ups,
    "chin_ups": _chin_ups,
    "deadlift": _deadlift,
    "running": _running,
    "rock_climbing": _rock_climbing,
    "dance_sync": _dance_sync,
}


# The demos where one clip is one set, so several clips are compared set by set.
SET_DEMOS = {"pull_ups", "chin_ups", "deadlift"}


def source_clip(folder: Path) -> str | None:
    """The name of the clip a pipeline's output *folder* was made from."""
    data = _load(folder / "run.json") or {}
    path = (data.get("input") or {}).get("path")
    return Path(path).name if path else None


def _by_set(builder, run_dir: Path, sets: list[str]) -> Export:
    """One Export across several sets: every row gains a `set` column, and each
    headline number reads set by set, `5 · 4`, in the order the sets were filmed."""
    folders = {source_clip(p.parent): p.parent for p in _find(run_dir, "run.json")}
    parts, missing = [], []
    for n, name in enumerate(sets, 1):
        folder = folders.get(name)
        part = builder(folder) if folder else Export()
        if not (part.summary or part.highlights):
            missing.append(f"set {n} ({name})")
        parts.append((n, part))
    done = [(n, e) for n, e in parts if e.summary or e.highlights]
    if not done:
        return Export(problem="no set wrote the files it reads")

    labels = list(dict.fromkeys(k for _, e in done for k, _ in e.highlights))
    highlights = [("Sets", str(len(done)))] + [
        (label, " · ".join(dict(e.highlights).get(label, "–") for _, e in done))
        for label in labels]
    first = done[0][1]
    return Export(
        highlights,
        [{"set": n, **row} for n, e in done for row in e.summary],
        [{"set": n, **row} for n, e in done for row in e.frames],
        summary_label=first.summary_label, frames_label=first.frames_label,
        frames_file=first.frames_file,
        problem=("nothing from " + ", ".join(missing)) if missing else "")


def build(demo_key: str, run_dir: Path, sets: list[str] = ()) -> Export:
    """*sets* names the clips in set order, when one run holds several sets."""
    builder = BUILDERS.get(demo_key)
    if builder is None:
        return Export()
    try:
        result = (_by_set(builder, run_dir, list(sets))
                  if demo_key in SET_DEMOS and len(sets) > 1 else builder(run_dir))
    except Exception as exc:   # a run that succeeded keeps its video either way
        return Export(problem=f"{type(exc).__name__}: {exc}")
    if not (result.highlights or result.summary or result.frames):
        result.problem = result.problem or "the run wrote none of the files it reads"
    return result


def write_csv(rows: list[dict], path: Path) -> Path | None:
    if not rows:
        return None
    # Union of keys, in first-seen order: rows can differ (a rep with no ROM read).
    fields: dict[str, None] = {}
    for row in rows:
        fields.update(dict.fromkeys(row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), restval="")
        writer.writeheader()
        writer.writerows(rows)
    return path


# Already compressed: deflating them again costs time and saves nothing.
_STORED = {".mp4", ".mov", ".m4v", ".png", ".jpg", ".jpeg", ".zip"}


def write_zip(entries: list[tuple[Path, str]], path: Path) -> Path | None:
    """*entries* (file, name inside the archive) as one zip at *path*.

    Files that have gone missing are skipped: an old run's directory may have
    been swept since it finished. None if nothing was left to put in it.
    """
    present = [(src, name) for src, name in entries if src.is_file()]
    if not present:
        return None
    with zipfile.ZipFile(path, "w") as archive:
        for src, name in present:
            method = zipfile.ZIP_STORED if src.suffix.lower() in _STORED else zipfile.ZIP_DEFLATED
            archive.write(src, name, compress_type=method)
    return path
