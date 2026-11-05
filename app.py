"""Vision Demos: a Gradio front end for the pipelines in this repository.

The front door, not the workshop. Each demo is a script driven entirely by
module-level constants in its own `config.py`, and this exposes the two or three
of those that decide what a result means, not the hundred that decide how it is
computed. Tuning a pipeline is still `config.py` plus `python main.py`.

Nothing here modifies the demos, and nothing in them refers to this file:
`webapp/runner.py` rewrites their config from outside, in a subprocess. Delete this and
every pipeline still runs exactly as before.

    python app.py
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
from gradio import processing_utils

from webapp import charts, exports, waiting

REPO_DIR = Path(__file__).resolve().parent
# The demo directories sit beside this file, one per pipeline.
DEMOS_DIR = REPO_DIR

# The gateway's anonymous tier. A caller with no key at all reaches it too, but
# the OpenAI SDK insists on a non-empty api_key, and this is the documented
# spelling of the same thing.
ANON_KEY = "vlmrun"

# Spaces sets SPACE_ID in the container. It decides how the anonymous budget is
# described (one egress IP, shared by every visitor), not whether a key is used.
ON_SPACES = bool(os.getenv("SPACE_ID"))

# Where the server listens, read the way `launch()` itself reads it, and passed
# to it explicitly below so the two cannot disagree. Only a server reachable
# from this machine alone counts as "the person clicking is the person paying":
# a Space, a VM bound to 0.0.0.0 or a share link all put strangers in front of
# it, and none of them should be billed to the key on disk by default.
SERVER_NAME = os.getenv("GRADIO_SERVER_NAME") or "127.0.0.1"
SHARE = os.getenv("GRADIO_SHARE", "").lower() in ("1", "true", "yes")
LOOPBACK = {"127.0.0.1", "localhost", "::1"}
LOCAL_ONLY = not ON_SPACES and not SHARE and SERVER_NAME in LOOPBACK

# The escape hatch for a deployment that does want to serve its own key.
# Off unless explicitly set, so a VLMRUN_API_KEY secret added to a public
# deployment for some other reason cannot silently start billing every
# visitor's run to it.
SERVE_AMBIENT_KEY = os.getenv("VLMRUN_SERVE_AMBIENT_KEY") == "1"

def _retention_mode() -> str:
    """Where a browser tab's files live.

    On this machine alone they stay, shared with the command line, so a local
    session keeps its history. A Space, a share link, or any other bind keeps
    them only while that tab is connected. Switching demos does not end the
    tab. ``VLMRUN_RETENTION=session`` or ``persistent`` overrides the choice.
    """
    choice = os.getenv("VLMRUN_RETENTION", "").strip().lower()
    if choice in ("session", "persistent"):
        return choice
    return "persistent" if LOCAL_ONLY else "session"


RETENTION = _retention_mode()
SESSION_RETENTION = RETENTION == "session"

# The gateway's published rate limits, copied from its docs rather than read at
# runtime: it reports the per-minute figure in a header but not the hourly or
# daily ones. Anonymous is bucketed by client IP, so a Space, which has one
# egress IP, shares a single bucket across every visitor. Keyed is per user.
# The caps below are about making that budget last, not about the model.
RATE_LIMITS_URL = "https://docs.vlm.run/gateway/rate-limits"
RATE_LIMITS_AS_OF = "10/9/26"
ANON_RATE_PER_MIN = 10
ANON_LIMITS = "10/min, 30/hr, 100/day"
KEYED_LIMITS = "240/min, 10,000/hr"
RATE_LIMITS_NOTE = (f"Limits shown as of {RATE_LIMITS_AS_OF} and may have changed. "
                    f"For the latest, see <a href='{RATE_LIMITS_URL}' "
                    f"target='_blank' rel='noopener'>{RATE_LIMITS_URL}</a>.")

# Budgets are in FRAMES, not seconds. Clip length is the wrong unit: the same
# 12 seconds is 360 frames at 30fps and 720 at 60fps, and frames are what gets
# uploaded, posed, and rendered. Seconds caps also do nothing for the request
# budget, since a pose call carries the whole clip in one request either way.
#
# 900 is the model's own default for video_max_frames. 3000 is not a limit the
# gateway imposes: a single 2250-frame request was measured working end to end,
# and this leaves headroom above that.
ANON_MAX_FRAMES = 900        # 30s at 30fps, 15s at 60fps
KEYED_MAX_FRAMES = 3000      # 100s at 30fps, 50s at 60fps

# A run is CPU-bound locally (ffmpeg, then the OpenCV render) and network-bound
# remotely. Neither is fast on a free Space, so one at a time.
# Fixed, and deliberately not tied to the render height. Pose coordinates come
# back normalized, so one inference serves any export size. Pinning this means
# re-rendering at a different height reuses the cached poses instead of paying
# for detection again. It costs a second ffmpeg pass when the two differ, which
# is local CPU rather than quota.
INFERENCE_HEIGHT = 720

RUN_TIMEOUT_SECONDS = 1800
KEEP_RUNS = 6                # per demo, before the oldest are swept
# The pipelines' caches (converted clips, poses, plate tracks, reads) are what
# make a repeat run free, but nothing else ever deletes them, and a Space's
# disk is shared by every visitor. Oldest go first past this, per demo.
CACHE_MAX_BYTES = 5 * 1024**3
# Largest file the upload control accepts. Both modes can preview and crop a
# long clip. The caps below are what a run is allowed to keep afterwards.
MAX_UPLOAD = "1gb"
_MB = 1024 ** 2
_GB = 1024 ** 3
# Anonymous is a full mode: 900 frames and 3 clips, with no key. A long upload
# is cut down to the window that budget analyzes, and only that window counts
# against this. It is a backstop for an enormous bitrate, not a reason to turn
# the mode off. A pasted key does not raise it until the gateway accepts it.
ANON_MAX_CLIP_BYTES = 500 * _MB
ANON_MAX_RUN_BYTES = 1500 * _MB
KEYED_MAX_CLIP_BYTES = _GB
KEYED_MAX_RUN_BYTES = 4 * _GB
# Climbing takes several clips in one run, and every clip is a full set of
# gateway calls: three is already most of the anonymous tier's hour.
MAX_CLIPS = {False: 3, True: 10}     # anonymous, keyed
# One cheap authenticated request, to tell a real key from any typed string.
GATEWAY_MODELS_URL = "https://gateway.vlm.run/v1/openai/models"
# Cropped previews live here, not in an untracked temp directory per crop.
CROP_DIR = Path(tempfile.gettempdir()) / "vision-demos-crops"
CROP_MAX_AGE = 3600

# Intermediates the pipelines write and then delete; never worth surfacing.
HIDDEN_VIDEOS = {"_raw.mp4", "_pair.mp4", "_solo.mp4"}


def _ambient_key() -> tuple[str | None, str]:
    """A key from the environment or the nearest `.env`, and where it came from.

    Mirrors what the demos do for themselves (`src/env.py`): an exported
    variable wins, then the nearest `.env` walking upward, which is what lets a
    checkout sit inside a larger tree that keeps one shared `.env` at its root.

    Resolved here rather than left to the demos because the subprocess env sets
    VLMRUN_API_KEY explicitly on every run, so by the time a pipeline looks, the
    decision has already been made.
    """
    key = os.getenv("VLMRUN_API_KEY")
    if key and key != ANON_KEY:
        return key, "VLMRUN_API_KEY in the environment"

    for directory in [REPO_DIR, *REPO_DIR.parents]:
        candidate = directory / ".env"
        if not candidate.is_file():
            continue
        try:
            from dotenv import dotenv_values
            value = (dotenv_values(candidate) or {}).get("VLMRUN_API_KEY")
        except Exception:
            value = None
        if value:
            # Relative to the repo, as `.env` or `../../.env`: the page shows
            # this, and an absolute path would show the home directory with it.
            return value, os.path.relpath(candidate, REPO_DIR)
    return None, ""


AMBIENT_KEY, AMBIENT_SOURCE = _ambient_key()

# Local development gets the author's own key without typing it every run; any
# server strangers can reach does not, because there the person clicking is
# not the person paying.
USE_AMBIENT = bool(AMBIENT_KEY) and (LOCAL_ONLY or SERVE_AMBIENT_KEY)

# Stands in for "the key found in the environment" in the key a run is given,
# so the key itself never makes a round trip through the browser. Only means
# anything where USE_AMBIENT allows that key to be used at all.
ENV_KEY_MARKER = "@env"


@dataclass
class Knob:
    """One exposed `config.py` constant, and how it is drawn."""

    key: str
    kind: str                       # slider | dropdown | checkbox | number | text | color
    label: str
    default: object = None
    info: str = ""
    choices: tuple = ()
    minimum: float = 0
    maximum: float = 100
    step: float = 1
    optional: bool = False          # tucked into the Optional section
    required: bool = False          # a run without it is refused, not guessed at


# The last entry of a color knob's dropdown; picking it shows a text box for
# any color the list does not have.
OTHER_COLOR = "Other…"
COLOR_MAX_CHARS = 30
COLOR_TEXT = re.compile(rf"[A-Za-z][A-Za-z \-]{{0,{COLOR_MAX_CHARS - 1}}}")
HOLD_COLORS = ("black", "blue", "gray", "green", "orange", "pink", "purple",
               "red", "white", "yellow")


@dataclass
class Demo:
    key: str
    title: str
    tagline: str
    model: str
    explainer: str                  # a .md file inside the demo directory, "#Section" for one part
    multi: bool = False             # several clips in one run
    # Each clip is one set of the same exercise: run one after another, in the
    # order they were filmed, and compared set against set.
    sets: bool = False
    upload_label: str = "Clips"
    # The constant naming the clip. The newer pipelines take `INPUT`, a clip or
    # a folder of them, and set INPUT_VIDEO themselves per clip.
    input_key: str = "INPUT_VIDEO"
    knobs: tuple[Knob, ...] = ()
    fixed: dict = field(default_factory=dict)   # overrides the UI does not expose
    notes: str = ""
    # What to keep, shown under the crop bar. Less video is less inference,
    # and the right window is different for every activity.
    crop_hint: str = ""
    # What a run goes through, read off the pipeline's log as it happens, and
    # the line under the figure.
    stages: tuple = waiting.POSE_STAGES
    wait_caption: str = "Running..."
    # The clip on top and a panel beneath it, as tall as the clip, rather than
    # side by side. EXPORT_HEIGHT is then the two together, and there is no
    # left half to cut out as an overlay-only video.
    stacked: bool = False

    @property
    def dir(self) -> Path:
        return DEMOS_DIR / self.key


# ── The demos ────────────────────────────────────────────────────────────────
# `fixed` is where a default that was tuned to the author's own clip gets
# neutralised. Those are not style choices. Left in place they quietly produce
# a wrong answer on someone else's upload, which is worse than an error.

DEMOS: tuple[Demo, ...] = (
    Demo(
        key="deadlift",
        multi=True, sets=True, upload_label="Clips, one per set",
        crop_hint="Crop to just before the bar leaves the floor on your first rep "
                  "and just after your last lockout.",
        title="Deadlift",
        tagline="Counts reps off the bar's height, and reads whether the back stays "
                "straight or rounds on each pull.",
        model="usyd-community/vitpose-plus-large + facebook/sam3.1 + "
              "google/gemma-4-26b-a4b-it",
        explainer="README.md#How it works",
        stages=waiting.DEADLIFT_STAGES,
        input_key="INPUT",
        knobs=(
            Knob("PLATE_DIAMETER_CM", "number", "Plate diameter (cm)", default=45,
                 minimum=20, maximum=60, optional=True,
                 info="The ruler for bar height. 45 is an Olympic plate; every "
                      "centimeter in the result scales with it."),
            Knob("SAMPLE_FPS", "slider", "Back reads per second", default=4,
                 minimum=1, maximum=15, step=1, optional=True,
                 info="Each read is one System One call; frames in between are "
                      "interpolated. More reads judge a fast pull more surely, and "
                      "cost proportionally more quota."),
        ),
        notes="Film **side on**, with the whole lifter and the near plate in shot. "
              "The bar's height is measured against the plate, and the back is "
              "read from the side.",
    ),
    Demo(
        key="pull_ups",
        multi=True, sets=True, upload_label="Clips, one per set",
        crop_hint="Crop to just before your first rep and just after your last.",
        title="Pull-ups",
        tagline="Counts reps, times each pull, and grades every rep's range of "
                "motion as full or partial.",
        model="usyd-community/vitpose-plus-large",
        explainer="how-it-works.md",
        stages=waiting.PULL_UP_STAGES,
        input_key="INPUT",
        fixed={
            # data/calibration.json is the author's own full set. Graded against
            # it, a stranger's reps are held to someone else's dead hang and
            # clearance; uncalibrated, each clip is its own reference.
            "USE_CALIBRATION": False,
        },
        notes="Film from **in front or behind**, with the whole body and the bar "
              "in shot. Range of motion is graded against this clip's own deepest "
              "hang, so include at least one full rep.",
    ),
    Demo(
        key="chin_ups",
        multi=True, sets=True, upload_label="Clips, one per set",
        crop_hint="Crop to just before your first rep and just after your last.",
        title="Chin-ups",
        tagline="Counts reps, and times the ascent and descent of each one.",
        model="usyd-community/vitpose-plus-large",
        explainer="rep-counting-explained.md",
        # The concentric phase, onset to peak, and only reps that get the chin
        # over the bar. Neither is a choice: a partial rep is not a rep.
        fixed={"REP_METRIC": "moving", "REQUIRE_HEAD_ABOVE_HANDS": True},
    ),
    Demo(
        key="running",
        crop_hint="Crop to steady running: skip speeding up at the start and "
                  "slowing down at the end.",
        title="Running",
        tagline="Cadence, every foot strike, and the average knee shape at contact.",
        model="usyd-community/vitpose-plus-large",
        explainer="cadence-explained.md",
        stages=waiting.RUNNING_STAGES,
        # Foot strikes measured against the ground: one stance level from the
        # whole clip, which suits the fixed camera almost every clip has.
        fixed={"FOOT_REFERENCE": "ground"},
    ),
    Demo(
        key="rock_climbing",
        crop_hint="Crop each clip from just before the climber leaves the "
                  "ground to the top.",
        title="Rock Climbing",
        tagline="Segments the holds, reads which ones the climber used and in what "
                "order, and compares attempts at the same route.",
        model="facebook/sam3.1 + usyd-community/vitpose-plus-large",
        explainer="route-reading-explained.md",
        stages=waiting.CLIMBING_STAGES,
        multi=True, upload_label="Clips of the same route",
        knobs=(
            Knob("HOLD_COLOR", "color", "Route color", choices=HOLD_COLORS,
                 required=True,
                 info="The color of the holds on the route."),
            Knob("HOLD_SAMPLE_FRAMES", "slider", "Frames sampled for holds",
                 default=8, minimum=4, maximum=16, step=1, optional=True,
                 info="Each is its own segmentation call. More frames find holds the "
                      "climber's body was hiding, and cost proportionally more quota."),
        ),
        fixed={
            # Upstream this renders only attempt 2, because analysing every clip
            # but drawing one is the fast path while iterating. A visitor wants
            # to see what they uploaded.
            "RENDER_ONLY": None,
            # The author's own route grade, which would otherwise be stamped
            # onto a stranger's climb now that there is no field to clear it.
            "ROUTE_GRADE": "",
        },
        notes="Upload two or more clips of the **same route** to get the comparison "
              "panel. That card is the whole point of the demo, and it cannot be "
              "built from a single attempt.",
    ),
    Demo(
        key="dance_sync",
        crop_hint="Crop to the routine itself: from once every dancer is in "
                  "shot to the final pose.",
        title="Dance Sync",
        tagline="Two dancers, one choreography. How closely do they match?",
        model="usyd-community/vitpose-plus-large",
        explainer="similarity-metric-explained.md",
        knobs=(
            Knob("SIMILARITY_METRIC", "dropdown", "Compared on",
                 default="angles", choices=("angles", "positions"),
                 info="Angles compare each limb's direction inside the dancer's own "
                      "torso frame, so a taller dancer is not penalised. Positions "
                      "compare hip-centred coordinates."),
            Knob("MAX_PERSONS", "number", "Cap dancers tracked", default=0,
                 minimum=0, maximum=12,
                 info="0 = no cap, which is the right answer for most clips: the "
                      "dancers enter at different times, and a cap drops whoever "
                      "arrives last rather than the spurious track you meant."),
        ),
        fixed={
            # Tuned to the author's own clip, the interesting 13 seconds of it.
            # On an arbitrary upload that window scores the wrong part, or
            # nothing at all.
            "SCORE_WINDOW": (None, None),
        },
        notes="Both dancers have to be **fully in shot** for a frame to be scored, so "
              "pick a window after everyone has entered. The start of a clip usually "
              "is not one. A run that scores no frames says so rather than guessing.",
    ),
    Demo(
        key="basketball",
        crop_hint="Crop to the shooting: from just before the first shot to "
                  "just after the last one lands.",
        title="Basketball",
        tagline="Calls every shot a make or a miss, plots each one on a court "
                "from above with the FG%, and graphs the shooting elbow.",
        model="usyd-community/vitpose-plus-large + facebook/sam3.1",
        explainer="how-it-works.md",
        stages=waiting.BASKETBALL_STAGES,
        input_key="INPUT",
        stacked=True,
        knobs=(
            Knob("PLAYER_HEIGHT_M", "number", "Shooter's height (m)", default=1.75,
                 minimum=1.2, maximum=2.3, step=0.01, optional=True,
                 info="The ruler for the court: every distance on the shot chart "
                      "scales with it, and so does which shots count as threes."),
            Knob("THREE_POINT_RADIUS_M", "dropdown", "Three-point line",
                 default=6.75, optional=True,
                 choices=(("FIBA, NCAA, WNBA (6.75 m)", 6.75), ("NBA (7.24 m)", 7.24),
                          ("US high school (6.02 m)", 6.02)),
                 info="Where the arc is drawn on the shot chart, and which shots "
                      "are threes."),
        ),
        fixed={
            # The author's own handles, which would otherwise be stamped in the
            # corner of a stranger's session if they were ever set upstream.
            "CREDIT": [],
        },
        notes="Film **landscape, on the 1x lens, with the phone set down and still**, "
              "the hoop and the shooter both in shot. The hoop is found once, so a "
              "camera that moves loses it. The elbow is only graded on side-on shots.",
    ),
)

DEMOS_BY_KEY = {d.key: d for d in DEMOS}


# ── Running one ──────────────────────────────────────────────────────────────

def _safe_session(session_hash: str | None) -> str | None:
    """A session id safe to use as one path component, or None."""
    if not session_hash:
        return None
    safe = re.sub(r"[^A-Za-z0-9_-]", "", session_hash)[:80]
    return safe or None


def _session_hash_of(request) -> str | None:
    if request is None:
        return None
    return _safe_session(getattr(request, "session_hash", None))


def _same_origin_browser_request(request) -> bool:
    """Whether a queued request came from the page that serves this app.

    The ambient key is a server-side credential, so callers do not get to use
    it merely by spelling ``@env`` in an API request. Browsers send Origin on
    the POSTs Gradio uses for events; requiring it to match Host prevents an
    unrelated page from driving a localhost app with the operator's key.
    """
    if request is None:
        return False
    try:
        headers = {str(k).lower(): str(v) for k, v in dict(request.headers).items()}
    except (AttributeError, TypeError, ValueError):
        return False
    origin = headers.get("origin", "")
    host = headers.get("x-forwarded-host", "").split(",", 1)[0].strip()
    host = host or headers.get("host", "").strip()
    fetch_site = headers.get("sec-fetch-site", "").lower()
    if not origin or not host or fetch_site not in ("", "same-origin", "none"):
        return False
    try:
        parsed = urllib.parse.urlsplit(origin)
    except ValueError:
        return False
    return (parsed.scheme in ("http", "https")
            and parsed.netloc.casefold() == host.casefold())


# Hashes whose tab is still connected, and Gradio temp files those tabs uploaded.
# The run directories themselves are found from the hash; this set is only the
# uploads Gradio stored somewhere else, plus the processes still writing.
_SESSION_LOCK = threading.Lock()
_LIVE_SESSIONS: set[str] = set()
_SESSION_PATHS: dict[str, set[Path]] = {}
_SESSION_PROCS: dict[str, set] = {}
ORPHAN_GRACE = 120
_SWEEPER_STARTED = False


def _hold_session(session_hash: str | None) -> None:
    safe = _safe_session(session_hash)
    if not SESSION_RETENTION or not safe:
        return
    with _SESSION_LOCK:
        _LIVE_SESSIONS.add(safe)


def _session_open(session_hash: str | None) -> bool:
    """False once this tab has disconnected. Local runs have no tab to lose."""
    if not SESSION_RETENTION:
        return True
    safe = _safe_session(session_hash)
    if not safe:
        return True
    with _SESSION_LOCK:
        return safe in _LIVE_SESSIONS


def _touch_session(request) -> str | None:
    """Mark *request*'s tab as still open, and return its safe id."""
    safe = _session_hash_of(request)
    if safe:
        _hold_session(safe)
    return safe


def _begin_session(request) -> tuple[str | None, bool]:
    """(session id, whether it exists only for this one call).

    A browser tab has an id and lives until the tab closes. A caller with no
    tab, such as the API, gets a directory that is removed when the call ends.
    """
    if not SESSION_RETENTION:
        return None, False
    safe = _touch_session(request)
    if safe:
        return safe, False
    safe = _safe_session("api-" + uuid.uuid4().hex)
    _hold_session(safe)
    return safe, True


def _remember_paths(session_hash: str | None, paths) -> None:
    """Gradio's own copies of uploads, deleted when the tab closes."""
    safe = _safe_session(session_hash)
    if not SESSION_RETENTION or not safe:
        return
    files = []
    for path in paths or []:
        if isinstance(path, dict):
            files.extend(v for v in (path.get("original"), path.get("current")) if v)
        elif path:
            files.append(path)
    if not files:
        return
    with _SESSION_LOCK:
        _SESSION_PATHS.setdefault(safe, set()).update(Path(p) for p in files)


def _track_process(session_hash: str | None, process) -> None:
    safe = _safe_session(session_hash)
    if not SESSION_RETENTION or not safe or process is None:
        return
    with _SESSION_LOCK:
        _SESSION_PROCS.setdefault(safe, set()).add(process)


def _untrack_process(session_hash: str | None, process) -> None:
    safe = _safe_session(session_hash)
    if not safe or process is None:
        return
    with _SESSION_LOCK:
        procs = _SESSION_PROCS.get(safe)
        if procs:
            procs.discard(process)


def _session_bases() -> list[Path]:
    """Every directory that holds per-tab files. The CLI never writes here."""
    return [*(demo.dir / "data" / "sessions" for demo in DEMOS),
            REPO_DIR / "data" / "sessions"]


def _under(path: Path, root: Path) -> bool:
    try:
        resolved, base = path.resolve(), root.resolve()
    except OSError:
        return False
    return resolved == base or base in resolved.parents


def _under_session_store(path: Path) -> bool:
    return any(_under(path, base) for base in _session_bases())


def _delete_upload(path: Path) -> None:
    """Delete one Gradio temp upload. A CLI file, or anything already inside a
    session directory, is left alone."""
    try:
        resolved = path.resolve()
    except OSError:
        return
    if not resolved.is_file() or _under_session_store(resolved):
        return
    allowed = [Path(tempfile.gettempdir()), CROP_DIR, Path.home() / ".gradio"]
    if not any(_under(resolved, root) for root in allowed):
        return
    for demo in DEMOS:
        for name in ("cache", "output", "runs"):
            if _under(resolved, demo.dir / "data" / name):
                return
    resolved.unlink(missing_ok=True)


def _forget_session(session_hash: str | None) -> None:
    """Delete everything this tab stored. Switching demos does not call this."""
    safe = _safe_session(session_hash)
    if not SESSION_RETENTION or not safe:
        return
    with _SESSION_LOCK:
        _LIVE_SESSIONS.discard(safe)
        extra = list(_SESSION_PATHS.pop(safe, ()))
        procs = list(_SESSION_PROCS.pop(safe, ()))
    for process in procs:
        if process.poll() is None:
            process.kill()
    for process in procs:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    for base in _session_bases():
        shutil.rmtree(base / safe, ignore_errors=True)
        try:
            base.rmdir()
        except OSError:
            pass
    for path in extra:
        _delete_upload(path)


def _sweep_finished_sessions() -> None:
    """Directories whose tab is gone and whose grace period has passed.

    A tab that is still open is in ``_LIVE_SESSIONS`` and is kept, including
    while the visitor is on a different demo. This only catches a call that
    ended without saying so.
    """
    if not SESSION_RETENTION:
        return
    cutoff = time.time() - ORPHAN_GRACE
    with _SESSION_LOCK:
        live = set(_LIVE_SESSIONS)
    for base in _session_bases():
        if not base.is_dir():
            continue
        for child in list(base.iterdir()):
            if not child.is_dir() or child.name in live:
                continue
            try:
                if child.stat().st_mtime >= cutoff:
                    continue
            except OSError:
                continue
            shutil.rmtree(child, ignore_errors=True)


def _sweep_sessions_on_startup() -> None:
    """No tab is connected yet, so every session directory is a leftover."""
    if not SESSION_RETENTION:
        return
    for base in _session_bases():
        shutil.rmtree(base, ignore_errors=True)


def _start_session_sweeper() -> None:
    global _SWEEPER_STARTED
    if not SESSION_RETENTION or _SWEEPER_STARTED:
        return
    _SWEEPER_STARTED = True

    def loop() -> None:
        while True:
            time.sleep(60)
            _sweep_finished_sessions()

    threading.Thread(target=loop, name="session-sweep", daemon=True).start()


def _work_root(demo: Demo, session_hash: str | None = None) -> Path:
    """Scratch space for runs, inside the demo directory.

    Two of the pipelines print paths via `path.relative_to(PROJECT_DIR)`, which
    raises on anything outside it, so the run directories have to live here
    rather than in /tmp. Locally that is the demo's own ``data/``, shared with
    the command line. On a public deployment it is ``data/sessions/<tab>/``,
    which is removed when the tab closes and which the command line never uses.
    """
    safe = _safe_session(session_hash) if SESSION_RETENTION else None
    if not safe:
        return demo.dir / "data"
    _hold_session(safe)
    root = demo.dir / "data" / "sessions" / safe
    root.mkdir(parents=True, exist_ok=True)
    return root


def _shared_session_dir(session_hash: str | None, kind: str) -> Path | None:
    """Crops and zips for one tab, shared across demos so switching keeps them."""
    safe = _safe_session(session_hash) if SESSION_RETENTION else None
    if not safe:
        return None
    _hold_session(safe)
    root = REPO_DIR / "data" / "sessions" / safe / kind
    root.mkdir(parents=True, exist_ok=True)
    return root


def _sweep_old_runs(runs_dir: Path, keep: int = KEEP_RUNS) -> None:
    """Drop all but the newest *keep* runs. A Space's disk is not large."""
    if not runs_dir.is_dir():
        return
    runs = sorted((p for p in runs_dir.iterdir() if p.is_dir()),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    for stale in runs[keep:]:
        shutil.rmtree(stale, ignore_errors=True)


def _sweep_cache(cache_dir: Path, limit: int = CACHE_MAX_BYTES) -> None:
    """Delete the oldest cache files until *cache_dir* is under *limit* bytes."""
    if not cache_dir.is_dir():
        return
    files = sorted((p for p in cache_dir.rglob("*") if p.is_file()),
                   key=lambda p: p.stat().st_mtime)
    total = sum(p.stat().st_size for p in files)
    for stale in files:
        if total <= limit:
            break
        total -= stale.stat().st_size
        stale.unlink(missing_ok=True)


def _redact(line: str, key: str) -> str:
    """*line* with no part of *key* left in it.

    The pipelines print where the key was loaded, never the key. This still
    masks the whole key, and a trailing fragment of it, in case a line echoes
    one. The anonymous tier's public key is left alone: it is not a secret.
    """
    if not key or key == ANON_KEY:
        return line
    return line.replace(key, "••••").replace(f"…{key[-4:]}", "…••••")


def _subprocess_env(api_key: str) -> dict:
    env = os.environ.copy()
    env.update({
        "VLMRUN_API_KEY": api_key,
        "PYTHONUNBUFFERED": "1",     # or the log arrives in one lump at the end
        "MPLBACKEND": "Agg",         # headless: no display to draw the plots on
        "NO_COLOR": "1",             # rich, writing to a pipe
        "TERM": "dumb",
        "COLUMNS": "100",
    })
    return env


def display_size(path: Path) -> tuple[int, int] | None:
    """(width, height) as the clip plays, rotation applied, or None.

    Phones store portrait video landscape and tag the rotation, so the stored
    size would call a 1662x2954 clip 2954x1662.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height:stream_tags=rotate"
             ":stream_side_data=rotation", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        stream = json.loads(out.stdout)["streams"][0]
        width, height = int(stream["width"]), int(stream["height"])
    except Exception:
        return None
    rotation = stream.get("tags", {}).get("rotate")
    for side in stream.get("side_data_list", []):
        rotation = side.get("rotation", rotation)
    if rotation is not None and abs(int(float(rotation))) % 180 == 90:
        width, height = height, width
    return width, height


def probe_clip(path: Path) -> tuple[float, float, int] | None:
    """(fps, duration_seconds, frame_count) for *path*, or None if unreadable.

    The frame rate is what turns a frame budget into a number of seconds, and it
    cannot be assumed: phones shoot 60fps as readily as 30, and that doubles the
    work for the same apparent clip length.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate:format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode != 0:
            return None
        rate, duration = [l for l in out.stdout.split() if l][:2]
        num, _, den = rate.partition("/")
        fps = float(num) / float(den or 1)
        seconds = float(duration)
    except Exception:
        return None
    if fps <= 0 or seconds <= 0:
        return None
    return fps, seconds, int(round(fps * seconds))


def recorded_at(path: Path) -> str | None:
    """When *path* was filmed, as an ISO timestamp, or None if it does not say.

    Apple's own tag first: `creation_time` on a clip off an iPhone is when it
    was exported or AirDropped, so two sets sent together share one, while
    `com.apple.quicktime.creationdate` is when recording started.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-of", "json", "-show_entries",
             "format_tags=com.apple.quicktime.creationdate,creation_time", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        tags = json.loads(out.stdout or "{}").get("format", {}).get("tags", {})
    except Exception:
        return None
    for name in ("com.apple.quicktime.creationdate", "creation_time"):
        value = tags.get(name)
        if value and not value.startswith("1970"):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat()
            except ValueError:
                continue
    return None


def order_sets(paths: list[Path]) -> tuple[list[Path], str]:
    """*paths* in the order they were filmed, and how that order was decided.

    Only reordered when every clip says when it was filmed: sorting the ones
    that do around the ones that do not would be a guess dressed as a fact.
    """
    times = [recorded_at(p) for p in paths]
    if len(paths) < 2:
        return paths, ""
    if all(times):
        moments = [datetime.fromisoformat(t) for t in times]
        # A clip tagged with no zone is compared as UTC, rather than refused.
        moments = [m if m.tzinfo else m.replace(tzinfo=timezone.utc) for m in moments]
        ordered = [p for _, p in sorted(zip(moments, paths), key=lambda x: x[0])]
        return ordered, "in the order they were filmed"
    return paths, "in upload order (not every clip says when it was filmed)"


def budget_frames(keyed: bool) -> int:
    return KEYED_MAX_FRAMES if keyed else ANON_MAX_FRAMES


def budget_bytes(keyed: bool) -> tuple[int, int]:
    """(per clip, per run) storage for the window a run actually keeps."""
    if keyed:
        return KEYED_MAX_CLIP_BYTES, KEYED_MAX_RUN_BYTES
    return ANON_MAX_CLIP_BYTES, ANON_MAX_RUN_BYTES


def _bytes_label(n: int) -> str:
    if n >= _GB:
        return f"{n / _GB:.1f} GB"
    return f"{max(1, round(n / _MB))} MB"


# sha256(key) -> (monotonic expiry, "ok" | "rejected"). The key itself is not kept.
_KEY_CACHE: dict[str, tuple[float, str]] = {}


def _gateway_key_status(key: str) -> str:
    """'ok', 'rejected', or 'unreachable'.

    The public anonymous key is not a billed key, even though the gateway
    accepts it. Any other string has to be accepted before it unlocks the
    larger frame budget and storage cap.
    """
    if not key or key == ANON_KEY:
        return "rejected"
    digest = hashlib.sha256(key.encode()).hexdigest()
    now = time.monotonic()
    cached = _KEY_CACHE.get(digest)
    if cached and cached[0] > now:
        return cached[1]
    request = urllib.request.Request(
        GATEWAY_MODELS_URL, headers={"Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            status = "ok" if 200 <= response.status < 300 else "unreachable"
    except urllib.error.HTTPError as exc:
        status = "rejected" if exc.code in (401, 403) else "unreachable"
    except (urllib.error.URLError, TimeoutError, OSError):
        status = "unreachable"
    if status != "unreachable":
        _KEY_CACHE[digest] = (now + (600 if status == "ok" else 60), status)
    return status


def _storage_message(keyed: bool, which: str) -> str:
    clip_cap, run_cap = budget_bytes(keyed)
    limit = (f"{_bytes_label(clip_cap)} per clip and {_bytes_label(run_cap)} per run")
    if keyed:
        return (f"{which} is over this key's storage limit ({limit}). "
                "Shorten the crop and run again.")
    return (f"{which} is over the anonymous storage limit ({limit}). "
            "Anonymous still runs: shorten the crop to the part you want measured, "
            "or switch to an API key the gateway accepts.")


def _window_bytes(path: Path, start: float, seconds: float,
                  duration: float | None, size: int) -> int:
    """Bytes the staged window will take. The whole file, when that is the window."""
    if not duration or duration <= 0 or (start <= 0.05 and seconds + 0.05 >= duration):
        return size
    return int(size * min(1.0, (seconds + 2) / duration))


def _unique_name(name: str, used: set[str]) -> str:
    """*name*, or `stem-2.suffix` and on, so two uploads can share a filename."""
    path = Path(name)
    candidate, stem, suffix = path.name, path.stem, path.suffix
    n = 2
    while candidate in used or not candidate:
        candidate = f"{stem}-{n}{suffix}" if stem else f"clip-{n}{suffix}"
        n += 1
    used.add(candidate)
    return candidate


def _stage_input(src: Path, dst: Path, start: float, seconds: float,
                 duration: float | None) -> bool:
    """Put *src* at *dst*. True when only the selected window was kept.

    The pipelines trim from the beginning (`TRIM_SECONDS`), so a window that
    starts later, or a long clip whose analysis window is its first stretch,
    is cut here. Stream-copied, like the crop bar: re-encoding would take over
    the rotation handling the demos are careful about. The cut can begin on
    the keyframe at or before *start*.
    """
    whole = not duration or duration <= 0 or (start <= 0.05 and seconds + 0.05 >= duration)
    if whole:
        shutil.copy2(src, dst)
        return False
    proc = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-ss", f"{max(0.0, start):.2f}", "-i", str(src), "-t", f"{seconds + 2:.2f}",
         "-c", "copy", "-map", "0",
         "-map_metadata", "0", "-movflags", "use_metadata_tags", str(dst)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not dst.is_file() or dst.stat().st_size == 0:
        dst.unlink(missing_ok=True)
        raise RuntimeError(proc.stderr.strip()[-500:] or "ffmpeg produced nothing")
    return True


def _stamp_by_content(path: Path) -> None:
    """Set *path*'s mtime from a hash of its bytes.

    The pipelines key their caches on `(name, size, mtime_ns)` of the input. That
    identifies a file on disk perfectly well, but Gradio writes every upload to a
    fresh temp file with a fresh mtime, so the same clip uploaded twice looked
    like two different clips and paid for inference twice.

    Deriving the mtime from the content makes the key describe the clip instead
    of the upload. Size and name are still part of it, so this only ever merges
    entries that are byte-identical: it cannot cause a hit across two different
    videos that happen to share a name.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    ns = int.from_bytes(digest.digest()[:6], "big")
    os.utime(path, ns=(ns, ns))


# The left half of every render, on its own. Each demo draws the clip with its
# overlay on the left and an equal-width panel of numbers on the right; the
# overlay alone is the part people cut into their own edits.
OVERLAY_SUFFIX = "_overlay_only"


def _overlay_only(video: Path) -> Path | None:
    """Crop *video* to its left panel, beside it as `<stem>_overlay_only.mp4`.

    The panel width comes from the run's own record of the clip it drew when
    there is one, and is half the frame otherwise: every demo's panel is drawn
    the clip's width. A render that is not two panels wide is left alone.
    """
    size = display_size(video)
    if not size:
        return None
    width, height = size
    clip_w = None
    record = video.parent / "run.json"
    if record.is_file():
        try:
            clip_w = json.loads(record.read_text()).get("export", {}).get("width")
        except (ValueError, AttributeError):
            clip_w = None
    if not clip_w:
        clip_w = width // 2
    if abs(width - 2 * clip_w) > 4:
        return None

    out = video.with_name(f"{video.stem}{OVERLAY_SUFFIX}.mp4")
    proc = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
         "-vf", f"crop={clip_w // 2 * 2}:{height // 2 * 2}:0:0",
         "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
         "-c:a", "copy", "-movflags", "+faststart", str(out)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not out.is_file() or out.stat().st_size == 0:
        out.unlink(missing_ok=True)
        return None
    return out


def _collect(run_dir: Path) -> tuple[list[Path], list[Path]]:
    """(videos, plots) from a finished run."""
    if not run_dir.is_dir():
        return [], []
    everything = sorted(p for p in run_dir.rglob("*") if p.is_file())
    videos = [p for p in everything if p.suffix == ".mp4" and p.name not in HIDDEN_VIDEOS
              and not p.stem.endswith(OVERLAY_SUFFIX)]
    images = [p for p in everything if p.suffix == ".png"]
    return videos, images


def _latest_run_dir(output_dir: Path) -> Path | None:
    if not output_dir.is_dir():
        return None
    stamps = [p for p in output_dir.iterdir() if p.is_dir()]
    return max(stamps, key=lambda p: p.stat().st_mtime) if stamps else None


def _resolve_knobs(demo: Demo, knob_values) -> tuple[dict, str]:
    """The overrides the knobs ask for, or a message naming what is missing.

    *knob_values* is flat, in the order `demo.knobs` lists them; a color knob
    takes two of them, its dropdown and its "other" text box.
    """
    values, out = iter(knob_values), {}
    for knob in demo.knobs:
        value = next(values, None)
        if knob.kind == "color":
            other = (next(values, None) or "").strip()
            picked_other = value == OTHER_COLOR
            value = (other if picked_other else value) or None
            if value is None and knob.required:
                name = knob.label.lower()
                return {}, (f"Type the {name} in the box under the list first."
                            if picked_other else f"Pick a {name} first.")
            # Typed text ends up in a segmentation prompt. A color is a few
            # words; anything else is refused rather than sent along.
            if picked_other and value and not COLOR_TEXT.fullmatch(value):
                return {}, (f"A {knob.label.lower()} is a color name: letters, "
                            f"spaces and hyphens, up to {COLOR_MAX_CHARS} characters.")
            out[knob.key] = value
        elif knob.kind == "number" and knob.key == "MAX_PERSONS":
            out[knob.key] = None if not value else int(value)
        elif knob.kind == "number":
            # A cleared box arrives as None; the pipeline wants a number.
            out[knob.key] = knob.default if value is None else value
        elif knob.kind == "slider":
            out[knob.key] = int(value)
        elif knob.kind == "text":
            text = (value or "").strip()
            out[knob.key] = text or (knob.default or None)
        else:
            out[knob.key] = value
    return out, ""


def run_demo(demo: Demo, uploads, api_key: str, start: float, seconds: float,
             export_height: int, *knob_values, session_hash: str | None = None,
             ambient_authorized: bool = False):
    """Generator driving one run. Yields UI updates as the log arrives."""

    log: list[str] = []

    def emit(status: str, *, video=None, highlights=None, table=None, plots=None,
             files=None, bundle=None, picker=None):
        keep = lambda v: v if v is not None else gr.update()
        return ("\n".join(log), status, keep(video), keep(highlights), keep(table),
                keep(plots), keep(files), keep(bundle), keep(picker))

    # ── inputs ───────────────────────────────────────────────────────────────
    typed = api_key.strip() if api_key else ""
    if typed == ENV_KEY_MARKER and (not USE_AMBIENT or not ambient_authorized):
        message = ("Blocked use of the key on disk because this run did not come "
                   "from this app's own page. Reload the app and try again.")
        log.append(message)
        yield emit(waiting.notice_html(message, "failed"))
        return

    paths = [Path(u) for u in (uploads or [])] if demo.multi else \
            ([Path(uploads)] if uploads else [])
    _remember_paths(session_hash, paths)
    if not paths:
        log.append("No video uploaded.")
        yield emit(waiting.notice_html("Upload a clip to start.", "failed"))
        return

    knob_overrides, missing = _resolve_knobs(demo, knob_values)
    if missing:
        log.append(missing)
        yield emit(waiting.notice_html(missing, "failed"))
        return

    # Blank is anonymous. The environment's key is used only when asked for,
    # by name: anonymous is the default even on a machine that has one.
    # Pasting the public anonymous key is that same mode, not a billed key.
    if typed == ENV_KEY_MARKER and USE_AMBIENT:
        key, source = AMBIENT_KEY, AMBIENT_SOURCE
    elif typed and typed not in (ENV_KEY_MARKER, ANON_KEY):
        key, source = typed, "the key you supplied"
    else:
        key, source = ANON_KEY, ""

    # A typed string is not a key. Anonymous never takes this trip: it is the
    # mode you get by leaving the key blank, and it stays available when a
    # pasted key is rejected.
    keyed = False
    if key != ANON_KEY:
        key_status = _gateway_key_status(key)
        if key_status == "ok":
            keyed = True
        else:
            message = ("That API key was rejected. Paste a valid one, or switch "
                       "to Free, anonymous." if key_status == "rejected" else
                       "Could not reach the gateway to check that API key. "
                       "Try again, or switch to Free, anonymous.")
            log.append(message)
            yield emit(waiting.notice_html(message, "failed"))
            return

    # Enforced here, from the clip's own frame rate, rather than trusting the
    # slider: the UI's validation is advisory, and an API caller never sees it.
    if len(paths) > MAX_CLIPS[keyed]:
        message = (f"Up to {MAX_CLIPS[keyed]} clips per run"
                   + ("." if keyed else
                      f" on the anonymous tier ({MAX_CLIPS[True]} with a key)."))
        log.append(message)
        yield emit(waiting.notice_html(message, "failed"))
        return

    clips = []
    for path in paths:
        if not path.is_file():
            message = f"Could not read {path.name}."
            log.append(message)
            yield emit(waiting.notice_html(message, "failed"))
            return
        clips.append((path, probe_clip(path), path.stat().st_size))

    # Every clip is held to the same window, so the fastest frame rate among
    # them is the one the budget binds on.
    budget = budget_frames(keyed)
    fps = max((info[0] for _, info, _ in clips if info), default=30.0)
    cap = budget / fps
    start, seconds = max(0.0, float(start)), float(seconds)
    if seconds > cap:
        log.append(
            f"Trimmed to {cap:.1f}s: the clip is {fps:.0f} fps, and "
            f"{budget} frames is the per-run budget"
            + ("" if keyed else ", raised to "
               f"{KEYED_MAX_FRAMES} with a key of your own") + "."
        )
        seconds = cap

    # Before any copy. Anonymous keeps the selected window, which the frame
    # budget already shortened; a long upload is not stored again.
    clip_cap, run_cap = budget_bytes(keyed)
    estimates = [
        _window_bytes(path, start, seconds, info[1] if info else None, size)
        for path, info, size in clips]
    over = [path.name for (path, _, _), estimate in zip(clips, estimates)
            if estimate > clip_cap]
    if over or sum(estimates) > run_cap:
        message = _storage_message(keyed, ", ".join(over) if over else "This run")
        log.append(message)
        yield emit(waiting.notice_html(message, "failed"))
        return

    if keyed:
        log.append(f"Key from {source}, billed to it. "
                   f"Analyzing {seconds:.0f}s of video.")
    else:
        where = ("This Space has one IP, so that budget is shared by everyone using "
                 "it right now." if ON_SPACES else
                 "Nothing supplied a key, so this run is on the shared guest budget.")
        log.append(
            f"Using the gateway's anonymous tier: {ANON_LIMITS} per IP "
            f"(as of {RATE_LIMITS_AS_OF}, may have changed; latest at "
            f"{RATE_LIMITS_URL}). {where}"
        )
        log.append(f"Analyzing {seconds:.1f}s, about {int(seconds * fps)} frames "
                   f"of the {budget}-frame anonymous budget.")

    # ── scratch ──────────────────────────────────────────────────────────────
    _sweep_crops()
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    data_dir = _work_root(demo, session_hash)
    runs_dir = data_dir / "runs"
    _sweep_old_runs(runs_dir)
    _sweep_cache(data_dir / "cache")

    run_root = runs_dir / run_id
    in_dir = run_root / "input"
    out_dir = run_root / "output"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    staged, names_used, cut = [], set(), 0
    for path, info, _size in clips:
        # Gradio hands over a temp file with a mangled name; the pipelines use
        # the stem to name their outputs, so keep something readable. Two clips
        # can share a name; the second gets a suffix instead of overwriting.
        target = in_dir / _unique_name(path.name, names_used)
        try:
            if _stage_input(path, target, start, seconds,
                            info[1] if info else None):
                cut += 1
        except Exception as exc:
            shutil.rmtree(run_root, ignore_errors=True)
            message = f"Could not prepare {path.name}: {exc}"
            log.append(message)
            yield emit(waiting.notice_html(message, "failed"))
            return
        staged.append(target)

    actual = []
    for path in staged:
        try:
            actual.append(path.stat().st_size)
        except OSError:
            actual.append(clip_cap + 1)
    over = [path.name for path, size in zip(staged, actual) if size > clip_cap]
    if over or sum(actual) > run_cap:
        shutil.rmtree(run_root, ignore_errors=True)
        message = _storage_message(keyed, ", ".join(over) if over else "This run")
        log.append(message)
        yield emit(waiting.notice_html(message, "failed"))
        return

    shown = [dst.name if dst.name == src.name else f"{dst.name} (from {src.name})"
             for (src, _, _), dst in zip(clips, staged)]
    log.append(f"Staged {len(staged)} clip(s): " + ", ".join(shown))
    if cut:
        log.append(f"Kept {start:.0f}s–{start + seconds:.0f}s from {cut} clip(s), "
                   "not the rest of the upload.")
    if demo.sets and len(staged) > 1:
        staged, how = order_sets(staged)
        log.append(f"Sets, {how}: "
                   + ", ".join(f"{i}. {p.name}" for i, p in enumerate(staged, 1)))

    for path in staged:
        _stamp_by_content(path)

    # ── config overrides ─────────────────────────────────────────────────────
    # 0 is "full resolution", which the pipelines spell None.
    export_h = int(export_height) or None
    if demo.stacked:
        # The dropdown names the clip's height, and the panel below is as tall
        # again. Full resolution is the source's own height twice over; the
        # pipeline never scales the clip past its source, so a guess too high
        # is harmless.
        size = _source_size(staged[0])
        export_h = 2 * (export_h or (size[1] if size else 4320))

    overrides: dict = {
        "TRIM_SECONDS": seconds,
        "INFERENCE_HEIGHT": INFERENCE_HEIGHT,
        "EXPORT_HEIGHT": export_h,
        "OUTPUT_DIR": str(out_dir),
        # Shared across runs on purpose: the cache key covers the clip and every
        # request field, so a repeat of the same upload and settings costs no
        # quota at all. That is the difference between a Space that survives a
        # front page and one that does not.
        "CACHE_DIR": str(data_dir / "cache"),
        "REUSE_POSES": True,
        "SAVE_REPORT": True,
    }

    if demo.sets:
        # One clip after another, through runner.py, in set order. Every
        # pipeline takes one clip per run; not all of them take a folder.
        overrides["__CLIPS__"] = [str(p) for p in staged]
        overrides["__CLIP_KEY__"] = demo.input_key
        overrides[demo.input_key] = str(staged[0])
    elif demo.multi:
        overrides["BATCH_MODE"] = True
        overrides["INPUT_DIR"] = str(in_dir)
        overrides["INPUT_VIDEO"] = str(staged[0])
        overrides["REUSE_HOLDS"] = True
        overrides["REUSE_FLOOR"] = True
        if not keyed:
            # 6 concurrent segmentation calls empties a 10/min bucket in one
            # breath. Serialised, the run is slower but it is not a 429.
            overrides["HOLD_REQUEST_WORKERS"] = 2
    else:
        overrides[demo.input_key] = str(staged[0])

    if "SAMPLE_FPS" in knob_overrides and not keyed:
        # The same reasoning as the hold workers: 8 reads in flight at once
        # empties the anonymous bucket in one breath.
        overrides["CONCURRENCY"] = 2

    overrides.update(knob_overrides)
    overrides.update(demo.fixed)

    # `fixed` carries tuples; JSON only has arrays, and runner.py turns arrays
    # back into tuples on the way in.
    overrides_path = run_root / "overrides.json"
    overrides_path.write_text(json.dumps(
        {k: (list(v) if isinstance(v, tuple) else v) for k, v in overrides.items()},
        indent=2,
    ))

    if demo.multi and not demo.sets and not keyed:
        # Per clip: one segmentation call per sampled frame for the holds, the
        # same again for the floor (it re-samples with HOLD_SAMPLE_FRAMES), and
        # one pose call carrying the whole clip. So 2n + 1, not n + 1.
        samples = int(overrides.get("HOLD_SAMPLE_FRAMES", 8))
        calls = len(staged) * (2 * samples + 1)
        log.append(
            f"This run needs roughly {calls} gateway calls: {len(staged)} clip(s) "
            f"x ({samples} hold + {samples} floor + 1 pose). The anonymous tier "
            f"allows {ANON_RATE_PER_MIN} a minute, so expect at least "
            f"{calls / ANON_RATE_PER_MIN:.0f} minutes, or supply a key."
        )
    elif "SAMPLE_FPS" in overrides and not keyed:
        # One pose call, a SAM call per 640-frame segment of plate tracking
        # (an upper bound: it is skipped if the run falls back to the hip hinge),
        # and a System One call per read.
        frames = int(seconds * fps)
        sam = -(-frames // (overrides.get("PLATE_TRACK_STRIDE", 5) * 128))
        reads = int(seconds * overrides["SAMPLE_FPS"])
        calls = 1 + sam + reads
        log.append(
            f"This run needs roughly {calls} gateway calls: 1 pose + {sam} plate "
            f"+ {reads} back reads. The anonymous tier allows {ANON_RATE_PER_MIN} "
            f"a minute and 30 an hour, so crop tight, lower the reads per second, "
            f"or supply a key."
        )

    log.append("")
    progress = waiting.Progress(demo.stages)
    yield emit(progress.html())

    # ── the run ──────────────────────────────────────────────────────────────
    started = time.perf_counter()
    process = subprocess.Popen(
        [sys.executable, "-u", str(REPO_DIR / "webapp" / "runner.py"),
         str(demo.dir), str(overrides_path)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=_subprocess_env(key), text=True, bufsize=1,
        cwd=str(demo.dir),
    )
    _track_process(session_hash, process)

    # Killed from a timer rather than checked between lines: a pipeline stuck
    # waiting on the network prints nothing, and would otherwise hold the
    # queue, and everyone in it, forever.
    timed_out = threading.Event()

    def _expire():
        timed_out.set()
        process.kill()

    watchdog = threading.Timer(RUN_TIMEOUT_SECONDS, _expire)
    watchdog.daemon = True
    watchdog.start()

    last_push = 0.0
    closed = False
    try:
        for line in process.stdout:
            if not _session_open(session_hash):
                closed = True
                break
            log.append(_redact(line.rstrip(), key))
            now = time.perf_counter()
            # A new stage goes out at once; otherwise, pushing every line makes
            # the queue the bottleneck on a chatty run.
            if progress.feed(line) or now - last_push > 0.4:
                last_push = now
                yield emit(progress.html())
        if not closed:
            code = process.wait()
    finally:
        watchdog.cancel()
        if process.poll() is None:
            process.kill()
        if closed:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        _untrack_process(session_hash, process)

    if closed:
        _forget_session(session_hash)
        log.append("\nThe tab closed, so this run was stopped and its files were deleted.")
        yield emit(waiting.notice_html("Stopped. This visit's files were deleted.", "failed"))
        return

    if timed_out.is_set():
        log.append(f"\nTimed out after {RUN_TIMEOUT_SECONDS}s and was killed.")
        yield emit(progress.html("failed", "Timed out."))
        return

    elapsed = time.perf_counter() - started

    if code != 0:
        log.append(f"\nExited {code} after {elapsed:.0f}s.")
        tail = "\n".join(log[-25:])
        hint = ""
        if "429" in tail or "rate limit" in tail.lower():
            hint = ("\n\nThat looks like the anonymous tier's rate limit "
                    f"({ANON_LIMITS}, shared by everyone on this Space). Supply "
                    f"your own VLM Run key, or try again later. Newest limits: "
                    f"{RATE_LIMITS_URL}")
        yield emit(progress.html("failed", f"Failed after {elapsed:.0f}s."),
                   highlights=waiting.notice_html(hint.strip(), "failed") if hint else "")
        return

    # ── outputs ──────────────────────────────────────────────────────────────
    run_dir = _latest_run_dir(out_dir)
    # Sets write a directory each; the run is all of them.
    if demo.sets and len(staged) > 1:
        run_dir = out_dir
    videos, images = _collect(run_dir) if run_dir else ([], [])
    set_names = [p.name for p in staged] if demo.sets else []

    if not videos and not images:
        log.append("\nThe run reported success but produced no video or plots.")
        yield emit(progress.html("failed", "Finished, but there is nothing to show."))
        return

    # The answer in two spreadsheets, named after the clip so a download folder
    # full of them still says which is which.
    result = exports.build(demo.key, run_dir, sets=set_names)
    if result.problem:
        log.append(f"\nCould not build the CSVs: {result.problem}")
    stem = demo.key if demo.multi else staged[0].stem
    csvs = [p for p in (
        exports.write_csv(result.summary, run_dir / f"{stem}_summary.csv"),
        exports.write_csv(result.frames, run_dir / f"{stem}_{result.frames_file}.csv"),
    ) if p]

    # Not for climbing's `_route` render: that one is already the right panel alone.
    overlays = [] if demo.stacked else [
        o for v in videos if not v.stem.endswith("_route") if (o := _overlay_only(v))]

    # Concentric time, set by set and rep by rep. Ahead of the pipelines' own
    # plots: with more than one set it is the one figure that spans them all.
    timing = charts.concentric_plot(demo.key, result.summary,
                                    run_dir / f"{stem}_concentric.png")
    if timing:
        images = [timing, *images]
    elif demo.sets:
        log.append("No concentric-time figure: fewer than two full reps to plot.")

    downloads = [*csvs, *videos, *overlays, *images]
    # Everything above in one file, for whoever wants the lot rather than
    # clicking down the list. Inside the run directory, so it is swept with it.
    bundle = exports.write_zip(
        [(p, str(p.relative_to(run_dir))) for p in downloads],
        run_dir / f"{stem}_results.zip")

    log.append(f"\nDone in {elapsed:.0f}s.")

    # A batch run renders a video per attempt, and each demo that draws a second
    # "route only" panel writes two. Showing the first and burying the rest in
    # the downloads loses the comparison the run was for.
    choices = [(_render_label(p, run_dir, set_names), str(p)) for p in videos]
    yield emit(
        progress.html("done", f"Done in {elapsed:.0f}s.", elapsed=elapsed),
        video=str(videos[0]) if videos else None,
        highlights=_highlights_html(result.highlights),
        table=gr.update(value=_table(result.summary), label=result.summary_label,
                        visible=bool(result.summary)),
        plots=_plots_html(images),
        files=gr.update(value=[str(p) for p in downloads], visible=True),
        bundle=gr.update(value=str(bundle) if bundle else None, visible=bool(bundle)),
        picker=gr.update(choices=choices,
                         value=str(videos[0]) if videos else None,
                         visible=len(videos) > 1),
    )


def _render_label(video: Path, run_dir: Path, set_names: list[str]) -> str:
    """How a render is named in the picker: by set for sets, else by folder."""
    if set_names:
        source = exports.source_clip(video.parent)
        if source in set_names:
            return f"Set {set_names.index(source) + 1} · {video.name}"
    return f"{video.parent.name}/{video.name}" if video.parent != run_dir else video.name


_UNITS = (("_s", " (s)"), ("_deg", " (°)"), ("_cm", " (cm)"), ("_percent", " (%)"))


def _header(column: str) -> str:
    """A CSV column as a person reads it: `ascent_s` is "Ascent (s)"."""
    unit = ""
    for suffix, label in _UNITS:
        if column.endswith(suffix):
            column, unit = column[:-len(suffix)], label
            break
    words = column.replace("_", " ")
    return words[:1].upper() + words[1:] + unit


def _table(rows: list[dict]):
    """Rows as the table component wants them. The CSV keeps snake_case, which
    is what a plotting script wants; on the page the headers are words."""
    if not rows:
        return None
    columns = list(dict.fromkeys(k for row in rows for k in row))
    return {"headers": [_header(c) for c in columns],
            "data": [["" if row.get(c) is None else row.get(c, "") for c in columns]
                     for row in rows]}


def _highlights_html(items: list[tuple[str, str]]) -> str:
    if not items:
        return ""
    tiles = "".join(f"<div class='stat'><div class='stat-value'>{html.escape(v)}</div>"
                    f"<div class='stat-label'>{html.escape(k)}</div></div>"
                    for k, v in items)
    # The first two, as a line for the history table, which reads it back
    # out of here the way it reads the elapsed time out of the status.
    headline = ", ".join(f"{k.lower()} {v}" for k, v in items[:2])
    return (f"<div class='stats' data-headline='{html.escape(headline, quote=True)}'>"
            f"{tiles}</div>")


def _plots_html(images: list[Path]) -> str:
    """Each plot at the column's full width and its own aspect ratio.

    Inlined rather than handed to a Gallery: a gallery tile is a fixed shape,
    so a wide timeline either shrinks to a strip or needs scrolling inside it.
    """
    figures = []
    for path in images:
        try:
            data = base64.b64encode(path.read_bytes()).decode()
        except OSError:
            continue
        figures.append(f"<figure><img src='data:image/png;base64,{data}' "
                       f"alt='{html.escape(path.stem)}'></figure>")
    return f"<div class='plots'>{''.join(figures)}</div>" if figures else ""


# Hovering a row in a Downloads list shows what the file is before it is saved:
# the image, the video playing muted, or a spreadsheet's first rows and columns.
# One listener on the document, since the lists are re-rendered after each run.
# Fetched from the same link the row downloads from, and only once per file.
PREVIEW_JS = r"""
() => {
  const ROWS = 8, COLS = 7, cache = new Map();
  let peek = null, current = null;
  const esc = s => s.replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

  function csvTable(text) {
    const lines = text.split(/\r?\n/).filter(Boolean);
    const cells = l => l.split(",").slice(0, COLS).map(c => esc(c.replace(/^"|"$/g, "")));
    const head = cells(lines[0] || "");
    const body = lines.slice(1, ROWS + 1).map(cells);
    const more = [];
    if (lines.length - 1 > ROWS) more.push(`${lines.length - 1} rows`);
    if ((lines[0] || "").split(",").length > COLS)
      more.push(`${lines[0].split(",").length} columns`);
    return `<table><tr>${head.map(h => `<th>${h}</th>`).join("")}</tr>` +
           body.map(r => `<tr>${r.map(c => `<td>${c}</td>`).join("")}</tr>`).join("") +
           `</table>` + (more.length ? `<p class="note">of ${more.join(", ")}</p>` : "");
  }

  function place(e) {
    if (!peek) return;
    const r = peek.getBoundingClientRect(), pad = 16;
    let x = e.clientX - r.width - pad, y = e.clientY - r.height / 2;
    if (x < pad) x = Math.min(e.clientX + pad, innerWidth - r.width - pad);
    y = Math.max(pad, Math.min(y, innerHeight - r.height - pad));
    peek.style.left = x + "px"; peek.style.top = y + "px";
  }

  function hide() { peek?.remove(); peek = null; current = null; }

  function show(row, e) {
    const link = row.querySelector("td.download a[href]");
    if (!link) return;
    const href = link.href, ext = (link.getAttribute("download") || href)
      .split("?")[0].split(".").pop().toLowerCase();
    if (!["png", "jpg", "jpeg", "mp4", "csv"].includes(ext)) return;
    hide(); current = row;
    peek = document.createElement("div"); peek.className = "dl-peek";
    if (ext === "mp4") {
      peek.innerHTML = `<video src="${href}" muted autoplay loop playsinline></video>`;
      peek.querySelector("video").addEventListener("loadedmetadata", () => place(e));
    } else if (ext === "csv") {
      peek.innerHTML = `<p class="note">Loading…</p>`;
      const got = cache.get(href) || fetch(href).then(r => r.text());
      cache.set(href, got);
      got.then(t => { if (current === row && peek) { peek.innerHTML = csvTable(t); place(e); } });
    } else {
      peek.innerHTML = `<img src="${href}" alt="">`;
      peek.querySelector("img").addEventListener("load", () => place(e));
    }
    document.body.appendChild(peek); place(e);
  }

  document.addEventListener("mouseover", e => {
    const row = e.target.closest?.(".downloads tr");
    if (row === current) return;
    row ? show(row, e) : hide();
  });
  document.addEventListener("mousemove", e => { if (current) place(e); });
  addEventListener("scroll", hide, true);
}
"""


# ── UI ───────────────────────────────────────────────────────────────────────
# One page. A toggle pinned to the top picks the demo, and everything below it
# belongs to that demo. Each demo has its own panel with its own components,
# so flipping away and back keeps the clip, the trim and the last result where
# they were; the toggle only ever changes which panel is showing.

# Every player on the page, input and result alike, is this tall, so a clip
# and what came back from it sit level with each other whatever their shape.
PLAYER_HEIGHT = 440

# Seconds only ever derived from frames, never assumed. 30fps is the common
# case and is only a starting point; a real clip is probed on upload.
ASSUMED_FPS = 30.0

def _build_knob(knob: Knob) -> list:
    """The component(s) for *knob*: one, or for a color knob, two."""
    if knob.kind == "color":
        pick = gr.Dropdown(choices=[*knob.choices, OTHER_COLOR], value=None,
                           label=knob.label + (" (required)" if knob.required else ""),
                           info=knob.info or None, filterable=False)
        other = gr.Textbox(label=f"Other {knob.label.lower()}", visible=False,
                           placeholder="e.g. teal, mint, neon green",
                           max_length=COLOR_MAX_CHARS)
        pick.change(lambda v: gr.update(visible=(v == OTHER_COLOR)), inputs=[pick],
                    outputs=[other], api_name=False)
        return [pick, other]
    return [_build_one(knob)]


def _build_one(knob: Knob):
    common = {"label": knob.label, "info": knob.info or None}
    if knob.kind == "slider":
        return gr.Slider(minimum=knob.minimum, maximum=knob.maximum, step=knob.step,
                         value=knob.default, **common)
    if knob.kind == "dropdown":
        return gr.Dropdown(choices=list(knob.choices), value=knob.default, **common)
    if knob.kind == "checkbox":
        return gr.Checkbox(value=bool(knob.default), **common)
    if knob.kind == "number":
        whole = float(knob.step).is_integer()
        return gr.Number(value=knob.default, precision=0 if whole else None,
                         step=knob.step, minimum=knob.minimum, maximum=knob.maximum,
                         **common)
    return gr.Textbox(value=knob.default or "", **common)


def _explainer(demo: Demo) -> str:
    """The demo's notes, or just one `## Section` of them when one is named."""
    name, _, section = demo.explainer.partition("#")
    path = demo.dir / name
    if not path.is_file():
        return ""
    text = path.read_text()
    if not section:
        return text
    # A README is mostly setup and output listings; only the method belongs
    # here. Its own heading is dropped, since the accordion already says it.
    found = re.search(rf"^##\s+{re.escape(section)}\s*$(.*?)(?=^##\s|\Z)", text,
                      flags=re.M | re.S)
    return found[1].strip() if found else ""


# Gradio's own svelte class hashes change between builds, so everything here
# hangs off semantic classes and our own elem_classes.
CSS = """
.hero { text-align: center; margin: 0.4rem 0 0.2rem 0; }
.hero h1 { margin: 0 0 0.3rem 0; font-size: 1.9rem; letter-spacing: -0.01em; }
.hero p { margin: 0; opacity: 0.72; }

/* The demo toggle, pinned so it stays in reach however far down the results
   run. The background keeps content from showing through as it scrolls under.
   Gradio's container is overflow:hidden, which makes it the scroll box sticky
   pins to, so nothing ever sticks; clip keeps the clipping without that. */
.gradio-container { overflow: clip !important; }
.topbar {
    position: sticky; top: 0; z-index: 50;
    background: var(--body-background-fill);
    padding: 0.6rem 0 0.5rem 0;
    border-bottom: 1px solid var(--border-color-primary);
    align-items: center !important; gap: 0.8rem;
}
.demo-toggle .wrap { display: flex !important; flex-wrap: wrap; gap: 0.35rem !important; }
.demo-toggle label {
    flex: 1 1 0; justify-content: center; min-width: 7.5rem;
    padding: 0.6rem 0.9rem !important; font-weight: 650; font-size: 1rem;
    border-radius: var(--radius-lg) !important; cursor: pointer;
}
.demo-toggle input[type=radio] { display: none; }
.demo-toggle label.selected {
    background: var(--button-primary-background-fill) !important;
    color: var(--button-primary-text-color) !important;
    border-color: var(--button-primary-background-fill) !important;
}
.billing-card {
    margin: 0.6rem 0 0.8rem 0; padding: 1rem 1.2rem 1.1rem 1.2rem;
    border: 1px solid var(--button-primary-background-fill);
    border-radius: var(--radius-xl); background: var(--block-background-fill);
}
.billing-card h3 { margin: 0 0 0.15rem 0; font-size: 1.15rem; }
.billing-card p { margin: 0; opacity: 0.72; font-size: 0.9rem; }
.billing-choice button { min-height: 3rem; font-size: 1.02rem !important;
                         font-weight: 650 !important; }
.billing-choice p { margin: 0.4rem 0 0 0; font-size: 0.85rem; }
.key-found, .key-missing { margin: 0.6rem 0 0 0 !important; font-size: 0.9rem; }
.key-found { color: #15803d; }
.key-missing { color: #b45309; }
.billing-confirm { justify-content: flex-end !important; margin-top: 0.3rem; }
.billing-confirm button { min-height: 2.6rem; font-weight: 650 !important; }
.billing {
    align-items: center !important; gap: 0.7rem; flex-wrap: wrap;
    margin: 0.4rem 0 0.6rem 0; padding: 0.35rem 0.5rem 0.35rem 0.9rem;
    border: 1px solid var(--border-color-primary); border-radius: var(--radius-lg);
    background: var(--block-background-fill);
}
.billing-note { margin: 0 !important; font-size: 0.88rem; }
.key-box { min-width: 14rem; }

.demo-head { margin: 1rem 0 0.4rem 0; }
.demo-head h2 { margin: 0 0 0.2rem 0; font-size: 1.35rem; }
.demo-head p { margin: 0; opacity: 0.72; }
.demo-note {
    font-size: 0.9rem; margin: 0.2rem 0 0.6rem 0; padding: 0.55rem 0.8rem;
    border-left: 3px solid var(--button-primary-background-fill);
    background: var(--block-background-fill); border-radius: 4px;
}

.go button { min-height: 3rem; font-size: 1.05rem !important;
             font-weight: 650 !important; }
.muted { opacity: 0.6; font-size: 0.85rem; text-align: center; }
.history-wrap { margin: 2rem auto 0 auto; }

/* Every player is PLAYER_HEIGHT tall (set on the components); a portrait
   clip and a landscape result both letterbox into it rather than crop. */
.clip-player video, .clip-player img {
    width: 100% !important;
    object-fit: contain !important;
    background: #000;
}

/* The run's answer: a row of numbers, then the table, then the plots. */
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(7.5rem, 1fr));
         gap: 0.5rem; margin: 0.2rem 0; }
.stat { padding: 0.65rem 0.8rem; border: 1px solid var(--border-color-primary);
        border-radius: var(--radius-lg); background: var(--block-background-fill); }
.stat-value { font-size: 1.35rem; font-weight: 700; line-height: 1.2;
              font-variant-numeric: tabular-nums; }
.stat-label { font-size: 0.8rem; opacity: 0.65; margin-top: 0.15rem; }
/* Downloads: a row shows what it is on hover (PREVIEW_JS), and the whole run
   comes as one zip from the button under the list. */
.downloads tr { cursor: default; }
.downloads tr:hover td { background: var(--background-fill-secondary); }
.bundle { margin-top: 0.3rem; }
.dl-peek {
    position: fixed; z-index: 1000; pointer-events: none;
    width: min(360px, calc(100vw - 32px)); max-height: 280px; overflow: hidden;
    padding: 6px; border-radius: var(--radius-lg);
    border: 1px solid var(--border-color-primary);
    background: var(--body-background-fill);
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.22);
}
.dl-peek img, .dl-peek video { display: block; width: 100%; max-height: 266px;
                               object-fit: contain; background: #000;
                               border-radius: 4px; }
.dl-peek img { background: #fff; }
.dl-peek table { border-collapse: collapse; font-size: 0.72rem; width: 100%;
                 font-variant-numeric: tabular-nums; }
.dl-peek th, .dl-peek td { padding: 2px 6px; text-align: left; white-space: nowrap;
                           border-bottom: 1px solid var(--border-color-primary); }
.dl-peek th { font-weight: 650; }
.dl-peek .note { font-size: 0.72rem; opacity: 0.6; margin: 4px 2px 0 2px; }
.session-zip { margin-top: 1.2rem; }
.zip-row { align-items: end !important; }
.plots figure { margin: 0 0 0.6rem 0; }
.plots img { display: block; width: 100%; height: auto;
             border: 1px solid var(--border-color-primary);
             border-radius: var(--radius-lg); background: #fff; }
""" + waiting.CSS


def _inline(text: str) -> str:
    """Inline markdown to HTML, for text placed inside an HTML block.

    Gradio's Markdown does not parse markdown nested inside a block-level HTML
    element, so a `**bold**` written for the prose reads as literal asterisks
    once it is wrapped in a styled <p>. Only the two inline forms these strings
    actually use are handled.
    """
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", text)


def _headline(highlights) -> str:
    """The run's first two numbers, for the history row."""
    found = re.search(r"data-headline='([^']*)'", highlights or "")
    return _clip_words(html.unescape(found[1])) if found and found[1] else "done"


def _clip_words(text: str, limit: int = 58) -> str:
    """Trim to *limit*, on a word boundary, so a row never ends mid-word."""
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(" ,;:")
    return f"{cut}…"


def _md_cell(text: str) -> str:
    """*text* as one markdown table cell: no column breaks, no markup."""
    return html.escape(text).replace("|", "&#124;").replace("`", "&#96;") \
               .replace("*", "&#42;").replace("_", "&#95;")


def _history_md(history: list[dict]) -> str:
    if not history:
        return ""
    rows = ["| Run | Demo | Clip | Result | Took |",
            "|---:|---|---|---|---:|"]
    for item in reversed(history):
        rows.append(
            f"| {item['n']} | {item['demo']} | {_md_cell(item['clip'])} | "
            f"{item['headline']} | {item['elapsed']} |"
        )
    return "\n".join(rows)


# What the session's zip can be narrowed to. Videos are most of the bytes, so
# someone after the numbers alone can leave them out.
KINDS = {"Spreadsheets": {".csv"}, "Videos": {".mp4"}, "Plots": {".png"}}
ZIP_DIR = Path(tempfile.gettempdir()) / "vision-demos-zips"
ZIP_MAX_AGE = 3600   # seconds a built session zip is kept for its download


def _on_history(history, chosen):
    """Each finished run joins the session's zip ticked; the rest keep theirs."""
    if not history:
        return gr.update(choices=[], value=[])
    choices = [(f"Run {h['n']} · {h['demo']} · {h['clip']}", h["n"])
               for h in reversed(history)]
    return gr.update(choices=choices,
                     value=sorted({*(chosen or []), history[-1]["n"]}))


def _size(n: int) -> str:
    return f"{n / 1e6:.1f} MB" if n >= 1e5 else f"{max(1, round(n / 1e3))} KB"


def _on_session_zip(history, chosen, kinds, request: gr.Request):
    """The ticked runs' files of the ticked kinds, as one zip, folder per run."""
    suffixes = set().union(*(KINDS[k] for k in kinds or [] if k in KINDS))
    entries = []
    safe_session = _session_hash_of(request) if SESSION_RETENTION else None
    # A hosted/API request without this tab's session id has no filesystem
    # namespace it is allowed to read.
    if SESSION_RETENTION and not safe_session:
        return gr.update(value=None, label="Nothing selected", interactive=False)
    for h in history or []:
        if h["n"] not in (chosen or []):
            continue
        demo = DEMOS_BY_KEY.get(h.get("key"))
        if demo is None:
            continue
        stem = re.sub(r"[^\w.-]+", "_", Path(h["clip"]).stem)
        folder = f"run{h['n']:02d}_{demo.key}_{stem}"
        data_dir = (demo.dir / "data" / "sessions" / safe_session
                    if safe_session else demo.dir / "data")
        allowed_root = data_dir / "runs"
        # Relative to the run's own top, so batch attempts that each wrote a
        # video of the same name stay apart, as they do in the run directory.
        paths = []
        for value in h.get("files", []):
            try:
                path = Path(value).resolve(strict=True)
            except (OSError, RuntimeError, TypeError):
                continue
            if (path.is_file() and path.suffix.lower() in suffixes
                    and _under(path, allowed_root)):
                paths.append(path)
        top = Path(os.path.commonpath(paths)) if paths else Path()
        top = top.parent if top in paths else top
        entries += [(path, f"{folder}/{path.relative_to(top)}") for path in paths]
    if not entries:
        return gr.update(value=None, label="Nothing selected", interactive=False)

    directory = _shared_session_dir(_touch_session(request), "zips")
    if directory is None:
        ZIP_DIR.mkdir(parents=True, exist_ok=True)
        cutoff = time.time() - ZIP_MAX_AGE
        for old in ZIP_DIR.glob("*.zip"):
            if old.stat().st_mtime < cutoff:
                old.unlink(missing_ok=True)
        directory = ZIP_DIR
    runs = len({e[1].split("/")[0] for e in entries})
    target = directory / f"vision-demos_{runs}-run{'s' if runs > 1 else ''}_{uuid.uuid4().hex[:6]}.zip"
    exports.write_zip(entries, target)
    size = sum(src.stat().st_size for src, _ in entries)
    return gr.update(value=str(target), interactive=True,
                     label=f"Download {len(entries)} file{'s' if len(entries) != 1 else ''} "
                           f"({_size(size)}, .zip)")


def _clip_path(value) -> Path | None:
    """A player's value as a path. The crop bar always belongs to one clip."""
    return Path(value) if value else None


def _is_keyed(run_key) -> bool:
    """Whether a run given *run_key* is billed to a key, and so budgeted as one."""
    run_key = (run_key or "").strip()
    return (USE_AMBIENT if run_key == ENV_KEY_MARKER else bool(run_key))


ANON_MODE, KEY_MODE = "Anonymous", "My API key"


def _run_key(mode, typed) -> str:
    """The key a run is given: blank for anonymous, the marker for the .env key."""
    if mode != KEY_MODE:
        return ""
    typed = (typed or "").strip()
    return typed or (ENV_KEY_MARKER if USE_AMBIENT else "")


def _billing_note(mode, typed) -> str:
    if mode != KEY_MODE:
        return "<b>Anonymous</b>"
    typed = (typed or "").strip()
    if typed:
        return "<b>Your API key</b> (entered manually)"
    if USE_AMBIENT:
        return "<b>Your API key</b> from .env"
    return "<b>Your API key</b> · paste one from app.vlm.run. Until then, runs stay anonymous"


def _on_pick(mode):
    """A click on one of the card's two options: select it, nothing more.

    Nothing about the runs changes until the check mark confirms it, so a
    half-typed key never gets used and picking the wrong option costs nothing.
    """
    keyed = mode == KEY_MODE
    return (mode,
            gr.update(variant="secondary" if keyed else "primary"),
            gr.update(variant="primary" if keyed else "secondary"),
            gr.update(visible=keyed), gr.update(visible=keyed))


def _on_confirm(pending, typed):
    """The check mark: apply the choice, and fold the card to one line."""
    if pending == KEY_MODE and not (typed or "").strip() and not USE_AMBIENT:
        gr.Warning("Paste a VLM Run API key first, or pick the free option.")
        return (gr.update(),) * 6
    return (pending, _run_key(pending, typed),
            f"<p class='billing-note'>{_billing_note(pending, typed)}</p>",
            gr.update(visible=False), gr.update(visible=True), pending)


def _on_change(current):
    """Reopen the card, with the option in force already selected."""
    return (gr.update(visible=True), gr.update(visible=False), *_on_pick(current))


def _on_restore(saved):
    """On load: a returning visitor's choice, or the card for a first visit.

    A pasted key is not remembered, so a returning key user with no .env key
    has nothing to run with and gets the card again.
    """
    if saved == ANON_MODE or (saved == KEY_MODE and USE_AMBIENT):
        return (saved, _run_key(saved, ""),
                f"<p class='billing-note'>{_billing_note(saved, '')}</p>",
                gr.update(visible=False), gr.update(visible=True),
                *_on_pick(saved))
    return (ANON_MODE, "",
            f"<p class='billing-note'>{_billing_note(ANON_MODE, '')}</p>",
            gr.update(visible=True), gr.update(visible=False),
            *_on_pick(KEY_MODE if saved == KEY_MODE else ANON_MODE))


# The trim bar's whole state, as one string: "start,end,duration,cap".
# One two-handle bar reads better than two sliders, and Gradio has no range
# component, so it is a small custom one (TRIM_HTML below). The cap is the
# longest window the frame budget allows at this clip's frame rate, carried
# along so the bar can warn without a round trip on every drag.

def _trim_value(start: float, end: float, duration: float, cap: float) -> str:
    return f"{start:.1f},{end:.1f},{duration:.1f},{cap:.1f}"


def _parse_trim(value) -> tuple[float, float]:
    """(start, end) from the bar, or the whole clip when there is none yet."""
    try:
        start, end = (float(x) for x in str(value).split(",")[:2])
        return start, end
    except ValueError:
        return 0.0, float("inf")


def _clip_cap(clip, typed_key) -> tuple[float, float] | None:
    """(duration, seconds the budget allows) for *clip*."""
    path = _clip_path(clip)
    probed = probe_clip(path) if path else None
    if probed is None:
        return None
    fps, duration, _ = probed
    return duration, budget_frames(_is_keyed(typed_key)) / fps


def _on_upload(clip, typed_key):
    """Size the bar to *clip*, selecting as much as the budget allows."""
    found = _clip_cap(clip, typed_key)
    if found is None:
        return gr.update(value="", visible=False)
    duration, cap = found
    return gr.update(value=_trim_value(0.0, min(duration, cap), duration, cap),
                     visible=True)


def _on_key(clip, typed_key, trim):
    """A key moves the budget under a clip that is already trimmed."""
    found = _clip_cap(clip, typed_key)
    if found is None:
        return gr.update()
    duration, cap = found
    start, end = _parse_trim(trim)
    return gr.update(value=_trim_value(start, min(end, duration), duration, cap))


def run_flow(demo: Demo, upload, api_key, trim, height, history, request, *knobs):
    """Drive one run, then record it in this session's history.

    *request* is the open browser tab. On a public deployment its files live
    until that tab closes. A caller with no tab gets a directory removed when
    the call finishes.
    """
    session_hash, ephemeral = _begin_session(request)
    try:
        yield from _drive_run(demo, upload, api_key, trim, height, history,
                              session_hash, _same_origin_browser_request(request),
                              *knobs)
    finally:
        if ephemeral:
            _forget_session(session_hash)


def _drive_run(demo: Demo, upload, api_key, trim, height, history,
               session_hash, ambient_authorized, *knobs):
    if demo.multi:
        # Each clip was cropped on its own and arrives already cut, so there
        # is no shared window: run all of each, as long as the longest one.
        # run_demo still holds that to the frame budget.
        start = 0.0
        durations = [probed[1] for u in (upload or [])
                     if (probed := probe_clip(Path(u)))]
        seconds = max(durations, default=0.5)
    else:
        # The bar is two edges; the pipeline wants an offset and a duration.
        start, end = _parse_trim(trim)
        seconds = max(0.5, end - start)

    final = None
    for out in run_demo(demo, upload, api_key, start, seconds, height, *knobs,
                        session_hash=session_hash,
                        ambient_authorized=ambient_authorized):
        final = out
        yield (*out, history, gr.update(), gr.update(), gr.update())

    # The figure goes once the run is over, however it ended, and the result
    # player comes back into its place.
    log, status, video, highlights, table, plots, files, bundle, picker = final
    video = (gr.update(value=video, visible=True) if isinstance(video, str)
             else gr.update(visible=True))
    final = (log, status, video, highlights, table, plots, files, bundle, picker)
    done = re.search(r"data-state='done' data-elapsed='(\d+)'", status or "")
    if not done:
        yield (*final, history, gr.update(), gr.update(), gr.update(visible=False))
        return

    names = ([Path(f).name for f in (upload or [])] if demo.multi
             else [Path(upload).name] if upload else [])
    clip = names[0] if len(names) == 1 else f"{len(names)} clips"
    history = history + [{
        "n": len(history) + 1,
        "key": demo.key,
        "demo": demo.title,
        "clip": clip,
        # Where the run's downloads are, for the session's zip further down.
        "files": list(files.get("value") or []) if isinstance(files, dict) else [],
        "headline": _headline(highlights if isinstance(highlights, str) else ""),
        "elapsed": f"{done[1]}s",
    }]
    yield (*final, history, gr.update(value=_history_md(history)),
           gr.update(visible=True), gr.update(visible=False))


def _begin_for(demo: Demo):
    """Empty the results before a run starts, and bring on the figure.

    Without this the previous run's video, numbers and plots stay on screen
    while the new one works, and a run that fails leaves them sitting next to
    the new run's error, reading as though they were its result.
    """
    return lambda: (
        "",                                     # log
        waiting.Progress(demo.stages).html(),   # status
        gr.update(value=None, visible=False),   # video: the figure has its place
        "",                                     # highlights
        gr.update(value=None, visible=False),   # per-rep table
        "",                                     # plots
        gr.update(value=None, visible=False),   # downloads
        gr.update(value=None, visible=False),   # downloads as one zip
        gr.update(choices=[], value=None, visible=False),   # render picker
        gr.update(value=waiting.wait_html(demo.wait_caption),
                  visible=True),                # the figure
    )


_SVG = ('<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true" '
        'fill="none" stroke="currentColor" stroke-width="2.6" '
        'stroke-linecap="round" stroke-linejoin="round">{}</svg>')
ICON_CHECK = _SVG.format('<path d="M5 12.5l4.5 4.5L19 7.5"/>')
ICON_CROSS = _SVG.format('<path d="M6.5 6.5l11 11M17.5 6.5l-11 11"/>')
ICON_PLAY = _SVG.format('<path d="M8.5 5.5v13L19 12z" fill="currentColor" stroke-width="1.5"/>')
ICON_PAUSE = _SVG.format('<rect x="6.5" y="5.5" width="3.8" height="13" rx="1" '
                         'fill="currentColor" stroke="none"/>'
                         '<rect x="13.7" y="5.5" width="3.8" height="13" rx="1" '
                         'fill="currentColor" stroke="none"/>')

TRIM_HTML = """
<div class="rs" data-v="${value}">
  <div class="head">
    <span class="label">Crop</span>
    <span class="readout"></span>
    <button type="button" class="play" title="Play selection">@play</button>
    <span class="actions">
      <button type="button" class="cancel" title="Undo">@cross</button>
      <button type="button" class="apply" title="Apply crop">@check</button>
    </span>
  </div>
  <div class="bar">
    <div class="track"><div class="fill"></div><div class="playhead"></div></div>
    <input type="range" class="lo" min="0" step="0.1" aria-label="Crop start">
    <input type="range" class="hi" min="0" step="0.1" aria-label="Crop end">
  </div>
  <div class="warn"></div>
  <div class="hint">@hint Shorter clips run faster.</div>
</div>
"""

TRIM_HTML = (TRIM_HTML.replace("@play", ICON_PLAY).replace("@cross", ICON_CROSS)
             .replace("@check", ICON_CHECK))

TRIM_CSS = """
.rs { padding: 6px 0 2px 0; }
.rs .head { display: flex; align-items: center; gap: 0.6rem; min-height: 32px; }
.rs .label { font-weight: 650; font-size: 0.9rem; }
.rs .readout { flex: 1; font-size: 0.88rem; opacity: 0.8; font-variant-numeric: tabular-nums; }
.rs .actions { display: flex; gap: 0.35rem; visibility: hidden; }
.rs.pending .actions { visibility: visible; }
.rs .play, .rs .actions button {
    width: 30px; height: 30px; border-radius: 50%; cursor: pointer;
    display: inline-flex; align-items: center; justify-content: center;
    padding: 0 !important; line-height: 0;
    border: 1px solid var(--border-color-primary, #d1d5db);
    background: var(--background-fill-primary, #fff); color: var(--body-text-color, #111);
}
.rs .actions .apply {
    background: var(--button-primary-background-fill, #6366f1);
    border-color: var(--button-primary-background-fill, #6366f1); color: #fff;
}
.rs .bar { position: relative; height: 28px; }
.rs .track { position: absolute; left: 8px; right: 8px; top: 10px; height: 8px;
             border-radius: 4px; background: var(--neutral-200, #e5e7eb); }
.rs .fill { position: absolute; top: 0; bottom: 0; border-radius: 4px;
            background: var(--button-primary-background-fill, #6366f1); }
.rs .fill.over { background: #d97706; }
.rs .playhead { position: absolute; top: -5px; bottom: -5px; width: 3px; margin-left: -1px;
                border-radius: 2px; background: var(--body-text-color, #111); display: none; }
.rs input[type=range] {
    position: absolute; left: 0; right: 0; top: 6px; width: 100%; height: 16px;
    margin: 0; background: none; pointer-events: none;
    -webkit-appearance: none; appearance: none;
}
.rs input[type=range]::-webkit-slider-runnable-track { background: none; }
.rs input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none; pointer-events: auto; cursor: grab;
    width: 18px; height: 18px; border-radius: 50%; background: #fff;
    border: 2px solid var(--button-primary-background-fill, #6366f1);
    box-shadow: 0 1px 3px rgba(0,0,0,0.25);
}
.rs input[type=range]::-moz-range-thumb {
    pointer-events: auto; cursor: grab;
    width: 14px; height: 14px; border-radius: 50%; background: #fff;
    border: 2px solid var(--button-primary-background-fill, #6366f1);
}
.rs .warn { font-size: 0.82rem; color: #b45309; min-height: 0; }
.rs .hint { font-size: 0.82rem; opacity: 0.65; margin-top: 2px; }
"""

# Dragging is local: the handles move, the player above seeks to the edge
# being dragged, and nothing is sent. The play button plays just the
# selection, from the start handle to the end one, with a playhead on the bar. The check mark commits the range and
# fires `apply`, which crops the clip server side; the cross puts the handles
# back. Writing the value re-renders the template, so it only happens on
# commit, never mid-drag.
TRIM_JS = """
const q = (sel) => element.querySelector(sel);
const ICON_PLAY = '@play', ICON_PAUSE = '@pause';
const MIN_GAP = 0.5;
let lastRs = null, lastV = null;

function parse() {
    const rs = q('.rs');
    const v = rs ? (rs.dataset.v || '').split(',').map(Number) : [];
    return v.length >= 4 && v.every(Number.isFinite) ? v : null;
}

function player() {
    const col = element.closest('.column');
    return col ? col.querySelector('video') : null;
}

function seek(t) {
    const v = player();
    if (!v || !Number.isFinite(t)) return;
    v.pause();
    v.currentTime = t;
}

let playing = false;

function stopPreview() {
    playing = false;
    const b = q('.play'), ph = q('.playhead');
    if (b) b.innerHTML = ICON_PLAY;
    if (ph) ph.style.display = 'none';
}

function tick() {
    const v = player(), vals = parse();
    if (!playing || !v || !vals) return stopPreview();
    const end = +q('.hi').value;
    if (v.currentTime >= end || v.ended) {
        v.pause();
        v.currentTime = end;
        return stopPreview();
    }
    const ph = q('.playhead');
    ph.style.display = 'block';
    ph.style.left = (100 * v.currentTime / vals[2]) + '%';
    requestAnimationFrame(tick);
}

function playSelection() {
    const v = player();
    if (!v) return;
    if (playing) { v.pause(); return stopPreview(); }
    v.currentTime = +q('.lo').value;
    playing = true;
    q('.play').innerHTML = ICON_PAUSE;
    // Tracking starts once playback really has. Paused after that, from the
    // player's own controls or by a drag, and the preview ends.
    v.addEventListener('playing', () => {
        if (!playing) return;
        v.addEventListener('pause', stopPreview, { once: true });
        tick();
    }, { once: true });
    // Started through the player's own button rather than v.play(): Gradio's
    // player keeps its own playing/paused state, and a play from outside it
    // can be undone moments later when that state catches up.
    const own = element.closest('.column')
        ?.querySelector('[aria-label="play-pause-replay-button"]');
    if (own) own.click(); else v.play().catch(stopPreview);
    // A browser that refuses to play leaves nothing to wait for.
    setTimeout(() => { if (playing && v.paused) stopPreview(); }, 1500);
}

function draw() {
    const v = parse();
    if (!v) return;
    const [s0, e0, d, cap] = v;
    const s = +q('.lo').value, e = +q('.hi').value;
    const fill = q('.fill');
    fill.style.left = (100 * s / d) + '%';
    fill.style.width = (100 * (e - s) / d) + '%';
    const over = e - s > cap + 0.05;
    fill.classList.toggle('over', over);
    q('.readout').textContent = s.toFixed(1) + 's – ' + e.toFixed(1) + 's';
    q('.warn').textContent = over
        ? 'Over the limit: only the first ' + cap.toFixed(1) + 's will run.' : '';
    const pending = Math.abs(s - s0) > 0.05 || Math.abs(e - e0) > 0.05;
    q('.rs').classList.toggle('pending', pending);
}

function sync() {
    const rs = q('.rs');
    if (!rs || (rs === lastRs && rs.dataset.v === lastV)) return;
    lastRs = rs; lastV = rs.dataset.v;
    const v = parse();
    if (!v) return;
    const lo = q('.lo'), hi = q('.hi');
    lo.max = hi.max = v[2];
    lo.value = v[0]; hi.value = v[1];
    draw();
}

element.addEventListener('input', (ev) => {
    const lo = q('.lo'), hi = q('.hi');
    if (!parse()) return;
    if (ev.target === lo && +lo.value > +hi.value - MIN_GAP) lo.value = +hi.value - MIN_GAP;
    if (ev.target === hi && +hi.value < +lo.value + MIN_GAP) hi.value = +lo.value + MIN_GAP;
    if (ev.target === lo || ev.target === hi) seek(+ev.target.value);
    draw();
});

element.addEventListener('click', (ev) => {
    const v = parse();
    if (!v) return;
    const lo = q('.lo'), hi = q('.hi');
    if (ev.target.closest('.play')) {
        playSelection();
    } else if (ev.target.closest('.cancel')) {
        lo.value = v[0]; hi.value = v[1];
        seek(v[0]);
        draw();
    } else if (ev.target.closest('.apply')) {
        q('.readout').textContent = 'Cropping…';
        props.value = [(+lo.value).toFixed(1), (+hi.value).toFixed(1), v[2], v[3]].join(',');
        trigger('apply');
    }
});

new MutationObserver(sync).observe(element, {
    childList: true, subtree: true, attributes: true, attributeFilter: ['data-v'],
});
sync();
"""


TRIM_JS = TRIM_JS.replace("@play", ICON_PLAY).replace("@pause", ICON_PAUSE)


def _sweep_crops(max_age: float = CROP_MAX_AGE) -> None:
    """Delete crop directories that have sat unused past *max_age* seconds."""
    if not CROP_DIR.is_dir():
        return
    cutoff = time.time() - max_age
    for child in list(CROP_DIR.iterdir()):
        try:
            if child.stat().st_mtime >= cutoff:
                continue
        except OSError:
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def _drop_crop(path) -> None:
    """Delete a crop this app made, once another file replaces it.

    A persistent crop is a directory under CROP_DIR. A hosted one is a
    directory under that tab's ``crops`` folder. Gradio's own upload, and
    either crop root, are not touched.
    """
    if not path:
        return
    try:
        folder = Path(path).resolve().parent
        parent = folder.parent
    except OSError:
        return
    owned = parent == CROP_DIR.resolve() or (
        parent.name == "crops" and _under_session_store(parent))
    if owned:
        shutil.rmtree(folder, ignore_errors=True)


def _crop_root(request) -> Path:
    """Where a crop is written: the open tab's folder, or the shared temp one."""
    shared = _shared_session_dir(_touch_session(request), "crops")
    if shared is not None:
        return shared
    return CROP_DIR


def _crop_clip(path: Path, start: float, end: float, directory: Path | None = None) -> Path:
    """A copy of *path* holding only *start* to *end*, under the same name.

    Same name because the pipelines name their outputs after the stem. Stream
    copied, like `_stage_input` and for the same reason: re-encoding would take
    over the rotation and HDR handling the demos are careful about. So the cut
    lands on the keyframe at or before *start*, a fraction of a second early.
    The copy lives under CROP_DIR locally, or under the open tab's own folder
    when the app is hosted. Either way it is removed when it is replaced, and
    a hosted one goes away with the tab.
    """
    root = directory or CROP_DIR
    if root == CROP_DIR:
        _sweep_crops()
    folder = root / uuid.uuid4().hex
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / path.name
    proc = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-ss", f"{start:.2f}", "-i", str(path), "-t", f"{end - start:.2f}",
         "-c", "copy", "-map", "0", "-avoid_negative_ts", "make_zero",
         # Keeps when it was filmed, which orders a run's sets.
         "-map_metadata", "0", "-movflags", "use_metadata_tags", str(out)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not out.is_file() or out.stat().st_size == 0:
        shutil.rmtree(folder, ignore_errors=True)
        raise RuntimeError(proc.stderr.strip()[-300:] or "ffmpeg produced nothing")
    return out


# The preview player only needs a clip a browser can play. Gradio re-encodes any
# upload that is not browser-playable (a phone's 4K HEVC .mov, say) at full size
# before it will show it, which took about 14s for a 30s 4K clip. A small
# H.264 copy made here takes about 4s, and is what the player and the sizing
# handlers read. Runs, crops and trims keep using the original.
PREVIEW_HEIGHT = 720
_SRC_TAG = "vd-src:"
_PREVIEW_CACHE: dict[tuple[str, int, int], str] = {}


def _preview_encoder() -> list[str]:
    """Hardware H.264 where ffmpeg has it (Macs), software otherwise."""
    if not hasattr(_preview_encoder, "found"):
        try:
            out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                                 capture_output=True, text=True, timeout=15).stdout
        except Exception:
            out = ""
        _preview_encoder.found = (
            ["-c:v", "h264_videotoolbox", "-b:v", "4M"]
            if "h264_videotoolbox" in out else
            ["-c:v", "libx264", "-preset", "veryfast", "-crf", "26"])
    return _preview_encoder.found


def _source_size(path: Path) -> tuple[int, int] | None:
    """Display size of the clip a preview was made from, else of *path*."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-of", "json",
             "-show_entries", "format_tags=comment", str(path)],
            capture_output=True, text=True, timeout=30)
        comment = json.loads(out.stdout or "{}").get("format", {}).get("tags", {}).get("comment", "")
        if comment.startswith(_SRC_TAG):
            w, h = comment[len(_SRC_TAG):].split("x")
            return int(w), int(h)
    except Exception:
        pass
    return display_size(path)


def _preview_clip(path: str, root: Path | None = None) -> str:
    """A small playable copy of *path* for the player, or *path* if that is not needed."""
    source = Path(path)
    try:
        stat = source.stat()
    except OSError:
        return path
    key = (str(source), stat.st_size, stat.st_mtime_ns)
    cached = _PREVIEW_CACHE.get(key)
    if cached and Path(cached).is_file():
        return cached
    size = display_size(source)
    if not size:
        return path
    try:
        playable = processing_utils.video_is_playable(str(source))
    except Exception:
        playable = False
    if playable and size[1] <= PREVIEW_HEIGHT * 1.5:
        return path

    root = root or CROP_DIR
    if root == CROP_DIR:
        _sweep_crops()
    folder = root / uuid.uuid4().hex
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"{source.stem}.mp4"
    proc = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source), "-map", "0:v:0",
         "-vf", f"scale=-2:'min({PREVIEW_HEIGHT},ih)'", *_preview_encoder(),
         "-pix_fmt", "yuv420p", "-an", "-metadata", f"comment={_SRC_TAG}{size[0]}x{size[1]}",
         "-movflags", "+faststart", str(out)],
        capture_output=True, text=True)
    if proc.returncode != 0 or not out.is_file() or out.stat().st_size == 0:
        shutil.rmtree(folder, ignore_errors=True)
        return path
    _PREVIEW_CACHE[key] = str(out)
    return str(out)


def _on_crop(upload, trim, request: gr.Request):
    """The check mark: replace the clip with its cropped copy.

    Replacing it fires the upload's change, which sizes the bar to the new,
    shorter clip. The upload itself is left where Gradio put it, so reverting
    is only a matter of pointing the player back at it. A batch has no single
    player to update and is never wired to this: there the committed range is
    simply what the run uses, on every clip.
    """
    if not upload:
        return gr.update(), gr.update()
    _remember_paths(_touch_session(request), [upload])
    start, end = _parse_trim(trim)
    try:
        cropped = str(_crop_clip(Path(upload), start, end, _crop_root(request)))
    except Exception as exc:
        gr.Warning(f"Could not crop that clip: {exc}")
        return gr.update(), gr.update()
    if str(upload) != cropped:
        _drop_crop(upload)
    return cropped, gr.update(visible=True)


# A batch keeps one entry per uploaded clip, {"original": path, "current":
# path}, where current is the latest crop. The picker chooses which clip the
# preview and the crop bar are showing; each keeps its own crop.

def _pick_choices(clips: list[dict]) -> list[tuple[str, str]]:
    return [(f"{i + 1}. {Path(c['original']).name}", str(i))
            for i, c in enumerate(clips)]


def _show_clip(clips: list[dict], idx: int, typed_key, root: Path | None = None):
    """(preview, bar, revert) for clip *idx*."""
    clip = clips[idx]
    return (_preview_clip(clip["current"], root), _on_upload(clip["current"], typed_key),
            gr.update(visible=clip["current"] != clip["original"]))


def _on_batch_upload(files, typed_key, previous, request: gr.Request):
    _remember_paths(_touch_session(request), files)
    for clip in previous or []:
        current = clip.get("current") if isinstance(clip, dict) else None
        original = clip.get("original") if isinstance(clip, dict) else None
        if current and current != original:
            _drop_crop(current)
    clips = [{"original": f, "current": f} for f in (files or [])]
    if not clips:
        return ([], gr.update(choices=[], value=None, visible=False),
                gr.update(value=None, visible=False),
                gr.update(value="", visible=False), gr.update(visible=False))
    preview, bar, revert = _show_clip(clips, 0, typed_key, _crop_root(request))
    return (clips,
            gr.update(choices=_pick_choices(clips), value="0",
                      visible=len(clips) > 1),
            gr.update(value=preview, visible=True), bar, revert)


def _on_batch_pick(clips, idx, typed_key, request: gr.Request):
    if not clips or idx is None:
        return gr.update(), gr.update(), gr.update()
    return _show_clip(clips, int(idx), typed_key, _crop_root(request))


def _on_batch_crop(clips, idx, trim, typed_key, request: gr.Request):
    """The check mark, for the clip on show. The others keep their own crop."""
    if not clips or idx is None:
        return clips, gr.update(), gr.update(), gr.update()
    i = int(idx)
    start, end = _parse_trim(trim)
    previous = clips[i]["current"]
    try:
        cropped = str(_crop_clip(Path(previous), start, end, _crop_root(request)))
    except Exception as exc:
        gr.Warning(f"Could not crop that clip: {exc}")
        return clips, gr.update(), gr.update(), gr.update()
    clips = [dict(c) for c in clips]
    clips[i]["current"] = cropped
    if str(previous) != cropped:
        _drop_crop(previous)
    return (clips, *_show_clip(clips, i, typed_key, _crop_root(request)))


def _on_batch_revert(clips, idx, typed_key, request: gr.Request):
    if not clips or idx is None:
        return clips, gr.update(), gr.update(), gr.update()
    i = int(idx)
    clips = [dict(c) for c in clips]
    previous = clips[i]["current"]
    clips[i]["current"] = clips[i]["original"]
    if str(previous) != str(clips[i]["original"]):
        _drop_crop(previous)
    return (clips, *_show_clip(clips, i, typed_key, _crop_root(request)))


def _on_revert(original, current):
    """Point the player back at the upload, and delete the crop it replaced."""
    if current and original and str(current) != str(original):
        _drop_crop(current)
    return original, gr.update(visible=False)


def _on_clear(current):
    _drop_crop(current)
    return None, gr.update(visible=False)


def _on_upload_seen(clip, typed_key, request: gr.Request):
    """Size the trim bar, and remember the upload so a hosted tab can delete it."""
    _remember_paths(_touch_session(request), [clip] if clip else [])
    return _on_upload(clip, typed_key)


def _on_uploaded(upload, request: gr.Request):
    """The visitor's own file, before any crop replaces it."""
    _remember_paths(_touch_session(request), [upload] if upload else [])
    return upload, gr.update(visible=False)


def _runner_for(demo: Demo):
    """run_flow with *demo* bound, as the generator Gradio streams from."""
    def run(upload, clips, api_key, trim, height, history, request: gr.Request, *knobs):
        # A batch runs each clip as cropped. `clips` is session state, which
        # an API caller cannot set, so the uploaded files stand in for it.
        _remember_paths(_touch_session(request), clips if demo.multi else [upload])
        if demo.multi and clips:
            upload = [c["current"] for c in clips]
        yield from run_flow(demo, upload, api_key, trim, height, history,
                            request, *knobs)
    return run


def _height_choices(size: tuple[int, int] | None) -> list[tuple[str, int]]:
    """Render heights, with full resolution named for the clip once known.

    The pipelines never scale up, so a height above the clip's own is only the
    clip's own again; those are left out rather than offered as if different.
    """
    full = "Full resolution" + (f" ({size[0]}\u00d7{size[1]})" if size else "")
    heights = [h for h in (480, 720, 1080) if not size or h < size[1]]
    return [(f"{h}p", h) for h in heights] + [(full, 0)]


def _on_clip_size(clip, current):
    """Name full resolution for this clip, keeping the choice where it stays valid."""
    size = _source_size(Path(clip)) if clip else None
    choices = _height_choices(size)
    values = [v for _, v in choices]
    value = current if current in values else (720 if 720 in values else 0)
    return gr.update(choices=choices, value=value)


@dataclass
class Panel:
    """One demo's components. Kept per demo so each holds its own state."""

    demo: Demo
    column: gr.Column
    upload: gr.components.Component
    clip: gr.Video                   # the one clip the crop bar is about
    trim: gr.HTML
    original: gr.State               # single clip: as uploaded, before any crop
    clips: gr.State                  # batch: each clip's original and crop
    picker: gr.Radio | None          # batch: which clip is on show
    revert: gr.Button
    knobs: list
    export_height: gr.Dropdown
    go: gr.Button
    outputs: list                    # log, status, video, highlights, table, plots, files, bundle, picker
    video_pick: gr.Dropdown
    video_out: gr.Video
    wait: gr.HTML                    # the figure shown while a run works


def _build_panel(demo: Demo, visible: bool) -> Panel:
    with gr.Column(visible=visible) as column:
        gr.Markdown(
            f"<div class='demo-head'><h2>{demo.title}</h2>"
            f"<p>{_inline(demo.tagline)}</p></div>"
        )
        if demo.notes:
            gr.Markdown(f"<div class='demo-note'>{_inline(demo.notes)}</div>")

        with gr.Row(equal_height=False):
            # ---- input ------------------------------------------------------
            with gr.Column(scale=5, min_width=320):
                picker = None
                if demo.multi:
                    upload = gr.File(label=demo.upload_label,
                                     file_count="multiple",
                                     file_types=[".mov", ".mp4", ".m4v", ".avi"])
                    picker = gr.Radio(label="Crop clip", choices=[], visible=False)
                    clip = gr.Video(label="Preview", interactive=False, visible=False,
                                    height=PLAYER_HEIGHT, elem_classes="clip-player")
                else:
                    upload = gr.Video(label="Clip", sources=["upload"],
                                      height=PLAYER_HEIGHT, elem_classes="clip-player")
                    clip = upload

                # The hint is fixed per demo, so it is baked into this demo's
                # copy of the template rather than passed as a prop.
                trim = gr.HTML("", visible=False,
                               html_template=TRIM_HTML.replace(
                                   "@hint", html.escape(demo.crop_hint)),
                               css_template=TRIM_CSS, js_on_load=TRIM_JS)
                original = gr.State(None)
                clips = gr.State([])
                revert = gr.Button("Revert to original", size="sm", visible=False)

                # The two or three settings that decide what a result means sit
                # in the open; they are the whole reason there is a choice here.
                # Built in two places but handed to the run in one order, the
                # order demo.knobs lists them.
                built = {k.key: _build_knob(k) for k in demo.knobs if not k.optional}
                with gr.Accordion("Optional", open=False):
                    built.update({k.key: _build_knob(k)
                                  for k in demo.knobs if k.optional})
                    export_height = gr.Dropdown(
                        choices=_height_choices(None), value=720,
                        label="Render height",
                        info="The height the result video is drawn at. Full "
                             "resolution keeps the clip's own size. Detection "
                             f"always runs at {INFERENCE_HEIGHT}p, so changing "
                             "this re-renders without calling the model again.")
                knobs = [c for k in demo.knobs for c in built[k.key]]
                go = gr.Button("Run", variant="primary", elem_classes="go")

            # ---- results ----------------------------------------------------
            with gr.Column(scale=6, min_width=320):
                video_pick = gr.Dropdown(label="Which render", visible=False,
                                         interactive=True, filterable=False)
                # While a run works the figure stands in the result's place, at
                # the result's height, so the result lands level with the clip
                # it came from. Separate from the stages so updating them does
                # not restart the animation: this is set once per run.
                wait = gr.HTML("", visible=False, js_on_load=waiting.WAIT_JS,
                               elem_classes="wait-slot",
                               min_height=PLAYER_HEIGHT)
                video_out = gr.Video(label="Result", interactive=False,
                                     height=PLAYER_HEIGHT, elem_classes="clip-player")
                status = gr.HTML(waiting.notice_html(
                    "Results appear here once a run finishes."))
                highlights = gr.HTML("")
                table = gr.Dataframe(interactive=False, visible=False,
                                     max_height=360, elem_classes="result-table")
                plots = gr.HTML("")
                files = gr.File(label="Downloads", file_count="multiple",
                                interactive=False, visible=False,
                                elem_classes="downloads")
                bundle = gr.DownloadButton("Download all (.zip)", visible=False,
                                           size="sm", elem_classes="bundle")
                with gr.Accordion("Run log", open=False):
                    log = gr.Textbox(label="", lines=20, max_lines=40,
                                     buttons=["copy"])

        with gr.Accordion("How it works", open=False):
            models = " + ".join(f"`{m.strip()}`" for m in demo.model.split("+"))
            gr.Markdown(f"**Models:** {models}")
            gr.Markdown(_explainer(demo) or "No notes for this one.")

    return Panel(
        demo=demo, column=column, upload=upload, clip=clip, trim=trim,
        original=original, clips=clips, picker=picker, revert=revert, knobs=knobs,
        export_height=export_height, go=go,
        outputs=[log, status, video_out, highlights, table, plots, files,
                 bundle, video_pick],
        video_pick=video_pick, video_out=video_out, wait=wait,
    )


with gr.Blocks(title="Vision Demos",
               delete_cache=(600, 3600) if SESSION_RETENTION else None) as demo_app:
    # Which demo was showing, remembered in the browser so a reload lands back
    # on it. History stays per session: it has no business outliving the tab.
    # A fixed secret, because the default is regenerated on every launch and
    # a restart would then forget the choice; a demo name is not a secret.
    st_last = gr.BrowserState(DEMOS[0].key, storage_key="vision-demos-tab",
                              secret="vision-demos-tab")
    st_history = gr.State([])

    gr.Markdown(
        "<div class='hero'><h1><a href='https://github.com/jeremyipark/vision-demos' target='_blank' rel='noopener' style='color:inherit;text-decoration:none'>Vision Demos</a></h1>"
        "<p>Real-world computer vision. Upload a clip, get the results back."
        "</p></div>"
    )

    # Billing is asked up front, above the demos, as a card, because a key box
    # tucked into a corner is one nobody finds. Picking an option selects it
    # and the check mark applies it; then it folds to a single line with a
    # Change button. The browser remembers the choice, never the key.
    st_billing = gr.BrowserState("", storage_key="vision-demos-billing",
                                 secret="vision-demos-billing")
    key_mode = gr.Textbox(ANON_MODE, visible=False)       # the choice in force
    key_pending = gr.Textbox(ANON_MODE, visible=False)    # the one selected

    with gr.Column(visible=False, elem_classes="billing-card") as billing_card:
        gr.Markdown("<h3>How should runs be billed?</h3>"
                    "<p>Pick one, then confirm. You can change it at any time.</p>")
        with gr.Row(equal_height=True):
            with gr.Column(elem_classes="billing-choice"):
                pick_anon = gr.Button("Free, anonymous", variant="primary")
                gr.Markdown(f"<p>No sign-up. {ANON_LIMITS}, shared per IP.</p>")
            with gr.Column(elem_classes="billing-choice"):
                pick_key = gr.Button("Use my API key", variant="secondary")
                gr.Markdown(
                    f"<p>Billed to your VLM Run key. {KEYED_LIMITS}, per user.</p>")
        gr.Markdown(f"<p class='billing-note'>{RATE_LIMITS_NOTE}</p>")
        # Where the key was found, never the key. On a deployment that does
        # not serve its own key this is always the not-found message.
        key_source = gr.Markdown(
            f"<p class='key-found'>&#x2713; Using the API key found in "
            f"<code>{html.escape(AMBIENT_SOURCE)}</code>.</p>" if USE_AMBIENT else
            "<p class='key-missing'>No API key found in <code>.env</code> or "
            "<code>VLMRUN_API_KEY</code>. Paste one below.</p>",
            visible=False)
        key_box = gr.Textbox(
            label=("Manual override (optional)" if USE_AMBIENT else "VLM Run API key"),
            type="password", value="", visible=False,
            placeholder=("paste a different key to use instead" if USE_AMBIENT
                         else "paste a key from app.vlm.run"),
            elem_classes="key-box")
        with gr.Row(elem_classes="billing-confirm"):
            billing_confirm = gr.Button("\u2713  Confirm", variant="primary",
                                        scale=0, min_width=150)

    with gr.Row(visible=False, elem_classes="billing") as billing_line:
        billing_note = gr.Markdown(
            f"<p class='billing-note'>{_billing_note(ANON_MODE, '')}</p>")
        billing_change = gr.Button("Change API key setting", size="sm", scale=0,
                                   min_width=180)

    with gr.Row(elem_classes="topbar"):
        toggle = gr.Radio(
            choices=[(d.title, d.key) for d in DEMOS], value=DEMOS[0].key,
            show_label=False, container=False, elem_classes="demo-toggle",
            scale=4,
        )
    # The key every run and every budget readout is given, kept in sync with
    # the two controls above. A hidden Textbox rather than State so the API
    # can set it too: blank for anonymous, or a key.
    run_key = gr.Textbox("", visible=False)

    panels = [_build_panel(d, visible=(i == 0)) for i, d in enumerate(DEMOS)]

    with gr.Column(visible=False, elem_classes="history-wrap") as history_panel:
        gr.Markdown("#### This session")
        history_md = gr.Markdown("")
        gr.Markdown("<p class='muted'>" + (
            "Kept until you close or refresh this tab. That deletes the clips "
            "and results from this visit. Switching demos does not."
            if SESSION_RETENTION else
            "Kept for as long as this browser tab is connected. "
            "Results themselves stay on disk."
        ) + "</p>")

        # Every run so far, across demos, in one download. Each run is a
        # folder in the zip, so two clips' summary.csv never collide.
        with gr.Column(elem_classes="session-zip"):
            gr.Markdown("#### Download results")
            zip_runs = gr.CheckboxGroup(label="Runs", choices=[], value=[])
            with gr.Row(elem_classes="zip-row"):
                zip_kinds = gr.CheckboxGroup(list(KINDS), value=list(KINDS),
                                             label="Include", scale=3)
                zip_all = gr.Button("Select all", size="sm", scale=0, min_width=110)
                zip_none = gr.Button("Select none", size="sm", scale=0, min_width=110)
            zip_go = gr.DownloadButton("Nothing selected", variant="primary",
                                       interactive=False)

    # ---- the session's zip -------------------------------------------------
    st_history.change(_on_history, inputs=[st_history, zip_runs],
                      outputs=[zip_runs], api_name=False)
    zip_all.click(lambda h: [x["n"] for x in h or []], inputs=[st_history],
                  outputs=[zip_runs], api_name=False)
    zip_none.click(lambda: [], outputs=[zip_runs], api_name=False)
    for _picker in (zip_runs, zip_kinds):
        _picker.change(_on_session_zip, inputs=[st_history, zip_runs, zip_kinds],
                       outputs=[zip_go], api_name=False,
                       trigger_mode="always_last", show_progress="minimal")

    # ---- the toggle --------------------------------------------------------
    def _show(key):
        key = key if key in DEMOS_BY_KEY else DEMOS[0].key
        return (key, key, *[gr.update(visible=(p.demo.key == key)) for p in panels])

    columns = [p.column for p in panels]
    toggle.change(_show, inputs=[toggle], outputs=[toggle, st_last, *columns],
                  api_name=False)
    demo_app.load(_show, inputs=[st_last], outputs=[toggle, st_last, *columns],
                  api_name=False)

    # ---- billing -----------------------------------------------------------
    pick_outputs = [key_pending, pick_anon, pick_key, key_source, key_box]
    for _btn, _mode in ((pick_anon, ANON_MODE), (pick_key, KEY_MODE)):
        _btn.click(lambda _m=_mode: _on_pick(_m), outputs=pick_outputs,
                   api_name=False)
    billing_confirm.click(
        _on_confirm, inputs=[key_pending, key_box],
        outputs=[key_mode, run_key, billing_note, billing_card, billing_line,
                 st_billing],
        api_name=False)
    billing_change.click(_on_change, inputs=[key_mode],
                         outputs=[billing_card, billing_line, *pick_outputs],
                         api_name=False)
    demo_app.load(_on_restore, inputs=[st_billing],
                  outputs=[key_mode, run_key, billing_note, billing_card,
                           billing_line, *pick_outputs],
                  api_name=False)

    # ---- per demo ----------------------------------------------------------
    for p in panels:
        if p.demo.multi:
            p.upload.change(_on_batch_upload, inputs=[p.upload, run_key, p.clips],
                            outputs=[p.clips, p.picker, p.clip, p.trim, p.revert],
                            api_name=False)
            # `input`, not `change`: only a click on the picker, not the
            # value set above when the clips first arrive.
            p.picker.input(_on_batch_pick, inputs=[p.clips, p.picker, run_key],
                           outputs=[p.clip, p.trim, p.revert], api_name=False)
            p.trim.apply(_on_batch_crop,
                         inputs=[p.clips, p.picker, p.trim, run_key],
                         outputs=[p.clips, p.clip, p.trim, p.revert],
                         api_name=False)
            p.revert.click(_on_batch_revert, inputs=[p.clips, p.picker, run_key],
                           outputs=[p.clips, p.clip, p.trim, p.revert],
                           api_name=False)
        else:
            p.upload.change(_on_upload_seen, inputs=[p.upload, run_key],
                            outputs=[p.trim], api_name=False)
            # `upload` fires for the user's own files only, never for the
            # cropped copy put back in their place, so it marks the original.
            p.upload.upload(_on_uploaded, inputs=[p.upload],
                            outputs=[p.original, p.revert], api_name=False)
            p.upload.clear(_on_clear, inputs=[p.upload],
                           outputs=[p.original, p.revert], api_name=False)
            p.trim.apply(_on_crop, inputs=[p.upload, p.trim],
                         outputs=[p.upload, p.revert], api_name=False)
            p.revert.click(_on_revert, inputs=[p.original, p.upload],
                           outputs=[p.upload, p.revert], api_name=False)
        p.clip.change(_on_clip_size, inputs=[p.clip, p.export_height],
                      outputs=[p.export_height], api_name=False)
        # A key moves the budget under a clip that is already trimmed.
        run_key.change(_on_key, inputs=[p.clip, run_key, p.trim],
                       outputs=[p.trim], api_name=False)

        # Gradio's own loading overlay is hidden: the figure and the stages
        # are the progress display, and its bars across every output read
        # as stray lines on top of them.
        p.go.click(_begin_for(p.demo), outputs=[*p.outputs, p.wait],
                   api_name=False, show_progress="hidden").then(
            _runner_for(p.demo),
            inputs=[p.upload, p.clips, run_key, p.trim, p.export_height,
                    st_history, *p.knobs],
            outputs=[*p.outputs, st_history, history_md, history_panel, p.wait],
            api_name=f"run_{p.demo.key}", show_progress="hidden",
            # One queue for every demo, not one each: six pipelines at once
            # would split a Space's two CPUs six ways.
            concurrency_id="pipeline", concurrency_limit=1,
        )
        p.video_pick.change(lambda chosen: chosen, inputs=[p.video_pick],
                            outputs=[p.video_out], api_name=False)

    gr.Markdown("<p class='muted' style='margin-top:2.5rem'>Created by Jeremy Park, PhD<br>"
        "<a href='https://www.linkedin.com/in/jeremyipark/' target='_blank' rel='noopener'>LinkedIn</a> &nbsp;·&nbsp; "
        "<a href='https://x.com/jeremyparkphd' target='_blank' rel='noopener'>X</a> &nbsp;·&nbsp; "
        "<a href='https://www.youtube.com/@jeremyparkphd' target='_blank' rel='noopener'>YouTube</a></p>")


def _on_unload(request: gr.Request):
    """The tab's connection dropped: close, refresh, or a lost network.

    Switching from one demo to another does not come through here, so that
    demo's clip and result stay while the visitor is still in the app.
    """
    _forget_session(getattr(request, "session_hash", None))


demo_app.unload(_on_unload)


if __name__ == "__main__":
    _sweep_crops()
    _sweep_sessions_on_startup()
    _start_session_sweeper()
    demo_app.queue(default_concurrency_limit=1).launch(
        server_name=SERVER_NAME, share=SHARE, max_file_size=MAX_UPLOAD,
        theme=gr.themes.Soft(), css=CSS, js=PREVIEW_JS)
