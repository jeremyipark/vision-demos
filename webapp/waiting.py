"""What the app shows while a run works: a moving figure, and the stages.

The figure is a pose skeleton, drawn the way ViTPose's own overlay draws one,
running on a loop. It is pure SVG animation, so it keeps
moving however long the gateway takes and costs nothing to serve.

The stages are read off the pipeline's own log. Every pipeline announces each
step with a numbered heading (`4 Pose estimation ─────`), so following along
needs nothing from the pipelines beyond what they already print.
"""

from __future__ import annotations

import html
import math
import re
import time
from dataclasses import dataclass, field

# ── The figure ───────────────────────────────────────────────────────────────
# Joints on a 100-unit box. "l" and "r" are the viewer's left and right.
BONES = (
    ("neck", "pelvis", "mid"),
    ("neck", "lsh", "l"), ("lsh", "lel", "l"), ("lel", "lwr", "l"),
    ("neck", "rsh", "r"), ("rsh", "rel", "r"), ("rel", "rwr", "r"),
    ("pelvis", "lhip", "l"), ("lhip", "lkn", "l"), ("lkn", "lan", "l"),
    ("pelvis", "rhip", "r"), ("rhip", "rkn", "r"), ("rkn", "ran", "r"),
)
COLOURS = {"l": "#22d3ee", "r": "#f472b6", "mid": "#a78bfa"}


# One gait cycle, at 24 frames. It started as the mean of 109 cycles of a real
# runner on a treadmill (ViTPose, side on, 180 steps a minute), smoothed to its
# first four harmonics so it loops without a seam. It was then changed on
# purpose so it is nobody in particular: a 15% longer stride, a 25-30% bigger
# arm swing, a lower heel kick, the torso set to a 5 degree forward lean,
# a gentler bounce, and a slightly quicker cadence than the source.
RUN_FRAMES = [
    dict(head=(61.8, 24.8), neck=(58.1, 31.3), lsh=(54.3, 34.9), rsh=(58.8, 33.3), lel=(43.8, 44.0), rel=(55.5, 42.7), lwr=(54.0, 49.1), rwr=(64.4, 45.5), pelvis=(55.7, 61.3), lhip=(57.2, 61.4), rhip=(54.2, 61.2), lkn=(64.2, 78.4), rkn=(48.9, 79.9), lan=(67.7, 96.8), ran=(32.1, 93.7)),
    dict(head=(61.7, 25.2), neck=(58.0, 31.7), lsh=(54.3, 35.3), rsh=(58.5, 33.7), lel=(44.1, 44.8), rel=(54.7, 43.1), lwr=(55.0, 49.1), rwr=(63.9, 46.4), pelvis=(55.7, 61.7), lhip=(57.2, 61.8), rhip=(54.2, 61.7), lkn=(64.2, 78.6), rkn=(50.9, 80.5), lan=(65.9, 97.2), ran=(31.9, 92.5)),
    dict(head=(61.6, 25.9), neck=(58.0, 32.4), lsh=(54.7, 35.8), rsh=(58.1, 34.4), lel=(44.9, 46.1), rel=(53.5, 44.0), lwr=(56.1, 50.1), rwr=(63.1, 48.0), pelvis=(55.7, 62.4), lhip=(57.3, 62.4), rhip=(54.1, 62.4), lkn=(63.8, 78.9), rkn=(53.4, 81.1), lan=(62.6, 97.6), ran=(32.6, 91.4)),
    dict(head=(61.7, 26.6), neck=(58.2, 33.0), lsh=(55.7, 36.4), rsh=(57.6, 35.2), lel=(46.9, 47.4), rel=(52.3, 45.0), lwr=(57.4, 51.9), rwr=(62.0, 50.2), pelvis=(55.7, 63.0), lhip=(57.3, 62.9), rhip=(54.1, 63.0), lkn=(62.8, 79.4), rkn=(56.3, 81.5), lan=(58.8, 98.1), ran=(34.6, 90.5)),
    dict(head=(62.0, 27.1), neck=(58.7, 33.6), lsh=(57.3, 36.9), rsh=(57.4, 35.8), lel=(49.8, 48.4), rel=(51.3, 45.7), lwr=(58.8, 53.9), rwr=(61.1, 52.2), pelvis=(55.7, 63.3), lhip=(57.1, 63.2), rhip=(54.3, 63.4), lkn=(61.3, 80.2), rkn=(59.4, 81.4), lan=(55.3, 98.6), ran=(37.7, 90.1)),
    dict(head=(62.7, 27.2), neck=(59.6, 33.8), lsh=(59.1, 37.1), rsh=(57.5, 36.0), lel=(53.1, 48.9), rel=(50.6, 45.8), lwr=(60.6, 54.7), rwr=(60.3, 53.2), pelvis=(55.7, 63.3), lhip=(56.6, 63.2), rhip=(54.7, 63.3), lkn=(59.6, 81.1), rkn=(62.5, 80.8), lan=(52.2, 98.8), ran=(41.7, 90.4)),
    dict(head=(63.4, 26.9), neck=(60.4, 33.5), lsh=(60.4, 37.0), rsh=(57.8, 35.7), lel=(55.7, 48.7), rel=(49.9, 45.0), lwr=(62.5, 53.7), rwr=(59.6, 52.9), pelvis=(55.7, 62.9), lhip=(56.0, 62.9), rhip=(55.3, 62.9), lkn=(57.7, 81.5), rkn=(65.2, 79.8), lan=(49.2, 98.7), ran=(46.0, 91.3)),
    dict(head=(63.9, 26.1), neck=(60.8, 32.9), lsh=(61.0, 36.6), rsh=(57.8, 35.0), lel=(57.0, 48.0), rel=(48.9, 43.7), lwr=(63.8, 51.5), rwr=(58.6, 51.5), pelvis=(55.7, 62.3), lhip=(55.5, 62.3), rhip=(55.8, 62.2), lkn=(55.6, 81.2), rkn=(67.1, 78.7), lan=(46.0, 98.2), ran=(50.3, 92.7)),
    dict(head=(63.9, 25.3), neck=(60.6, 32.1), lsh=(60.9, 35.9), rsh=(57.5, 34.2), lel=(57.3, 47.2), rel=(47.7, 42.1), lwr=(64.4, 49.2), rwr=(57.3, 49.8), pelvis=(55.7, 61.5), lhip=(55.3, 61.6), rhip=(56.0, 61.5), lkn=(53.3, 80.4), rkn=(67.8, 77.6), lan=(42.7, 97.3), ran=(54.7, 94.1)),
    dict(head=(63.7, 24.5), neck=(60.2, 31.4), lsh=(60.6, 35.2), rsh=(56.8, 33.4), lel=(57.2, 46.3), rel=(46.5, 40.8), lwr=(64.2, 47.7), rwr=(56.0, 48.3), pelvis=(55.7, 60.9), lhip=(55.5, 60.9), rhip=(55.9, 60.8), lkn=(51.2, 79.5), rkn=(67.6, 77.0), lan=(39.8, 96.0), ran=(59.2, 95.4)),
    dict(head=(63.3, 24.0), neck=(59.8, 30.8), lsh=(60.4, 34.8), rsh=(56.1, 32.8), lel=(57.1, 45.5), rel=(45.6, 39.9), lwr=(63.8, 47.0), rwr=(55.1, 47.4), pelvis=(55.7, 60.3), lhip=(55.8, 60.3), rhip=(55.6, 60.3), lkn=(50.0, 79.0), rkn=(66.7, 76.9), lan=(37.5, 94.7), ran=(63.5, 96.5)),
    dict(head=(63.1, 23.7), neck=(59.6, 30.6), lsh=(60.4, 34.6), rsh=(55.8, 32.5), lel=(57.2, 44.9), rel=(45.2, 39.5), lwr=(63.6, 46.6), rwr=(54.7, 47.0), pelvis=(55.7, 59.9), lhip=(56.0, 59.9), rhip=(55.4, 59.9), lkn=(49.9, 78.9), rkn=(65.7, 77.3), lan=(35.9, 93.4), ran=(66.7, 97.2)),
    dict(head=(63.2, 23.9), neck=(59.7, 30.7), lsh=(60.4, 34.7), rsh=(55.9, 32.5), lel=(57.1, 44.6), rel=(45.4, 39.5), lwr=(63.7, 46.3), rwr=(55.0, 46.9), pelvis=(55.7, 59.9), lhip=(55.9, 59.9), rhip=(55.4, 60.0), lkn=(51.0, 79.3), rkn=(65.0, 77.9), lan=(35.0, 92.1), ran=(68.1, 97.8)),
    dict(head=(63.3, 24.3), neck=(59.8, 31.1), lsh=(60.3, 35.1), rsh=(56.3, 32.9), lel=(56.8, 44.8), rel=(45.9, 40.0), lwr=(63.9, 46.7), rwr=(55.7, 47.3), pelvis=(55.7, 60.4), lhip=(55.7, 60.3), rhip=(55.7, 60.4), lkn=(52.8, 80.0), rkn=(64.6, 78.6), lan=(35.0, 91.1), ran=(67.2, 98.2)),
    dict(head=(63.3, 25.0), neck=(59.9, 31.7), lsh=(60.1, 35.7), rsh=(56.8, 33.5), lel=(56.3, 45.7), rel=(46.7, 41.0), lwr=(63.7, 48.4), rwr=(56.7, 48.2), pelvis=(55.7, 61.1), lhip=(55.5, 61.1), rhip=(55.9, 61.1), lkn=(54.9, 80.6), rkn=(64.2, 79.1), lan=(35.9, 90.2), ran=(64.5, 98.5)),
    dict(head=(63.3, 25.7), neck=(59.9, 32.4), lsh=(59.9, 36.3), rsh=(57.1, 34.4), lel=(55.8, 47.0), rel=(47.8, 42.4), lwr=(63.3, 51.4), rwr=(58.1, 49.7), pelvis=(55.7, 61.9), lhip=(55.5, 61.8), rhip=(55.9, 61.9), lkn=(57.0, 81.1), rkn=(63.6, 79.6), lan=(37.8, 89.6), ran=(60.7, 98.7)),
    dict(head=(63.3, 26.3), neck=(60.1, 33.0), lsh=(60.0, 36.7), rsh=(57.5, 35.1), lel=(55.3, 48.2), rel=(49.4, 44.0), lwr=(62.6, 54.5), rwr=(60.0, 51.0), pelvis=(55.7, 62.5), lhip=(55.9, 62.4), rhip=(55.5, 62.5), lkn=(59.3, 81.2), rkn=(62.4, 80.0), lan=(40.6, 89.5), ran=(56.6, 98.9)),
    dict(head=(63.4, 26.6), neck=(60.3, 33.2), lsh=(60.1, 36.8), rsh=(57.9, 35.4), lel=(54.2, 48.7), rel=(51.2, 45.3), lwr=(61.9, 56.3), rwr=(62.2, 51.8), pelvis=(55.7, 62.8), lhip=(56.5, 62.7), rhip=(54.9, 62.9), lkn=(61.9, 80.6), rkn=(60.5, 80.4), lan=(44.0, 90.1), ran=(52.7, 99.0)),
    dict(head=(63.5, 26.4), neck=(60.4, 33.1), lsh=(59.8, 36.4), rsh=(58.4, 35.3), lel=(52.3, 48.4), rel=(53.1, 45.9), lwr=(61.1, 55.9), rwr=(64.2, 51.4), pelvis=(55.7, 62.9), lhip=(57.0, 62.7), rhip=(54.4, 63.0), lkn=(64.3, 79.6), rkn=(57.9, 80.8), lan=(47.7, 91.2), ran=(48.8, 99.0)),
    dict(head=(63.4, 26.0), neck=(60.2, 32.5), lsh=(58.9, 35.8), rsh=(58.8, 34.8), lel=(49.7, 47.4), rel=(54.5, 45.8), lwr=(59.7, 53.8), rwr=(65.4, 50.0), pelvis=(55.7, 62.6), lhip=(57.4, 62.5), rhip=(54.0, 62.8), lkn=(66.0, 78.4), rkn=(54.8, 80.9), lan=(51.6, 92.8), ran=(44.7, 98.7)),
    dict(head=(63.1, 25.5), neck=(59.7, 31.9), lsh=(57.5, 35.0), rsh=(58.9, 34.2), lel=(47.0, 46.0), rel=(55.4, 45.1), lwr=(57.9, 51.4), rwr=(65.8, 48.3), pelvis=(55.7, 62.3), lhip=(57.5, 62.2), rhip=(53.8, 62.4), lkn=(66.5, 77.5), rkn=(51.9, 80.7), lan=(55.9, 94.2), ran=(40.8, 98.2)),
    dict(head=(62.7, 25.0), neck=(59.1, 31.3), lsh=(56.1, 34.5), rsh=(58.9, 33.6), lel=(45.0, 44.8), rel=(55.9, 44.2), lwr=(55.9, 49.9), rwr=(65.4, 46.8), pelvis=(55.7, 61.8), lhip=(57.5, 61.8), rhip=(53.9, 61.9), lkn=(66.0, 77.2), rkn=(49.4, 80.2), lan=(60.4, 95.3), ran=(37.4, 97.3)),
    dict(head=(62.3, 24.7), neck=(58.6, 31.0), lsh=(55.1, 34.3), rsh=(58.9, 33.2), lel=(44.0, 43.9), rel=(56.0, 43.4), lwr=(54.3, 49.4), rwr=(64.9, 45.8), pelvis=(55.7, 61.4), lhip=(57.4, 61.4), rhip=(54.0, 61.4), lkn=(65.1, 77.5), rkn=(48.0, 79.8), lan=(64.5, 96.0), ran=(34.9, 96.1)),
    dict(head=(62.0, 24.6), neck=(58.3, 31.0), lsh=(54.5, 34.5), rsh=(58.9, 33.1), lel=(43.8, 43.7), rel=(55.9, 42.8), lwr=(53.7, 49.2), rwr=(64.6, 45.4), pelvis=(55.7, 61.2), lhip=(57.3, 61.2), rhip=(54.1, 61.2), lkn=(64.4, 78.0), rkn=(47.9, 79.6), lan=(67.2, 96.4), ran=(33.1, 94.9)),
]
RUN_SECONDS = 0.62          # one cycle; the source took 0.67

# A relaxed, upright runner. The torso and the head are rotated back toward
# vertical, and each arm is rebuilt with fixed bone lengths: it swings from
# the shoulder (a wider arc than the source, which reads as running faster)
# with the elbow held in a soft, open bend instead of a tight one.
TORSO_TURN = -5.0           # degrees, about the pelvis (negative = more upright)
HEAD_TURN = -14.0           # degrees, about the neck
UPPER_ARM, FOREARM = 12.5, 11.5
ARM_CENTRE = -0.27          # radians; the arm swings about here, behind the hip
ARM_ARC = 1.4               # stretch of the shoulder swing about its average
ELBOW_BEND = (82, 100)      # interior elbow angle, in degrees, across the cycle


def _turn(p, about, deg):
    t, dx, dy = math.radians(deg), p[0] - about[0], p[1] - about[1]
    return (about[0] + dx * math.cos(t) - dy * math.sin(t),
            about[1] + dx * math.sin(t) + dy * math.cos(t))


def _relax(frames):
    out = []
    for f in frames:
        o = dict(f)
        for j in ("neck", "head", "lsh", "rsh", "lel", "rel", "lwr", "rwr"):
            o[j] = _turn(f[j], f["pelvis"], TORSO_TURN)
        o["head"] = _turn(o["head"], o["neck"], HEAD_TURN)
        out.append(o)
    for sh, el, wr in (("lsh", "lel", "lwr"), ("rsh", "rel", "rwr")):
        swing = [math.atan2(f[el][0] - f[sh][0], f[el][1] - f[sh][1]) for f in out]
        bend = []
        for f in out:
            v1 = (f[sh][0] - f[el][0], f[sh][1] - f[el][1])
            v2 = (f[wr][0] - f[el][0], f[wr][1] - f[el][1])
            bend.append(math.acos((v1[0] * v2[0] + v1[1] * v2[1])
                                  / math.hypot(*v1) / math.hypot(*v2)))
        mean = sum(swing) / len(swing)
        lo, hi = min(bend), max(bend)
        for o, th, bd in zip(out, swing, bend):
            th = ARM_CENTRE + (th - mean) * ARM_ARC
            k = (bd - lo) / (hi - lo)
            bd = math.radians(ELBOW_BEND[0] + k * (ELBOW_BEND[1] - ELBOW_BEND[0]))
            x, y = o[sh]
            ex, ey = x + UPPER_ARM * math.sin(th), y + UPPER_ARM * math.cos(th)
            # the forearm folds forward of the upper arm
            fa = th + math.pi - bd
            o[el] = (ex, ey)
            o[wr] = (ex + FOREARM * math.sin(fa), ey + FOREARM * math.cos(fa))
    return out


RUN_FRAMES = _relax(RUN_FRAMES)

# Scenery: a ground line that fades out at both ends, and a soft shadow.
GROUND = (
    '<defs><linearGradient id="wait-ground" x1="0" x2="1">'
    '<stop offset="0" stop-color="#94a3b8" stop-opacity="0"/>'
    '<stop offset="0.2" stop-color="#94a3b8" stop-opacity="0.9"/>'
    '<stop offset="0.8" stop-color="#94a3b8" stop-opacity="0.9"/>'
    '<stop offset="1" stop-color="#94a3b8" stop-opacity="0"/>'
    '</linearGradient></defs>'
    '<ellipse cx="50" cy="100.5" rx="17" ry="2.2" fill="#94a3b8" opacity="0.25"/>'
    '<line x1="4" y1="100" x2="96" y2="100" stroke="url(#wait-ground)" '
    'stroke-width="1.6" stroke-linecap="round"/>'
)


def _values(frames, key: str, axis: int) -> str:
    # The first frame again at the end, so the loop closes on itself.
    return ";".join(f"{f[key][axis]:g}" for f in (*frames, frames[0]))


def figure_svg() -> str:
    """The runner, looping forever with SMIL animation.

    Every demo shares it: it is the one that reads as a person moving at a
    glance, and a real gait cycle does that better than anything posed by hand.
    """
    frames, dur = RUN_FRAMES, RUN_SECONDS
    anim = (lambda attr, vals:
            f'<animate attributeName="{attr}" values="{vals}" dur="{dur}s" '
            f'repeatCount="indefinite"/>')
    parts = [GROUND]
    for a, b, side in BONES:
        parts.append(
            f'<line stroke="{COLOURS[side]}" stroke-width="3.2" stroke-linecap="round">'
            + anim("x1", _values(frames, a, 0)) + anim("y1", _values(frames, a, 1))
            + anim("x2", _values(frames, b, 0)) + anim("y2", _values(frames, b, 1))
            + "</line>")
    for joint in ("lsh", "rsh", "lel", "rel", "lwr", "rwr",
                  "lhip", "rhip", "lkn", "rkn", "lan", "ran"):
        parts.append('<circle r="2.3" style="fill: var(--body-text-color, #334155)">'
                     + anim("cx", _values(frames, joint, 0))
                     + anim("cy", _values(frames, joint, 1)) + "</circle>")
    parts.append(f'<circle r="6" fill="none" stroke="{COLOURS["mid"]}" '
                 f'stroke-width="3">' + anim("cx", _values(frames, "head", 0))
                 + anim("cy", _values(frames, "head", 1)) + "</circle>")
    return (f'<svg viewBox="0 8 100 98" width="120" height="118" '
            f'aria-hidden="true">{"".join(parts)}</svg>')


def wait_html(caption: str) -> str:
    """The figure, a clock the browser keeps (see WAIT_JS), and a caption."""
    return (f"<div class='wait'>{figure_svg()}"
            f"<div class='wait-clock'>0s</div>"
            f"<div class='wait-caption'>{html.escape(caption)}</div></div>")


# The clock runs in the browser: the server only speaks when the log does, and
# a gateway call can be silent for a minute. Each new figure restarts it.
WAIT_JS = """
setInterval(() => {
    const el = element.querySelector('.wait-clock');
    if (!el) return;
    if (!el.dataset.t0) el.dataset.t0 = Date.now();
    const s = Math.floor((Date.now() - el.dataset.t0) / 1000);
    el.textContent = s < 60 ? s + 's' : Math.floor(s / 60) + 'm ' + (s % 60) + 's';
}, 250);
"""


# ── The stages ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Stage:
    now: str                 # while it runs: "Waiting for ViTPose"
    done: str                # once past it: "ViTPose"
    keys: tuple[str, ...]    # matched against the pipeline's step headings
    model: bool = False      # a gateway call, so "cached" means something


PREPARE = Stage("Preparing the clip", "Clip prepared",
                ("api key", "source video", "convert"))
VITPOSE = Stage("Waiting for ViTPose", "ViTPose", ("pose estimation", "vitpose"), True)
RENDER = Stage("Rendering", "Rendered", ("render", "artifacts"))

POSE_STAGES = (PREPARE, VITPOSE, RENDER)
RUNNING_STAGES = (PREPARE, VITPOSE,
                  Stage("Measuring the stride", "Stride measured", ("gait",)), RENDER)
SAM = Stage("Waiting for SAM 3.1", "SAM 3.1", ("sam 3.1",), True)
REPS = Stage("Counting reps", "Reps counted", ("reps",))
PULL_UP_STAGES = (PREPARE, VITPOSE, REPS, RENDER)
DEADLIFT_STAGES = (PREPARE, VITPOSE, SAM, REPS,
                   Stage("Waiting for System One", "System One",
                         ("system one", "back position"), True),
                   RENDER)
BASKETBALL_STAGES = (PREPARE, VITPOSE, SAM,
                     Stage("Calling makes and misses", "Shots called", ("make or miss",)),
                     RENDER)
CLIMBING_STAGES = (
    Stage("Preparing the clips", "Clips prepared",
          ("api key", "source video", "convert", "camera", "wall")),
    SAM,
    VITPOSE,
    Stage("Reading the climb", "Climb read", ("the climb", "same route")),
    RENDER,
)

# "4 Pose estimation ─────", after rich has drawn it into the log.
_HEADING = re.compile(r"^\s*(\d+)\s+(.+?)\s*─{3,}\s*$")
_ATTEMPT = re.compile(r"^\s*attempt\s+(\d+)/(\d+)\b")


@dataclass
class Progress:
    """Follows a run through *stages* from its log, and draws where it is."""

    stages: tuple[Stage, ...]
    current: int = -1
    since: float = field(default_factory=time.perf_counter)
    took: dict = field(default_factory=dict)       # stage index -> seconds
    cached: set = field(default_factory=set)       # model stages that skipped the call
    attempt: tuple[int, int] | None = None

    def feed(self, line: str) -> bool:
        """Take one log line. True when what is drawn should change."""
        if m := _ATTEMPT.match(line):
            self.attempt = (int(m[1]), int(m[2]))
            return True
        if m := _HEADING.match(line):
            title = m[2].lower()
            for i, stage in enumerate(self.stages):
                if any(k in title for k in stage.keys):
                    if i != self.current:
                        self._enter(i)
                        return True
                    return False
            return False
        if ("cache hit" in line and 0 <= self.current < len(self.stages)
                and self.stages[self.current].model and self.current not in self.cached):
            self.cached.add(self.current)
            return True
        return False

    def _enter(self, i: int) -> None:
        now = time.perf_counter()
        if self.current >= 0:
            self.took[self.current] = self.took.get(self.current, 0) + now - self.since
        self.current, self.since = i, now

    def html(self, state: str = "running", message: str = "",
             elapsed: float | None = None) -> str:
        """The checklist. *state* is running, done or failed."""
        if state == "done":
            self._enter(len(self.stages))            # close the last stage
        rows = []
        for i, stage in enumerate(self.stages):
            if state == "done" or i < self.current:
                note = ("cached, no model call" if i in self.cached else
                        f"{self.took[i]:.1f}s" if self.took.get(i, 0) >= 0.1 else "")
                rows.append(f"<li class='done'>{_TICK}<span>{stage.done}</span>"
                            f"<em>{note}</em></li>")
            elif i == self.current and state == "failed":
                rows.append(f"<li class='failed'>{_CROSS}<span>{stage.now}</span></li>")
            elif i == self.current:
                rows.append(f"<li class='now'>{_SPIN}<span>{stage.now}"
                            f"</span><em>{'cached' if i in self.cached else ''}</em></li>")
            else:
                rows.append(f"<li class='todo'><i class='dot'></i>"
                            f"<span>{stage.now}</span></li>")
        head = {
            "running": "Working on it"
                       + (f" &middot; attempt {self.attempt[0]} of {self.attempt[1]}"
                          if self.attempt and self.attempt[1] > 1 else ""),
            "done": f"{_TICK} {html.escape(message)}",
            "failed": f"{_CROSS} {html.escape(message)}",
        }[state]
        stamp = f" data-elapsed='{elapsed:.0f}'" if elapsed is not None else ""
        return (f"<div class='stages' data-state='{state}'{stamp}>"
                f"<p class='stages-head'>{head}</p><ul>{''.join(rows)}</ul></div>")


def notice_html(message: str, kind: str = "idle") -> str:
    """A one-line status with no stages: before a run, or when one never started."""
    safe_kind = kind if kind in {"idle", "running", "done", "failed"} else "idle"
    return (f"<div class='stages' data-state='{safe_kind}'>"
            f"<p class='stages-head'>{html.escape(message)}</p></div>")


_TICK = ('<svg viewBox="0 0 24 24" width="15" height="15" fill="none" '
         'stroke="currentColor" stroke-width="3" stroke-linecap="round" '
         'stroke-linejoin="round" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>')
# An arc, turned by CSS. SVG rather than a bordered box: Gradio's base styles
# zero the border on it.
_SPIN = ('<svg class="spin" viewBox="0 0 24 24" width="15" height="15" '
         'aria-hidden="true"><circle cx="12" cy="12" r="9" fill="none" '
         'stroke="currentColor" stroke-opacity="0.2" stroke-width="3.2"/>'
         '<path d="M12 3a9 9 0 0 1 9 9" fill="none" stroke="currentColor" '
         'stroke-width="3.2" stroke-linecap="round"/></svg>')
_CROSS = ('<svg viewBox="0 0 24 24" width="15" height="15" fill="none" '
          'stroke="currentColor" stroke-width="3" stroke-linecap="round" '
          'aria-hidden="true"><path d="M6.5 6.5l11 11M17.5 6.5l-11 11"/></svg>')

CSS = """
/* Gradio puts the class on its block and again on the inner prose div, so
   the card is drawn on the block alone and the content centred inside it. */
.block.wait-slot { border: 1px solid var(--border-color-primary);
                   border-radius: var(--radius-lg); background: var(--block-background-fill); }
.block.wait-slot .html-container { height: 100%; display: flex;
                                   align-items: center; justify-content: center; }
.prose.wait-slot { border: none !important; background: none !important; }
.wait { display: flex; flex-direction: column; align-items: center; padding: 0.2rem 0.4rem; }
.wait svg { width: 200px; height: 196px; }
.wait svg { display: block; }
.wait-clock { font-size: 1.1rem; font-weight: 700; font-variant-numeric: tabular-nums;
              margin-top: 0.2rem; }
.wait-caption { font-size: 0.75rem; opacity: 0.6; text-align: center; max-width: 9.5rem;
                margin-top: 0.15rem; }
.stages .stages-head { margin: 0.2rem 0 0.5rem 0; font-size: 1.05rem; font-weight: 650;
                       display: flex; align-items: center; gap: 0.4rem; }
.stages[data-state=done] .stages-head { color: #15803d; }
.stages[data-state=failed] .stages-head { color: #b91c1c; }
.stages[data-state=idle] .stages-head { font-weight: 400; opacity: 0.75; font-size: 0.95rem; }
.stages ul { list-style: none; margin: 0; padding: 0; }
.stages li { display: flex; align-items: center; gap: 0.55rem; padding: 0.22rem 0;
             font-size: 0.95rem; }
.stages li em { font-style: normal; font-size: 0.8rem; opacity: 0.6; margin-left: auto;
                font-variant-numeric: tabular-nums; }
.stages li.done { color: #15803d; }
.stages li.done span { color: var(--body-text-color); }
.stages li.failed { color: #b91c1c; font-weight: 650; }
.stages li.now { font-weight: 650; }
.stages li.todo { opacity: 0.45; }
.stages .dot { width: 9px; height: 9px; border-radius: 50%; margin: 0 3px;
               border: 2px solid currentColor; box-sizing: border-box; }
.stages .spin { flex: none; color: var(--button-primary-background-fill, #6366f1);
                animation: stage-spin 0.8s linear infinite; }
@keyframes stage-spin { to { transform: rotate(360deg); } }
"""
