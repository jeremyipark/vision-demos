"""Reading the climb: which limb used which hold, in what order, and for how long.

A hold is *used* when a hand or a foot stays inside it. Dwell rather than a
single frame, because a hand passing over a hold on the way to another one
touches it for two or three frames and has not used it. The route is over when
both wrists are on the top hold — matching a gym's own rule for topping out.

Contact is tracked per *limb*, not per hold, which is what makes the utilization
split possible: the same hold can be a handhold early and a foothold later, and
those are different events by the same body.

Four corrections do most of the work here:

  * **the mask, not the box.** A box around a hold is mostly wall; the mask is
    the hold. With the box, a hand resting beside a hold reads as on it.
  * **the ankle is not the foot.** COCO-17 stops at the ankle joint and the
    contact is at the toes, a few percent of frame height below it.
  * **hysteresis.** Getting onto a hold and staying on it are different
    thresholds. A keypoint that has settled on a hold still wanders a few pixels
    a frame, and with one threshold that wander reads as letting go and
    re-gripping several times a second. So contact is *entered* on a tight
    margin and *held* on a loose one — the grip has to clearly break, not just
    graze the boundary.
  * **one hold per limb.** A hand is on one hold at a time. Where masks abut,
    the deepest containment wins, so a limb cannot bank time on two holds at once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

ANKLE_KPTS = frozenset({15, 16})
WRIST_KPTS = frozenset({9, 10})


@dataclass(frozen=True)
class Limb:
    """One of the four points of contact."""

    key: str
    kpt: int
    label: str      # for the panel
    name: str       # for prose


# Order is the panel's reading order: hands above feet, left before right.
LIMBS: tuple[Limb, ...] = (
    Limb("left_hand", 9, "LH", "left hand"),
    Limb("right_hand", 10, "RH", "right hand"),
    Limb("left_foot", 15, "LF", "left foot"),
    Limb("right_foot", 16, "RF", "right foot"),
)
LIMB_BY_KEY = {limb.key: limb for limb in LIMBS}
WRIST_KEYS = tuple(l.key for l in LIMBS if l.kpt in WRIST_KPTS)
ANKLE_KEYS = tuple(l.key for l in LIMBS if l.kpt in ANKLE_KPTS)


@dataclass
class Contact:
    """One limb's uninterrupted use of one hold."""

    limb: str
    hold_id: int
    start: int      # frame the dwell was satisfied on, backdated to first touch
    end: int        # last frame still in contact

    def frames(self) -> int:
        return self.end - self.start + 1


@dataclass
class ClimbAnalysis:
    """What the climb did, in frame indices."""

    activated_at: dict[int, int]            # hold id -> frame it was first used
    order: dict[int, int]                   # hold id -> 1-based activation rank
    midline: dict[int, tuple[float, float]] # frame -> torso centroid, normalized
    contacts: list[Contact]
    holding: dict[int, dict[str, int]]      # frame -> {limb key: hold id} in use now
    start_frame: int | None
    completion_frame: int | None
    final_hold_id: int | None
    fps: float
    holds: list[dict] = field(default_factory=list)
    window: tuple[int, int] | None = None   # the climb; set by restrict()
    start_rule: str = "holds"               # "floor" when a floor was available

    @property
    def elapsed(self) -> float | None:
        """Wall-clock seconds from both feet on the route to both hands on top."""
        if self.start_frame is None or self.completion_frame is None:
            return None
        return (self.completion_frame - self.start_frame) / self.fps

    @property
    def n_used(self) -> int:
        return len(self.activated_at)

    def summary(self) -> dict:
        return {
            "holds_detected": len(self.holds),
            "holds_used": self.n_used,
            "start_frame": self.start_frame,
            "completion_frame": self.completion_frame,
            "final_hold_id": self.final_hold_id,
            "elapsed_seconds": round(self.elapsed, 3) if self.elapsed is not None else None,
            "topped_out": self.completion_frame is not None,
            "start_rule": self.start_rule,
            "order": [hid for hid, _ in sorted(self.order.items(), key=lambda kv: kv[1])],
        }


class HoldGeometry:
    """Point-in-hold tests, with the mask where there is one and the box otherwise.

    ``depth`` is a signed distance: positive inside the hold, negative outside,
    in units of **frame height**. Returning a distance rather than a bool is what
    lets the caller run two thresholds on it (enter tight, hold loose) and pick
    between abutting holds by which one the point is furthest inside.

    Coordinates arrive normalized independently in x and y, which means one
    normalized unit is a different number of pixels on each axis. Testing in that
    space makes every margin an ellipse rather than a circle — on this portrait
    clip, 1.78x more generous vertically than horizontally, so a hand drifting
    sideways off a hold releases while the same drift downward does not. Scaling
    x by the aspect ratio puts both axes in the same physical unit, so a margin
    means one distance in every direction.
    """

    def __init__(self, holds: list[dict], *, bbox_margin: float, aspect: float = 1.0):
        self.holds = holds
        self.by_id = {h["id"]: h for h in holds}
        self.bbox_margin = bbox_margin
        self.aspect = aspect
        self._contours = {}
        for hold in holds:
            polygon = hold.get("polygon")
            if polygon and len(polygon) >= 3:
                arr = np.asarray(polygon, dtype=np.float32).copy()
                arr[:, 0] *= aspect
                self._contours[hold["id"]] = arr.reshape(-1, 1, 2)

    def depth(self, hold: dict, x: float, y: float) -> float:
        contour = self._contours.get(hold["id"])
        if contour is not None:
            return float(cv2.pointPolygonTest(
                contour, (float(x) * self.aspect, float(y)), True))
        # No usable mask: fall back to the box, expressed as the same signed
        # distance so both paths answer to one threshold.
        bx, by, bw, bh = hold["bbox"]
        dx = min(x - bx, bx + bw - x) * self.aspect
        dy = min(y - by, by + bh - y)
        return float(min(dx, dy) + self.bbox_margin)

    def best(self, x: float, y: float, margin: float):
        """The hold this point is most inside, or None. Ties broken by depth."""
        best_hold, best_depth = None, -margin
        for hold in self.holds:
            d = self.depth(hold, x, y)
            if d >= -margin and d > best_depth:
                best_hold, best_depth = hold, d
        return best_hold


def limb_point(poses, frame: int, limb: Limb, *, cfg, toe_offset: float | None = None):
    """A limb's position for the inclusion test, or None when not visible.

    *toe_offset* is how far below the ankle joint to put the contact point, in
    canvas units — :func:`scales` derives it from the climber's own height, so
    it is a foot rather than whatever fraction of the canvas the camera happened
    to leave. Omitted, the raw config number is used, which is only right if the
    caller has already scaled it.
    """
    if not poses.valid[frame][limb.kpt]:
        return None
    x, y = poses.kpts[frame][limb.kpt]
    if cfg.ANKLE_LENIENCY and limb.kpt in ANKLE_KPTS:
        y = y + (cfg.ANKLE_TO_TOE_OFFSET if toe_offset is None else toe_offset)
    return float(x), float(y)


def scales(holds: list[dict], poses) -> tuple[float, float]:
    """``(hold_unit, body_unit)`` — the two lengths every threshold is quoted in.

    The thresholds below used to be fractions of the frame, which worked while
    the frame was the wall: one tripod shot, the boulder filling it, so "1.8% of
    the frame" was a fixed distance on the rock. Neither half of that survives
    here. The canvas is sized by wherever the camera wandered, so the boulder
    occupies whatever fraction of it that turned out to be — on this clip about
    half, with ceiling above and mats below. The identical config number is
    therefore 0.58 hold-heights on the tripod clip and 0.92 here: over half as
    generous again, which is how a hand ends up "on" a hold it is a hold's width
    away from.

    So the thresholds stop being fractions of anything the camera decides, and
    become multiples of two things the scene decides:

    ``hold_unit``  the median hold's height. Contact margins scale with it,
                   because whether a hand is on a hold is a question about the
                   hold's own size — a fingertip off a crimp is off it, and a
                   fingertip off a volume is still on it.

    ``body_unit``  the climber's median bounding-box height. Body clearances
                   scale with it, because how far a toe hangs below an ankle and
                   how far off the mat counts as "off the mat" are questions
                   about the person, not about the wall.

    Both are measured from this clip, so the numbers in `config.py` mean the
    same thing on the next one however it was framed.
    """
    hold_unit = float(np.median([h["bbox"][3] for h in holds])) if holds else 0.02
    boxes = [poses.bboxes[f][3] for f in poses.frames() if f in poses.bboxes]
    body_unit = float(np.median(boxes)) if boxes else 0.25
    return max(hold_unit, 1e-4), max(body_unit, 1e-4)


def analyze(holds: list[dict], poses, *, fps: float, aspect: float, cfg,
            floor=None) -> ClimbAnalysis:
    """Walk the clip once, in order, and record what each limb held."""
    frames = poses.frames()
    if not holds or not frames:
        return ClimbAnalysis({}, {}, {}, [], {}, None, None, None, fps, holds)

    hold_unit, body_unit = scales(holds, poses)
    geometry = HoldGeometry(holds, bbox_margin=cfg.HOLD_BBOX_MARGIN * hold_unit,
                            aspect=aspect)
    dwell_frames = max(1, int(round(cfg.HOLD_DWELL_SECONDS * fps)))
    final_frames = max(1, int(round(cfg.FINAL_HOLD_DWELL_SECONDS * fps)))
    # The top hold is often tapped rather than settled onto, so it confirms on a
    # shorter dwell than the rest. The route still is not complete until both
    # hands hold it together for FINAL_HOLD_DWELL_SECONDS, which is the real gate.
    final_touch_frames = max(1, int(round(cfg.FINAL_HOLD_TOUCH_SECONDS * fps)))
    # A contact that never gets within `graze_depth` of the hold's outline has to
    # last `graze_frames` rather than the ordinary dwell. See HOLD_GRAZE_DEPTH.
    graze_depth = cfg.HOLD_GRAZE_DEPTH * hold_unit
    graze_frames = max(1, int(round(cfg.HOLD_GRAZE_DWELL_SECONDS * fps)))
    enter = cfg.HOLD_MASK_MARGIN * hold_unit
    release = cfg.HOLD_RELEASE_MARGIN * hold_unit
    clearance = cfg.FLOOR_CLEARANCE * body_unit
    toe_offset = cfg.ANKLE_TO_TOE_OFFSET * body_unit
    start_frames = max(1, int(round(cfg.START_DWELL_SECONDS * fps)))

    final_hold = min(holds, key=lambda h: h["bbox"][1] + h["bbox"][3] / 2)

    activated_at: dict[int, int] = {}
    contacts: list[Contact] = []
    holding: dict[int, dict[str, int]] = {}

    # Per-limb contact state: which hold, since when, and whether the dwell has
    # been satisfied yet. `since` backdates a confirmed contact to the first
    # frame the limb touched, so the dwell requirement filters out brushes
    # without also discounting the half-second it takes to earn one.
    on: dict[str, int | None] = {limb.key: None for limb in LIMBS}
    since: dict[str, int] = {limb.key: 0 for limb in LIMBS}
    held: dict[str, int] = {limb.key: 0 for limb in LIMBS}     # consecutive frames
    confirmed: dict[str, bool] = {limb.key: False for limb in LIMBS}
    last_seen: dict[str, int] = {limb.key: 0 for limb in LIMBS}
    # Deepest this contact ever reached, to tell a touch from a pass-by.
    peak: dict[str, float] = {limb.key: -math.inf for limb in LIMBS}

    def close(limb_key: str):
        """Bank the current contact if it ever cleared the dwell threshold."""
        if on[limb_key] is not None and confirmed[limb_key]:
            contacts.append(Contact(limb_key, on[limb_key], since[limb_key],
                                    last_seen[limb_key]))
        on[limb_key] = None
        held[limb_key] = 0
        confirmed[limb_key] = False
        peak[limb_key] = -math.inf

    final_dwell = 0
    completion_frame = None
    start_dwell = 0
    start_frame = None

    midline: dict[int, tuple[float, float]] = {}

    for frame in frames:
        kpts, valid = poses.kpts[frame], poses.valid[frame]

        torso = [k for k in cfg.TORSO_KP_INDICES if valid[k]]
        if torso:
            midline[frame] = (float(np.mean([kpts[k][0] for k in torso])),
                              float(np.mean([kpts[k][1] for k in torso])))

        now: dict[str, int] = {}
        for limb in LIMBS:
            key = limb.key
            point = limb_point(poses, frame, limb, cfg=cfg, toe_offset=toe_offset)

            if point is None:
                close(key)
                continue

            # Hysteresis: if this limb already has a hold, it keeps it as long as
            # it is within the *release* margin — a looser boundary than the one
            # it had to cross to get there. Only once it is clearly off does the
            # limb look for a new hold, on the tight margin.
            #
            # With one exception, and it matters: hysteresis exists to stop a
            # limb letting go of a hold for *nothing*, not to stop it moving to a
            # better one. Two long rails on this wall are set parallel a hand's
            # width apart — closer than the release margin — so a hand that
            # matched the lower one first stayed matched to it after it had
            # visibly transferred to the upper one, and the upper rail never lit
            # up despite being held for two seconds. A hold the limb is *further
            # inside* than its current one is not a candidate to be resisted, it
            # is the answer.
            current = geometry.by_id.get(on[key]) if on[key] is not None else None
            depth = geometry.depth(current, *point) if current is not None else None
            if depth is not None and depth >= -release:
                better = geometry.best(*point, margin=enter)
                chosen = better if (better is not None
                                    and better["id"] != current["id"]
                                    and geometry.depth(better, *point) > depth) else current
                if chosen is not current:
                    depth = None
            else:
                chosen = geometry.best(*point, margin=enter)
                depth = None

            if chosen is None:
                close(key)
                continue

            if on[key] != chosen["id"]:
                close(key)
                on[key] = chosen["id"]
                since[key] = frame

            held[key] += 1
            last_seen[key] = frame
            if depth is None:
                depth = geometry.depth(chosen, *point)
            peak[key] = max(peak[key], depth)

            need = (final_touch_frames if chosen["id"] == final_hold["id"]
                    else dwell_frames)
            # A limb that never came within `graze_depth` of the outline was
            # passing the hold, not using it, and has to stay far longer to
            # count. Both conditions, so a real contact still confirms on the
            # ordinary dwell and a graze is not rejected outright — see
            # HOLD_GRAZE_DEPTH for why depth alone cannot separate the two.
            close_enough = peak[key] >= -graze_depth or held[key] >= graze_frames
            if not confirmed[key] and held[key] >= need and close_enough:
                confirmed[key] = True
                activated_at.setdefault(chosen["id"], frame)
            if confirmed[key]:
                now[key] = chosen["id"]

        holding[frame] = now

        if completion_frame is None:
            # Topping out: both wrists in *confirmed contact* with the final
            # hold at once — the same state that lights the hold on the panel,
            # not a second opinion about it.
            #
            # This used to be its own raw distance test, run against the release
            # margin and independent of the contact tracker, which made it
            # strictly more permissive. On this clip it fired one frame before
            # the tracker acquired the hold for the right hand, so the route was
            # marked complete while the right hand's contact with the top hold
            # started *after* the window closed — the hold went green for a hand
            # that, by the panel's own list, had never touched it.
            #
            # Deriving it from the contacts makes the two agree by construction:
            # nothing can top out without both hands holding the top hold.
            both = all(now.get(key) == final_hold["id"] for key in WRIST_KEYS)
            if both:
                # Backdated to when the two hands were both *touching* the hold,
                # not to when the second one finished confirming — the same
                # correction `since` makes for ordinary contacts, and for the
                # same reason. A hand needs FINAL_HOLD_TOUCH_SECONDS on the top
                # hold before the tracker will say it is on it, and those frames
                # were spent on the hold. Counting only from the moment both are
                # confirmed charges the match for the second hand's confirmation
                # twice, so a rule that reads "both hands for 0.8s" really asks
                # for 1.05s, and a genuine match that was held for a second is
                # rejected with no way to see why.
                #
                # The running counter is kept alongside it and the larger wins,
                # so FINAL_DWELL_DECAY still governs jitter across a broken grip
                # while a clean match is measured as the overlap it actually is.
                overlap = frame - max(since[key] for key in WRIST_KEYS) + 1
                final_dwell = max(final_dwell + 1, overlap)
                if final_dwell >= final_frames:
                    completion_frame = frame
            else:
                final_dwell = max(0, final_dwell - cfg.FINAL_DWELL_DECAY)

        # Starting the clock. The gym's rule is both feet off the ground, which
        # is not the same as both feet on holds: a foot smeared flat against the
        # wall, on no hold at all, is a legitimate placement and the climb has
        # begun. Measuring against the floor covers both; measuring against the
        # holds misses the smear and starts the clock late, or never.
        #
        # Both feet at *once*, and sustained. Latching each foot the first time
        # it ever came up and waiting for the pair answers a different question —
        # "was each foot ever off the ground" — which a climber walking to the
        # wall satisfies one step at a time.
        #
        # And a hand on a hold, because "off the floor" alone is a claim about
        # two keypoints and nothing else. A climber entering frame is half out of
        # it, where the pose is at its worst: on this clip ViTPose puts both
        # ankles a sixth of a frame above the mat while they are plainly walking
        # on it. Being on the wall is not provable from the feet alone, and
        # nobody gets off the ground on a boulder without holding something.
        #
        # Without a floor to measure against, fall back to the feet themselves
        # being on holds — the only other thing in frame that says "not standing".
        if start_frame is None:
            if floor is not None:
                feet_up = True
                for limb in LIMBS:
                    if limb.kpt not in ANKLE_KPTS:
                        continue
                    point = limb_point(poses, frame, limb, cfg=cfg,
                                       toe_offset=toe_offset)
                    # A foot on a hold is off the ground, whatever the floor line
                    # says about it. Two independent pieces of evidence rather
                    # than one, which matters on a low start: the toe and the mat
                    # are then a couple of centimetres apart and the floor test
                    # is deciding the climb on pose noise.
                    if point is None:
                        feet_up = False
                        break
                    if on[limb.key] is None and not floor.is_clear(*point, clearance,
                                                                  frame=frame):
                        feet_up = False
                        break
                gripping = any(now.get(limb.key) is not None for limb in LIMBS)
                started = feet_up and gripping
            else:
                started = all(on[limb.key] is not None
                              for limb in LIMBS if limb.kpt in ANKLE_KPTS)

            # Leaky, not a reset. A single frame where ViTPose drops an ankle
            # into the mat should cost the counter one frame, not all of them:
            # resetting means the clock needs START_DWELL_SECONDS of *flawless*
            # pose, and on a climber hanging a hand's width off the ground it
            # never gets it. The clock started seven seconds late on this clip
            # for exactly that reason.
            start_dwell = (start_dwell + 1 if started
                           else max(0, start_dwell - cfg.START_DWELL_DECAY))
            if start_dwell >= start_frames:
                # Backdate to the first frame of the run, not the frame the dwell
                # matured on: the climber left the ground half a second ago.
                start_frame = frame - start_dwell + 1

    for limb in LIMBS:
        close(limb.key)   # the clip ending is not the same as letting go

    order = {hid: rank for rank, (hid, _) in
             enumerate(sorted(activated_at.items(), key=lambda kv: kv[1]), start=1)}

    return ClimbAnalysis(
        activated_at=activated_at, order=order, midline=midline, contacts=contacts,
        holding=holding, start_frame=start_frame, completion_frame=completion_frame,
        final_hold_id=final_hold["id"], fps=fps, holds=holds,
        start_rule="floor" if floor is not None else "holds")


def renumber(analysis: ClimbAnalysis, mapping: dict[int, int]) -> None:
    """Rewrite every hold id in *analysis*, in place, through *mapping*.

    Called once the comparison knows how this clip's wall lines up with the
    rest, so that the ids drawn on the video are the ids the card compares. The
    hold dicts are mutated rather than copied, which also renumbers the caller's
    own `route` — it is the same list, and one wall with two numberings is the
    bug this exists to remove.

    Everything derived from the analysis has to be rebuilt afterwards:
    `Utilization` and `hold_times` read these ids and cache their own copies.
    """
    missing = [h["id"] for h in analysis.holds if h["id"] not in mapping]
    if missing:
        raise KeyError(f"no number for hold(s) {missing}; the mapping must be total")

    for hold in analysis.holds:
        hold["id"] = mapping[hold["id"]]
    analysis.activated_at = {mapping[h]: f for h, f in analysis.activated_at.items()}
    analysis.order = {mapping[h]: rank for h, rank in analysis.order.items()}
    analysis.holding = {frame: {limb: mapping[h] for limb, h in now.items()}
                        for frame, now in analysis.holding.items()}
    for contact in analysis.contacts:
        contact.hold_id = mapping[contact.hold_id]
    if analysis.final_hold_id is not None:
        analysis.final_hold_id = mapping[analysis.final_hold_id]


def check(analysis: ClimbAnalysis, util: "Utilization") -> list[str]:
    """Consistency between what the panel draws and what the tables say.

    These are invariants, not thresholds: each one is a statement the two halves
    of the output would be contradicting each other about if it failed. Returned
    as messages rather than raised, so a questionable clip still produces a
    render you can look at to see *why*.
    """
    problems = []
    final = analysis.final_hold_id

    if analysis.completion_frame is not None:
        # Topping out means both hands held the top hold. If the hold is green
        # but a hand's list does not contain it, one of the two is lying.
        for key in WRIST_KEYS:
            if final not in util.holds_by_limb[key]:
                problems.append(
                    f"{LIMB_BY_KEY[key].name} is missing hold {final} from its list, "
                    f"but the route is marked topped out — which requires both "
                    f"hands on it")
        if final not in analysis.activated_at:
            problems.append(f"hold {final} is drawn complete but never activated")

    for key, holds in util.holds_by_limb.items():
        unknown = [h for h in holds if h not in analysis.activated_at]
        if unknown:
            problems.append(f"{LIMB_BY_KEY[key].name} lists hold(s) "
                            f"{', '.join(map(str, unknown))} that never lit up")

    if analysis.start_frame is not None and analysis.completion_frame is not None \
            and analysis.completion_frame < analysis.start_frame:
        problems.append("the route completed before it started")

    return problems


def restrict(analysis: ClimbAnalysis, window: tuple[int, int]) -> int:
    """Re-derive activation and order from the contacts inside *window*.

    One window has to govern everything the panel shows, or it contradicts
    itself: a climber pulls on and steps off a few times before committing, and
    counting those touches in the pips while excluding them from the percentages
    puts a hand on a hold the table says was never used by a hand.

    The contacts themselves are left whole — the report still knows about the
    warm-up and says how many holds it touched. Returns that count.
    """
    lo, hi = window
    activated: dict[int, int] = {}
    for contact in analysis.contacts:
        start = max(contact.start, lo)
        if start > min(contact.end, hi):
            continue
        if contact.hold_id not in activated or start < activated[contact.hold_id]:
            activated[contact.hold_id] = start

    analysis.activated_at = activated
    analysis.order = {hid: rank for rank, (hid, _) in
                      enumerate(sorted(activated.items(), key=lambda kv: kv[1]), start=1)}
    analysis.window = window
    return len({c.hold_id for c in analysis.contacts if c.end < lo})


# ── utilization ──────────────────────────────────────────────────────────────

class Utilization:
    """How the climb's contact time divides between the four limbs and the holds.

    **coverage** is the headline: this limb's contact time over the climb's own
    duration. It does not add to 100% across the four limbs and cannot — limbs
    hold simultaneously — and that is the point. It answers "how much of the
    climb was this limb actually on something", which stands on its own and can
    be read against another climber's number for the same limb.

    **share** — this limb over the total contact time summed across all four —
    is also kept, in the JSON. It is a real decomposition and it adds to 100%,
    but every limb's figure moves when any other limb's does, so two climbers'
    shares are not comparable unless their totals happen to match.

    **sequence** is the beta: the holds each limb used, in the order it used
    them, revisits included. That is the thing that actually differs between two
    people on one route.
    """

    def __init__(self, analysis: ClimbAnalysis, *, window: tuple[int, int] | None = None):
        self.fps = analysis.fps
        self.window = window
        lo, hi = window if window else (-10**9, 10**9)

        self.per_limb: dict[str, int] = {limb.key: 0 for limb in LIMBS}
        self.per_pair: dict[tuple[str, int], int] = {}
        self.holds_by_limb: dict[str, list[int]] = {limb.key: [] for limb in LIMBS}
        self.sequence_by_limb: dict[str, list[int]] = {limb.key: [] for limb in LIMBS}

        # Contacts close in chronological order within a limb, so appending as
        # they come gives the order the holds were used in.
        for contact in sorted(analysis.contacts, key=lambda c: c.start):
            start, end = max(contact.start, lo), min(contact.end, hi)
            if end < start:
                continue
            frames = end - start + 1
            self.per_limb[contact.limb] += frames
            pair = (contact.limb, contact.hold_id)
            self.per_pair[pair] = self.per_pair.get(pair, 0) + frames
            if contact.hold_id not in self.holds_by_limb[contact.limb]:
                self.holds_by_limb[contact.limb].append(contact.hold_id)
            self.sequence_by_limb[contact.limb].append(contact.hold_id)

        self.total_frames = sum(self.per_limb.values())
        self.journey_frames = (hi - lo + 1) if window else 0

    def sequence(self, limb_key: str) -> list[int]:
        """The holds this limb used, in order. The beta.

        Consecutive repeats are collapsed: a foot that comes off hold 10 and
        gets straight back on it has repositioned, not moved, and "10 10"
        describes the same path as "10" while reading like a typo. A genuine
        return to a hold after visiting another one is kept, because that is a
        different path. The uncollapsed contacts are still in `per_pair` and in
        limb_usage.csv.
        """
        out: list[int] = []
        for hold_id in self.sequence_by_limb[limb_key]:
            if not out or out[-1] != hold_id:
                out.append(hold_id)
        return out

    def seconds(self, limb_key: str) -> float:
        return self.per_limb[limb_key] / self.fps if self.fps else 0.0

    def share(self, limb_key: str) -> float:
        """This limb's fraction of all contact time. The four sum to 1."""
        return self.per_limb[limb_key] / self.total_frames if self.total_frames else 0.0

    def coverage(self, limb_key: str) -> float:
        """Fraction of the climb this limb was on a hold. These do not sum to 1."""
        return self.per_limb[limb_key] / self.journey_frames if self.journey_frames else 0.0

    def pair_share(self, limb_key: str, hold_id: int) -> float:
        frames = self.per_pair.get((limb_key, hold_id), 0)
        return frames / self.total_frames if self.total_frames else 0.0

    def limbs_on(self, hold_id: int) -> list[str]:
        """Which limbs used this hold, in panel order."""
        return [limb.key for limb in LIMBS if (limb.key, hold_id) in self.per_pair]

    def rows(self) -> list[dict]:
        """One row per (limb, hold) that actually happened, biggest first."""
        out = []
        for (limb_key, hold_id), frames in self.per_pair.items():
            out.append({
                "limb": limb_key,
                "hold_id": hold_id,
                "seconds": round(frames / self.fps, 3) if self.fps else 0.0,
                "frames": frames,
                "share_of_contact": round(frames / self.total_frames, 5)
                if self.total_frames else 0.0,
            })
        out.sort(key=lambda r: (-r["frames"], r["hold_id"]))
        return out

    def as_dict(self) -> dict:
        return {
            "total_contact_seconds": round(self.total_frames / self.fps, 3)
            if self.fps else 0.0,
            "per_limb": {
                limb.key: {
                    "seconds": round(self.seconds(limb.key), 3),
                    "share_of_contact": round(self.share(limb.key), 5),
                    "coverage_of_climb": round(self.coverage(limb.key), 5),
                    "holds": sorted(self.holds_by_limb[limb.key]),
                    "n_holds": len(self.holds_by_limb[limb.key]),
                    "sequence": self.sequence(limb.key),
                    "sequence_raw": self.sequence_by_limb[limb.key],
                }
                for limb in LIMBS
            },
            "per_limb_hold": self.rows(),
        }


def running_utilization(analysis: ClimbAnalysis, frames: list[int]) -> dict[int, dict[str, int]]:
    """Cumulative contact frames per limb, at every frame. For the live panel.

    Clipped to the same window as everything else, so the bars on the panel land
    exactly on the percentages in the report. Built by sweeping the contact
    intervals once rather than re-summing them per frame, so the panel costs one
    pass over the climb instead of one per frame.
    """
    lo, hi = analysis.window if analysis.window else (-10**9, 10**9)
    delta: dict[int, dict[str, int]] = {}
    for contact in analysis.contacts:
        for frame in range(max(contact.start, lo), min(contact.end, hi) + 1):
            delta.setdefault(frame, {})
            delta[frame][contact.limb] = delta[frame].get(contact.limb, 0) + 1

    running = {limb.key: 0 for limb in LIMBS}
    out: dict[int, dict[str, int]] = {}
    for frame in frames:
        for key, count in delta.get(frame, {}).items():
            running[key] += count
        out[frame] = dict(running)
    return out


def running_sequence(analysis: ClimbAnalysis,
                     frames: list[int]) -> dict[int, dict[str, list[int]]]:
    """Each limb's hold sequence as it stood at every frame. For the live panel.

    A hold joins the list the moment that contact is confirmed, so the list on
    screen only ever claims what has already happened.
    """
    lo, hi = analysis.window if analysis.window else (-10**9, 10**9)
    starts: dict[int, list[tuple[str, int]]] = {}
    for contact in sorted(analysis.contacts, key=lambda c: c.start):
        start = max(contact.start, lo)
        if start > min(contact.end, hi):
            continue
        starts.setdefault(start, []).append((contact.limb, contact.hold_id))

    running: dict[str, list[int]] = {limb.key: [] for limb in LIMBS}
    out: dict[int, dict[str, list[int]]] = {}
    for frame in frames:
        for limb_key, hold_id in starts.get(frame, ()):
            # Collapsed the same way Utilization.sequence does it, so the line
            # on screen is the line in the report.
            if not running[limb_key] or running[limb_key][-1] != hold_id:
                running[limb_key].append(hold_id)
        out[frame] = {k: list(v) for k, v in running.items()}
    return out


# ── the sequence ─────────────────────────────────────────────────────────────
# The line that differs between two attempts at one route, in the two forms that
# are worth keeping. Both are restricted to the climb itself, like everything
# else the report says, so a hold brushed while pulling on is not a move.

def activation_sequence(analysis: ClimbAnalysis) -> list[int]:
    """The holds in the order they were first used, whatever touched them.

    Hand or foot makes no difference here, and that is deliberate: this is the
    *path up the wall*, the thing two attempts at one problem can be laid
    side by side and read against each other. Which appendage arrived is a
    different question, and `limb_sequence` answers it.
    """
    return [hid for hid, _ in sorted(analysis.order.items(), key=lambda kv: kv[1])]


def limb_sequence(analysis: ClimbAnalysis) -> list[dict]:
    """Every move of the climb in order: which limb went to which hold, when.

    One entry per contact, so a hold used by a hand and later by a foot appears
    twice — those are two moves. A limb that comes off a hold and goes straight
    back onto it is merged into one entry, the same rule `Utilization.sequence`
    collapses on: that is a grip settling, not a move. Merging is judged per
    limb, not against the previous entry in the list — the other three limbs
    keep moving while a hand resettles, and a foot placed in between does not
    turn one hand's fidget into two moves.
    """
    lo, hi = analysis.window if analysis.window else (-10**9, 10**9)
    origin = lo if analysis.window else min(analysis.midline, default=0)

    merged: list[dict] = []
    latest: dict[str, dict] = {}        # limb -> its most recent entry
    for contact in sorted(analysis.contacts, key=lambda c: (c.start, c.limb)):
        start, end = max(contact.start, lo), min(contact.end, hi)
        if end < start:
            continue
        previous = latest.get(contact.limb)
        if previous is not None and previous["hold_id"] == contact.hold_id:
            previous["end"] = max(previous["end"], end)
            continue
        entry = {"limb": contact.limb, "hold_id": contact.hold_id,
                 "start": start, "end": end}
        merged.append(entry)
        latest[contact.limb] = entry

    fps = analysis.fps or 1.0
    return [{
        "step": step,
        "hold_id": move["hold_id"],
        "limb": move["limb"],
        "limb_label": LIMB_BY_KEY[move["limb"]].label,
        "time": round((move["start"] - origin) / fps, 3),
        "contact_seconds": round((move["end"] - move["start"] + 1) / fps, 3),
    } for step, move in enumerate(merged, start=1)]


def first_limb_by_hold(analysis: ClimbAnalysis) -> dict[int, str]:
    """Which limb got to each hold first. Keyed by hold id."""
    first: dict[int, tuple[int, str]] = {}
    for move in limb_sequence(analysis):
        hid = move["hold_id"]
        if hid not in first or move["step"] < first[hid][0]:
            first[hid] = (move["step"], move["limb"])
    return {hid: limb for hid, (_, limb) in first.items()}


def hold_times(analysis: ClimbAnalysis, *, cfg) -> list[dict]:
    """Per hold: when it was first used, when contact ended, and by which limbs.

    Restricted to the climb, so every figure here agrees with the panel and with
    the utilization split. Touches from before the clock started are counted
    separately, in the summary, rather than mixed in with the send.
    """
    lo, hi = analysis.window if analysis.window else (
        min(analysis.midline, default=0), max(analysis.midline, default=0))

    by_hold: dict[int, list[Contact]] = {}
    for contact in analysis.contacts:
        start, end = max(contact.start, lo), min(contact.end, hi)
        if end < start:
            continue
        by_hold.setdefault(contact.hold_id, []).append(Contact(
            contact.limb, contact.hold_id, start, end))

    rows = []
    for hold in sorted(analysis.holds, key=lambda h: h["id"]):
        hid = hold["id"]
        found = by_hold.get(hid)
        if not found:
            rows.append({"hold_id": hid, "order": None, "limbs": "",
                         "time_start": None, "time_end": None,
                         "contact_seconds": None})
            continue

        first = min(c.start for c in found)
        last = max(c.end for c in found)
        # Summed over limbs: two hands on one hold for a second is two
        # limb-seconds of use, and the split below has to add up to that.
        frames = sum(c.frames() for c in found)
        limbs = [limb.label for limb in LIMBS
                 if any(c.limb == limb.key for c in found)]
        rows.append({
            "hold_id": hid,
            "order": analysis.order.get(hid),
            "limbs": "+".join(limbs),
            "time_start": round((first - lo) / analysis.fps, 3),
            "time_end": round((last - lo) / analysis.fps, 3),
            "contact_seconds": round(frames / analysis.fps, 3),
        })
    return rows


def route_path(analysis: ClimbAnalysis, *, sigma: float,
               n_points: int = 60) -> list[tuple[float, float]]:
    """The smoothed midline path from pulling on to topping out, normalized.

    Normalized rather than in pixels, and on the canvas rather than in a frame:
    the line is a statement about the wall, so it is the renderer's job to place
    it in whatever panel it ends up drawn on — and it no longer has to be
    recomputed if that panel changes size.

    Smoothed because the raw centroid wobbles with every reach; the line is
    meant to read as the shape of the climb, not as a seismograph of it.
    """
    from scipy.ndimage import gaussian_filter1d

    if analysis.start_frame is None:
        return []
    end = analysis.completion_frame
    if end is None:
        end = analysis.activated_at.get(analysis.final_hold_id)
    if end is None:
        return []

    frames = sorted(f for f in analysis.midline if analysis.start_frame <= f <= end)
    if len(frames) < 4:
        return []

    xs = gaussian_filter1d(np.array([analysis.midline[f][0] for f in frames]),
                           sigma=sigma, mode="nearest")
    ys = gaussian_filter1d(np.array([analysis.midline[f][1] for f in frames]),
                           sigma=sigma, mode="nearest")
    idx = np.linspace(0, len(xs) - 1, min(n_points, len(xs))).astype(int)
    return [(float(xs[i]), float(ys[i])) for i in idx]


def summary_text(analysis: ClimbAnalysis, rows: list[dict], util: Utilization, *,
                 source: str, color: str, numbering: str, warmup: int = 0) -> str:
    """The climb, human-readable."""
    lines = [
        f"Route: {color} holds — {source}",
        "=" * 64,
        f"{'holds detected':<28}{len(analysis.holds)}",
        f"{'holds used':<28}{analysis.n_used}",
    ]
    if analysis.elapsed is not None:
        lines.append(f"{'time on the wall':<28}{analysis.elapsed:.2f}s")
    elif analysis.start_frame is not None:
        lines.append(f"{'time on the wall':<28}did not top out")
    else:
        lines.append(f"{'time on the wall':<28}start never detected")
    lines += [
        f"{'topped out':<28}{'yes' if analysis.completion_frame is not None else 'no'}",
        f"{'start rule':<28}" + ("both feet clear of the floor, one hand on a hold"
                                   if analysis.start_rule == "floor"
                                   else "both feet on holds (no floor found)"),
        f"{'hold numbering':<28}bottom to top, "
        f"{'left to right' if numbering == 'ltr' else 'right to left'} within a row",
        "",
        "Limb utilization",
        "-" * 64,
        "on hold = this limb's contact time over the climb's duration. The four",
        "do not add to 100% and cannot — limbs hold at the same time. Each one",
        "stands alone, and reads against another climber's same limb.",
        "",
        f"{'limb':<12}{'contact':>9}{'on hold':>9}{'holds':>7}   used",
    ]
    for limb in LIMBS:
        used = util.holds_by_limb[limb.key]
        lines.append(
            f"{limb.name:<12}{util.seconds(limb.key):>8.2f}s"
            f"{util.coverage(limb.key) * 100:>8.1f}%"
            f"{len(used):>7}   {', '.join(str(h) for h in sorted(used)) or '—'}")

    lines += [
        "",
        "Beta — the holds each limb used, in order (revisits included)",
        "-" * 64,
        "This is the line to compare between two people on the same route.",
        "",
    ]
    for limb in LIMBS:
        seq = util.sequence(limb.key)
        lines.append(f"{limb.name:<12}{' '.join(str(h) for h in seq) or '—'}")

    lines += [
        "",
        "Holds, by standardized id",
        "-" * 64,
        "Times are from the start of the climb — both feet off the ground — and",
        "cover the climb only. Touches from before the clock started are counted",
        "at the end, not mixed in with the send.",
        "",
        f"{'id':>3}  {'#':>3}  {'limbs':<12}{'first use':>10}{'last use':>10}{'contact':>10}",
    ]
    for row in rows:
        if row["order"] is None:
            lines.append(f"{row['hold_id']:>3}    –  {'never used':<12}"
                         f"{'—':>10}{'—':>10}{'—':>10}")
        else:
            lines.append(
                f"{row['hold_id']:>3}  {row['order']:>3}  {row['limbs']:<12}"
                f"{row['time_start']:>9.2f}s{row['time_end']:>9.2f}s"
                f"{row['contact_seconds']:>9.2f}s")

    if warmup:
        lines += ["", f"{warmup} hold(s) were also touched while pulling on, "
                      "before the clock started; not counted above."]

    busiest = util.rows()[:5]
    if busiest:
        lines += ["", "Longest single limb-on-hold contacts", "-" * 64]
        for row in busiest:
            lines.append(f"  {LIMB_BY_KEY[row['limb']].name:<12} on hold "
                         f"{row['hold_id']:<3} {row['seconds']:>7.2f}s"
                         f"  ({row['share_of_contact'] * 100:.1f}% of contact)")
    return "\n".join(lines) + "\n"
