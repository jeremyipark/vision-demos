"""Four attempts at one route, read against each other.

A single clip cannot show you that you climbed the same problem four different
ways. You watch one, and the sequence it lit up looks like *the* sequence. The
comparison is only visible from outside any one video, which is what this module
builds: every attempt's path up the wall, in one numbering, with the differences
between them marked.

The whole thing rests on one assumption, and it is the assumption most likely to
be quietly false: that hold 7 is the same lump of resin in every clip. Each clip
gets its own SAM pass and its own consensus, and the numbering is positional —
bottom to top, along the route's lean. So a clip that missed one hold does not
produce a sequence with a gap in it. It produces a sequence where every number
above the missing hold is one too low, and nothing about it looks wrong.

Hence `align`. Holds are matched between clips by where they are on the wall,
not by what they were numbered, and every sequence is re-expressed in one
reference clip's numbering before anything is compared. `align` itself stays
partial — a hold with no counterpart is absent from its map, not guessed at —
and `_numbering` closes it into the total map the run renumbers by, giving the
unmatched ones fresh ids above the reference's last.

That numbering is then adopted by the clip itself: the ids drawn on the wall in
the video are these, not the ones its own SAM pass happened to assign. One run,
one numbering, in the overlay and the artifacts and the card alike — otherwise
the video says you matched hold 15 and the card under it says 14.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def centroid(hold: dict) -> tuple[float, float]:
    x, y, w, h = hold["bbox"]
    return x + w / 2, y + h / 2


def _distance(a: dict, b: dict, aspect: float) -> float:
    """Centroid distance in frame heights, isotropic.

    Coordinates are normalized independently per axis, so a raw distance is an
    ellipse: on a portrait clip a hold 0.02 away sideways is nearly twice as far
    in pixels as one 0.02 away vertically. Scaling x by the aspect ratio puts
    both axes in the same physical unit, the way HoldGeometry does.
    """
    (ax, ay), (bx, by) = centroid(a), centroid(b)
    dx = (ax - bx) * aspect
    dy = ay - by
    return (dx * dx + dy * dy) ** 0.5


def align(reference: list[dict], other: list[dict], *, max_distance: float,
          aspect: float = 1.0) -> tuple[dict[int, int], dict]:
    """Map *other*'s hold ids onto *reference*'s, by position on the wall.

    Greedy over every candidate pair in distance order, nearest first, each hold
    claimed at most once. Greedy is the right shape here rather than a lazy
    nearest-neighbour: two holds set close together would otherwise both match
    whichever reference hold sits between them, and one of the two would be
    silently dropped in favour of a worse pairing.

    Returns ``(id_map, stats)``. Ids absent from the map had no counterpart
    within *max_distance* and are genuinely unmatched — not zero, not guessed.
    """
    pairs = sorted(
        ((_distance(r, o, aspect), r["id"], o["id"]) for r in reference for o in other),
        key=lambda p: (p[0], p[1], p[2]))

    id_map: dict[int, int] = {}
    claimed: set[int] = set()
    distances: list[float] = []
    for distance, ref_id, other_id in pairs:
        if distance > max_distance:
            break
        if other_id in id_map or ref_id in claimed:
            continue
        id_map[other_id] = ref_id
        claimed.add(ref_id)
        distances.append(distance)

    # Matched holds that kept their own number are the good case: it means both
    # clips saw the same wall and numbered it the same way, and the map is a
    # formality. A low agreement with a high match rate means the clips found
    # the same holds and disagree about their order, which the map fixes and
    # which is worth saying out loud.
    agreed = sum(1 for other_id, ref_id in id_map.items() if other_id == ref_id)
    return id_map, {
        "matched": len(id_map),
        "reference_holds": len(reference),
        "other_holds": len(other),
        "unmatched_in_other": sorted(h["id"] for h in other if h["id"] not in id_map),
        "unmatched_in_reference": sorted(h["id"] for h in reference
                                         if h["id"] not in claimed),
        "id_agreement": round(agreed / len(id_map), 4) if id_map else 0.0,
        "renumbered": len(id_map) - agreed,
        "max_offset": round(max(distances), 5) if distances else None,
        "mean_offset": round(sum(distances) / len(distances), 5) if distances else None,
    }


@dataclass
class Attempt:
    """One clip's climb, in its own hold numbering until `Comparison` aligns it."""

    label: str                      # the clip's stem — what the panel calls it
    video: str
    holds: list[dict]
    sequence: list[int]             # activation order, this clip's own ids
    limb_sequence: list[dict]
    elapsed: float | None
    topped_out: bool
    n_detected: int
    n_used: int

    # Filled in by `build`, in the reference clip's numbering.
    id_map: dict[int, int] = field(default_factory=dict)
    # Total where `id_map` is partial: every hold this clip found, including the
    # ones the reference wall has no counterpart for. This is the map the run
    # renumbers by, so it cannot leave a hold without a number.
    numbering: dict[int, int] = field(default_factory=dict)
    aligned_sequence: list[int] = field(default_factory=list)
    alignment: dict = field(default_factory=dict)
    is_reference: bool = False
    index: int = 0                  # 1-based, by filename ascending

    @property
    def n_moves(self) -> int:
        return len(self.sequence)

    def display(self, template: str = "ATTEMPT {n}") -> str:
        """What the panel calls this attempt.

        Its position in the batch, not its filename: the filename is whatever the
        camera happened to name it and means nothing to anyone watching, while
        "ATTEMPT 2" is a fact about the session. The filename stays in every
        artifact, so a row on the card can still be traced back to a file.
        """
        return template.format(n=self.index, label=self.label, video=self.video)

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "index": self.index,
            "video": self.video,
            "topped_out": self.topped_out,
            "elapsed_seconds": self.elapsed,
            "holds_detected": self.n_detected,
            "holds_used": self.n_used,
            "sequence": self.sequence,
            "sequence_aligned": self.aligned_sequence,
            "limb_sequence": self.limb_sequence,
            "alignment": {"is_reference": self.is_reference,
                          "id_map": {str(k): v for k, v in sorted(self.id_map.items())},
                          **self.alignment},
        }


@dataclass
class Comparison:
    """Every attempt at one route, in one numbering."""

    attempts: list[Attempt]
    reference: str
    aligned: bool                   # False when ids were left as each clip found them
    route: dict = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    def by_label(self, label: str) -> Attempt | None:
        return next((a for a in self.attempts if a.label == label), None)

    def others(self, label: str) -> list[Attempt]:
        """The other attempts, shortest first — the panel's reading order."""
        return sorted((a for a in self.attempts if a.label != label),
                      key=lambda a: (not a.topped_out, len(a.aligned_sequence),
                                     a.elapsed if a.elapsed is not None else 1e9))

    def _sends(self) -> list[Attempt]:
        """Only attempts that topped out: a bailed one used fewer holds by not
        finishing, and calling that the shortest route would be a lie."""
        return [a for a in self.attempts if a.topped_out]

    def shortest(self) -> Attempt | None:
        sends = self._sends()
        return min(sends, key=lambda a: (len(a.aligned_sequence),
                                         a.elapsed if a.elapsed is not None else 1e9),
                   default=None)

    def fastest(self) -> Attempt | None:
        sends = [a for a in self._sends() if a.elapsed is not None]
        return min(sends, key=lambda a: a.elapsed, default=None)

    def shared_holds(self) -> set[int]:
        """Holds every attempt that topped out used. The context, not the news."""
        sends = self._sends()
        if not sends:
            return set()
        common = set(sends[0].aligned_sequence)
        for attempt in sends[1:]:
            common &= set(attempt.aligned_sequence)
        return common

    def as_dict(self) -> dict:
        shortest, fastest = self.shortest(), self.fastest()
        return {
            "route": self.route,
            "reference": self.reference,
            "ids_aligned": self.aligned,
            "n_attempts": len(self.attempts),
            "shortest": shortest.label if shortest else None,
            "fastest": fastest.label if fastest else None,
            "shared_holds": sorted(self.shared_holds()),
            "consistency_problems": self.problems,
            "attempts": [a.as_dict() for a in self.attempts],
        }


def build(attempts: list[Attempt], *, aspect: float = 1.0, max_distance: float = 0.04,
          drift_warn: float = 0.02, align_ids: bool = True,
          route: dict | None = None) -> Comparison:
    """Align every attempt onto one clip's numbering and collect the problems.

    The reference is the wall the clips *agree* on: the most common hold count,
    not the largest. Taking the largest lets a single clip's extra detection
    become canonical for all of them — with three clips finding 16 holds and one
    finding 17, the odd one out would set the numbering and every sequence would
    shift by one above the phantom hold. It is also visible, because the render
    draws each clip's own ids: the video says you matched hold 16 and the panel
    under it says 17.

    Ties in frequency go to the larger count (a hold two clips missed is more
    likely real than imagined), then to the label, so the choice is stable
    between runs.
    """
    if not attempts:
        return Comparison([], reference="", aligned=False, route=route or {})

    # Numbered by filename, ascending — the order the clips were shot, which is
    # the order "attempt 2" means anything in. Assigned here rather than trusting
    # the caller's ordering, so the panel and the artifacts cannot disagree.
    for i, one in enumerate(sorted(attempts, key=lambda a: a.label), start=1):
        one.index = i

    from collections import Counter
    counts = Counter(a.n_detected for a in attempts)
    agreed_count = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
    reference = min((a for a in attempts if a.n_detected == agreed_count),
                    key=lambda a: a.label)
    problems: list[str] = []

    for attempt in attempts:
        attempt.is_reference = attempt is reference
        if attempt is reference:
            attempt.id_map = {h["id"]: h["id"] for h in attempt.holds}
            attempt.alignment = {
                "matched": len(attempt.holds), "reference_holds": len(attempt.holds),
                "other_holds": len(attempt.holds), "unmatched_in_other": [],
                "unmatched_in_reference": [], "id_agreement": 1.0, "renumbered": 0,
                "max_offset": 0.0, "mean_offset": 0.0,
            }
        else:
            attempt.id_map, attempt.alignment = align(
                reference.holds, attempt.holds,
                max_distance=max_distance, aspect=aspect)

        attempt.numbering = _numbering(attempt, reference) if align_ids else \
            {h["id"]: h["id"] for h in attempt.holds}
        attempt.aligned_sequence = [attempt.numbering[h] for h in attempt.sequence]

        problems.extend(_problems(attempt, reference, drift_warn=drift_warn))

    return Comparison(attempts=attempts, reference=reference.label,
                      aligned=align_ids, route=route or {}, problems=problems)


def _numbering(attempt: Attempt, reference: Attempt) -> dict[int, int]:
    """Every one of *attempt*'s hold ids, in the reference clip's numbering.

    `align` is deliberately partial — a hold with no counterpart is absent from
    its map rather than guessed at — but a *renumbering* cannot be: the run
    relabels the wall in the video by this map, and a hold left out would be a
    hold with no number under the climber's hand.

    So the unmatched ones are given fresh ids above the reference's last, in
    their own order. That says what is true about them — they are on this wall
    and not on the reference's — while keeping them findable, which a shared
    placeholder would not. They are also still reported as a problem: a clip
    that needs one has segmented the wall differently from its siblings.
    """
    ceiling = max((h["id"] for h in reference.holds), default=0)
    out: dict[int, int] = {}
    for hold in sorted(attempt.holds, key=lambda h: h["id"]):
        own = hold["id"]
        if own in attempt.id_map:
            out[own] = attempt.id_map[own]
        else:
            ceiling += 1
            out[own] = ceiling
    return out


def _problems(attempt: Attempt, reference: Attempt, *,
              drift_warn: float = 0.02) -> list[str]:
    """What is wrong with this clip's wall, relative to the reference clip's.

    Stated as sentences rather than raised, the way `climb.check` does it: a
    clip that disagrees still renders, and the render is how you see why.
    """
    if attempt is reference:
        return []

    out = []
    stats = attempt.alignment
    name, ref = attempt.label, reference.label

    if attempt.n_detected != reference.n_detected:
        out.append(f"{name} found {attempt.n_detected} holds, {ref} found "
                   f"{reference.n_detected} — the same wall should segment to the "
                   f"same count")

    missing = stats["unmatched_in_reference"]
    if missing:
        out.append(f"{name} is missing hold(s) {', '.join(map(str, missing))} that "
                   f"{ref} found; they are dropped from its sequence, not renumbered")

    extra = stats["unmatched_in_other"]
    if extra:
        out.append(f"{name} has {len(extra)} hold(s) with no counterpart on {ref}'s "
                   f"wall; they are numbered above {ref}'s last hold, so they "
                   f"cannot collide with it")

    if stats["renumbered"]:
        out.append(f"{stats['renumbered']} of {name}'s holds were numbered "
                   f"differently than {ref}'s and have been re-expressed in {ref}'s "
                   f"numbering")

    # A drift this large is not a hold the camera saw from a slightly different
    # angle. Either the tripod moved between takes or two different holds were
    # matched to each other, and both make the comparison meaningless.
    if stats["max_offset"] is not None and stats["max_offset"] > drift_warn:
        out.append(f"{name}'s holds sit up to {stats['max_offset']:.3f} of a frame "
                   f"height from {ref}'s — check the camera did not move between takes")

    return out


def summary_text(comparison: Comparison) -> str:
    """The comparison, human-readable. Sits next to the videos as sequences.txt."""
    shortest = comparison.shortest()
    fastest = comparison.fastest()
    route = comparison.route

    header = " · ".join(str(v) for v in (route.get("color"), route.get("grade"),
                                         route.get("name")) if v)
    lines = [
        f"Route: {header or 'unnamed'} — {len(comparison.attempts)} attempts",
        "=" * 72,
        "The sequence is the holds in the order they were first used, whatever",
        f"touched them. Numbering is {comparison.reference}'s — the hold count the",
        "clips agree on; every other clip's holds were matched to it by position",
        "on the wall.",
        "",
        f"{'attempt':<12}{'clip':<12}{'holds':>6}{'time':>9}   sequence",
    ]
    for attempt in sorted(comparison.attempts,
                          key=lambda a: (not a.topped_out, len(a.aligned_sequence))):
        tags = []
        if shortest and attempt.label == shortest.label:
            tags.append("shortest")
        if fastest and attempt.label == fastest.label:
            tags.append("fastest")
        if not attempt.topped_out:
            tags.append("did not top out")
        sequence = " ".join(str(h) for h in attempt.aligned_sequence) or "—"
        lines.append(
            f"{attempt.display():<12}{attempt.label:<12}"
            f"{len(attempt.aligned_sequence):>6}"
            f"{(f'{attempt.elapsed:.2f}s' if attempt.elapsed is not None else '—'):>9}"
            f"   {sequence}"
            + (f"   [{', '.join(tags)}]" if tags else ""))

    shared = comparison.shared_holds()
    if shared:
        lines += ["", f"Used in every send: {' '.join(map(str, sorted(shared)))}"]
    for attempt in comparison.attempts:
        unique = [h for h in attempt.aligned_sequence
                  if all(
                      h not in other.aligned_sequence
                      for other in comparison.attempts if other is not attempt)]
        if unique:
            lines.append(f"Only {attempt.display().lower()} ({attempt.label}): "
                         f"{' '.join(map(str, unique))}")

    lines += ["", "Moves, by attempt", "-" * 72,
              "Every contact in order, with the appendage that made it.", ""]
    for attempt in comparison.attempts:
        lines.append(f"{attempt.display()}  ({attempt.label})")
        if not attempt.limb_sequence:
            lines.append("  —")
        for move in attempt.limb_sequence:
            hold = attempt.id_map.get(move["hold_id"], move["hold_id"]) \
                if comparison.aligned else move["hold_id"]
            lines.append(f"  {move['step']:>3}. {move['limb_label']:<3} -> hold "
                         f"{hold if hold is not None else '?':<4} "
                         f"at {move['time']:>6.2f}s  ({move['contact_seconds']:.2f}s)")
        lines.append("")

    if comparison.problems:
        lines += ["Consistency", "-" * 72]
        lines += [f"  ! {p}" for p in comparison.problems]
        lines.append("")

    return "\n".join(lines) + "\n"
