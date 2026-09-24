"""The two panels.

Left is the clip with what the models saw drawn onto it: every hold on the route
outlined and filled, the climber's skeleton over the top. Right is the same wall
with the climber taken away — the holds start dim, light up in the order they are
used and carry that order as a number, and the body is reduced to the one dot the
whole climb is measured from. The clock runs from pulling on, and stops on the top.

Taking the climber away is the point of the right panel: the left one shows that
the models work, and the right one shows what they *found*, which is a route.

Handheld, the two panels stop sharing a coordinate system, and the split is what
the demo is about. The left panel is the camera's view, so the route is projected
into every frame through that frame's homography and rides the pan. The right
panel is the *wall's* view — the mosaic from :mod:`src.mosaic`, which holds
still no matter what the operator does — so the route is drawn on it once and
never moves again. A rectangle on the right traces what the left is currently
looking at, because once the camera has zoomed in, nothing else says which part
of the wall the live shot is a detail of.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from . import (climb, floor as floor_mod, holds as holds_mod, skeleton,
               text as text_mod)


def _px(base: float, scale: float) -> int:
    return max(1, int(round(base * scale)))


def _polygon_px(hold: dict, width: int, height: int):
    polygon = hold.get("polygon")
    if not polygon or len(polygon) < 3:
        return None
    arr = np.asarray(polygon, dtype=float)
    arr[:, 0] *= width
    arr[:, 1] *= height
    return arr.reshape(-1, 1, 2).astype(np.int32)


def _dashed_polyline(frame, poly, color, thickness, *, dash: int = 6, gap: int = 4):
    """A closed polygon drawn as a dashed outline, walked at constant arc length."""
    pts = poly.reshape(-1, 2).astype(float)
    loop = np.vstack([pts, pts[:1]])
    drawing, budget = True, float(dash)
    for a, b in zip(loop, loop[1:]):
        length = float(np.hypot(*(b - a)))
        walked = 0.0
        while walked < length:
            step = min(budget, length - walked)
            if drawing:
                p0 = a + (b - a) * (walked / length)
                p1 = a + (b - a) * ((walked + step) / length)
                cv2.line(frame, tuple(np.round(p0).astype(int)),
                         tuple(np.round(p1).astype(int)), color, thickness, cv2.LINE_AA)
            walked += step
            budget -= step
            if budget <= 1e-6:
                drawing = not drawing
                budget = float(dash if drawing else gap)


def draw_hold(frame, hold, width, height, color, thickness, *, fill: bool,
              alpha: float, dashed: bool = False):
    """Outline a hold, optionally filling it. Falls back to the box with no mask.

    *dashed* marks a hold recovered by `src.recover` rather than segmented from
    the prompt. It rests on one box-prompted look plus the climber's behaviour,
    not on a hundred sightings agreeing, and a dashed outline says so instead of
    letting it pass as one of the others.
    """
    poly = _polygon_px(hold, width, height)
    if poly is None:
        x, y, w, h = hold["bbox"]
        cv2.rectangle(frame, (int(x * width), int(y * height)),
                      (int((x + w) * width), int((y + h) * height)),
                      color, thickness, cv2.LINE_AA)
        return

    if fill and alpha > 0:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [poly], color)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    if dashed:
        _dashed_polyline(frame, poly, color, thickness)
    else:
        cv2.polylines(frame, [poly], True, color, thickness, cv2.LINE_AA)


def draw_dashed_spline(frame, points, color, thickness, dash, gap):
    """A dashed cubic spline through *points*, walked at constant arc length.

    Dashed rather than solid because the line crosses the holds it is explaining;
    a solid stroke would hide the very shapes the panel is there to show.
    """
    if len(points) < 2:
        return
    from scipy.interpolate import make_interp_spline

    arr = np.asarray(points, dtype=float)
    n = len(arr)
    t = np.linspace(0, 1, n)
    fine = np.linspace(0, 1, max(400, n * 60))
    k = min(n - 1, 3)
    xs = make_interp_spline(t, arr[:, 0], k=k)(fine)
    ys = make_interp_spline(t, arr[:, 1], k=k)(fine)

    drawing, budget = True, float(dash)
    prev = (int(xs[0]), int(ys[0]))
    seg_start = prev
    for i in range(1, len(xs)):
        curr = (int(xs[i]), int(ys[i]))
        step = float(np.hypot(curr[0] - prev[0], curr[1] - prev[1]))
        if step < 0.5:
            continue
        while step > 0:
            if step <= budget:
                budget -= step
                if drawing:
                    cv2.line(frame, seg_start, curr, color, thickness, cv2.LINE_AA)
                step = 0
            else:
                frac = budget / step
                mid = (int(prev[0] + frac * (curr[0] - prev[0])),
                       int(prev[1] + frac * (curr[1] - prev[1])))
                if drawing:
                    cv2.line(frame, seg_start, mid, color, thickness, cv2.LINE_AA)
                drawing = not drawing
                budget = float(dash if drawing else gap)
                seg_start = mid
                step = float(np.hypot(curr[0] - mid[0], curr[1] - mid[1]))
                prev = mid
        prev = curr


def draw_limb_pips(frame, hold, width, height, limb_keys, *, colors, radius, gap,
                   offset, active: set):
    """A row of dots under a hold, one per limb that has used it.

    Under the hold rather than on it: the mask is the hold's shape and covering
    it with markers would trade the thing being shown for the annotation. A limb
    currently holding gets a white ring, so the panel distinguishes "this hand
    used this hold" from "this hand is on it now".
    """
    if not limb_keys:
        return
    x, y, w, h = hold["bbox"]
    cx = int((x + w / 2) * width)
    cy = int((y + h) * height) + offset + radius
    span = len(limb_keys) * (2 * radius) + (len(limb_keys) - 1) * gap
    px = cx - span // 2 + radius
    for key in limb_keys:
        cv2.circle(frame, (px, cy), radius, colors[key], -1, cv2.LINE_AA)
        if key in active:
            cv2.circle(frame, (px, cy), radius + max(1, radius // 2),
                       (255, 255, 255), 1, cv2.LINE_AA)
        px += 2 * radius + gap


def draw_limb_panel(frame, limbs, running, holding, sequences, elapsed_frames, *,
                    origin, cfg, font, scale, _px, right_reserve=0):
    """One row per limb: the holds it has used, in order, and its on-hold share.

    The percentage is this limb's contact time over the climb *so far* — not a
    slice of a pie shared with the other three. The four do not add to 100% and
    cannot, because limbs hold at the same time; each one stands alone, which is
    what makes it readable against another climber's same limb.

    The sequence is the beta, and it is the line that actually differs between
    two people on one route. It grows as the climb does; when it outruns the
    panel the oldest entries are dropped, since the recent ones are the ones
    still changing.
    """
    x0, y0 = origin
    swatch = _px(cfg.LIMB_PANEL_SWATCH, scale)
    bar_w = _px(cfg.LIMB_PANEL_BAR_WIDTH, scale)
    bar_h = _px(cfg.LIMB_PANEL_BAR_HEIGHT, scale)
    label_size = _px(cfg.LIMB_PANEL_LABEL_SIZE, scale)
    value_size = _px(cfg.LIMB_PANEL_VALUE_SIZE, scale)
    title_size = _px(cfg.LIMB_PANEL_TITLE_SIZE, scale)
    row_gap = _px(cfg.LIMB_PANEL_ROW_GAP, scale)
    pad = _px(6, scale)

    # Column geometry is measured, not assumed: the widest limb name and the
    # widest percentage set the gutter, so the two columns line up no matter
    # what the font renders at, and the sequences all start at one x rather than
    # each hanging off the end of its own percentage.
    name_w = max(text_mod.measure(limb.name, font, label_size)[0] for limb in limbs)
    pct_w = text_mod.measure("100%", font, value_size)[0]
    col1_x = x0 + swatch + pad
    bar_x = col1_x + name_w + pad * 2
    pct_x = bar_x + bar_w + pad * 2
    col2_x = pct_x + pct_w + _px(cfg.LIMB_PANEL_COLUMN_GUTTER, scale)

    y = y0
    if cfg.LIMB_PANEL_TITLE or cfg.LIMB_PANEL_TITLE_2:
        if cfg.LIMB_PANEL_TITLE:
            text_mod.draw(frame, cfg.LIMB_PANEL_TITLE, font, size=title_size,
                          xy=(x0, y), color=(150, 150, 150), anchor="lt")
        if cfg.LIMB_PANEL_TITLE_2:
            text_mod.draw(frame, cfg.LIMB_PANEL_TITLE_2, font, size=title_size,
                          xy=(col2_x, y), color=(150, 150, 150), anchor="lt")
        y += title_size + row_gap * 2

    row_h = max(swatch, bar_h, value_size)
    for limb in limbs:
        color = cfg.LIMB_COLORS[limb.key]
        on = holding.get(limb.key) is not None
        opacity = 1.0 if on else cfg.LIMB_PANEL_IDLE_DIM
        mid = y + row_h // 2

        dim = tuple(int(c * opacity) for c in color)
        cv2.rectangle(frame, (x0, mid - swatch // 2),
                      (x0 + swatch, mid + swatch - swatch // 2), dim, -1)
        if on:
            cv2.rectangle(frame, (x0 - 1, mid - swatch // 2 - 1),
                          (x0 + swatch + 1, mid + swatch - swatch // 2 + 1),
                          (255, 255, 255), 1, cv2.LINE_AA)

        text_mod.draw(frame, limb.name, font, size=label_size, xy=(col1_x, mid),
                      color=(230, 230, 230), anchor="lm", opacity=opacity)

        cv2.rectangle(frame, (bar_x, mid - bar_h // 2),
                      (bar_x + bar_w, mid + bar_h - bar_h // 2),
                      cfg.LIMB_PANEL_BAR_BG, -1)
        # Of the climb so far, not of the other limbs. A limb that has held
        # something for every frame reads 100% and is not competing with anyone.
        coverage = (running.get(limb.key, 0) / elapsed_frames) if elapsed_frames else 0.0
        coverage = min(1.0, coverage)
        filled = int(round(bar_w * coverage))
        if filled > 0:
            cv2.rectangle(frame, (bar_x, mid - bar_h // 2),
                          (bar_x + filled, mid + bar_h - bar_h // 2), dim, -1)

        # Right-aligned against the gutter, so 9% and 100% end at the same edge
        # and the second column starts at one x for every row.
        text_mod.draw(frame, f"{coverage * 100:.0f}%", font, size=value_size,
                      xy=(pct_x + pct_w, mid), color=(255, 255, 255), anchor="rm",
                      opacity=opacity)

        # The beta. Dropped from the front when it outgrows the panel: the tail
        # is the part still moving, and the whole list is in summary.txt anyway.
        sequence = sequences.get(limb.key) or []
        if sequence:
            # The clock and the credit sit level with the bottom rows, so the
            # space they occupy is reserved for every row — a column that is
            # rectangular for two rows and not for the others is not a column.
            budget = (frame.shape[1] - right_reserve
                      - _px(cfg.LIMB_PANEL_SEQ_RIGHT_PAD, scale) - col2_x)
            text = " ".join(str(h) for h in sequence)
            while sequence and text_mod.measure(text, font, label_size)[0] > budget:
                sequence = sequence[1:]
                text = "… " + " ".join(str(h) for h in sequence)
            text_mod.draw(frame, text, font, size=label_size, xy=(col2_x, mid),
                          color=cfg.LIMB_COLORS[limb.key], anchor="lm",
                          opacity=1.0 if on else 0.75)
        y += row_h + row_gap
    return y


# ── the completion panel ─────────────────────────────────────────────────────
# What the batch is for. One clip can only ever show you the sequence you are
# watching, and while you are watching it, it looks like *the* sequence. So when
# the route tops out both panels go down behind a scrim and one card comes up
# over the pair: the time, this attempt's holds in order, and underneath, the
# same line for every other attempt at the same problem — with the holds that
# differ picked out, because the differences are the entire point.

def _token_width(tokens, font, size: int) -> int:
    return sum(text_mod.measure(text, font, size)[0] for text, _ in tokens)


def _draw_tokens(frame, tokens, font, *, size: int, xy, opacity: float = 1.0) -> int:
    """Draw ``(text, colour)`` runs left to right from *xy*, returning the width.

    One string per colour rather than one call per sequence, because the holds
    that differ between two attempts have to be the ones that catch the eye and
    a single-colour line hides exactly the information the panel exists for.
    """
    x, y = xy
    for text, color in tokens:
        text_mod.draw(frame, text, font, size=size, xy=(x, y), color=color,
                      anchor="lm", opacity=opacity)
        x += text_mod.measure(text, font, size)[0]
    return x - xy[0]


def _fit_common(token_sets, font, *, size: int, budget: int, min_size: int) -> int:
    """One size at which *every* sequence fits the card.

    Solved across all of them rather than per row, and this is the whole point.
    Fitting each row on its own lets a short sequence keep a large face while a
    long one shrinks to fit, so the two end up about the same width on screen —
    and the card's only claim is that one of those lines is shorter than the
    others. Width has to mean holds and nothing else, which it does exactly when
    the size is shared.

    Shrinks, never grows: the configured size is the target.
    """
    while size > min_size and any(_token_width(t, font, size) > budget
                                  for t in token_sets):
        size -= 1
    return size


def _fit_text(text: str, font, *, size: int, budget: int, floor: float = 0.55) -> int:
    """Largest size up to *size* at which *text* fits *budget*.

    The sequences are fitted as a group; the headline is one string and is
    fitted on its own. Without it a long clock ("Completed in 13.63s!") runs off
    the card on the narrow route-only export, where the card is half the width
    but every size is quoted at the same scale.
    """
    low = max(1, int(size * floor))
    while size > low and text_mod.measure(text, font, size)[0] > budget:
        size -= 1
    return size


def _truncate(tokens, font, *, size: int, budget: int):
    """Drop entries off the end until the row fits, marking it with an ellipsis.

    Off the end, unlike the live panel, which drops from the front: there the
    tail is the part still moving, here the climb is over and it reads from
    hold 1. Only reached when even the minimum size will not fit.
    """
    if _token_width(tokens, font, size) <= budget:
        return tokens
    ellipsis = ("…" if text_mod.has_glyph("…", font) else "...",
                tokens[-1][1] if tokens else (150, 150, 150))
    kept = list(tokens)
    while kept and _token_width(kept + [ellipsis], font, size) > budget:
        kept.pop()
    return kept + [ellipsis]


def _separator(font, cfg) -> str:
    """The first separator this face can actually draw.

    Pillow renders a character the font lacks as .notdef — an empty box — and
    reports a normal width for it, so an arrow that is missing does not fail,
    it just fills the sequence with tofu. Avenir Next, the first face picked on
    macOS, has no U+2192.
    """
    for candidate in (cfg.COMPARE_SEQ_SEPARATOR, *cfg.COMPARE_SEQ_SEPARATOR_FALLBACKS):
        if text_mod.has_glyph(candidate.strip() or " ", font):
            return candidate
    return " "


def _sequence_tokens(sequence, *, unique: set, cfg, separator: str):
    """Two colours: gold for a hold only this row used, grey for the rest.

    Grading the grey by how many attempts shared a hold sounds like more
    information and is not — the card only glosses the gold, so a second shade
    of grey is a distinction the reader is invited to decode and given no key
    for. One grey, and the colour means exactly the one thing it says.
    """
    tokens = []
    for i, hold in enumerate(sequence):
        if i:
            tokens.append((separator, cfg.COMPARE_SEP_COLOR))
        if cfg.COMPARE_HIGHLIGHT_DIFF and hold in unique:
            tokens.append((str(hold), cfg.COMPARE_UNIQUE_COLOR))
        else:
            tokens.append((str(hold), cfg.COMPARE_COMMON_COLOR))
    return tokens


def _draw_tag(frame, text, font, *, size: int, right: int, mid: int, cfg):
    """A filled chip, right-aligned, ending at *right*. Returns its left edge."""
    tw, th = text_mod.measure(text, font, size)
    pad_x, pad_y = max(3, size // 3), max(2, size // 4)
    x1, x0 = right, right - tw - pad_x * 2
    cv2.rectangle(frame, (x0, mid - th // 2 - pad_y), (x1, mid + th - th // 2 + pad_y),
                  cfg.COMPARE_TAG_BG, -1)
    text_mod.draw(frame, text, font, size=size, xy=(x0 + pad_x, mid),
                  color=cfg.COMPARE_TAG_COLOR, anchor="lm")
    return x0


def _draw_legend(frame, *, right: int, top: int, budget: int, cfg, font,
                 scale: float) -> int:
    """What the gold numbers mean, right-aligned. Returns the height drawn.

    The colour carries the comparison — a gold hold is one nobody else used —
    and a reader who does not know that is looking at an arbitrary highlight on
    a list of numbers. Drawn from the right edge inward so it lands in the
    corner the rows leave empty.
    """
    size = _px(cfg.COMPARE_LEGEND_SIZE, scale)
    swatch = _px(cfg.COMPARE_LEGEND_SWATCH, scale)
    gap = _px(cfg.COMPARE_LEGEND_GAP, scale)
    pad = _px(6, scale)

    # Only the gold is glossed. It is the one colour that carries a claim; the
    # greys are just the holds it is being contrasted against, and naming all
    # three turns a one-line gloss into a key the reader has to study.
    entries = ((cfg.COMPARE_UNIQUE_COLOR, cfg.COMPARE_LEGEND_UNIQUE),)

    def span(at: int) -> int:
        return (sum(swatch + pad + text_mod.measure(t, font, at)[0]
                    for _, t in entries) + gap * (len(entries) - 1))

    while size > _px(9, scale) and span(size) > budget:
        size -= 1
    height = max(size, swatch)
    mid = top + height // 2

    widths = [swatch + pad + text_mod.measure(text, font, size)[0]
              for _, text in entries]
    x = right - sum(widths) - gap * (len(entries) - 1)
    for (color, text), width in zip(entries, widths):
        cv2.rectangle(frame, (x, mid - swatch // 2),
                      (x + swatch, mid + swatch - swatch // 2), color, -1)
        text_mod.draw(frame, text, font, size=size, xy=(x + swatch + pad, mid),
                      color=cfg.COMPARE_LEGEND_COLOR, anchor="lm")
        x += width + gap
    return height


def completion_panel(frame, comparison, label: str, *, cfg, font, scale: float,
                     alpha: float = 1.0, region: tuple[int, int] | None = None) -> None:
    """Draw the comparison card over *frame*, faded in at *alpha*.

    Every attempt is drawn to one template — a name, its count and time, then its
    sequence on its own line, all starting at the same x and all set at the same
    size. That is what makes the card readable at a glance: the lines are the
    same kind of thing, so the only difference between them is their length, and
    their length is the number of holds.

    Composed on a copy and blended in one pass, so the fade carries the scrim,
    the card and every glyph at once — fading them individually would have the
    text arrive through a background that was already opaque.
    """
    if alpha <= 0.0 or comparison is None:
        return

    attempt = comparison.by_label(label)
    if attempt is None:
        return
    others = comparison.others(label)
    if not others:
        return          # nothing to compare against; the card would be a caption

    work = frame.copy()
    height, full_width = work.shape[:2]
    # The band the card lives in — the left panel on the paired render, the whole
    # frame on the route-only one.
    rx0, rx1 = region if region else (0, full_width)
    width = rx1 - rx0

    # Everything behind the card goes down, not away: the climber is still there,
    # and the panel reads as laid over the climb rather than as a cut to a slate.
    # Only within the band, so the finished route on the other side stays lit —
    # it is what the numbers on the card are numbers *of*.
    band = work[:, rx0:rx1]
    cv2.convertScaleAbs(band, dst=band, alpha=1.0 - cfg.COMPARE_PANEL_SCRIM)

    pad = _px(cfg.COMPARE_PANEL_PAD, scale)
    gap = _px(cfg.COMPARE_PANEL_ROW_GAP, scale)
    title_size = _px(cfg.COMPARE_TITLE_SIZE, scale)
    subtitle_size = _px(cfg.COMPARE_SUBTITLE_SIZE, scale)
    header_size = _px(cfg.COMPARE_HEADER_SIZE, scale)
    label_size = _px(cfg.COMPARE_LABEL_SIZE, scale)
    min_seq_size = _px(cfg.COMPARE_SEQ_MIN_SIZE, scale)
    meta_size = _px(cfg.COMPARE_META_SIZE, scale)
    tag_size = _px(cfg.COMPARE_TAG_SIZE, scale)

    separator = _separator(font, cfg)
    attempt_gap = _px(cfg.COMPARE_ATTEMPT_GAP, scale)
    seq_gap = max(1, gap // 2)
    section_gap = gap * cfg.COMPARE_SECTION_GAP
    legend_h = (max(_px(cfg.COMPARE_LEGEND_SIZE, scale),
                    _px(cfg.COMPARE_LEGEND_SWATCH, scale)) + section_gap
                if cfg.COMPARE_LEGEND else 0)

    card_w = int(width * cfg.COMPARE_PANEL_WIDTH)
    content_w = card_w - pad * 2

    shortest = comparison.shortest()
    fastest = comparison.fastest()

    def unique_to(one):
        """Holds no other attempt used. The news on every row."""
        return {h for h in one.aligned_sequence
                if all(h not in other.aligned_sequence
                       for other in comparison.attempts if other is not one)}

    def tag_for(one):
        if shortest and fastest and shortest.label == fastest.label == one.label:
            return cfg.COMPARE_BOTH_TAG
        if shortest and one.label == shortest.label:
            return cfg.COMPARE_SHORTEST_TAG
        if fastest and one.label == fastest.label:
            return cfg.COMPARE_FASTEST_TAG
        return None

    def meta_for(one):
        # Keyed on whether it topped out, not on whether there happens to be a
        # time: an attempt that bailed used fewer holds by not finishing, and a
        # row that quietly reports its clock reads as a send that beat this one.
        text = f"{len(one.aligned_sequence)} holds"
        if not one.topped_out:
            return text + "  ·  did not top out"
        if one.elapsed is not None:
            return text + f"  ·  {one.elapsed:.2f}s"
        return text

    # Every row's tokens up front, so one size can be solved for all of them.
    rows = [(attempt, True)] + [(one, False) for one in others]
    tokens_for = {
        id(one): _sequence_tokens(
            one.aligned_sequence, unique=unique_to(one), cfg=cfg,
            separator=separator)
        for one, _ in rows
    }
    # The tag sits on the name line, not the sequence line — on the sequence line
    # it would steal width from one row and not the others, which is the same
    # distortion as a per-row font size, just in a different currency.
    seq_size = _fit_common(list(tokens_for.values()), font,
                           size=_px(cfg.COMPARE_SEQ_SIZE, scale),
                           budget=max(content_w, 1), min_size=min_seq_size)

    chip_h = tag_size + _px(8, scale)
    head_h = max(label_size, meta_size, chip_h)
    row_h = head_h + seq_gap + seq_size

    card_h = (pad * 2 + title_size + gap + subtitle_size + section_gap
              + header_size + gap + row_h + section_gap
              + header_size + gap
              + (row_h + attempt_gap) * len(others) - attempt_gap
              + legend_h)
    card_h = min(card_h, height - _px(12, scale) * 2)

    x0 = rx0 + (width - card_w) // 2
    y0 = (height - card_h) // 2
    cv2.rectangle(work, (x0, y0), (x0 + card_w, y0 + card_h), cfg.COMPARE_PANEL_BG, -1)
    cv2.rectangle(work, (x0, y0), (x0 + card_w, y0 + card_h),
                  cfg.COMPARE_PANEL_BORDER, max(1, _px(1, scale)), cv2.LINE_AA)

    left = x0 + pad
    right = x0 + card_w - pad

    def draw_row(y: int, one, *, highlight: bool) -> int:
        head_mid = y + head_h // 2
        text_mod.draw(work, one.display(cfg.COMPARE_ATTEMPT_LABEL), font,
                      size=label_size, xy=(left, head_mid),
                      color=cfg.COMPARE_TITLE_COLOR if highlight
                      else cfg.COMPARE_LABEL_COLOR, anchor="lm")

        meta = meta_for(one)
        text_mod.draw(work, meta, font, size=meta_size, xy=(right, head_mid),
                      color=cfg.COMPARE_META_COLOR, anchor="rm")

        tag = tag_for(one)
        if tag:
            _draw_tag(work, tag, font, size=tag_size,
                      right=right - text_mod.measure(meta, font, meta_size)[0]
                      - _px(18, scale), mid=head_mid, cfg=cfg)

        y += head_h + seq_gap
        tokens = _truncate(tokens_for[id(one)], font, size=seq_size,
                           budget=max(content_w, 1))
        _draw_tokens(work, tokens, font, size=seq_size, xy=(left, y + seq_size // 2))
        return y + seq_size

    # ── the headline ────────────────────────────────────────────────────────
    y = y0 + pad
    elapsed = attempt.elapsed
    title = (cfg.COMPARE_TITLE.format(elapsed=elapsed) if elapsed is not None
             else "COMPLETED!")
    title_size = _fit_text(title, font, size=title_size, budget=content_w)
    text_mod.draw(work, title, font, size=title_size, xy=(left, y),
                  color=cfg.COMPARE_TITLE_COLOR, anchor="lt")
    y += title_size + gap

    parts = [str(comparison.route.get("color") or "").strip()]
    if cfg.COMPARE_PANEL_SHOW_GRADE and comparison.route.get("grade"):
        parts.append(str(comparison.route["grade"]))
    if comparison.route.get("name"):
        parts.append(str(comparison.route["name"]))
    parts.append(f"{attempt.n_detected} holds on the route")
    parts.append(f"{len(comparison.attempts)} attempts")
    subtitle = "  ·  ".join(p for p in parts if p)
    subtitle_size = _fit_text(subtitle, font, size=subtitle_size, budget=content_w)
    text_mod.draw(work, subtitle, font, size=subtitle_size,
                  xy=(left, y), color=cfg.COMPARE_SUBTITLE_COLOR, anchor="lt")
    y += subtitle_size + section_gap

    # ── this climb ──────────────────────────────────────────────────────────
    text_mod.draw(work, cfg.COMPARE_THIS_HEADER, font, size=header_size, xy=(left, y),
                  color=cfg.COMPARE_HEADER_COLOR, anchor="lt")
    y += header_size + gap
    y = draw_row(y, attempt, highlight=True)
    y += section_gap

    # ── the other attempts ──────────────────────────────────────────────────
    heading, caption = cfg.COMPARE_OTHERS_HEADER, "same route"
    text_mod.draw(work, heading, font, size=header_size, xy=(left, y),
                  color=cfg.COMPARE_HEADER_COLOR, anchor="lt")
    # A gloss on the heading, so it is set at the row scale and sits on the
    # heading's baseline — at the heading's own size it reads as a second
    # heading arguing with the first. Dropped rather than crowded when the card
    # is too narrow to hold both.
    if (text_mod.measure(heading, font, header_size)[0]
            + text_mod.measure(caption, font, meta_size)[0]
            + _px(24, scale)) <= content_w:
        text_mod.draw(work, caption, font, size=meta_size,
                      xy=(right, y + header_size), color=cfg.COMPARE_META_COLOR,
                      anchor="rb")
    y += header_size + gap

    for one in others:
        if y + row_h > y0 + card_h - pad - legend_h + attempt_gap:
            break        # more attempts than the card can hold; sequences.txt has them all
        y = draw_row(y, one, highlight=False) + attempt_gap

    if cfg.COMPARE_LEGEND:
        _draw_legend(work, right=right, top=y0 + card_h - pad - legend_h + section_gap,
                     budget=content_w, cfg=cfg, font=font, scale=scale)

    cv2.addWeighted(work, alpha, frame, 1.0 - alpha, 0.0, dst=frame)


def holds_preview(frame, holds, color, *, scale: float = 1.0):
    """A still of the detected route, for checking the prompt found the right wall."""
    height, width = frame.shape[:2]
    out = frame.copy()
    thickness = _px(2, scale)
    for hold in holds:
        draw_hold(out, hold, width, height, color, thickness, fill=True, alpha=0.35,
                  dashed=hold.get("recovered", False))
        x, y, w, h = hold["bbox"]
        cv2.putText(out, str(hold["id"]),
                    (int(x * width), max(12, int(y * height) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4 * scale, color, thickness, cv2.LINE_AA)
    return out


def render(info, holds, poses, analysis, route_points, out_path: Path, *,
           cfg, console, camera, wall, fit, frame_poses=None, colour=None,
           right_path: Path | None = None, ground=None,
           comparison=None, label: str | None = None) -> dict:
    """Write the paired panels (and optionally the right one alone) as raw MP4.

    OpenCV's writer emits MPEG-4 Part 2; `video.encode_h264` is what makes the
    result playable. Kept separate so a killed render leaves an intermediate,
    not a broken deliverable.

    With a *comparison* and a *label*, the render ends on the completion panel:
    the card comes up when the route tops out, and the last frame is held past
    the end of the source for long enough to read it.

    The two panels now live in two different coordinate systems, which is the
    whole shape of the handheld version:

    * **left** is the clip as shot, so the route has to be projected *into* each
      frame through that frame's homography — the holds move because the camera
      does, and they have to move with the wall exactly.
    * **right** is the canvas: the wall mosaic, holding still, with the route on
      it. Nothing here moves at all, which is the point of it.

    ``holds``, ``poses``, ``analysis`` and ``route_points`` are all in canvas
    coordinates. ``frame_poses`` is the same climber back in frame coordinates,
    for drawing the skeleton on the left; it is passed rather than un-projected
    here because the un-projection is lossy at the frame edges and the caller
    already has the original.
    """
    width, height = info.width, info.height
    # Every size in config is quoted at the inference width, so an export at a
    # different resolution scales rather than being re-tuned.
    scale = width / 608.0

    font = text_mod.resolve_font(cfg.PANEL_FONT, cfg.PANEL_FONT_INDEX)
    edges, points = skeleton.visible_parts(cfg.DRAW_FACE)
    hold_color = cfg.HOLD_RENDER_COLOR.get(colour or cfg.HOLD_COLOR,
                                           cfg.HOLD_COLOR_FALLBACK)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    pair = cv2.VideoWriter(str(out_path), fourcc, info.fps, (width * 2, height))
    solo = cv2.VideoWriter(str(right_path), fourcc, info.fps, (width, height)) \
        if right_path else None

    cap = cv2.VideoCapture(str(info.path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {info.path}")

    # The route as the right panel will draw it: canvas coordinates placed into
    # the panel's box. Computed once — the panel does not move.
    panel_holds = [fit.hold(h) for h in holds]
    centroids = {h["id"]: (int((h["bbox"][0] + h["bbox"][2] / 2) * width),
                           int((h["bbox"][1] + h["bbox"][3] / 2) * height))
                 for h in panel_holds}
    panel_route = ([tuple(p) for p in
                    fit.point(np.asarray(route_points, dtype=float)) * [width, height]]
                   if route_points else [])
    backdrop = (fit.backdrop(wall, dim=cfg.ROUTE_BACKDROP_DIM)
                if wall is not None and cfg.ROUTE_BACKDROP else
                np.zeros((height, width, 3), dtype=np.uint8))
    viewport_quad = None

    inactive_thick = _px(cfg.HOLD_INACTIVE_THICK, scale)
    active_thick = _px(cfg.HOLD_ACTIVE_THICK, scale)
    outline_thick = _px(cfg.HOLD_OUTLINE_THICK, scale)
    dot_radius = _px(cfg.MIDLINE_DOT_RADIUS, scale)
    margin = _px(cfg.PANEL_MARGIN, scale)
    gap = _px(cfg.TIMER_CREDIT_GAP, scale)
    label_size = _px(cfg.HOLD_LABEL_SIZE, scale)

    pip_radius = _px(cfg.LIMB_PIP_RADIUS, scale)
    pip_gap = _px(cfg.LIMB_PIP_GAP, scale)
    pip_offset = _px(cfg.LIMB_PIP_OFFSET, scale)

    # Per-hold limb usage, and when each (limb, hold) pair was first used, so a
    # pip appears at the moment that limb arrives rather than from frame one.
    limb_keys_by_hold: dict[int, list[str]] = {}
    first_use: dict[tuple[str, int], int] = {}
    win_lo, win_hi = analysis.window if analysis.window else (-10**9, 10**9)
    for contact in analysis.contacts:
        start, end = max(contact.start, win_lo), min(contact.end, win_hi)
        if end < start:
            continue          # warm-up only; the report does not count it either
        limb_keys_by_hold.setdefault(contact.hold_id, [])
        if contact.limb not in limb_keys_by_hold[contact.hold_id]:
            limb_keys_by_hold[contact.hold_id].append(contact.limb)
        key = (contact.limb, contact.hold_id)
        first_use[key] = min(first_use.get(key, start), start)
    # Keep pips in the panel's own limb order, not in arrival order, so a hold's
    # dots mean the same thing everywhere on the wall.
    order_index = {limb.key: i for i, limb in enumerate(climb.LIMBS)}
    for keys in limb_keys_by_hold.values():
        keys.sort(key=lambda k: order_index[k])

    running = climb.running_utilization(analysis, sorted(analysis.holding))
    sequences = climb.running_sequence(analysis, sorted(analysis.holding))

    # Widest the bottom-right stack will ever get: the finished clock and the
    # credit. Measured once so the sequence column can steer clear of both.
    corner_reserve = 0
    if cfg.TIMER or cfg.CREDIT_TEXT:
        widths = [text_mod.measure("00:00 s", font, _px(cfg.TIMER_SIZE, scale))[0]]
        if cfg.CREDIT_TEXT:
            widths.append(text_mod.measure(cfg.CREDIT_TEXT, font,
                                           _px(cfg.CREDIT_SIZE, scale))[0])
        corner_reserve = max(widths) + margin + _px(16, scale)

    # The completion panel, and the tail that makes it readable. The clock stops
    # on the top-out but the clip keeps rolling for however long it takes to
    # climb down — on some of these takes, barely at all. So the last composed
    # frame is held past the end of the source rather than the comparison
    # flashing past in the six frames the clip happened to have left.
    panel_at = None
    if (comparison is not None and label is not None and cfg.COMPARE_PANEL
            and analysis.completion_frame is not None
            and comparison.others(label)):
        panel_at = analysis.completion_frame
    fade_frames = max(1, int(round(cfg.COMPARE_PANEL_FADE_SECONDS * info.fps)))
    tail_frames = (max(0, int(round(cfg.COMPARE_PANEL_HOLD_SECONDS * info.fps)))
                   if panel_at is not None else 0)

    # On the pair, the card is confined to the left panel; on the route-only
    # export there is no left panel and it takes the frame.
    panel_region = (0, width) if cfg.COMPARE_PANEL_ON_LEFT else (0, width * 2)

    def panel_alpha(index: int) -> float:
        if panel_at is None or index < panel_at:
            return 0.0
        return min(1.0, (index - panel_at + 1) / fade_frames)

    trail: list[tuple[int, int]] = []
    stats = {"frames_written": 0, "frames_with_pose": 0, "tail_frames": 0,
             "completion_panel": panel_at is not None}
    frame_index = 0
    last_running = {limb.key: 0 for limb in climb.LIMBS}
    tail_left = tail_right = None

    columns = [TextColumn("[cyan]rendering[/]"), BarColumn(), TaskProgressColumn(),
               TextColumn("{task.completed}/{task.total} frames"), TimeElapsedColumn()]

    with Progress(*columns, console=console, transient=True) as progress:
        task = progress.add_task("render", total=info.n_frames or None)
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # ── left: the clip, with what the models saw on it ───────────────
            left = frame
            if ground is not None and cfg.DRAW_FLOOR_LINE and frame_index < camera.n_frames:
                # The ground line lives on the canvas now, so it comes back into
                # the frame the same way the holds do.
                floor_mod.draw(left, ground, color=cfg.FLOOR_LINE_COLOR,
                               thickness=_px(cfg.FLOOR_LINE_THICK, scale),
                               transform=lambda pts, i=frame_index: camera.to_frame(pts, i))
            # The route projected into *this* frame. The holds are stored once,
            # on the canvas, and this is the only place they become pixels on
            # the moving image — so the outlines track the wall through every
            # pan and zoom without ever having been detected in this frame.
            live = ([{**hold,
                      "polygon": holds_mod.in_frame(hold, camera, frame_index).tolist()}
                     for hold in holds] if frame_index < camera.n_frames else [])
            for hold in live:
                draw_hold(left, hold, width, height, hold_color, outline_thick,
                          fill=True, alpha=cfg.HOLD_FILL_ALPHA,
                          dashed=cfg.RECOVERED_DASH and hold.get("recovered", False))

            shown = frame_poses if frame_poses is not None else poses
            has_pose = frame_index in shown.kpts
            if has_pose:
                stats["frames_with_pose"] += 1
                skeleton.draw_person(
                    left, shown.kpts[frame_index], shown.valid[frame_index],
                    width=width, height=height, edges=edges, points=points,
                    colors=cfg.SKELETON_COLORS,
                    thickness=_px(cfg.LINE_THICKNESS, scale),
                    radius=_px(cfg.POINT_RADIUS, scale),
                    bbox=shown.bboxes[frame_index] if cfg.DRAW_PERSON_BBOX else None,
                    bbox_color=cfg.PERSON_BBOX_COLOR,
                    bbox_thickness=_px(cfg.PERSON_BBOX_THICK, scale))

            # ── right: the route, with the climber taken away ────────────────
            right = backdrop.copy()

            # Where the live shot is looking, drawn on the canvas. Without it the
            # two panels are hard to relate once the camera has zoomed: the left
            # is a detail of the right and nothing says which detail.
            if cfg.DRAW_VIEWPORT and frame_index < camera.n_frames:
                viewport_quad = np.round(
                    fit.point(camera.coverage(frame_index)) * [width, height]
                ).astype(np.int32)
                cv2.polylines(right, [viewport_quad], True, cfg.VIEWPORT_COLOR,
                              _px(cfg.VIEWPORT_THICK, scale), cv2.LINE_AA)

            # After topping out the panel holds its final state, so the last
            # frames read as a finished route rather than continuing to update.
            effective = (min(frame_index, analysis.completion_frame)
                         if analysis.completion_frame is not None else frame_index)
            topped = (analysis.completion_frame is not None
                      and frame_index >= analysis.completion_frame)

            now_holding = analysis.holding.get(effective, {})
            last_running = running.get(effective, last_running)

            # Every hold carries its number from the first frame, lit or not.
            # The numbering is a property of the wall, so a hold nobody touched
            # is still hold 3 — and "you could have used 3" needs 3 to be on
            # screen, not inferred from the gap between 2 and 4.
            for hold in panel_holds:
                hid = hold["id"]
                if hid not in analysis.activated_at or analysis.activated_at[hid] > effective:
                    draw_hold(right, hold, width, height, cfg.HOLD_INACTIVE_COLOR,
                              inactive_thick, fill=False, alpha=0.0,
                              dashed=cfg.RECOVERED_DASH and hold.get("recovered", False))
                    if cfg.HOLD_LABEL == "id":
                        text_mod.draw(right, str(hid), font, size=label_size,
                                      xy=(int(hold["bbox"][0] * width),
                                          int(hold["bbox"][1] * height) - _px(2, scale)),
                                      color=cfg.HOLD_INACTIVE_LABEL_COLOR, anchor="lb")

            if topped and panel_route and cfg.ROUTE_SPLINE:
                draw_dashed_spline(right, panel_route, cfg.ROUTE_SPLINE_COLOR,
                                   _px(cfg.ROUTE_SPLINE_THICKNESS, scale),
                                   _px(cfg.ROUTE_SPLINE_DASH, scale),
                                   _px(cfg.ROUTE_SPLINE_GAP, scale))

            active = sorted(
                (h for h in panel_holds
                 if h["id"] in analysis.activated_at
                 and analysis.activated_at[h["id"]] <= effective),
                key=lambda h: analysis.order[h["id"]])

            for hold in active:
                hid = hold["id"]
                done = topped and hid == analysis.final_hold_id
                color = cfg.HOLD_COMPLETE_COLOR if done else hold_color
                alpha = cfg.HOLD_COMPLETE_ALPHA if done else cfg.HOLD_ACTIVE_FILL_ALPHA
                draw_hold(right, hold, width, height, color, active_thick,
                          fill=True, alpha=alpha,
                          dashed=cfg.RECOVERED_DASH and hold.get("recovered", False))
                cv2.circle(right, centroids[hid], _px(3, scale), color, -1, cv2.LINE_AA)

                if cfg.HOLD_LABEL != "none":
                    # `chip`, not `label`: the render's own `label` is the clip
                    # this attempt belongs to, and the completion panel needs it.
                    chip = str(hid if cfg.HOLD_LABEL == "id" else analysis.order[hid])
                    tw, th = text_mod.measure(chip, font, label_size)
                    x1 = int(hold["bbox"][0] * width)
                    y1 = int(hold["bbox"][1] * height)
                    pad = _px(3, scale)
                    cv2.rectangle(right, (x1, max(0, y1 - th - pad * 2)),
                                  (x1 + tw + pad * 2, y1), color, -1)
                    text_mod.draw(right, chip, font, size=label_size,
                                  xy=(x1 + pad, max(0, y1 - th - pad)),
                                  color=cfg.HOLD_LABEL_COLOR, anchor="lt")

                if cfg.LIMB_PIPS:
                    # Which limbs have used this hold *by now* — the panel should
                    # not show a foot that has not arrived yet.
                    used = [k for k in limb_keys_by_hold.get(hid, ())
                            if first_use.get((k, hid), 10**9) <= effective]
                    draw_limb_pips(right, hold, width, height, used,
                                   colors=cfg.LIMB_COLORS, radius=pip_radius,
                                   gap=pip_gap, offset=pip_offset,
                                   active={k for k, v in now_holding.items() if v == hid})

            # The midline: where the body is, and the last second of where it was.
            if analysis.start_frame is not None and frame_index >= analysis.start_frame:
                position = analysis.midline.get(frame_index)
                if position is not None:
                    # The midline is on the canvas, so it is where the body is
                    # on the *wall*: a climber holding a rest stays put on the
                    # trail even while the camera drifts around them.
                    px, py = fit.point(np.array([position]))[0]
                    trail.append((int(px * width), int(py * height)))
                if len(trail) > cfg.MIDLINE_TRAIL_FRAMES:
                    trail.pop(0)

            n = len(trail)
            for i, (tx, ty) in enumerate(trail[:-1]):
                frac = (i + 1) / max(n, 1)
                color = tuple(int(cfg.MIDLINE_TRAIL_DIM[c]
                                  + (cfg.MIDLINE_DOT_COLOR[c] - cfg.MIDLINE_TRAIL_DIM[c]) * frac)
                              for c in range(3))
                cv2.circle(right, (tx, ty), max(1, int(dot_radius * frac * 0.7)),
                           color, -1, cv2.LINE_AA)
            if trail:
                cv2.circle(right, trail[-1], dot_radius, cfg.MIDLINE_DOT_COLOR, -1, cv2.LINE_AA)
                cv2.circle(right, trail[-1], dot_radius + _px(2, scale),
                           (255, 255, 255), 1, cv2.LINE_AA)

            # Bottom-left: the running contact split.
            if cfg.LIMB_PANEL:
                panel_h = (_px(cfg.LIMB_PANEL_TITLE_SIZE, scale)
                           + _px(cfg.LIMB_PANEL_ROW_GAP, scale) * 2
                           + 4 * (max(_px(cfg.LIMB_PANEL_SWATCH, scale),
                                      _px(cfg.LIMB_PANEL_BAR_HEIGHT, scale),
                                      _px(cfg.LIMB_PANEL_VALUE_SIZE, scale))
                                  + _px(cfg.LIMB_PANEL_ROW_GAP, scale)))
                start = analysis.window[0] if analysis.window else 0
                draw_limb_panel(right, climb.LIMBS, last_running, now_holding,
                                sequences.get(effective, {}),
                                max(0, effective - start + 1),
                                origin=(margin, height - margin - panel_h),
                                cfg=cfg, font=font, scale=scale, _px=_px,
                                right_reserve=corner_reserve)

            # Bottom-right: the clock over the credit.
            bottom = height - margin
            if cfg.CREDIT_TEXT:
                size = _px(cfg.CREDIT_SIZE, scale)
                text_mod.draw(right, cfg.CREDIT_TEXT, font, size=size,
                              xy=(width - margin, bottom), color=cfg.CREDIT_COLOR,
                              anchor="rb")
                bottom -= size + gap

            if cfg.TIMER and analysis.start_frame is not None \
                    and frame_index >= analysis.start_frame:
                if analysis.completion_frame is not None \
                        and frame_index >= analysis.completion_frame:
                    elapsed = analysis.elapsed
                else:
                    elapsed = (frame_index - analysis.start_frame) / info.fps
                text_mod.draw(right, f"{int(elapsed):02d}:{int(elapsed % 1 * 100):02d} s",
                              font, size=_px(cfg.TIMER_SIZE, scale),
                              xy=(width - margin, bottom), color=cfg.TIMER_COLOR,
                              anchor="rb")

            # Hold the pristine panels before the card goes over them, so the
            # held tail can be composed at a fade the source frames never saw.
            if tail_frames and panel_at is not None and frame_index >= panel_at:
                tail_left, tail_right = left.copy(), right.copy()

            alpha = panel_alpha(frame_index)
            combined = np.hstack([left, right])
            if alpha > 0.0:
                # The left half only: the right panel is the finished route the
                # card's numbers refer to, and covering it would leave the
                # sequence pointing at nothing.
                completion_panel(combined, comparison, label, cfg=cfg, font=font,
                                 scale=scale, alpha=alpha, region=panel_region)
            pair.write(combined)
            if solo is not None:
                if alpha > 0.0 and cfg.COMPARE_PANEL_ON_ROUTE:
                    completion_panel(right, comparison, label, cfg=cfg, font=font,
                                     scale=scale, alpha=alpha)
                solo.write(right)

            stats["frames_written"] += 1
            frame_index += 1
            progress.update(task, completed=frame_index)

        # The tail: the same last frame, recomposed each time so the card can
        # still be fading up if the clip ran out mid-fade. Once the fade is done
        # every remaining frame is identical, so the first finished one is kept
        # and written again — the alternative is laying out the same card a few
        # hundred more times to get the same pixels. VideoWriter.write does not
        # touch the array it is handed, so one buffer serves them all.
        if tail_frames and tail_left is not None:
            done_pair = done_solo = None
            for i in range(tail_frames):
                alpha = panel_alpha(frame_index + i)
                if done_pair is not None:
                    pair.write(done_pair)
                    if solo is not None:
                        solo.write(done_solo)
                    continue

                combined = np.hstack([tail_left, tail_right])
                completion_panel(combined, comparison, label, cfg=cfg, font=font,
                                 scale=scale, alpha=alpha, region=panel_region)
                pair.write(combined)
                held = None
                if solo is not None:
                    held = tail_right.copy()
                    if cfg.COMPARE_PANEL_ON_ROUTE:
                        completion_panel(held, comparison, label, cfg=cfg, font=font,
                                         scale=scale, alpha=alpha)
                    solo.write(held)
                if alpha >= 1.0:
                    done_pair, done_solo = combined, held
            stats["tail_frames"] = tail_frames
            stats["frames_written"] += tail_frames

    cap.release()
    pair.release()
    if solo is not None:
        solo.release()
    return stats
