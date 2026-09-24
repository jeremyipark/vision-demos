"""The two panels.

Left is the clip with what the models saw drawn onto it: every hold on the route
outlined and filled, the climber's skeleton over the top. Right is the same wall
with the climber taken away — the holds start dim, light up in the order they are
used and carry that order as a number, and the body is reduced to the one dot the
whole climb is measured from. The clock runs from pulling on, and stops on the top.

Taking the climber away is the point of the right panel: the left one shows that
the models work, and the right one shows what they *found*, which is a route.

The two panels share one world, the one the depth measured. The left panel is
the camera's view, so each 3D hold is projected into every frame through that
frame's pose and rides the pan. The right panel is a virtual camera in the
same world, looking at the fused wall — chasing the climber up it, or orbiting
it — and the route is re-projected into that view every frame.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from . import climb, skeleton, space as space_mod, text as text_mod


def hold_label(hold_id: int, cfg=None) -> str:
    """How a hold is named on screen: A, B, C … by default, or its number.

    Letters, because the finale numbers the holds 1, 2, 3 in the order they
    were *used*, and a wall already numbered 1-16 bottom to top would put two
    different "3"s on the screen at once. Hold 1 is A, 26 is Z, 27 is AA. The
    id underneath is unchanged, so every artifact still joins on it.
    """
    if cfg is not None and getattr(cfg, "HOLD_LABEL_STYLE", "letter") != "letter":
        return str(hold_id)
    n, out = int(hold_id), ""
    while n > 0:
        n, rem = divmod(n - 1, 26)
        out = chr(65 + rem) + out
    return out or "?"


def credit_lines(cfg) -> list[tuple[str, str]]:
    """The credit as ``[(label, value), ...]``, top line first. Empty for none.

    Reads ``CREDIT`` — each entry a ``(label, value)`` pair or a plain string —
    and falls back to the older single ``CREDIT_TEXT`` string if that is all a
    config has. Blank entries are skipped, so a template with an unfilled
    handle does not leave a gap.
    """
    entries = getattr(cfg, "CREDIT", None)
    if entries is None:
        text = getattr(cfg, "CREDIT_TEXT", None)
        entries = [text] if text else []
    if isinstance(entries, (str, tuple)):
        entries = [entries]
    lines = []
    for entry in entries:
        if isinstance(entry, (tuple, list)):
            label, value = (entry[0], entry[1]) if len(entry) >= 2 else ("", entry[0])
        else:
            label, value = "", entry
        label, value = str(label or "").strip(), str(value or "").strip()
        if value:
            lines.append((label, value))
    return lines


def _credit_parts(label: str, value: str) -> tuple[str, str]:
    return (f"{label}: " if label else "", value)


def measure_credit(cfg, font, scale: float) -> tuple[int, int]:
    """``(width, height)`` of the whole credit stack, in pixels."""
    size = _px(cfg.CREDIT_SIZE, scale)
    lines = credit_lines(cfg)
    if not lines:
        return 0, 0
    widths = [text_mod.measure(a + b, font, size)[0]
              for a, b in (_credit_parts(*line) for line in lines)]
    gap = _px(getattr(cfg, "CREDIT_LINE_GAP", 5), scale)
    return max(widths), len(lines) * size + (len(lines) - 1) * gap


def draw_credit(img, cfg, font, scale: float, *, right: int, bottom: int) -> int:
    """Draw the credit right-aligned with its last line on *bottom*.

    Returns the y the next thing up (the clock) should sit on: the top of the
    stack less the usual gap, or *bottom* unchanged when there is no credit.
    """
    lines = credit_lines(cfg)
    if not lines:
        return bottom
    size = _px(cfg.CREDIT_SIZE, scale)
    gap = _px(getattr(cfg, "CREDIT_LINE_GAP", 5), scale)
    label_colour = getattr(cfg, "CREDIT_LABEL_COLOR", cfg.CREDIT_COLOR)
    y = bottom
    for label, value in reversed(lines):
        prefix, text = _credit_parts(label, value)
        # Both parts placed from one ascender line, so they share a baseline;
        # the line as a whole sits with its lowest ink on *y*.
        parts = [(text, cfg.CREDIT_COLOR)] + ([(prefix, label_colour)] if prefix else [])
        boxes = [text_mod.ink_box(s, font, size) for s, _ in parts]
        ascender = y - max(b[3] for b in boxes)
        x = right
        for (s, colour), (l, t, r_, b) in zip(parts, boxes):
            text_mod.draw(img, s, font, size=size, xy=(x, ascender + t), color=colour,
                          anchor="rt")
            x -= r_ - l + (1 if s == text else 0)
        y -= size + gap
    return y + gap - _px(cfg.TIMER_CREDIT_GAP, scale)


def format_clock(seconds: float) -> str:
    """The clock as a stopwatch writes it: ``16.63 s``, or ``1:05.42`` past a minute.

    Rounded to the hundredth, not truncated, so it agrees with the time the
    run reports: ``int(16.63 % 1 * 100)`` is 62, because 16.63 is stored as
    16.6299…. A colon only ever separates minutes from seconds.
    """
    hundredths = int(round(seconds * 100))
    minutes, rest = divmod(hundredths, 6000)
    if minutes:
        return f"{minutes}:{rest // 100:02d}.{rest % 100:02d}"
    return f"{rest // 100}.{rest % 100:02d} s"


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
            text = " ".join(hold_label(h, cfg) for h in sequence)
            while sequence and text_mod.measure(text, font, label_size)[0] > budget:
                sequence = sequence[1:]
                text = "… " + " ".join(hold_label(h, cfg) for h in sequence)
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


def render(info, poses, analysis, out_path: Path, *, cfg, console, scene,
           frame_poses=None, colour=None, frame_range=None,
           right_path: Path | None = None, ground=None,
           comparison=None, label: str | None = None) -> dict:
    """Write the paired panels (and optionally the right one alone) as raw MP4.

    OpenCV's writer emits MPEG-4 Part 2; `video.encode_h264` is what makes the
    result playable. Kept separate so a killed render leaves an intermediate,
    not a broken deliverable.

    With a *comparison* and a *label*, the render ends on the completion panel:
    the card comes up when the route tops out, and the last frame is held past
    the end of the source for long enough to read it.

    * **left** is the clip as shot, with each 3D hold projected into the frame
      through that frame's camera pose — the holds move because the camera does,
      and they move with the wall exactly.
    * **right** is the fused wall seen from a virtual camera
      (``SPACE_VIEW_LIDAR``): a chase camera that follows the climber's torso,
      or an orbit around the route. The holds light up as they are used.

    ``frame_poses`` is the climber as ViTPose returned them, drawn on the left.
    """
    width, height = info.width, info.height
    # Every size in config is quoted at the inference width, so an export at a
    # different resolution scales rather than being re-tuned.
    scale = width / 608.0

    font = text_mod.resolve_font(cfg.PANEL_FONT, cfg.PANEL_FONT_INDEX)
    console.print(f"  [dim]panel font: {font[2] if font else 'OpenCV Hershey (no TrueType face found)'}[/]")
    edges, points = skeleton.visible_parts(cfg.DRAW_FACE)
    body_colors = skeleton.palette(cfg.SKELETON_COLORS, cfg.LIMB_COLORS,
                                   by_limb=cfg.SKELETON_BY_LIMB)
    hold_color = cfg.HOLD_RENDER_COLOR.get(colour or cfg.HOLD_COLOR,
                                           cfg.HOLD_COLOR_FALLBACK)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    pair = cv2.VideoWriter(str(out_path), fourcc, info.fps, (width * 2, height))
    solo = cv2.VideoWriter(str(right_path), fourcc, info.fps, (width, height)) \
        if right_path else None

    cap = cv2.VideoCapture(str(info.path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {info.path}")

    # The odometry may not place every frame, and a frame with no camera
    # pose has no route — not a route in the wrong place, none at all. Rendering
    # those frames produces a video that is mostly two dead panels, so the export
    # is trimmed to the span that has something to show.
    first_frame, last_frame = frame_range if frame_range else (0, info.n_frames - 1)
    if first_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(first_frame))

    # The right panel is a virtual camera in the reconstruction, so the holds are
    # re-projected every frame and their panel positions come back from the
    # projection rather than being computed here. See src/space.py.
    space_centroids: dict[int, tuple[int, int]] = {}
    panel_holds: list[dict] = []
    centroids: dict[int, tuple[int, int]] = {}
    blank = np.zeros((height, width, 3), dtype=np.uint8)

    chase = (space_mod.chase_path(
                 scene, range(first_frame, last_frame + 1), cfg,
                 hold_from=(analysis.completion_frame + int(round(info.fps))
                            if analysis.completion_frame is not None else None))
             if cfg.SPACE_VIEW_LIDAR == "chase" else None)
    if chase:
        az = np.degrees([v[1] for v in chase.values()])
        heights = [v[0][1] for v in chase.values()]
        console.print(f"  [dim]chase camera: azimuth {az.min():+.0f}° to {az.max():+.0f}°, "
                      f"steadiest step {np.abs(np.diff(az)).max():.2f}°/frame; follows the "
                      f"torso {min(heights):+.2f} to {max(heights):+.2f} m up the wall[/]")
    torso_path: list[tuple[int, np.ndarray]] = []   # (frame, body centre in the world)
    # Everything that does not change per frame, done once: the wall in route
    # axes, its colours dimmed so the route is the subject.
    dense_local = scene["axes"].to_wall(scene["recon"].points)
    dense_colours = (scene["colours"].astype(np.float32)
                     * (1.0 - cfg.SPACE_CLOUD_DIM)).astype(np.uint8)
    # Framed on the route plus the body's reach, not on the whole cloud: the
    # cloud runs to the mats and the side walls, and filling the panel with
    # those leaves the route small in the middle of it.
    frame_on = [scene["axes"].to_wall(h.polygon) for h in scene["holds"].values()]
    for world, ok in list(scene.get("skeleton", {}).values())[::15]:
        frame_on.append(scene["axes"].to_wall(world[ok]))
    dense_outline = np.vstack(frame_on)
    point_size = _px(cfg.SPACE_POINT_SIZE, scale)
    space_colors = {"active": hold_color, "inactive": cfg.HOLD_INACTIVE_COLOR,
                    # Over a wall, not over black: the grey that reads as
                    # "unused" on black vanishes against plywood.
                    "inactive_dense": (200, 200, 200)}

    inactive_thick = _px(cfg.HOLD_INACTIVE_THICK, scale)
    active_thick = _px(cfg.HOLD_ACTIVE_THICK, scale)
    outline_thick = _px(cfg.HOLD_OUTLINE_THICK, scale)
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
    if cfg.TIMER or credit_lines(cfg):
        widths = [text_mod.measure(format_clock(59.99), font, _px(cfg.TIMER_SIZE, scale))[0]]
        if credit_lines(cfg):
            widths.append(measure_credit(cfg, font, scale)[0])
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

    # The finale: see `render_finale`. It replaces the climb-down, so the clip
    # itself stops where the chase camera stops following.
    finale_at = None
    # Chase only: the finale starts from where the chase camera left off, and
    # stopping the clip early without one to follow would just cut it short.
    if (cfg.FINALE and chase and analysis.completion_frame is not None
            and panel_at is None):
        finale_at = min(last_frame, analysis.completion_frame
                        + int(round(cfg.FINALE_START_AFTER_S * info.fps)))
        last_frame = finale_at
    finale_start = None

    def panel_alpha(index: int) -> float:
        if panel_at is None or index < panel_at:
            return 0.0
        return min(1.0, (index - panel_at + 1) / fade_frames)

    stats = {"frames_written": 0, "frames_with_pose": 0, "tail_frames": 0,
             "completion_panel": panel_at is not None}
    frame_index = int(first_frame)
    last_running = {limb.key: 0 for limb in climb.LIMBS}
    tail_left = tail_right = None

    columns = [TextColumn("[cyan]rendering[/]"), BarColumn(), TaskProgressColumn(),
               TextColumn("{task.completed}/{task.total} frames"), TimeElapsedColumn()]

    with Progress(*columns, console=console, transient=True) as progress:
        task = progress.add_task("render", total=(last_frame - first_frame + 1) or None)
        while True:
            ok, frame = cap.read()
            if not ok or frame_index > last_frame:
                break

            # ── left: the clip, with what the models saw on it ───────────────
            left = frame
            if ground is not None and cfg.DRAW_FLOOR_LINE:
                # The junction is a line in the world, so projected it is
                # already this frame's own pixels.
                junction = ground.polyline(frame_index)
                if junction is not None and len(junction) >= 2:
                    cv2.polylines(left, [np.round(junction * [width, height]
                                                  ).astype(np.int32)],
                                  False, cfg.FLOOR_LINE_COLOR,
                                  _px(cfg.FLOOR_LINE_THICK, scale), cv2.LINE_AA)
            # The route projected into *this* frame. The holds are stored once,
            # in the world, and this is the only place they become pixels on the
            # moving image — so the outlines track the wall through every pan
            # and zoom without ever having been detected in this frame.
            live = []
            for hold3d in scene["geometry"].holds_at(frame_index).values():
                polygon = scene["geometry"].project(hold3d, frame_index)
                if polygon is not None and len(polygon) >= 3:
                    live.append({"id": hold3d.id, "polygon": polygon.tolist(),
                                 "bbox": [0.0, 0.0, 0.0, 0.0]})
            for hold in live:
                draw_hold(left, hold, width, height, hold_color, outline_thick,
                          fill=True, alpha=cfg.HOLD_FILL_ALPHA)

            shown = frame_poses if frame_poses is not None else poses
            has_pose = frame_index in shown.kpts
            if has_pose:
                stats["frames_with_pose"] += 1
                skeleton.draw_person(
                    left, shown.kpts[frame_index], shown.valid[frame_index],
                    width=width, height=height, edges=edges, points=points,
                    colors=body_colors,
                    thickness=_px(cfg.LINE_THICKNESS, scale),
                    radius=_px(cfg.POINT_RADIUS, scale),
                    bbox=shown.bboxes[frame_index] if cfg.DRAW_PERSON_BBOX else None,
                    bbox_color=cfg.PERSON_BBOX_COLOR,
                    bbox_thickness=_px(cfg.PERSON_BBOX_THICK, scale))

            # ── right: the route, with the climber taken away ────────────────
            right = blank.copy()

            # After topping out the panel holds its final state, so the last
            # frames read as a finished route rather than continuing to update.
            effective = (min(frame_index, analysis.completion_frame)
                         if analysis.completion_frame is not None else frame_index)
            topped = (analysis.completion_frame is not None
                      and frame_index >= analysis.completion_frame)

            now_holding = analysis.holding.get(effective, {})
            last_running = running.get(effective, last_running)

            if chase is not None:
                target, azimuth, elevation = chase[frame_index]
                finale_start = (target, azimuth, elevation)
                view = space_mod.chase_view((width, height), target, azimuth,
                                            elevation, span_m=cfg.SPACE_CHASE_SPAN_M)
                # Points as big as one voxel is on screen at this zoom, so
                # the wall reads as a surface rather than a dot grid.
                point_size = max(_px(cfg.SPACE_POINT_SIZE, scale), int(np.ceil(
                    1.2 * cfg.LIDAR_VOXEL_M * view.focal / view.distance)))
            else:
                azimuth, elevation = space_mod.orbit_lidar(
                    cfg, (frame_index - first_frame) / max(last_frame - first_frame, 1))
                view = space_mod.fit_view(scene["axes"], (width, height),
                                          azimuth=azimuth, elevation=elevation,
                                          margin=1.25, points=dense_outline)
            space_centroids, _ = space_mod.draw_dense(
                right, scene["axes"], scene["holds"], dense_local, dense_colours,
                view, activated=analysis.activated_at, effective=effective,
                colors=space_colors, cfg=cfg, point_size=point_size)
            # The top hold goes gold the frame the route completes. The 2D
            # code that does this on the flat panel has no outline to draw
            # here (the 3D holds are projected, not stored), so without this
            # the gold only ever appeared once the finale drew it itself.
            if topped and analysis.final_hold_id in scene["holds"]:
                fill_hold3d(right, view, scene["axes"],
                            scene["holds"][analysis.final_hold_id],
                            cfg.HOLD_COMPLETE_COLOR, cfg.HOLD_COMPLETE_ALPHA,
                            _px(cfg.HOLD_ACTIVE_THICK, scale))
            panel_holds = [
                {"id": hid,
                 "bbox": [xy[0] / width, xy[1] / height, 0.0, 0.0],
                 "polygon": []}
                for hid, xy in space_centroids.items()]
            centroids = dict(space_centroids)

            body = scene.get("skeleton", {}).get(frame_index)
            if body is not None:
                world, ok = body
                torso = [k for k in cfg.TORSO_KP_INDICES if ok[k]]
                if (torso and analysis.start_frame is not None
                        and frame_index >= analysis.start_frame and not topped):
                    torso_path.append((frame_index, world[torso].mean(axis=0)))
            # The body's path up the wall, in the world: re-projected every
            # frame because the view moves, so it stays on the wall.
            if len(torso_path) >= 2:
                pix, depth = view.project(scene["axes"].to_wall(
                    np.array([p for _, p in torso_path])))
                pts = [tuple(p) for p, d in zip(pix, depth) if d > 0]
                if topped and cfg.ROUTE_SPLINE:
                    draw_dashed_spline(right, pts, cfg.ROUTE_SPLINE_COLOR,
                                       _px(cfg.ROUTE_SPLINE_THICKNESS, scale),
                                       _px(cfg.ROUTE_SPLINE_DASH, scale),
                                       _px(cfg.ROUTE_SPLINE_GAP, scale))
                else:
                    recent = np.round(pix[-cfg.MIDLINE_TRAIL_FRAMES:]).astype(np.int32)
                    cv2.polylines(right, [recent], False, cfg.MIDLINE_DOT_COLOR,
                                  _px(2, scale), cv2.LINE_AA)
            if body is not None and cfg.SPACE_DRAW_SKELETON:
                world, ok = body
                pix, depth = view.project(scene["axes"].to_wall(world))
                ok = ok & (depth > 0)
                skeleton.draw_person(
                    right, pix / [width, height], ok, width=width, height=height,
                    edges=edges, points=points, colors=body_colors,
                    thickness=_px(cfg.LINE_THICKNESS, scale),
                    radius=_px(cfg.POINT_RADIUS, scale) - 1)


            # Every hold carries its number from the first frame, lit or not.
            # The numbering is a property of the wall, so a hold nobody touched
            # is still hold 3 — and "you could have used 3" needs 3 to be on
            # screen, not inferred from the gap between 2 and 4.
            for hold in panel_holds:
                hid = hold["id"]
                if hid not in analysis.activated_at or analysis.activated_at[hid] > effective:
                    draw_hold(right, hold, width, height, cfg.HOLD_INACTIVE_COLOR,
                              inactive_thick, fill=False, alpha=0.0)
                    if cfg.HOLD_LABEL == "id":
                        text_mod.draw(right, hold_label(hid, cfg), font, size=label_size,
                                      xy=(int(hold["bbox"][0] * width),
                                          int(hold["bbox"][1] * height) - _px(2, scale)),
                                      color=cfg.HOLD_INACTIVE_LABEL_COLOR, anchor="lb")

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
                          fill=True, alpha=alpha)
                cv2.circle(right, centroids[hid], _px(3, scale), color, -1, cv2.LINE_AA)

                if cfg.HOLD_LABEL != "none":
                    # `chip`, not `label`: the render's own `label` is the clip
                    # this attempt belongs to, and the completion panel needs it.
                    chip = (hold_label(hid, cfg) if cfg.HOLD_LABEL == "id"
                            else str(analysis.order[hid]))
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
            bottom = draw_credit(right, cfg, font, scale, right=width - margin,
                                 bottom=height - margin)

            if cfg.TIMER and analysis.start_frame is not None \
                    and frame_index >= analysis.start_frame:
                if analysis.completion_frame is not None \
                        and frame_index >= analysis.completion_frame:
                    elapsed = analysis.elapsed
                else:
                    elapsed = (frame_index - analysis.start_frame) / info.fps
                text_mod.draw(right, format_clock(elapsed),
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
            last_left = left
            if solo is not None:
                if alpha > 0.0 and cfg.COMPARE_PANEL_ON_ROUTE:
                    completion_panel(right, comparison, label, cfg=cfg, font=font,
                                     scale=scale, alpha=alpha)
                solo.write(right)

            stats["frames_written"] += 1
            frame_index += 1
            progress.update(task, completed=frame_index - first_frame)

        if finale_at is not None and finale_start is not None:
            travel = space_mod.body_travel(
                torso_path, scene["axes"], fps=info.fps,
                sigma_s=cfg.FINALE_TRAVEL_SMOOTH_S, start=analysis.start_frame,
                end=analysis.completion_frame)
            # The floor's height along the wall frame's up axis, so reach and
            # the top hold can be quoted above the ground.
            floor_fit = scene["models"][0].get("floor") if scene.get("models") else None
            floor_height = (float(floor_fit[1] - floor_fit[0] @ scene["axes"].origin)
                            if floor_fit is not None else None)
            extent = space_mod.body_extent(
                scene.get("skeleton", {}), scene["axes"], start=analysis.start_frame,
                end=analysis.completion_frame, floor_height=floor_height or 0.0)
            top_hold = scene["holds"].get(analysis.final_hold_id)
            top_hold_m = (float(scene["axes"].to_wall(top_hold.centre.reshape(1, 3))[0, 1]
                                - floor_height)
                          if top_hold is not None and floor_height is not None else None)
            if travel is not None:
                travel = {**travel, "extent": extent, "top_hold_above_floor_m": top_hold_m,
                          "floor_known": floor_height is not None,
                          "floor_local_y": floor_height}
            stats["body_travel"] = travel
            if extent:
                name = lambda e: (e["limb"] or skeleton.KPT_NAMES[e["joint"]]).replace("_", " ")
                console.print(
                    f"  [dim]body reach: {extent['vertical_m']:.2f} m vertical "
                    f"({name(extent['bottom'])} at {extent['bottom_above_floor_m']:.2f} m "
                    f"to {name(extent['top'])} at {extent['top_above_floor_m']:.2f} m), "
                    f"{extent['horizontal_m']:.2f} m horizontal "
                    f"({name(extent['left'])} to {name(extent['right'])}); "
                    f"top hold {top_hold_m:.2f} m off the ground[/]")
            if travel:
                console.print(
                    f"  [dim]body travel: {travel['travel_vertical_m']:.2f} m vertical, "
                    f"{travel['travel_horizontal_m']:.2f} m horizontal "
                    f"(net {travel['net_up_m']:+.2f} m up, "
                    f"{travel['net_across_m']:+.2f} m across; "
                    f"{travel['path_m']:.2f} m of path)[/]")
            written = render_finale(
                pair, solo, left=last_left, scene=scene, analysis=analysis, travel=travel,
                torso_path=torso_path, start=finale_start, size=(width, height),
                fps=info.fps, cfg=cfg, font=font, scale=scale, hold_color=hold_color,
                local=dense_local, colours=dense_colours, colors=space_colors)
            stats["finale_frames"] = written
            stats["frames_written"] += written

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


def fill_hold3d(panel, view, axes, hold, colour, alpha: float, thickness: int):
    """Fill and outline one 3D hold as *view* sees it. Returns the polygon, or None."""
    pix, depth = view.project(axes.to_wall(hold.polygon))
    if not np.all(depth > 0):
        return None
    poly = np.round(pix).astype(np.int32)
    overlay = panel.copy()
    cv2.fillPoly(overlay, [poly], colour)
    cv2.addWeighted(overlay, alpha, panel, 1 - alpha, 0, panel)
    cv2.polylines(panel, [poly], True, colour, thickness, cv2.LINE_AA)
    return poly


def _ease(x: float) -> float:
    """Smoothstep: starts and ends at rest, so no move begins or ends with a jolt."""
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def render_finale(pair, solo, *, left, scene, analysis, torso_path, start, size, fps,
                  cfg, font, scale, hold_color, local, colours, colors,
                  travel: dict | None = None) -> int:
    """The route, revealed in 3D, once it is climbed. Returns frames written.

    Nothing during the climb says *this is 3D*: the chase camera follows the
    climber so closely that the right panel reads as a second video. So the
    ending says it outright, in three beats that overlap:

    1. **expand** — the 3D panel widens to take the whole frame, sliding the
       clip off to the left; the story is now the route, not the footage.
    2. **reveal** — at the same time the camera pulls back from the climber to
       the whole route, and swings out to one side on the way (the swing is
       what makes relief legible: the holds slide against the wall behind
       them), settling face-on with the route centred.
    3. **hold** — the finished route, every used hold numbered in the order it
       was used, the top hold gold, the body's line up it, and what the depth
       measured about the wall.

    The camera path starts exactly where the chase camera left off, so there
    is no cut: the same view, starting to move.
    """
    width, height = size
    axes = scene["axes"]
    holds = scene["holds"]
    n_expand = max(1, int(round(cfg.FINALE_EXPAND_S * fps)))
    n_move = max(1, int(round(cfg.FINALE_MOVE_S * fps)))
    n_hold = max(0, int(round(cfg.FINALE_HOLD_S * fps)))
    n_settle = max(n_expand, n_move)
    # The replay, once the camera is square to the wall: the lit holds dim, then
    # come back one at a time in the order the climber reached them.
    n_dim = int(round(cfg.FINALE_REPLAY_DIM_S * fps)) if cfg.FINALE_REPLAY else 0
    n_step = max(1, int(round(cfg.FINALE_REPLAY_STEP_S * fps)))
    n_seq = len([h for h in analysis.activated_at
                 if analysis.completion_frame is None
                 or analysis.activated_at[h] <= analysis.completion_frame])
    n_replay = (n_dim + n_seq * n_step) if cfg.FINALE_REPLAY else 0
    total = n_settle + n_replay + n_hold

    # Where it ends up: the whole route, centred, framed for the full width.
    outline = np.vstack([axes.to_wall(h.polygon) for h in holds.values()])
    lo, hi = outline.min(axis=0), outline.max(axis=0)
    end_target = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2,
                           float(np.median(outline[:, 2]))])
    end_span = max((hi[1] - lo[1]) * cfg.FINALE_MARGIN,
                   (hi[0] - lo[0]) * cfg.FINALE_MARGIN * height / (2 * width))
    t0, a0, e0 = np.asarray(start[0], dtype=float), float(start[1]), float(start[2])
    s0 = cfg.SPACE_CHASE_SPAN_M
    e1 = np.radians(cfg.FINALE_ELEVATION_DEG)
    swing = np.radians(cfg.FINALE_SWING_DEG)
    # Swing out on the side the chase camera was already on, so the move
    # continues the motion rather than reversing it.
    side = 1.0 if a0 >= 0 else -1.0

    final_id = analysis.final_hold_id
    used = {h: f for h, f in analysis.activated_at.items()
            if analysis.completion_frame is None or f <= analysis.completion_frame}
    complete = cfg.HOLD_COMPLETE_COLOR
    label_size = _px(cfg.HOLD_LABEL_SIZE, scale)
    pad = _px(3, scale)

    # No height here: the triangle says how high the top hold is, and a second
    # height in the caption (the spread between the lowest and highest hold,
    # which is what this used to show) read as a contradiction of it.
    lean = getattr(axes, "lean_deg", None)
    caption = [f"{len(used)} holds used"
               + (f"  ·  wall {abs(lean):.0f}° {'overhang' if lean > 0 else 'slab'}"
                  if lean is not None and abs(lean) >= 1 else "  ·  vertical wall"
                  if lean is not None else "")]
    caption.append(cfg.FINALE_SUBTITLE)
    elapsed = analysis.elapsed

    # The sequence, and when each hold was reached — the replay's script.
    sequence = [h for h in climb.activation_sequence(analysis) if h in used and h in holds]
    reached = [used[h] for h in sequence]
    path_frames = np.array([f for f, _ in torso_path]) if torso_path else np.zeros(0)
    path_local = (axes.to_wall(np.array([p for _, p in torso_path]))
                  if torso_path else np.zeros((0, 3)))
    glow_sigma = max(2.0, 10.0 * scale)
    separator = _separator(font, cfg)
    seq_size = _px(20, scale)

    # Which limbs used each hold during the climb, and the colour that makes:
    # one limb's own colour, or an even blend of every limb that was on it.
    window_lo, window_hi = analysis.window if analysis.window else (-10 ** 9, 10 ** 9)
    if analysis.completion_frame is not None:
        window_hi = min(window_hi, analysis.completion_frame)
    limbs_on: dict[int, list[str]] = {}
    for contact in analysis.contacts:
        if min(contact.end, window_hi) < max(contact.start, window_lo):
            continue
        keys = limbs_on.setdefault(contact.hold_id, [])
        if contact.limb not in keys:
            keys.append(contact.limb)
    order_index = {limb.key: n for n, limb in enumerate(climb.LIMBS)}
    limb_colour: dict[int, tuple[int, int, int]] = {}
    for hid in sequence:
        keys = sorted(limbs_on.get(hid, []), key=order_index.get)
        if keys:
            mix = np.mean([cfg.LIMB_COLORS[k] for k in keys], axis=0)
            limb_colour[hid] = tuple(int(round(v)) for v in mix)
        else:
            limb_colour[hid] = hold_color
    step_of = {hid: n + 1 for n, hid in enumerate(sequence)}
    badge_size = _px(13, scale)

    def draw(panel, view, *, light: dict, path_until: float | None, shown: int | None,
             newest: int | None, by_limb: bool = False, steps: bool = False,
             legs: float = 0.0, card=None):
        """One finale frame. *light* is hold -> level: 0 dim, 1 lit, >1 flashing.

        *by_limb* colours each lit hold by the limb(s) that used it rather than
        the route's colour; *steps* adds each hold's place in the sequence.
        """
        point_size = max(_px(cfg.SPACE_POINT_SIZE, scale), int(np.ceil(
            1.2 * cfg.LIDAR_VOXEL_M * view.focal / view.distance)))
        # Every hold dim first; the lit ones are laid over it below, so the
        # replay can take any hold from dark to lit to flashing.
        centroids, _ = space_mod.draw_dense(
            panel, axes, holds, local, colours, view, activated={},
            effective=0, colors=colors, cfg=cfg, point_size=point_size)

        # On the wall, under every lit hold: a measurement painted on the
        # plywood, not a sticker over the route.
        leg_labels = draw_triangle(panel, view, legs)

        if path_until is not None and len(path_local) >= 2:
            keep = path_frames <= path_until
            if keep.sum() >= 2:
                pix, depth = view.project(path_local[keep])
                pts = [tuple(p) for p, d in zip(pix, depth) if d > 0]
                draw_dashed_spline(panel, pts, cfg.ROUTE_SPLINE_COLOR,
                                   _px(cfg.ROUTE_SPLINE_THICKNESS, scale) + 1,
                                   _px(cfg.ROUTE_SPLINE_DASH, scale),
                                   _px(cfg.ROUTE_SPLINE_GAP, scale))

        def colour_of(hid):
            if by_limb:
                return limb_colour[hid]
            return complete if hid == final_id else hold_color

        glow = None
        for hid in sequence:
            level = light.get(hid, 0.0)
            if level <= 0:
                continue
            base = colour_of(hid)
            flash = max(0.0, level - 1.0)          # 0 settled, up to FINALE_REPLAY_GLOW
            # Brighter while it flashes: toward white, and more opaque.
            colour = tuple(int(c + (255 - c) * 0.55 * min(flash, 1.0)) for c in base)
            alpha = min(1.0, (cfg.HOLD_COMPLETE_ALPHA if hid == final_id or by_limb
                              else cfg.HOLD_ACTIVE_FILL_ALPHA) * min(level, 1.0)
                        + 0.3 * min(flash, 1.0))
            poly = fill_hold3d(panel, view, axes, holds[hid], colour, alpha,
                               _px(cfg.HOLD_ACTIVE_THICK, scale) + (1 if flash > 0.2 else 0))
            if poly is not None and by_limb and hid == final_id:
                # Limb colours for the fill, but the top hold keeps a gold rim:
                # it is still the one that finished the route.
                cv2.polylines(panel, [poly], True, complete,
                              _px(cfg.HOLD_ACTIVE_THICK, scale) + 2, cv2.LINE_AA)
            if poly is not None and flash > 0.02:
                if glow is None:
                    glow = np.zeros_like(panel)
                cv2.fillPoly(glow, [poly], tuple(int(c * min(flash, 1.0)) for c in colour))
        if glow is not None:
            # A soft halo, added rather than blended: light, not paint.
            glow = cv2.GaussianBlur(glow, (0, 0), glow_sigma)
            cv2.add(panel, glow, dst=panel)

        for hid in sequence:
            level = light.get(hid, 0.0)
            if level <= 0 or hid not in centroids:
                continue
            colour = colour_of(hid)
            chip = hold_label(hid, cfg)
            tw, th = text_mod.measure(chip, font, label_size)
            x, y = centroids[hid]
            x1, y1 = x - tw // 2 - pad, y - th - pad * 2 - _px(6, scale)
            layer = panel.copy()
            cv2.rectangle(layer, (x1, y1), (x1 + tw + pad * 2, y1 + th + pad * 2),
                          colour, -1)
            text_mod.draw(layer, chip, font, size=label_size, xy=(x1 + pad, y1 + pad),
                          color=cfg.HOLD_LABEL_COLOR, anchor="lt")
            if steps:
                # Its place in the sequence, in a white disc on the chip's
                # shoulder: the letter says which hold, the number says when.
                number = str(step_of[hid])
                nw, nh = text_mod.measure(number, font, badge_size)
                radius = max(nw, nh) // 2 + _px(4, scale)
                cx, cy = x1 + tw + pad * 2 + radius - _px(2, scale), y1 - radius // 3
                cv2.circle(layer, (cx, cy), radius, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(layer, (cx, cy), radius, (40, 40, 40), 1, cv2.LINE_AA)
                text_mod.draw(layer, number, font, size=badge_size,
                              xy=(cx - nw // 2, cy - nh // 2 - 1), color=(20, 20, 20),
                              anchor="lt")
            a = min(level, 1.0)
            cv2.addWeighted(layer, a, panel, 1 - a, 0, panel)

        margin = _px(cfg.PANEL_MARGIN, scale)
        if shown:
            # The sequence, written out as it is replayed, bottom-left — each
            # hold in the colour it lit up, the newest one in white.
            y = panel.shape[0] - margin - seq_size
            text_mod.draw(panel, "SEQUENCE", font, size=_px(13, scale),
                          xy=(margin, y - _px(8, scale)), color=(150, 150, 150),
                          anchor="lb")
            x = margin
            for n, hid in enumerate(sequence[:shown]):
                if n:
                    text_mod.draw(panel, separator, font, size=seq_size, xy=(x, y),
                                  color=cfg.COMPARE_SEP_COLOR, anchor="lt")
                    x += text_mod.measure(separator, font, seq_size)[0]
                token = hold_label(hid, cfg)
                colour = (255, 255, 255) if hid == newest else colour_of(hid)
                text_mod.draw(panel, token, font, size=seq_size, xy=(x, y),
                              color=colour, anchor="lt")
                x += text_mod.measure(token, font, seq_size)[0]
        if leg_labels:
            mask = hold_mask(view, panel.shape)
            # The caption (top-left) and the legend + sequence (bottom) are
            # taken too: a label is a claim about the wall, not about them.
            mask[:_px(64, scale), :_px(460, scale)] = 1
            mask[panel.shape[0] - _px(96, scale):, :] = 1
            if triangle is not None:
                pts = np.array([triangle["across"][0], triangle["corner"], triangle["up"][1]])
                pix, _ = view.project(pts)
                # Thin: a label *beside* a leg must not count as covering it,
                # or every spot next to the leg is "taken" and the least-bad one
                # wins — which put the height label on top of two holds.
                cv2.polylines(mask, [np.round(pix).astype(np.int32)], False, 1,
                              _px(6, scale))
            if card is not None and card[0] == "card":
                cx, cy = card[1]
                mask[max(0, cy - _px(6, scale)):cy + card_h + _px(6, scale),
                     max(0, cx - _px(6, scale)):cx + card_w + _px(6, scale)] = 1
            for text, a, b, horizontal, colour in leg_labels:
                label(panel, text, a, b, horizontal, mask, colour)
            # How far the torso travelled, on its own curving line.
            if len(path_local) >= 2 and travel:
                pix, depth = view.project(path_local)
                pix = pix[depth > 0]
                if len(pix) >= 4:
                    label_on_path(panel, f"{travel['path_m']:.2f} m torso traveled", pix,
                                  mask)

        if by_limb:
            # What the colours mean, above the sequence.
            y = panel.shape[0] - margin - seq_size - _px(34, scale)
            x = margin
            swatch = _px(10, scale)
            for limb in climb.LIMBS:
                cv2.rectangle(panel, (x, y - swatch), (x + swatch, y),
                              cfg.LIMB_COLORS[limb.key], -1)
                x += swatch + _px(5, scale)
                text_mod.draw(panel, limb.name, font, size=_px(13, scale), xy=(x, y + 1),
                              color=(200, 200, 200), anchor="lb")
                x += text_mod.measure(limb.name, font, _px(13, scale))[0] + _px(14, scale)
            text_mod.draw(panel, "blended where limbs shared a hold", font,
                          size=_px(11, scale), xy=(x, y + 1), color=(130, 130, 130),
                          anchor="lb")

    def overlay_text(panel, alpha):
        margin = _px(cfg.PANEL_MARGIN, scale)
        # The clock and credit stay exactly where they were through the whole
        # finale, so the cut from the clip is not also a cut in the corner.
        bottom = panel.shape[0] - margin
        w = panel.shape[1]
        bottom = draw_credit(panel, cfg, font, scale, right=w - margin, bottom=bottom)
        if cfg.TIMER and elapsed is not None:
            text_mod.draw(panel, format_clock(elapsed),
                          font, size=_px(cfg.TIMER_SIZE, scale), xy=(w - margin, bottom),
                          color=cfg.TIMER_COLOR, anchor="rb")
        if alpha <= 0:
            return
        # The caption is the new information, so it is the only thing that fades in.
        layer = panel.copy()
        text_mod.draw(layer, caption[0], font, size=_px(22, scale),
                      xy=(margin, margin), color=(255, 255, 255), anchor="lt")
        text_mod.draw(layer, caption[1], font, size=_px(14, scale),
                      xy=(margin, margin + _px(30, scale)), color=(170, 170, 170),
                      anchor="lt")
        cv2.addWeighted(layer, alpha, panel, 1 - alpha, 0, panel)

    # ── the travel card ──
    # Placed where it covers nothing: the final framing is rendered once as an
    # occupancy mask (every hold's outline, grown a little, and the body's
    # line), the corners already taken by the caption, sequence and clock are
    # ruled out, and the least-covered of a handful of spots wins.
    end_azimuth = a0 if cfg.FINALE_ANGLE == "keep" else 0.0
    end_elevation = e0 if cfg.FINALE_ANGLE == "keep" else (
        e1 if cfg.FINALE_ANGLE == "swing" else 0.0)
    card_w, card_h = _px(300, scale), _px(192, scale)

    def place_card(panel_size, span):
        pw, ph = panel_size
        view = space_mod.chase_view(panel_size, end_target, end_azimuth, end_elevation,
                                    span_m=span)
        busy = np.zeros((ph, pw), np.uint8)
        for hold in holds.values():
            pix, depth = view.project(axes.to_wall(hold.polygon))
            if np.all(depth > 0):
                cv2.fillPoly(busy, [np.round(pix).astype(np.int32)], 1)
        if len(path_local) >= 2:
            pix, depth = view.project(path_local)
            cv2.polylines(busy, [np.round(pix[depth > 0]).astype(np.int32)], False, 1,
                          _px(6, scale))
        if triangle is not None:
            pts = np.array([triangle["across"][0], triangle["corner"], triangle["up"][1]])
            pix, _ = view.project(pts)
            cv2.polylines(busy, [np.round(pix).astype(np.int32)], False, 1, _px(40, scale))
            for point, _ in triangle["marks"].values():
                p, _ = view.project(point.reshape(1, 3))
                cv2.circle(busy, tuple(np.round(p[0]).astype(int)), _px(20, scale), 1, -1)
        busy = cv2.dilate(busy, np.ones((_px(14, scale),) * 2, np.uint8))
        margin = _px(cfg.PANEL_MARGIN, scale)
        # Reserved: caption (top-left), legend + sequence (bottom-left), clock.
        busy[:_px(64, scale), :_px(460, scale)] = 1
        busy[ph - _px(96, scale):, :] = 1
        spots = []
        for fx in (0.0, 0.5, 1.0):
            for fy in (0.0, 0.33, 0.66, 1.0):
                x = int(margin + fx * (pw - card_w - 2 * margin))
                y = int(margin + fy * (ph - card_h - 2 * margin))
                spots.append((x, y))
        # Least covered wins; ties go to the spot nearest a corner, which reads
        # as an annotation rather than as something floating over the route.
        def cost(spot):
            x, y = spot
            covered = float(busy[y:y + card_h, x:x + card_w].mean())
            edge = min(x, pw - x - card_w) + min(y, ph - y - card_h)
            return (round(covered, 3), edge)
        best = min(spots, key=cost)
        # "Least covered" is not "clear". On a narrow frame there may be no gap
        # the card fits in, and settling for the least bad spot puts it over a
        # hold — so then it becomes one line under the caption instead.
        if cost(best)[0] > 0.01:
            return ("line", (margin, margin + _px(52, scale)))
        return ("card", best)

    # The triangle: two proxies for how far the climb went, measured on things
    # that do not move. Up is the top hold's height off the floor; across is
    # from the hold furthest from it (the leftmost used, on a route that tops
    # out to the right) over to the top hold. It closes at the top hold, with
    # its corner on the floor beneath it and no hypotenuse. On the wall's
    # surface (the holds' plane), set out from the route like a drawing's
    # dimension lines so the climb sits inside it with room to spare.
    def leg_colour_for(which):
        return complete if which == "top" else cfg.FINALE_TRAVEL_COLOR

    triangle = None
    floor_y = travel.get("floor_local_y") if travel else None
    if (travel and cfg.FINALE_TRAVEL and cfg.FINALE_TRAVEL_TRIANGLE
            and floor_y is not None and final_id in holds and len(sequence) >= 2):
        wall_z = float(end_target[2])
        def at(x, y):
            return np.array([x, y, wall_z])
        def centre_of(h):
            return axes.to_wall(holds[h].centre.reshape(1, 3))[0]
        top_c = centre_of(final_id)
        across_of = {h: centre_of(h)[0] for h in sequence}
        rightward = top_c[0] >= float(np.median(list(across_of.values())))
        far = (min(across_of, key=across_of.get) if rightward
               else max(across_of, key=across_of.get))
        far_c = centre_of(far)
        leg_x = top_c[0] + (cfg.FINALE_TRAVEL_OFFSET_M if rightward
                            else -cfg.FINALE_TRAVEL_OFFSET_M)
        leg_y = floor_y                                  # along the floor itself
        triangle = {
            "corner": at(leg_x, leg_y),
            "across": (at(far_c[0], leg_y), at(leg_x, leg_y)),   # drawn leg
            "up": (at(leg_x, leg_y), at(leg_x, top_c[1])),
            "across_span": (at(far_c[0], leg_y), at(top_c[0], leg_y)),   # measured
            "up_span": (at(leg_x, floor_y), at(leg_x, top_c[1])),
            # What each leg is measured from: a dot on the hold, and a faint
            # line out to the leg.
            "marks": {"left": (far_c, leg_colour_for("far")),
                      "top": (top_c, leg_colour_for("top"))},
            "vertical": float(top_c[1] - floor_y),
            "horizontal": float(abs(top_c[0] - far_c[0])),
            "far_hold": far,
        }
        lo_x, hi_x = sorted([far_c[0], leg_x])
        hi_y = top_c[1]
        # Frame the final view on everything it draws — holds, triangle, the
        # extremes — not on the holds alone, or the legs run off the frame.
        # Extra room below for the legend and sequence, above for the caption.
        drawn = np.vstack([outline[:, :2], [[leg_x, leg_y], [lo_x, leg_y], [hi_x, leg_y],
                                            [leg_x, hi_y]]])
        dlo, dhi = drawn.min(axis=0), drawn.max(axis=0)
        room = dhi[1] - dlo[1]
        dlo = dlo - [0.10 * (dhi[0] - dlo[0]), 0.16 * room]
        dhi = dhi + [0.10 * (dhi[0] - dlo[0]), 0.10 * room]
        end_target = np.array([(dlo[0] + dhi[0]) / 2, (dlo[1] + dhi[1]) / 2, end_target[2]])
        end_span = max(dhi[1] - dlo[1], (dhi[0] - dlo[0]) * height / (2 * width))
        lo[:2], hi[:2] = dlo, dhi

    card_pair = card_solo = None
    if travel and cfg.FINALE_TRAVEL and cfg.FINALE_TRAVEL_CARD:
        card_pair = place_card((2 * width, height), end_span)
        card_solo = place_card((width, height),
                               max(end_span, (hi[0] - lo[0]) * cfg.FINALE_MARGIN
                                   * height / width))

    def arrows(layer, cx, cy, half, axis, thick):
        a, b = ((cx, cy - half), (cx, cy + half)) if axis == "v" else \
               ((cx - half, cy), (cx + half, cy))
        for p, q in ((a, b), (b, a)):
            cv2.arrowedLine(layer, q, p, (255, 255, 255), thick, cv2.LINE_AA,
                            tipLength=0.35)

    extent_info = travel.get("extent") if travel else None
    top_hold_m = travel.get("top_hold_above_floor_m") if travel else None

    def card_rows():
        rows = []
        if triangle is not None:
            rows += [("vertical reach", triangle["vertical"], "v"),
                     ("horizontal reach", triangle["horizontal"], "h")]
        rows.append(("body-centre path", travel["path_m"], "path"))
        if top_hold_m is not None:
            rows.append(("top hold, off the ground", top_hold_m, "top"))
        return rows

    def part(e):
        return (e["limb"] or skeleton.KPT_NAMES[e["joint"]]).replace("_", " ")

    def footnote():
        if extent_info:
            return [f"lowest point: {part(extent_info['bottom'])}, "
                    f"{extent_info['bottom_above_floor_m']:.2f} m off the ground",
                    f"highest point: {part(extent_info['top'])}, "
                    f"{extent_info['top_above_floor_m']:.2f} m off the ground"]
        return ["heights above the floor"]

    def draw_icon(layer, icon, at):
        cx, cy = at
        half = _px(9, scale)
        thick = max(1, _px(2, scale))
        if icon in ("v", "h"):
            arrows(layer, cx, cy, half, icon, thick)
        elif icon == "path":
            pts = np.array([[cx - half, cy + half // 2], [cx - half // 3, cy - half // 2],
                            [cx + half // 3, cy + half // 2], [cx + half, cy - half // 2]])
            cv2.polylines(layer, [pts.astype(np.int32)], False, (255, 255, 255), thick,
                          cv2.LINE_AA)
        elif icon == "top":
            tri = np.array([[cx, cy - half], [cx + half, cy + half], [cx - half, cy + half]])
            cv2.fillPoly(layer, [tri.astype(np.int32)], complete, cv2.LINE_AA)

    def draw_line(panel, at, alpha):
        """The compact form: one line, under the caption."""
        x, y = at
        layer = panel.copy()
        size = _px(15, scale)
        pass
        short = {"vertical reach": "up", "horizontal reach": "across",
                 "body-centre path": "path", "top hold, off the ground": "top hold"}
        for label, meters, icon in card_rows():
            draw_icon(layer, icon, (x + _px(8, scale), y + size // 2 + 1))
            x += _px(22, scale)
            text = f"{meters:.2f} m {short.get(label, label)}"
            text_mod.draw(layer, text, font, size=size, xy=(x, y), color=(255, 255, 255),
                          anchor="lt")
            x += text_mod.measure(text, font, size)[0] + _px(14, scale)
        cv2.addWeighted(layer, alpha, panel, 1 - alpha, 0, panel)

    def draw_card(panel, placed, alpha):
        if placed is None or alpha <= 0:
            return
        kind, at = placed
        if kind == "line":
            draw_line(panel, at, alpha)
            return
        x, y = at
        layer = panel.copy()
        cv2.rectangle(layer, (x, y), (x + card_w, y + card_h), (24, 24, 24), -1)
        cv2.rectangle(layer, (x, y), (x + card_w, y + card_h), (80, 80, 80), 1)
        inner = _px(12, scale)
        text_mod.draw(layer, "THE CLIMB, MEASURED", font, size=_px(12, scale),
                      xy=(x + inner, y + inner), color=(150, 150, 150), anchor="lt")
        rows = card_rows()
        row_y = y + inner + _px(22, scale)
        value_size = _px(22, scale)
        for label_text, meters, icon in rows:
            icon_at = (x + inner + _px(9, scale), row_y + value_size // 2)
            draw_icon(layer, icon, icon_at)
            value = f"{meters:.2f} m"
            text_mod.draw(layer, value, font, size=value_size,
                          xy=(x + inner + _px(26, scale), row_y), color=(255, 255, 255),
                          anchor="lt")
            vw = text_mod.measure(value, font, value_size)[0]
            text_mod.draw(layer, label_text, font, size=_px(13, scale),
                          xy=(x + inner + _px(34, scale) + vw, row_y + _px(7, scale)),
                          color=(170, 170, 170), anchor="lt")
            row_y += value_size + _px(8, scale)
        lines = footnote()
        for n, line in enumerate(reversed(lines)):
            text_mod.draw(layer, line, font, size=_px(11, scale),
                          xy=(x + inner, y + card_h - inner - n * _px(15, scale)),
                          color=(140, 140, 140), anchor="lb")
        cv2.addWeighted(layer, alpha, panel, 1 - alpha, 0, panel)

    card_from = n_settle + n_replay     # after the replay: the route is complete
    n_draw_legs = max(1, int(round(0.9 * fps)))
    leg_colour = cfg.FINALE_TRAVEL_COLOR

    def dashed(panel, a, b, thick, *, colour=None, dash=9, gap=6):
        colour = leg_colour if colour is None else colour
        length = float(np.hypot(*(np.subtract(b, a))))
        dash, gap = _px(dash, scale), _px(gap, scale)
        if length < 1:
            return
        step = (np.subtract(b, a)) / length
        t = 0.0
        while t < length:
            p = np.add(a, step * t)
            q = np.add(a, step * min(t + dash, length))
            cv2.line(panel, tuple(np.round(p).astype(int)), tuple(np.round(q).astype(int)),
                     colour, thick, cv2.LINE_AA)
            t += dash + gap

    def hold_mask(view, shape):
        """Where the holds are in this view, grown a little: labels keep off it."""
        mask = np.zeros(shape[:2], np.uint8)
        for hold in holds.values():
            pix, depth = view.project(axes.to_wall(hold.polygon))
            if np.all(depth > 0):
                cv2.fillPoly(mask, [np.round(pix).astype(np.int32)], 1)
        return cv2.dilate(mask, np.ones((_px(10, scale),) * 2, np.uint8))

    def label(panel, text, a, b, horizontal, mask, colour=None):
        """A leg's length, beside the leg, in the first spot that touches no hold.

        Tried from the middle of the leg outward, on both sides of it; if every
        spot touches something, the least-touching one is used.
        """
        size = _px(17, scale)
        tw, th = text_mod.measure(text, font, size)
        p, gap = _px(4, scale), _px(14, scale)
        h, w = mask.shape
        best, best_cover = None, None
        for t in (0.5, 0.38, 0.62, 0.26, 0.74, 0.14, 0.86):
            mx, my = np.add(a, np.subtract(b, a) * t)
            if horizontal:
                sides = [(mx - tw / 2, my + gap), (mx - tw / 2, my - gap - th)]
            else:
                sides = [(mx + gap, my - th / 2), (mx - gap - tw, my - th / 2)]
            for x, y in sides:
                x, y = int(round(x)), int(round(y))
                # Wholly inside the frame, or not at all.
                if x - p < 0 or y - p < 0 or x + tw + p > w or y + th + p > h:
                    continue
                x0, y0, x1, y1 = x - p, y - p, x + tw + p, y + th + p
                cover = float(mask[y0:y1, x0:x1].mean())
                if best_cover is None or cover < best_cover:
                    best, best_cover = (x, y), cover
                if cover == 0.0:
                    break
            if best_cover == 0.0:
                break
        if best is None:
            return
        x, y = best
        layer = panel.copy()
        cv2.rectangle(layer, (x - p, y - p), (x + tw + p, y + th + p), (22, 22, 22), -1)
        cv2.addWeighted(layer, 0.7, panel, 0.3, 0, panel)
        text_mod.draw(panel, text, font, size=size, xy=(x, y),
                      color=leg_colour if colour is None else colour, anchor="lt")
        mask[max(0, y - 3 * p):y + th + 3 * p, max(0, x - 3 * p):x + tw + 3 * p] = 1

    def boxed(panel, text, x, y, colour, size):
        tw, th = text_mod.measure(text, font, size)
        p = _px(4, scale)
        layer = panel.copy()
        cv2.rectangle(layer, (x - p, y - p), (x + tw + p, y + th + p), (22, 22, 22), -1)
        cv2.addWeighted(layer, 0.72, panel, 0.28, 0, panel)
        text_mod.draw(panel, text, font, size=size, xy=(x, y), color=colour, anchor="lt")

    def clear_spot(candidates, tw, th, mask):
        p = _px(4, scale)
        h, w = mask.shape
        best, best_cover = None, None
        for x, y in candidates:
            x, y = int(round(x)), int(round(y))
            x0, y0, x1, y1 = max(0, x - p), max(0, y - p), min(w, x + tw + p), min(h, y + th + p)
            if x1 <= x0 or y1 <= y0 or x < 0 or y < 0 or x + tw > w or y + th > h:
                continue
            cover = float(mask[y0:y1, x0:x1].mean())
            if best_cover is None or cover < best_cover:
                best, best_cover = (x, y), cover
            if cover == 0.0:
                break
        return best

    def label_on_path(panel, text, pix, mask):
        size = _px(15, scale)
        tw, th = text_mod.measure(text, font, size)
        gap = _px(12, scale)
        candidates = []
        for t in (0.45, 0.35, 0.55, 0.25, 0.65, 0.15, 0.75):
            k = int(t * (len(pix) - 1))
            a, b = pix[max(0, k - 3)], pix[min(len(pix) - 1, k + 3)]
            d = b - a
            normal = np.array([-d[1], d[0]]) / max(np.hypot(*d), 1e-6)
            for side in (1, -1):
                c = pix[k] + side * normal * (gap + th)
                candidates.append((c[0] - tw / 2, c[1] - th / 2))
        spot = clear_spot(candidates, tw, th, mask)
        if spot:
            boxed(panel, text, *spot, cfg.ROUTE_SPLINE_COLOR, size)

    def grow_mask(mask, poly, pad=10):
        """Mark a placed label's neighbourhood taken, so the next one steers clear."""
        (x0, y0), (x1, y1) = np.asarray(poly).min(axis=0), np.asarray(poly).max(axis=0)
        p = _px(pad, scale)
        mask[max(0, int(y0) - p):int(y1) + p, max(0, int(x0) - p):int(x1) + p] = 1

    def label_beside(panel, text, poly, mask, colour, size=15):
        size = _px(size, scale)
        tw, th = text_mod.measure(text, font, size)
        gap = _px(12, scale)
        (x0, y0), (x1, y1) = poly.min(axis=0), poly.max(axis=0)
        cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
        candidates = [(x1 + gap, cy - th / 2), (x0 - gap - tw, cy - th / 2),
                      (cx - tw / 2, y0 - gap - th), (cx - tw / 2, y1 + gap),
                      (x1 + gap, y0 - th), (x0 - gap - tw, y0 - th)]
        spot = clear_spot(candidates, tw, th, mask)
        if spot:
            boxed(panel, text, *spot, colour, size)
            grow_mask(mask, [spot, (spot[0] + tw, spot[1] + th)])

    def draw_triangle(panel, view, progress):
        """The two legs, growing across then up with *progress* 0-1.

        Returns the labels to place, which `draw` puts on last, over everything.
        """
        if triangle is None or progress <= 0:
            return []
        segs = ["across", "up", "across_span", "up_span"]
        pts = np.array([triangle["corner"]] + [p for sg in segs for p in triangle[sg]])
        pix, depth = view.project(pts)
        if not np.all(depth > 0):
            return []
        P = {"corner": pix[0]}
        for n, sg in enumerate(segs):
            P[sg] = (pix[1 + 2 * n], pix[2 + 2 * n])
        thick = max(1, _px(2, scale))
        thin = max(1, _px(1, scale))
        across = min(1.0, progress * 2.0)
        up = max(0.0, progress * 2.0 - 1.0)
        faint = tuple(int(c * 0.75) for c in leg_colour)

        a0, a1 = P["across"]
        dashed(panel, a0, a0 + (a1 - a0) * across, thick)
        if up > 0:
            u0, u1 = P["up"]
            dashed(panel, u0, u0 + (u1 - u0) * up, thick)

        # Each extreme: a dot where that body part was, in its limb's colour
        # (slate for any joint that is not a hand or a foot), with a faint
        # extension line out to the leg it bounds.
        def mark(key, to_leg):
            point, colour = triangle["marks"][key]
            p, d = view.project(point.reshape(1, 3))
            if d[0] <= 0:
                return
            p = p[0]
            dashed(panel, p, to_leg(p), thin, colour=faint, dash=4, gap=5)
            cv2.circle(panel, tuple(np.round(p).astype(int)), _px(5, scale), colour, -1,
                       cv2.LINE_AA)
            cv2.circle(panel, tuple(np.round(p).astype(int)), _px(5, scale), (30, 30, 30),
                       1, cv2.LINE_AA)

        to_across = lambda p: np.array([p[0], a0[1] + (a1[1] - a0[1])
                                        * ((p[0] - a0[0]) / max(a1[0] - a0[0], 1e-6))])
        to_up = lambda p: np.array([u0[0], p[1]]) if up > 0 else p

        def tick(point, horizontal_leg):
            half = _px(7, scale)
            x, y = point
            a, b = (((x, y - half), (x, y + half)) if horizontal_leg
                    else ((x - half, y), (x + half, y)))
            cv2.line(panel, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)),
                     leg_colour, thick, cv2.LINE_AA)

        jobs = []
        if across >= 1.0:
            if "left" in triangle["marks"]:
                mark("left", to_across)
            for point in P["across_span"]:
                tick(point, True)
            jobs.append((f"{triangle['horizontal']:.2f} m across", *P["across_span"], True,
                         leg_colour))
        if up >= 1.0:
            if "top" in triangle["marks"]:
                mark("top", to_up)
            for point in P["up_span"]:
                tick(point, False)
            # The right angle, as a draughtsman would mark it.
            c = P["corner"]
            m = _px(11, scale)
            sx = -1 if a0[0] < c[0] else 1
            sy = -1 if u1[1] < c[1] else 1
            box = np.array([[c[0] + sx * m, c[1]], [c[0] + sx * m, c[1] + sy * m],
                            [c[0], c[1] + sy * m]])
            cv2.polylines(panel, [np.round(box).astype(np.int32)], False, leg_colour,
                          max(1, _px(1.5, scale)), cv2.LINE_AA)
            jobs.append((f"{triangle['vertical']:.2f} m off the ground", *P["up_span"], False,
                         complete))
        return jobs

    for i in range(total):
        grow = _ease(i / n_expand)
        q = _ease(i / n_move)
        held = max(0, i - max(n_expand, n_move))
        target = (1 - q) * t0 + q * end_target
        span = (1 - q) * s0 + q * end_span
        # A slow drift through the hold, so the last frames are still alive.
        drift = np.radians(cfg.FINALE_DRIFT_DEG) * (max(0, held - n_replay) / max(n_hold, 1))
        if cfg.FINALE_ANGLE == "keep":
            # A straight pull-back: the wall stays at exactly the angle the
            # climb was watched from, and only the framing changes.
            azimuth, elevation = a0 + drift, e0
        elif cfg.FINALE_ANGLE == "swing":
            azimuth = (1 - q) * a0 + side * swing * np.sin(np.pi * q) + drift
            elevation = (1 - q) * e0 + q * e1
        else:
            # Face-on: the camera settles square to the wall, looking straight
            # along its normal, with the route centred. The same ease as the
            # pull-back and never past zero, so the wall straightens once and
            # does not tilt on the way.
            azimuth, elevation = (1 - q) * a0 + drift, (1 - q) * e0

        # What is lit, how brightly, and how far the body's line has got.
        r_i = i - n_settle
        full_path = float(analysis.completion_frame or 10 ** 9)
        if r_i < 0 or not cfg.FINALE_REPLAY:
            state = dict(light={h: 1.0 for h in sequence}, path_until=full_path,
                         shown=None, newest=None)
        elif r_i < n_dim:
            fade = 1.0 - _ease(r_i / max(n_dim, 1))
            state = dict(light={h: fade for h in sequence},
                         path_until=full_path if fade > 0.5 else None,
                         shown=None, newest=None)
        elif r_i < n_replay:
            t = (r_i - n_dim) / n_step          # holds reached so far, fractional
            k = int(t)
            light = {}
            for n, hid in enumerate(sequence):
                if n <= k:
                    since = (t - n) * cfg.FINALE_REPLAY_STEP_S
                    light[hid] = 1.0 + cfg.FINALE_REPLAY_GLOW * np.exp(
                        -since / cfg.FINALE_REPLAY_FLASH_S)
            # The line runs from where the climb began to this hold, then eases
            # on toward the next as the next one is about to light.
            if k + 1 < len(reached):
                frac = _ease(t - k)
                until = reached[k] + (reached[k + 1] - reached[k]) * frac
            else:
                until = full_path
            state = dict(light=light, path_until=until, shown=min(k + 1, len(sequence)),
                         newest=sequence[min(k, len(sequence) - 1)] if sequence else None,
                         by_limb=cfg.FINALE_REPLAY_BY_LIMB, steps=True)
        else:
            state = dict(light={h: 1.0 for h in sequence}, path_until=full_path,
                         shown=len(sequence) if cfg.FINALE_REPLAY else None, newest=None,
                         by_limb=cfg.FINALE_REPLAY and cfg.FINALE_REPLAY_BY_LIMB,
                         steps=cfg.FINALE_REPLAY)

        state["legs"] = _ease((i - card_from) / n_draw_legs) if triangle is not None else 0.0
        panel_w = width + int(round(width * grow))
        panel = np.zeros((height, panel_w, 3), np.uint8)
        view = space_mod.chase_view((panel_w, height), target, azimuth, elevation,
                                    span_m=span)
        draw(panel, view, **state, card=card_pair if panel_w == 2 * width else None)
        text_alpha = _ease((i - n_move * 0.7) / max(1, n_move * 0.3))
        overlay_text(panel, text_alpha)
        card_alpha = _ease((i - card_from) / max(1, int(0.5 * fps)))
        if panel_w == 2 * width:
            draw_card(panel, card_pair, card_alpha)
        # The clip slides off to the left as the panel takes its place, and
        # dims as it goes, so the eye follows the route rather than the exit.
        shown = left[:, panel_w - width:]
        if shown.shape[1]:
            shown = (shown.astype(np.float32) * (1 - 0.6 * grow)).astype(np.uint8)
            frame = np.hstack([shown, panel])
        else:
            frame = panel
        pair.write(frame)

        if solo is not None:
            solo_panel = np.zeros((height, width, 3), np.uint8)
            solo_view = space_mod.chase_view(
                (width, height), target, azimuth, elevation,
                span_m=(1 - q) * s0 + q * max(end_span, (hi[0] - lo[0])
                                              * cfg.FINALE_MARGIN * height / width))
            draw(solo_panel, solo_view, **state, card=card_solo)
            overlay_text(solo_panel, text_alpha)
            draw_card(solo_panel, card_solo, card_alpha)
            solo.write(solo_panel)
    return total
