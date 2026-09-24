"""Holds the colour prompt missed, found by watching where the climber held on.

`"green climbing hold"` finds the green holds. It does not find the one that has
been chalked over until it reads grey — and on this wall there is exactly such a
hold, a foothold between two SAM found, dusted pale enough that nothing about it
is green any more. Lowering the score threshold does not help, because the
tracker never proposed it at all; broadening the prompt to `"climbing hold"`
finds every hold in the gym and puts the colour problem back one step.

What does know that hold is there is the climber. A foot that sits in one place
for a second and a half, on the wall, touching nothing the model found, is
evidence — and it is better evidence than appearance, because it is the thing
the route is actually made of.

So this is a second pass that uses the body as the prompt:

1. Walk the climb. Note every stretch where a limb held still, on the wall, and
   matched no hold.
2. Cluster those stretches in canvas space. Drop the ones that land on a hold we
   already have, and the ones on the mats.
3. For each site left, pick a frame that can see it with the climber elsewhere,
   and ask SAM 3.1 `segment_box` what is inside a box there.

`segment_box` is the right call rather than another text prompt, because a box
prompt has no opinion about colour: told where to look, SAM segments the object
that is there, chalk and all.

## What stops it inventing holds

A smear — a foot flat on blank wall, no hold under it — is the failure mode, and
two things guard against it. The site has to be somewhere a limb *stopped*, for
longer than an ordinary hold dwell, which a foot sliding down a smear does not
do. And what comes back has to be the size and shape of a hold: a mask that
covers a tenth of the wall is the wall, and a mask of four pixels is noise.
Anything outside those bounds is dropped and reported, so a recovered hold is
always a decision you can see rather than one that quietly appears.

Recovered holds carry ``"recovered": True``, and `holds.png` outlines them
differently, because a hold found this way rests on weaker evidence than one
thirteen sightings agreed on and should not be able to hide among them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import cv2
import numpy as np

from src import holds as holds_mod
from src.camera import Track


@dataclass
class Site:
    """Somewhere a limb stopped and nothing was found. A candidate hold."""

    centre: np.ndarray            # normalized canvas coords
    frames: list[int]
    limbs: list[str]

    @property
    def seconds(self) -> float:
        return float(len(self.frames))


def _clusters(points: list[tuple[np.ndarray, int, str]], radius: float
              ) -> list[Site]:
    """Greedy spatial clustering of lingering-limb samples."""
    sites: list[Site] = []
    for point, frame, limb in points:
        for site in sites:
            if np.linalg.norm(site.centre - point) <= radius:
                n = len(site.frames)
                site.centre = (site.centre * n + point) / (n + 1)
                site.frames.append(frame)
                if limb not in site.limbs:
                    site.limbs.append(limb)
                break
        else:
            sites.append(Site(centre=point.copy(), frames=[frame], limbs=[limb]))
    return sites


def candidates(poses, route: list[dict], geometry, *, fps: float, cfg,
               hold_unit: float, toe_offset: float, wall,
               window: tuple[int | None, int | None] = (None, None)) -> list[Site]:
    """Where a limb lingered on the wall and matched nothing. Sorted by dwell.

    Two filters do most of the work, and both are about not inventing holds.

    "On the wall" is checked against the route's own dilated footprint, which
    keeps a foot resting on the mat out of the list. It is the same mask
    `pose.pick_track` uses to decide who the climber is.

    *window* is the climb itself — pulling on to topping out. Outside it a still
    limb means nothing: before the start the climber is standing on the ground,
    and afterwards they are sitting on top of the boulder with a hand flat on
    the summit, which is a long, still, hold-free contact and not a hold.
    """
    from src.climb import LIMBS, limb_point

    need = max(1, int(round(cfg.RECOVER_DWELL_SECONDS * fps)))
    radius = cfg.RECOVER_CLUSTER_RADIUS * hold_unit
    samples: list[tuple[np.ndarray, int, str]] = []
    low = window[0] if window[0] is not None else -1
    high = window[1] if window[1] is not None else 10 ** 9

    for limb in LIMBS:
        run: list[tuple[np.ndarray, int]] = []
        for frame in poses.frames():
            if not low <= frame <= high:
                if len(run) >= need:
                    for position, at in run:
                        samples.append((position, at, limb.key))
                run = []
                continue
            point = limb_point(poses, frame, limb, cfg=cfg, toe_offset=toe_offset)
            loose = None
            if point is not None:
                loose = geometry.best(*point, margin=cfg.HOLD_RELEASE_MARGIN * hold_unit)

            still = False
            if point is not None and loose is None:
                position = np.array(point)
                # "Held still" is measured against the hold scale: a limb that
                # wandered further than a hold's width was moving past, not
                # resting on something.
                if not run or np.linalg.norm(position - run[0][0]) <= radius:
                    run.append((position, frame))
                    still = True
            if not still:
                if len(run) >= need:
                    for position, at in run:
                        samples.append((position, at, limb.key))
                run = []
        if len(run) >= need:
            for position, at in run:
                samples.append((position, at, limb.key))

    sites = _clusters(samples, radius)

    kept = []
    for site in sites:
        if len(site.frames) < need:
            continue
        if holds_mod.wall_overlap(
                [site.centre[0] - hold_unit, site.centre[1] - hold_unit,
                 2 * hold_unit, 2 * hold_unit], wall) < cfg.RECOVER_MIN_ON_WALL:
            continue
        # Already explained by a hold we have, generously: the point of this pass
        # is holds that are missing, not a second opinion on the ones that are not.
        if any(geometry.depth(hold, *site.centre) >= -cfg.RECOVER_NEAR_HOLD * hold_unit
               for hold in route):
            continue
        kept.append(site)
    return sorted(kept, key=lambda s: -len(s.frames))


def _best_frame(site: Site, poses, camera: Track, *, inset: float = 0.12
                ) -> int | None:
    """A frame that sees this site with the climber as far from it as possible.

    The site is somewhere a hand or foot was, so in most frames a hand or foot is
    on top of it. Asking SAM to segment the hold under a shoe gets a shoe. The
    climber's own keypoints say which frames are clear, and the best of those is
    a frame where they have moved on and the site is in open view.
    """
    best, best_gap = None, -1.0
    for frame in poses.frames():
        if frame >= camera.n_frames:
            continue
        local = camera.to_frame(site.centre.reshape(1, 2), frame)[0]
        if not (inset <= local[0] <= 1 - inset and inset <= local[1] <= 1 - inset):
            continue
        points = poses.kpts[frame][poses.valid[frame]]
        if not len(points):
            continue
        gap = float(np.min(np.linalg.norm(points - site.centre, axis=1)))
        if gap > best_gap:
            best, best_gap = frame, gap
    return best


def _frame_image(video, index: int) -> np.ndarray | None:
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, image = capture.read()
    capture.release()
    return image if ok else None


def _segment_box(client, image: np.ndarray, box, *, model: str
                 ) -> tuple[list[dict], dict | None, dict | None]:
    """SAM 3.1 `segment_box`: what object is inside this box. No colour involved."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [{
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{holds_mod.encode_jpeg(image)}"},
        }]}],
        response_format={"type": "json_object"},
        extra_body={"method": "segment_box",
                    "method_params": {"bbox_xywh": [float(v) for v in box],
                                      "mask_format": "png"}},
    )
    payload = json.loads(response.choices[0].message.content)
    content = payload.get("content") or {}
    usage = response.usage.model_dump() if response.usage else None
    return content.get("items") or [], content.get("mask"), usage


def _pick(items, mask, site: Site, camera: Track, frame: int, route: list[dict], *,
          hold_unit: float, aspect: float, cfg):
    """The returned instance that is actually the hold at *site*, or ``(None, why)``.

    `segment_box` does not hand back one object. The box steers it — a box at the
    site and a box in the far corner of the same frame come back with 50 and 25
    instances respectively — but what arrives is a proposal set over the whole
    image, so taking the highest-scoring row lands on whatever SAM was surest of
    anywhere in shot. On the first run that was a hold two-thirds of the frame
    away, which then got placed on the canvas nowhere near the foot that
    prompted the search.

    So the row is chosen by *position*: the instance whose outline the site falls
    in, or failing that the nearest one within a hold's width. Instances that
    coincide with a hold we already have are skipped, since the point of the
    pass is what is missing.
    """
    labels = holds_mod.decode_label_map(mask)
    local = camera.to_frame(site.centre.reshape(1, 2), frame)[0]

    best = None
    for item in items:
        box = item.get("bbox_xywh")
        if not box:
            continue
        # Cheap rejection on the box before decoding the mask: at fifty
        # instances a frame, decoding every one of them to find the near ones is
        # most of the work for none of the answer.
        cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
        if np.hypot((cx - local[0]) * aspect, cy - local[1]) > cfg.RECOVER_SEARCH_RADIUS:
            continue

        polygon = np.zeros((0, 2))
        blob = holds_mod.instance_mask(labels, item)
        if blob is not None:
            contour = holds_mod._contour(blob)
            if len(contour) >= 3:
                polygon = contour / [blob.shape[1], blob.shape[0]]
        if len(polygon) < 3:
            bx, by, bw, bh = box
            polygon = np.array([[bx, by], [bx + bw, by],
                                [bx + bw, by + bh], [bx, by + bh]])

        canvas = camera.to_canvas(polygon, frame)
        lo, hi = canvas.min(axis=0), canvas.max(axis=0)
        bbox = [float(lo[0]), float(lo[1]), float(hi[0] - lo[0]), float(hi[1] - lo[1])]

        span = max(bbox[2], bbox[3]) / hold_unit
        if not cfg.RECOVER_MIN_SPAN <= span <= cfg.RECOVER_MAX_SPAN:
            continue
        if any(holds_mod._iou(bbox, hold["bbox"]) > cfg.RECOVER_SAME_HOLD_IOU
               for hold in route):
            continue

        # Signed distance from the site to this outline, isotropic. Positive is
        # inside, so the max over candidates prefers the one the foot was on.
        ring = canvas.astype(np.float32).copy()
        ring[:, 0] *= aspect
        depth = float(cv2.pointPolygonTest(ring.reshape(-1, 1, 2),
                                           (float(site.centre[0]) * aspect,
                                            float(site.centre[1])), True))
        if depth < -cfg.RECOVER_SITE_TOLERANCE * hold_unit:
            continue
        if best is None or depth > best[0]:
            best = (depth, canvas, bbox, float(item.get("score") or 0.0))

    if best is None:
        return None, "no returned instance is a hold-sized object at that spot"
    return (best[1], best[2], best[3]), ""


def recover(client, video, sites: list[Site], poses, camera: Track, *, model: str,
            hold_unit: float, aspect: float, route: list[dict], cfg,
            console=None) -> tuple[list[dict], list[dict], list[dict]]:
    """Box-prompt each site. ``(holds, rejected, usages)``, holds in canvas coords."""
    found: list[dict] = []
    rejected: list[dict] = []
    usages: list[dict] = []

    for site in sites[:cfg.RECOVER_MAX_SITES]:
        frame = _best_frame(site, poses, camera)
        if frame is None:
            rejected.append({"site": site.centre.round(3).tolist(),
                             "reason": "never in clear view"})
            continue
        image = _frame_image(video, frame)
        if image is None:
            continue

        # The box is built on the canvas and carried into the frame, so it is a
        # constant size *on the wall* however zoomed that frame happened to be.
        half = cfg.RECOVER_BOX_SIZE * hold_unit / 2
        corners = np.array([[site.centre[0] - half, site.centre[1] - half],
                            [site.centre[0] + half, site.centre[1] + half]])
        local = camera.to_frame(corners, frame)
        x0, y0 = np.clip(local.min(axis=0), 0.0, 1.0)
        x1, y1 = np.clip(local.max(axis=0), 0.0, 1.0)
        if x1 - x0 < 1e-3 or y1 - y0 < 1e-3:
            continue

        items, mask, usage = _segment_box(
            client, image, [x0, y0, x1 - x0, y1 - y0], model=model)
        if usage:
            usages.append(usage)
        if not items:
            rejected.append({"site": site.centre.round(3).tolist(),
                             "reason": "SAM found nothing in the box"})
            continue

        chosen, why = _pick(items, mask, site, camera, frame, route,
                            hold_unit=hold_unit, aspect=aspect, cfg=cfg)
        if chosen is None:
            rejected.append({"site": site.centre.round(3).tolist(),
                             "reason": why, "frame": frame})
            continue
        canvas, bbox, score = chosen

        found.append({
            "bbox": bbox,
            "polygon": canvas.tolist(),
            "score": round(score, 4),
            "appearances": len(site.frames),
            "appearance_fraction": 1.0,
            "drift_px": 0.0,
            "tracks": [],
            "frames": sorted(site.frames),
            "recovered": True,
            "recovered_from": {"frame": int(frame), "limbs": site.limbs,
                               "dwell_frames": len(site.frames)},
        })
        if console:
            console.print(f"  [green]recovered a hold[/] at "
                          f"({site.centre[0]:.3f}, {site.centre[1]:.3f}) — "
                          f"{'/'.join(site.limbs)} rested there for "
                          f"{len(site.frames)} frames and nothing was under it; "
                          f"[dim]box-prompted SAM on frame {frame}, score "
                          f"{score:.2f}[/]")
    return found, rejected, usages
