# How the route is read

The same steps serve a phone on a tripod and a phone somebody is holding. Off a
tripod, step 1 finds every homography to be the identity and the canvas is
simply the frame; everything after it reads the same either way.

1. Solve the camera: match every frame directly to one reference frame, and use
   the resulting homography to place it on a shared **canvas**.
2. Build the wall: warp every frame onto the canvas and take the per-pixel
   median, which erases the climber and leaves the boulder.
3. Ask SAM 3.1 to `track` the colour prompt through the whole clip in one call —
   118 sampled frames here, one every 0.37 s, each instance carrying a
   `track_id`.
4. Warp every sighting onto the canvas. Drop the ones that land far from their
   own track's median position: a hold is bolted to a wall, so those are the
   tracker's identity switches.
5. Merge tracks whose canvas outlines nearly coincide — one hold SAM lost and
   re-acquired under a new id — then pixel-vote each survivor into one shape.
6. Drop holds outside the climber's own hull, expanded: they are on the next
   wall. Number what is left bottom to top, along the route's lean.
7. Track the climber with ViTPose on every frame, project the keypoints onto the
   canvas, and smooth them **there**.
8. Start the clock when both feet clear the floor with a hand on a hold; stop it
   when both wrists hold the top hold. Activate each hold after continuous
   contact.
9. Look for holds the prompt missed: anywhere a limb rested on the wall inside
   the climb with nothing under it, box-prompt SAM there. Anything recovered
   rejoins the route and the climb is read again.

Code: [`src/camera.py`](src/camera.py), [`src/mosaic.py`](src/mosaic.py),
[`src/holds.py`](src/holds.py), [`src/climb.py`](src/climb.py),
[`src/recover.py`](src/recover.py). Every parameter, and why each default is what
it is: [`config.py`](config.py).

## Why these rules

- **The wall needs a coordinate system before anything else can have one.**
  Handheld, a hold is at different pixels in every frame, so "hold 7" cannot be
  a location any more. Steps 1–2 give the wall one fixed frame of reference —
  the canvas — and everything downstream lives in it. The renderer projects back
  out to the moving image only at the last step.
- **A homography is exact here, and that is measured.** It is exact for a planar
  scene or a purely rotating camera; this boulder is a faceted prow, so it rests
  entirely on the second. `tools/parallax_check.py` fits one homography between
  two frames and reports the residual by depth band — ceiling, prow, face, mats.
  On this clip the worst band is 2.1 px across the full 43 seconds, which only
  happens when the optical centre does not move. A clip shot *walking past* the
  wall is the one this cannot hold: run the check first, and a ceiling band in
  the tens of pixels says the camera translated.
- **The tracker and the canvas check each other.** SAM's `track_id` carries an
  identity through motion the canvas cannot follow, using appearance information
  that independent per-frame detection throws away. The canvas supplies the
  geometry that identity has to obey, and catches the cases where it does not —
  one track on this clip jumps 126 px across the wall partway through. Neither
  alone is enough.
- **The appearance test divides by what was visible, not by what was sampled.**
  A hold off-screen for half the clip is not a hold the model kept losing. The
  camera track knows where each frame was pointing, so the question can be asked
  properly.
- **Touch tests read the canonical outline, not the frame's mask.** The frames
  where SAM loses a hold are overwhelmingly the frames where the climber is
  covering it — exactly the frames in which a touch is happening. The canvas
  outline is still there when the evidence for it is hidden.
- **A margin means one distance, because it is a distance on the wall.**
  Measuring in canvas space is half of it: under a zoom, a fraction of the frame
  is a hand's width on the wide shots and a fingertip on the tight ones, so a
  hold would get easier to touch exactly as the operator pushed in on the crux.
  Smoothing too — in frame coordinates a hand locked onto a hold still travels
  as the camera pans, so smoothing there filters the camera's motion into the
  body's.
- **And the unit is set by the scene, not the camera.** Contact margins are
  multiples of the **median hold height**, body clearances multiples of the
  **climber's own height** (`climb.scales`). The old fractions-of-the-frame
  worked while the frame was the wall; on a canvas sized by wherever the camera
  wandered, the same number was 0.58 hold-heights on the tripod clip and 0.92
  here. Two separate bugs were that one fact.
- **Hysteresis resists letting go, not transferring.** A limb keeps its hold
  until it clears a looser release margin — otherwise a settled hand's wander
  reads as regripping. But the two long rails here are set parallel *closer than
  that margin*, so a hand that matched the lower one stayed matched to it after
  visibly moving to the upper one, which then never lit up despite two seconds of
  weight on it. A hold the limb is further *inside* than its current one is not a
  candidate to be resisted; it is the answer.
- **The clock needs evidence, not perfect pose.** Both feet clear of the floor
  with a hand on a hold — one hand. The dwell counter leaks by a frame on a bad
  frame instead of resetting, because on a climber hanging a hand's width off the
  ground it never gets half a second of flawless ankles; and a foot already on a
  hold skips the floor test, since a foot on a hold is off the ground whatever
  the floor line says.
- **The climber finds the holds the prompt could not.** A colour prompt cannot
  find a hold that is no longer the colour, and this wall has one chalked over
  until it reads grey. A limb that stops there for six seconds with nothing under
  it is stronger evidence than appearance, so `segment_box` is asked what is in a
  box at that spot — a box prompt has no opinion about colour. The guards against
  inventing a hold out of a smear are that the limb must have *stopped*, inside
  the climb window, and that what comes back must be hold-shaped.
- **The route is a consensus, not a detection.** SAM's edge moves between looks
  at the same hold, so the outline is the agreement across a hundred sightings
  from a hundred camera angles. A hold nested inside a better-scoring one is
  dropped: IoU is blind to containment, and a phantom hold shifts every id above
  it.
- **The ankle is not the foot.** COCO-17 stops at the joint and contact is at the
  toes, so `ANKLE_TO_TOE_OFFSET` shifts it down for the inclusion test.
- **Contact has two thresholds, and touch is dwell.** A settled keypoint's wander
  reads as regripping against one boundary, so contact is entered at
  `HOLD_MASK_MARGIN` and held until `HOLD_RELEASE_MARGIN`, both isotropic; and a
  hand passing over a hold has not used it, so activation needs
  `HOLD_DWELL_SECONDS`. The top hold confirms on a tap.
- **The clock starts when both feet leave the floor**, not when they find holds:
  a foot smeared flat against the wall is a legitimate placement. The floor is
  segmented too, and its edge is reduced **on the canvas** — a ground line
  attached to the gym rather than to the lens, which is what the rule wanted all
  along.

## Comparing attempts

A sequence is limb-agnostic: `1 2 5 8 10` is the path up the wall, and that is
what two attempts can be laid side by side. Which limb arrived is kept as
`moves`, the per-limb beta as `per_limb_sequence`, both in `sequence.json`.
**on hold %** is contact time over the climb's duration, so the four do not sum
to 100% and each compares against another climber's same limb.

**Hold 7 has to be the same hold in every clip.** Each clip gets its own SAM
pass, so one that missed a hold produces no gap: every number above it is one too
low and nothing looks wrong. Holds are matched by centroid within
`HOLD_MATCH_DISTANCE` onto the wall the clips agree on, and every clip adopts
that numbering everywhere. Step 9 warns on a differing count, an unmatched hold,
a renumbering, or drift past `HOLD_DRIFT_WARN`.

**Every clip has its own canvas, so the canvases are registered first.** Each
clip's wall mosaic is matched onto the first clip's — the same SIFT matcher as
the camera track, one more homography — and its holds are carried across before
any centroid is compared. The mosaics are the right images for it: the climber
has been medianed out of both, so nothing on them moves. Off a tripod the
homography is the identity; handheld it is what makes two takes shot from
different spots comparable at all. A clip whose wall shares fewer than
`COMPARE_REGISTER_MIN_INLIERS` matches with the first is left out of the
comparison with a warning rather than aligned on a guess.

## If it comes out wrong

Look at `holds.png` first: nearly everything downstream is a consequence of it.
It is drawn on the canvas rather than on a frame, so it shows the whole boulder
instead of whatever the operator happened to be pointing at when the clip
started.

If the holds do not sit on the wall *at all*, the problem is upstream of SAM —
check the camera line in step 4 before touching any hold knob.

| symptom | knob |
|---|---|
| holds slide off the wall as the camera moves | the camera track: see the rows below |
| many frames reported `filled` or `chained` | `CAMERA_MIN_INLIERS` down, `CAMERA_FEATURES` up |
| the route shimmers against a still wall | `CAMERA_SMOOTH_SIGMA` up |
| the canvas is mostly empty with the wall in one corner | `CAMERA_CANVAS_LIMIT` down; one wild frame set the extent |
| the wall image is ghosted or the climber is still in it | `WALL_MAX_SAMPLES` up — the median needs a majority of clean looks |
| the wall image is soft | `WALL_SCALE` up, or check `WALL_SHARPNESS_WEIGHT` is on |
| holds missing from the route | `HOLD_MIN_APPEARANCE` down, `HOLD_TRACK_STRIDE` down |
| a hold the climber used is not detected at all | it is probably not its colour any more — `RECOVER_MISSED_HOLDS` is the pass for that; `RECOVER_DWELL_SECONDS` down, `RECOVER_NEAR_HOLD` down |
| a recovered hold appeared where there is only wall | `RECOVER_MIN_SPAN` up, `RECOVER_DWELL_SECONDS` up |
| a real missed hold was rejected as "no instance at that spot" | `RECOVER_SEARCH_RADIUS` up, `RECOVER_SITE_TOLERANCE` up |
| a hold the climber transferred to never lights up | two holds closer than `HOLD_RELEASE_MARGIN`; lower it |
| the clock starts seconds late | `ANKLE_TO_TOE_OFFSET` down, `START_DWELL_SECONDS` down, `FLOOR_CLEARANCE` down |
| every hold is "touched" at once | the margins are in hold heights now; `HOLD_MASK_MARGIN` down |
| the numbering flipped between two runs | `HOLD_LEAN_DEADBAND` up, or pin `HOLD_NUMBERING` |
| a hold flickers between two places | `HOLD_REJECT_RADIUS` down: the tracker is switching identity |
| one real hold appears as two | `HOLD_MERGE_IOU` down |
| two adjacent holds merged into one | `HOLD_MERGE_IOU` up — parallel rails need 0.6+ |
| one hold outlined and numbered twice | `HOLD_NMS_CONTAINMENT` down |
| holds from the next wall over | `HOLD_SPATIAL_MARGIN` down |
| two same-height holds numbered "backwards" | `HOLD_NUMBER_BAND` up |
| a hold the climber used never lights up | `HOLD_DWELL_SECONDS` down, `HOLD_MASK_MARGIN` up |
| footholds never light up | `ANKLE_TO_TOE_OFFSET` up |
| contact fragments, or a limb keeps a hold it has left | `HOLD_RELEASE_MARGIN` up, down |
| no top-out detected | `FINAL_HOLD_DWELL_SECONDS` down |
| the clock starts early or never | `FLOOR_CLEARANCE`; check the floor line |
| the floor line hangs a spur off the canvas edge | `FLOOR_MIN_SUPPORT` up |
| the floor was not found | `FLOOR_PROMPT`: name your gym's floor, e.g. `"blue mat"` |
| the whole clip looks washed out | HDR that was not tone-mapped; step 3 says why |
| every track dies halfway through a long clip | expected — SAM's tracker does not re-detect; segmenting handles it, see `holds.segment_plan` |
| a clip's route is the wrong colour | `HOLD_COLOR_BY_CLIP` |
| a clip was left out of the comparison | its wall did not register onto the first clip's; film from closer to the same spot, or `COMPARE_REGISTER_MIN_INLIERS` down |
| the comparison matches the wrong holds between clips | `HOLD_MATCH_DISTANCE` down; check each clip's `holds.png` |

`TONEMAP = "auto"` converts HDR sources and leaves SDR ones alone. It runs on
`libplacebo`, which needs a Vulkan device (`moltenvk` on macOS), and Homebrew's
ffmpeg has none, so check `which ffmpeg` finds the environment's one.

## The handheld clip this was tuned on

43 s, 1298 frames, 4K HLG shot on an iPhone in portrait, handheld: the operator
pans, tilts, zooms and follows the climber up a green problem on the left arête
and over the prow. All 1298 frames solved for camera pose directly against the
reference, none chained, none interpolated, worst frame 225 RANSAC inliers. SAM
returned 16 raw tracks, which consolidated to 13 holds agreeing to a median 2.1
px on a 1204 px canvas. Nine of them were used, in 24.77 s. Total gateway cost,
all three models: $0.05.
