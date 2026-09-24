# How the route is read

1. **The wall.** Each frame's depth is turned into 3D points. The camera's path
   is solved frame by frame by matching features against recent frames, and
   the depth maps are fused into one point cloud in meters.
2. **The holds.** SAM 3.1 tracks the route's colour through the whole video,
   giving each hold an ID. Every sighting is placed in 3D from the depth under
   it. A hold is bolted to the wall, so sightings that land far from the rest
   are the tracker mixing up two holds, and are dropped.
3. **The floor.** The largest flat surface below the camera is the floor. It
   gives "up", the wall's angle, and the line the clock is measured against.
4. **The climber.** ViTPose finds the climber in every frame. The wall is fused
   again with the climber cut out, and each joint is placed in 3D.
5. **The route.** Holds that never come near the climber are dropped. The rest
   are numbered bottom to top.
6. **The climb.** A hand or foot is on a hold when it is inside the hold's
   outline in the image *and* physically close enough in 3D. The clock starts
   when both feet leave the floor with a hand on a hold, and stops when both
   hands are on the top hold.

Code: [`src/lidar.py`](src/lidar.py) (3D), [`src/holds.py`](src/holds.py)
(SAM), [`src/climb.py`](src/climb.py) (contact and timing). Every setting, and
why its default is what it is: [`config.py`](config.py).

## A few rules worth knowing

- **Contact needs a moment, not a touch.** A hand passing over a hold hasn't
  used it, so a hold only lights up after `HOLD_DWELL_SECONDS` of contact.
- **Letting go needs a clear break.** A limb joins a hold at
  `HOLD_MASK_MARGIN` but stays on it until `HOLD_RELEASE_MARGIN`, so a hand
  settling on a hold doesn't flicker on and off.
- **The ankle isn't the foot.** Pose stops at the ankle, so
  `ANKLE_TO_TOE_OFFSET` moves the contact point down to the toes.
- **Distances are in hold heights and body heights,** not fractions of the
  frame, so zooming in doesn't make holds easier to touch.
- **Hands are checked by reach, not wrist depth.** Filmed from behind, the
  depth under a gripping wrist is the forearm in front of it, so a hand counts
  if the hold is within reach of the shoulder.

## If it comes out wrong

Look at `route3d.png` first: it shows whether the holds landed on the wall in
the right places.

| symptom | setting |
|---|---|
| "not a depth capture" | the folder needs `meta.json` and `depth.bin`; a regular video belongs in `../rock_climbing` |
| the wall looks doubled or smeared | `LIDAR_FEATURES` up, `LIDAR_MIN_INLIERS` up |
| holds missing from the route | `HOLD_MIN_SCORE` down, `HOLD_TRACK_STRIDE` down |
| one hold appears as two | `LIDAR_HOLD_MERGE_M` up |
| two nearby holds merged into one | `LIDAR_HOLD_MERGE_M` down |
| holds from the next wall over | `HOLD_SPATIAL_MARGIN` down |
| the 3D view is tilted | no floor was found; keep some floor in shot |
| a hold the climber used never lights up | `HOLD_DWELL_SECONDS` down, `HOLD_MASK_MARGIN` up |
| footholds never light up | `ANKLE_TO_TOE_OFFSET` up, `CONTACT_3D_FOOT_M` up |
| the clock starts late | `START_DWELL_SECONDS` down, `FLOOR_CLEARANCE` down |
| no top-out detected | `FINAL_HOLD_DWELL_SECONDS` down |
| the 3D camera is jumpy | `SPACE_CHASE_SMOOTH_SECONDS` up |
| the route is the wrong colour | `HOLD_COLOR` |

Comparing several attempts at the same route (`COMPARE_ATTEMPTS`) is off: each
capture has its own coordinate system, and lining them up isn't built yet.
