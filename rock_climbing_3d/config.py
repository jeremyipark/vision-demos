"""Params for the rock climbing run, in 3D from a depth capture.

Edit values here, then run `python main.py`. There are no command-line flags on
purpose: `run.json` in each output directory snapshots these values, so a result
can always be traced back to its settings.
"""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

# ── Input ────────────────────────────────────────────────────────────────────
# Every input is a depth capture: a directory holding video.mov, depth.bin (a
# metric depth map per video frame) and meta.json (the camera's intrinsics),
# as the iPhone depth app records it. Another depth camera works if its
# recording is written in the same layout — see src/lidar.py. A plain video
# has no depth and is refused; the 2D demo, ../rock_climbing, is built for it.
#
# Batch mode is what makes the comparison possible. Every capture in INPUT_DIR
# is treated as another attempt at the *same* route — same wall, same colour,
# same problem — so each render can finish on a panel holding that attempt's
# sequence of holds against the others.
#
# Off, INPUT_CAPTURE alone is processed and there is nothing to compare
# against, so the completion panel is skipped.
BATCH_MODE = True
INPUT_DIR = DATA_DIR / "input" / "current"

# Render one attempt instead of all of them. Every capture is still analyzed,
# since the card each one ends on is made of all of them; this skips only the
# drawing, which is the part that takes the time. Give the attempt number as the
# card labels it (1-based, by name) or a capture name. None renders every attempt.
RENDER_ONLY = None        # e.g. 2, or a capture's folder name

# Used when BATCH_MODE is False.
INPUT_CAPTURE = DATA_DIR / "input" / "current" / "my_capture"

# ── Conversion (.mov -> .mp4) ────────────────────────────────────────────────
# Always converted, so every step downstream decodes the same frames. A capture's
# video.mov is already stored upright with no rotation tag, and is passed through
# at its own frame count: depth slab i belongs to video frame i.
INFERENCE_HEIGHT = 1080   # uploaded to both models; ViTPose caps at a 2048px long edge
# Rendered; each panel is one frame wide, so the output is 2x this across.
# Capped rather than left at the source's own resolution: the right panel is drawn
# from a cloud fused out of 320x240 depth maps, so a 4K export upsamples it for
# nothing and multiplies the file size.
EXPORT_HEIGHT = 1080
TRIM_SECONDS = None       # e.g. 8.0 for a fast test run
CONVERT_CRF = 23
FORCE_RECONVERT = False   # True re-runs ffmpeg even when a cached MP4 matches

# ── HDR ──────────────────────────────────────────────────────────────────────
# An iPhone recording in HDR writes HLG-encoded Rec. 2020, and nothing
# downstream of the conversion knows what to do with that: OpenCV reads the
# values as if they were ordinary sRGB, which lifts the midtones and flattens
# the contrast. That is the washed-out look, and it is not only cosmetic — the
# hold and pose models were trained on normal photographs, and a wall whose
# colours have been flattened is a harder wall to pick a red hold out of.
#
#   "auto"  tone-map only when the source is tagged HDR (HLG or PQ)
#   "off"   never; take the source as it is
#   "force" tone-map whatever the tags say
#
# "auto" is the one to leave it on: an SDR clip is passed through untouched, so
# the same setting is right for every video you point this at.
TONEMAP = "auto"

# How the HDR range is squeezed into SDR. `bt.2446a` is the ITU's own method
# for exactly this conversion and is the safe default. `spline` keeps midtones
# a little brighter, `hable` is filmic and rolls the highlights off harder.
TONEMAP_ALGORITHM = "bt.2446a"

# ── The wall, in three dimensions ────────────────────────────────────────────
# Every frame comes with metric depth, so nothing is triangulated. The camera
# path is RGB-D odometry, the holds are placed by the depth under their masks,
# and everything is in meters. See src/lidar.py.
REUSE_LIDAR = True            # False re-runs the odometry instead of using the cache

LIDAR_FEATURES = 2500         # SIFT keypoints per frame
LIDAR_MATCH_RATIO = 0.8       # Lowe ratio against the local map's keyframes
LIDAR_MAX_REPROJ_PX = 3.0     # PnP RANSAC inlier threshold, at inference size
LIDAR_LOCAL_MAP = 5           # keyframes each frame is matched against
LIDAR_KEYFRAME_EVERY = 12     # frames between keyframes, at most
LIDAR_MIN_INLIERS = 30        # fewer and the frame is left unplaced
LIDAR_FILL_GAP = 6            # unplaced runs this short are interpolated
# Temporal smoothing of the camera path, in frames. The per-frame PnP carries a
# millimeter of independent noise, which three meters out is a hold outline
# shimmering by a pixel. 0 disables.
LIDAR_POSE_SMOOTH_SIGMA = 1.0

# The fused wall. Every LIDAR_FUSE_STRIDE-th frame's depth map is placed in the
# world and voxelized; the climber is masked out of each one first (their pose
# box, grown by LIDAR_BODY_MARGIN), or a whole climb of them is smeared up the
# face.
LIDAR_FUSE_STRIDE = 6
LIDAR_VOXEL_M = 0.015
LIDAR_MAX_DEPTH_M = 8.0
LIDAR_BODY_MARGIN = 0.04      # normalized, around the pose box

# A sighting further than this from its hold's median position is SAM handing
# the id to a different hold. Meters now, not hold heights: depth placed it.
LIDAR_HOLD_SPREAD_M = 0.20
LIDAR_HOLD_MIN_VIEWS = 3
LIDAR_HOLD_NORMAL_RADIUS_M = 0.12   # wall around a hold its orientation is read from
LIDAR_HOLD_MERGE_M = 0.04     # two placed holds this close are one hold, picked up twice
LIDAR_HOLD_MAX_REACH = 3.0    # scene radii from the route's middle; further is a stray
# Where the ground sits when no floor plane is found in the cloud, as a
# percentile of height along the route's up-axis. Not the minimum: a few stray
# depth readings always land below everything else.
LIDAR_GROUND_PERCENTILE = 2.0

# ── The 3D panel ─────────────────────────────────────────────────────────────
# There is a real, dense, coloured wall to look at, and the climber's own
# skeleton can be placed on it in meters, so the right panel is a virtual
# camera of its own:
#   "chase"   follows the climber's torso, from the same side of the wall the
#             real phone was on, smoothed so it drifts rather than jerks
#   "orbit"   a fixed sweep across the whole route, on its own schedule
SPACE_VIEW_LIDAR = "chase"
SPACE_CHASE_SPAN_M = 3.6            # meters of wall the panel shows, top to bottom
SPACE_CHASE_SMOOTH_SECONDS = 1.0    # how slowly it follows; higher is steadier
SPACE_CHASE_AZIMUTH_FOLLOW = 0.6    # share of the real phone's angle it copies
SPACE_CHASE_MAX_AZIMUTH = 30.0      # degrees either side of face-on, at most
SPACE_CHASE_ELEVATION = 6.0         # degrees; a little above, looking slightly down
SPACE_DRAW_SKELETON = True    # the climber in 3D, from the depth under each joint
SPACE_POINT_SIZE = 2          # px at the inference width, per cloud point
SPACE_CLOUD_DIM = 0.55        # 0 keeps the wall's own colours, 1 is black
# Where the orbit looks from. A dense wall reads as solid, so a wide angle shows
# the relief instead of breaking the illusion.
SPACE_LIDAR_ORBIT_DEGREES = 70.0
SPACE_LIDAR_ELEVATION_DEGREES = 12.0

# The odometry may not place every frame — a stretch too blurred to match, say.
# A frame with no camera pose has no route at all, so exporting it gives two dead panels; this trims the export
# to the span that has something on it. False exports the whole clip, most of it
# blank, which is occasionally what you want to see.
SPACE_RENDER_COVERED_ONLY = True

SPACE_PREVIEW_AZIMUTH = -18.0  # route3d.png: slightly off-axis, so depth shows

# ── Gateway ──────────────────────────────────────────────────────────────────
GATEWAY_BASE_URL = "https://gateway.vlm.run/v1/openai"
REQUEST_TIMEOUT = 1800.0   # seconds; video pose is minutes, not seconds

# ── The holds: SAM 3.1 ───────────────────────────────────────────────────────
# Promptable segmentation. The prompt is the whole route definition — change
# the colour and the demo follows a different route up the same wall.
HOLD_MODEL = "facebook/sam3.1"
HOLD_COLOR = "orange"                        # the route being climbed

HOLD_PROMPT = "{color} climbing hold"       # {color} is substituted from HOLD_COLOR
REUSE_HOLDS = True        # False re-runs segmentation instead of using the cache

# Route metadata. Recorded into every artifact — run.json, climb.json,
# sequence.json, comparison.json — so a sequence can be traced back to the
# problem it came off. Nothing draws the grade yet; COMPARE_PANEL_SHOW_GRADE
# below is the switch when that changes.
ROUTE_GRADE = "VB"
ROUTE_NAME = None         # e.g. "the green one by the door"; None omits it

# How often the holds are looked at. SAM 3.1's `track` method takes the whole
# clip in one call and carries a `track_id` forward through it, so this is a
# sampling rate rather than a set of independent looks: every HOLD_TRACK_STRIDE
# source frames, the tracker is asked where each hold went.
#
# The camera moves, so a hold is somewhere different in every frame. At 30 fps a
# stride of 11 is a look every 0.37s, which is short enough that no hold moves
# far between two of them and the tracker never has to guess.
#
# The cap is the gateway's, not ours: SAM's tracker holds a frame of memory per
# sample and `video_max_frames` tops out at 128. A longer clip is covered by
# raising the stride, not the cap.
HOLD_TRACK_STRIDE = 11
HOLD_TRACK_MAX_FRAMES = 128
HOLD_MIN_SCORE = 0.40      # drop instances SAM itself is unsure about

# A `track_id` is SAM's claim that two sightings are the same hold, and it is
# usually right and occasionally very wrong. The depth is what makes that
# visible: a hold is bolted to a wall, so every sighting of it must land in the
# same place (LIDAR_HOLD_SPREAD_M above).

# Drops holds outside the climber's route region: each hold is projected into
# each covered frame and kept only when it enters the climber's expanded box. A
# correctly-coloured hold around the corner is still not part of this climb.
HOLD_SPATIAL_FILTER = True
HOLD_SPATIAL_MARGIN = 0.08    # normalized margin around the body/trajectory

# ── The floor ────────────────────────────────────────────────────────────────
# The clock starts when both feet leave the ground, which is the gym's own rule
# and is NOT the same as both feet being on holds: a foot smeared flat against
# the wall, on no hold at all, is a legitimate placement and the climb has begun.
# Measuring against holds misses that and starts the clock late, or never.
#
# The floor is already in the depth: the plane fitted to the cloud's mats, where
# it meets the route's own wall plane, is the wall-floor junction the clock
# measures against. No segmentation call. Off, the start rule falls back to
# both feet on holds.
DETECT_FLOOR = True

# How far above the floor line a foot has to be to count as off the ground, in
# **climber heights** (see `climb.scales`). The toe point sits about at the mat
# when standing, so this only has to clear the noise in that estimate — and a
# foot already matched to a hold skips the test entirely, since a foot on a hold
# is off the ground whatever the floor line says.
FLOOR_CLEARANCE = 0.03

# The line the clock is measured against, drawn on the left panel only — there
# it sits over the actual mat and can be checked; on the black panel it would be
# an unverifiable line crossing the contact block.
DRAW_FLOOR_LINE = True
FLOOR_LINE_COLOR = (120, 230, 255)
FLOOR_LINE_THICK = 2

# ── The climber: ViTPose ─────────────────────────────────────────────────────
POSE_MODEL = "usyd-community/vitpose-plus-large"
REUSE_POSES = True        # False re-runs pose instead of using the cache

# Pose runs on every decoded frame either way; video_fps is the *detector*
# cadence and reaches stride 1 once it is >= the decoded rate. A hand arriving
# on a hold is a few frames, so nothing here decimates.
EVERY_FRAME = True
VIDEO_FPS = 10.0          # detector cadence, used only when EVERY_FRAME is False
VIDEO_MAX_FRAMES = None   # None -> every frame of the clip when EVERY_FRAME
PRECISION = 4             # decimal places on normalized coordinates (1-8)

# The pose contract now returns one confidence per joint beside the coordinates,
# so a keypoint is no longer just visible-or-sentinel: a wrist the model placed
# behind the body comes back with a low score rather than at (0, 0).
#
# 0.0 keeps every placed joint and leaves only the (0, 0) sentinel to mark a
# joint invisible — which is how the dwell and margin numbers below were tuned,
# so it is the default. Raise it to drop joints the model is guessing at; expect
# to re-check the contact thresholds if you do, because a wrist that stops being
# valid is a wrist that stops touching holds.
POSE_MIN_KPT_SCORE = 0.0

# Which body the overlay follows. The gateway's ViTPose tracks people across
# frames, so the choice is made once per track rather than per frame: each
# track is scored on how much of it sits on the holds, and a passerby walking
# in front of the wall cannot take the lock from the climber halfway up it.
LOCK_TO_WALL = True
MIN_WALL_OVERLAP = 0.05       # a track below this is not on the wall at all

# Temporal smoothing of keypoints, in frames. ~2 removes per-frame jitter
# without visible lag. 0 disables.
POSE_SMOOTH_SIGMA = 2.0

# ── Touch detection ──────────────────────────────────────────────────────────
# A limb activates a hold by staying inside it. Dwell rather than a single
# frame, so a hand swinging past a hold on the way to another one is not a use.
# The clock starts on both feet clear of the ground with a hand on a hold, held
# for this long. Its own dwell rather than HOLD_DWELL_SECONDS: this one is
# measuring a body position against a noisy floor line, not a limb against a
# mask, and it wants to be shorter.
START_DWELL_SECONDS = 0.4
# …and the counter leaks rather than resetting. One frame where ViTPose drops an
# ankle into the mat costs a frame, not the whole run — otherwise the clock
# needs START_DWELL_SECONDS of flawless pose, which on a low start it never gets.
START_DWELL_DECAY = 1

HOLD_DWELL_SECONDS = 0.5
FINAL_HOLD_DWELL_SECONDS = 0.5    # both wrists on the top hold: the route is done
FINAL_HOLD_TOUCH_SECONDS = 0.25   # …but the top hold itself confirms on a tap.
                                  # Topping out is a match, not a rest, and a
                                  # hand that has to settle for half a second
                                  # first would have the route complete before
                                  # the panel agreed a hand was ever on it.
# These two do NOT stack any more, and the distinction is what the number means.
# Each hand needs TOUCH_SECONDS on the top hold before the tracker will say it is
# on it; the match is then measured from when both hands were *touching*, not
# from when the second one finished confirming. Counted the other way, "both
# hands for 0.8s" silently asked for 1.05s of the later hand — and on three of
# the four test attempts the match was held for 0.5-0.67s, so the route was
# climbed, the top hold lit up, both hands showed on it, and nothing completed.
#
# 0.5s is set below the shortest of those with room to spare. Raise it if a hand
# brushing the top hold on the way past is being read as a send; the console
# prints the top-out frame in step 6 either way.
FINAL_DWELL_DECAY = 0             # counter -= this per missed frame, floored at 0.
                                  # 0 forgives jitter on the completion check only;
                                  # ordinary activation still resets hard on a miss.

# Slack on the mask edge, normalized. Two thresholds, not one, because getting
# onto a hold and staying on it are different questions. A keypoint that has
# settled on a hold still wanders a few pixels a frame, and against a single
# boundary that wander reads as letting go and re-gripping several times a
# second — which would shred the contact into fragments and undercount every
# limb. So contact is ENTERED at the tight margin and HELD until the looser one
# is crossed: the grip has to clearly break, not just graze the edge.
# Both are in units of FRAME HEIGHT, and isotropic: x is scaled by the aspect
# ratio before the test, so a margin means the same distance sideways as it does
# vertically. (Testing in raw normalized coordinates made every margin an
# ellipse — on this portrait clip, 1.78x more generous vertically.)
# In **hold heights**, not fractions of the frame — see `climb.scales` for why a
# fraction of a handheld frame is not a fixed distance on the wall.
HOLD_MASK_MARGIN = 0.58       # enter: the mask hugs the hold
HOLD_RELEASE_MARGIN = 0.96    # stay:  must clear this to let go
HOLD_BBOX_MARGIN = 0.58       # fallback for a hold with no usable mask

# How *close* a contact got, not just how long it lasted. The dwell above asks
# only how many frames a limb spent inside the margin, which a limb travelling
# past a hold at a constant distance satisfies as well as one resting on it: on
# one test clip the left foot crossed a hold's margin for exactly the dwell, never
# coming within 17px of the outline, and the hold activated.
#
# So a contact that never gets closer than GRAZE_DEPTH to the hold's edge is
# treated as a graze and has to last GRAZE_DWELL_SECONDS instead. Not rejected
# outright: depth alone does not separate them. The toe point is an estimate
# (ankle + ANKLE_TO_TOE_OFFSET, since COCO-17 has no foot), so a foot genuinely
# standing on a hold for seven seconds can sit a few pixels outside the mask the
# whole time, as one test clip's left foot does. What the two real shallow
# contacts have that the grazes do not is duration.
#
# 0.012 sits in the gap the four attempts leave between the deepest graze
# (-0.0158) and the shallowest real contact (-0.0088), which is where it should
# be re-checked if either moves.
HOLD_GRAZE_DEPTH = 0.38       # hold heights; closer than this is a real touch
HOLD_GRAZE_DWELL_SECONDS = 1.5

# With LiDAR, a contact must also be physically possible. The 2D test above
# cannot tell a limb on a hold from a limb that merely overlaps it in the image;
# depth can, where it can see the limb. Unknown depth never rejects: it falls
# back to the 2D test alone.
#
# Feet: the ankle is visible from behind and reads within ~10 cm of the hold it
# stands on (median 16 cm to the hold's surface on the test capture), so it is
# gated on real distance. The limit allows for the ankle being a foot's length
# from the toes that are actually on the hold.
#
# Hands are gated on REACH, not on wrist depth, and that is a measured choice.
# Filmed from behind, a gripping hand is behind the forearm, and at 240x320 with
# iOS's smoothing the depth under the wrist is the arm: real grips read a median
# 22 cm and a p10 47 cm in front of the hold — no margin separates that from a
# hand reaching past. Set CONTACT_3D_HAND_M to a distance to gate on it anyway
# (for footage shot from the side, where the hand is visible). The reach test
# is honest about being a backstop: real grips here were at most 1.04 m from
# the shoulder, so it rejects only a hold no arm could be on.
CONTACT_3D = True
CONTACT_3D_FOOT_M = 0.25
CONTACT_3D_HAND_M = None        # wrist-depth gate; None = off (see above)
CONTACT_3D_HAND_REACH_M = 1.15  # shoulder to hold, at most
CONTACT_3D_RELEASE_M = 0.08   # extra slack to *stay* on, as the 2D margins have

# COCO-17 has no foot: the ankle keypoint sits at the joint, but the contact is
# at the toes, well below it. Without this a foothold goes unactivated while the
# foot is plainly on it, and a hold just above the ankle activates when nothing
# is touching it. Shifts ankles down by the offset for the inclusion test only.
ANKLE_LENIENCY = True
# How far below the ankle joint the contact point sits, in **climber heights**.
# A foot is about this much of a standing person. As a fraction of the frame it
# would change with every zoom, and zoomed out it comes to nearly a shin — which
# eats the whole gap between the toe and the mat and stops the clock starting.
ANKLE_TO_TOE_OFFSET = 0.045

# The four points of contact are defined in src/climb.LIMBS (wrists 9/10,
# ankles 15/16) — they are what the utilization split is computed over.
TORSO_KP_INDICES = [5, 6, 11, 12]   # shoulder L/R, hip L/R — the midline centroid
FINAL_HOLD_WRISTS = [9, 10]         # both wrists must match the top hold

# ── Hold numbering ───────────────────────────────────────────────────────────
# The numbering has to be a property of the wall, not of the climber, or two
# people's runs cannot be compared hold for hold. Bottom to top, with holds
# within HOLD_NUMBER_BAND of each other in height treated as one row.
#
# Height alone is climber-independent but not stable: on this wall two holds sit
# 0.0008 apart, so mask noise decides their order and a re-run can silently
# renumber the route. The band fixes that; the direction settles what "first" in
# a row means.
#
#   "auto" — follow the route's own lean. A route running bottom-left to
#            top-right numbers left to right, a mirrored one right to left, so
#            hold 1 is the first hold of the climb either way.
#   "ltr" / "rtl" — force it.
HOLD_NUMBERING = "auto"
HOLD_NUMBER_BAND = 0.025      # ~ one hold height; same row if closer than this
# Below this |lean| there is no lean to read and "auto" uses the fixed default.
# Without it, auto is a coin toss on a route that runs straight up: this one
# measured -0.005 over thirteen holds and +0.018 over fourteen, so recovering a
# single hold near the bottom flipped the sign and renumbered every row.
HOLD_LEAN_DEADBAND = 0.05

# Align several captures onto one hold numbering so their sequences can be read
# against each other. **Not valid yet**: the alignment matches holds between
# clips by position, which needs one coordinate system, and each capture's world
# is anchored to wherever its own camera started. Two captures are not
# comparable until their walls are registered to each other — both are metric,
# so that is one rigid transform between two fused clouds. It is a tractable
# job and it is not done, so this defaults off rather than silently aligning
# holds that have nothing to do with each other.
COMPARE_ATTEMPTS = False

# ── Same wall, four attempts ─────────────────────────────────────────────────
# Comparing sequences across clips only means something if hold 7 is the same
# lump of resin in every one of them. Each clip gets its own SAM pass, and the
# masks will not come back identical — the phone sat a little differently, the
# light moved, consensus voted a slightly different edge. That is fine. What is
# not fine is a clip that found 13 holds where the others found 14: the
# numbering is positional, so one missing hold renumbers everything above it and
# "1 2 5 8" from that clip is a sentence in a different language.
#
# So the clips are checked against each other. Holds are matched by centroid
# between the reference clip (the one that found the most) and each of the
# others; anything further away than the threshold is not the same hold.
CHECK_HOLD_CONSISTENCY = True
HOLD_MATCH_DISTANCE = 0.04    # frame heights, isotropic — about two hold widths
HOLD_CONSISTENCY_STRICT = False   # True stops the run when the walls disagree

# Matched holds that still sit further apart than this are worth saying out loud.
# Under it is the ordinary disagreement between two consensus runs on the same
# wall; over it, either the tripod moved between takes or two different holds
# were matched to each other, and both make the comparison meaningless.
HOLD_DRIFT_WARN = 0.02

# Matched holds are then re-expressed in the reference clip's numbering, so a
# clip that missed one still compares correctly on the holds it did find. A hold
# with no counterpart is drawn as "?" in the panel rather than guessed at.
ALIGN_HOLD_IDS = True

# ── Render ───────────────────────────────────────────────────────────────────
# The clip on the left, the route being read on the right, side by side.
OUTPUT_CRF = 18

# All colours are BGR, because OpenCV.
#
# The overlay is drawn in the route's own colour: on the right panel the holds
# are all there is, so painting them anything else would make the viewer
# translate. Saturated rather than literal — these sit on a black panel and over
# the holds themselves, and a washed-out orange reads as neither.
HOLD_RENDER_COLOR = {
    "orange": (0, 140, 255),
    "blue":   (255, 130, 30),
    "yellow": (0, 215, 255),
    "red":    (40, 40, 235),
    "green":  (80, 200, 60),
    "purple": (220, 80, 170),
    "pink":   (180, 105, 255),
    "white":  (245, 245, 245),
    # Black holds cannot be drawn black: the right panel's ground is black, and
    # the outline would vanish. This is the lightest grey still read as "that
    # dark one" rather than as white.
    "black":  (150, 150, 150),
}
HOLD_COLOR_FALLBACK = (0, 140, 255)

HOLD_OUTLINE_THICK = 2      # px at the inference width; scales with the export

# Two alphas, because the two panels blend against different grounds. On the
# left the fill sits over the hold itself, so it has to stay light enough to let
# the hold's own shading through — it is a highlight, not a sticker. On the
# right the ground is black, and the same 0.35 turns orange into brown: the
# colour the panel is keyed to has to survive the blend.
HOLD_FILL_ALPHA = 0.35        # left panel, over the photo
HOLD_ACTIVE_FILL_ALPHA = 0.65 # right panel, over black
HOLD_COMPLETE_ALPHA = 0.85    # the final hold, after the route is topped

# The tracker's box around the climber. Off: the skeleton already says where the
# body is, and on a wall the box mostly frames holds the climber is not on.
DRAW_PERSON_BBOX = False
PERSON_BBOX_COLOR = (0, 255, 255)   # cyan
PERSON_BBOX_THICK = 2

DRAW_FACE = False           # face keypoints read as a scribble over the face
LINE_THICKNESS = 2          # px at the inference width
POINT_RADIUS = 4

# Each arm (shoulder to wrist) and leg (hip to ankle) drawn in its limb's colour
# from LIMB_COLORS, on both panels — the same colours the limb panel and the
# finale's replay use, so a hand on a hold and the arm it is on read as one.
# False draws both arms in "arms" and both legs in "legs" below.
SKELETON_BY_LIMB = True
SKELETON_COLORS = {
    "arms":  (0, 200, 255),
    # Slate, #CBD5E1. Neutral so the four limb colours carry all the meaning;
    # grey rather than white so the white joint dots and the white route line
    # still stand out against it; cool to sit with the Aurora limbs.
    "torso": (225, 213, 203),
    "legs":  (255, 80, 200),
}

# ── The right panel ──────────────────────────────────────────────────────────
# The wall with the climber taken away: every hold on the route, dim until it is
# used and lit in the order it was used, over the path the body took.
HOLD_INACTIVE_COLOR = (55, 55, 55)   # not yet used
HOLD_INACTIVE_THICK = 1      # dim: a hint of the shape, not a claim about it
HOLD_ACTIVE_THICK = 2
# The final hold, once topped. Deliberately not the route colour — it is the one
# hold that has to say something the others do not, and on a green route the
# green it used to be was the same green as every hold under it. Amber-gold, the
# same tag colour the completion panel marks the shortest attempt in.
HOLD_COMPLETE_COLOR = (90, 210, 255)

# What the chip on each hold says.
#   "id"    — the standardized hold number (see HOLD_NUMBERING). Comparable
#             between climbers on the same route, which is the point of it.
#   "order" — the order this climber used them in. Not comparable: it is a
#             property of the run. The lighting-up sequence shows it anyway.
#   "none"  — no chip.
HOLD_LABEL = "id"
# How a hold's id is written on screen: "letter" (A, B, C … — hold 1 is A) or
# "number". Letters by default, because the finale numbers holds 1, 2, 3 in the
# order they were used, and two different "3"s on screen is a puzzle.
HOLD_LABEL_STYLE = "letter"
HOLD_LABEL_SIZE = 15                 # px at the inference width
HOLD_LABEL_COLOR = (0, 0, 0)         # on a filled chip in the hold's own colour
# Unused holds are numbered too, without a chip — dim enough to stay in the
# background, bright enough to read when someone asks "what was 3?".
HOLD_INACTIVE_LABEL_COLOR = (120, 120, 120)

# ── Limb utilization ─────────────────────────────────────────────────────────
# Which limb held which hold, and for how long. Pips under each used hold say
# which limbs used it; the block at the bottom-left keeps a running split.
LIMB_PIPS = True
LIMB_PIP_RADIUS = 4                  # px at the inference width
LIMB_PIP_GAP = 3
LIMB_PIP_OFFSET = 7                  # below the hold's lowest point

# BGR. "Aurora": four distinct hues, evenly spaced (~55°) from cyan to rose.
#
# Distinct hues, not light and dark of two, because the finale colours each
# hold by the limbs that used it and blends them where several did — two blues
# a shade apart could not be told apart, and neither could their blend.
# Hands and feet each sit side by side on the wheel, so the blends that happen
# most (both hands on a hold, both feet) come out clean — sky blue, orchid —
# rather than muddy. And the whole range stays clear of the orange route, the
# gold top hold and the wall's green volumes.
# Change these if you set HOLD_COLOR to something in the blue or magenta family.
LIMB_COLORS = {
    "left_hand":  (238, 211, 34),    # cyan     #22D3EE
    "right_hand": (241, 102, 99),    # indigo   #6366F1
    "left_foot":  (249, 121, 232),   # magenta  #E879F9
    "right_foot": (133, 113, 251),   # rose     #FB7185
}

LIMB_PANEL = True                    # the running split, bottom-left
# Two headers, one per column. The gutter between them is what makes the
# sequences read as their own column rather than as a tail on the percentage.
LIMB_PANEL_TITLE = "ON A HOLD"
LIMB_PANEL_TITLE_2 = "HOLDS USED, IN ORDER"
LIMB_PANEL_COLUMN_GUTTER = 30
LIMB_PANEL_TITLE_SIZE = 15
LIMB_PANEL_LABEL_SIZE = 15
LIMB_PANEL_VALUE_SIZE = 16
LIMB_PANEL_ROW_GAP = 7
LIMB_PANEL_BAR_WIDTH = 46            # the full-width bar, at 100%
LIMB_PANEL_BAR_HEIGHT = 9
LIMB_PANEL_BAR_BG = (38, 38, 38)
LIMB_PANEL_SWATCH = 9
LIMB_PANEL_SEQ_RIGHT_PAD = 18  # keep the sequence off the panel's right edge
# A limb currently on a hold gets its row marked, so the block reads as live
# rather than as a caption that happens to change.
LIMB_PANEL_IDLE_DIM = 0.45           # opacity of a row whose limb is holding nothing

# The body's midline: the torso centroid, with a short trail behind it.
MIDLINE_DOT_COLOR = (0, 0, 255)      # bright red
MIDLINE_TRAIL_FRAMES = 45

# The route line, drawn on completion: a dashed spline through the torso's path
# in 3D, so what the panel finishes on is the line the body actually took.
ROUTE_SPLINE = True
ROUTE_SPLINE_COLOR = (255, 255, 255)
ROUTE_SPLINE_THICKNESS = 1
ROUTE_SPLINE_DASH = 5
ROUTE_SPLINE_GAP = 12

# Bottom-right stack: the clock over the credit. Sizes are px at the inference
# width and scale with the export.
TIMER = True
TIMER_SIZE = 26
TIMER_COLOR = (255, 255, 255)

# The credit, under the clock: one entry per line, top to bottom. Each entry is
# either ("LABEL", "value") — drawn as "LABEL: value", the label a shade dimmer —
# or a plain string, drawn as it is. Any mix, any number of lines:
#
#   CREDIT = ["Jeremy Park"]                        # a name alone
#   CREDIT = [("X", "@jeremyparkphd")]              # a handle alone
#   CREDIT = ["Jeremy Park", ("IG", "@jpclimbs")]   # a name, then a handle
#   CREDIT = []                                     # no credit at all
#
# Ships blank. To add your own, fill it in — mine, for example:
#
#   CREDIT = [
#       ("LI", "Jeremy Park, PhD"),
#       ("X", "@jeremyparkphd"),
#   ]
CREDIT = []
CREDIT_SIZE = 17                   # per line; the stack grows upward from the corner
CREDIT_LINE_GAP = 5
CREDIT_COLOR = (180, 180, 180)
CREDIT_LABEL_COLOR = (120, 120, 120)   # the "LI:" / "X:" part

PANEL_MARGIN = 18            # inset from the right and bottom edges
TIMER_CREDIT_GAP = 22

# The face every panel label, number and the clock are set in. "auto" is Avenir
# Next Bold on a Mac, falling back to Arial (Windows), DejaVu or Liberation Sans
# (Linux), and finally the DejaVu Sans inside matplotlib, so it runs everywhere.
# "opencv" forces Hershey; a path picks a file. The run prints which it used.
PANEL_FONT = "auto"
PANEL_FONT_INDEX = None      # face index inside a .ttc; None uses the default

# ── The finale (LiDAR) ───────────────────────────────────────────────────────
# One second after the top-out, the 3D panel expands to fill the frame while
# the camera pulls back from the climber, swings out to one side (the swing is
# what makes the depth legible) and settles face-on on the whole route, centred.
# It replaces the climb-down. Only with SPACE_VIEW_LIDAR = "chase", and not when
# the batch completion card is showing.
FINALE = True
FINALE_START_AFTER_S = 1.0
FINALE_EXPAND_S = 1.2         # the panel widening to the full frame
FINALE_MOVE_S = 3.0           # the camera pulling back to the whole route
FINALE_HOLD_S = 2.5           # the finished route, held
# Where the camera ends up:
#   "face-on"  eases from the chase angle to square-on to the wall, route
#              centred — the wall flat, the view perpendicular to it
#   "keep"     holds the angle it had at the top-out and only pulls back
#   "swing"    arcs out to the side on the way (shows the relief), then face-on
FINALE_ANGLE = "face-on"
FINALE_SWING_DEG = 28.0       # "swing" only: how far out to the side it arcs
FINALE_DRIFT_DEG = 0.0        # slow drift through the hold; 0 holds still
FINALE_ELEVATION_DEG = 8.0
FINALE_MARGIN = 1.3           # room around the route in the final framing
FINALE_SUBTITLE = "reconstructed in 3D from iPhone LiDAR"
# Once the camera is square to the wall, the route is replayed: the lit holds
# dim, then light again one at a time in the order they were reached, each with
# a bright flash and a glow that settles, while the body's line grows to meet it
# and the sequence is written out bottom-left. The top hold is last, in gold.
FINALE_REPLAY = True
FINALE_REPLAY_DIM_S = 0.4     # the lit holds fading back before it starts
FINALE_REPLAY_STEP_S = 0.4    # between one hold lighting and the next
FINALE_REPLAY_FLASH_S = 0.25  # how quickly a hold's flash settles
FINALE_REPLAY_GLOW = 0.9      # how bright the flash gets, above lit
# In the replay each hold lights in the colour of the limb that used it
# (LIMB_COLORS), blended evenly where several did; the top hold keeps a gold
# rim. The climb itself stays in the route colour. False keeps the route colour.
FINALE_REPLAY_BY_LIMB = True
# How far the climb went, on the final view: the top hold's height off the
# floor (up), from the furthest used hold across to the top hold (across), and
# how far the torso travelled, labelled on its own curving line. Proxies on
# static holds, never placed over a hold. The torso's path is smoothed over
# FINALE_TRAVEL_SMOOTH_S first so per-frame depth noise is not counted as travel.
FINALE_TRAVEL = True
FINALE_TRAVEL_SMOOTH_S = 0.5
# The two measurements drawn as dashed legs of a right-angled triangle on the
# wall, corner on the floor beneath the top hold, no hypotenuse.
FINALE_TRAVEL_TRIANGLE = True
FINALE_TRAVEL_COLOR = (225, 213, 203)   # slate, the torso's colour: it is its path
# Clear wall between the climb and each leg, in meters: the legs are drawn
# outside the body's path, like a drawing's dimension lines, with ticks and
# faint extension lines marking exactly what the labelled distance spans.
FINALE_TRAVEL_OFFSET_M = 0.35
# The summary card of all the numbers. Off: the two legs and the torso's line
# carry them on the picture itself.
FINALE_TRAVEL_CARD = False

# ── The completion panel ─────────────────────────────────────────────────────
# What the whole batch is for. The route tops out, both panels dim behind a
# scrim, and one card comes up over the pair: how long it took, the holds this
# attempt used in order, and underneath, the same line for every other attempt
# at the same problem. The climb you just watched is the only one of the four
# you can see; the panel is the only place the other three exist.
#
# Needs BATCH_MODE and at least two attempts; with nothing to compare against it
# is skipped rather than drawn empty.
COMPARE_PANEL = True
COMPARE_PANEL_ON_ROUTE = True    # also on the right-panel-only export

# The card sits over the LEFT panel only — the clip — so the right panel is
# still readable underneath it: the finished route, every hold lit in the order
# it was used, next to the numbers explaining it. A card spanning both covered
# up the one picture the sequence refers to. Only the left panel is dimmed for
# the same reason. On the route-only export there is no left half and the card
# takes the whole frame.
#
# Every size below is therefore quoted against a card about half as wide as the
# render, which is why they are smaller than they look.
COMPARE_PANEL_ON_LEFT = True

# The clock stops with the top-out, but the clip keeps rolling for however long
# it takes to climb down — sometimes seconds, sometimes not at all. Neither is
# long enough to read a comparison off, so the last composed frame is held for
# this long past the end of the source. 0 disables, and the panel then gets
# whatever the clip had left.
COMPARE_PANEL_HOLD_SECONDS = 5.0
COMPARE_PANEL_FADE_SECONDS = 0.45   # the card fading up, so it arrives rather than cuts

COMPARE_PANEL_SCRIM = 0.80       # how far the video behind the card is taken down
COMPARE_PANEL_BG = (20, 20, 20)
COMPARE_PANEL_BORDER = (72, 72, 72)
COMPARE_PANEL_WIDTH = 0.90       # fraction of the *card's half* the card spans
COMPARE_PANEL_PAD = 36           # px at the inference width, inside the card
COMPARE_PANEL_ROW_GAP = 24
COMPARE_ATTEMPT_GAP = 42         # extra air between one attempt's row and the next
COMPARE_SECTION_GAP = 3          # multiples of the row gap, above each section rule

COMPARE_TITLE = "Completed in {elapsed:.2f}s!"
COMPARE_TITLE_SIZE = 33
COMPARE_TITLE_COLOR = (255, 255, 255)
COMPARE_SUBTITLE_SIZE = 15
COMPARE_SUBTITLE_COLOR = (150, 150, 150)

# The section rules. Deliberately *larger* than the row labels under them: they
# are the structure of the card — this climb, then the others — and the row
# label is a name inside that structure, so reading them the other way round
# makes the card a list of attempts with two captions lost in it.
COMPARE_THIS_HEADER = "THIS CLIMB"
COMPARE_OTHERS_HEADER = "OTHER ATTEMPTS"
COMPARE_HEADER_SIZE = 23
COMPARE_HEADER_COLOR = (205, 205, 205)
COMPARE_LABEL_SIZE = 17          # each row's "ATTEMPT n"
COMPARE_LABEL_COLOR = (170, 170, 170)

# ONE size for every sequence on the card, this climb's included. The panel's
# whole claim is that one of these lines is shorter than the others, and a line
# set in a bigger face than the line below it is wider *for a reason that has
# nothing to do with how many holds are in it* — which is how the shortest route
# ended up looking like the longest. So the size is solved once, for the longest
# sequence in the batch, and every row is drawn at it: width is then holds, and
# only holds.
#
# This is the target. It shrinks, never grows, and only as far as the minimum
# before a sequence is truncated instead.
COMPARE_SEQ_SIZE = 29
COMPARE_SEQ_MIN_SIZE = 12
# The arrow is checked against the font before it is drawn: Avenir Next, the
# first face this picks on macOS, has no U+2192 and Pillow renders a missing
# glyph as an empty box without complaining. The first separator the face can
# actually draw wins.
COMPARE_SEQ_SEPARATOR = " → "
COMPARE_SEQ_SEPARATOR_FALLBACKS = (" › ", " · ", " > ")
COMPARE_META_SIZE = 14           # "8 holds · 14.20s", right-aligned on each row
COMPARE_META_COLOR = (140, 140, 140)

# Rows are named by position, not by file: "ATTEMPT 2" is the second clip in
# INPUT_DIR sorted by filename, ascending. The filename is whatever the camera
# called it, which says nothing to anyone watching; the artifacts keep it.
COMPARE_ATTEMPT_LABEL = "ATTEMPT {n}"

COMPARE_PANEL_SHOW_GRADE = False  # the grade is recorded either way; this draws it

# The tag on the attempt that used the fewest holds — the line the panel exists
# to make obvious. The fastest attempt is tagged too when it is a different one;
# when the same attempt is both, it says so once.
COMPARE_SHORTEST_TAG = "SHORTEST ROUTE"
COMPARE_FASTEST_TAG = "FASTEST TIME"
COMPARE_BOTH_TAG = "SHORTEST + FASTEST"
COMPARE_TAG_SIZE = 12
COMPARE_TAG_COLOR = (25, 25, 25)         # on a filled chip
COMPARE_TAG_BG = (90, 210, 255)          # amber-gold; not the route's own colour

# Holds this attempt used that no other attempt did — and, on the other rows,
# holds they used that this one did not. The differences are the comparison; the
# holds shared with anybody else are context, so they sit back a shade.
# Two colours, not three: the card glosses the gold and nothing else, so a
# second shade of grey would be a distinction with no key to read it by.
# Used identically on every row — including this climb's. The rows have to be
# directly comparable, and a sequence that is white here and grey three lines
# down is two different encodings of the same fact. Which row is yours is said
# by the section rule and the bold ATTEMPT label instead.
COMPARE_HIGHLIGHT_DIFF = True
COMPARE_COMMON_COLOR = (170, 170, 170)   # a hold some other attempt used too
COMPARE_UNIQUE_COLOR = (90, 210, 255)    # only this row used it
COMPARE_SEP_COLOR = (95, 95, 95)         # the arrows between them

# A gold number means nothing on its own, and the whole point of the colour is
# that it is the difference between this attempt and the others. So the card
# says what it means, bottom-right, where there is room for it.
COMPARE_LEGEND = True
COMPARE_LEGEND_UNIQUE = "unique to that attempt"
COMPARE_LEGEND_SIZE = 13
COMPARE_LEGEND_COLOR = (130, 130, 130)
COMPARE_LEGEND_SWATCH = 9
COMPARE_LEGEND_GAP = 20                  # between entries, if there is ever more than one

# ── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = DATA_DIR / "output"
CACHE_DIR = DATA_DIR / "cache"     # converted MP4s + model responses, reused across runs
RUN_STAMP_FORMAT = "%Y%m%d-%H%M%S"

SAVE_HOLDS_PREVIEW = True    # route3d.png: the route face-on over the fused wall
SAVE_RIGHT_PANEL = True      # the right panel on its own, as well as the pair
