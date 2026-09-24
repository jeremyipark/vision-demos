"""Params for the rock climbing run.

Edit values here, then run `python main.py`. There are no command-line flags on
purpose: `run.json` in each output directory snapshots these values, so a result
can always be traced back to its settings.
"""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

# ── Caches ───────────────────────────────────────────────────────────────────
# Every expensive step is cached in CACHE_DIR, keyed on the clip and the settings
# that shaped it, so changing a setting already misses on its own. Set one of
# these to False to make that step redo its work anyway. The gateway ones cost
# money; the local ones cost time.
REUSE_HOLDS = True       # SAM 3.1 hold tracking         (gateway)
REUSE_FLOOR = True       # SAM 3.1 floor segmentation    (gateway)
REUSE_POSES = True       # ViTPose                       (gateway)
REUSE_CAMERA = True      # camera track + still check    (local)
REUSE_WALL = True        # stitched wall image           (local)
FORCE_RECONVERT = False   # True re-runs ffmpeg even when a cached MP4 matches

# ── Input ────────────────────────────────────────────────────────────────────
# Batch mode is what makes the comparison possible. Every clip in INPUT_DIR is
# treated as another attempt at the *same* route — same wall, same colour, same
# problem — so each render can finish on a panel holding that attempt's sequence
# of holds against the others. Four tries up one VB is four different sets of
# holds, and the whole point of the panel is that you cannot see that from any
# one of the four videos.
#
# Off, INPUT_VIDEO alone is processed and there is nothing to compare against,
# so the completion panel is skipped.
BATCH_MODE = True
INPUT_DIR = DATA_DIR / "input" / "current"
INPUT_SUFFIXES = (".mov", ".mp4", ".m4v", ".avi")

# Render one attempt instead of all of them. Every clip is still analyzed, since
# the card each one ends on is made of all of them; this skips only the drawing,
# which is the part that takes the time. Give the attempt number as the card
# labels it (1-based, by filename) or a clip name. None renders every attempt.
RENDER_ONLY = None           # e.g. 2, or "IMG_8843"

# Used when BATCH_MODE is False.
INPUT_VIDEO = DATA_DIR / "input" / "climbing.mov"

# ── Conversion (.mov -> .mp4) ────────────────────────────────────────────────
# Always converted: ffmpeg applies the source rotation, OpenCV ignores it. This
# clip is stored 3840x2160 with a -90 rotation, so the decoded frame is the
# portrait 2160x3840 it was actually shot as.
INFERENCE_HEIGHT = 1080   # uploaded to both models; ViTPose caps at a 2048px long edge
# Rendered; each panel is one frame wide, so the output is 2x this across.
# Capped rather than left at the source's own 3840: the right panel is a canvas
# built from 1080p frames, so a 4K export upsamples it for nothing and turns a
# 15 MB file into 290 MB.
EXPORT_HEIGHT = 1080
TRIM_SECONDS = None       # e.g. 8.0 for a fast test run
CONVERT_CRF = 23

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

# ── The camera ───────────────────────────────────────────────────────────────
# What makes a handheld clip work. Every frame is matched to one reference frame
# and the resulting homography puts it on a shared canvas, so the wall has fixed
# coordinates again even though the camera does not. A clip off a tripod needs
# no special case: every homography comes out as the identity, the canvas is the
# frame, and everything downstream reads exactly as it would have.
#
# It is worth knowing *why* a homography is enough, because it is not a
# compromise here. `tools/parallax_check.py` fits one between two frames and
# reports the residual by depth band; between the first and last frames of a
# 43-second handheld clip it gives 0.74px on the ceiling trusses, 0.54px on the
# prow, 0.48px on the main face and 2.10px on the mats — four surfaces metres
# apart in depth, all but satisfied by a single planar map. Only a camera whose
# optical centre barely moves can do that, and for pure rotation a homography
# is exact rather than approximate. The operator standing on the mat, panning
# and zooming to follow their friend up, is that camera.
#
# Walking along the wall is not. Run the check on a clip like that first: a
# ceiling band that blows up to tens of pixels is the camera translating, and
# no flat canvas will hold it.

# None picks the most *central* frame — the one with the best median inlier
# count against a spread of others — which is the view the canvas is built
# around. Pin an integer to force one.
CAMERA_REFERENCE = None
CAMERA_PROBES = 12            # candidates considered when picking the reference

# The probes double as a stillness test. If no probe frame's corners move more
# than this many pixels against any other, the camera was on a tripod or a water
# bottle: the per-frame pass is skipped and every frame gets the identity, in
# ~1s instead of ~25s per 30s clip.
#
# Pixels at INFERENCE_HEIGHT. Measured on real clips at 1080: four water-bottle
# takes gave 0.8, 1.9, 2.6 and 3.7px (the phone settling, plus fitting noise
# magnified at the corners); a handheld take gave 125px. 5px sits in that gap
# and is under a quarter of a hold's height — less than the 2-7px the full pass
# itself disagrees with itself by. None always runs the full pass.
CAMERA_STILL_PX = 5.0

CAMERA_FEATURES = 3000        # SIFT keypoints per frame
CAMERA_MATCH_RATIO = 0.75     # Lowe ratio; lower is stricter
CAMERA_RANSAC_PX = 3.0        # MAGSAC++ inlier threshold, in source pixels
CAMERA_MIN_INLIERS = 60       # below this a frame falls back to chaining

# Temporal smoothing of the solved path, in frames, applied to the image corners
# rather than to the matrix entries — see `_corner_smooth`. This removes the
# last few tenths of a pixel of per-frame estimation noise, which the eye reads
# as the route shimmering against a wall that is plainly not moving.
CAMERA_SMOOTH_SIGMA = 1.5

CAMERA_CANVAS_SCALE = 1.0     # canvas pixels per reference-frame pixel
CAMERA_CANVAS_LIMIT = 3.0     # cap the canvas at this many frame-widths of slop

# ── The wall canvas ──────────────────────────────────────────────────────────
# The right panel's backdrop: every frame warped onto the canvas and reduced per
# pixel. The reduction is a median, which is what erases the climber — they are
# somewhere different in every frame, so no wall pixel sees them in more than a
# handful of the stack, and a median ignores a minority outright.
WALL_STRIDE = 6           # every nth frame considered
WALL_MAX_SAMPLES = 160    # …capped here; the whole stack is held to take a median
WALL_SCALE = 1.0          # canvas pixels per canvas pixel, for a finer mosaic
WALL_SHARPNESS_WEIGHT = True   # favour the frames that resolved each pixel best
WALL_MIN_COVERAGE = 3     # canvas seen by fewer frames than this is cropped away

# ── Gateway ──────────────────────────────────────────────────────────────────
GATEWAY_BASE_URL = "https://gateway.vlm.run/v1/openai"
REQUEST_TIMEOUT = 1800.0   # seconds; video pose is minutes, not seconds

# ── The holds: SAM 3.1 ───────────────────────────────────────────────────────
# Promptable segmentation. The prompt is the whole route definition — change
# the colour and the demo follows a different route up the same wall.
HOLD_MODEL = "facebook/sam3.1"
HOLD_COLOR = "green"                        # the route being climbed

# Per-clip override, keyed on the file's stem. A batch is normally one route
# filmed several times, so one colour serves — but a folder of unrelated clips
# is the common case while developing, and re-running the whole batch to change
# one clip's colour is the sort of friction that stops you checking.
HOLD_COLOR_BY_CLIP = {}      # e.g. {"IMG_4918": "pink"}
HOLD_PROMPT = "{color} climbing hold"       # {color} is substituted from HOLD_COLOR

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
# The tripod version sampled twelve frames in the whole clip and got away with
# it, because the wall never moved and one detection served every frame. Nothing
# about that survives a moving camera. At 30 fps a stride of 11 is a look every
# 0.37s, which is short enough that no hold moves far between two of them and
# the tracker never has to guess.
#
# The cap is the gateway's, not ours: SAM's tracker holds a frame of memory per
# sample and `video_max_frames` tops out at 128. A longer clip is covered by
# raising the stride, not the cap.
HOLD_TRACK_STRIDE = 11
HOLD_TRACK_MAX_FRAMES = 128
HOLD_MIN_SCORE = 0.40      # drop instances SAM itself is unsure about

# A `track_id` is SAM's claim that two sightings are the same hold, and on this
# clip it is usually right and occasionally very wrong — one track jumps 125px
# across the wall partway through, which is the tracker handing the id to a
# different hold. The canvas is what makes that visible: a hold is bolted to a
# wall, so every sighting of it must land on the same canvas pixels.
#
# A sighting further than this from its track's median position is dropped as an
# identity switch. In multiples of the hold's own median size, not pixels: a big
# volume's centroid legitimately wanders further than a crimp's.
HOLD_REJECT_RADIUS = 2.5
HOLD_MIN_SIGHTINGS = 8        # a track seen fewer times than this is not a hold

# Fraction of the frames that *could have seen* this hold in which it was
# actually found. The denominator is the camera track's answer, not the sample
# count: handheld, a hold off-screen for half the clip would otherwise look like
# a hold the model kept losing.
HOLD_MIN_APPEARANCE = 0.30

# SAM sometimes drops a track and picks the same hold back up under a new id.
# On the canvas those are two outlines in the same place — a question only the
# canvas can answer, since in the video they never coexist.
#
# Keep this high. The genuine re-identifications here overlap at 0.85+, because
# they are literally the same hold twice; the pair that overlaps at 0.30 is the
# two long rails set parallel a hand's width apart, and merging those loses a
# hold and invents a shape spanning both.
HOLD_MERGE_IOU = 0.60

# IoU is symmetric, so it cannot see a small box nested inside a large one: a
# knob segmented off its own hold scores ~0.28 against it and becomes an extra
# hold. Drop a survivor with at least this much of its area inside a better one.
# Set to None to keep every cluster.
HOLD_NMS_CONTAINMENT = 0.80

# Per-hold shape averaging, in canvas space. Each track's surviving outlines are
# rasterized into a shared grid over its own extent, pixel-voted, then
# re-contoured — so the drawn shape is the agreement across a hundred looks from
# a hundred camera angles rather than any one frame's noisy edge.
HOLD_RASTER_SIZE = 256        # raster long-side for the vote grid
HOLD_VOTE_FRACTION = 0.5      # keep a pixel covered by >= this share of outlines
HOLD_MIN_AREA_PX = 20         # drop a voted blob smaller than this, in vote-grid px
HOLD_FILL_HOLES = True        # fill interior holes left by chalk / bolt shadows

# Drops holds outside the climber's route region: the expanded convex hull of
# the body trajectory on the canvas. A correctly-coloured hold around the corner
# is still not part of this climb.
HOLD_SPATIAL_FILTER = True
HOLD_SPATIAL_MARGIN = 0.08    # normalized margin around the body/trajectory

# ── The floor: SAM 3.1 again ─────────────────────────────────────────────────
# The clock starts when both feet leave the ground, which is the gym's own rule
# and is NOT the same as both feet being on holds: a foot smeared flat against
# the wall, on no hold at all, is a legitimate placement and the climb has begun.
# Measuring against holds misses that and starts the clock late, or never.
#
# So the floor gets segmented too — same model, one more prompt. What is kept is
# its top edge per column, not the region: the wall-floor junction is not level
# in frame (it drops 35px across our test clips), so one threshold would be
# wrong by that much at one end.
#
# With no floor found, the start rule falls back to both feet on holds.
DETECT_FLOOR = True
FLOOR_SAMPLE_FRAMES = 14      # stills the floor is read off; it does not need tracking
FLOOR_REQUEST_WORKERS = 6     # those stills are independent calls; run them together
FLOOR_PROMPT = "gray floor"   # "climbing mat" finds the wall padding instead
FLOOR_MIN_SCORE = 0.50
FLOOR_MIN_AREA = 0.02         # a floor smaller than this is something else
FLOOR_EDGE_RESOLUTION = 256   # columns the edge is stored at
# Canvas columns with fewer than this many frames agreeing are left unknown. The
# outermost columns sit at the edge of what any camera saw, and one frame's
# noisy last pixel there would anchor the line and hang a spur off each end.
FLOOR_MIN_SUPPORT = 3

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
WALL_MASK_RES = 256           # the hold mask is rasterized at this resolution
WALL_MASK_DILATE = 0.04       # dilate it by this fraction of the long side, so a
                              # torso spanning the gap between holds still counts
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
# In **hold heights** now, not fractions of the frame — see `climb.scales` for
# why the old unit stopped meaning anything once the canvas replaced the frame.
# These reproduce the tripod clip's effective tuning: 0.018 of that frame was
# 0.58 of a hold there, and the same number was 0.92 of a hold here.
HOLD_MASK_MARGIN = 0.58       # enter: the mask hugs the hold
HOLD_RELEASE_MARGIN = 0.96    # stay:  must clear this to let go
HOLD_BBOX_MARGIN = 0.58       # fallback for a hold with no usable mask

# How *close* a contact got, not just how long it lasted. The dwell above asks
# only how many frames a limb spent inside the margin, which a limb travelling
# past a hold at a constant distance satisfies as well as one resting on it: on
# IMG_8843 the left foot crossed hold 2's margin for exactly the dwell, never
# coming within 17px of the outline, and the hold activated.
#
# So a contact that never gets closer than GRAZE_DEPTH to the hold's edge is
# treated as a graze and has to last GRAZE_DWELL_SECONDS instead. Not rejected
# outright: depth alone does not separate them. The toe point is an estimate
# (ankle + ANKLE_TO_TOE_OFFSET, since COCO-17 has no foot), so a foot genuinely
# standing on a hold for seven seconds can sit a few pixels outside the mask the
# whole time — IMG_8845's left foot on hold 10 does. What the two real shallow
# contacts have that the grazes do not is duration.
#
# 0.012 sits in the gap the four attempts leave between the deepest graze
# (-0.0158) and the shallowest real contact (-0.0088), which is where it should
# be re-checked if either moves.
HOLD_GRAZE_DEPTH = 0.38       # hold heights; closer than this is a real touch
HOLD_GRAZE_DWELL_SECONDS = 1.5

# COCO-17 has no foot: the ankle keypoint sits at the joint, but the contact is
# at the toes, well below it. Without this a foothold goes unactivated while the
# foot is plainly on it, and a hold just above the ankle activates when nothing
# is touching it. Shifts ankles down by the offset for the inclusion test only.
ANKLE_LENIENCY = True
# How far below the ankle joint the contact point sits, in **climber heights**.
# A foot is about this much of a standing person. It used to be a fraction of
# the frame, which on the canvas came out at nearly a shin — it ate the whole
# gap between the toe and the mat and stopped the clock ever starting.
ANKLE_TO_TOE_OFFSET = 0.045

# The four points of contact are defined in src/climb.LIMBS (wrists 9/10,
# ankles 15/16) — they are what the utilization split is computed over.
TORSO_KP_INDICES = [5, 6, 11, 12]   # shoulder L/R, hip L/R — the midline centroid
FINAL_HOLD_WRISTS = [9, 10]         # both wrists must match the top hold

# ── Holds the prompt missed ──────────────────────────────────────────────────
# A second pass that uses the climber as the prompt. A limb that sits still on
# the wall for RECOVER_DWELL_SECONDS with nothing under it is evidence of a hold
# the colour prompt did not find — on this wall, a foothold chalked over until
# nothing about it reads green. SAM 3.1's `segment_box` is then asked what is
# inside a box there, and a box prompt has no opinion about colour.
#
# Costs one small image call per site, and only runs where there is evidence.
RECOVER_MISSED_HOLDS = True
RECOVER_DWELL_SECONDS = 1.2   # longer than an ordinary hold dwell: a smear moves
RECOVER_MAX_SITES = 6         # most sites to spend a call on, best-dwelt first

# Clustering and exclusion, all in hold heights.
RECOVER_CLUSTER_RADIUS = 1.0  # samples this close are one site
# A site this close to a known hold is taken to *be* that hold. It only has to
# cover sites sitting just outside the contact test, because the release margin
# (HOLD_RELEASE_MARGIN, 0.96 hold heights) has already excluded everything
# nearer than that — set it much higher and it swallows the case this pass
# exists for, which is a missed hold set right beside a found one. On this wall
# the chalked-over foothold sits 1.27 hold heights from its neighbour.
RECOVER_NEAR_HOLD = 1.05
RECOVER_BOX_SIZE = 4.0        # the box put to SAM, centred on the site

# `segment_box` steers on the box but answers with a proposal set over the whole
# frame — fifty instances at the site, twenty-five for a box in the far corner —
# so the row is chosen by position rather than by score. These bound that search.
RECOVER_SEARCH_RADIUS = 0.10  # frame widths; instances further out are not decoded
RECOVER_SITE_TOLERANCE = 1.0  # hold heights the site may sit outside the outline
RECOVER_SAME_HOLD_IOU = 0.35  # above this the instance is a hold we already have
RECOVER_MIN_ON_WALL = 0.25    # a site with less of the route's footprint is not on it

# What comes back has to be hold-shaped. A mask many hold-heights across is the
# panel behind the hold; a fraction of one is noise. This is the guard against a
# foot smeared on blank wall inventing a hold.
RECOVER_MIN_SPAN = 0.35
RECOVER_MAX_SPAN = 4.0

# Recovered holds are drawn with a dashed outline: they rest on one box-prompted
# look plus the climber's behaviour, not on a hundred sightings agreeing, and
# should not be able to hide among the ones that do.
RECOVERED_DASH = True

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

# Align several clips onto one hold numbering so their sequences can be read
# against each other. The alignment matches holds between clips by position,
# and each clip builds its own canvas around its own reference frame — so before
# anything is matched, every clip's wall mosaic is registered onto the first
# clip's (`camera.register`: the same feature matcher, one more homography) and
# its holds are carried across. Off a tripod that homography is the identity and
# this is exactly the old comparison; handheld it is what makes it valid at all.
#
# A clip whose wall will not register — shot from somewhere that shares too
# little of the wall with the first clip — is left out of the comparison with a
# warning rather than aligned on a guess.
COMPARE_ATTEMPTS = True
COMPARE_REGISTER_FEATURES = 6000   # SIFT keypoints per mosaic
COMPARE_REGISTER_MIN_INLIERS = 60  # below this two walls are not the same wall

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
COPY_AUDIO = True          # mux the source audio back onto the render

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

SKELETON_COLORS = {
    "arms":  (0, 200, 255),
    "torso": (0, 255, 100),
    "legs":  (255, 80, 200),
}

# ── The right panel ──────────────────────────────────────────────────────────
# The wall with the climber taken away: every hold on the route, dim until it is
# used and lit in the order it was used, over the path the body took.
# The backdrop is the wall canvas from src/mosaic.py — the whole boulder, built
# out of every frame and with the climber medianed away. The tripod version drew
# the route on black because there was nothing else to draw it on; here there is
# a photograph of the wall that no single frame of the clip contains.
#
# Dimmed, because the panel's subject is the route and not the photograph: at
# full brightness the lit holds compete with every other hold in the gym.
# Off by default, which puts the route on black exactly as the tripod version
# did: the panel's subject is the route, and a photograph of every other hold in
# the gym competes with it. The mosaic is still built and still written out as
# `wall.png` and `holds.png`, where checking the segmentation against the real
# wall is the whole point. Turn it on for a panel that shows the boulder.
ROUTE_BACKDROP = False
ROUTE_BACKDROP_DIM = 0.55     # 0 keeps the mosaic at full brightness, 1 is black

# The right panel is framed on the route, not on the whole canvas. The canvas is
# as big as the camera's wanderings made it — on this clip the boulder is about
# half of it, with ceiling above and mats below — and drawing all of that on
# black leaves the route as a small cluster in the middle of an empty panel.
ROUTE_FRAME_ON_HOLDS = True
ROUTE_FRAME_MARGIN = 0.10     # of the route's own extent, added on every side

# What the left panel is currently looking at, traced on the canvas. Once the
# operator has zoomed in, nothing else says which part of the wall the live shot
# is a detail of.
DRAW_VIEWPORT = False
VIEWPORT_COLOR = (150, 150, 150)
VIEWPORT_THICK = 1

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

# BGR. Hands one hue family, feet another, so the pairing reads before the
# labels do; left and right of a pair differ in lightness rather than hue.
#
# Deliberately away from the orange the holds are drawn in — a pip in almost the
# route's own colour reads as part of the hold rather than as a marker on it.
# Change these if you set HOLD_COLOR to something in the blue or magenta family.
LIMB_COLORS = {
    "left_hand":  (255, 225, 140),   # light azure
    "right_hand": (235, 150, 30),    # deep azure
    "left_foot":  (235, 150, 245),   # light magenta
    "right_foot": (200, 60, 190),    # deep magenta
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
MIDLINE_DOT_RADIUS = 5
MIDLINE_TRAIL_FRAMES = 45
MIDLINE_TRAIL_DIM = (0, 0, 80)

# The route line, drawn on completion: a dashed spline through the smoothed
# midline path, so what the panel finishes on is the line the body actually took.
ROUTE_SPLINE = True
ROUTE_SPLINE_COLOR = (255, 255, 255)
ROUTE_SPLINE_THICKNESS = 1
ROUTE_SPLINE_DASH = 5
ROUTE_SPLINE_GAP = 12
ROUTE_SPLINE_SIGMA = 12.0            # gaussian smoothing of the path, in frames

# Bottom-right stack: the clock over the credit. Sizes are px at the inference
# width and scale with the export.
TIMER = True
TIMER_SIZE = 26
TIMER_COLOR = (255, 255, 255)

CREDIT_TEXT = "Jeremy Park"           # drawn under the clock; None omits it
CREDIT_SIZE = 21
CREDIT_COLOR = (180, 180, 180)

PANEL_MARGIN = 18            # inset from the right and bottom edges
TIMER_CREDIT_GAP = 14

PANEL_FONT = "auto"          # "auto", "opencv" to force Hershey, or a font path
PANEL_FONT_INDEX = None      # face index inside a .ttc; None uses the default

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
# INPUT_DIR sorted by filename, ascending. IMG_8843 is what the camera called
# it, which says nothing to anyone watching; the artifacts keep the filename.
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

SAVE_HOLDS_PREVIEW = True    # holds.png: the detected route on a clean frame
SAVE_RIGHT_PANEL = True      # the right panel on its own, as well as the pair
