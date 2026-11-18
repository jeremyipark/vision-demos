"""Settings for the golf run. Edit here, or override per run with flags (`python main.py -h`)."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

# ── Input ────────────────────────────────────────────────────────────────────
INPUT = DATA_DIR / "input"   # a clip, or a folder of clips
VIDEO_EXTENSIONS = (".mov", ".mp4", ".m4v", ".avi", ".mkv")

# ── Conversion ───────────────────────────────────────────────────────────────
# Long side of what the models see. The ball is a few pixels across, so keep it high.
INFERENCE_LONG_SIDE = 1920
TRIM_SECONDS = None       # e.g. 3.0 for a quick test
CONVERT_CRF = 18
FORCE_RECONVERT = False

# iPhone HDR looks washed out in OpenCV. "auto" tone-maps HDR clips, "off" never, "force" always.
TONEMAP = "auto"
TONEMAP_ALGORITHM = "bt.2446a"

# ── Gateway ──────────────────────────────────────────────────────────────────
GATEWAY_BASE_URL = "https://gateway.vlm.run/v1/openai"
REQUEST_TIMEOUT = 1800.0
REUSE_CACHE = True        # reuse a cached reply when the request matches

# ── Pose: ViTPose ────────────────────────────────────────────────────────────
POSE_MODEL = "usyd-community/vitpose-plus-large"
POSE_FPS = 15.0           # detector cadence; the pose still runs on every frame
PRECISION = 4
MAX_PERSONS = 1
PERSON_IOU_THRESHOLD = 0.6

LINE_THICKNESS = 3        # px at 720 px wide; scales with the output
POINT_RADIUS = 2

SKELETON_COLORS = {       # BGR; the same per-limb colors as the basketball demo
    "left_arm":  (238, 211, 34),    # cyan
    "right_arm": (241, 102, 99),    # indigo
    "left_leg":  (249, 121, 232),   # magenta
    "right_leg": (133, 113, 251),   # rose
    "torso":     (225, 213, 203),   # slate
}

# ── Club and ball: SAM 3.1 tracking ──────────────────────────────────────────
SAM_MODEL = "facebook/sam3.1"
CLUB_PROMPT = "golf club"
BALL_PROMPT = "golf ball"
SAM_STRIDE = 1                 # every frame: the club moves a long way between frames
CLUB_MIN_SCORE = 0.3
BALL_MIN_SCORE = 0.3

# ── Club ─────────────────────────────────────────────────────────────────────
CLUB_MAX_HAND_GAP = 0.6        # torso lengths: a club mask farther than this from the hands isn't held
CLUB_TRAIL_SECONDS = 0.5       # the clubhead's trail
CLUB_FILL_ALPHA = 0.45

# ── Ball ─────────────────────────────────────────────────────────────────────
BALL_MAX_SIZE = 0.12           # torso lengths: anything bigger isn't the ball
BALL_REST_MOVE = 0.5           # ball sizes it may wander and still be at rest on the tee
BALL_FLIGHT_MAX_STEP = 1.5     # torso lengths a ball in flight may jump between frames
FAINT_SEARCH = True            # past SAM's last sighting, follow the ball by motion (it's a few px)
FAINT_MAX_SECONDS = 2.0
TRAJECTORY_EXTEND_SECONDS = 0.5  # how far past the last sighting the fitted path is drawn
BALL_TRACE_GAP = 28            # px at 720 wide: the flight trace stops this far behind the ball,
                               # outside its ring, so the ball shows for a human to check

# ── Ball speed, launch, carry: a flight model fitted to the sightings ────────
FLIGHT_ESTIMATE = True
# The phone's focal length, px. None: from the field of view across the frame's short side
# (an iPhone's 1x lens filming video: ~41°; it's ~1460 px on a 1080-wide portrait frame).
CAMERA_FOCAL_PX = None
CAMERA_SHORT_SIDE_FOV_DEG = 40.7
# How far the phone looks down. None: from where the poles in shot converge.
CAMERA_PITCH_DEG = None
CAMERA_PITCH_FALLBACK_DEG = 15.0   # if there are no poles to read it from

# ── Video ────────────────────────────────────────────────────────────────────
SLOWMO_FACTOR = 3              # the whole swing replayed this many times slower (1 = no replay)
SLOWMO_BEFORE_SECONDS = 0.1    # the replay starts this long before takeaway...
SLOWMO_AFTER_SECONDS = None    # ...and ends this long after the finish; None = the end of the clip
# Credit: a strip along the bottom of the video and swing.png, the golfer on the left and
# your handles on the right as ("LABEL", "value") pairs. Off when both are empty.
# Or pass --golfer, --x, --li, --ig (--no-credit for none).
CREDIT_GOLFER = None           # e.g. "Your Name"
CREDIT = []                    # e.g. [("X", "@you"), ("LI", "Your Name")]
PANEL_FONT = "auto"            # "auto", "opencv", or a font path
PANEL_FONT_INDEX = None
OUTPUT_CRF = 20

# ── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = DATA_DIR / "output"
CACHE_DIR = DATA_DIR / "cache"
RUN_STAMP_FORMAT = "%Y%m%d-%H%M%S"
