"""Settings for the basketball run. Edit here, or override per run with flags (`python main.py -h`)."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

# ── Input ────────────────────────────────────────────────────────────────────
INPUT = DATA_DIR / "input"   # a clip, or a folder of clips
VIDEO_EXTENSIONS = (".mov", ".mp4", ".m4v", ".avi", ".mkv")

# ── Conversion ───────────────────────────────────────────────────────────────
INFERENCE_HEIGHT = 1080   # what the models see
EXPORT_HEIGHT = 1080      # the output's total height: the clip on top, the court below, each half
TRIM_SECONDS = None       # e.g. 10.0 for a quick test
CONVERT_CRF = 23
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
POSE_FPS = 15.0          # detector cadence; the pose still runs on every frame
PRECISION = 4             # decimals on normalized coordinates
MAX_PERSONS = 1
PERSON_IOU_THRESHOLD = 0.6

LINE_THICKNESS = 3        # px at 720 px wide; scales with the output
POINT_RADIUS = 2

SKELETON_COLORS = {       # BGR
    "left_arm":  (238, 211, 34),    # cyan
    "right_arm": (241, 102, 99),    # indigo
    "left_leg":  (249, 121, 232),   # magenta
    "right_leg": (133, 113, 251),   # rose
    "torso":     (225, 213, 203),   # slate
}

# ── Hoop: SAM 3.1, one frame ─────────────────────────────────────────────────
# The phone stays still, so the hoop is found once.
SAM_MODEL = "facebook/sam3.1"
HOOP_PROMPT = "basketball hoop"   # finds rim + net; "basketball rim" finds nothing
HOOP_FRAME_SECONDS = None         # None = the middle of the clip
RIM_DIAMETER_M = 0.49     # outer edge, tube included: what the mask spans (inner is 0.457)

# ── Ball: SAM 3.1, around each shot only ─────────────────────────────────────
BALL_PROMPT = "basketball"
BALL_BEFORE_SECONDS = 0.6     # window start, before the hands go up
BALL_AFTER_SECONDS = 2.4      # window end
BALL_STRIDE = 1               # every frame, so the ball can't skip past the rim
BALL_MIN_SCORE = 0.5
BALL_SIZE_TOLERANCE = 0.35    # boxes this far off the ball's size are dropped (hands, net)
BALL_DRAW_SIZE_TOLERANCE = 0.6   # looser, for drawing only: keeps the ball in the hands and near the camera
BALL_EXTEND_SECONDS = 1.5     # more tracking when the ball is still at the rim
BALL_MAX_EXTENSIONS = 2

# ── Shots ────────────────────────────────────────────────────────────────────
# A shot: both wrists above the head, after both were below the shoulders.
SHOT_ARMED_BELOW = "shoulder"
SHOT_MIN_GAP_SECONDS = 1.5    # closer than this is one shot
SHOOTING_HAND = "auto"        # "auto" (the arm that straightens most), "left" or "right"
HELD_REACH = 1.5              # ball sizes from a wrist that still count as in the hands

# ── Make or miss (see how-it-works.md) ───────────────────────────────────────
RIM_INNER_FRACTION = 0.80     # share of the rim's width that counts as inside
NET_MIN_SECONDS = 0.15        # rim to net bottom: ~0.1 s falling past it, 0.2 s+ through it
NET_SIDE_TOLERANCE = 0.35     # rim widths the ball may drift sideways in the net
REBOUND_MARGIN_RIMS = 0.15    # rising this far back over the rim is an in-and-out
RIM_GAP_SECONDS = 0.5         # a crossing may span a gap this long (the ball hidden by the rim and net)
BALL_DIAMETER_M = 0.239
FRONT_OF_RIM_SIZE = 1.2       # a ball this much bigger than one at the rim is in front of it

# ── Elbow (elbow.png) ────────────────────────────────────────────────────────
# Measured in 2D, so only side-on shots count.
SIDE_ON_MAX_SHOULDER_RATIO = 0.55   # shoulder width / torso: ~0.1-0.45 side-on, ~0.6 facing
ELBOW_MIN_PROJECTED = 0.5           # skip frames where an arm segment points at the camera
SET_WINDOW_SECONDS = 0.3            # set point: deepest bend from this long before the hands go up
ELBOW_BEFORE_SECONDS = 0.8          # the plot's span around each release
ELBOW_AFTER_SECONDS = 0.4
ELBOW_STRIP = ("-0.6", "-0.4", "set", "-0.1", "0", "0.2")   # frames per shot: seconds from release, or "set"

# ── Court: an estimate from one camera ───────────────────────────────────────
CAMERA_HFOV_DEGREES = 70.0    # iPhone 1x lens, landscape
PLAYER_HEIGHT_M = 1.75        # scales every distance
COURT_YAW_DEGREES = 0.0       # turn the court if the camera isn't square to the backboard
THREE_POINT_RADIUS_M = 6.75   # FIBA; 7.24 NBA, 6.02 US high school
FREE_THROW_DISTANCE_M = 4.19  # hoop center to the free-throw line

# ── Video ────────────────────────────────────────────────────────────────────
COURT_MIN_DEPTH_M = 9.0       # court drawn past the hoop: the arc plus room for the tallies
VERDICT_SECONDS = 1.2         # how long the rim stays lit green or red
BALL_TRAIL_SECONDS = 0.5
BALL_FILL_SECONDS = 0.33      # missed detections up to this long are filled in, for drawing
PANEL_FONT = "auto"           # "auto", "opencv", or a font path
PANEL_FONT_INDEX = None
OUTPUT_CRF = 20

# ── Hot streak ───────────────────────────────────────────────────────────────
HOT_STREAK = 3                # makes in a row before the ball and rim catch fire
BALL_FIRE_SECONDS = 0.3       # length of the ball's flame
BALL_FIRE_FADE_SECONDS = 0.3  # the flame fades over this once the ball drops through
RIM_COOL_SECONDS = 1.5        # the rim's flare cools over this
RIM_WARM = 0.25               # the glow left while the streak lasts (flare = 1)
STREAK_FADE = 0.5             # the HOT STREAK! text's opacity while the ball passes behind it

# Credit, bottom-right: ("LABEL", "value") or a string. Or pass --li, --x, --ig.
CREDIT = []
CREDIT_SIZE = 20              # px at 1280 wide
CREDIT_FOOTER = 52
CREDIT_SEPARATION = 28
CREDIT_MARGIN = 22
CREDIT_COLOR = (180, 180, 180)
CREDIT_LABEL_COLOR = (120, 120, 120)

# ── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = DATA_DIR / "output"
CACHE_DIR = DATA_DIR / "cache"
RUN_STAMP_FORMAT = "%Y%m%d-%H%M%S"
