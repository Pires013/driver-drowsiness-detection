"""Control panel of the project: every index, threshold, factor, duration and demo layout value.

Changing a value here changes the behavior of the whole project; nothing is hard-coded in the scripts.
"""
from pathlib import Path

import numpy as np

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"                      # git-ignored (face videos)
UTA_RLDD_DIR = DATA_DIR / "uta-rldd"                  # UTA-RLDD dataset videos
FEATURES_DIR = DATA_DIR / "features"                  # per-video feature CSVs
FEATURES_TEST_DIR = DATA_DIR / "features_test"        # output of partial test runs (--max-seconds)
SANITY_DIR = DATA_DIR / "sanity_check"                # sanity-check reports and plots
SESSIONS_DIR = PROJECT_ROOT / "collection" / "sessions"     # own webcam collection sessions
TEST_PHOTOS_DIR = PROJECT_ROOT / "demos" / "test_photos"    # photos for the photo demo

# --- MediaPipe Face Mesh ---
MAX_NUM_FACES = 1                    # only the driver is tracked
REFINE_LANDMARKS = True              # needed for accurate eyelid and iris landmarks
MIN_DETECTION_CONFIDENCE = 0.5       # Face Mesh detection confidence
MIN_TRACKING_CONFIDENCE = 0.5        # Face Mesh tracking confidence

# --- Landmark indices (official MediaPipe numbering) ---
# Geometric order [P1..P6] required by the EAR and MAR formulas.
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]    # left eye points for EAR
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]    # right eye points for EAR
MOUTH_INDICES = [61, 291, 37, 84, 267, 314]          # mouth points for MAR
# Order: nose, chin, left eye, right eye, left mouth, right mouth (matches FACE_3D_MODEL)
HEAD_POSE_INDICES = [1, 152, 33, 263, 61, 291]
CHIN_INDEX = 152                     # chin landmark (head-drop rule)

# --- Head pose (solvePnP) ---
# Generic 3D face model in OpenCV convention (Y positive = down), nose at the origin.
FACE_3D_MODEL = np.array([
    [0.0, 0.0, 0.0],            # nose tip
    [0.0, 330.0, -65.0],        # chin
    [-225.0, -170.0, -135.0],   # left eye corner
    [225.0, -170.0, -135.0],    # right eye corner
    [-150.0, 150.0, -125.0],    # left mouth corner
    [150.0, 150.0, -125.0],     # right mouth corner
], dtype=np.float64)
FOCAL_LENGTH_FACTOR = 1             # approximate focal length = factor * image width
# With Y down the model is equivalent to a 180 deg rotation about X: an upright head gives roll ~ +-180.
# Subtracting half the period and wrapping to [-180, 180) makes roll = 0 mean an upright head.
ROLL_PERIOD_DEG = 360.0             # period of the roll angle
POSE_FRONT_MIN_TZ = 500.0           # tz above this = mirrored solution (good ones sit at ~-2200, mirrored at ~+2300);
                                    # tz between 0 and 500 is a non-converged fit (~1200 px error), left untouched
POSE_FRONT_SOLUTION_FIX = np.diag([-1.0, -1.0, 1.0])   # Rz(180 deg): turns a tz > 0 solution into the usual one (tz < 0)
ROLL_OFFSET_DEG = 180.0             # subtracted so that roll = 0 means an upright head

# --- Webcam demo calibration and alerts ---
CALIBRATION_FRAMES = 90              # ~3 s at 30 fps with a neutral face
EAR_DEFAULT_THRESHOLD = 0.20         # fixed threshold of the photo demo (no calibration phase)
EAR_CLOSING_FACTOR = 0.75            # EAR below 75% of the open-eye EAR = eye closing
EAR_CLOSED_FRAMES = 15               # consecutive frames below threshold to alert (~0.5 s at 30 fps)
MAR_YAWN_FACTOR = 1.6                # MAR above 1.6x the resting MAR = mouth open like a yawn
YAWN_FRAMES = 30                     # consecutive frames above threshold to alert (~1 s at 30 fps)
CRITICAL_EYE_CLOSED_MIN_S = 1.0      # Critical: eye closed for >= 1 s ...
CRITICAL_EYE_LOOKBACK_S = 3.0        # ... within the 3 s BEFORE the head drop starts, or during it
ROLL_TILT_LIMIT_DEG = 25.0           # sideways drop: |roll - reference roll| above this for 2 s WITH eye closed
PITCH_DROP_OFFSET_DEG = -18.0        # head drop = neutral pitch minus 18 deg
CHIN_DROP_OFFSET_PX = 45.0           # or chin 45 px below the neutral position
HEAD_DROP_FRAMES = 50                # consecutive drop frames to alert

# --- Own data collection (webcam) ---
BLOCK_DURATION_S = 3.0               # length of each labeled block
DEFAULT_PERSON = "eu"                # default person id
DEFAULT_SCENARIO = "padrao"          # default scenario name
LABEL_ALERT = 0                      # label: alert
LABEL_FATIGUE = 1                    # label: fatigued
BEEP_BLOCK_SAVED = (800, 200)        # (frequency Hz, duration ms)
BEEP_ALERT_MODE = (1200, 150)        # beep when switching to alert mode
BEEP_FATIGUE_MODE = (500, 300)       # beep when switching to fatigue mode
SESSION_COLUMNS = ["session_id", "person_id", "scenario", "block_id", "frame", "t",
                   "ear", "mar", "yaw", "pitch", "label"]    # columns of a session CSV

# --- Demo interface: keys, BGR colors and text styles ---
WAIT_KEY_MS = 5                      # cv2.waitKey delay
KEY_MASK = 0xFF                      # keeps the low byte of waitKey
KEY_ESC = 27                         # quit key
KEY_QUIT = ord("q")                  # quit key
KEY_ALERT_MODE = ord("0")            # collection: alert mode
KEY_FATIGUE_MODE = ord("1")          # collection: fatigue mode
KEY_PAUSE = ord("s")                 # collection: pause/save

GREEN = (0, 255, 0)                  # BGR colors
RED = (0, 0, 255)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)


def _style(origin: tuple, scale: float, color: tuple, thickness: int) -> dict:
    return {"origin": origin, "scale": scale, "color": color, "thickness": thickness}


FPS_STYLE = _style((20, 70), 3, GREEN, 3)                   # FPS counter
EAR_CALIBRATING_STYLE = _style((20, 50), 1.3, YELLOW, 2)    # EAR demo: calibrating message
EAR_BOTH_EYES_STYLE = _style((20, 120), 2, RED, 2)          # EAR demo: left/right readout
EAR_MEAN_STYLE = _style((20, 160), 2, BLUE, 2)              # EAR demo: mean readout
EAR_ALERT_STYLE = _style((20, 210), 2, RED, 2)              # EAR demo: alert message
MAR_CALIBRATING_STYLE = _style((20, 50), 1.5, YELLOW, 2)    # MAR demo: calibrating message
MAR_READOUT_STYLE = _style((20, 50), 2, CYAN, 2)            # MAR demo: readout
YAWN_ALERT_STYLE = _style((20, 100), 2, RED, 3)             # MAR demo: yawn alert
HEAD_CALIBRATING_STYLE = _style((20, 50), 1.3, YELLOW, 2)   # head demo: calibrating message
HEAD_PITCH_STYLE = _style((20, 50), 2, GREEN, 2)            # head demo: pitch readout
HEAD_YAW_STYLE = _style((20, 90), 2, CYAN, 2)               # head demo: yaw readout
HEAD_ALERT_STYLE = _style((20, 170), 2, RED, 3)             # head demo: drop alert
COLLECTION_BLOCK_STYLE = _style((20, 150), 1.8, GREEN, 2)   # collection: block status (color follows the label)
COLLECTION_STANDBY_STYLE = _style((20, 150), 1.8, CYAN, 2)  # collection: standby status
PHOTO_BOTH_EYES_STYLE = _style((30, 60), 2, RED, 2)         # photo demo: left/right readout
PHOTO_MEAN_STYLE = _style((30, 100), 2, BLUE, 2)            # photo demo: mean readout
PHOTO_STATUS_STYLE = _style((30, 150), 2, GREEN, 2)         # photo demo: status (color follows the state)

EAR_POINT_RADIUS = 2                 # landmark dot radius, EAR demo
PHOTO_POINT_RADIUS = 3               # landmark dot radius, photo demo
MAR_POINT_RADIUS = 3                 # landmark dot radius, MAR demo
HEAD_POINT_RADIUS = 4                # landmark dot radius, head demo
PHOTO_HEIGHT_PX = 720                # test photos are resized to this height

# --- UTA-RLDD pipeline: extraction ---
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}    # accepted video files
FEATURE_COLUMNS = ["ear_left", "ear_right", "ear_mean", "mar", "pitch", "yaw", "roll"]    # per-frame features
FEATURES_CSV_COLUMNS = ["participant", "label", "fold", "timestamp_s", "frame_idx",
                        "face_detected", *FEATURE_COLUMNS]    # columns of a feature CSV
EXTRACTION_LOG_COLUMNS = ["video", "fold", "participant", "label", "width", "height",
                          "rotation_deg", "nominal_fps", "frames", "duration_s", "pct_no_face",
                          "ts_non_monotonic", "processing_s", "processing_fps"]    # columns of the extraction log
FACE_FOUND = 1                       # face_detected value when a face was found
FACE_NOT_FOUND = 0                   # face_detected value when no face was found
EXTRACTION_LOG_NAME = "extraction_log.csv"    # extraction log file name
MS_PER_SECOND = 1000.0               # unit conversion

# Debug frame (landmarks drawn) saved by the extraction
DEBUG_FRAME_AT_S = 30.0              # video time of the saved debug frame
DEBUG_MAX_SIDE_PX = 1280             # debug frame is downscaled to this longest side
DEBUG_POINT_MIN_RADIUS = 2           # minimum dot radius
DEBUG_POINT_WIDTH_DIVISOR = 300      # dot radius = width / 300
DEBUG_TEXT_REFERENCE_WIDTH = 900     # text grows from this width
DEBUG_TEXT_X = 10                    # text left margin
DEBUG_TEXT_LINE_HEIGHT = 30          # text line spacing
DEBUG_TEXT_SCALE = 0.7               # text font scale
DEBUG_TEXT_THICKNESS = 2             # text stroke width

# Contact sheet of inspect_frames
TILE_SIZE = 440                      # tile side in px
TILE_GRID_COLUMNS = 4                # tiles per row
TILE_CROP_MARGIN = 1.5               # crop = face x 1.5
TILE_POINT_WIDTH_DIVISOR = 400       # dot radius = tile width / 400
TILE_NO_FACE_CROP_DIVISOR = 4        # no face: show the central half of the image
TILE_TEXT_X = 6                      # text left margin
TILE_TEXT_BASELINE = 20              # first text line baseline
TILE_TEXT_LINE_HEIGHT = 22           # text line spacing
TILE_TEXT_PADDING = 6                # padding around the text
TILE_TEXT_SCALE = 0.5                # text font scale
EAR_REPRODUCTION_TOLERANCE = 1e-4    # reprocessing must reproduce the CSV EAR within this

# --- UTA-RLDD pipeline: analysis (thresholds relative to each video) ---
# Invalid frame: becomes NaN ONLY in the analysis (raw CSVs are never modified).
# With the head almost in profile the eye landmarks fall off the eye and solvePnP can return the mirrored solution.
VALID_YAW_MAX_DEG = 45.0             # |yaw| above this: profile
VALID_ROLL_MAX_DEG = 45.0            # |roll| above this: mirrored solvePnP
VALID_EAR_EYE_MAX = 0.45             # single-eye EAR above this is implausible (p99.99 of clean frames is 0.441)

OPEN_EYE_PERCENTILE = 75             # video's "open eye" level; p90 was inflated by spurious peaks
MAR_REST_PERCENTILE = 50             # resting MAR = median (mouth closed most of the time)
EAR_CLOSED_P80_FACTOR = 0.20         # classic PERCLOS P80: eye >= 80% closed
BLINK_MIN_S = 0.05                   # shorter than this is 1-frame noise
BLINK_MAX_S = 0.50                   # >= 0.5 s is no longer a blink: long closure
BRIDGE_MAX_GAP_S = 0.3               # gap of invalid frames up to this, between two event frames, does not break the event
HEAD_NOD_MIN_S = 1.7                 # head down for >= 1.7 s = nod (equals the demo's 50 frames at 30 fps)
PITCH_SMOOTH_S = 0.3                 # rolling median of pitch before detecting nods and measuring dispersion
HEAD_NEUTRAL_PERCENTILE = 50         # neutral pitch = median of the video itself (camera changes between recordings)
PITCH_LOW_PERCENTILE = 5             # percentile used as the low-pitch level
YAWN_MIN_S = 1.0                     # mouth above threshold for >= 1 s
JUMP_DEG = 10.0                      # pitch/yaw jump between consecutive frames
MAR_PERCENTILES = (5, 25, 50, 75, 90, 95, 99)    # MAR percentiles reported
SECONDS_PER_MINUTE = 60.0            # unit conversion
PERCENT = 100                        # unit conversion

LABEL_NAMES = {0: "0 alerta", 5: "5 baixa vigilância", 10: "10 sonolento"}    # display names per label
ALERT_LABEL = 0                      # UTA-RLDD label: alert
DROWSY_LABEL = 10                    # UTA-RLDD label: drowsy
LOW_VIGILANCE_LABEL = 5              # UTA-RLDD label: low vigilance
FPS_MISMATCH_RATIO = 0.8             # fps of a participant's label-0 and label-10 videos: min/max below this is flagged

# Plots
LABEL_COLORS = {0: "#8fb4d9", 5: "#4f86c0", 10: "#1f4e8c"}   # single-hue ramp: darker = drowsier
INK_COLOR = "#2b2b2b"                # text color
GRID_COLOR = "#e3e3e3"               # grid color
YAWN_MARK_COLOR = "#c0392b"          # yawn marker color
MAR_FIXED_REFERENCE = 0.50           # old fixed threshold, drawn only as a reference
MAR_FIXED_REFERENCE_COLOR = "#999"   # reference line color
MAR_FIXED_LABEL_COLOR = "#777"       # reference label color
TITLE_FONT_SIZE = 10                 # subplot title font size
LABEL_FONT_SIZE = 8                  # axis label font size
SUPTITLE_FONT_SIZE = 11              # figure title font size
NOTE_FONT_SIZE = 7                   # footnote font size
GRID_LINE_WIDTH = 0.8                # grid line width
COMPARISON_FIGSIZE = (13, 7)         # comparison figure size (inches)
COMPARISON_DPI = 140                 # comparison figure DPI
BAR_WIDTH = 0.26                     # bar width in the comparison plot
BAR_FILL = 0.92                      # fraction of the slot a bar group fills
TIMESERIES_WIDTH = 13                # time-series figure width (inches)
TIMESERIES_ROW_HEIGHT = 2.3          # height of each time-series row (inches)
TIMESERIES_DPI = 130                 # time-series figure DPI
TIMESERIES_SMOOTH_FRAMES = 15        # rolling median for display only (~0.5 s)
TIMESERIES_LINE_WIDTH = 0.6          # time-series line width
MAR_PLOT_WIDTH = 10                  # MAR plot width (inches)
MAR_PLOT_ROW_HEIGHT = 0.38           # height per row in the MAR plot (inches)
MAR_PLOT_PADDING = 1.5               # extra figure height (inches)
MAR_PLOT_DPI = 140                   # MAR plot DPI
MAR_PLOT_THIN_LINE = 2               # width of the thin range line
MAR_PLOT_THICK_LINE = 7              # width of the thick range line
MAR_PLOT_MEDIAN_SIZE = 4             # median marker size
MAR_PLOT_THRESHOLD_SIZE = 11         # threshold marker size

# --- Shared preprocessing (training and live demo): detection/preprocessing.py ---
RESAMPLE_HZ = 10                     # resampling rate
STEP_S = 1.0 / RESAMPLE_HZ           # duration of one resampled step
MAX_INTERP_GAP_S = 0.3               # gaps up to 0.3 s are interpolated; longer ones stay missing
MAX_INTERP_GAP_STEPS = round(MAX_INTERP_GAP_S / STEP_S)    # same limit in steps
WINDOW_S = 60                        # network window length
WINDOW_STEPS = WINDOW_S * RESAMPLE_HZ    # window length in steps
MIN_OBSERVED_FRACTION = 0.5          # a window with fewer observed steps than this is discarded
N_MEASURES_BASE = 5                  # EAR, MAR, pitch, yaw, roll (means over each 0.1 s step)
MEASURE_NAMES = ["ear_mean", "mar", "pitch", "yaw", "roll"]    # base measure names
RELATIVE_CHANNELS = ["ear_rel", "mar_rel", "pitch_rel", "yaw_rel", "roll_rel", "ear_min_rel", "missing"]    # network input channels
EAR_MIN_CHANNEL = N_MEASURES_BASE    # index of the minimum-EAR channel (from cycle 2)
MODEL_NORM_CLIP = 6.0                # network normalization clipped to +-6 standard deviations

# --- Live demo (demos/main_demo.py) and event detectors (detection/events.py) ---
LIVE_CALIBRATION_S = 10              # calibration at start (and when pressing r)
LIVE_MIN_CALIBRATION_FRAMES = 30     # fewer valid frames than this: calibration restarts
LIVE_UPDATE_S = 1.0                  # the network runs once per second
KEY_RECALIBRATE = ord("r")           # key to recalibrate

EYE_CLOSED_ALERT_S = 2.0             # eye closed (EAR < 75% of reference) for >= 2 s
YAWN_MIN_S_LIVE = 1.0                # MAR > 1.6x rest for >= 1 s = yawn
YAWN_COUNT_WINDOW_MIN = 15           # yawns counted over the last 15 min (default)
FAIR_YAWN_COUNT_WINDOW_MIN = 2       # fair mode (--feira): short memory, one person after another
HEAD_DROP_MIN_S = 2.0                # pitch 18 deg below reference for >= 2 s
FACE_LOST_MIN_S = 1.5                # Critical on lost face: face absent for at least 1.5 s ...
FACE_LOST_EYE_CLOSED_MIN_S = 0.5     # ... AND eye closed for >= 0.5 s (tolerating BRIDGE_MAX_GAP_S) ...
FACE_LOST_EYE_LOOKBACK_S = 3.0       # ... in the 3 s before the face vanished (or a head drop already started with eye closed)
EVENT_BRIDGE_S = 0.3                 # an invalid/no-face gap up to 0.3 s does not break an event

# Levels: Normal < Attention < Alert < Critical
LEVEL_NAMES = ["NORMAL", "ATENCAO", "ALERTA", "CRITICO"]    # no accents: cv2.putText cannot draw them
LEVEL_ATTENTION_YAWNS = 3            # Attention: >= 3 yawns in the last 15 min
LEVEL_ATTENTION_PROB = 0.50          # or network above 50% (for LEVEL_ATTENTION_PERSIST_S; exits below ..._EXIT)
LEVEL_ATTENTION_PROB_EXIT = 0.40     # network probability below which Attention turns off
LEVEL_ATTENTION_PERSIST_S = 5.0      # Attention values are untuned (only Alert was chosen from the data)
# Network Alert: above ENTER for PERSIST_S consecutive seconds; exits only below EXIT (hysteresis).
# Chosen on out-of-fold predictions of 51 people (excluding person 44 and the 6 demo test people) in
# training/persistencia.py: alert videos <= 5% of the time in Alert, maximizing drowsy time in Alert.
# Before (no persistence): 70% / 60% / 0 s. Eye closed >= 2 s still triggers Alert immediately.
LEVEL_ALERT_PROB = 0.90              # network probability to enter Alert
LEVEL_ALERT_PROB_EXIT = 0.70         # network probability below which Alert turns off
LEVEL_ALERT_PERSIST_S = 10.0         # seconds above the entry probability before Alert
LEVEL_HOLD_S = 2.0                   # the level only drops after staying lower for 2 s (avoids flicker)
LEVEL_COLORS = [(80, 160, 60), (0, 200, 255), (0, 120, 255), (40, 40, 220)]    # BGR, one per level
PERCLOS_WINDOW_S = 60                # simple PERCLOS rule shown next to the network
EAR_PLOT_S = 30                      # seconds shown in the live eye-opening plot
SOUND_LEVEL_ALERT_REPEAT_S = 3.0     # Alert from eye closed or head drop: beep every 3 s
SOUND_NET_ALERT_REPEAT_S = 10.0      # Alert from the network only: one beep on entry, then every 10 s
SOUND_LEVEL_CRITICAL_REPEAT_S = 1.5  # Critical: beep every 1.5 s
SOUND_ALERT_HZ = 880                 # Alert beep pitch
SOUND_CRITICAL_HZ = 1320             # Critical beep pitch
SOUND_SAMPLE_RATE = 22050            # beep sample rate
SOUND_BEEP_S = 0.25                  # beep duration
LIVE_DETECT_HEIGHT_PX = 480          # frame is downscaled to this height before Face Mesh (speed)
LIVE_WINDOW_TITLE = "Deteccao de sonolencia (demo ao vivo)"    # window title

# --- Recording to validate yawn and head drop (collection/validation_recording.py) ---
OWN_DIR = DATA_DIR / "own"                            # own recordings (git-ignored like all of data/)
VALIDATION_KEYS = {ord("b"): "b", ord("h"): "h", ord("c"): "c", ord("p"): "p", ord("n"): "n"}    # marking keys
VALIDATION_EVENT_NAMES = {"b": "bocejo", "h": "queda de cabeca", "c": "olhar o celular",
                          "p": "olhar painel/retrovisor", "n": "normal"}    # event name per key
VALIDATION_FRAME_COLUMNS = ["t", "face_detected", *FEATURE_COLUMNS, "marcas"]    # per-frame CSV columns
VALIDATION_VIDEO_HEIGHT_PX = 480     # recorded video height
VALIDATION_VIDEO_FPS = 30.0          # recorded video fps
VALIDATION_FLASH_S = 2.0             # the latest mark stays highlighted on screen for 2 s
VALIDATION_MATCH_TOLERANCE_S = 2.0   # a detection matches a marked event if within 2 s of it
