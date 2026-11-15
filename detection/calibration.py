"""Reference values and thresholds relative to the user/video.

Two reference methods on purpose: mean of calibration frames (webcam demo) and percentile of a whole video (offline analysis)."""
import numpy as np

from detection import parameters as P


def reference_from_first_frames(values: list[float]) -> float:
    return sum(values) / len(values)


def reference_from_percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.nanpercentile(values, percentile))


def ear_threshold(open_eye_reference: float, factor: float = P.EAR_CLOSING_FACTOR) -> float:
    return open_eye_reference * factor


def mar_threshold(rest_reference: float, factor: float = P.MAR_YAWN_FACTOR) -> float:
    return rest_reference * factor


def pitch_drop_threshold(neutral_pitch: float, offset_deg: float = P.PITCH_DROP_OFFSET_DEG) -> float:
    return neutral_pitch + offset_deg


def chin_drop_threshold(neutral_chin_y: float, offset_px: float = P.CHIN_DROP_OFFSET_PX) -> float:
    return neutral_chin_y + offset_px
