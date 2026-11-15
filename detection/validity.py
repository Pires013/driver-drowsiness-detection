"""Invalid-frame rule for the analysis (raw CSVs are never modified)."""
import numpy as np

from detection import parameters as P


def is_valid_frame(face_detected: np.ndarray, yaw: np.ndarray, roll: np.ndarray,
                   ear_left: np.ndarray, ear_right: np.ndarray) -> np.ndarray:
    # NaN comparisons are False, so no-face frames come out invalid; fmax ignores a NaN eye only if the other exists
    max_eye_ear = np.fmax(ear_left, ear_right)
    return ((face_detected == P.FACE_FOUND)
            & (np.abs(yaw) <= P.VALID_YAW_MAX_DEG)
            & (np.abs(roll) <= P.VALID_ROLL_MAX_DEG)
            & (max_eye_ear <= P.VALID_EAR_EYE_MAX))
