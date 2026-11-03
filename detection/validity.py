"""Regra de frame inválido da análise (os CSVs brutos nunca são alterados)."""
import numpy as np

from detection import parameters as P


def is_valid_frame(face_detected: np.ndarray, yaw: np.ndarray, roll: np.ndarray,
                   ear_left: np.ndarray, ear_right: np.ndarray) -> np.ndarray:
    """Máscara de frames válidos: rosto detectado, pose não extrema nem espelhada e EAR plausível.

    Comparações com NaN dão False, então frame sem rosto também sai inválido. fmax ignora o
    NaN de um olho só se o outro existir.
    """
    max_eye_ear = np.fmax(ear_left, ear_right)
    return ((face_detected == P.FACE_FOUND)
            & (np.abs(yaw) <= P.VALID_YAW_MAX_DEG)
            & (np.abs(roll) <= P.VALID_ROLL_MAX_DEG)
            & (max_eye_ear <= P.VALID_EAR_EYE_MAX))
