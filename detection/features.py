"""Os 7 valores por frame (EAR esq/dir/médio, MAR, pitch, yaw, roll) usados pela coleta e pelo pipeline."""
from detection.ear import compute_ears
from detection.head_pose import compute_head_pose
from detection.landmarks import FaceLandmarks
from detection.mar import compute_mar


def compute_features(face_landmarks: FaceLandmarks, width: int, height: int) -> dict[str, float]:
    """Features de um frame, com as chaves de parameters.FEATURE_COLUMNS."""
    ear_left, ear_right, ear_mean = compute_ears(face_landmarks, width, height)
    yaw, pitch, roll = compute_head_pose(face_landmarks, width, height)
    return {
        "ear_left": ear_left,
        "ear_right": ear_right,
        "ear_mean": ear_mean,
        "mar": compute_mar(face_landmarks, width, height),
        "pitch": pitch,
        "yaw": yaw,
        "roll": roll,
    }
