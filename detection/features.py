"""The 7 per-frame values (EAR left/right/mean, MAR, pitch, yaw, roll) used by collection and pipeline."""
from detection.ear import compute_ears
from detection.head_pose import compute_head_pose
from detection.landmarks import FaceLandmarks
from detection.mar import compute_mar


def compute_features(face_landmarks: FaceLandmarks, width: int, height: int) -> dict[str, float]:
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
