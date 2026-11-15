"""EAR (Eye Aspect Ratio): eye height over eye width."""
import math

from detection import parameters as P
from detection.landmarks import FaceLandmarks, Point, to_pixel_points


def ear_from_points(eye_points: list[Point]) -> float:
    p1, p2, p3, p4, p5, p6 = eye_points
    vertical_1 = math.dist(p2, p6)
    vertical_2 = math.dist(p3, p5)
    horizontal = math.dist(p1, p4)
    # corrupted frame: avoid division by zero
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def compute_ears(face_landmarks: FaceLandmarks, width: int, height: int) -> tuple[float, float, float]:
    left = ear_from_points(to_pixel_points(face_landmarks, P.LEFT_EYE_INDICES, width, height))
    right = ear_from_points(to_pixel_points(face_landmarks, P.RIGHT_EYE_INDICES, width, height))
    return left, right, (left + right) / 2.0
