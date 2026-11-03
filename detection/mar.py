"""MAR (Mouth Aspect Ratio): razão entre a abertura e a largura da boca."""
import math

from detection import parameters as P
from detection.landmarks import FaceLandmarks, Point, to_pixel_points


def mar_from_points(mouth_points: list[Point]) -> float:
    """MAR a partir dos 6 pontos da boca em pixels."""
    p1, p2, p3, p4, p5, p6 = mouth_points
    vertical_1 = math.dist(p3, p4)
    vertical_2 = math.dist(p5, p6)
    horizontal = math.dist(p1, p2)
    if horizontal == 0:         # frame corrompido: evita divisão por zero
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def compute_mar(face_landmarks: FaceLandmarks, width: int, height: int) -> float:
    """MAR bruto (sem calibração nem limiar) do rosto."""
    return mar_from_points(to_pixel_points(face_landmarks, P.MOUTH_INDICES, width, height))
