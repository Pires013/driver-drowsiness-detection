"""Pose da cabeça: ângulos de Euler a partir de 6 landmarks e um modelo 3D genérico (solvePnP)."""
import cv2
import numpy as np

from detection import parameters as P
from detection.landmarks import FaceLandmarks, to_pixel_points


def compute_head_pose(face_landmarks: FaceLandmarks, width: int, height: int) -> tuple[float, float, float]:
    """(yaw, pitch, roll) em graus. Roll já normalizado: 0 = cabeça reta."""
    face_2d = np.array(to_pixel_points(face_landmarks, P.HEAD_POSE_INDICES, width, height),
                       dtype=np.float64)

    focal_length = P.FOCAL_LENGTH_FACTOR * width
    camera_matrix = np.array([
        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1],
    ])
    no_distortion = np.zeros((4, 1), dtype=np.float64)

    _, rotation_vector, translation = cv2.solvePnP(P.FACE_3D_MODEL, face_2d, camera_matrix, no_distortion)
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    if translation[2, 0] > P.POSE_FRONT_MIN_TZ:
        # Ambiguidade do solvePnP: o modelo 3D tem o Z invertido, então a solução usual sai com a cabeça "atrás"
        # da câmera (tz < 0). Às vezes, sobretudo com a cabeça baixa ou de lado, ele devolve a outra solução (tz > 0),
        # que é a mesma pose girada 180° em Z (roll saltava para ~±177°). Desfazemos esse giro para sair sempre
        # na mesma representação (mediu-se: Rz(180°) leva uma solução na outra, erro de ~0,4 na norma da matriz).
        rotation_matrix = P.POSE_FRONT_SOLUTION_FIX @ rotation_matrix
    angles = cv2.RQDecomp3x3(rotation_matrix)[0]

    pitch = angles[0]   # inclinação vertical
    yaw = angles[1]     # rotação horizontal
    roll = (angles[2] % P.ROLL_PERIOD_DEG) - P.ROLL_OFFSET_DEG   # inclinação lateral
    return yaw, pitch, roll


def chin_y_pixels(face_landmarks: FaceLandmarks, height: int) -> int:
    """Altura (em pixels) do queixo na imagem; apoia a regra de queda de cabeça dos demos."""
    return int(face_landmarks.landmark[P.CHIN_INDEX].y * height)
