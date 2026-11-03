"""Acesso ao MediaPipe: criar o Face Mesh, detectar o rosto e converter landmarks em pixels."""
import cv2
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import face_mesh as mp_face_mesh

from detection import parameters as P

FaceLandmarks = landmark_pb2.NormalizedLandmarkList
Point = tuple[float, float]


def create_face_mesh(static_image: bool = False) -> mp_face_mesh.FaceMesh:
    """Cria o Face Mesh com a configuração única do projeto."""
    return mp_face_mesh.FaceMesh(
        static_image_mode=static_image,
        max_num_faces=P.MAX_NUM_FACES,
        refine_landmarks=P.REFINE_LANDMARKS,
        min_detection_confidence=P.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=P.MIN_TRACKING_CONFIDENCE,
    )


def detect_face(face_mesh: mp_face_mesh.FaceMesh, frame_bgr) -> FaceLandmarks | None:
    """Landmarks do rosto em um frame BGR do OpenCV, ou None se não houver rosto."""
    results = face_mesh.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))   # MediaPipe exige RGB
    if results.multi_face_landmarks:
        return results.multi_face_landmarks[0]
    return None


def to_pixel_points(face_landmarks: FaceLandmarks, indices: list[int],
                    width: int, height: int) -> list[Point]:
    """Converte landmarks normalizados (0 a 1) em coordenadas de pixel, em ponto flutuante.

    É a única conversão do projeto. Não arredonda: o arredondamento para pixel inteiro só
    acontece no desenho (drawing.draw_points), para não adicionar ruído a EAR, MAR e pose.
    """
    return [(face_landmarks.landmark[i].x * width, face_landmarks.landmark[i].y * height)
            for i in indices]
