"""Desenho sobre a imagem: pontos, malha facial e texto. Só usado por demos e depuração."""
import cv2
from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import face_mesh as mp_face_mesh

from detection import parameters as P
from detection.landmarks import FaceLandmarks, to_pixel_points


def draw_points(image, face_landmarks: FaceLandmarks, indices: list[int],
                color: tuple, radius: int) -> None:
    """Círculos cheios sobre os landmarks indicados."""
    height, width = image.shape[:2]
    for point in to_pixel_points(face_landmarks, indices, width, height):
        cv2.circle(image, (round(point[0]), round(point[1])), radius, color, -1)


def draw_feature_points(image, face_landmarks: FaceLandmarks, radius: int) -> None:
    """Os 6 pontos de cada olho (vermelho e azul) e da boca (amarelo) que entram no EAR e no MAR."""
    draw_points(image, face_landmarks, P.LEFT_EYE_INDICES, P.RED, radius)
    draw_points(image, face_landmarks, P.RIGHT_EYE_INDICES, P.BLUE, radius)
    draw_points(image, face_landmarks, P.MOUTH_INDICES, P.YELLOW, radius)


def draw_tesselation(image, face_landmarks: FaceLandmarks) -> None:
    """Malha facial inteira."""
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())


def draw_contours(image, face_landmarks: FaceLandmarks) -> None:
    """Contornos de olhos, sobrancelhas, lábios e rosto."""
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())


def put_text(image, text: str, style: dict, color: tuple | None = None) -> None:
    """Escreve texto com um estilo de parameters; `color` troca só a cor do estilo."""
    cv2.putText(image, text, style["origin"], cv2.FONT_HERSHEY_PLAIN, style["scale"],
                color if color is not None else style["color"], style["thickness"])
