"""MediaPipe access: Face Mesh creation, face detection and landmark-to-pixel conversion."""
import cv2
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import face_mesh as mp_face_mesh

from detection import parameters as P

FaceLandmarks = landmark_pb2.NormalizedLandmarkList
Point = tuple[float, float]


def create_face_mesh(static_image: bool = False) -> mp_face_mesh.FaceMesh:
    return mp_face_mesh.FaceMesh(
        static_image_mode=static_image,
        max_num_faces=P.MAX_NUM_FACES,
        refine_landmarks=P.REFINE_LANDMARKS,
        min_detection_confidence=P.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=P.MIN_TRACKING_CONFIDENCE,
    )


def detect_face(face_mesh: mp_face_mesh.FaceMesh, frame_bgr) -> FaceLandmarks | None:
    # MediaPipe expects RGB
    results = face_mesh.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        return results.multi_face_landmarks[0]
    return None


def to_pixel_points(face_landmarks: FaceLandmarks, indices: list[int],
                    width: int, height: int) -> list[Point]:
    # floats on purpose: rounding only happens when drawing, to avoid adding noise to EAR/MAR/pose
    return [(face_landmarks.landmark[i].x * width, face_landmarks.landmark[i].y * height)
            for i in indices]
