"""Head pose: Euler angles from 6 landmarks and a generic 3D face model (solvePnP)."""
import cv2
import numpy as np

from detection import parameters as P
from detection.landmarks import FaceLandmarks, to_pixel_points


def compute_head_pose(face_landmarks: FaceLandmarks, width: int, height: int) -> tuple[float, float, float]:
    """Return (yaw, pitch, roll) in degrees; roll is normalized so 0 = upright head."""
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
        # solvePnP ambiguity: the usual solution has tz < 0, but with the head down or turned it sometimes
        # returns the mirrored one (tz > 0), the same pose rotated 180 deg about Z (roll jumped to ~+-177).
        # Undo that rotation so the output always uses the same representation.
        rotation_matrix = P.POSE_FRONT_SOLUTION_FIX @ rotation_matrix
    angles = cv2.RQDecomp3x3(rotation_matrix)[0]

    pitch = angles[0]
    yaw = angles[1]  
    roll = (angles[2] % P.ROLL_PERIOD_DEG) - P.ROLL_OFFSET_DEG
    return yaw, pitch, roll


def chin_y_pixels(face_landmarks: FaceLandmarks, height: int) -> int:
    """Chin height in image pixels (supports the demos' head-drop rule)."""
    return int(face_landmarks.landmark[P.CHIN_INDEX].y * height)
