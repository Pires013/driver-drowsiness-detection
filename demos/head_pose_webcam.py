"""Webcam head-pose demo for head drops (q quits).

The mean pitch and chin height of the first frames (looking straight ahead) are the reference for the drop."""
from detection import parameters as P
from detection.calibration import chin_drop_threshold, pitch_drop_threshold, reference_from_first_frames
from detection.drawing import draw_points, put_text
from detection.head_pose import chin_y_pixels, compute_head_pose
from detection.landmarks import FaceLandmarks, create_face_mesh, detect_face
from detection.webcam import capture_frames, show

WINDOW_TITLE = "Head Pose Estimation"
CALIBRATING_TEXT = "CALIBRANDO HEAD... Olhe reto para a frente"
ALERT_TEXT = "ALERTA: QUEDA DE CABECA!"


def new_state() -> dict:
    return {"calibrated": False, "pitch_values": [], "chin_values": [],
            "pitch_threshold": None, "chin_threshold": None, "drop_frames": 0}


def calibrate(state: dict, pitch: float, chin_y: int) -> None:
    state["pitch_values"].append(pitch)
    state["chin_values"].append(chin_y)
    if len(state["pitch_values"]) >= P.CALIBRATION_FRAMES:
        neutral_pitch = reference_from_first_frames(state["pitch_values"])
        neutral_chin_y = reference_from_first_frames(state["chin_values"])
        state["pitch_threshold"] = pitch_drop_threshold(neutral_pitch)
        state["chin_threshold"] = chin_drop_threshold(neutral_chin_y)
        state["calibrated"] = True
        print(f"Calibração concluída! Pitch Neutro: {neutral_pitch:.2f} | Queixo Neutro Y: {neutral_chin_y:.2f}")


def update_drop_frames(state: dict, pitch: float, chin_y: int) -> None:
    head_dropping = pitch < state["pitch_threshold"] or chin_y > state["chin_threshold"]
    state["drop_frames"] = state["drop_frames"] + 1 if head_dropping else 0


def process_face(image, face_landmarks: FaceLandmarks, state: dict) -> None:
    height, width = image.shape[:2]
    draw_points(image, face_landmarks, P.HEAD_POSE_INDICES, P.GREEN, P.HEAD_POINT_RADIUS)
    chin_y = chin_y_pixels(face_landmarks, height)
    yaw, pitch, _ = compute_head_pose(face_landmarks, width, height)

    if not state["calibrated"]:
        put_text(image, CALIBRATING_TEXT, P.HEAD_CALIBRATING_STYLE)
        calibrate(state, pitch, chin_y)
        return

    put_text(image, f"Pitch: {int(pitch)} (Lim: {int(state['pitch_threshold'])})", P.HEAD_PITCH_STYLE)
    put_text(image, f"Yaw   (Lado): {int(yaw)}", P.HEAD_YAW_STYLE)
    update_drop_frames(state, pitch, chin_y)
    if state["drop_frames"] > P.HEAD_DROP_FRAMES:
        put_text(image, ALERT_TEXT, P.HEAD_ALERT_STYLE)


def main() -> None:
    face_mesh = create_face_mesh()
    state = new_state()
    for image in capture_frames():
        face_landmarks = detect_face(face_mesh, image)
        if face_landmarks is not None:
            process_face(image, face_landmarks, state)
        if show(WINDOW_TITLE, image) == P.KEY_QUIT:
            break


if __name__ == "__main__":
    main()
