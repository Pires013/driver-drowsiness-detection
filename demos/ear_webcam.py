"""Webcam EAR demo with dynamic calibration (q quits).

The mean EAR of the first frames (eyes open) is the reference; the eye-closing threshold is a fraction of it."""
from detection import parameters as P
from detection.calibration import ear_threshold, reference_from_first_frames
from detection.drawing import draw_contours, draw_points, draw_tesselation, put_text
from detection.ear import compute_ears
from detection.landmarks import FaceLandmarks, create_face_mesh, detect_face
from detection.webcam import capture_frames, measure_fps, show

WINDOW_TITLE = "Face Mesh - Deteccao de Fadiga"
CALIBRATING_TEXT = "CALIBRANDO EAR... Mantenha os olhos abertos normalmente"
ALERT_TEXT = "ALERTA: OLHOS FECHADOS!"


def new_state() -> dict:
    return {"calibrated": False, "calibration_values": [], "threshold": None, "closed_frames": 0}


def calibrate(state: dict, ear_mean: float) -> None:
    state["calibration_values"].append(ear_mean)
    if len(state["calibration_values"]) >= P.CALIBRATION_FRAMES:
        neutral = reference_from_first_frames(state["calibration_values"])
        state["threshold"] = ear_threshold(neutral)
        state["calibrated"] = True
        print(f"Calibração concluída! EAR Neutro: {neutral:.3f} | Limiar: {state['threshold']:.3f}")


def update_closed_frames(state: dict, ear_mean: float) -> None:
    state["closed_frames"] = state["closed_frames"] + 1 if ear_mean < state["threshold"] else 0


def draw_readout(image, ear_left: float, ear_right: float, ear_mean: float, threshold: float) -> None:
    put_text(image, f"EAR E: {ear_left:.2f} | D: {ear_right:.2f}", P.EAR_BOTH_EYES_STYLE)
    put_text(image, f"EAR Media: {ear_mean:.2f} (Lim: {threshold:.2f})", P.EAR_MEAN_STYLE)


def process_face(image, face_landmarks: FaceLandmarks, state: dict) -> None:
    height, width = image.shape[:2]
    ear_left, ear_right, ear_mean = compute_ears(face_landmarks, width, height)
    draw_points(image, face_landmarks, P.LEFT_EYE_INDICES, P.RED, P.EAR_POINT_RADIUS)
    draw_points(image, face_landmarks, P.RIGHT_EYE_INDICES, P.BLUE, P.EAR_POINT_RADIUS)

    if not state["calibrated"]:
        put_text(image, CALIBRATING_TEXT, P.EAR_CALIBRATING_STYLE)
        calibrate(state, ear_mean)
        return

    draw_readout(image, ear_left, ear_right, ear_mean, state["threshold"])
    update_closed_frames(state, ear_mean)
    if state["closed_frames"] > P.EAR_CLOSED_FRAMES:
        put_text(image, ALERT_TEXT, P.EAR_ALERT_STYLE)
    draw_tesselation(image, face_landmarks)
    draw_contours(image, face_landmarks)


def main() -> None:
    face_mesh = create_face_mesh()
    state = new_state()
    previous_time = 0
    for image in capture_frames():
        face_landmarks = detect_face(face_mesh, image)
        if face_landmarks is not None:
            process_face(image, face_landmarks, state)

        fps, previous_time = measure_fps(previous_time)
        put_text(image, f"FPS: {int(fps)}", P.FPS_STYLE)
        if show(WINDOW_TITLE, image) == P.KEY_QUIT:
            break


if __name__ == "__main__":
    main()
