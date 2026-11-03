"""Demo de webcam do MAR (bocejo) com calibração dinâmica. Rodar: python -m demos.mar_webcam (q sai).

Nos primeiros frames o usuário mantém a boca fechada: a média vira o MAR de repouso e o limiar
de bocejo é um múltiplo dele.
"""
from detection import parameters as P
from detection.calibration import mar_threshold, reference_from_first_frames
from detection.drawing import draw_points, put_text
from detection.landmarks import FaceLandmarks, create_face_mesh, detect_face
from detection.mar import compute_mar
from detection.webcam import capture_frames, show

WINDOW_TITLE = "Deteccao de Bocejo (MAR)"
CALIBRATING_TEXT = "CALIBRANDO... Mantenha a boca fechada"
ALERT_TEXT = "BOCEJO DETECTADO!"


def new_state() -> dict:
    return {"calibrated": False, "calibration_values": [], "threshold": None, "yawn_frames": 0}


def calibrate(state: dict, mar: float) -> None:
    """Acumula o MAR de repouso e, ao juntar frames suficientes, fixa o limiar."""
    state["calibration_values"].append(mar)
    if len(state["calibration_values"]) >= P.CALIBRATION_FRAMES:
        rest = reference_from_first_frames(state["calibration_values"])
        state["threshold"] = mar_threshold(rest)
        state["calibrated"] = True
        print(f"Calibração concluída! Repouso: {rest:.2f} | Novo Limiar MAR: {state['threshold']:.2f}")


def update_yawn_frames(state: dict, mar: float) -> None:
    """Conta frames seguidos acima do limiar; fala e sorriso rápidos zeram o contador."""
    state["yawn_frames"] = state["yawn_frames"] + 1 if mar > state["threshold"] else 0


def process_face(image, face_landmarks: FaceLandmarks, state: dict) -> None:
    height, width = image.shape[:2]
    mar = compute_mar(face_landmarks, width, height)
    draw_points(image, face_landmarks, P.MOUTH_INDICES, P.CYAN, P.MAR_POINT_RADIUS)

    if not state["calibrated"]:
        put_text(image, CALIBRATING_TEXT, P.MAR_CALIBRATING_STYLE)
        calibrate(state, mar)
        return

    put_text(image, f"MAR: {mar:.2f} (Lim: {state['threshold']:.2f})", P.MAR_READOUT_STYLE)
    update_yawn_frames(state, mar)
    if state["yawn_frames"] > P.YAWN_FRAMES:
        put_text(image, ALERT_TEXT, P.YAWN_ALERT_STYLE)


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
