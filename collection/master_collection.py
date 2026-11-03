"""Coleta própria pela webcam: grava blocos rotulados de 3 s com EAR, MAR, yaw e pitch por frame.

Rodar da raiz: python -m collection.master_collection
Teclas: 0 = alerta, 1 = fadiga, s = pausar, ESC ou q = sair.

Cada execução é uma sessão com seu próprio CSV em collection/sessions/. Assim PC e notebook não
geram conflito no git; depois é só concatenar os arquivos. (session_id, block_id) identifica
o bloco no dataset todo.
"""
import csv
import time
import winsound
from datetime import datetime

from detection import parameters as P
from detection.drawing import put_text
from detection.features import compute_features
from detection.landmarks import create_face_mesh, detect_face
from detection.webcam import capture_frames, measure_fps, show

WINDOW_TITLE = "Coleta Mestre Unificada"
STANDBY_TEXT = "STANDBY (Pressione 0 ou 1)"
INSTRUCTIONS = "[INFO] Pressione '0' para Alerta, '1' para Fadiga, 's' para pausar e ESC/Q para sair."


def ask_session_info() -> tuple[str, str]:
    """Pessoa e cenário da sessão (Enter aceita o padrão)."""
    person = input("Pessoa (ex: eu, mae) [eu]: ").strip() or P.DEFAULT_PERSON
    scenario = input("Cenário/iluminação (ex: noite-luz-quarto, dia-janela) [padrao]: ").strip() or P.DEFAULT_SCENARIO
    return person, scenario


def open_session_csv(session_id: str):
    """Abre o CSV da sessão com o cabeçalho; devolve (arquivo, writer, caminho)."""
    P.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = P.SESSIONS_DIR / f"session_{session_id}.csv"
    csv_file = open(path, mode="w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(P.SESSION_COLUMNS)
    return csv_file, writer, path


def new_state() -> dict:
    return {"recording": False, "label": P.LABEL_ALERT, "block_start": time.time(),
            "block_frames": [], "block_id": 0}


def start_recording(state: dict, label: int, message: str, beep: tuple[int, int]) -> None:
    """Começa a gravar um novo bloco com o rótulo escolhido."""
    state.update(recording=True, label=label, block_frames=[], block_start=time.time())
    print(message)
    winsound.Beep(*beep)


def handle_key(state: dict, key: int) -> None:
    if key == P.KEY_ALERT_MODE:
        start_recording(state, P.LABEL_ALERT, "[INFO] Modo Alerta ativado (0)", P.BEEP_ALERT_MODE)
    elif key == P.KEY_FATIGUE_MODE:
        start_recording(state, P.LABEL_FATIGUE, "[INFO] Modo Fadiga ativado (1)", P.BEEP_FATIGUE_MODE)
    elif key == P.KEY_PAUSE:
        state["recording"] = False
        print("[INFO] Gravação pausada.")


def save_block(state: dict, session_fields: list, csv_file, writer) -> None:
    """Grava TODOS os frames do bloco (sem média) com um id de bloco novo."""
    state["block_id"] += 1
    for index, (t, ear, mar, yaw, pitch) in enumerate(state["block_frames"]):
        writer.writerow([*session_fields, state["block_id"], index, f"{t:.3f}", f"{ear:.4f}",
                         f"{mar:.4f}", f"{yaw:.4f}", f"{pitch:.4f}", state["label"]])
    csv_file.flush()
    print(f"[SALVO] Bloco #{state['block_id']} | Rótulo: {state['label']} | {len(state['block_frames'])} frames")


def record_frame(image, state: dict, features: dict, session_fields: list, csv_file, writer) -> None:
    """Acumula o frame no bloco; ao completar a duração, grava o bloco e apita."""
    elapsed = time.time() - state["block_start"]
    state["block_frames"].append(
        (elapsed, features["ear_mean"], features["mar"], features["yaw"], features["pitch"]))
    remaining = max(0.0, P.BLOCK_DURATION_S - elapsed)
    color = P.RED if state["label"] == P.LABEL_FATIGUE else P.GREEN
    put_text(image, f"BLOCO 3S ({state['label']}): {remaining:.1f}s", P.COLLECTION_BLOCK_STYLE, color=color)

    if elapsed >= P.BLOCK_DURATION_S:
        save_block(state, session_fields, csv_file, writer)
        state["block_frames"] = []
        state["block_start"] = time.time()
        winsound.Beep(*P.BEEP_BLOCK_SAVED)


def main() -> None:
    face_mesh = create_face_mesh()
    person, scenario = ask_session_info()
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file, writer, csv_path = open_session_csv(session_id)
    session_fields = [session_id, person, scenario]
    print(f"[INFO] Sessão {session_id} | pessoa: {person} | cenário: {scenario}")
    print(f"[INFO] Gravando em: {csv_path}")
    print(INSTRUCTIONS)

    state = new_state()
    previous_time = 0
    try:
        for image in capture_frames():
            height, width = image.shape[:2]
            face_landmarks = detect_face(face_mesh, image)
            features = compute_features(face_landmarks, width, height) if face_landmarks is not None else None

            if state["recording"] and features is not None:
                record_frame(image, state, features, session_fields, csv_file, writer)
            else:
                put_text(image, STANDBY_TEXT, P.COLLECTION_STANDBY_STYLE)

            fps, previous_time = measure_fps(previous_time)
            put_text(image, f"FPS: {int(fps)}", P.FPS_STYLE)

            key = show(WINDOW_TITLE, image)
            if key in (P.KEY_ESC, P.KEY_QUIT):
                break
            handle_key(state, key)
    finally:
        csv_file.close()


if __name__ == "__main__":
    main()
