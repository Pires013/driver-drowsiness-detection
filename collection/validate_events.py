"""Compares hand-marked events (collection/validation_recording.py) with the yawn and head-drop detectors.

A detection matches a mark of the same type if the intervals touch within a 2 s tolerance; reports precision, recall and F1."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from detection import parameters as P
from detection.events import EventTracker, decide_level
from detection.preprocessing import LiveBuffer
from detection.validity import is_valid_frame

DETECTABLE = {"b": "yawns", "h": "head_drops"}
LONG_MARK_S = 20.0  # a longer mark means an old session (one key toggled the event on/off)


def detect(frames: pd.DataFrame, calibration_start: float = 0.0) -> dict[str, list[tuple[float, float]]]:
    face = frames["face_detected"].to_numpy() == P.FACE_FOUND
    valid = is_valid_frame(frames["face_detected"].to_numpy(), frames["yaw"].to_numpy(),
                           frames["roll"].to_numpy(), frames["ear_left"].to_numpy(),
                           frames["ear_right"].to_numpy())
    values = frames[P.MEASURE_NAMES].to_numpy()
    valid &= np.isfinite(values).all(axis=1)
    times = frames["t"].to_numpy()
    t0 = max(times[0], calibration_start)  # the 10 s calibration starts here (default: start of the recording)
    buffer, tracker = LiveBuffer(P.LIVE_CALIBRATION_S), None
    restarts, lost_after_drop, open_start = 0, [], None
    critical, crit_start = [], None
    for i, t in enumerate(times):
        if t < t0:
            continue
        if tracker is None:
            buffer.add(t - t0, values[i].tolist(), bool(valid[i]))
            if buffer.try_calibrate(t - t0):
                ref = buffer.ref
                tracker = EventTracker(ref_ear=ref[0], ref_mar=ref[1], ref_pitch=ref[2], ref_roll=ref[4])
            elif buffer.calibration_failed:
                buffer, t0, restarts = LiveBuffer(P.LIVE_CALIBRATION_S), t, restarts + 1
            continue
        state = tracker.update(t, bool(face[i]), bool(valid[i]), values[i, 0], values[i, 1], values[i, 2], values[i, 4])
        is_critical = decide_level(False, False, state) == 3  # no network: only the eye and head rules
        if is_critical and crit_start is None:
            crit_start = t
        elif not is_critical and crit_start is not None:
            critical.append((crit_start, t)); crit_start = None
        if state.face_lost_after_drop and open_start is None:
            open_start = t
        elif not state.face_lost_after_drop and open_start is not None:
            lost_after_drop.append((open_start, t)); open_start = None
    if tracker is None:
        raise SystemExit("gravação curta demais para a calibração (menos de 10 s com rosto)")
    if open_start is not None:
        lost_after_drop.append((open_start, times[-1]))
    if crit_start is not None:
        critical.append((crit_start, times[-1]))
    tracker.finish()
    return {"yawns": tracker.yawns, "head_drops": tracker.head_drops, "lost_after_drop": lost_after_drop, "critical": critical,
            "calibrated_at": t0 + P.LIVE_CALIBRATION_S, "restarts": restarts, "ref_pitch": tracker.ref_pitch}


def overlaps(a: tuple[float, float], b: tuple[float, float], tol: float) -> bool:
    return a[0] <= b[1] + tol and b[0] <= a[1] + tol


def compare(marked: list[tuple[float, float]], detected: list[tuple[float, float]], tol: float) -> dict:
    hits = sum(any(overlaps(m, d, tol) for d in detected) for m in marked)
    good = sum(any(overlaps(m, d, tol) for m in marked) for d in detected)
    precision = good / len(detected) if detected else float("nan")
    recall = hits / len(marked) if marked else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if detected and marked and precision + recall > 0 else float("nan"))
    return {"marcados": len(marked), "detectados": len(detected), "acertos": hits, "perdas": len(marked) - hits,
            "alarmes_falsos": len(detected) - good, "precisao": precision, "recall": recall, "f1": f1}


def validate(session: Path, tolerance: float = P.VALIDATION_MATCH_TOLERANCE_S,
             calibration_start: float = 0.0) -> tuple[pd.DataFrame, list[str]]:
    frames = pd.read_csv(session / "frames.csv")
    events = pd.read_csv(session / "events.csv")
    detected = detect(frames, calibration_start)
    start = detected["calibrated_at"]
    by_type = {k: [(s, e) for s, e in zip(g.inicio_s, g.fim_s) if e >= start]
               for k, g in events.groupby("tipo")}
    rows = []
    for kind, key in DETECTABLE.items():
        rows.append({"tipo": P.VALIDATION_EVENT_NAMES[kind],
                     **compare(by_type.get(kind, []), detected[key], tolerance)})
    drops = detected["head_drops"]
    for kind in ("c", "p"):
        spans = by_type.get(kind, [])
        false_drops = [d for d in drops if any(overlaps(d, s, tolerance) for s in spans)]
        rows.append({"tipo": P.VALIDATION_EVENT_NAMES[kind], "marcados": len(spans), "detectados": "--",
                     "acertos": "--", "perdas": "--", "alarmes_falsos": f"{len(false_drops)} quedas falsas",
                     "precisao": float("nan"), "recall": float("nan"), "f1": float("nan")})
    notes = [f"Calibração concluída em t = {start:.1f} s ({detected['restarts']} recomeço(s) por falta de quadros "
             f"válidos), pitch de referência {detected['ref_pitch']:.1f}° (queda = abaixo de "
             f"{detected['ref_pitch'] + P.PITCH_DROP_OFFSET_DEG:.1f}°); marcas antes disso ignoradas. Tolerância de casamento: {tolerance} s. "
             f"Duração gravada: {frames['t'].iloc[-1]:.0f} s."]
    long_marks = [(k, s, e) for k, g in events.groupby("tipo") for s, e in zip(g.inicio_s, g.fim_s)
                  if e - s > LONG_MARK_S]
    if long_marks:
        notes.append("ATENÇÃO: " + "; ".join(f"'{k}' de {s:.0f} a {e:.0f} s" for k, s, e in long_marks)
                     + ". Marcas desse tamanho vêm da gravação antiga (cada tecla ligava e desligava o evento): "
                     "uma tecla apertada uma vez fica 'aberta' até o fim e todo evento depois dela casa com ela. "
                     "Nesse caso os acertos por evento NÃO são confiáveis; grave de novo apertando a tecla a cada ação.")
    lost = detected["lost_after_drop"]
    notes.append(f"Quedas de cabeça detectadas (pitch 18° abaixo por 2 s ou mais): {len(drops)}. "
                 f"Rosto perdido com olho fechado antes (a regra que dá nível Crítico sem esperar a rede): "
                 f"{len(lost)} ocorrência(s)" + (": " + ", ".join(f"{s:.0f}-{e:.0f} s" for s, e in lost) if lost else "")
                 + ". Essa regra exige "
                 f"rosto ausente por {P.FACE_LOST_MIN_S:g} s ou mais, com olho fechado por {P.FACE_LOST_EYE_CLOSED_MIN_S:g} s ou mais "
                 f"nos {P.FACE_LOST_EYE_LOOKBACK_S:g} s antes de sumir (ou queda já iniciada com olho fechado).")
    for kind in ("c", "p"):
        spans = by_type.get(kind, [])
        n = sum(any(overlaps(d, s, tolerance) for s in spans) for d in lost)
        notes.append(f"Rosto perdido após queda durante '{P.VALIDATION_EVENT_NAMES[kind]}': {n} de {len(lost)}.")
    crit = detected["critical"]
    notes.append(f"Crítico pelas regras de olho e cabeça (sem a rede): {len(crit)} ocorrência(s)"
                 + (": " + ", ".join(f"{s:.0f}-{e:.0f} s" for s, e in crit) if crit else "")
                 + f". Durante marcas 'celular': {sum(any(overlaps(c, s, 0.0) for s in by_type.get('c', [])) for c in crit)}; "
                 f"'painel/retrovisor': {sum(any(overlaps(c, s, 0.0) for s in by_type.get('p', [])) for c in crit)}; "
                 f"'queda de cabeça': {sum(any(overlaps(c, s, tolerance) for s in by_type.get('h', [])) for c in crit)}.")
    notes.append(f"Marcas 'normal': {len(by_type.get('n', []))}.")
    lines = ["Bocejos detectados (início-fim, s): " + (", ".join(f"{s:.0f}-{e:.0f}" for s, e in detected["yawns"]) or "nenhum"),
             "Quedas de cabeça detectadas (início-fim, s): " + (", ".join(f"{s:.0f}-{e:.0f}" for s, e in drops) or "nenhuma")]
    return pd.DataFrame(rows), notes + lines


def main() -> None:
    ap = argparse.ArgumentParser(description="Compara eventos marcados com os detectores")
    ap.add_argument("session", type=Path, help="pasta da gravação (data/own/<sessão>)")
    ap.add_argument("--tolerance", type=float, default=P.VALIDATION_MATCH_TOLERANCE_S)
    ap.add_argument("--calibration-start", type=float, default=0.0,
                    help="segundo em que a calibração de 10 s começa (use se você só se ajeitou depois do início)")
    args = ap.parse_args()
    table, notes = validate(args.session, args.tolerance, args.calibration_start)
    shown = table.copy()
    for col in ("precisao", "recall", "f1"):
        shown[col] = shown[col].map(lambda v: "--" if pd.isna(v) else f"{100 * v:.0f}%")
    text = "\n".join(["| " + " | ".join(shown.columns) + " |", "|" + "---|" * len(shown.columns)]
                     + ["| " + " | ".join(str(v) for v in row) + " |" for row in shown.itertuples(index=False)]
                     + [""] + notes)
    print(text)
    (args.session / "validacao.md").write_text(text + "\n", encoding="utf-8")
    table.to_csv(args.session / "validacao.csv", index=False)


if __name__ == "__main__":
    main()
