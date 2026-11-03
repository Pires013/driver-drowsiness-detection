"""
Sanity check das features extraídas: compara, para cada participante, os vídeos alerta (0),
baixa vigilância (5) e sonolento (10).

Por vídeo calcula: EAR médio, PERCLOS, piscadas/min, bocejos, pitch médio, frames com EAR
exatamente 0, distribuição do MAR e instabilidade de pitch/yaw. Gera resumo.csv,
comparacao_0_vs_10.csv, bocejos.csv e os gráficos em data/sanity_check/.

Todos os limiares são RELATIVOS ao próprio vídeo (parâmetros em detection/parameters.py),
porque resolução, câmera, iluminação e rosto mudam entre vídeos.

Uso (da raiz do projeto):
    python -m pipeline.sanity_check [--fold Fold1] [--features data/features] [--out data/sanity_check]
Sem --fold lê todos os folds de --features (<features>/<FoldN>/<participante>/<rótulo>.csv).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from detection import parameters as P
from detection.calibration import ear_threshold, mar_threshold, pitch_drop_threshold, reference_from_percentile
from detection.validity import is_valid_frame
from pipeline.sanity_plots import plot_comparison, plot_ear_timeseries, plot_mar_distribution

# Direção esperada do vídeo sonolento em relação ao alerta (None = sem expectativa clara)
EXPECTED_DIRECTION = {"mean_ear": -1, "perclos_pct": +1, "blinks_per_min": None, "yawns": +1,
                      "mean_pitch": None, "long_closures": +1, "head_nods_per_min": +1,
                      "pct_head_down": +1, "pitch_std": +1, "pitch_p5_minus_median": -1}


# ----------------------------------------------------------------------
# Sequências contínuas
# ----------------------------------------------------------------------
def find_runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(início, fim) em posições de linha de cada sequência contínua de True. Frame inválido
    tem mask=False, então quebra a sequência (não conta como olho fechado)."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.array([], dtype=int), np.array([], dtype=int)
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.where(edges == 1)[0], np.where(edges == -1)[0] - 1


def bridge_invalid_gaps(mask: np.ndarray, valid: np.ndarray, timestamps: np.ndarray,
                        frame_dt: float) -> np.ndarray:
    """Cópia de mask com True nos buracos de frames INVÁLIDOS de até BRIDGE_MAX_GAP_S entre dois True.

    Só para detectar eventos (piscada, fechamento longo, bocejo): um frame inválido no meio do
    evento não deve parti-lo em dois. Frame válido com mask False (olho aberto de fato) nunca é
    ponteado, e as porcentagens (PERCLOS etc.) continuam usando a máscara original.
    """
    bridged = np.asarray(mask, dtype=bool).copy()
    starts, ends = find_runs(~np.asarray(valid, dtype=bool))
    for start, end in zip(starts, ends):
        if start == 0 or end == len(bridged) - 1:
            continue
        gap_s = timestamps[end] - timestamps[start] + frame_dt
        if gap_s <= P.BRIDGE_MAX_GAP_S and bridged[start - 1] and bridged[end + 1]:
            bridged[start:end + 1] = True
    return bridged


def run_durations(mask: np.ndarray, timestamps: np.ndarray, frame_dt: float) -> np.ndarray:
    """Duração (s) de cada sequência contínua de True."""
    starts, ends = find_runs(mask)
    return timestamps[ends] - timestamps[starts] + frame_dt


# ----------------------------------------------------------------------
# Métricas por assunto (cada uma devolve um dict; a ordem das chaves define as colunas do resumo)
# ----------------------------------------------------------------------
def ear_metrics(data: pd.DataFrame, valid: np.ndarray, timestamps: np.ndarray, frame_dt: float) -> dict:
    """EAR médio, PERCLOS (referência: olho aberto do próprio vídeo), piscadas e fechamentos longos."""
    ear = data["ear_mean"].to_numpy()
    open_eye_ear = reference_from_percentile(ear, P.OPEN_EYE_PERCENTILE)
    closing = ear < ear_threshold(open_eye_ear)             # NaN < x é False
    durations = run_durations(bridge_invalid_gaps(closing, valid, timestamps, frame_dt), timestamps, frame_dt)
    blinks = int(((durations >= P.BLINK_MIN_S) & (durations < P.BLINK_MAX_S)).sum())
    valid_minutes = valid.sum() * frame_dt / P.SECONDS_PER_MINUTE
    return {
        "mean_ear": data[valid]["ear_mean"].mean(),
        "open_eye_ear": open_eye_ear,
        "perclos_pct": P.PERCENT * closing[valid].mean(),
        "perclos80_pct": P.PERCENT * (ear < ear_threshold(open_eye_ear, P.EAR_CLOSED_P80_FACTOR))[valid].mean(),
        "blinks": blinks,
        "blinks_per_min": blinks / valid_minutes,
        "long_closures": int((durations >= P.BLINK_MAX_S).sum()),
    }


def zero_ear_metrics(valid_rows: pd.DataFrame) -> dict:
    """Frames com EAR exatamente 0 e sua origem provável.

    O CSV tem 6 casas decimais, então "0.000000" = zero exato do cálculo (ou < 5e-7).
    A trava de divisão por zero (horizontal == 0) só dispara se os DOIS cantos do olho caírem no
    mesmo pixel: só acontece com malha facial colapsada, e nesse caso o MAR (que também tem
    trava) vem 0 junto. Zero isolado, com MAR normal, é o outro caso: numerador 0 (as duas
    distâncias verticais viram 0 px por truncamento int()).
    """
    zero_left, zero_right, zero_mean = (valid_rows[c] == 0 for c in ("ear_left", "ear_right", "ear_mean"))
    zero_eye = zero_left | zero_right
    with_zero_mar = int((zero_eye & (valid_rows["mar"] == 0)).sum())
    return {
        "zeros_ear_left": int(zero_left.sum()),
        "zeros_ear_right": int(zero_right.sum()),
        "zeros_ear_mean": int(zero_mean.sum()),
        "zeros_with_mar_zero": with_zero_mar,
        "zeros_isolated": int(zero_eye.sum() - with_zero_mar),
    }


def mar_metrics(data: pd.DataFrame, valid: np.ndarray, timestamps: np.ndarray, frame_dt: float,
                video_id: dict) -> tuple[dict, list]:
    """Distribuição do MAR e bocejos (MAR acima de um múltiplo do MAR de repouso por tempo mínimo)."""
    mar = data["mar"].to_numpy()
    metrics = {f"mar_p{p}": reference_from_percentile(mar, p) for p in P.MAR_PERCENTILES}
    metrics["mar_max"] = float(np.nanmax(mar))
    rest = reference_from_percentile(mar, P.MAR_REST_PERCENTILE)   # mediana = boca fechada a maior parte do tempo
    threshold = mar_threshold(rest)
    above = mar > threshold
    metrics["mar_yawn_threshold"] = threshold
    metrics["pct_mar_above_threshold"] = P.PERCENT * above[valid].mean()

    starts, ends = find_runs(bridge_invalid_gaps(above, valid, timestamps, frame_dt))
    long_enough = (timestamps[ends] - timestamps[starts] + frame_dt) >= P.YAWN_MIN_S
    metrics["yawns"] = int(long_enough.sum())
    yawns = []
    for start, end in zip(starts[long_enough], ends[long_enough]):
        peak = start + int(np.nanargmax(mar[start:end + 1]))
        yawns.append({"video": f"{video_id['participant']}/{video_id['label']}",
                      "participant": video_id["participant"], "label": video_id["label"],
                      "start_s": round(timestamps[start], 2), "end_s": round(timestamps[end], 2),
                      "duration_s": round(timestamps[end] - timestamps[start] + frame_dt, 2),
                      "peak_frame": int(data["frame_idx"].iloc[peak]), "peak_mar": round(float(mar[peak]), 3)})
    return metrics, yawns


def smooth_pitch(pitch: np.ndarray, valid: np.ndarray, frame_dt: float) -> np.ndarray:
    """Mediana móvel centrada (PITCH_SMOOTH_S) do pitch; frame inválido continua NaN."""
    window = max(1, round(P.PITCH_SMOOTH_S / frame_dt)) | 1          # nº ímpar de frames
    smooth = pd.Series(pitch).rolling(window, center=True, min_periods=1).median().to_numpy()
    return np.where(valid, smooth, np.nan)


def head_pose_events(pitch: np.ndarray, valid: np.ndarray, timestamps: np.ndarray,
                     frame_dt: float) -> dict:
    """Cabeceios e dispersão do pitch, sobre o pitch suavizado dos frames válidos.

    Neutro = mediana do próprio vídeo; cabeça baixa = pitch abaixo de neutro + PITCH_DROP_OFFSET_DEG.
    Cabeceio = cabeça baixa por >= HEAD_NOD_MIN_S (com a ponte de buracos inválidos).
    """
    smooth = smooth_pitch(pitch, valid, frame_dt)
    neutral = reference_from_percentile(smooth, P.HEAD_NEUTRAL_PERCENTILE)
    head_down = smooth < pitch_drop_threshold(neutral)                # NaN < x é False
    durations = run_durations(bridge_invalid_gaps(head_down, valid, timestamps, frame_dt),
                              timestamps, frame_dt)
    nods = int((durations >= P.HEAD_NOD_MIN_S).sum())
    valid_minutes = valid.sum() * frame_dt / P.SECONDS_PER_MINUTE
    return {
        "head_nods": nods,
        "head_nods_per_min": nods / valid_minutes,
        "pct_head_down": P.PERCENT * head_down[valid].mean(),
        "pitch_std": float(np.nanstd(smooth, ddof=1)),
        "pitch_p5_minus_median": reference_from_percentile(smooth, P.PITCH_LOW_PERCENTILE) - neutral,
    }


def head_metrics(raw: pd.DataFrame, face_ok: np.ndarray, valid_rows: pd.DataFrame) -> dict:
    """Pitch/yaw médios (frames válidos) e instabilidade do solvePnP.

    Salto entre frames CONSECUTIVOS (índice consecutivo e rosto nos dois). Só mede, e mede no
    dado CRU (sem o filtro de validade): o objetivo é quantificar a instabilidade.
    """
    consecutive = (raw["frame_idx"].diff().to_numpy() == 1) & face_ok & np.roll(face_ok, 1)
    consecutive[0] = False
    pitch_jump = np.abs(raw["pitch"].diff().to_numpy())[consecutive] > P.JUMP_DEG
    yaw_jump = np.abs(raw["yaw"].diff().to_numpy())[consecutive] > P.JUMP_DEG
    has_pairs = len(pitch_jump) > 0
    return {
        "mean_pitch": valid_rows["pitch"].mean(),
        "mean_yaw": valid_rows["yaw"].mean(),
        "pct_jump_pitch": P.PERCENT * pitch_jump.mean() if has_pairs else np.nan,
        "pct_jump_yaw": P.PERCENT * yaw_jump.mean() if has_pairs else np.nan,
        "pct_jump_any": P.PERCENT * (pitch_jump | yaw_jump).mean() if has_pairs else np.nan,
    }


# ----------------------------------------------------------------------
# Um vídeo
# ----------------------------------------------------------------------
def analyze_video(csv_path: Path, apply_filter: bool = True) -> tuple[dict, pd.DataFrame, list]:
    """(resumo, dados com frames inválidos em NaN, bocejos) de um CSV de features."""
    data = pd.read_csv(csv_path, dtype={"participant": str})
    timestamps = data["timestamp_s"].to_numpy()
    frame_dt = float(np.median(np.diff(timestamps)))
    face_ok = data["face_detected"].to_numpy() == P.FACE_FOUND
    raw = data[["frame_idx", "pitch", "yaw"]].copy()    # a instabilidade do solvePnP se mede no dado cru

    if apply_filter:
        valid = is_valid_frame(data["face_detected"].to_numpy(), data["yaw"].to_numpy(),
                               data["roll"].to_numpy(), data["ear_left"].to_numpy(),
                               data["ear_right"].to_numpy())
    else:
        valid = face_ok
    data.loc[~valid, P.FEATURE_COLUMNS] = np.nan        # só na memória; o CSV em disco não muda
    valid_rows = data[valid]

    video_id = {"participant": data["participant"].iloc[0], "label": int(data["label"].iloc[0])}
    summary = {
        **video_id,
        "frames": len(data), "duration_s": round(timestamps[-1] - timestamps[0], 1),
        "dt_s": round(frame_dt, 4), "fps": round(1 / frame_dt, 2),
        "pct_no_face": round(P.PERCENT * (~face_ok).mean(), 2),
        "pct_invalid": round(P.PERCENT * (face_ok & ~valid).mean(), 2),
    }
    summary.update(ear_metrics(data, valid, timestamps, frame_dt))
    summary.update(zero_ear_metrics(valid_rows))
    mar_summary, yawns = mar_metrics(data, valid, timestamps, frame_dt, video_id)
    summary.update(mar_summary)
    summary.update(head_metrics(raw, face_ok, valid_rows))
    summary.update(head_pose_events(data["pitch"].to_numpy(), valid, timestamps, frame_dt))
    return summary, data, yawns


# ----------------------------------------------------------------------
# Tabelas e relatório
# ----------------------------------------------------------------------
def add_alert_reference_perclos(table: pd.DataFrame, data: dict) -> None:
    """Coluna extra: PERCLOS com o EAR 'aberto' do vídeo ALERTA do mesmo participante como referência.

    Se o vídeo sonolento já tem olho meio fechado o tempo todo, o EAR aberto dele próprio é baixo e
    o PERCLOS relativo ao próprio vídeo SUBESTIMA a sonolência; esta coluna expõe isso.
    """
    reference = table[table.label == P.ALERT_LABEL].set_index("participant")["open_eye_ear"]
    table["perclos_alert_ref_pct"] = [
        P.PERCENT * (data[(row.participant, row.label)]["ear_mean"].dropna()     # só frames válidos
                     < ear_threshold(reference[row.participant])).mean()
        if row.participant in reference.index else np.nan for row in table.itertuples()]


def add_resolution(table: pd.DataFrame, log_path: Path) -> None:
    """Colunas width/height (já com a rotação aplicada) lidas do log da extração."""
    if not log_path.exists():
        table["width"] = table["height"] = np.nan
        return
    log = pd.read_csv(log_path, dtype={"participant": str}).drop_duplicates(["participant", "label"], keep="last")
    sizes = log.set_index(["participant", "label"])[["width", "height"]]
    keys = list(zip(table["participant"], table["label"]))
    table["width"] = [sizes["width"].get(k, np.nan) for k in keys]
    table["height"] = [sizes["height"].get(k, np.nan) for k in keys]


def fps_mismatch(table: pd.DataFrame) -> dict:
    """{participante: menor/maior fps} dos vídeos 0 e 10, para quem fica abaixo de FPS_MISMATCH_RATIO."""
    flagged = {}
    for participant, group in table.groupby("participant"):
        fps = group[group.label.isin([P.ALERT_LABEL, P.DROWSY_LABEL])]["fps"]
        if len(fps) == 2 and fps.min() / fps.max() < P.FPS_MISMATCH_RATIO:
            flagged[participant] = fps.min() / fps.max()
    return flagged


def compare_alert_vs_drowsy(table: pd.DataFrame) -> pd.DataFrame:
    """Diferença (sonolento - alerta) de cada métrica esperada, por participante."""
    rows = []
    for participant, group in table.groupby("participant"):
        alert, drowsy = group[group.label == P.ALERT_LABEL], group[group.label == P.DROWSY_LABEL]
        if len(alert) and len(drowsy):
            rows.append({"participant": participant,
                         **{f"delta_{c}": drowsy.iloc[0][c] - alert.iloc[0][c] for c in EXPECTED_DIRECTION}})
    return pd.DataFrame(rows)


def stage_table(table: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por participante com as 3 etapas (0, 5, 10) de cada métrica esperada."""
    labels = (P.ALERT_LABEL, P.LOW_VIGILANCE_LABEL, P.DROWSY_LABEL)
    rows = []
    for participant, group in table.groupby("participant"):
        by_label = group.set_index("label")
        if all(label in by_label.index for label in labels):
            rows.append({"participant": participant,
                         **{f"{c}_{label}": by_label.loc[label, c]
                            for c in EXPECTED_DIRECTION for label in labels}})
    return pd.DataFrame(rows)


def print_stages(stages: pd.DataFrame) -> None:
    """Em qual etapa cada sinal aparece: média por etapa e participantes na direção esperada."""
    alert, low, drowsy = P.ALERT_LABEL, P.LOW_VIGILANCE_LABEL, P.DROWSY_LABEL
    print(f"\n=== Etapas 0 / 5 / 10 ({len(stages)} participantes com os 3 vídeos) ===")
    rows = []
    for column, sign in EXPECTED_DIRECTION.items():
        if not sign:
            continue
        a, b, c = (stages[f"{column}_{label}"] for label in (alert, low, drowsy))
        rows.append({"metrica": column, "media_0": a.mean(), "media_5": b.mean(), "media_10": c.mean(),
                     "5 vs 0": int((np.sign(b - a) == sign).sum()),
                     "10 vs 5": int((np.sign(c - b) == sign).sum()),
                     "10 vs 0": int((np.sign(c - a) == sign).sum()),
                     "0<5<10": int(((np.sign(b - a) == sign) & (np.sign(c - b) == sign)).sum())})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("(contagens = participantes na direção esperada; 0<5<10 = piora a cada etapa)")


def print_report(table: pd.DataFrame, comparison: pd.DataFrame, out_dir: Path) -> None:
    pd.set_option("display.width", 250, "display.max_columns", 50)
    print("\n=== Resumo por vídeo ===")
    print(table[["participant", "label", "frames", "pct_no_face", "pct_invalid", "mean_ear",
                 "perclos_pct", "perclos80_pct", "perclos_alert_ref_pct", "blinks_per_min",
                 "long_closures", "yawns", "mean_pitch"]].round(2).to_string(index=False))
    flagged = fps_mismatch(table)
    print(f"\n=== fps e resolução (* = fps do 0 e do 10 muito diferentes, razão < {P.FPS_MISMATCH_RATIO}) ===")
    resolution = table[["participant", "label", "fps", "width", "height"]].copy()
    resolution["flag"] = ["*" if p in flagged and l in (P.ALERT_LABEL, P.DROWSY_LABEL) else ""
                          for p, l in zip(resolution.participant, resolution.label)]
    print(resolution.to_string(index=False))
    print("\n=== EAR exatamente 0 ===")
    print(table[["participant", "label", "zeros_ear_left", "zeros_ear_right", "zeros_ear_mean",
                 "zeros_with_mar_zero", "zeros_isolated"]].to_string(index=False))
    print("\n=== MAR ===")
    print(table[["participant", "label", "mar_p5", "mar_p25", "mar_p50", "mar_p75", "mar_p95", "mar_p99",
                 "mar_max", "mar_yawn_threshold", "pct_mar_above_threshold", "yawns"]].round(3).to_string(index=False))
    print("\n=== Cabeça (pitch suavizado; cabeceio = cabeça baixa por >= "
          f"{P.HEAD_NOD_MIN_S} s) ===")
    print(table[["participant", "label", "head_nods", "head_nods_per_min", "pct_head_down", "pitch_std",
                 "pitch_p5_minus_median"]].round(2).to_string(index=False))
    nods = table.pivot(index="participant", columns="label", values="head_nods")
    never = nods[(nods.fillna(0) == 0).all(axis=1)]
    print(f"Participantes com zero cabeceios em todos os vídeos: {len(never)}/{len(nods)}"
          + (f" ({', '.join(never.index)})" if len(never) else ""))
    print(f"\n=== Instabilidade (% de pares de frames consecutivos com salto > {P.JUMP_DEG:.0f}°) ===")
    print(table[["participant", "label", "pct_jump_pitch", "pct_jump_yaw", "pct_jump_any"]]
          .round(3).to_string(index=False))
    if len(comparison):
        print("\n=== Direção esperada (10 vs 0) ===")
        for column, sign in EXPECTED_DIRECTION.items():
            if sign:
                matches = int((np.sign(comparison[f"delta_{column}"]) == sign).sum())
                print(f"{column}: {matches}/{len(comparison)} participantes na direção esperada "
                      f"({'maior' if sign > 0 else 'menor'} no sonolento)")
    print_stages(stage_table(table))
    print(f"\nSaídas em {out_dir}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=P.FEATURES_DIR)
    ap.add_argument("--fold", help="analisa só este fold (ex.: Fold1); padrão: todos")
    ap.add_argument("--out", type=Path, default=P.SANITY_DIR)
    ap.add_argument("--sem-filtro", action="store_true",
                    help="não invalida frames (só rosto detectado): reproduz a análise anterior")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(args.features.glob(f"{args.fold or '*'}/*/*.csv"))
    if not csv_files:
        raise SystemExit(f"Nenhum CSV em {args.features / (args.fold or '*')}")
    summaries, data, all_yawns = [], {}, []
    for csv_file in csv_files:
        summary, video_data, yawns = analyze_video(csv_file, apply_filter=not args.sem_filtro)
        summaries.append(summary)
        all_yawns += yawns
        data[(summary["participant"], summary["label"])] = video_data
        print(f"analisado {summary['participant']}/{summary['label']} ({summary['frames']} frames)")
    table = pd.DataFrame(summaries).sort_values(["participant", "label"]).reset_index(drop=True)
    add_alert_reference_perclos(table, data)
    add_resolution(table, P.FEATURES_DIR / P.EXTRACTION_LOG_NAME)

    table.round(4).to_csv(args.out / "resumo.csv", index=False)
    pd.DataFrame(all_yawns).to_csv(args.out / "bocejos.csv", index=False)
    comparison = compare_alert_vs_drowsy(table)
    if len(comparison):
        comparison.round(4).to_csv(args.out / "comparacao_0_vs_10.csv", index=False)

    stage_table(table).round(4).to_csv(args.out / "etapas_0_5_10.csv", index=False)
    plot_comparison(table, args.out)
    plot_ear_timeseries(data, args.out)
    plot_mar_distribution(table, args.out)
    print_report(table, comparison, args.out)


if __name__ == "__main__":
    main()
