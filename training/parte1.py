"""Part 1: calibration length (5, 10 and 60 s), metrics tables and the final model (10 s calibration).

Usage: python -m training.parte1 tabelas|final (after prepare_data and run_experiments for cycles 4, 5 and 6)."""
import json
import sys

import numpy as np
import pandas as pd

from detection import parameters as P
from training import config as C
from training import models as M
from training.report import METRICS, MODEL_TITLES, cm_markdown, evaluate

METRIC_TITLES = {"acc": "Acurácia", "precision": "Precisão", "recall": "Recall", "f1": "F1-score",
                 "bal_acc": "Acurácia balanceada"}
METRIC_ORDER = ["acc", "precision", "recall", "f1", "bal_acc"]
MODELS = ("regra_aprendida", "lstm")
# (label, cycle, experiment)
SOURCES = [("Realista, calibração de 5 s", 4, "A"), ("Realista, calibração de 10 s", 5, "A"),
           ("Realista, calibração de 60 s", 6, "A"),
           ("Otimista (vídeo alerta), ciclo 3, sem o 44", 3, "B"),
           ("Otimista (vídeo alerta), ciclo 2, sem 26, 44 e 45", 2, "B")]
FINAL_DIR = C.TRAINING_DIR / "modelo_final"
METRICS_CSV = C.TRAINING_DIR / "metricas_parte1.csv"
REPORT_PART1 = C.TRAINING_DIR / "relatorio_parte1.md"


def collect() -> list[dict]:
    rows = []
    for level in ("video", "janela"):
        for label, cycle, key in SOURCES:
            for model in MODELS:
                ev = evaluate(cycle, key, model)
                if ev is None:
                    continue
                frame = ev["video" if level == "video" else "window"]
                cm = ev["cm_video" if level == "video" else "cm_window"]
                row = {"nivel": level, "calibracao": label, "modelo": MODEL_TITLES[model],
                       "n_sementes": ev["n_seeds"]}
                for m in METRIC_ORDER:
                    row[f"{m}_media"] = 100 * frame[m].mean()
                    row[f"{m}_dp"] = 100 * frame[m].std(ddof=1)
                row.update(tn=cm[0, 0], fp=cm[0, 1], fn=cm[1, 0], tp=cm[1, 1])
                rows.append(row)
    return rows


def fmt_table(rows: list[dict], level: str) -> str:
    lines = ["| Calibração | Modelo | " + " | ".join(METRIC_TITLES[m] for m in METRIC_ORDER) + " |",
             "|---|---|" + "---|" * len(METRIC_ORDER)]
    for r in rows:
        if r["nivel"] == level:
            lines.append(f"| {r['calibracao']} | {r['modelo']} | "
                         + " | ".join(f"{r[f'{m}_media']:.1f} ± {r[f'{m}_dp']:.1f}" for m in METRIC_ORDER) + " |")
    return "\n".join(lines)


def fmt_confusions(rows: list[dict], level: str) -> list[str]:
    out = []
    for r in rows:
        if r["nivel"] != level:
            continue
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        out += [f"**{r['calibracao']} — {r['modelo']}**", "", cm_markdown(cm, [0, 10]), ""]
    return out


def write_tables() -> list[dict]:
    rows = collect()
    pd.DataFrame(rows).round(2).to_csv(METRICS_CSV, index=False)
    print(f"{len(rows)} linhas em {METRICS_CSV}")
    return rows


def table_markdown(rows: list[dict]) -> list[str]:
    seeds = sorted({r["n_sementes"] for r in rows if r["modelo"] == MODEL_TITLES["lstm"]})
    lines = ["Classe positiva = sonolento (10). Média ± desvio padrão entre os 5 folds oficiais, em %. "
             f"A LSTM é a média de {', '.join(map(str, seeds))} sementes por fold. As janelas de treino e de "
             "teste começam depois do 1º minuto nos três casos de calibração realista, então os três "
             "veem as mesmas janelas e só muda a referência. As linhas \"otimista\" usam o vídeo alerta da "
             "própria pessoa como referência (só para comparação; não existe num uso real).", "",
             "### Por vídeo (métrica principal)", "", fmt_table(rows, "video"), "",
             "### Por janela (secundária)", "", fmt_table(rows, "janela"), "",
             "### Matrizes de confusão por vídeo (soma dos 5 folds; LSTM = média das sementes)", ""]
    lines += fmt_confusions(rows, "video")
    lines += ["### Matrizes de confusão por janela (soma dos 5 folds)", ""] + fmt_confusions(rows, "janela")
    return lines


def train_final(log=print) -> None:
    import torch
    from training.run_experiments import load_experiment, result_path

    cycle = 5
    exp = load_experiment("A", cycle)
    epochs = []
    for k in range(1, C.N_FOLDS + 1):
        for s in range(C.N_SEEDS[cycle]):
            meta = json.loads(str(np.load(result_path(cycle, "A", "lstm", f"Fold{k}", s))["meta"]))
            epochs.append(meta["best_epoch"])
    n_epochs = int(np.median(epochs))
    log(f"Modelo final: {len(exp['X'])} janelas de {len(set(exp['person']))} pessoas, {n_epochs} épocas "
        f"(mediana das melhores épocas dos folds)")
    model, norm, _ = M.train_lstm(exp["X"], exp["y"], exp["n_classes"], C.SEED, fixed_epochs=n_epochs,
                                  log=lambda m: None)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = "lstm_final_cal10.pt"
    torch.save({"state": model.state_dict(), "norm": {k: v.tolist() for k, v in norm.items()}},
               FINAL_DIR / checkpoint)
    config = {
        "descricao": "LSTM final: calibração de 10 s, todos os participantes do UTA-RLDD menos o 44",
        "modelo": {"tipo": "lstm", "checkpoint": checkpoint, "n_entradas": len(norm["mean"]) + 1,
                   "unidades": C.LSTM_HIDDEN, "camadas": 1, "dropout": C.LSTM_DROPOUT, "n_classes": 2,
                   "rotulos": [0, 10], "classe_sonolento": 1, "epocas": n_epochs,
                   "participantes_excluidos": C.CYCLE3_EXCLUDE},
        "calibracao": {"segundos": 10,
                       "estatisticas": {"ear": f"percentil {P.OPEN_EYE_PERCENTILE}",
                                        "mar": f"percentil {P.MAR_REST_PERCENTILE} (mediana)",
                                        "pitch": "mediana", "yaw": "mediana", "roll": "mediana"},
                       "ordem_da_referencia": ["ear", "mar", "pitch", "yaw", "roll"],
                       "relativo": {"ear": "razão da referência", "mar": "razão da referência",
                                    "ear_min": "razão da referência de EAR",
                                    "pitch_yaw_roll": "diferença em graus"}},
        "reamostragem": {"hz": P.RESAMPLE_HZ, "media_por_intervalo": True, "ear_minimo_por_intervalo": True,
                         "buraco_interpolado_max_s": P.MAX_INTERP_GAP_S,
                         "frame_invalido": {"rosto": True, "yaw_max_graus": P.VALID_YAW_MAX_DEG,
                                            "roll_max_graus": P.VALID_ROLL_MAX_DEG,
                                            "ear_max_de_um_olho": P.VALID_EAR_EYE_MAX}},
        "janela": {"segundos": P.WINDOW_S, "passos": P.WINDOW_STEPS,
                   "fracao_minima_observada": P.MIN_OBSERVED_FRACTION, "so_depois_da_calibracao": True,
                   "atualizacao_ao_vivo_s": P.LIVE_UPDATE_S},
        "features": {"medidas_da_janela": P.MEASURE_NAMES + ["ear_min"],
                     "canais_relativos": P.RELATIVE_CHANNELS[:-1],
                     "ordem_na_entrada_da_rede": P.RELATIVE_CHANNELS},
        "normalizacao": {"media": norm["mean"].tolist(), "desvio": norm["std"].tolist(),
                         "limite_em_desvios": M.NORM_CLIP, "falta_vira": 0.0,
                         "canal_de_falta_acrescentado_no_fim": True},
    }
    (FINAL_DIR / "config_modelo_final.json").write_text(json.dumps(config, indent=2, ensure_ascii=False),
                                                         encoding="utf-8")
    log(f"Modelo final e configuração gravados em {FINAL_DIR}")


PRESENTATION_PATH = C.TRAINING_DIR / "tabela_apresentacao.md"
PRESENTATION_METRICS = ["acc", "precision", "recall", "f1"]


def presentation_table() -> str:
    rows = [("Regra com limiar aprendido (calibração de 10 s)", 5, "regra_aprendida"),
            ("Rede LSTM (calibração de 10 s)", 5, "lstm"),
            ("Rede LSTM, calibração otimista (referência: vídeo alerta da própria pessoa)", 3, "lstm")]
    key = {5: "A", 3: "B"}
    lines = ["| Modelo | " + " | ".join(METRIC_TITLES[m] for m in PRESENTATION_METRICS) + " |",
             "|---|" + "---|" * len(PRESENTATION_METRICS)]
    for title, cycle, model in rows:
        frame = evaluate(cycle, key[cycle], model)["video"]
        lines.append(f"| {title} | " + " | ".join(
            f"{100 * frame[m].mean():.1f} ± {100 * frame[m].std(ddof=1):.1f}" for m in PRESENTATION_METRICS) + " |")
    gru = evaluate(5, "A", "gru")["video"]
    lstm = evaluate(5, "A", "lstm")["video"]
    diff = gru["bal_acc"].to_numpy() - lstm["bal_acc"].to_numpy()
    return "\n".join([
        "# Tabela final para a apresentação", "",
        "Por vídeo (cada vídeo recebe uma previsão: a média das janelas de 60 s). Classe positiva = sonolento (10), "
        "alerta (0) é a negativa. Valores em %, média ± desvio padrão entre os 5 folds oficiais (cada pessoa só "
        "aparece em um fold, então ninguém está no treino e no teste). Calibração realista = referência "
        "(olho aberto, boca e cabeça em repouso) dos primeiros 10 s do vídeo. 57 pessoas (sem o 44). "
        "As redes são a média de 5 sementes por fold.", "", *lines, "",
        "A calibração otimista usa o vídeo alerta da própria pessoa como referência; só serve de comparação, "
        "não existe num uso real (ciclo 3, experimento B).", "",
        "## Por que a LSTM e não a GRU", "",
        f"Com a calibração de 10 s a GRU fez {100 * gru['acc'].mean():.1f} ± {100 * gru['acc'].std(ddof=1):.1f} de "
        f"acurácia contra {100 * lstm['acc'].mean():.1f} ± {100 * lstm['acc'].std(ddof=1):.1f} da LSTM. A diferença "
        f"de acurácia balanceada por fold é {100 * diff.mean():.1f} pontos com erro padrão "
        f"{100 * diff.std(ddof=1) / np.sqrt(len(diff)):.1f} (5 folds), ou seja, empate "
        "estatístico; pela regra combinada o empate fica com a LSTM, que é o modelo final do demo.", ""])


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "tabelas"
    if command == "tabelas":
        print("\n".join(table_markdown(write_tables())))
    elif command == "apresentacao":
        PRESENTATION_PATH.write_text(presentation_table(), encoding="utf-8")
        print(PRESENTATION_PATH.read_text(encoding="utf-8"))
    elif command == "final":
        train_final()
