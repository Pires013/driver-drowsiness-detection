"""Persistência e histerese para os níveis da rede no demo (escolha de N e dos limites).

Uso (da raiz, depois de treinar o ciclo 5 com a rede escolhida):
    python -m training.persistencia probs lstm|gru     # previsões fora do fold, uma por segundo
    python -m training.persistencia grade lstm|gru     # busca dos parâmetros e tabela de resultados

As previsões imitam o demo: referência dos primeiros 10 s do vídeo, janela de 60 s que só começa depois
da calibração, uma saída por segundo, feita pelo modelo do fold em que a pessoa NÃO entrou no
treino (semente 0). A escolha dos parâmetros usa só os vídeos 0 e 10 das pessoas fora do 44 e dos 6
participantes do teste do demo (04, 08, 13, 25, 37, 50); estes ficam para a comparação antes/depois.
"""
import itertools
import sys

import numpy as np
import pandas as pd
import torch

from detection import parameters as P
from detection.events import NetworkGate
from detection.preprocessing import reference_of, relative
from training import config as C
from training import models as M
from training.prepare_data import resample_video

CYCLE = 5
DEMO_PERSONS = ["04", "08", "13", "25", "37", "50"]
OUT_DIR = C.TRAINING_DIR / "persistencia"
FIRST_OUTPUT_S = P.LIVE_CALIBRATION_S + P.WINDOW_S      # a rede só responde com 60 s depois da calibração
ENTER = [0.5, 0.6, 0.7, 0.8, 0.9]
GAP = [0.1, 0.2, 0.3]                                    # saída = entrada - gap
PERSIST_S = [0, 5, 10, 20, 30, 60]
MAX_ALERT_VIDEOS_PCT = 5.0                               # meta: vídeos alerta <= 5% do tempo em Alerta


def probs_path(kind: str):
    return OUT_DIR / f"probs_{kind}.npz"


def compute_probs(kind: str) -> None:
    """Probabilidade de sonolento a cada 1 s, de cada vídeo 0 e 10, pelo modelo do fold da pessoa."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    models = {}
    for k in range(1, C.N_FOLDS + 1):
        data = torch.load(C.models_dir(CYCLE) / f"{kind}_A_Fold{k}_s0.pt", map_location="cpu")
        net = M.make_rnn(kind, len(P.RELATIVE_CHANNELS), 2)
        net.load_state_dict(data["state"]); net.eval()
        models[f"Fold{k}"] = (net, {a: np.array(b, dtype=np.float32) for a, b in data["norm"].items()})
    out = {}
    cal = P.LIVE_CALIBRATION_S * P.RESAMPLE_HZ
    for csv_path in sorted(P.FEATURES_DIR.glob("Fold*/*/*.csv")):
        series, meta = resample_video(csv_path, C.n_measures(CYCLE))
        if meta["label"] not in (0, 10) or meta["participant"] == "44":
            continue
        ref = reference_of(series[:cal])
        if ref is None:
            continue
        rel = relative(series, ref)
        starts = list(range(cal, len(rel) - P.WINDOW_STEPS + 1, P.RESAMPLE_HZ))
        wins, idx = [], []
        for s in starts:
            w = rel[s:s + P.WINDOW_STEPS]
            if np.isfinite(w[:, 0]).mean() >= P.MIN_OBSERVED_FRACTION:
                wins.append(w.astype(np.float32)); idx.append(s)
        net, norm = models[meta["fold"]]
        p = np.full(len(starts), np.nan)
        if wins:
            probs = M.predict_proba(net, M.to_input(np.stack(wins), norm))[:, 1]
            p[[(s - cal) // P.RESAMPLE_HZ for s in idx]] = probs
        out[f"{meta['participant']}_{meta['label']}"] = p
    np.savez_compressed(probs_path(kind), **out)
    print(f"{len(out)} vídeos em {probs_path(kind)}")


def simulate(probs: np.ndarray, enter: float, exit_: float, persist_s: float) -> np.ndarray:
    """Estado (bool) do portão de Alerta a cada segundo."""
    gate = NetworkGate(enter, exit_, persist_s)
    return np.array([gate.update(float(i), None if np.isnan(p) else float(p)) for i, p in enumerate(probs)])


def load_probs(kind: str) -> dict:
    data = np.load(probs_path(kind))
    return {k: data[k] for k in data.files}


def video_stats(probs: dict, persons, params) -> dict:
    """% do tempo em Alerta (só pela rede) de cada classe e % de vídeos com mais de 10% do tempo em Alerta."""
    res = {0: [], 10: []}
    for key, p in probs.items():
        person, label = key.split("_")
        if person in persons and np.isfinite(p).any():
            state = simulate(p, *params)
            res[int(label)].append(100 * state.mean())
    return {"alerta_t": np.mean(res[0]), "sonolento_t": np.mean(res[10]),
            "alerta_acima10": 100 * np.mean(np.array(res[0]) > 10),
            "sonolento_acima10": 100 * np.mean(np.array(res[10]) > 10),
            "n_alerta": len(res[0]), "n_sonolento": len(res[10])}


def grid(kind: str) -> pd.DataFrame:
    probs = load_probs(kind)
    persons = {k.split("_")[0] for k in probs} - set(DEMO_PERSONS)
    rows = []
    for enter, gap, persist in itertools.product(ENTER, GAP, PERSIST_S):
        stats = video_stats(probs, persons, (enter, round(enter - gap, 2), persist))
        rows.append({"entra": enter, "sai": round(enter - gap, 2), "N_s": persist, **stats})
    return pd.DataFrame(rows)


def choose(table: pd.DataFrame) -> pd.Series:
    """Entre as combinações com vídeos alerta <= 5% do tempo em Alerta, a que mais mantém dos sonolentos."""
    ok = table[table.alerta_t <= MAX_ALERT_VIDEOS_PCT]
    return (ok if len(ok) else table).sort_values(["sonolento_t", "alerta_t"], ascending=[False, True]).iloc[0]


if __name__ == "__main__":
    command, kind = sys.argv[1], sys.argv[2]
    if command == "probs":
        compute_probs(kind)
    elif command == "grade":
        table = grid(kind)
        table.round(2).to_csv(OUT_DIR / f"grade_{kind}.csv", index=False)
        print(table.round(1).sort_values("alerta_t").head(25).to_string())
        print(choose(table))
