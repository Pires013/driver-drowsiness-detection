"""Roda os experimentos (validação cruzada nos 5 folds oficiais) com os modelos de cada ciclo.

Uso (da raiz):
    python -m training.run_experiments                       # ciclo 1: A, B, C, D e o modelo final
    python -m training.run_experiments --cycle 2             # ciclo 2: A e B, LSTM e GRU x 5 sementes
    python -m training.run_experiments --cycle 3             # ciclo 3: como o 2, com 26 e 45 de volta
    python -m training.run_experiments --cycle 2 --exp A     # só alguns experimentos
    python -m training.run_experiments --benchmark           # mede o tempo de uma época e sai

Cada (experimento, fold, modelo[, semente]) grava as probabilidades em results/ e é pulado se já
existir, então dá para retomar uma rodada interrompida. O relatório sai de training.report.
"""
import argparse
import json
import time

import numpy as np
import torch

from training import config as C
from training import models as M
from training.report import evaluate, write_report


def load_experiment(key: str, cycle: int = 1) -> dict:
    """Janelas do experimento: filtra as classes e as pessoas excluídas e remapeia os rótulos."""
    spec = C.experiments(cycle)[key]
    data = np.load(C.cycle_dir(cycle) / f"windows_{spec['calibration']}.npz")
    keep = np.isin(data["y"], spec["labels"]) & ~np.isin(data["person"], spec["exclude"])
    labels = np.array(spec["labels"])
    return {"X": data["X"][keep], "label": data["y"][keep],
            "y": np.searchsorted(labels, data["y"][keep]), "person": data["person"][keep],
            "fold": data["fold"][keep], "n_classes": len(labels), "spec": spec}


def result_path(cycle: int, key: str, model: str, fold: str, seed: int | None = None):
    suffix = "" if seed is None else f"_s{seed}"
    return C.results_dir(cycle) / f"{key}_{model}_{fold}{suffix}.npz"


def inner_split(train_persons: np.ndarray, seed: int) -> np.ndarray:
    """Pessoas de treino separadas para a parada antecipada (a rede não treina com elas)."""
    persons = np.unique(train_persons)
    rng = np.random.RandomState(seed)
    return rng.choice(persons, size=max(1, round(C.INNER_VAL_FRACTION * len(persons))), replace=False)


def run_fold(cycle: int, key: str, exp: dict, fold: str, stats: np.ndarray, log, only_models=None,
             shuffle_labels: bool = False, tag: str = "", n_seeds: int | None = None) -> None:
    """Treina nos outros 4 folds e prevê o fold de teste, para cada modelo (e semente)."""
    test = exp["fold"] == fold
    train = ~test
    # Uma pessoa nunca está em treino e teste ao mesmo tempo (os folds oficiais separam pessoas)
    assert not set(exp["person"][train]) & set(exp["person"][test]), "pessoa em treino e teste"
    y_train = exp["y"][train].copy()
    if shuffle_labels:    # controle de vazamento: rótulos sorteados por vídeo no treino
        rng = np.random.RandomState(C.SEED)
        videos = sorted({(p, l) for p, l in zip(exp["person"][train], exp["label"][train])})
        new = {v: rng.randint(exp["n_classes"]) for v in videos}
        y_train = np.array([new[(p, l)] for p, l in zip(exp["person"][train], exp["label"][train])])
    fold_n = int(fold[-1])
    n_classes = exp["n_classes"]
    wanted = only_models or (C.base_models(cycle) + C.RNN_KINDS[cycle])
    C.results_dir(cycle).mkdir(parents=True, exist_ok=True)

    def save(model: str, probs: np.ndarray, score=None, extra=None, seed=None) -> None:
        np.savez(result_path(cycle, key + tag, model, fold, seed), test_idx=np.flatnonzero(test),
                 probs=probs, score=np.zeros(len(probs)) if score is None else score,
                 meta=json.dumps(extra or {}))

    def missing(model: str, seed=None) -> bool:
        return not result_path(cycle, key + tag, model, fold, seed).exists()

    if "regra_fixa" in wanted and n_classes == 2 and missing("regra_fixa"):
        pred = M.rule_predict(stats[test, 0], [C.RULE_PERCLOS_PCT])
        save("regra_fixa", np.eye(n_classes)[pred], stats[test, 0], {"thresholds": [C.RULE_PERCLOS_PCT]})
    if "regra_aprendida" in wanted and missing("regra_aprendida"):
        thresholds = M.fit_rule(stats[train, 0], y_train, n_classes)
        pred = M.rule_predict(stats[test, 0], thresholds)
        save("regra_aprendida", np.eye(n_classes)[pred], stats[test, 0], {"thresholds": thresholds})
    if "random_forest" in wanted and missing("random_forest"):
        forest = M.fit_forest(stats[train], y_train, C.SEED + fold_n)
        save("random_forest", forest.predict_proba(stats[test]),
             extra={"importances": dict(zip(M.STAT_NAMES, forest.feature_importances_.round(4).tolist()))})

    # Redes: a divisão de validação é a mesma em todas as sementes do fold (muda só o sorteio
    # dos pesos iniciais e a ordem dos lotes)
    val_persons = inner_split(exp["person"][train], C.SEED + fold_n)
    is_val = np.isin(exp["person"], val_persons) & train
    is_fit = train & ~is_val
    y_all = exp["y"].copy()
    y_all[train] = y_train
    seeds = range(n_seeds if n_seeds is not None else C.N_SEEDS[cycle])
    for kind in C.RNN_KINDS[cycle]:
        if kind not in wanted:
            continue
        for s in seeds:
            seed_arg = s if C.N_SEEDS[cycle] > 1 else None
            if not missing(kind, seed_arg):
                continue
            log(f"  {kind.upper()} {key}{tag} {fold} semente {s}: {is_fit.sum()} janelas de treino, "
                f"{is_val.sum()} de validação ({len(val_persons)} pessoas)")
            started = time.time()
            model, norm, best_epoch = M.train_lstm(exp["X"][is_fit], y_all[is_fit], n_classes,
                                                   C.SEED + 100 * fold_n + s, exp["X"][is_val],
                                                   y_all[is_val], log=lambda m: None, kind=kind)
            probs = M.predict_proba(model, M.to_input(exp["X"][test], norm))
            save(kind, probs, extra={"best_epoch": best_epoch, "train_s": round(time.time() - started, 1),
                                     "val_persons": val_persons.tolist()}, seed=seed_arg)
            if not shuffle_labels:
                C.models_dir(cycle).mkdir(parents=True, exist_ok=True)
                name = f"{kind}_{key}_{fold}" + (f"_s{s}" if seed_arg is not None else "")
                torch.save({"state": model.state_dict(), "norm": {k: v.tolist() for k, v in norm.items()},
                            "n_classes": n_classes, "labels": exp["spec"]["labels"], "kind": kind,
                            "channels": C.channels(cycle), "calibration": exp["spec"]["calibration"]},
                           C.models_dir(cycle) / f"{name}.pt")
            log(f"    melhor época {best_epoch}, {time.time() - started:.0f} s")


def train_final_model(log) -> None:
    """Modelo final do experimento A (ciclo 1): todos os dados, sem validação, nº de épocas =
    mediana das melhores épocas dos 5 folds (a parada antecipada precisa de validação)."""
    exp = load_experiment("A", 1)
    epochs = [json.loads(str(np.load(result_path(1, "A", "lstm", f"Fold{k}"))["meta"]))["best_epoch"]
              for k in range(1, C.N_FOLDS + 1)]
    n_epochs = int(np.median(epochs))
    log(f"Modelo final (experimento A): {len(exp['X'])} janelas, {n_epochs} épocas (melhores dos folds: {epochs})")
    model, norm, _ = M.train_lstm(exp["X"], exp["y"], exp["n_classes"], C.SEED, fixed_epochs=n_epochs, log=log)
    torch.save({"state": model.state_dict(), "norm": {k: v.tolist() for k, v in norm.items()},
                "n_classes": 2, "labels": exp["spec"]["labels"], "channels": C.CHANNELS,
                "calibration": exp["spec"]["calibration"], "epochs": n_epochs,
                "usage": "janela de 60 s a 10 Hz; referência = 1º minuto da sessão (EAR p75, MAR e "
                         "pose medianas); EAR e MAR em razão da referência, pose em diferença"},
               C.models_dir(1) / "lstm_final_A.pt")


def leakage_controls(cycle: int, keys, log) -> None:
    """Se algum resultado passa de 90% (por janela ou por vídeo), repete o treino com rótulos
    sorteados por vídeo; sem vazamento o resultado cai para perto do acaso."""
    for key in keys:
        suspicious = []
        for model in ("regra_aprendida", "random_forest", *C.RNN_KINDS[cycle]):
            ev = evaluate(cycle, key, model)
            if ev is not None and max(ev["window"].bal_acc.mean(), ev["video"].acc.mean()) > C.SUSPICIOUS_SCORE:
                suspicious.append(model)
        if not suspicious:
            continue
        log(f"Experimento {key}: resultado > {C.SUSPICIOUS_SCORE:.0%} em {suspicious}; rodando o controle com rótulos sorteados")
        exp = load_experiment(key, cycle)
        stats = M.all_stats(exp["X"])
        for k in range(1, C.N_FOLDS + 1):
            run_fold(cycle, key, exp, f"Fold{k}", stats, log, only_models=suspicious,
                     shuffle_labels=True, tag="_perm", n_seeds=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", type=int, default=1, choices=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--exp", nargs="+", default=None)
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--skip-nets", action="store_true")
    args = ap.parse_args()
    cycle = args.cycle
    keys = args.exp or list(C.experiments(cycle))
    C.results_dir(cycle).mkdir(parents=True, exist_ok=True)
    C.models_dir(cycle).mkdir(parents=True, exist_ok=True)
    log_file = open(C.cycle_dir(cycle) / "treino.log", "a", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    if args.benchmark:
        exp = load_experiment("A", cycle)
        idx = np.flatnonzero(exp["fold"] != "Fold1")[:320]
        norm = M.fit_normalization(exp["X"][idx])
        inputs = M.to_input(exp["X"][idx], norm)
        torch.set_num_threads(C.TORCH_THREADS)
        for kind in C.RNN_KINDS[cycle]:
            model = M.make_rnn(kind, inputs.shape[2], 2)
            opt = torch.optim.Adam(model.parameters())
            start = time.time()
            for i in range(0, len(inputs), C.LSTM_BATCH):
                opt.zero_grad()
                torch.nn.functional.cross_entropy(model(inputs[i:i + C.LSTM_BATCH]),
                                                  torch.from_numpy(exp["y"][idx][i:i + C.LSTM_BATCH])).backward()
                opt.step()
            print(f"{kind}: 320 janelas em {time.time() - start:.1f} s")
        return

    wanted = C.base_models(cycle) + ([] if args.skip_nets else C.RNN_KINDS[cycle])
    for key in keys:
        exp = load_experiment(key, cycle)
        stats = M.all_stats(exp["X"])
        log(f"=== Ciclo {cycle}, experimento {key}: {exp['spec']['desc']} ({len(exp['X'])} janelas, "
            f"{len(set(exp['person']))} pessoas) ===")
        for k in range(1, C.N_FOLDS + 1):
            run_fold(cycle, key, exp, f"Fold{k}", stats, log, only_models=wanted)
        log(f"Experimento {key} concluído")

    if cycle == 1 and "A" in keys and not args.skip_nets and not (C.models_dir(1) / "lstm_final_A.pt").exists():
        train_final_model(log)
    leakage_controls(cycle, keys, log)
    write_report(cycle, log)


if __name__ == "__main__":
    main()
