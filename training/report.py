"""Metrics and report (data/training/relatorio.md) built from the predictions in results/.

The main metric is per VIDEO; per window is secondary because neighboring windows overlap by 50%."""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

from training import config as C
from training import models as M

MODEL_TITLES = {"regra_fixa": "Regra PERCLOS > 20%", "regra_aprendida": "Regra com limiar aprendido",
                "random_forest": "Random Forest", "lstm": "LSTM", "gru": "GRU"}
MODEL_NAMES = list(MODEL_TITLES)
METRICS = ["acc", "bal_acc", "f1", "precision", "recall"]
CLASS_NAMES = {0: "alerta (0)", 5: "baixa vigilância (5)", 10: "sonolento (10)"}


def load_dataset(cycle: int, key: str) -> dict:
    spec = C.experiments(cycle)[key]
    data = np.load(C.cycle_dir(cycle) / f"windows_{spec['calibration']}.npz")
    keep = np.isin(data["y"], spec["labels"]) & ~np.isin(data["person"], spec["exclude"])
    labels = np.array(spec["labels"])
    return {"label": data["y"][keep], "y": np.searchsorted(labels, data["y"][keep]),
            "person": data["person"][keep], "fold": data["fold"][keep], "labels": labels}


def scores(y_true, y_pred, n_classes) -> dict:
    average = "binary" if n_classes == 2 else "macro"
    # positive class = the last one (drowsy in the binary experiments)
    return {"acc": accuracy_score(y_true, y_pred), "bal_acc": balanced_accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
            "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
            "recall": recall_score(y_true, y_pred, average=average, zero_division=0)}


def result_files(cycle: int, key: str, model: str, tag: str, k: int) -> list:
    folder = C.results_dir(cycle)
    if C.N_SEEDS[cycle] > 1 and model in C.RNN_KINDS[cycle]:
        return sorted(folder.glob(f"{key}{tag}_{model}_Fold{k}_s*.npz"))
    path = folder / f"{key}{tag}_{model}_Fold{k}.npz"
    return [path] if path.exists() else []


def evaluate(cycle: int, key: str, model: str, tag: str = "", restrict_exclude=None) -> dict | None:
    data = load_dataset(cycle, key)
    n_classes = len(data["labels"])
    per_fold = [result_files(cycle, key, model, tag, k) for k in range(1, C.N_FOLDS + 1)]
    n_seeds = len(per_fold[0])
    if n_seeds == 0 or any(len(files) != n_seeds for files in per_fold):
        return None
    window = np.zeros((n_seeds, C.N_FOLDS, len(METRICS)))
    video = np.zeros((n_seeds, C.N_FOLDS, len(METRICS)))
    cm_window = np.zeros((n_seeds, n_classes, n_classes))
    cm_video = np.zeros((n_seeds, n_classes, n_classes))
    video_table = []
    for k, files in enumerate(per_fold):
        for s, path in enumerate(files):
            res = np.load(path)
            idx, probs = res["test_idx"], res["probs"]
            meta = json.loads(str(res["meta"]))
            if restrict_exclude:
                keep = ~np.isin(data["person"][idx], restrict_exclude)
                idx, probs, score = idx[keep], probs[keep], res["score"][keep]
            else:
                score = res["score"]
            y = data["y"][idx]
            pred = probs.argmax(1)
            window[s, k] = list(scores(y, pred, n_classes).values())
            cm_window[s] += confusion_matrix(y, pred, labels=range(n_classes))
            frame = pd.DataFrame({"person": data["person"][idx], "label": data["label"][idx], "y": y,
                                  "score": score})
            for c in range(n_classes):
                frame[f"p{c}"] = probs[:, c]
            by_video = frame.groupby(["person", "label"]).mean(numeric_only=True).reset_index()
            if model.startswith("regra"):
                by_video["pred"] = M.rule_predict(by_video["score"].to_numpy(), meta["thresholds"])
            else:
                by_video["pred"] = by_video[[f"p{c}" for c in range(n_classes)]].to_numpy().argmax(1)
            by_video["y"] = by_video["y"].round().astype(int)
            video[s, k] = list(scores(by_video["y"], by_video["pred"], n_classes).values())
            cm_video[s] += confusion_matrix(by_video["y"], by_video["pred"], labels=range(n_classes))
            if s == 0:
                by_video["fold"] = f"Fold{k + 1}"
                video_table.append(by_video)
    columns = METRICS
    return {"window": pd.DataFrame(window.mean(axis=0), columns=columns),
            "video": pd.DataFrame(video.mean(axis=0), columns=columns),
            "seed_std_video_acc": float(video[:, :, 0].mean(axis=1).std(ddof=1)) if n_seeds > 1 else None,
            "n_seeds": n_seeds, "cm_window": cm_window.mean(axis=0), "cm_video": cm_video.mean(axis=0),
            "pooled_video_acc": float(np.mean([np.trace(c) / c.sum() for c in cm_video])),
            "labels": data["labels"], "videos": pd.concat(video_table)}


def pm(series: pd.Series) -> str:
    return f"{100 * series.mean():.1f} ± {100 * series.std(ddof=1):.1f}"


def cm_markdown(cm: np.ndarray, labels) -> str:
    names = [CLASS_NAMES[int(l)] for l in labels]
    fmt = (lambda v: f"{v:.1f}") if not float(cm.sum()).is_integer() else (lambda v: str(int(v)))
    lines = ["| real \\ previsto | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    lines += [f"| {names[i]} | " + " | ".join(fmt(v) for v in row) + " |" for i, row in enumerate(cm)]
    return "\n".join(lines)


def df_markdown(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def counts_section(cycle: int) -> list[str]:
    exclude = C.experiments(cycle)["A"]["exclude"]
    data = np.load(C.cycle_dir(cycle) / "windows_realistic.npz")
    df = pd.DataFrame({"fold": data["fold"], "y": data["y"], "person": data["person"]})
    df = df[~df.person.isin(exclude)]
    lines = ["## 1. Dados", "",
             f"Janelas de {C.WINDOW_S} s, passo de {C.WINDOW_STRIDE_S} s, a {C.RESAMPLE_HZ} Hz "
             f"({C.WINDOW_STEPS} passos), depois do 1º minuto de cada vídeo. As duas calibrações usam "
             "exatamente as mesmas janelas (só muda a referência).", "",
             f"Total: {len(df)} janelas de {df.person.nunique()} pessoas"
             + (f" (fora: {', '.join(exclude)})." if exclude else "."), "",
             "| Fold | Pessoas | Janelas classe 0 | Janelas classe 5 | Janelas classe 10 | Total |",
             "|---|---|---|---|---|---|"]
    for fold, g in df.groupby("fold"):
        c = g.y.value_counts()
        lines.append(f"| {fold} | {g.person.nunique()} | {c.get(0, 0)} | {c.get(5, 0)} | {c.get(10, 0)} | {len(g)} |")
    c = df.y.value_counts()
    lines.append(f"| **Total** | {df.person.nunique()} | {c.get(0, 0)} | {c.get(5, 0)} | {c.get(10, 0)} | {len(df)} |")
    lines += ["", "Os experimentos binários usam só as classes 0 e 10 (a classe 5 aparece acima apenas "
              "para o ciclo 1, experimento C).", ""]
    return lines


def experiment_sections(cycle: int) -> tuple[list[str], list[dict]]:
    lines, final_rows = [], []
    models = [m for m in MODEL_NAMES if m in C.RNN_KINDS[cycle] or m in
              ("regra_fixa", "regra_aprendida", "random_forest")]
    for key, spec in C.experiments(cycle).items():
        lines += ["", f"## Experimento {key}: {spec['desc']}", ""]
        table = ["| Modelo | Vídeo: acurácia | Vídeo: ac. balanceada | Vídeo: F1 | Vídeo: acurácia (todos juntos) | "
                 "Janela: acurácia | Janela: ac. balanceada | Janela: F1 | dp entre sementes (vídeo) |",
                 "|---|---|---|---|---|---|---|---|---|"]
        details = []
        for model in models:
            ev = evaluate(cycle, key, model)
            if ev is None:
                continue
            w, v = ev["window"], ev["video"]
            seed_std = "—" if ev["seed_std_video_acc"] is None else f"{100 * ev['seed_std_video_acc']:.1f}"
            table.append(f"| {MODEL_TITLES[model]} | {pm(v.acc)} | {pm(v.bal_acc)} | {pm(v.f1)} | "
                         f"{100 * ev['pooled_video_acc']:.1f} | {pm(w.acc)} | {pm(w.bal_acc)} | {pm(w.f1)} | {seed_std} |")
            seeds_note = f" (média de {ev['n_seeds']} sementes)" if ev["n_seeds"] > 1 else ""
            details += [f"**{MODEL_TITLES[model]}**{seeds_note} — matriz de confusão por vídeo "
                        "(média das probabilidades das janelas"
                        + ("; regra: PERCLOS médio do vídeo" if model.startswith("regra") else "")
                        + ", soma dos 5 folds):", "", cm_markdown(ev["cm_video"], ev["labels"]), "",
                        "Por janela (soma dos 5 folds):", "", cm_markdown(ev["cm_window"], ev["labels"]), ""]
            final_rows.append({"Experimento": key, "Modelo": MODEL_TITLES[model],
                               "Vídeo: acurácia": pm(v.acc), "Vídeo: ac. balanceada": pm(v.bal_acc),
                               "Janela: ac. balanceada": pm(w.bal_acc)})
        lines += table + [""] + details
        for model in ("regra_aprendida", "random_forest", *C.RNN_KINDS[cycle]):
            metas = []
            for k in range(1, C.N_FOLDS + 1):
                metas.append([json.loads(str(np.load(f)["meta"])) for f in result_files(cycle, key, model, "", k)])
            if not metas[0]:
                continue
            if model == "regra_aprendida":
                lines.append("Limiares aprendidos (PERCLOS %) por fold: "
                             + "; ".join(str([round(t, 1) for t in m[0]["thresholds"]]) for m in metas) + ".")
            elif model == "random_forest":
                imp = pd.DataFrame([m[0]["importances"] for m in metas]).mean().sort_values(ascending=False)
                lines.append("Random Forest: importância média: " + ", ".join(f"{n} {v:.2f}" for n, v in imp.items()) + ".")
            else:
                epochs = [np.mean([x["best_epoch"] for x in m]) for m in metas]
                seconds = [np.mean([x["train_s"] for x in m]) for m in metas]
                lines.append(f"{MODEL_TITLES[model]}: melhor época média por fold: "
                             + ", ".join(f"{e:.1f}" for e in epochs) + "; tempo médio de treino por fold (s): "
                             + ", ".join(f"{s:.0f}" for s in seconds) + ".")
        lines.append("")
    return lines, final_rows


def leakage_section(cycle: int) -> list[str]:
    perm = []
    for key in C.experiments(cycle):
        for model in MODEL_NAMES:
            ev = evaluate(cycle, key, model, "_perm")
            if ev is not None:
                perm.append(f"| {key} | {MODEL_TITLES[model]} | {100 * ev['pooled_video_acc']:.1f} | "
                            f"{pm(ev['window'].bal_acc)} |")
    lines = ["## Checagem de vazamento", "",
             "Sempre: o script confere (assert) que nenhuma pessoa está em treino e teste do mesmo fold; "
             "a normalização das redes e os limiares da regra aprendida usam só as janelas de treino; a "
             "parada antecipada usa pessoas separadas dentro do treino; o 1º minuto (calibração) não "
             "entra em nenhuma janela no modo realista.", ""]
    if perm:
        lines += ["Resultado acima de 90% em algum modelo: controle com rótulos sorteados por vídeo no treino "
                  "(teste com rótulos reais); sem vazamento deve cair para perto de 50%.", "",
                  "| Exp. | Modelo | Acurácia por vídeo (%) | Ac. balanceada por janela (%) |", "|---|---|---|---|"] + perm
    else:
        lines.append("Nenhum resultado passou de 90%; o controle com rótulos sorteados não foi necessário.")
    return lines + [""]


def comparison_section() -> list[str]:
    rows = []
    restrict = C.CYCLE2_EXCLUDE
    for key in ("A", "B"):
        for model in MODEL_NAMES:
            variants = [("Ciclo 1 (58 pessoas)", evaluate(1, key, model)),
                        ("Ciclo 1 (55 pessoas, sem 26, 44, 45)", evaluate(1, key, model, restrict_exclude=restrict)),
                        ("Ciclo 2 (55 pessoas)", evaluate(2, key, model))]
            for name, ev in variants:
                if ev is None:
                    continue
                rows.append({"Exp.": key, "Modelo": MODEL_TITLES[model], "Ciclo": name,
                             "Vídeo: acurácia": pm(ev["video"].acc),
                             "Vídeo: ac. balanceada": pm(ev["video"].bal_acc),
                             "Janela: ac. balanceada": pm(ev["window"].bal_acc)})
    return ["## Comparação ciclo 1 × ciclo 2 (experimentos A e B)", "",
            "Métrica principal: por vídeo. O ciclo 1 usava 1 semente (LSTM); o ciclo 2 usa a média de 5 "
            "sementes por fold e acrescenta o canal de EAR mínimo (só as redes o recebem; a regra e o "
            "Random Forest usam as mesmas estatísticas do ciclo 1, então a diferença neles vem só de "
            "quem está no teste). A linha \"55 pessoas\" do ciclo 1 reaproveita as previsões originais, "
            "feitas por modelos treinados com as 58 pessoas.", "",
            df_markdown(pd.DataFrame(rows)) if rows else "(sem resultados)", ""]


def rotation_section() -> list[str]:
    return ["## Rotação do vídeo (26/5, 26/10, 45/0) e o roll do 26", "",
            "`rotation_deg` 180 (26/5, 45) e 270 (26/10) é aplicado do jeito certo em "
            "`pipeline/extract_features.py`: nos três vídeos o rosto sai em pé com a rotação da metadata "
            "(checado com landmarks desenhados). Nada foi corrigido nem reextraído, e a hipótese do "
            "relatório do ciclo 2 (rotação errada) estava errada.", "",
            "O que acontece no 26/5: com o rosto em pé e quase de frente (frames 0, 7261 e 17877: yaw "
            "-33°, 7° e 2°; pitch 20°, 0° e -1°), o roll gravado é 162°, 177° e 177°. É a solução "
            "invertida do `solvePnP` (decomposição em ângulos de Euler com o roll perto de ±180°), não a "
            "pose real. Por isso a regra de |roll| > 45° em `detection/validity.py` descarta frames "
            "bons nesse vídeo (e no 26/10). Não foi corrigido: mexer no cálculo de pose exigiria "
            "reextrair e é uma decisão à parte.", ""]


def comparison3_section() -> list[str]:
    rows = []
    for key in ("A", "B"):
        for model in MODEL_NAMES:
            variants = [("Ciclo 2 (55 pessoas, sem 26, 44, 45)", evaluate(2, key, model)),
                        ("Ciclo 3 (57 pessoas, sem 44)", evaluate(3, key, model)),
                        ("Ciclo 3 restrito às 55 pessoas do ciclo 2",
                         evaluate(3, key, model, restrict_exclude=C.CYCLE2_EXCLUDE))]
            for name, ev in variants:
                if ev is None:
                    continue
                rows.append({"Exp.": key, "Modelo": MODEL_TITLES[model], "Ciclo": name,
                             "Vídeo: acurácia": pm(ev["video"].acc),
                             "Vídeo: ac. balanceada": pm(ev["video"].bal_acc),
                             "Janela: ac. balanceada": pm(ev["window"].bal_acc)})
    return ["## Comparação ciclo 2 × ciclo 3 (experimentos A e B)", "",
            "Métrica principal: por vídeo. Mesma configuração nos dois ciclos (canal de EAR mínimo, 5 "
            "sementes, GRU); muda só quem participa: o ciclo 3 inclui o 26 e o 45. A linha \"restrito\" "
            "usa as previsões do ciclo 3 só nas pessoas que também estão no ciclo 2.", "",
            df_markdown(pd.DataFrame(rows)) if rows else "(sem resultados)", ""]


def write_report(cycle: int, log=print) -> None:
    if cycle == 3:
        lines = ["# Relatório do treino: ciclo 3 (ciclo 2 com o 26 e o 45 de volta)", "",
                 "Mesma configuração do ciclo 2 (canal de EAR mínimo; LSTM e GRU com 5 sementes por fold, "
                 "média; só A realista e B otimista, binários), excluindo só o participante 44. "
                 "**Métrica principal: por vídeo**; por janela é secundária. Médias ± desvio entre os 5 "
                 "folds, em %.", ""]
        lines += counts_section(3)
        body, final_rows = experiment_sections(3)
        lines += body + ["## Tabela final do ciclo 3", "", df_markdown(pd.DataFrame(final_rows)), ""]
        lines += comparison3_section() + leakage_section(3) + rotation_section()
        C.REPORT_CYCLE3_PATH.write_text("\n".join(lines), encoding="utf-8")
        log(f"Relatório do ciclo 3 gravado em {C.REPORT_CYCLE3_PATH}")
        return
    if cycle == 1:
        lines = ["# Relatório do primeiro treino (ciclo 1: regra, Random Forest e LSTM)", "",
                 "Validação cruzada nos 5 folds oficiais do UTA-RLDD (uma pessoa nunca está em treino e "
                 "teste). Médias ± desvio entre os 5 folds, em %. F1 = classe sonolento (2 classes) ou "
                 "macro (3 classes).", ""]
        lines += counts_section(1)
        body, final_rows = experiment_sections(1)
        lines += body + ["## Tabela final", "", df_markdown(pd.DataFrame(final_rows)), ""] + leakage_section(1)
        C.REPORT_CYCLE1_PATH.write_text("\n".join(lines), encoding="utf-8")
        log(f"Relatório do ciclo 1 gravado em {C.REPORT_CYCLE1_PATH}")
        return
    lines = ["# Relatório do treino: ciclo 2 (com o ciclo 1 ao final)", "",
             "Ciclo 2 = ciclo 1 com: canal de EAR mínimo por intervalo de 0,1 s; participantes 26, 44 e 45 "
             "fora; LSTM e GRU com 5 sementes por fold (média); só os experimentos A (calibração realista) e "
             "B (otimista), binários. **Métrica principal: por vídeo** (média das probabilidades das janelas); "
             "por janela é secundária. Médias ± desvio entre os 5 folds, em %.", ""]
    lines += counts_section(2)
    body, final_rows = experiment_sections(2)
    lines += body
    lines += ["## Tabela final do ciclo 2", "", df_markdown(pd.DataFrame(final_rows)), ""]
    lines += comparison_section() + leakage_section(2)
    lines += ["## Por que o 26/5 foi excluído", "",
              "No 26/5, 99,4% dos frames com rosto são inválidos pela regra de `detection/validity.py`: "
              "64,7% só pela regra de **|roll| > 45°**, 24,7% só pela de |yaw| > 45° e 9,7% pelas duas. "
              "A mediana do roll é 163° (63% dos frames entre 135° e 180°), isto é, o rosto sai de cabeça "
              "para baixo no solvePnP. No 26/10 o mesmo acontece (82,6% inválidos, 77,5% só pelo roll; "
              "mediana do roll 155°). CORREÇÃO (ciclo 3): a hipótese de rotação de vídeo estava errada; o "
              "rosto sai em pé e quase de frente, e o roll de ~177° é a solução invertida do solvePnP "
              "(ver o relatório do ciclo 3). Nada foi corrigido; os três participantes só ficaram fora.", ""]
    if C.REPORT_CYCLE1_PATH.exists():
        lines += ["---", "", "# Ciclo 1 (relatório original, preservado)", "",
                  C.REPORT_CYCLE1_PATH.read_text(encoding="utf-8").replace("\n# ", "\n## ", 1)]
    C.REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(f"Relatório gravado em {C.REPORT_PATH}")
