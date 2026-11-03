"""Os três modelos: regra de PERCLOS, Random Forest sobre estatísticas da janela e LSTM pequena.

Todas as funções recebem as janelas já relativas à referência (shape (N, 600, 5), NaN = falta,
canais: EAR relativo, MAR relativo, pitch, yaw e roll relativos). Nada aqui lê arquivo nem sabe o
que é fold: quem decide quais janelas entram no treino é o run_experiments.
"""
import copy
import time

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from torch import nn

from detection import parameters as P
from training import config as C

STAT_NAMES = ["perclos_pct", "ear_medio", "fechamentos_longos", "piscadas", "bocejos", "pitch_std"]


# ----------------------------------------------------------------------
# Estatísticas da janela (usadas pela regra e pelo Random Forest)
# ----------------------------------------------------------------------
def run_lengths(mask: np.ndarray) -> np.ndarray:
    """Tamanho (em passos) de cada sequência contínua de True."""
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


def window_stats(window: np.ndarray) -> np.ndarray:
    """[PERCLOS %, EAR médio, fechamentos longos, piscadas, bocejos, desvio do pitch] de uma janela.

    Mesmos limiares da análise de vídeo (EAR < 75% da referência = fechando; MAR > 1,6x a
    referência por >= 1 s = bocejo), só que contados a 10 Hz. A falta quebra as sequências.
    """
    ear, mar, pitch = window[:, 0], window[:, 1], window[:, 2]
    observed = np.isfinite(ear)
    closed = ear < P.EAR_CLOSING_FACTOR                         # NaN < x é False
    lengths = run_lengths(closed)
    return np.array([
        P.PERCENT * closed[observed].mean(),
        np.nanmean(ear),
        (lengths >= C.LONG_CLOSURE_MIN_STEPS).sum(),
        ((lengths >= 1) & (lengths <= C.BLINK_MAX_STEPS)).sum(),
        (run_lengths(mar > P.MAR_YAWN_FACTOR) >= C.YAWN_MIN_STEPS).sum(),
        np.nanstd(pitch)])


def all_stats(X: np.ndarray) -> np.ndarray:
    return np.stack([window_stats(w) for w in X])


# ----------------------------------------------------------------------
# Regra de PERCLOS
# ----------------------------------------------------------------------
def rule_predict(perclos: np.ndarray, thresholds: list[float]) -> np.ndarray:
    """Classe = quantos limiares o PERCLOS ultrapassa (1 limiar: 0/1; 2 limiares: 0/1/2)."""
    return sum((perclos > t).astype(int) for t in thresholds)


def fit_rule(perclos: np.ndarray, y: np.ndarray, n_classes: int) -> list[float]:
    """Limiares que maximizam a acurácia balanceada NO TREINO (1 limiar ou 2 para 3 classes)."""
    candidates = np.unique(np.percentile(perclos, np.linspace(0, 100, 101)))
    if n_classes == 2:
        scores = [balanced_accuracy_score(y, rule_predict(perclos, [t])) for t in candidates]
        return [float(candidates[int(np.argmax(scores))])]
    best, best_pair = -1.0, None
    for i, t1 in enumerate(candidates):
        for t2 in candidates[i + 1:]:
            score = balanced_accuracy_score(y, rule_predict(perclos, [t1, t2]))
            if score > best:
                best, best_pair = score, [float(t1), float(t2)]
    return best_pair


# ----------------------------------------------------------------------
# Random Forest
# ----------------------------------------------------------------------
def fit_forest(stats: np.ndarray, y: np.ndarray, seed: int) -> RandomForestClassifier:
    forest = RandomForestClassifier(n_estimators=C.RF_TREES, min_samples_leaf=C.RF_MIN_LEAF,
                                    class_weight="balanced", n_jobs=-1, random_state=seed)
    return forest.fit(stats, y)


# ----------------------------------------------------------------------
# LSTM
# ----------------------------------------------------------------------
class SmallLSTM(nn.Module):
    """1 camada de 64 unidades; a saída de todos os passos é promediada no tempo (a janela tem 600
    passos e o rótulo vale para a janela inteira) e passa por dropout e uma camada linear."""

    def __init__(self, n_inputs: int, n_classes: int):
        super().__init__()
        self.lstm = nn.LSTM(n_inputs, C.LSTM_HIDDEN, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(C.LSTM_DROPOUT)
        self.out = nn.Linear(C.LSTM_HIDDEN, n_classes)

    def forward(self, x):
        seq, _ = self.lstm(x)
        return self.out(self.dropout(seq.mean(dim=1)))


class SmallGRU(nn.Module):
    """Igual à SmallLSTM, com GRU no lugar (mesma configuração: 1 camada, 64 unidades, dropout 0,3)."""

    def __init__(self, n_inputs: int, n_classes: int):
        super().__init__()
        self.gru = nn.GRU(n_inputs, C.LSTM_HIDDEN, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(C.LSTM_DROPOUT)
        self.out = nn.Linear(C.LSTM_HIDDEN, n_classes)

    def forward(self, x):
        seq, _ = self.gru(x)
        return self.out(self.dropout(seq.mean(dim=1)))


def make_rnn(kind: str, n_inputs: int, n_classes: int) -> nn.Module:
    return {"lstm": SmallLSTM, "gru": SmallGRU}[kind](n_inputs, n_classes)


def fit_normalization(X: np.ndarray) -> dict:
    """Média e desvio de cada medida SÓ nos passos observados das janelas de treino."""
    flat = X.reshape(-1, X.shape[2])
    mean = np.nanmean(flat, axis=0)
    std = np.nanstd(flat, axis=0)
    return {"mean": mean.astype(np.float32), "std": np.maximum(std, 1e-6).astype(np.float32)}


NORM_CLIP = 6.0


def to_input(X: np.ndarray, norm: dict) -> torch.Tensor:
    """(N, 600, 6): medidas normalizadas (falta = 0, limitadas a ±6 desvios) + canal de falta."""
    missing = ~np.isfinite(X[:, :, 0])
    z = np.clip((X - norm["mean"]) / norm["std"], -NORM_CLIP, NORM_CLIP)
    z = np.nan_to_num(z, nan=0.0)
    return torch.from_numpy(np.concatenate([z, missing[:, :, None]], axis=2).astype(np.float32))


def class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    return torch.from_numpy(len(y) / (n_classes * np.maximum(counts, 1)))


def predict_proba(model: nn.Module, inputs: torch.Tensor, batch: int = 128) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return np.concatenate([torch.softmax(model(inputs[i:i + batch]), dim=1).numpy()
                               for i in range(0, len(inputs), batch)])


def train_lstm(X_train, y_train, n_classes, seed, X_val=None, y_val=None, fixed_epochs=None,
               log=print, kind: str = "lstm") -> tuple[nn.Module, dict, int]:
    """Treina a LSTM. Com X_val: parada antecipada pela perda de validação (guarda o melhor
    estado). Com fixed_epochs (modelo final, sem validação): treina esse número de épocas.
    Devolve (modelo, normalização, melhor época)."""
    torch.set_num_threads(C.TORCH_THREADS)
    torch.manual_seed(seed)
    norm = fit_normalization(X_train)
    inputs = to_input(X_train, norm)
    targets = torch.from_numpy(y_train.astype(np.int64))
    weights = class_weights(y_train, n_classes)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    model = make_rnn(kind, inputs.shape[2], n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=C.LSTM_LR)
    val_inputs = to_input(X_val, norm) if X_val is not None else None
    val_targets = torch.from_numpy(y_val.astype(np.int64)) if X_val is not None else None
    generator = torch.Generator().manual_seed(seed)

    best_loss, best_state, best_epoch, waited = np.inf, None, 0, 0
    max_epochs = fixed_epochs or C.LSTM_MAX_EPOCHS
    started = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(inputs), generator=generator)
        total = 0.0
        for i in range(0, len(order), C.LSTM_BATCH):
            batch = order[i:i + C.LSTM_BATCH]
            optimizer.zero_grad()
            loss = loss_fn(model(inputs[batch]), targets[batch])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), C.LSTM_GRAD_CLIP)
            optimizer.step()
            total += loss.item() * len(batch)
        if val_inputs is None:
            best_epoch = epoch
            log(f"    época {epoch}: treino {total / len(inputs):.4f} ({time.time() - started:.0f} s)")
            if time.time() - started > C.LSTM_MAX_TRAIN_S:
                break
            continue
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val_inputs), val_targets).item()
        log(f"    época {epoch}: treino {total / len(inputs):.4f} val {val_loss:.4f} "
            f"({time.time() - started:.0f} s)")
        if val_loss < best_loss - 1e-4:
            best_loss, best_state, best_epoch, waited = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            waited += 1
            if waited >= C.LSTM_PATIENCE:
                break
        if time.time() - started > C.LSTM_MAX_TRAIN_S:
            log("    tempo limite de treino atingido")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, norm, best_epoch
