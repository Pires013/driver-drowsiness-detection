"""The three models: PERCLOS rule, Random Forest on window statistics and a small LSTM (plus GRU).

Functions take windows already relative to the reference (shape (N, 600, 5), NaN = missing; channels:
relative EAR, relative MAR, relative pitch, yaw, roll). Nothing here reads files or knows about folds:
run_experiments decides which windows go into training.
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


def run_lengths(mask: np.ndarray) -> np.ndarray:
    """Length (in steps) of each run of consecutive True values."""
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


def window_stats(window: np.ndarray) -> np.ndarray:
    """Window statistics: [PERCLOS %, mean EAR, long closures, blinks, yawns, pitch std].

    Same thresholds as the video analysis (EAR < 75% of reference = closing; MAR > 1.6x reference
    for >= 1 s = yawn), counted at 10 Hz. Missing steps break runs.
    """
    ear, mar, pitch = window[:, 0], window[:, 1], window[:, 2]
    observed = np.isfinite(ear)
    closed = ear < P.EAR_CLOSING_FACTOR                         # NaN < x is False
    lengths = run_lengths(closed)
    return np.array([
        P.PERCENT * closed[observed].mean(),
        np.nanmean(ear),
        (lengths >= C.LONG_CLOSURE_MIN_STEPS).sum(),
        ((lengths >= 1) & (lengths <= C.BLINK_MAX_STEPS)).sum(),
        (run_lengths(mar > P.MAR_YAWN_FACTOR) >= C.YAWN_MIN_STEPS).sum(),
        np.nanstd(pitch)])


def all_stats(X: np.ndarray) -> np.ndarray:
    """Window statistics for every window in X."""
    return np.stack([window_stats(w) for w in X])


def rule_predict(perclos: np.ndarray, thresholds: list[float]) -> np.ndarray:
    """Class = number of thresholds the PERCLOS exceeds (1 threshold: 0/1; 2 thresholds: 0/1/2)."""
    return sum((perclos > t).astype(int) for t in thresholds)


def fit_rule(perclos: np.ndarray, y: np.ndarray, n_classes: int) -> list[float]:
    """Thresholds maximizing balanced accuracy ON THE TRAINING SET (1 threshold, or 2 for 3 classes)."""
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


def fit_forest(stats: np.ndarray, y: np.ndarray, seed: int) -> RandomForestClassifier:
    """Random Forest on the window statistics."""
    forest = RandomForestClassifier(n_estimators=C.RF_TREES, min_samples_leaf=C.RF_MIN_LEAF,
                                    class_weight="balanced", n_jobs=-1, random_state=seed)
    return forest.fit(stats, y)


class SmallLSTM(nn.Module):
    """One 64-unit layer; outputs of all steps are averaged over time (the window has 600 steps and
    the label applies to the whole window), then pass through dropout and a linear layer."""

    def __init__(self, n_inputs: int, n_classes: int):
        super().__init__()
        self.lstm = nn.LSTM(n_inputs, C.LSTM_HIDDEN, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(C.LSTM_DROPOUT)
        self.out = nn.Linear(C.LSTM_HIDDEN, n_classes)

    def forward(self, x):
        seq, _ = self.lstm(x)
        return self.out(self.dropout(seq.mean(dim=1)))


class SmallGRU(nn.Module):
    """Same as SmallLSTM with a GRU instead (same configuration)."""

    def __init__(self, n_inputs: int, n_classes: int):
        super().__init__()
        self.gru = nn.GRU(n_inputs, C.LSTM_HIDDEN, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(C.LSTM_DROPOUT)
        self.out = nn.Linear(C.LSTM_HIDDEN, n_classes)

    def forward(self, x):
        seq, _ = self.gru(x)
        return self.out(self.dropout(seq.mean(dim=1)))


def make_rnn(kind: str, n_inputs: int, n_classes: int) -> nn.Module:
    """Build the recurrent network: kind is "lstm" or "gru"."""
    return {"lstm": SmallLSTM, "gru": SmallGRU}[kind](n_inputs, n_classes)


def fit_normalization(X: np.ndarray) -> dict:
    """Mean and std of each measure over observed steps of the training windows ONLY."""
    flat = X.reshape(-1, X.shape[2])
    mean = np.nanmean(flat, axis=0)
    std = np.nanstd(flat, axis=0)
    return {"mean": mean.astype(np.float32), "std": np.maximum(std, 1e-6).astype(np.float32)}


NORM_CLIP = 6.0


def to_input(X: np.ndarray, norm: dict) -> torch.Tensor:
    """Network input: normalized measures (missing = 0, clipped to +-6 std) plus a missing-data channel."""
    missing = ~np.isfinite(X[:, :, 0])
    z = np.clip((X - norm["mean"]) / norm["std"], -NORM_CLIP, NORM_CLIP)
    z = np.nan_to_num(z, nan=0.0)
    return torch.from_numpy(np.concatenate([z, missing[:, :, None]], axis=2).astype(np.float32))


def class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights for the loss."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    return torch.from_numpy(len(y) / (n_classes * np.maximum(counts, 1)))


def predict_proba(model: nn.Module, inputs: torch.Tensor, batch: int = 128) -> np.ndarray:
    """Softmax probabilities, computed in batches."""
    model.eval()
    with torch.no_grad():
        return np.concatenate([torch.softmax(model(inputs[i:i + batch]), dim=1).numpy()
                               for i in range(0, len(inputs), batch)])


def train_lstm(X_train, y_train, n_classes, seed, X_val=None, y_val=None, fixed_epochs=None,
               log=print, kind: str = "lstm") -> tuple[nn.Module, dict, int]:
    """Train the network. With X_val: early stopping on validation loss (keeps the best state).
    With fixed_epochs (final model, no validation): trains that many epochs.
    Returns (model, normalization, best epoch)."""
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
