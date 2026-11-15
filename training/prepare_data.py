"""Builds the training windows from the per-frame CSVs in data/features/.

Per video: invalid frames become gaps -> resample to 10 Hz (mean of valid frames per 0.1 s step; gaps
up to 0.3 s interpolated, longer ones stay missing) -> measure each channel against a reference ->
60 s windows with a 30 s stride.

Two calibrations, saved in separate files with the SAME windows:
  realistic : reference = first minute of the video itself (excluded from the windows)
  optimistic: reference = the person's alert (0) video (as in the UTA-RLDD paper)

Usage (from the root): python -m training.prepare_data [cycle]   (default 1; cycles 2+ add the minimum-EAR channel)
Output: data/training[/ciclo2]/windows_<calibration>.npz and videos.csv
"""
import numpy as np
import pandas as pd

from detection import parameters as P
from detection.preprocessing import reference_of, relative, resample
from detection.validity import is_valid_frame
from training import config as C


def resample_video(csv_path, n_measures: int = C.N_MEASURES) -> tuple[np.ndarray, dict]:
    """Return (10 Hz series of shape (T, n_measures) with NaN gaps, metadata) for one CSV.

    The resampling itself lives in detection/preprocessing.py, the same code as the live demo."""
    data = pd.read_csv(csv_path, dtype={"participant": str})
    data = data.sort_values("timestamp_s").drop_duplicates("timestamp_s")
    valid = is_valid_frame(data["face_detected"].to_numpy(), data["yaw"].to_numpy(),
                           data["roll"].to_numpy(), data["ear_left"].to_numpy(),
                           data["ear_right"].to_numpy())
    values = data[P.MEASURE_NAMES].to_numpy()
    valid &= np.isfinite(values).all(axis=1)
    series, observed = resample(data["timestamp_s"].to_numpy(), values, valid, n_measures)
    n_steps = len(series)
    filled = int(np.isfinite(series[:, 0]).sum() - observed.sum())
    meta = {"participant": data["participant"].iloc[0], "label": int(data["label"].iloc[0]),
            "fold": data["fold"].iloc[0], "n_steps": n_steps,
            "pct_observed_raw": 100 * observed.mean(),
            "pct_interpolated": 100 * filled / n_steps,
            "pct_missing": 100 * np.isnan(series[:, 0]).mean()}
    return series, meta


def windows_of(rel: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """Windows (start step, 60 s x 10 Hz tensor) after the calibration minute."""
    out = []
    start = C.CALIBRATION_STEPS
    while start + C.WINDOW_STEPS <= len(rel):
        win = rel[start:start + C.WINDOW_STEPS]
        if np.isfinite(win[:, 0]).mean() >= C.MIN_OBSERVED_FRACTION:
            out.append((start, win))
        start += C.WINDOW_STRIDE_STEPS
    return out


def main(cycle: int = 1) -> None:
    """Resample every video and save the windows for each calibration."""
    out_dir = C.cycle_dir(cycle)
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = {}
    for csv_path in sorted(P.FEATURES_DIR.glob("Fold*/*/*.csv")):
        series, meta = resample_video(csv_path, C.n_measures(cycle))
        videos[(meta["participant"], meta["label"])] = (series, meta)
    print(f"{len(videos)} vídeos reamostrados a {C.RESAMPLE_HZ} Hz")

    persons = sorted({p for p, _ in videos})
    incomplete = [p for p in persons if any((p, lab) not in videos for lab in (0, 5, 10))]
    assert not incomplete, f"participantes sem os 3 vídeos: {incomplete}"

    rows = []
    calibrations = ((C.CALIBRATION_REALISTIC,) if cycle in C.PART1_CYCLES
                    else (C.CALIBRATION_REALISTIC, C.CALIBRATION_OPTIMISTIC))
    cal_steps = C.calibration_s(cycle) * C.RESAMPLE_HZ    # realistic reference = first segment of the video
    for calibration in calibrations:
        X, y, person, fold, start = [], [], [], [], []
        for (p, label), (series, meta) in sorted(videos.items()):
            if calibration == C.CALIBRATION_REALISTIC:
                ref = reference_of(series[:cal_steps])
            else:
                ref = reference_of(videos[(p, P.ALERT_LABEL)][0])
            if ref is None:
                print(f"  [{calibration}] {p}/{label}: sem dado válido para a referência; vídeo fora")
                continue
            wins = windows_of(relative(series, ref))
            for s, w in wins:
                X.append(w.astype(np.float32)); y.append(label); person.append(p)
                fold.append(meta["fold"]); start.append(s * C.STEP_S)
            if calibration == C.CALIBRATION_REALISTIC:
                rows.append({**meta, "n_windows": len(wins),
                             "ref_ear": ref[0], "ref_mar": ref[1], "ref_pitch": ref[2]})
        X = np.stack(X)
        np.savez_compressed(out_dir / f"windows_{calibration}.npz", X=X, y=np.array(y),
                            person=np.array(person), fold=np.array(fold), start_s=np.array(start))
        print(f"[{calibration}] {len(X)} janelas, shape {X.shape}, "
              f"{100 * np.isnan(X[:, :, 0]).mean():.1f}% dos passos ausentes")
    pd.DataFrame(rows).to_csv(out_dir / "videos.csv", index=False)


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
