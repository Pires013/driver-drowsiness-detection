"""Resampling to 10 Hz, per-person reference and network window assembly.

Single code path shared by training (training/prepare_data.py) and the live demo (demos/main_demo.py).
Per video/session: invalid frames become gaps -> mean of valid frames per 0.1 s step (plus the step's
minimum EAR) -> gaps up to 0.3 s interpolated, longer ones stay missing -> values relative to the
person's reference (EAR and MAR as ratios, pose as differences).
"""
import numpy as np

from detection import parameters as P
from detection.calibration import reference_from_percentile


def resample(t: np.ndarray, values: np.ndarray, valid: np.ndarray, n_measures: int = P.N_MEASURES_BASE + 1,
             first_step: int = 0, n_steps: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Resample to 10 Hz: returns (series (n_steps, n_measures) with NaN gaps, observed-step mask).

    values is (N, 5) = mean EAR, MAR, pitch, yaw, roll. Channels 0-4 are the mean per 0.1 s step;
    channel 5 (if n_measures > 5) is the step's minimum EAR, since a 2-frame blink nearly vanishes
    in the mean. first_step/n_steps crop a segment (live use).
    """
    t = np.clip(np.asarray(t, dtype=float), 0, None)
    steps = (t / P.STEP_S).astype(int)
    if n_steps is None:
        n_steps = int(steps.max()) - first_step + 1
        bins = steps - first_step
    else:
        bins = steps - first_step
    inside = valid & (bins >= 0) & (bins < n_steps)
    bins = np.minimum(bins, n_steps - 1)

    series = np.full((n_steps, n_measures), np.nan)
    counts = np.bincount(bins[inside], minlength=n_steps)
    observed = counts > 0
    for ch in range(P.N_MEASURES_BASE):
        sums = np.bincount(bins[inside], weights=values[inside, ch], minlength=n_steps)
        series[observed, ch] = sums[observed] / counts[observed]
    if n_measures > P.N_MEASURES_BASE:
        ear_min = np.full(n_steps, np.inf)
        np.minimum.at(ear_min, bins[inside], values[inside, 0])
        series[observed, P.EAR_MIN_CHANNEL] = ear_min[observed]

    # interpolate only gaps up to MAX_INTERP_GAP_STEPS between two observed steps
    idx = np.flatnonzero(observed)
    for a, b in zip(idx[:-1], idx[1:]):
        gap = b - a - 1
        if 0 < gap <= P.MAX_INTERP_GAP_STEPS:
            frac = (np.arange(a + 1, b) - a) / (b - a)
            series[a + 1:b] = series[a] + frac[:, None] * (series[b] - series[a])
    return series, observed


def reference_of(series: np.ndarray) -> np.ndarray | None:
    """Reference [EAR p75, MAR median, pitch/yaw/roll medians] over observed steps; None if there are none."""
    if np.isnan(series[:, 0]).all():
        return None
    return np.array([
        reference_from_percentile(series[:, 0], P.OPEN_EYE_PERCENTILE),
        reference_from_percentile(series[:, 1], P.MAR_REST_PERCENTILE),
        *[reference_from_percentile(series[:, ch], 50) for ch in (2, 3, 4)]])


def wrap_deg(angle):
    """Wrap an angle (or angle difference) in degrees to [-180, 180): 177 - (-177) = 354 becomes -6."""
    return (np.asarray(angle, dtype=float) + 180.0) % 360.0 - 180.0


def relative(series: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """EAR (and minimum EAR) and MAR as ratios of the reference; pitch/yaw/roll as wrapped differences in degrees."""
    out = series.copy()
    out[:, 0] /= ref[0]
    out[:, 1] /= ref[1]
    out[:, 2:P.N_MEASURES_BASE] = wrap_deg(out[:, 2:P.N_MEASURES_BASE] - ref[2:])
    if out.shape[1] > P.N_MEASURES_BASE:
        out[:, P.N_MEASURES_BASE:] /= ref[0]
    return out


def window_ok(window: np.ndarray) -> bool:
    """True if enough steps have an EAR reading (MIN_OBSERVED_FRACTION)."""
    return bool(np.isfinite(window[:, 0]).mean() >= P.MIN_OBSERVED_FRACTION)


class LiveBuffer:
    """Keeps recent frames of a live session and builds the relative window of the last 60 s.

    Times count from session start (or the last recalibration). The first `calibration_s` seconds
    form the reference; the network window only uses steps AFTER them.
    """

    def __init__(self, calibration_s: float = P.LIVE_CALIBRATION_S, n_measures: int = P.N_MEASURES_BASE + 1):
        self.calibration_s = calibration_s
        self.n_measures = n_measures
        self.reset()

    def reset(self) -> None:
        self.t: list[float] = []
        self.values: list[list[float]] = []
        self.valid: list[bool] = []
        self.ref: np.ndarray | None = None
        self.calibration_failed = False

    def add(self, t: float, values: list[float], valid: bool) -> None:
        self.t.append(t)
        self.values.append(values)
        self.valid.append(valid)
        keep_from = t - (P.WINDOW_S + 10)        # keep only the last 70 s once calibrated
        if self.ref is not None and self.t[0] < keep_from:
            n = int(np.searchsorted(self.t, keep_from))
            del self.t[:n], self.values[:n], self.valid[:n]

    @property
    def calibrated(self) -> bool:
        return self.ref is not None

    def try_calibrate(self, now: float) -> bool:
        """Fix the reference once `calibration_s` has elapsed; flags failure (restart) if there is too little data."""
        if self.ref is not None:
            return True
        if now < self.calibration_s:
            return False
        t = np.array(self.t)
        use = t < self.calibration_s
        valid = np.array(self.valid)[use]
        if valid.sum() < P.LIVE_MIN_CALIBRATION_FRAMES:
            self.calibration_failed = True
            return False
        series, _ = resample(t[use], np.array(self.values)[use], valid, self.n_measures,
                             n_steps=round(self.calibration_s / P.STEP_S))
        self.ref = reference_of(series)
        return self.ref is not None

    def window(self, now: float) -> tuple[np.ndarray | None, int]:
        """Return (relative window of the last 60 s, observed steps); window is None before 60 s of post-calibration data or if too sparse."""
        if self.ref is None or now - self.calibration_s < P.WINDOW_S:
            return None, 0
        end_step = int(now / P.STEP_S)           # the current (incomplete) step is excluded
        first_step = end_step - P.WINDOW_STEPS
        t = np.array(self.t)
        use = t >= first_step * P.STEP_S - 1.0
        series, _ = resample(t[use], np.array(self.values)[use], np.array(self.valid)[use],
                             self.n_measures, first_step=first_step, n_steps=P.WINDOW_STEPS)
        win = relative(series, self.ref)
        observed = int(np.isfinite(win[:, 0]).sum())
        return (win if window_ok(win) else None), observed
