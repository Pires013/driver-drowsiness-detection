"""Reamostragem a 10 Hz, referência (calibração) e montagem da janela da rede.

Código ÚNICO do treino (training/prepare_data.py) e da demonstração ao vivo (demos/main_demo.py):
os dois tratam os dados do mesmo jeito.

Por vídeo/sessão: frame inválido (detection/validity.py) vira falta -> média dos frames válidos de
cada intervalo de 0,1 s (e EAR mínimo do intervalo) -> buraco de até 0,3 s interpolado, maior fica
como falta -> medida relativa à referência da pessoa (EAR e MAR em razão, pose em diferença).
"""
import numpy as np

from detection import parameters as P
from detection.calibration import reference_from_percentile


def resample(t: np.ndarray, values: np.ndarray, valid: np.ndarray, n_measures: int = P.N_MEASURES_BASE + 1,
             first_step: int = 0, n_steps: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Série a 10 Hz (n_steps, n_measures), NaN nas faltas, e a máscara de passos observados.

    t: instantes (s); values: (N, 5) = EAR médio, MAR, pitch, yaw, roll; valid: máscara de frames
    válidos. Canais 0-4 = média do intervalo de 0,1 s; canal 5 (se n_measures > 5) = EAR mínimo do
    intervalo (uma piscada de 2 frames quase some na média). `first_step` e `n_steps` recortam um
    trecho (uso ao vivo); o passo k cobre [k*0,1 s, (k+1)*0,1 s).
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

    # Interpola só os buracos de até MAX_INTERP_GAP_STEPS passos entre dois passos observados
    idx = np.flatnonzero(observed)
    for a, b in zip(idx[:-1], idx[1:]):
        gap = b - a - 1
        if 0 < gap <= P.MAX_INTERP_GAP_STEPS:
            frac = (np.arange(a + 1, b) - a) / (b - a)
            series[a + 1:b] = series[a] + frac[:, None] * (series[b] - series[a])
    return series, observed


def reference_of(series: np.ndarray) -> np.ndarray | None:
    """[EAR p75, MAR mediana, pitch, yaw, roll medianas] dos passos observados; None se não há."""
    if np.isnan(series[:, 0]).all():
        return None
    return np.array([
        reference_from_percentile(series[:, 0], P.OPEN_EYE_PERCENTILE),
        reference_from_percentile(series[:, 1], P.MAR_REST_PERCENTILE),
        *[reference_from_percentile(series[:, ch], 50) for ch in (2, 3, 4)]])


def wrap_deg(angle):
    """Leva um ângulo (ou diferença de ângulos) em graus para [-180, 180): 177 - (-177) = 354 vira -6."""
    return (np.asarray(angle, dtype=float) + 180.0) % 360.0 - 180.0


def relative(series: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """EAR (e EAR mínimo) e MAR como razão da referência; pitch, yaw e roll como diferença (graus),
    sempre em [-180, 180) (a diferença "dá a volta" em vez de passar de 180)."""
    out = series.copy()
    out[:, 0] /= ref[0]
    out[:, 1] /= ref[1]
    out[:, 2:P.N_MEASURES_BASE] = wrap_deg(out[:, 2:P.N_MEASURES_BASE] - ref[2:])
    if out.shape[1] > P.N_MEASURES_BASE:
        out[:, P.N_MEASURES_BASE:] /= ref[0]
    return out


def window_ok(window: np.ndarray) -> bool:
    """A janela tem observação suficiente (>= 50% dos passos com EAR)?"""
    return bool(np.isfinite(window[:, 0]).mean() >= P.MIN_OBSERVED_FRACTION)


class LiveBuffer:
    """Guarda os frames recentes de uma sessão ao vivo e monta a janela relativa dos últimos 60 s.

    Os instantes são contados desde o início da sessão (ou da última recalibração). Os primeiros
    `calibration_s` segundos formam a referência; a janela da rede só usa passos DEPOIS deles.
    """

    def __init__(self, calibration_s: float = P.LIVE_CALIBRATION_S, n_measures: int = P.N_MEASURES_BASE + 1):
        self.calibration_s = calibration_s
        self.n_measures = n_measures
        self.reset()

    def reset(self) -> None:
        self.t: list[float] = []
        self.values: list[list[float]] = []      # EAR médio, MAR, pitch, yaw, roll (NaN sem rosto)
        self.valid: list[bool] = []
        self.ref: np.ndarray | None = None
        self.calibration_failed = False

    def add(self, t: float, values: list[float], valid: bool) -> None:
        self.t.append(t)
        self.values.append(values)
        self.valid.append(valid)
        keep_from = t - (P.WINDOW_S + 10)        # só os últimos 70 s ficam (depois da calibração)
        if self.ref is not None and self.t[0] < keep_from:
            n = int(np.searchsorted(self.t, keep_from))
            del self.t[:n], self.values[:n], self.valid[:n]

    @property
    def calibrated(self) -> bool:
        return self.ref is not None

    def try_calibrate(self, now: float) -> bool:
        """Fixa a referência quando passaram `calibration_s` s; None se faltou dado (recomeça)."""
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
        """(janela (600, n_measures) relativa, passos observados) dos últimos 60 s; None se ainda não
        há 60 s depois da calibração ou se a janela tem pouca observação."""
        if self.ref is None or now - self.calibration_s < P.WINDOW_S:
            return None, 0
        end_step = int(now / P.STEP_S)           # passo atual (incompleto) não entra
        first_step = end_step - P.WINDOW_STEPS
        t = np.array(self.t)
        use = t >= first_step * P.STEP_S - 1.0
        series, _ = resample(t[use], np.array(self.values)[use], np.array(self.valid)[use],
                             self.n_measures, first_step=first_step, n_steps=P.WINDOW_STEPS)
        win = relative(series, self.ref)
        observed = int(np.isfinite(win[:, 0]).sum())
        return (win if window_ok(win) else None), observed
