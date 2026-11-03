"""Referências e limiares relativos ao usuário/vídeo.

Há DOIS métodos de referência, de propósito separados e com nomes distintos:
  - reference_from_first_frames: o do demo de webcam (média dos frames de calibração, em que
    o usuário olha para a câmera de olhos abertos e boca fechada);
  - reference_from_percentile: o da análise de vídeos prontos (percentil do vídeo inteiro,
    sem fase de calibração e robusto a piscadas e picos).
Os limiares são funções do fator, que recebe a constante de parameters como padrão.
"""
import numpy as np

from detection import parameters as P


def reference_from_first_frames(values: list[float]) -> float:
    """Média dos valores coletados na calibração do demo."""
    return sum(values) / len(values)


def reference_from_percentile(values: np.ndarray, percentile: float) -> float:
    """Percentil dos valores válidos de um vídeo (NaN é ignorado)."""
    return float(np.nanpercentile(values, percentile))


def ear_threshold(open_eye_reference: float, factor: float = P.EAR_CLOSING_FACTOR) -> float:
    """EAR abaixo disto = olho fechando."""
    return open_eye_reference * factor


def mar_threshold(rest_reference: float, factor: float = P.MAR_YAWN_FACTOR) -> float:
    """MAR acima disto = boca aberta como em bocejo."""
    return rest_reference * factor


def pitch_drop_threshold(neutral_pitch: float, offset_deg: float = P.PITCH_DROP_OFFSET_DEG) -> float:
    """Pitch abaixo disto = cabeça caindo."""
    return neutral_pitch + offset_deg


def chin_drop_threshold(neutral_chin_y: float, offset_px: float = P.CHIN_DROP_OFFSET_PX) -> float:
    """Queixo abaixo (y maior) disto = cabeça caindo."""
    return neutral_chin_y + offset_px
