"""Constantes do treino (preparação dos dados, modelos e experimentos).

As constantes da detecção (limiar de EAR fechando, fator de bocejo, regra de frame inválido)
continuam em detection/parameters.py e são reaproveitadas daqui; este arquivo só tem o que é
específico do treino.
"""
from detection import parameters as P

TRAINING_DIR = P.DATA_DIR / "training"
REPORT_PATH = TRAINING_DIR / "relatorio.md"
REPORT_CYCLE1_PATH = TRAINING_DIR / "relatorio_ciclo1.md"   # relatório do ciclo 1, preservado
REPORT_CYCLE3_PATH = TRAINING_DIR / "relatorio_ciclo3.md"


# Parte 1 (duração da calibração): "ciclos" 4, 5 e 6 = calibração realista de 5, 10 e 60 s
PART1_CYCLES = {4: 5, 5: 10, 6: 60}
CYCLE_DIR_NAMES = {4: "parte1_cal5", 5: "parte1_cal10", 6: "parte1_cal60"}


def cycle_dir(cycle: int):
    """Ciclo 1 grava direto em data/training/; os seguintes em data/training/ciclo<N>/ (parte 1: parte1_cal<S>/)."""
    if cycle == 1:
        return TRAINING_DIR
    return TRAINING_DIR / CYCLE_DIR_NAMES.get(cycle, f"ciclo{cycle}")


def calibration_s(cycle: int) -> int:
    """Segundos do início do vídeo usados na referência da calibração realista (padrão: 60 s)."""
    return PART1_CYCLES.get(cycle, CALIBRATION_S)


def results_dir(cycle: int):
    return cycle_dir(cycle) / "results"


def models_dir(cycle: int):
    return cycle_dir(cycle) / "models"

SEED = 42

# ----------------------------------------------------------------------
# Reamostragem e janelas
# ----------------------------------------------------------------------
RESAMPLE_HZ = P.RESAMPLE_HZ          # constantes compartilhadas com a demonstração (detection/parameters.py)
STEP_S = P.STEP_S
MAX_INTERP_GAP_S = P.MAX_INTERP_GAP_S
MAX_INTERP_GAP_STEPS = P.MAX_INTERP_GAP_STEPS
WINDOW_S = P.WINDOW_S
WINDOW_STRIDE_S = 30
WINDOW_STEPS = WINDOW_S * RESAMPLE_HZ
WINDOW_STRIDE_STEPS = WINDOW_STRIDE_S * RESAMPLE_HZ
CALIBRATION_S = 60                   # 1º minuto do vídeo: calibração (fica fora das janelas)
CALIBRATION_STEPS = CALIBRATION_S * RESAMPLE_HZ
MIN_OBSERVED_FRACTION = P.MIN_OBSERVED_FRACTION   # janela com menos passos observados é descartada

# Canais do tensor da janela: medidas relativas à referência + máscara (1 = dado ausente).
# Ciclo 2 acrescenta o EAR mínimo de cada intervalo de 0,1 s (a média sozinha apaga piscadas).
CHANNELS = ["ear_rel", "mar_rel", "pitch_rel", "yaw_rel", "roll_rel", "missing"]
CHANNELS_CYCLE2 = ["ear_rel", "mar_rel", "pitch_rel", "yaw_rel", "roll_rel", "ear_min_rel", "missing"]
N_MEASURES = 5                       # ciclo 1
N_MEASURES_CYCLE2 = 6                # ciclo 2: as 5 de cima + EAR mínimo (índice 5)


def n_measures(cycle: int) -> int:
    return N_MEASURES if cycle == 1 else N_MEASURES_CYCLE2      # ciclos 2 e 3 têm o EAR mínimo


def channels(cycle: int) -> list[str]:
    return CHANNELS if cycle == 1 else CHANNELS_CYCLE2

CALIBRATION_REALISTIC = "realistic"    # referência = 1º minuto do próprio vídeo
CALIBRATION_OPTIMISTIC = "optimistic"  # referência = vídeo alerta (0) da pessoa

# ----------------------------------------------------------------------
# Eventos por janela (a 10 Hz, 1 passo = 0,1 s)
# ----------------------------------------------------------------------
LONG_CLOSURE_MIN_STEPS = round(P.BLINK_MAX_S / STEP_S)   # fechamento longo: >= 0,5 s
BLINK_MAX_STEPS = LONG_CLOSURE_MIN_STEPS - 1             # piscada: de 1 passo até < 0,5 s
YAWN_MIN_STEPS = round(P.YAWN_MIN_S / STEP_S)
RULE_PERCLOS_PCT = 20.0              # regra fixa: PERCLOS da janela acima disto = sonolento

# ----------------------------------------------------------------------
# Modelos
# ----------------------------------------------------------------------
N_FOLDS = 5
RF_TREES = 300
RF_MIN_LEAF = 3
LSTM_HIDDEN = 64
LSTM_DROPOUT = 0.3
LSTM_LR = 1e-3
LSTM_BATCH = 32
LSTM_MAX_EPOCHS = 30
LSTM_PATIENCE = 6
LSTM_GRAD_CLIP = 1.0
LSTM_MAX_TRAIN_S = 900               # teto de tempo de treino por modelo (s)
INNER_VAL_FRACTION = 0.2             # fração das PESSOAS de treino usada na parada antecipada
TORCH_THREADS = 8

SUSPICIOUS_SCORE = 0.90              # resultado acima disto dispara a checagem de vazamento

N_SEEDS = {1: 1, 2: 5, 3: 5, 4: 5, 5: 5, 6: 5}               # sementes da rede por fold
RNN_KINDS = {1: ["lstm"], 2: ["lstm", "gru"], 3: ["lstm", "gru"], 4: ["lstm"], 5: ["lstm", "gru"], 6: ["lstm"]}
BASE_MODELS = ["regra_fixa", "regra_aprendida", "random_forest"]


def base_models(cycle: int) -> list[str]:
    """Modelos sem rede de cada ciclo (a parte 1 só compara a regra com limiar aprendido e a LSTM)."""
    return ["regra_aprendida"] if cycle in PART1_CYCLES else BASE_MODELS

EXPERIMENTS = {
    "A": {"labels": [0, 10], "calibration": CALIBRATION_REALISTIC, "exclude": [],
          "desc": "alerta (0) contra sonolento (10), calibração realista"},
    "B": {"labels": [0, 10], "calibration": CALIBRATION_OPTIMISTIC, "exclude": [],
          "desc": "alerta contra sonolento, calibração otimista (referência = vídeo 0)"},
    "C": {"labels": [0, 5, 10], "calibration": CALIBRATION_REALISTIC, "exclude": [],
          "desc": "três classes (0, 5, 10), calibração realista"},
    "D": {"labels": [0, 10], "calibration": CALIBRATION_REALISTIC, "exclude": ["44", "45"],
          "desc": "como A sem os participantes 44 e 45 (vídeos de qualidade muito baixa)"},
}

# Ciclo 2: só A e B (binários); 44 e 45 (vídeos de qualidade muito baixa) e 26 (vídeos 5 e 10 quase
# todo descartados pela regra de |roll| > 45°) ficam fora de treino e teste.
CYCLE2_EXCLUDE = ["26", "44", "45"]
EXPERIMENTS_CYCLE2 = {k: {**EXPERIMENTS[k], "exclude": CYCLE2_EXCLUDE} for k in ("A", "B")}


# Ciclo 3 = ciclo 2 (mesma configuração) com o 26 e o 45 de volta; só o 44 fica fora. A rotação do
# vídeo não era a causa do 26/5 (o rosto sai em pé), então o 26 volta com o que a regra de frame deixa.
CYCLE3_EXCLUDE = ["44"]
EXPERIMENTS_CYCLE3 = {k: {**EXPERIMENTS[k], "exclude": CYCLE3_EXCLUDE} for k in ("A", "B")}


# Parte 1: só o experimento A (0 contra 10, calibração realista), sem o 44, com 1, 5 ou 60 s de calibração
EXPERIMENTS_PART1 = {"A": {**EXPERIMENTS["A"], "exclude": CYCLE3_EXCLUDE}}


def experiments(cycle: int) -> dict:
    if cycle in PART1_CYCLES:
        return EXPERIMENTS_PART1
    return {1: EXPERIMENTS, 2: EXPERIMENTS_CYCLE2, 3: EXPERIMENTS_CYCLE3}[cycle]
