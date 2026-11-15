"""Training constants (data preparation, models and experiments).

Detection constants stay in detection/parameters.py and are reused here; this file only holds what is training-specific."""
from detection import parameters as P

TRAINING_DIR = P.DATA_DIR / "training"
REPORT_PATH = TRAINING_DIR / "relatorio.md"
REPORT_CYCLE1_PATH = TRAINING_DIR / "relatorio_ciclo1.md"
REPORT_CYCLE3_PATH = TRAINING_DIR / "relatorio_ciclo3.md"


# part 1 (calibration length): cycles 4, 5 and 6 = realistic calibration of 5, 10 and 60 s
PART1_CYCLES = {4: 5, 5: 10, 6: 60}
CYCLE_DIR_NAMES = {4: "parte1_cal5", 5: "parte1_cal10", 6: "parte1_cal60"}


def cycle_dir(cycle: int):
    if cycle == 1:
        return TRAINING_DIR
    return TRAINING_DIR / CYCLE_DIR_NAMES.get(cycle, f"ciclo{cycle}")


def calibration_s(cycle: int) -> int:
    return PART1_CYCLES.get(cycle, CALIBRATION_S)


def results_dir(cycle: int):
    return cycle_dir(cycle) / "results"


def models_dir(cycle: int):
    return cycle_dir(cycle) / "models"

SEED = 42

RESAMPLE_HZ = P.RESAMPLE_HZ
STEP_S = P.STEP_S
MAX_INTERP_GAP_S = P.MAX_INTERP_GAP_S
MAX_INTERP_GAP_STEPS = P.MAX_INTERP_GAP_STEPS
WINDOW_S = P.WINDOW_S
WINDOW_STRIDE_S = 30
WINDOW_STEPS = WINDOW_S * RESAMPLE_HZ
WINDOW_STRIDE_STEPS = WINDOW_STRIDE_S * RESAMPLE_HZ
CALIBRATION_S = 60  # first minute of the video: calibration (kept out of the windows)
CALIBRATION_STEPS = CALIBRATION_S * RESAMPLE_HZ
MIN_OBSERVED_FRACTION = P.MIN_OBSERVED_FRACTION

# window tensor channels: measures relative to the reference + a mask (1 = missing)
# cycle 2 adds the per-step minimum EAR (the mean alone erases blinks)
CHANNELS = ["ear_rel", "mar_rel", "pitch_rel", "yaw_rel", "roll_rel", "missing"]
CHANNELS_CYCLE2 = ["ear_rel", "mar_rel", "pitch_rel", "yaw_rel", "roll_rel", "ear_min_rel", "missing"]
N_MEASURES = 5
N_MEASURES_CYCLE2 = 6


def n_measures(cycle: int) -> int:
    return N_MEASURES if cycle == 1 else N_MEASURES_CYCLE2


def channels(cycle: int) -> list[str]:
    return CHANNELS if cycle == 1 else CHANNELS_CYCLE2

CALIBRATION_REALISTIC = "realistic"  # reference = first minute of the video itself
CALIBRATION_OPTIMISTIC = "optimistic"  # reference = the person's alert (0) video

LONG_CLOSURE_MIN_STEPS = round(P.BLINK_MAX_S / STEP_S)
BLINK_MAX_STEPS = LONG_CLOSURE_MIN_STEPS - 1
YAWN_MIN_STEPS = round(P.YAWN_MIN_S / STEP_S)
RULE_PERCLOS_PCT = 20.0  # fixed rule: window PERCLOS above this = drowsy

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
LSTM_MAX_TRAIN_S = 900
INNER_VAL_FRACTION = 0.2  # fraction of training PEOPLE used for early stopping
TORCH_THREADS = 8

SUSPICIOUS_SCORE = 0.90  # a result above this triggers the leakage check

N_SEEDS = {1: 1, 2: 5, 3: 5, 4: 5, 5: 5, 6: 5}
RNN_KINDS = {1: ["lstm"], 2: ["lstm", "gru"], 3: ["lstm", "gru"], 4: ["lstm"], 5: ["lstm", "gru"], 6: ["lstm"]}
BASE_MODELS = ["regra_fixa", "regra_aprendida", "random_forest"]


def base_models(cycle: int) -> list[str]:
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

# cycle 2: 44 and 45 (very low quality videos) and 26 (videos 5 and 10 mostly discarded by the 
# roll
#  > 45 rule) are out of training and test
CYCLE2_EXCLUDE = ["26", "44", "45"]
EXPERIMENTS_CYCLE2 = {k: {**EXPERIMENTS[k], "exclude": CYCLE2_EXCLUDE} for k in ("A", "B")}


# cycle 3 = cycle 2 with 26 and 45 back; only 44 stays out. Video rotation was not the cause of 26/5, so 26 returns with what the frame rule keeps
CYCLE3_EXCLUDE = ["44"]
EXPERIMENTS_CYCLE3 = {k: {**EXPERIMENTS[k], "exclude": CYCLE3_EXCLUDE} for k in ("A", "B")}


EXPERIMENTS_PART1 = {"A": {**EXPERIMENTS["A"], "exclude": CYCLE3_EXCLUDE}}


def experiments(cycle: int) -> dict:
    if cycle in PART1_CYCLES:
        return EXPERIMENTS_PART1
    return {1: EXPERIMENTS, 2: EXPERIMENTS_CYCLE2, 3: EXPERIMENTS_CYCLE3}[cycle]
