"""Parâmetros do projeto: único lugar com índices, limiares, fatores, durações e layout dos demos.

Mudar um valor aqui muda o comportamento de todo o projeto; nada fica solto nos scripts.
"""
from pathlib import Path

import numpy as np

# ======================================================================
# Caminhos
# ======================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"                      # ignorado pelo git (vídeos de rostos)
UTA_RLDD_DIR = DATA_DIR / "uta-rldd"
FEATURES_DIR = DATA_DIR / "features"
FEATURES_TEST_DIR = DATA_DIR / "features_test"        # saída de testes parciais (--max-seconds)
SANITY_DIR = DATA_DIR / "sanity_check"
SESSIONS_DIR = PROJECT_ROOT / "collection" / "sessions"
TEST_PHOTOS_DIR = PROJECT_ROOT / "demos" / "test_photos"

# ======================================================================
# MediaPipe Face Mesh
# ======================================================================
MAX_NUM_FACES = 1
REFINE_LANDMARKS = True              # necessário para mapear bem pálpebras e íris
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# ======================================================================
# Índices dos landmarks (documentação oficial do MediaPipe)
# ======================================================================
# Ordem geométrica [P1..P6] exigida pelas fórmulas de EAR e MAR.
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
MOUTH_INDICES = [61, 291, 37, 84, 267, 314]
# Ordem: nariz, queixo, olho esq, olho dir, boca esq, boca dir (casa com FACE_3D_MODEL)
HEAD_POSE_INDICES = [1, 152, 33, 263, 61, 291]
CHIN_INDEX = 152

# ======================================================================
# Pose da cabeça (solvePnP)
# ======================================================================
# Modelo 3D genérico em convenção OpenCV (Y positivo = para baixo), nariz na origem.
FACE_3D_MODEL = np.array([
    [0.0, 0.0, 0.0],            # ponta do nariz
    [0.0, 330.0, -65.0],        # queixo
    [-225.0, -170.0, -135.0],   # canto do olho esquerdo
    [225.0, -170.0, -135.0],    # canto do olho direito
    [-150.0, 150.0, -125.0],    # canto esquerdo da boca
    [150.0, 150.0, -125.0],     # canto direito da boca
], dtype=np.float64)
FOCAL_LENGTH_FACTOR = 1             # distância focal aproximada = fator * largura da imagem
# Com Y para baixo o modelo equivale a girar 180° em X: cabeça em pé sai com roll ≈ ±180°.
# Subtrair o período/2 e embrulhar em [-180, 180) faz roll = 0 significar cabeça reta.
ROLL_PERIOD_DEG = 360.0
POSE_FRONT_MIN_TZ = 500.0                             # tz acima disto = solução "de frente" (as boas ficam em ~-2200, as invertidas em ~+2300);
                                                      # tz entre 0 e 500 é fit que não convergiu (erro de ~1200 px), não se mexe
POSE_FRONT_SOLUTION_FIX = np.diag([-1.0, -1.0, 1.0])   # Rz(180°): converte a solução com tz > 0 na usual (tz < 0)
ROLL_OFFSET_DEG = 180.0

# ======================================================================
# Calibração e alertas dos demos de webcam
# ======================================================================
CALIBRATION_FRAMES = 90              # ~3 s a 30 fps com o rosto em posição neutra
EAR_DEFAULT_THRESHOLD = 0.20         # limiar fixo do demo de fotos (que não tem fase de calibração)
EAR_CLOSING_FACTOR = 0.75            # EAR < 75% do EAR de olho aberto = olho fechando
EAR_CLOSED_FRAMES = 15               # frames seguidos abaixo do limiar para alertar (~0,5 s a 30 fps)
MAR_YAWN_FACTOR = 1.6                # MAR > 1,6 x MAR de repouso = boca aberta como em bocejo
YAWN_FRAMES = 30                     # frames seguidos acima do limiar para alertar (~1 s a 30 fps)
CRITICAL_EYE_CLOSED_MIN_S = 1.0       # Crítico: olho fechado por >= 1 s ...
CRITICAL_EYE_LOOKBACK_S = 3.0         # ... em algum momento nos 3 s ANTES do início da queda de cabeça, ou durante ela
ROLL_TILT_LIMIT_DEG = 25.0            # queda lateral: |roll - roll da referência| acima disto por 2 s COM olho fechado
PITCH_DROP_OFFSET_DEG = -18.0        # queda de cabeça = pitch neutro menos 18°
CHIN_DROP_OFFSET_PX = 45.0           # ou queixo 45 px abaixo da posição neutra
HEAD_DROP_FRAMES = 50                # frames seguidos de queda para alertar

# ======================================================================
# Coleta própria (webcam)
# ======================================================================
BLOCK_DURATION_S = 3.0
DEFAULT_PERSON = "eu"
DEFAULT_SCENARIO = "padrao"
LABEL_ALERT = 0
LABEL_FATIGUE = 1
BEEP_BLOCK_SAVED = (800, 200)        # (frequência Hz, duração ms)
BEEP_ALERT_MODE = (1200, 150)
BEEP_FATIGUE_MODE = (500, 300)
SESSION_COLUMNS = ["session_id", "person_id", "scenario", "block_id", "frame", "t",
                   "ear", "mar", "yaw", "pitch", "label"]

# ======================================================================
# Interface dos demos: teclas, cores BGR e estilo dos textos
# ======================================================================
WAIT_KEY_MS = 5
KEY_MASK = 0xFF
KEY_ESC = 27
KEY_QUIT = ord("q")
KEY_ALERT_MODE = ord("0")
KEY_FATIGUE_MODE = ord("1")
KEY_PAUSE = ord("s")

GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)


def _style(origin: tuple, scale: float, color: tuple, thickness: int) -> dict:
    return {"origin": origin, "scale": scale, "color": color, "thickness": thickness}


FPS_STYLE = _style((20, 70), 3, GREEN, 3)
EAR_CALIBRATING_STYLE = _style((20, 50), 1.3, YELLOW, 2)
EAR_BOTH_EYES_STYLE = _style((20, 120), 2, RED, 2)
EAR_MEAN_STYLE = _style((20, 160), 2, BLUE, 2)
EAR_ALERT_STYLE = _style((20, 210), 2, RED, 2)
MAR_CALIBRATING_STYLE = _style((20, 50), 1.5, YELLOW, 2)
MAR_READOUT_STYLE = _style((20, 50), 2, CYAN, 2)
YAWN_ALERT_STYLE = _style((20, 100), 2, RED, 3)
HEAD_CALIBRATING_STYLE = _style((20, 50), 1.3, YELLOW, 2)
HEAD_PITCH_STYLE = _style((20, 50), 2, GREEN, 2)
HEAD_YAW_STYLE = _style((20, 90), 2, CYAN, 2)
HEAD_ALERT_STYLE = _style((20, 170), 2, RED, 3)
COLLECTION_BLOCK_STYLE = _style((20, 150), 1.8, GREEN, 2)     # a cor muda com o rótulo
COLLECTION_STANDBY_STYLE = _style((20, 150), 1.8, CYAN, 2)
PHOTO_BOTH_EYES_STYLE = _style((30, 60), 2, RED, 2)
PHOTO_MEAN_STYLE = _style((30, 100), 2, BLUE, 2)
PHOTO_STATUS_STYLE = _style((30, 150), 2, GREEN, 2)           # a cor muda com o estado

EAR_POINT_RADIUS = 2
PHOTO_POINT_RADIUS = 3
MAR_POINT_RADIUS = 3
HEAD_POINT_RADIUS = 4
PHOTO_HEIGHT_PX = 720                # as fotos de teste são redimensionadas para esta altura

# ======================================================================
# Pipeline UTA-RLDD: extração
# ======================================================================
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
FEATURE_COLUMNS = ["ear_left", "ear_right", "ear_mean", "mar", "pitch", "yaw", "roll"]
FEATURES_CSV_COLUMNS = ["participant", "label", "fold", "timestamp_s", "frame_idx",
                        "face_detected", *FEATURE_COLUMNS]
EXTRACTION_LOG_COLUMNS = ["video", "fold", "participant", "label", "width", "height",
                          "rotation_deg", "nominal_fps", "frames", "duration_s", "pct_no_face",
                          "ts_non_monotonic", "processing_s", "processing_fps"]
FACE_FOUND = 1
FACE_NOT_FOUND = 0
EXTRACTION_LOG_NAME = "extraction_log.csv"
MS_PER_SECOND = 1000.0

# Frame de depuração (landmarks desenhados) salvo pela extração
DEBUG_FRAME_AT_S = 30.0
DEBUG_MAX_SIDE_PX = 1280
DEBUG_POINT_MIN_RADIUS = 2
DEBUG_POINT_WIDTH_DIVISOR = 300      # raio do ponto = largura / 300
DEBUG_TEXT_REFERENCE_WIDTH = 900     # o texto cresce a partir desta largura
DEBUG_TEXT_X = 10
DEBUG_TEXT_LINE_HEIGHT = 30
DEBUG_TEXT_SCALE = 0.7
DEBUG_TEXT_THICKNESS = 2

# Folha de contato do inspect_frames
TILE_SIZE = 440
TILE_GRID_COLUMNS = 4
TILE_CROP_MARGIN = 1.5               # recorte = rosto x 1,5
TILE_POINT_WIDTH_DIVISOR = 400
TILE_NO_FACE_CROP_DIVISOR = 4        # sem rosto: mostra a metade central da imagem
TILE_TEXT_X = 6
TILE_TEXT_BASELINE = 20
TILE_TEXT_LINE_HEIGHT = 22
TILE_TEXT_PADDING = 6
TILE_TEXT_SCALE = 0.5
EAR_REPRODUCTION_TOLERANCE = 1e-4    # reprocessar tem que reproduzir o EAR do CSV

# ======================================================================
# Pipeline UTA-RLDD: análise (limiares relativos ao próprio vídeo)
# ======================================================================
# Frame inválido: vira NaN SÓ na análise (os CSVs brutos nunca são alterados).
# Com a cabeça quase de perfil os landmarks do olho caem fora do olho e o solvePnP pode
# devolver a solução espelhada.
VALID_YAW_MAX_DEG = 45.0             # |yaw| acima disso: perfil
VALID_ROLL_MAX_DEG = 45.0            # |roll| acima disso: solvePnP espelhado
VALID_EAR_EYE_MAX = 0.45             # EAR de UM olho acima disso é implausível (p99,99 dos
                                     # frames "limpos" é 0,441)

OPEN_EYE_PERCENTILE = 75             # "olho aberto" do vídeo; p90 era inflado por picos espúrios
MAR_REST_PERCENTILE = 50             # MAR de repouso = mediana (boca fechada a maior parte do tempo)
EAR_CLOSED_P80_FACTOR = 0.20         # PERCLOS clássico P80: olho >= 80% fechado
BLINK_MIN_S = 0.05                   # abaixo disso é ruído de 1 frame
BLINK_MAX_S = 0.50                   # >= 0,5 s deixa de ser piscada: fechamento longo
BRIDGE_MAX_GAP_S = 0.3               # buraco de frames inválidos até isso, entre dois frames do evento, não o interrompe
HEAD_NOD_MIN_S = 1.7                 # cabeça baixa por >= 1,7 s = cabeceio (equivale a 50 frames a 30 fps do demo)
PITCH_SMOOTH_S = 0.3                 # mediana móvel do pitch antes de detectar cabeceio e medir dispersão
HEAD_NEUTRAL_PERCENTILE = 50         # pitch neutro = mediana do PRÓPRIO vídeo (a câmera muda entre gravações)
PITCH_LOW_PERCENTILE = 5
YAWN_MIN_S = 1.0                     # boca acima do limiar por >= 1 s
JUMP_DEG = 10.0                      # salto de pitch/yaw entre frames consecutivos
MAR_PERCENTILES = (5, 25, 50, 75, 90, 95, 99)
SECONDS_PER_MINUTE = 60.0
PERCENT = 100

LABEL_NAMES = {0: "0 alerta", 5: "5 baixa vigilância", 10: "10 sonolento"}
ALERT_LABEL = 0
DROWSY_LABEL = 10
LOW_VIGILANCE_LABEL = 5
FPS_MISMATCH_RATIO = 0.8             # fps do vídeo 0 e do 10 do mesmo participante: menor/maior abaixo disso = marcado

# Gráficos
LABEL_COLORS = {0: "#8fb4d9", 5: "#4f86c0", 10: "#1f4e8c"}   # rampa de uma cor: escuro = sonolento
INK_COLOR = "#2b2b2b"
GRID_COLOR = "#e3e3e3"
YAWN_MARK_COLOR = "#c0392b"
MAR_FIXED_REFERENCE = 0.50           # limiar fixo antigo, desenhado só como referência
MAR_FIXED_REFERENCE_COLOR = "#999"
MAR_FIXED_LABEL_COLOR = "#777"
TITLE_FONT_SIZE = 10
LABEL_FONT_SIZE = 8
SUPTITLE_FONT_SIZE = 11
NOTE_FONT_SIZE = 7
GRID_LINE_WIDTH = 0.8
COMPARISON_FIGSIZE = (13, 7)
COMPARISON_DPI = 140
BAR_WIDTH = 0.26
BAR_FILL = 0.92
TIMESERIES_WIDTH = 13
TIMESERIES_ROW_HEIGHT = 2.3
TIMESERIES_DPI = 130
TIMESERIES_SMOOTH_FRAMES = 15        # mediana móvel só para visualizar (~0,5 s)
TIMESERIES_LINE_WIDTH = 0.6
MAR_PLOT_WIDTH = 10
MAR_PLOT_ROW_HEIGHT = 0.38
MAR_PLOT_PADDING = 1.5
MAR_PLOT_DPI = 140
MAR_PLOT_THIN_LINE = 2
MAR_PLOT_THICK_LINE = 7
MAR_PLOT_MEDIAN_SIZE = 4
MAR_PLOT_THRESHOLD_SIZE = 11

# ======================================================================
# Pré-processamento compartilhado (treino e demonstração ao vivo): detection/preprocessing.py
# ======================================================================
RESAMPLE_HZ = 10
STEP_S = 1.0 / RESAMPLE_HZ
MAX_INTERP_GAP_S = 0.3               # buraco de até 0,3 s é interpolado; maior fica como falta
MAX_INTERP_GAP_STEPS = round(MAX_INTERP_GAP_S / STEP_S)
WINDOW_S = 60                        # janela da rede
WINDOW_STEPS = WINDOW_S * RESAMPLE_HZ
MIN_OBSERVED_FRACTION = 0.5          # janela com menos passos observados que isso não vale
N_MEASURES_BASE = 5                  # EAR, MAR, pitch, yaw, roll (médias do intervalo de 0,1 s)
MEASURE_NAMES = ["ear_mean", "mar", "pitch", "yaw", "roll"]
RELATIVE_CHANNELS = ["ear_rel", "mar_rel", "pitch_rel", "yaw_rel", "roll_rel", "ear_min_rel", "missing"]
EAR_MIN_CHANNEL = N_MEASURES_BASE    # índice do canal de EAR mínimo (a partir do ciclo 2)
MODEL_NORM_CLIP = 6.0                # normalização da rede limitada a ±6 desvios

# ======================================================================
# Demonstração ao vivo (demos/main_demo.py) e detectores de evento (detection/events.py)
# ======================================================================
LIVE_CALIBRATION_S = 10              # calibração no início (e ao apertar r)
LIVE_MIN_CALIBRATION_FRAMES = 30     # menos frames válidos que isso: a calibração recomeça
LIVE_UPDATE_S = 1.0                  # a rede roda uma vez por segundo
KEY_RECALIBRATE = ord("r")

EYE_CLOSED_ALERT_S = 2.0             # olho fechado (EAR < 75% da referência) por >= 2 s
YAWN_MIN_S_LIVE = 1.0                # MAR > 1,6 x repouso por >= 1 s = bocejo
YAWN_COUNT_WINDOW_MIN = 15           # bocejos contados nos últimos 15 min (padrão)
FAIR_YAWN_COUNT_WINDOW_MIN = 2       # modo feira (--feira): memória curta, para uma pessoa depois da outra
HEAD_DROP_MIN_S = 2.0                # pitch 18° abaixo da referência por >= 2 s
FACE_LOST_MIN_S = 1.5                # Crítico por rosto perdido: rosto ausente por pelo menos 1,5 s ...
FACE_LOST_EYE_CLOSED_MIN_S = 0.5     # ... E olho fechado por >= 0,5 s (tolerância BRIDGE_MAX_GAP_S) ...
FACE_LOST_EYE_LOOKBACK_S = 3.0       # ... nos 3 s antes de o rosto sumir (ou queda de cabeça já iniciada com olho fechado)
EVENT_BRIDGE_S = 0.3                 # frame inválido/sem rosto de até 0,3 s não interrompe o evento

# Níveis: Normal < Atenção < Alerta < Crítico
LEVEL_NAMES = ["NORMAL", "ATENCAO", "ALERTA", "CRITICO"]    # sem acento: o cv2.putText não desenha
LEVEL_ATTENTION_YAWNS = 3            # Atenção: >= 3 bocejos nos últimos 15 min
LEVEL_ATTENTION_PROB = 0.50          # ou rede acima de 50% (por LEVEL_ATTENTION_PERSIST_S; sai abaixo de ..._EXIT)
LEVEL_ATTENTION_PROB_EXIT = 0.40
LEVEL_ATTENTION_PERSIST_S = 5.0      # Atenção: valores não ajustados (só o Alerta foi escolhido nos dados)
# Alerta pela rede: acima de ENTER por PERSIST_S segundos seguidos; só sai abaixo de EXIT (histerese).
# Escolhidos com as previsões fora do fold de 51 pessoas (sem o 44 e sem os 6 do teste do demo), em
# training/persistencia.py: vídeos alerta <= 5% do tempo em Alerta, maximizando o tempo dos sonolentos.
# Antes (sem persistência): 70% / 60% / 0 s. Olho fechado >= 2 s continua disparando Alerta na hora.
LEVEL_ALERT_PROB = 0.90
LEVEL_ALERT_PROB_EXIT = 0.70
LEVEL_ALERT_PERSIST_S = 10.0
LEVEL_HOLD_S = 2.0                   # o nível só desce depois de ficar assim por 2 s (evita piscar)
LEVEL_COLORS = [(80, 160, 60), (0, 200, 255), (0, 120, 255), (40, 40, 220)]    # BGR
PERCLOS_WINDOW_S = 60                # regra simples ao lado da rede
EAR_PLOT_S = 30                      # gráfico ao vivo da abertura do olho
SOUND_LEVEL_ALERT_REPEAT_S = 3.0      # Alerta por olho fechado ou queda de cabeça: bipe a cada 3 s
SOUND_NET_ALERT_REPEAT_S = 10.0       # Alerta só pela rede: um bipe ao entrar e depois a cada 10 s enquanto continuar
SOUND_LEVEL_CRITICAL_REPEAT_S = 1.5
SOUND_ALERT_HZ = 880
SOUND_CRITICAL_HZ = 1320
SOUND_SAMPLE_RATE = 22050
SOUND_BEEP_S = 0.25
LIVE_DETECT_HEIGHT_PX = 480          # o quadro é reduzido a esta altura antes do Face Mesh (velocidade)
LIVE_WINDOW_TITLE = "Deteccao de sonolencia (demo ao vivo)"

# ======================================================================
# Gravação para validar bocejo e queda de cabeça (collection/validation_recording.py)
# ======================================================================
OWN_DIR = DATA_DIR / "own"                            # gravações próprias (fora do git, como todo data/)
VALIDATION_KEYS = {ord("b"): "b", ord("h"): "h", ord("c"): "c", ord("p"): "p", ord("n"): "n"}
VALIDATION_EVENT_NAMES = {"b": "bocejo", "h": "queda de cabeca", "c": "olhar o celular",
                          "p": "olhar painel/retrovisor", "n": "normal"}
VALIDATION_FRAME_COLUMNS = ["t", "face_detected", *FEATURE_COLUMNS, "marcas"]
VALIDATION_VIDEO_HEIGHT_PX = 480
VALIDATION_VIDEO_FPS = 30.0
VALIDATION_FLASH_S = 2.0              # a última marca fica em destaque na tela por 2 s
VALIDATION_MATCH_TOLERANCE_S = 2.0   # detecção casa com o evento marcado se cai até 2 s dele
