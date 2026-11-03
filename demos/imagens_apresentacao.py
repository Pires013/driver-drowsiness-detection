"""Quatro imagens de apoio para a apresentação, com frames do UTA-RLDD (participante 05; a cabeça
inclinada usa o 35, que abaixa a cabeça mais que o 05).

Rodar da raiz: python -m demos.imagens_apresentacao   (grava em data/apresentacao/, fora do git)
  a) olho_aberto_fechado.png   b) boca_repouso_bocejo.png   c) cabeca_reta_inclinada.png
  d) abertura_do_olho_participante05.png (EAR ao longo do vídeo alerta e do sonolento)
Os frames são escolhidos pelos valores do CSV de features (sem ajuste manual).
"""
import cv2
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from detection import parameters as P
from detection.drawing import draw_points
from detection.features import compute_features
from detection.landmarks import create_face_mesh, detect_face, to_pixel_points
from detection.validity import is_valid_frame
from pipeline.extract_features import ROTATIONS, open_video

PERSON, HEAD_PERSON = "05", "35"
OUT = P.DATA_DIR / "apresentacao"
FONT, FONT_BOLD = "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"
GREEN, RED = (80, 170, 60), (40, 40, 220)


def video_path(label: int, person: str = PERSON):
    return next(P.UTA_RLDD_DIR.glob(f"Fold*/{person}/{label}.*"))


def features(label: int, person: str = PERSON) -> pd.DataFrame:
    data = pd.read_csv(next(P.FEATURES_DIR.glob(f"Fold*/{person}/{label}.csv")), dtype={"participant": str})
    data["ok"] = is_valid_frame(data["face_detected"].to_numpy(), data["yaw"].to_numpy(),
                                data["roll"].to_numpy(), data["ear_left"].to_numpy(),
                                data["ear_right"].to_numpy())
    return data


def grab(label: int, t: float, mesh, person: str = PERSON):
    """(frame girado, landmarks) do instante t (s)."""
    cap, rotation = open_video(video_path(label, person))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    cap.release()
    assert ok, f"não leu {label}.mp4 em {t} s"
    if rotation:
        frame = cv2.rotate(frame, ROTATIONS[rotation])
    return frame, detect_face(mesh, frame)


def crop(frame, landmarks, indices, margin, size=(640, 420)):
    """Recorte em torno dos pontos escolhidos, na proporção de `size`."""
    height, width = frame.shape[:2]
    pts = np.array(to_pixel_points(landmarks, indices, width, height))
    cx, cy = pts.mean(axis=0)
    half_w = max(np.ptp(pts[:, 0]) * margin, 20)
    half_h = half_w * size[1] / size[0]
    x0, x1 = int(max(cx - half_w, 0)), int(min(cx + half_w, width))
    y0, y1 = int(max(cy - half_h, 0)), int(min(cy + half_h, height))
    return cv2.resize(frame[y0:y1, x0:x1], size, interpolation=cv2.INTER_AREA), (x0, y0, x1 - x0, y1 - y0)


def label(image, title, value, color):
    """Faixa branca com título e valor (PIL, para desenhar acentos que o cv2.putText não tem)."""
    cv2.rectangle(image, (0, 0), (image.shape[1], 100), (255, 255, 255), -1)
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.text((14, 6), title, font=ImageFont.truetype(FONT, 34), fill=(40, 40, 40))
    draw.text((14, 46), value, font=ImageFont.truetype(FONT_BOLD, 50), fill=color[::-1])
    image[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def pair(name, left, right):
    sep = np.full((left.shape[0], 12, 3), 255, np.uint8)
    cv2.imwrite(str(OUT / name), np.hstack([left, sep, right]))


def eyes_image(mesh):
    data = features(0)
    data = data[data.ok & (data.timestamp_s > 60)]
    opened = data.loc[data.ear_mean.sub(data.ear_mean.quantile(0.75)).abs().idxmin()]
    closed = data.loc[data[data.ear_mean > 0.02].ear_mean.idxmin()]         # olho fechado, mas medido
    tiles = []
    for row, title, color in ((opened, "Olho aberto", GREEN), (closed, "Olho fechado", RED)):
        frame, lm = grab(0, row.timestamp_s, mesh)
        view, (x0, y0, w, h) = crop(frame, lm, P.LEFT_EYE_INDICES + P.RIGHT_EYE_INDICES, 1.9)
        draw_points(frame, lm, P.LEFT_EYE_INDICES + P.RIGHT_EYE_INDICES, (0, 0, 255), 3)
        view, _ = crop(frame, lm, P.LEFT_EYE_INDICES + P.RIGHT_EYE_INDICES, 1.9)
        label(view, title, f"EAR = {row.ear_mean:.2f}", color)
        tiles.append(view)
    pair("olho_aberto_fechado.png", *tiles)


def mouth_image(mesh):
    data = features(0)
    data = data[data.ok & (data.timestamp_s > 60)]
    rest = data.loc[data.mar.sub(data.mar.median()).abs().idxmin()]
    yawn = data.loc[data.mar.idxmax()]
    tiles = []
    for row, title, color in ((rest, "Boca em repouso", GREEN), (yawn, "Bocejo", RED)):
        frame, lm = grab(0, row.timestamp_s, mesh)
        draw_points(frame, lm, P.MOUTH_INDICES, (0, 200, 255), 4)
        view, _ = crop(frame, lm, P.MOUTH_INDICES, 1.7)
        label(view, title, f"MAR = {row.mar:.2f}", color)
        tiles.append(view)
    pair("boca_repouso_bocejo.png", *tiles)


def head_image(mesh):
    data = features(10, HEAD_PERSON)         # o 05 quase não abaixa a cabeça; o 35 abaixa (pitch menor = para baixo)
    data = data[data.ok & (data.timestamp_s > 60) & (data.yaw.abs() < 25) & (data.roll.abs() < 25)]
    straight = data.loc[data.pitch.sub(data.pitch.median()).abs().idxmin()]
    down = data.loc[data.pitch.sub(data.pitch.quantile(0.01)).abs().idxmin()]    # 1º percentil: ignora pontos soltos
    ref = straight.pitch
    tiles = []
    for row, title, color in ((straight, "Cabeça reta", GREEN), (down, "Cabeça inclinada para baixo", RED)):
        frame, lm = grab(10, row.timestamp_s, mesh, HEAD_PERSON)
        view, _ = crop(frame, lm, list(range(468)), 1.0, size=(560, 560))
        diff = abs(row.pitch - ref)
        label(view, title, f"{diff:.0f} graus" if row is down else "referência (0 graus)", color)
        tiles.append(view)
    pair("cabeca_reta_inclinada.png", *tiles)
    return straight.pitch, down.pitch


def curve_image():
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=130)
    for label_, name, color in ((0, "Alerta", "#2e7d32"), (10, "Sonolento", "#c62828")):
        d = features(label_)
        d = d[d.ok]
        minutes = d.timestamp_s / 60
        smooth = d.ear_mean.rolling(300, min_periods=30, center=True).median()   # ~10 s
        ax.plot(minutes, smooth, color=color, lw=2.6, label=name)
    ax.set_xlabel("Tempo de vídeo (minutos)", fontsize=16)
    ax.set_ylabel("Abertura do olho (EAR)", fontsize=16)
    ax.set_title("Participante 05: abertura do olho ao longo do vídeo", fontsize=18)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=16, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "abertura_do_olho_participante05.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mesh = create_face_mesh(static_image=True)
    eyes_image(mesh)
    mouth_image(mesh)
    print("pitch reta/inclinada:", head_image(mesh))
    curve_image()
    print("imagens em", OUT)


if __name__ == "__main__":
    main()
