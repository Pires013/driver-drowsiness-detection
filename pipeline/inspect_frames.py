"""
Salva uma folha de contato com frames específicos de um vídeo, com os landmarks desenhados e as
features anotadas, para inspeção visual de artefatos.

O vídeo é reprocessado EM SEQUÊNCIA desde o frame 0 (como na extração), porque o FaceMesh em modo
vídeo usa tracking: o resultado de um frame depende dos anteriores. Cada frame inspecionado é
comparado com o valor gravado no CSV, para provar que o reprocessamento reproduz a extração.

Uso (da raiz do projeto):
    python -m pipeline.inspect_frames --video 04/5.mp4 --frames 6318 347 --out data/inspect/p04_5.png
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from detection import parameters as P
from detection.drawing import draw_contours, draw_feature_points
from detection.features import compute_features
from detection.landmarks import FaceLandmarks, create_face_mesh, detect_face
from pipeline.extract_features import ROTATIONS, open_video, parse_video_path


def crop_around_face(image, face_landmarks: FaceLandmarks):
    """Recorte quadrado em torno do rosto, com margem."""
    height, width = image.shape[:2]
    xs = [lm.x * width for lm in face_landmarks.landmark]
    ys = [lm.y * height for lm in face_landmarks.landmark]
    center_x, center_y = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    side = max(max(xs) - min(xs), max(ys) - min(ys)) * P.TILE_CROP_MARGIN
    x0, y0 = int(max(0, center_x - side / 2)), int(max(0, center_y - side / 2))
    return image[y0:int(min(height, center_y + side / 2)), x0:int(min(width, center_x + side / 2))]


def caption_lines(frame_idx: int, timestamp_s: float, features: dict | None, csv_ear_mean: float) -> list[str]:
    if features is None:
        return [f"f={frame_idx} t={timestamp_s:.2f}s", "SEM ROSTO"]
    return [f"f={frame_idx} t={timestamp_s:.2f}s",
            f"EAR L={features['ear_left']:.2f} R={features['ear_right']:.2f} (csv {csv_ear_mean:.2f})",
            f"MAR={features['mar']:.2f}",
            f"yaw={features['yaw']:.0f} pitch={features['pitch']:.0f} roll={features['roll']:.0f}"]


def draw_tile(frame, face_landmarks: FaceLandmarks | None, lines: list[str]):
    """Recorte do rosto com landmarks (ou o centro do frame, sem rosto) e a legenda embaixo."""
    image = frame.copy()
    if face_landmarks is None:
        height, width = image.shape[:2]
        d = P.TILE_NO_FACE_CROP_DIVISOR
        crop = image[height // d: (d - 1) * height // d, width // d: (d - 1) * width // d]
    else:
        draw_contours(image, face_landmarks)
        draw_feature_points(image, face_landmarks,
                            max(P.DEBUG_POINT_MIN_RADIUS, image.shape[1] // P.TILE_POINT_WIDTH_DIVISOR))
        crop = crop_around_face(image, face_landmarks)
    tile = cv2.resize(crop, (P.TILE_SIZE, P.TILE_SIZE), interpolation=cv2.INTER_AREA)

    caption = np.zeros((P.TILE_TEXT_LINE_HEIGHT * len(lines) + P.TILE_TEXT_PADDING, P.TILE_SIZE, 3), np.uint8)
    for k, text in enumerate(lines):
        cv2.putText(caption, text, (P.TILE_TEXT_X, P.TILE_TEXT_BASELINE + P.TILE_TEXT_LINE_HEIGHT * k),
                    cv2.FONT_HERSHEY_SIMPLEX, P.TILE_TEXT_SCALE, P.GREEN, 1, cv2.LINE_AA)
    return np.vstack([tile, caption])


def contact_sheet(tiles: list, columns: int):
    """Junta os tiles em grade; completa a última linha com tiles pretos."""
    tiles = list(tiles)
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="PARTICIPANTE/ARQUIVO, ex.: 04/5.mp4")
    ap.add_argument("--frames", type=int, nargs="+", required=True, help="frame_idx (como no CSV)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cols", type=int, default=P.TILE_GRID_COLUMNS)
    ap.add_argument("--input", type=Path, default=P.UTA_RLDD_DIR)
    ap.add_argument("--features", type=Path, default=P.FEATURES_DIR)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    found = list(args.input.glob(f"*/{args.video}"))
    if not found:
        raise SystemExit(f"vídeo não encontrado: {args.video}")
    video = found[0]
    fold, participant, label = parse_video_path(video, args.input)
    csv_data = pd.read_csv(args.features / fold / participant / f"{label}.csv").set_index("frame_idx")
    targets = sorted(set(args.frames))

    cap, rotation = open_video(video)
    face_mesh = create_face_mesh()
    tiles, frame_idx = {}, 0
    while frame_idx <= targets[-1]:
        success, frame = cap.read()
        if not success:
            break
        if rotation:
            frame = cv2.rotate(frame, ROTATIONS[rotation])
        face_landmarks = detect_face(face_mesh, frame)      # roda em TODOS os frames: tracking
        if frame_idx in targets:
            height, width = frame.shape[:2]
            features = compute_features(face_landmarks, width, height) if face_landmarks is not None else None
            csv_ear_mean = csv_data.loc[frame_idx, "ear_mean"]
            if features is not None and abs(features["ear_mean"] - csv_ear_mean) > P.EAR_REPRODUCTION_TOLERANCE:
                print(f"AVISO frame {frame_idx}: EAR reprocessado {features['ear_mean']:.4f} != CSV {csv_ear_mean:.4f}")
            lines = caption_lines(frame_idx, csv_data.loc[frame_idx, "timestamp_s"], features, csv_ear_mean)
            tiles[frame_idx] = draw_tile(frame, face_landmarks, lines)
        frame_idx += 1
    cap.release()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), contact_sheet([tiles[i] for i in targets if i in tiles], args.cols))
    print(f"salvo {args.out} ({len(tiles)} frames)")


if __name__ == "__main__":
    main()
