"""
Extrai EAR, MAR e pose da cabeça de cada vídeo do UTA-RLDD e salva um CSV por vídeo
(uma linha por frame). As fórmulas vêm do pacote detection, as mesmas da coleta própria.

Uso (da raiz do projeto):
    python -m pipeline.extract_features                       # tudo em data/uta-rldd
    python -m pipeline.extract_features --max-seconds 60 --only 01/10.MOV 04/0.mp4 \
        --debug-frame-dir data/features_test                  # teste rápido

Estrutura esperada: <input>/<FoldN_partM>/<participante>/<rótulo>.<ext>
O rótulo (0, 5, 10) vem do NOME DO ARQUIVO; não existe outro metadado.
"""
import argparse
import csv
import math
import os
import re
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
from tqdm import tqdm

from detection import parameters as P
from detection.drawing import draw_contours, draw_feature_points, draw_tesselation
from detection.features import compute_features
from detection.landmarks import FaceLandmarks, create_face_mesh, detect_face

ROTATIONS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def parse_video_path(video: Path, input_dir: Path) -> tuple[str, str, int]:
    parts = video.relative_to(input_dir).parts
    if len(parts) != 3:
        raise ValueError(f"Estrutura inesperada (esperava fold/participante/arquivo): {video}")
    fold_folder, participant, filename = parts
    match = re.match(r"(Fold\d+)", fold_folder, re.IGNORECASE)
    fold = match.group(1) if match else fold_folder
    stem = Path(filename).stem
    if not stem.isdigit():
        raise ValueError(f"Nome de arquivo sem rótulo numérico: {video}")
    return fold, participant, int(stem)


def open_video(path: Path) -> tuple[cv2.VideoCapture, int]:
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(str(path))
    # OpenCV auto-rotation is off for determinism (it did not apply it on these videos); the metadata rotation is applied manually
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    degrees = int(cap.get(cv2.CAP_PROP_ORIENTATION_META)) % 360
    return cap, degrees


def iter_frames(cap: cv2.VideoCapture, rotation_deg: int, max_seconds: float | None) -> Iterator:
    while True:
        success, frame = cap.read()
        if not success:
            return
        # REAL container timestamp (not frame_idx / fps): fps varies between videos
        timestamp_s = cap.get(cv2.CAP_PROP_POS_MSEC) / P.MS_PER_SECOND
        if max_seconds and timestamp_s >= max_seconds:
            return
        if rotation_deg:
            frame = cv2.rotate(frame, ROTATIONS[rotation_deg])
        yield timestamp_s, frame


def estimate_total_frames(cap: cv2.VideoCapture, nominal_fps: float, max_seconds: float | None) -> int:
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_seconds:
        total = min(total, int(max_seconds * nominal_fps) + 1)
    return total


def csv_row(participant: str, label: int, fold: str, timestamp_s: float, frame_idx: int,
            features: dict | None) -> list:
    if features is None:
        face_detected, values = P.FACE_NOT_FOUND, [math.nan] * len(P.FEATURE_COLUMNS)
    else:
        face_detected, values = P.FACE_FOUND, [features[column] for column in P.FEATURE_COLUMNS]
    return [participant, label, fold, f"{timestamp_s:.4f}", frame_idx, face_detected,
            *(f"{value:.6f}" for value in values)]


def save_debug_frame(frame, face_landmarks: FaceLandmarks, features: dict,
                     timestamp_s: float, destination: Path) -> None:
    image = frame.copy()
    draw_tesselation(image, face_landmarks)
    draw_contours(image, face_landmarks)
    height, width = image.shape[:2]
    draw_feature_points(image, face_landmarks,
                        max(P.DEBUG_POINT_MIN_RADIUS, width // P.DEBUG_POINT_WIDTH_DIVISOR))

    scale = max(1.0, width / P.DEBUG_TEXT_REFERENCE_WIDTH)
    lines = [f"t={timestamp_s:.1f}s  EAR L={features['ear_left']:.2f} R={features['ear_right']:.2f} "
             f"mean={features['ear_mean']:.2f}",
             f"MAR={features['mar']:.2f}  pitch={features['pitch']:.0f} yaw={features['yaw']:.0f} "
             f"roll={features['roll']:.0f}"]
    for k, text in enumerate(lines):
        origin = (P.DEBUG_TEXT_X, int(P.DEBUG_TEXT_LINE_HEIGHT * scale * (k + 1)))
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, P.DEBUG_TEXT_SCALE * scale,
                    P.GREEN, max(1, int(P.DEBUG_TEXT_THICKNESS * scale)))

    shrink = P.DEBUG_MAX_SIDE_PX / max(height, width)
    if shrink < 1:
        image = cv2.resize(image, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(destination), image)


def process_video(video: Path, input_dir: Path, csv_path: Path, face_mesh,
                  max_seconds: float | None, debug_dir: Path | None, debug_at_s: float) -> dict:
    fold, participant, label = parse_video_path(video, input_dir)
    cap, rotation = open_video(video)
    if not cap.isOpened():
        raise RuntimeError(f"Não consegui abrir {video}")
    if rotation not in (0, *ROTATIONS):
        raise RuntimeError(f"Rotação inesperada ({rotation}°) em {video}")
    nominal_fps = cap.get(cv2.CAP_PROP_FPS)

    tmp_path = csv_path.with_suffix(".csv.tmp")
    frame_count = no_face_count = non_monotonic_count = 0
    previous_ts, last_ts = -1.0, 0.0
    width = height = 0
    debug_saved = debug_dir is None
    started = time.perf_counter()

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "w", newline="") as f, \
            tqdm(total=estimate_total_frames(cap, nominal_fps, max_seconds),
                 desc=f"{participant}/{label}", unit="fr", leave=False) as progress:
        writer = csv.writer(f)
        writer.writerow(P.FEATURES_CSV_COLUMNS)
        for timestamp_s, frame in iter_frames(cap, rotation, max_seconds):
            height, width = frame.shape[:2]
            face_landmarks = detect_face(face_mesh, frame)
            features = compute_features(face_landmarks, width, height) if face_landmarks is not None else None

            if features is None:
                no_face_count += 1
            elif not debug_saved and timestamp_s >= debug_at_s:
                save_debug_frame(frame, face_landmarks, features, timestamp_s,
                                 debug_dir / f"{participant}_{label}_landmarks.png")
                debug_saved = True

            if timestamp_s < previous_ts:
                non_monotonic_count += 1
            previous_ts = last_ts = timestamp_s
            writer.writerow(csv_row(participant, label, fold, timestamp_s, frame_count, features))
            frame_count += 1
            progress.update(1)
    cap.release()
    os.replace(tmp_path, csv_path)  # only becomes "done" once the whole video was processed

    elapsed = time.perf_counter() - started
    return {
        "video": f"{participant}/{video.name}", "fold": fold, "participant": participant,
        "label": label, "width": width, "height": height, "rotation_deg": rotation,
        "nominal_fps": round(nominal_fps, 3), "frames": frame_count, "duration_s": round(last_ts, 2),
        "pct_no_face": round(P.PERCENT * no_face_count / frame_count, 2) if frame_count else math.nan,
        "ts_non_monotonic": non_monotonic_count, "processing_s": round(elapsed, 1),
        "processing_fps": round(frame_count / elapsed, 2) if elapsed else math.nan,
    }


def list_videos(input_dir: Path, only: list[str] | None) -> list[Path]:
    if only:
        videos = []
        for name in only:
            found = [p for p in input_dir.glob(f"*/{name}") if p.is_file()]
            if not found:
                raise SystemExit(f"--only: '{name}' não encontrado em {input_dir}/<fold>/")
            videos += found
    else:
        videos = [p for p in input_dir.glob("*/*/*") if p.suffix.lower() in P.VIDEO_EXTENSIONS]
    return sorted(videos)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=P.UTA_RLDD_DIR)
    ap.add_argument("--output", type=Path, default=None,
                    help="padrão: data/features (ou data/features_test com --max-seconds)")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="processa só os primeiros N segundos (teste)")
    ap.add_argument("--only", nargs="+", metavar="PARTICIPANTE/ARQUIVO",
                    help="ex.: 01/10.MOV 04/0.mp4")
    ap.add_argument("--debug-frame-dir", type=Path, default=None,
                    help="salva 1 frame por vídeo com landmarks desenhados")
    ap.add_argument("--debug-frame-at", type=float, default=P.DEBUG_FRAME_AT_S,
                    help="segundo aproximado do frame de depuração (padrão 30)")
    return ap.parse_args()


def append_log(log_path: Path, info: dict, write_header: bool) -> None:
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=P.EXTRACTION_LOG_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(info)


def main() -> None:
    args = parse_args()
    # partial runs NEVER write to the final dir, or "skip if exists" would treat a 60 s CSV as a full video
    if args.output is None:
        args.output = P.FEATURES_TEST_DIR if args.max_seconds else P.FEATURES_DIR
    if args.debug_frame_dir:
        args.debug_frame_dir.mkdir(parents=True, exist_ok=True)

    videos = list_videos(args.input, args.only)
    print(f"{len(videos)} vídeo(s) em {args.input} -> CSVs em {args.output}")

    log_path = args.output / P.EXTRACTION_LOG_NAME
    args.output.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    face_mesh = create_face_mesh()

    done = skipped = 0
    for i, video in enumerate(videos, 1):
        fold, participant, label = parse_video_path(video, args.input)
        csv_path = args.output / fold / participant / f"{label}.csv"
        if csv_path.exists():
            skipped += 1
            print(f"[{i}/{len(videos)}] pulando (CSV já existe): {participant}/{video.name}")
            continue
        print(f"[{i}/{len(videos)}] {participant}/{video.name}")
        # reset tracking so it does not leak from the last frame of one video into the first of the next
        face_mesh.reset()
        info = process_video(video, args.input, csv_path, face_mesh, args.max_seconds,
                             args.debug_frame_dir, args.debug_frame_at)
        append_log(log_path, info, write_header)
        write_header = False
        print(f"    {info['frames']} frames | {info['pct_no_face']}% sem rosto | "
              f"rot={info['rotation_deg']}° | {info['width']}x{info['height']} | "
              f"{info['processing_fps']} fps de processamento ({info['processing_s']} s)")
        done += 1
    print(f"Concluído: {done} processado(s), {skipped} pulado(s).")


if __name__ == "__main__":
    main()
