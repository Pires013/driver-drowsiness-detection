"""Records the webcam (or a video) with per-frame measures and hand-marked events, to validate the
yawn and head-drop detectors (collection/validate_events.py compares the two).

Run from the project root:
    python -m collection.validation_recording                 # webcam
    python -m collection.validation_recording --video x.mp4 --simulate "b:23-25,h:60-64" --headless
Keys (ONE MARK PER KEY PRESS, at the exact instant; press at every action, not once per block):
b = yawn, h = head drop, c = looking at the phone, p = looking at the dashboard/mirror, n = normal,
q or ESC = quit.

Output in data/own/<session>/ (git-ignored): video.mp4, frames.csv (per-frame measures plus the key
pressed on that frame) and events.csv (one row per key press: type, start = end = key time in seconds).
--simulate injects marks by time ("b:24" or "b:23-25", which marks the end) to test without a webcam.
"""
import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2

from detection import parameters as P
from detection.features import compute_features
from detection.landmarks import create_face_mesh, detect_face
from pipeline.extract_features import ROTATIONS, open_video

NAN = float("nan")


def parse_simulation(spec: str | None) -> list[tuple[str, float]]:
    """Parse 'b:24,h:60-64' into [('b', 24.0), ('h', 64.0)]; a span marks its end instant."""
    marks = []
    for part in (spec or "").split(","):
        if part.strip():
            kind, span = part.strip().split(":")
            marks.append((kind, float(span.split("-")[-1])))
    return marks


class EventMarker:
    """Stores one mark (type, instant) per key press."""

    def __init__(self):
        self.intervals: list[tuple[str, float, float]] = []     # (type, start, end); start = end
        self.last: tuple[str, float] | None = None

    def press(self, kind: str, t: float) -> None:
        self.intervals.append((kind, t, t))
        self.last = (kind, t)


def frames_from(args):
    """Yield (frame, time in s) from the webcam (wall clock) or a file (video timestamp, rotation applied)."""
    if args.video:
        if not Path(args.video).is_file():
            raise SystemExit(f"vídeo não encontrado: {args.video}")
        cap, rotation = open_video(Path(args.video))
        if args.start_s:
            cap.set(cv2.CAP_PROP_POS_MSEC, args.start_s * 1000)
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    return
                t = cap.get(cv2.CAP_PROP_POS_MSEC) / P.MS_PER_SECOND - (args.start_s or 0.0)
                if args.max_seconds and t >= args.max_seconds:
                    return
                yield (cv2.rotate(frame, ROTATIONS[rotation]) if rotation else frame), t
        finally:
            cap.release()
    else:
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        started = time.perf_counter()
        try:
            while cap.isOpened():
                ok, frame = cap.read()
                if ok:
                    yield frame, time.perf_counter() - started
        finally:
            cap.release()


def record(args) -> Path:
    """Record frames, measures and marks into a session folder; returns the folder."""
    session = args.session or datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) / session
    out.mkdir(parents=True, exist_ok=True)
    face_mesh = create_face_mesh()
    marker = EventMarker()
    spans = parse_simulation(args.simulate)
    simulated = sorted((t, k) for k, t in spans)
    writer = None
    last_t = 0.0
    typed: list[str] = []
    with open(out / "frames.csv", "w", newline="", encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(P.VALIDATION_FRAME_COLUMNS)
        for frame, t in frames_from(args):
            height, width = frame.shape[:2]
            if height > P.VALIDATION_VIDEO_HEIGHT_PX:
                frame = cv2.resize(frame, (round(width * P.VALIDATION_VIDEO_HEIGHT_PX / height),
                                           P.VALIDATION_VIDEO_HEIGHT_PX), interpolation=cv2.INTER_AREA)
                height, width = frame.shape[:2]
            if writer is None and not args.no_video:
                writer = cv2.VideoWriter(str(out / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                         P.VALIDATION_VIDEO_FPS, (width, height))
            pressed, typed = typed, []
            while simulated and simulated[0][0] <= t:
                pressed.append(simulated.pop(0)[1])
                marker.press(pressed[-1], t)
            landmarks = detect_face(face_mesh, frame)
            features = compute_features(landmarks, width, height) if landmarks is not None else None
            csv_writer.writerow([f"{t:.4f}", int(features is not None),
                                 *[f"{features[c]:.6f}" if features else NAN for c in P.FEATURE_COLUMNS],
                                 "|".join(pressed)])
            if writer is not None:
                writer.write(frame)
            last_t = t
            if not args.headless:
                shown = frame.copy()
                recent = marker.last is not None and t - marker.last[1] < P.VALIDATION_FLASH_S
                label = P.VALIDATION_EVENT_NAMES[marker.last[0]] if recent else "-"
                cv2.putText(shown, f"{t:6.1f} s  marcas: {len(marker.intervals)}  ultima: {label}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, P.RED if recent else P.GREEN, 2)
                cv2.imshow("Gravacao de validacao (b h c p n, q sai)", shown)
                key = cv2.waitKey(1) & P.KEY_MASK
                if key in (P.KEY_QUIT, P.KEY_ESC):
                    break
                if key in P.VALIDATION_KEYS:
                    marker.press(P.VALIDATION_KEYS[key], t)
                    typed.append(P.VALIDATION_KEYS[key])           # goes into the "marcas" column of the next frame
    if writer is not None:
        writer.release()
    if not args.headless:
        cv2.destroyAllWindows()
    with open(out / "events.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tipo", "inicio_s", "fim_s"])
        w.writerows([(k, f"{s:.2f}", f"{e:.2f}") for k, s, e in marker.intervals])
    print(f"gravado em {out} ({len(marker.intervals)} marcas)")
    return out


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Gravação para validar bocejo e queda de cabeça")
    ap.add_argument("--video", help="arquivo de vídeo no lugar da webcam")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--out", type=Path, default=P.OWN_DIR)
    ap.add_argument("--session", help="nome da pasta da sessão (padrão: data e hora)")
    ap.add_argument("--simulate", help="marcas por tempo, ex.: 'b:23-25,h:60-64' (teste sem webcam)")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--no-video", action="store_true", help="não grava o video.mp4")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--start-s", type=float, default=None)
    return ap.parse_args(argv)


if __name__ == "__main__":
    record(parse_args())
