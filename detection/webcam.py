"""Webcam capture, display window and FPS helpers shared by demos and collection."""
import time
from collections.abc import Iterator

import cv2

from detection import parameters as P


def capture_frames(source: int = 0) -> Iterator:
    cap = cv2.VideoCapture(source)
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignorando frame vazio da câmera.")
                continue
            yield frame
    finally:
        cap.release()
        cv2.destroyAllWindows()


def show(title: str, image) -> int:
    cv2.imshow(title, image)
    return cv2.waitKey(P.WAIT_KEY_MS) & P.KEY_MASK


def measure_fps(previous_time: float) -> tuple[float, float]:
    now = time.time()
    elapsed = now - previous_time
    return (1 / elapsed if elapsed > 0 else 0), now
