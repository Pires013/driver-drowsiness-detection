"""Minimal demo: MediaPipe face mesh on the webcam with FPS (q quits)."""
from detection import parameters as P
from detection.drawing import draw_contours, draw_tesselation, put_text
from detection.landmarks import create_face_mesh, detect_face
from detection.webcam import capture_frames, measure_fps, show

WINDOW_TITLE = "Face Mesh - Deteccao de Fadiga"


def main() -> None:
    face_mesh = create_face_mesh()
    previous_time = 0
    for image in capture_frames():
        face_landmarks = detect_face(face_mesh, image)
        if face_landmarks is not None:
            draw_tesselation(image, face_landmarks)
            draw_contours(image, face_landmarks)

        fps, previous_time = measure_fps(previous_time)
        put_text(image, f"FPS: {int(fps)}", P.FPS_STYLE)
        if show(WINDOW_TITLE, image) == P.KEY_QUIT:
            break


if __name__ == "__main__":
    main()
