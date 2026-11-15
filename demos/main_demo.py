"""Live drowsiness demo (webcam or video file).

Run from the project root:
    python -m demos.main_demo                       # webcam
    python -m demos.main_demo --video path.mp4      # video file
Keys: r = recalibrate (new visitor), d = debug panel, q or ESC = quit.

Layer 1 (state): the LSTM analyzes the last 60 s every second and outputs a drowsiness probability.
Layer 2 (immediate events): closed eye >= 2 s, yawns in the last 15 min and head drop
(detection/events.py). Both combine into a level: Normal, Attention, Alert or Critical
(rules in detection/parameters.py). A simple PERCLOS rule (last 60 s) is shown next to it.

Preprocessing is shared with training (detection/preprocessing.py); the configuration comes from the
JSON saved with the final model (data/training/modelo_final/). --headless runs without a window (tests).
"""
import argparse
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from detection import parameters as P
from detection.drawing import draw_contours, draw_feature_points
from detection.events import EventState, EventTracker, LevelHold, NetworkGate, decide_level, level_reason
from detection.features import compute_features
from detection.landmarks import create_face_mesh, detect_face
from detection.preprocessing import LiveBuffer, wrap_deg
from detection.validity import is_valid_frame
from pipeline.extract_features import ROTATIONS, open_video
from training import models as M

DEFAULT_CONFIG = P.DATA_DIR / "training" / "modelo_final" / "config_modelo_final.json"
PANEL_W = 380
VIEW_H = 480
FONT = cv2.FONT_HERSHEY_SIMPLEX
NAN5 = [float("nan")] * P.N_MEASURES_BASE


class Network:
    """Final network (LSTM or GRU) with the training normalization; returns the drowsy probability of a window."""

    def __init__(self, config_path: Path, checkpoint: Path | None = None):
        self.config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        info = self.config["modelo"]
        path = Path(checkpoint) if checkpoint else Path(config_path).parent / info["checkpoint"]
        data = torch.load(path, map_location="cpu")
        self.model = M.make_rnn(data.get("kind", info["tipo"]), info["n_entradas"], info["n_classes"])
        self.model.load_state_dict(data["state"])
        self.model.eval()
        self.norm = {k: np.array(v, dtype=np.float32) for k, v in data["norm"].items()}
        self.positive = info["classe_sonolento"]
        torch.set_num_threads(2)

    def predict(self, window: np.ndarray) -> float:
        inputs = M.to_input(window[None].astype(np.float32), self.norm)
        return float(M.predict_proba(self.model, inputs)[0, self.positive])


class LiveSession:
    """Session logic without any UI, so it also serves tests with video files."""

    def __init__(self, network: Network, yawn_window_min: float = P.YAWN_COUNT_WINDOW_MIN):
        self.network = network
        self.yawn_window_min = yawn_window_min
        self.calibration_s = network.config["calibracao"]["segundos"]
        self.face_mesh = create_face_mesh()
        self.recalibrate(0.0)
        self.timeline: list[dict] = []

    def recalibrate(self, now: float) -> None:
        """Restart from scratch (new visitor): reference, window, events and level."""
        self.t0 = now
        self.buffer = LiveBuffer(self.calibration_s)
        self.tracker: EventTracker | None = None
        self.hold = LevelHold()
        self.gate_attention = NetworkGate(P.LEVEL_ATTENTION_PROB, P.LEVEL_ATTENTION_PROB_EXIT,
                                          P.LEVEL_ATTENTION_PERSIST_S)
        self.gate_alert = NetworkGate(P.LEVEL_ALERT_PROB, P.LEVEL_ALERT_PROB_EXIT, P.LEVEL_ALERT_PERSIST_S)
        self.events = EventState()
        self.prob: float | None = None
        self.perclos: float | None = None
        self.observed_steps = 0
        self.level = 0
        self.reason = ""
        self.alert_source = ""            # origin of the last Alert: "eye" (closed eye) or "net" (network only)
        self.last_update = -1e9
        self.ear_history: deque = deque()
        self.frame_log: deque = deque()      # (t, face, valid, raw EAR) of the last 60 s (debug panel)
        self.status = ""
        self.last_features: dict | None = None

    def process(self, frame_bgr, now: float) -> tuple[float, object]:
        """Detect the face and update buffer, events, network and level. Returns (session time, landmarks)."""
        t = now - self.t0
        height, width = frame_bgr.shape[:2]
        if height > P.LIVE_DETECT_HEIGHT_PX:
            scale = P.LIVE_DETECT_HEIGHT_PX / height
            frame_bgr = cv2.resize(frame_bgr, (round(width * scale), P.LIVE_DETECT_HEIGHT_PX),
                                   interpolation=cv2.INTER_AREA)
            height, width = frame_bgr.shape[:2]
        landmarks = detect_face(self.face_mesh, frame_bgr)
        features = compute_features(landmarks, width, height) if landmarks is not None else None
        self.last_features = features
        face = features is not None
        if face:
            valid = bool(is_valid_frame(P.FACE_FOUND, features["yaw"], features["roll"],
                                        features["ear_left"], features["ear_right"]))
            values = [features[name] for name in P.MEASURE_NAMES]
            valid &= bool(np.isfinite(values).all())
        else:
            valid, values = False, NAN5
        self.buffer.add(t, values, valid)
        self.frame_log.append((t, face, valid, features["ear_mean"] if face else float("nan")))
        while self.frame_log[0][0] < t - P.WINDOW_S:
            self.frame_log.popleft()

        if not self.buffer.calibrated:
            if self.buffer.try_calibrate(t):
                ref = self.buffer.ref
                self.tracker = EventTracker(ref_ear=ref[0], ref_mar=ref[1], ref_pitch=ref[2], ref_roll=ref[4],
                                          yawn_window_min=self.yawn_window_min)
            elif self.buffer.calibration_failed:
                self.recalibrate(now)           # no face during calibration: start over
                self.status = "Calibracao falhou (rosto nao encontrado). Recomecando."
            return t, landmarks

        ref = self.buffer.ref
        self.events = self.tracker.update(t, face, valid, features["ear_mean"] if face else float("nan"),
                                          features["mar"] if face else float("nan"),
                                          features["pitch"] if face else float("nan"),
                                          features["roll"] if face else float("nan"))
        if face and valid:
            self.ear_history.append((t, features["ear_mean"] / ref[0]))
        while self.ear_history and self.ear_history[0][0] < t - P.EAR_PLOT_S:
            self.ear_history.popleft()

        if t - self.last_update >= P.LIVE_UPDATE_S:           # the network runs once per second
            self.last_update = t
            window, self.observed_steps = self.buffer.window(t)
            if window is not None:
                self.prob = self.network.predict(window)
                self.perclos = float(M.window_stats(window)[0])
            elif t - self.buffer.calibration_s >= P.WINDOW_S:
                self.prob = self.perclos = None                # window with too few observations
        net_attention = self.gate_attention.update(t, self.prob)
        net_alert = self.gate_alert.update(t, self.prob)
        raw = decide_level(net_attention, net_alert, self.events)
        self.level = self.hold.update(t, raw)
        if raw >= 2:
            self.alert_source = "eye" if self.events.eye_closed_alert else "net"
        if self.level > raw:        # the level only drops after LEVEL_HOLD_S: show that it is leaving
            self.reason = f"saindo de {P.LEVEL_NAMES[self.level]} (aguarda {P.LEVEL_HOLD_S:g} s)"
        else:
            self.reason = level_reason(raw, self.prob, net_attention, net_alert, self.events,
                                       self.yawn_window_min)
        return t, landmarks

    def record(self, t: float, fps: float) -> None:
        """Append one timeline row per second of session."""
        if not self.buffer.calibrated or (self.timeline and t - self.timeline[-1]["t"] < 1.0):
            return
        e = self.events
        self.timeline.append({"t": round(t, 1), "nivel": P.LEVEL_NAMES[self.level],
                              "rede": "" if self.prob is None else round(self.prob, 3),
                              "perclos": "" if self.perclos is None else round(self.perclos, 1),
                              "bocejos_15min": e.yawn_count, "olho_fechado_s": round(e.eye_closed_s, 1),
                              "queda_cabeca": int(e.head_drop), "rosto_perdido_pos_queda": int(e.face_lost_after_drop),
                              "fps": round(fps, 1)})


def text(img, s, org, scale=0.6, color=(255, 255, 255), thick=1) -> None:
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def bar(img, x, y, w, h, fraction, color, marker=None) -> None:
    cv2.rectangle(img, (x, y), (x + w, y + h), (70, 70, 70), -1)
    cv2.rectangle(img, (x, y), (x + int(w * max(0.0, min(1.0, fraction))), y + h), color, -1)
    if marker is not None:
        cv2.line(img, (x + int(w * marker), y - 3), (x + int(w * marker), y + h + 3), (255, 255, 255), 2)


def draw_ear_plot(img, session: LiveSession, x, y, w, h) -> None:
    """Eye opening (EAR / reference) over the last 30 s, with the eye-closing threshold line."""
    cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 40), -1)
    top = 1.4
    limit_y = y + h - int(h * P.EAR_CLOSING_FACTOR / top)
    cv2.line(img, (x, limit_y), (x + w, limit_y), (60, 60, 230), 1)
    text(img, "limiar", (x + 3, limit_y - 3), 0.4, (60, 60, 230))
    if not session.ear_history:
        return
    t_end = session.ear_history[-1][0]
    pts = [(x + int(w * (1 - (t_end - t) / P.EAR_PLOT_S)), y + h - int(h * min(v, top) / top))
           for t, v in session.ear_history]
    cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, (255, 220, 120), 2, cv2.LINE_AA)


def head_state(e: EventState) -> tuple[str, tuple]:
    if e.face_lost_after_drop:
        return "ROSTO PERDIDO APOS QUEDA", P.LEVEL_COLORS[3]
    if e.head_drop:
        return "QUEDA DE CABECA", P.LEVEL_COLORS[3]
    if e.head_down_now:
        return "cabeca baixando", P.LEVEL_COLORS[1]
    return "normal", (200, 200, 200)


def draw_debug(view, session: LiveSession) -> None:
    """Temporary debug panel (key d): raw and relative pose, EAR and share of valid frames."""
    ref = session.buffer.ref
    f = session.last_features
    log = list(session.frame_log)
    n = max(len(log), 1)
    valid = sum(1 for _, _, v, _ in log if v)
    no_face = sum(1 for _, fc, _, _ in log if not fc)
    ratios = [e / ref[0] for _, _, v, e in log if v and ref is not None and np.isfinite(e)]
    raw_perclos = 100 * np.mean([r < P.EAR_CLOSING_FACTOR for r in ratios]) if ratios else float("nan")
    lines = [f"quadros 60 s: {n} | validos {100 * valid / n:.0f}% | sem rosto {100 * no_face / n:.0f}%"]
    if f is not None:
        lines.append(f"pitch/yaw/roll bruto: {f['pitch']:.0f} {f['yaw']:.0f} {f['roll']:.0f}")
    if ref is not None:
        lines.append(f"referencia p/y/r: {ref[2]:.0f} {ref[3]:.0f} {ref[4]:.0f}")
        if f is not None:
            rel = wrap_deg(np.array([f["pitch"], f["yaw"], f["roll"]]) - ref[2:])
            lines.append(f"relativo p/y/r: {rel[0]:.0f} {rel[1]:.0f} {rel[2]:.0f}")
            lines.append(f"EAR {f['ear_mean']:.3f} | ref {ref[0]:.3f} | razao {f['ear_mean'] / ref[0]:.2f} "
                         f"(limiar {P.EAR_CLOSING_FACTOR})")
        lines.append(f"PERCLOS dos quadros validos (60 s): {raw_perclos:.0f}%  | painel: "
                     f"{'--' if session.perclos is None else f'{session.perclos:.0f}%'}")
        lines.append(f"passos observados na janela: {session.observed_steps}/{P.WINDOW_STEPS}")
    box_h, box_w = 18 * len(lines) + 8, min(view.shape[1], 520)
    y0 = view.shape[0] - box_h - 16                      # box at the bottom, over the video and the panel
    region = view[y0:y0 + box_h, :box_w]
    region[:] = (region * 0.3).astype(np.uint8)
    for i, line in enumerate(lines):
        text(view, line, (6, y0 + 18 * (i + 1)), 0.45, (0, 255, 255))


def render(session: LiveSession, frame_bgr, landmarks, t: float, fps: float, debug: bool = False) -> np.ndarray:
    """Frame with face points plus the side panel (level, network, PERCLOS, yawns, head, plot)."""
    h, w = frame_bgr.shape[:2]
    view = cv2.resize(frame_bgr, (round(w * VIEW_H / h), VIEW_H))
    if landmarks is not None:
        draw_contours(view, landmarks)
        draw_feature_points(view, landmarks, 2)
    panel = np.full((VIEW_H, PANEL_W, 3), 30, np.uint8)
    e = session.events

    if not session.buffer.calibrated:
        left = max(0.0, session.calibration_s - t)
        text(view, f"CALIBRANDO: olhe para a camera, olhos abertos e boca fechada ({left:.0f} s)",
             (10, 30), 0.6, (0, 255, 255), 2)
        text(panel, "CALIBRANDO", (20, 60), 1.0, (0, 255, 255), 2)
        bar(panel, 20, 90, PANEL_W - 40, 16, min(1.0, t / session.calibration_s), (0, 200, 255))
        if session.status:
            text(panel, session.status[:42], (10, 140), 0.45, (80, 80, 255))
    else:
        color = P.LEVEL_COLORS[session.level]
        cv2.rectangle(panel, (0, 0), (PANEL_W, 70), color, -1)
        text(panel, P.LEVEL_NAMES[session.level], (20, 50), 1.5, (255, 255, 255), 3)
        if session.reason:
            text(panel, session.reason[:52], (10, 90), 0.45, (230, 230, 230))
        if session.prob is None:
            done = min(P.WINDOW_S, max(0.0, t - session.buffer.calibration_s))
            text(panel, f"REDE (ultimos 60 s): coletando {done:.0f}/{P.WINDOW_S} s", (20, 113), 0.5, (0, 255, 255), 1)
            bar(panel, 20, 122, PANEL_W - 40, 14, done / P.WINDOW_S, (0, 200, 255))
        else:
            c = P.LEVEL_COLORS[2 if session.prob > P.LEVEL_ALERT_PROB else 1 if session.prob > P.LEVEL_ATTENTION_PROB else 0]
            label = f"REDE (LSTM): {100 * session.prob:.0f}%"
            text(panel, label, (20, 113), 0.8, c, 2)
            width = cv2.getTextSize(label, FONT, 0.8, 2)[0][0]
            text(panel, "(ultimos 60 s)", (20 + width + 8, 113), 0.45, (200, 200, 200))
            bar(panel, 20, 122, PANEL_W - 40, 14, session.prob, c, marker=P.LEVEL_ALERT_PROB)
        if session.perclos is None:
            text(panel, "REGRA (PERCLOS 60 s): --", (20, 165), 0.6, (200, 200, 200))
        else:
            rule_on = session.perclos > 20.0
            text(panel, f"REGRA PERCLOS 60 s: {session.perclos:.0f}%  ({'sonolento' if rule_on else 'alerta'})",
                 (20, 165), 0.6, P.LEVEL_COLORS[2] if rule_on else (200, 200, 200))
            bar(panel, 20, 175, PANEL_W - 40, 12, session.perclos / 100, (200, 160, 80), marker=0.20)
        text(panel, f"Bocejos (ultimos {session.yawn_window_min:g} min): {e.yawn_count}", (20, 220), 0.6,
             P.LEVEL_COLORS[1] if e.yawn_count >= P.LEVEL_ATTENTION_YAWNS else (230, 230, 230))
        label, hc = head_state(e)
        text(panel, f"Cabeca: {label}", (20, 250), 0.6, hc)
        eye_color = P.LEVEL_COLORS[2] if e.eye_closed_alert else (230, 230, 230)
        text(panel, f"Olho fechado: {e.eye_closed_s:.1f} s", (20, 280), 0.6, eye_color)
        if session.last_features is None:
            text(panel, f"SEM ROSTO ({e.face_lost_s:.1f} s)", (20, 310), 0.6, P.LEVEL_COLORS[3])
        text(panel, "Abertura do olho (ultimos 30 s)", (20, 345), 0.5, (200, 200, 200))
        draw_ear_plot(panel, session, 20, 355, PANEL_W - 40, 95)
    text(panel, f"{fps:.0f} fps | t = {t:.0f} s | r recalibra, d depuracao, q sai", (10, VIEW_H - 8), 0.45, (150, 150, 150))
    canvas = np.hstack([view, panel])
    if debug:
        draw_debug(canvas, session)
    return canvas


class Beeper:
    """Beeps on Alert and Critical levels (sounddevice); stays silent without a sound device."""

    def __init__(self, enabled: bool):
        self.sd = None
        self.next_t = 0.0
        self.next_net_t = 0.0
        if enabled:
            try:
                import sounddevice
                self.sd = sounddevice
            except Exception as error:                   # missing library or sound device
                print(f"Sem som: {error}")

    def _wave(self, hz: int) -> np.ndarray:
        n = int(P.SOUND_SAMPLE_RATE * P.SOUND_BEEP_S)
        return (0.4 * np.sin(2 * np.pi * hz * np.arange(n) / P.SOUND_SAMPLE_RATE)).astype(np.float32)

    def update(self, level: int, now: float, source: str = "eye") -> None:
        """Critical and eye/drop Alert beep periodically (1.5 s and 3 s). Network-only Alert (`source` = "net")
        beeps once on entry and then every SOUND_NET_ALERT_REPEAT_S."""
        if level < 2:
            self.next_net_t = 0.0                 # on re-entering network Alert the first beep is immediate
        if self.sd is None or level < 2:
            return
        net_only = level == 2 and source == "net"
        # separate clocks: the slow network beep never delays the closed-eye/drop or Critical beeps
        due = now >= (self.next_net_t if net_only else self.next_t)
        if not due:
            return
        critical = level == 3
        try:
            self.sd.play(self._wave(P.SOUND_CRITICAL_HZ if critical else P.SOUND_ALERT_HZ), P.SOUND_SAMPLE_RATE)
        except Exception as error:
            print(f"Sem som: {error}")
            self.sd = None
            return
        if net_only:
            self.next_net_t = now + P.SOUND_NET_ALERT_REPEAT_S
        else:
            self.next_t = now + (P.SOUND_LEVEL_CRITICAL_REPEAT_S if critical else P.SOUND_LEVEL_ALERT_REPEAT_S)


def frames(args):
    """Yield (frame, time in s): webcam = wall clock; file = video timestamp (metadata rotation applied)."""
    if args.video:
        if not Path(args.video).is_file():
            raise SystemExit(f"vídeo não encontrado: {args.video}")
        cap, rotation = open_video(Path(args.video))
        start_s = args.start_s or 0.0
        if start_s:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    return
                t = cap.get(cv2.CAP_PROP_POS_MSEC) / P.MS_PER_SECOND
                if args.max_seconds and t - start_s >= args.max_seconds:
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
                if not ok:
                    continue
                yield frame, time.perf_counter() - started
        finally:
            cap.release()


def run(args, session: LiveSession | None = None) -> LiveSession:
    """Main loop: read frames, update the session, render and handle keys."""
    session = session or LiveSession(Network(args.config, args.checkpoint),
                                     P.FAIR_YAWN_COUNT_WINDOW_MIN if args.feira else P.YAWN_COUNT_WINDOW_MIN)
    beeper = Beeper(not args.no_sound and not args.headless)
    shots_dir = Path(args.shots) if args.shots else None
    if shots_dir:
        shots_dir.mkdir(parents=True, exist_ok=True)
    saved: set = set()
    fps, last = 0.0, time.perf_counter()
    debug = args.debug
    for n, (frame, now) in enumerate(frames(args)):
        if n == 0:
            session.recalibrate(now)
        t, landmarks = session.process(frame, now)
        tick = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(tick - last, 1e-6) if n else 1 / max(tick - last, 1e-6)
        last = tick
        session.record(t, fps)
        beeper.update(session.level, now, session.alert_source)
        want = None
        if shots_dir and session.buffer.calibrated:
            if session.level >= 1 and f"nivel{session.level}" not in saved:
                want = f"nivel{session.level}_{P.LEVEL_NAMES[session.level]}"
                saved.add(f"nivel{session.level}")
            elif int(t) // 120 > 0 and f"t{int(t) // 120}" not in saved and int(t) % 120 < 1:
                want = f"t{int(t)}s"
                saved.add(f"t{int(t) // 120}")
        if not args.headless or want:
            canvas = render(session, frame, landmarks, t, fps, debug)
            if want:
                cv2.imwrite(str(shots_dir / f"{want}_{int(t)}s.png"), canvas)
            if not args.headless:
                cv2.imshow(P.LIVE_WINDOW_TITLE, canvas)
                key = cv2.waitKey(1) & P.KEY_MASK
                if key in (P.KEY_QUIT, P.KEY_ESC):
                    break
                if key == ord("d"):
                    debug = not debug
                if key == P.KEY_RECALIBRATE:
                    session.recalibrate(now)
                    saved.clear()
    if not args.headless:
        cv2.destroyAllWindows()
    return session


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Demonstração ao vivo de sonolência")
    ap.add_argument("--video", help="arquivo de vídeo no lugar da webcam")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="JSON do modelo final")
    ap.add_argument("--checkpoint", type=Path, default=None, help="outro .pt (ex.: modelo de um fold)")
    ap.add_argument("--headless", action="store_true", help="sem janela (testes)")
    ap.add_argument("--feira", action="store_true",
                    help=f"memória curta para visitantes: bocejos contados nos últimos {P.FAIR_YAWN_COUNT_WINDOW_MIN} min")
    ap.add_argument("--no-sound", action="store_true")
    ap.add_argument("--debug", action="store_true", help="começa com o painel de depuração ligado (tecla d liga e desliga)")
    ap.add_argument("--max-seconds", type=float, default=None, help="para de ler o vídeo depois disto")
    ap.add_argument("--start-s", type=float, default=None, help="começa o vídeo neste instante")
    ap.add_argument("--timeline", type=Path, default=None, help="grava a linha do tempo (CSV)")
    ap.add_argument("--shots", type=Path, default=None, help="pasta para capturas de tela")
    return ap.parse_args(argv)


def main() -> None:
    args = parse_args()
    session = run(args)
    if args.timeline and session.timeline:
        import csv
        args.timeline.parent.mkdir(parents=True, exist_ok=True)
        with open(args.timeline, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(session.timeline[0]))
            writer.writeheader()
            writer.writerows(session.timeline)
        print(f"linha do tempo em {args.timeline}")


if __name__ == "__main__":
    main()
