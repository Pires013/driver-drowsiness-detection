"""Detectores de evento imediato (olho fechado, bocejo, queda de cabeça, rosto perdido) e níveis.

Usados pela demonstração ao vivo (demos/main_demo.py) e pela validação com eventos marcados à mão
(collection/validate_events.py). Trabalham frame a frame, com os instantes reais, e com a referência
de uma calibração: EAR de olho aberto, MAR de repouso e pitch neutro. Os limiares e durações estão em
detection/parameters.py.
"""
from collections import deque
from dataclasses import dataclass, field

from detection import parameters as P
from detection.calibration import ear_threshold, mar_threshold, pitch_drop_threshold
from detection.preprocessing import wrap_deg


class Run:
    """Condição contínua no tempo. Valor desconhecido (frame inválido ou sem rosto) até `bridge_s`
    não interrompe a sequência; um valor falso (ex.: olho aberto) interrompe na hora."""

    def __init__(self, bridge_s: float = P.EVENT_BRIDGE_S):
        self.bridge_s = bridge_s
        self.start: float | None = None
        self.last_true: float | None = None

    def update(self, t: float, cond: bool | None) -> tuple[float, float | None]:
        """Devolve (duração atual da sequência, início da sequência que acabou de terminar ou None)."""
        ended = None
        if cond:
            if self.start is None:
                self.start = t
            self.last_true = t
        elif self.start is not None and (cond is False or t - self.last_true > self.bridge_s):
            ended = (self.start, self.last_true)
            self.start = None
        return (self.last_true - self.start if self.start is not None else 0.0), ended

    def close(self) -> tuple[float, float] | None:
        if self.start is None:
            return None
        ended = (self.start, self.last_true)
        self.start = None
        return ended


class ClosedRun:
    """Olho fechado ao longo do tempo, tolerando aberturas curtas (tremido dos pontos, óculos).

    Uma abertura (olho aberto ou frame sem leitura) de até `bridge_s` entre dois quadros fechados não
    zera a contagem, mas o tempo de abertura NÃO conta: a duração é o tempo fechado de verdade
    (um fechamento de 2 s exige 2 s de olho fechado, não 2 s de relógio com piscadas no meio)."""

    def __init__(self, bridge_s: float = P.BRIDGE_MAX_GAP_S):
        self.bridge_s = bridge_s
        self.start: float | None = None
        self.last_true: float | None = None
        self._in_gap = False
        self.open_s = 0.0                 # soma das aberturas toleradas desde o início da sequência
        self.ended_closed_s = 0.0         # tempo fechado da última sequência que terminou

    def closed_s(self) -> float:
        return self.last_true - self.start - self.open_s if self.start is not None else 0.0

    def update(self, t: float, closed: bool) -> tuple[float, tuple[float, float] | None]:
        """Devolve (tempo fechado da sequência atual, (início, fim) da sequência que acabou de terminar)."""
        ended = None
        if closed:
            if self.start is None:
                self.start, self.open_s = t, 0.0
            elif self._in_gap:
                self.open_s += t - self.last_true       # a abertura tolerada não conta como olho fechado
            self.last_true, self._in_gap = t, False
        elif self.start is not None:
            if t - self.last_true > self.bridge_s:
                ended = self.close()
            else:
                self._in_gap = True
        return self.closed_s(), ended

    def close(self) -> tuple[float, float] | None:
        if self.start is None:
            return None
        self.ended_closed_s = self.closed_s()
        ended = (self.start, self.last_true)
        self.start, self._in_gap = None, False
        return ended


@dataclass
class EventState:
    """Estado dos eventos num instante."""
    eye_closed_s: float = 0.0          # duração da sequência atual de olho fechado
    eye_closed_alert: bool = False     # >= 2 s
    yawning: bool = False              # MAR acima do limiar agora
    yawn_count: int = 0                # bocejos nos últimos 15 min
    head_drop: bool = False            # queda: pitch abaixo do limiar por >= 2 s, ou inclinação lateral + olho fechado por >= 2 s
    head_drop_eye: bool = False        # queda + olho fechado >= 1 s nos 3 s antes do início da queda ou durante (Crítico)
    head_tilt_drop: bool = False       # a queda é pela inclinação lateral (roll) com olho fechado
    head_down_now: bool = False        # pitch abaixo do limiar neste frame
    face_lost_after_drop: bool = False
    face_lost_s: float = 0.0


@dataclass
class EventTracker:
    """Acompanha os eventos a partir da referência (ear, mar, pitch) da calibração."""
    ref_ear: float
    ref_mar: float
    ref_pitch: float
    ref_roll: float = 0.0              # roll da calibração (inclinação lateral relativa a ela)
    yawn_window_min: float = P.YAWN_COUNT_WINDOW_MIN    # bocejos contam nesta janela (min)
    yawns: list = field(default_factory=list)          # (início, fim) de cada bocejo concluído
    head_drops: list = field(default_factory=list)     # (início, fim) de cada queda concluída
    eye_closures: list = field(default_factory=list)   # (início, fim) de cada fechamento >= 2 s
    _yawn_starts: deque = field(default_factory=deque)

    def __post_init__(self):
        self.ear_limit = ear_threshold(self.ref_ear)
        self.mar_limit = mar_threshold(self.ref_mar)
        self.pitch_limit = pitch_drop_threshold(self.ref_pitch)
        self._eye, self._yawn, self._head, self._tilt = Run(), Run(), Run(), Run()
        self._eye_tol = ClosedRun()        # olho fechado COM tolerância de aberturas: só para o Crítico
        self._eye_long: list[list[float]] = []      # [início, fim] dos fechamentos com >= 1 s de olho fechado
        self._yawn_counted = False
        self._last_face_t: float | None = None
        self._last_face_eye_ok = False     # no último quadro com rosto: olho fechado (com tolerância) >= 0,5 s nos 3 s antes
        self._last_face_head_eye = False   # ... ou queda de cabeça já iniciada com olho fechado
        self._eye_short: list[list[float]] = []      # [início, fim] dos fechamentos com >= FACE_LOST_EYE_CLOSED_MIN_S

    def update(self, t: float, face: bool, valid: bool, ear: float, mar: float, pitch: float,
               roll: float = float("nan")) -> EventState:
        """Um frame: `face` = rosto detectado; `valid` = frame válido pela regra de validity.py."""
        def cond(test: bool) -> bool | None:
            return bool(test) if valid else None

        # Alerta por olho fechado (>= 2 s): sem tolerância, como sempre foi
        eye_s, eye_end = self._eye.update(t, cond(valid and ear < self.ear_limit))
        if eye_end and eye_end[1] - eye_end[0] >= P.EYE_CLOSED_ALERT_S:
            self.eye_closures.append(eye_end)
        # Olho fechado para o Crítico (>= 1 s antes ou durante a queda): tolera aberturas de até BRIDGE_MAX_GAP_S
        tol_s, _ = self._eye_tol.update(t, bool(valid and ear < self.ear_limit))
        for closures, min_s in ((self._eye_long, P.CRITICAL_EYE_CLOSED_MIN_S), (self._eye_short, P.FACE_LOST_EYE_CLOSED_MIN_S)):
            if tol_s >= min_s:
                if closures and closures[-1][0] == self._eye_tol.start:
                    closures[-1][1] = t
                else:
                    closures.append([self._eye_tol.start, t])
            while closures and closures[0][1] < t - 60:
                closures.pop(0)
        yawn_s, yawn_end = self._yawn.update(t, cond(valid and mar > self.mar_limit))
        if yawn_end:
            if self._yawn_counted:
                self.yawns.append(yawn_end)
            self._yawn_counted = False
        if yawn_s >= P.YAWN_MIN_S_LIVE and not self._yawn_counted:
            self._yawn_counted = True
            self._yawn_starts.append(self._yawn.start)
        head_below = bool(valid and wrap_deg(pitch - self.ref_pitch) < P.PITCH_DROP_OFFSET_DEG)
        head_s, head_end = self._head.update(t, cond(head_below))
        if head_end and head_end[1] - head_end[0] >= P.HEAD_DROP_MIN_S:
            self.head_drops.append(head_end)

        # Inclinação lateral: |roll relativo| acima do limite COM olho fechado, por >= 2 s. Usa só rosto + roll finito
        # (não a regra de frame válido, que descarta |roll| > 45°); olho aberto por até 0,3 s é "desconhecido" (piscada).
        tilted = bool(face and roll == roll and abs(wrap_deg(roll - self.ref_roll)) > P.ROLL_TILT_LIMIT_DEG)
        eye_closed_now = bool(ear == ear and ear < self.ear_limit)
        tilt_cond = None if not face or roll != roll else (False if not tilted else (True if eye_closed_now else None))
        tilt_s, tilt_end = self._tilt.update(t, tilt_cond)
        if tilt_end and tilt_end[1] - tilt_end[0] >= P.HEAD_DROP_MIN_S:
            self.head_drops.append(tilt_end)

        starts = [r.start for r, s in ((self._head, head_s), (self._tilt, tilt_s)) if s >= P.HEAD_DROP_MIN_S]
        drop_start = min(starts) if starts else None
        eye_before = drop_start is not None and any(
            b >= drop_start - P.CRITICAL_EYE_LOOKBACK_S and a <= t for a, b in self._eye_long)

        if face:
            self._last_face_t = t
            # olho fechado nos 3 s antes (ou queda já iniciada com olho fechado): a leitura do olho de frente é confiável
            self._last_face_eye_ok = any(b >= t - P.FACE_LOST_EYE_LOOKBACK_S and a <= t for a, b in self._eye_short)
            self._last_face_head_eye = eye_before
        lost_s = (t - self._last_face_t) if (not face and self._last_face_t is not None) else 0.0
        after_drop = lost_s >= P.FACE_LOST_MIN_S and (self._last_face_eye_ok or self._last_face_head_eye)

        horizon = t - self.yawn_window_min * 60
        while self._yawn_starts and self._yawn_starts[0] < horizon:
            self._yawn_starts.popleft()
        return EventState(eye_closed_s=eye_s, eye_closed_alert=eye_s >= P.EYE_CLOSED_ALERT_S,
                          yawning=self._yawn_counted, yawn_count=len(self._yawn_starts),
                          head_drop=head_s >= P.HEAD_DROP_MIN_S or tilt_s >= P.HEAD_DROP_MIN_S,
                          head_drop_eye=eye_before, head_tilt_drop=tilt_s >= P.HEAD_DROP_MIN_S, head_down_now=head_below or tilted,
                          face_lost_after_drop=after_drop, face_lost_s=lost_s)

    def finish(self) -> None:
        """Fecha as sequências abertas ao fim do vídeo (para a lista de intervalos)."""
        eye = self._eye.close()
        if eye and eye[1] - eye[0] >= P.EYE_CLOSED_ALERT_S:
            self.eye_closures.append(eye)
        yawn = self._yawn.close()
        if yawn and self._yawn_counted:
            self.yawns.append(yawn)
        head = self._head.close()
        if head and head[1] - head[0] >= P.HEAD_DROP_MIN_S:
            self.head_drops.append(head)
        tilt = self._tilt.close()
        if tilt and tilt[1] - tilt[0] >= P.HEAD_DROP_MIN_S:
            self.head_drops.append(tilt)


def decide_level(net_attention: bool, net_alert: bool, events: EventState) -> int:
    """0 Normal, 1 Atenção, 2 Alerta, 3 Crítico (regras de parameters.py). net_attention e net_alert são
    os estados dos NetworkGate da rede (persistência e histerese), já decididos por quem chama."""
    alert = net_alert or events.eye_closed_alert
    if events.head_drop_eye or (net_alert and events.head_drop) or events.face_lost_after_drop:
        return 3
    if alert:
        return 2
    if events.yawn_count >= P.LEVEL_ATTENTION_YAWNS or net_attention:
        return 1
    return 0


def level_reason(level: int, prob: float | None, net_attention: bool, net_alert: bool, events: EventState,
                 yawn_window_min: float = P.YAWN_COUNT_WINDOW_MIN) -> str:
    """Motivo do nível `level` (o que decide_level calculou), em texto curto sem acento (cv2.putText)."""
    why = []
    if level == 3:
        if events.face_lost_after_drop:
            why.append("rosto perdido logo apos queda de cabeca")
        else:
            why.append(("inclinacao lateral" if events.head_tilt_drop else "queda de cabeca") + " + "
                       + ("olho fechado" if events.head_drop_eye else "rede alta"))
    elif level == 2:
        if events.eye_closed_alert:
            why.append(f"olho fechado {events.eye_closed_s:.0f} s")
        if net_alert:
            held = prob is not None and prob <= P.LEVEL_ALERT_PROB
            why.append(f"rede > {100 * P.LEVEL_ALERT_PROB:.0f}% por {P.LEVEL_ALERT_PERSIST_S:.0f} s"
                       + (f" (segue ate cair de {100 * P.LEVEL_ALERT_PROB_EXIT:.0f}%)" if held else ""))
    elif level == 1:
        if events.yawn_count >= P.LEVEL_ATTENTION_YAWNS:
            why.append(f"{events.yawn_count} bocejos em {yawn_window_min:g} min")
        if net_attention:
            held = prob is not None and prob <= P.LEVEL_ATTENTION_PROB
            why.append(f"rede > {100 * P.LEVEL_ATTENTION_PROB:.0f}% por {P.LEVEL_ATTENTION_PERSIST_S:.0f} s"
                       + (f" (segue ate cair de {100 * P.LEVEL_ATTENTION_PROB_EXIT:.0f}%)" if held else ""))
    return " + ".join(why)


class LevelHold:
    """O nível sobe na hora e só desce depois de ficar mais baixo por LEVEL_HOLD_S (evita piscar)."""

    def __init__(self, hold_s: float = P.LEVEL_HOLD_S):
        self.hold_s = hold_s
        self.level = 0
        self._lower_since: float | None = None

    def update(self, t: float, raw_level: int) -> int:
        if raw_level >= self.level:
            self.level, self._lower_since = raw_level, None
        else:
            if self._lower_since is None:
                self._lower_since = t
            if t - self._lower_since >= self.hold_s:
                self.level, self._lower_since = raw_level, None
        return self.level


class NetworkGate:
    """Persistência e histerese da saída da rede: liga só depois de ficar acima de `enter` por
    `persist_s` segundos seguidos e desliga só quando cai abaixo de `exit_` (menor que `enter`).
    Sem saída (None) o estado é mantido e a contagem recomeça."""

    def __init__(self, enter: float, exit_: float, persist_s: float):
        self.enter, self.exit_, self.persist_s = enter, exit_, persist_s
        self.on = False
        self._above_since: float | None = None

    def update(self, t: float, prob: float | None) -> bool:
        if prob is None:
            self._above_since = None
        elif self.on:
            if prob < self.exit_:
                self.on = False
        elif prob > self.enter:
            if self._above_since is None:
                self._above_since = t
            if t - self._above_since >= self.persist_s - 1e-9:
                self.on, self._above_since = True, None
        else:
            self._above_since = None
        return self.on
