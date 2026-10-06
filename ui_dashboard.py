"""
ui_dashboard.py

Camada de apresentação do Hand Control.

Responsabilidade ÚNICA: desenhar o painel (OpenCV) a partir de um estado já
calculado. Não acessa câmera, MediaPipe, socket nem regras de movimento.

Observação: as fontes do OpenCV (Hershey) NÃO suportam acentos. Todos os
textos desenhados aqui são ASCII de propósito (ex.: "MAO", "NAO").
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

log = logging.getLogger("hand_control.ui")

# ---------------------------------------------------------------------------
# Layout (px). Canvas fixo 1280x720.
# ---------------------------------------------------------------------------
CANVAS_W, CANVAS_H = 1280, 720
HEADER_H = 84
MARGIN = 24

CAM_CARD = (24, 108, 936, 635)          # x1, y1, x2, y2
CAM_PAD = 16
CAM_W = CAM_CARD[2] - CAM_CARD[0] - 2 * CAM_PAD   # 880
CAM_H = CAM_CARD[3] - CAM_CARD[1] - 2 * CAM_PAD   # 495
CAM_ORIGIN = (CAM_CARD[0] + CAM_PAD, CAM_CARD[1] + CAM_PAD)

PACKET_CARD = (24, 651, 936, 696)
GRIPPER_CARD = (960, 108, 1256, 208)
AXES_CARD = (960, 224, 1256, 496)
LINK_CARD = (960, 512, 1256, 696)

# ---------------------------------------------------------------------------
# Paleta branco + azul (BGR).
# ---------------------------------------------------------------------------
WHITE = (255, 255, 255)
BG = (253, 248, 244)          # #F4F8FD
BLUE_900 = (138, 58, 30)      # #1E3A8A
BLUE_700 = (216, 78, 29)      # #1D4ED8
BLUE_600 = (235, 99, 37)      # #2563EB
BLUE_500 = (246, 130, 59)     # #3B82F6
BLUE_200 = (254, 219, 191)    # #BFDBFE
BLUE_100 = (254, 234, 219)    # #DBEAFE
BLUE_50 = (255, 246, 239)     # #EFF6FF
INK = (42, 23, 15)            # #0F172A
MUTED = (139, 116, 100)       # #64748B
BORDER = (242, 230, 220)      # #DCE6F2
SHADOW = (246, 238, 232)      # sombra suave

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX
AA = cv2.LINE_AA

AXIS_LABELS = ("ROLL", "COTOVELO", "OMBRO")
AXIS_MAX_DEG = 180


@dataclass
class UiState:
    """Tudo que o painel precisa para desenhar um frame."""

    hand_detected: bool
    gripper_closed: Optional[bool]          # None = ainda desconhecido
    angles: Optional[Sequence[int]]         # (roll, cotovelo, ombro) ou None
    in_sync: bool
    last_packet: Optional[str]
    debug: Optional[dict]
    target: str                             # "ip:porta"


# ---------------------------------------------------------------------------
# Primitivas de desenho
# ---------------------------------------------------------------------------
def _rounded_rect(img, p1, p2, color, radius):
    x1, y1 = p1
    x2, y2 = p2
    r = int(max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2)))
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
    for cx, cy in (
        (x1 + r, y1 + r),
        (x2 - r, y1 + r),
        (x1 + r, y2 - r),
        (x2 - r, y2 - r),
    ):
        cv2.circle(img, (cx, cy), r, color, -1, AA)


def _card(img, box, radius=18):
    x1, y1, x2, y2 = box
    _rounded_rect(img, (x1, y1 + 3), (x2, y2 + 3), SHADOW, radius)   # sombra
    _rounded_rect(img, (x1, y1), (x2, y2), BORDER, radius)           # borda
    _rounded_rect(img, (x1 + 1, y1 + 1), (x2 - 1, y2 - 1), WHITE, radius - 1)


def _text(img, text, org, scale=0.55, color=INK, thickness=1, font=FONT,
          align="left"):
    (w, _), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    if align == "right":
        x -= w
    elif align == "center":
        x -= w // 2
    cv2.putText(img, text, (int(x), int(y)), font, scale, color, thickness, AA)
    return w


def _pill(img, right_x, cy, text, fg, bg, border=None, height=32):
    """Pílula alinhada à direita. Retorna o x da borda esquerda."""
    (tw, _), _ = cv2.getTextSize(text, FONT, 0.5, 1)
    w = tw + 40
    x1, x2 = int(right_x - w), int(right_x)
    y1, y2 = int(cy - height // 2), int(cy + height // 2)
    r = height // 2
    if border is not None:
        _rounded_rect(img, (x1, y1), (x2, y2), border, r)
        _rounded_rect(img, (x1 + 1, y1 + 1), (x2 - 1, y2 - 1), bg, r - 1)
    else:
        _rounded_rect(img, (x1, y1), (x2, y2), bg, r)
    cv2.circle(img, (x1 + 16, int(cy)), 4, fg, -1, AA)
    _text(img, text, (x1 + 28, int(cy) + 5), 0.5, fg, 1)
    return x1


def _overlay_rgba(dst, src, x, y):
    h, w = src.shape[:2]
    roi = dst[y:y + h, x:x + w]
    if src.shape[2] == 4:
        a = src[:, :, 3:4].astype(np.float32) / 255.0
        blended = src[:, :, :3].astype(np.float32) * a + roi.astype(np.float32) * (1 - a)
        roi[:] = blended.astype(np.uint8)
    else:
        roi[:] = src


def _load_logo(path, target_h):
    if not os.path.isfile(path):
        log.warning("Logo não encontrada em %s (seguindo sem logo).", path)
        return None
    logo = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if logo is None:
        log.warning("Não foi possível ler a logo %s.", path)
        return None
    scale = target_h / logo.shape[0]
    size = (max(1, int(logo.shape[1] * scale)), target_h)
    return cv2.resize(logo, size, interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Camada estática (desenhada uma vez)
# ---------------------------------------------------------------------------
def build_static_layer(logo_path):
    canvas = np.full((CANVAS_H, CANVAS_W, 3), BG, dtype=np.uint8)

    # Header
    cv2.rectangle(canvas, (0, 0), (CANVAS_W, HEADER_H), WHITE, -1)
    cv2.line(canvas, (0, HEADER_H), (CANVAS_W, HEADER_H), BORDER, 2, AA)
    cv2.rectangle(canvas, (0, 0), (CANVAS_W, 4), BLUE_600, -1)

    text_x = MARGIN
    logo = _load_logo(logo_path, target_h=62)
    if logo is not None:
        _overlay_rgba(canvas, logo, MARGIN, 14)
        text_x = MARGIN + logo.shape[1] + 16
    _text(canvas, "HAND CONTROL", (text_x, 46), 0.95, BLUE_900, 2, FONT_BOLD)
    _text(canvas, "Controle do braco por gestos  |  ESP32",
          (text_x, 68), 0.5, MUTED, 1)

    # Cards
    for box in (CAM_CARD, PACKET_CARD, GRIPPER_CARD, AXES_CARD, LINK_CARD):
        _card(canvas, box)

    # Títulos de seção
    gx, gy = GRIPPER_CARD[0] + 20, GRIPPER_CARD[1] + 28
    _text(canvas, "GARRA", (gx, gy), 0.5, MUTED, 1)
    _text(canvas, "EIXOS  (0-180)", (AXES_CARD[0] + 20, AXES_CARD[1] + 28),
          0.5, MUTED, 1)
    _text(canvas, "ENLACE ESP32", (LINK_CARD[0] + 20, LINK_CARD[1] + 28),
          0.5, MUTED, 1)
    _text(canvas, "ULTIMO ENVIO", (PACKET_CARD[0] + 20, PACKET_CARD[1] + 28),
          0.5, MUTED, 1)
    _text(canvas, "Q  para sair", (PACKET_CARD[2] - 20, PACKET_CARD[1] + 28),
          0.5, MUTED, 1, align="right")

    return canvas


# ---------------------------------------------------------------------------
# Elementos dinâmicos
# ---------------------------------------------------------------------------
def _draw_header_status(canvas, state):
    cy = HEADER_H // 2 + 2
    x = CANVAS_W - MARGIN

    if state.in_sync:
        x = _pill(canvas, x, cy, "SINCRONIZADO", WHITE, BLUE_600)
    else:
        x = _pill(canvas, x, cy, "NAO SINCRONIZADO", MUTED, WHITE, border=BORDER)

    x -= 10
    if state.hand_detected:
        _pill(canvas, x, cy, "MAO DETECTADA", BLUE_700, BLUE_50, border=BLUE_200)
    else:
        _pill(canvas, x, cy, "SEM MAO", MUTED, WHITE, border=BORDER)


def _draw_camera(canvas, frame, hand_detected):
    view = cv2.resize(frame, (CAM_W, CAM_H), interpolation=cv2.INTER_AREA)

    # Badge "AO VIVO" translúcido
    overlay = view.copy()
    _rounded_rect(overlay, (14, 14), (118, 44), WHITE, 15)
    cv2.addWeighted(overlay, 0.85, view, 0.15, 0, view)
    cv2.circle(view, (32, 29), 5, BLUE_600, -1, AA)
    _text(view, "AO VIVO", (44, 34), 0.45, BLUE_900, 1)

    if not hand_detected:
        text = "Nenhuma mao detectada"
        (tw, _), _ = cv2.getTextSize(text, FONT, 0.7, 1)
        cx, cy = CAM_W // 2, CAM_H - 44
        overlay = view.copy()
        _rounded_rect(overlay, (cx - tw // 2 - 24, cy - 24),
                      (cx + tw // 2 + 24, cy + 18), WHITE, 21)
        cv2.addWeighted(overlay, 0.88, view, 0.12, 0, view)
        _text(view, text, (cx, cy + 6), 0.7, BLUE_900, 1, align="center")

    x, y = CAM_ORIGIN
    canvas[y:y + CAM_H, x:x + CAM_W] = view


def _draw_gripper(canvas, closed):
    x1, y1, x2, y2 = GRIPPER_CARD
    if closed is True:
        label, filled, color = "FECHADA", True, BLUE_600
    elif closed is False:
        label, filled, color = "ABERTA", False, BLUE_600
    else:
        label, filled, color = "AGUARDANDO", False, MUTED

    scale = 1.0 if closed is not None else 0.8   # "AGUARDANDO" é mais longo
    _text(canvas, label, (x1 + 20, y1 + 72), scale,
          INK if closed is not None else MUTED, 2, FONT_BOLD)

    cx, cy = x2 - 40, y1 + 58
    cv2.circle(canvas, (cx, cy), 20, BLUE_50, -1, AA)
    if filled:
        cv2.circle(canvas, (cx, cy), 12, color, -1, AA)
    else:
        cv2.circle(canvas, (cx, cy), 12, color, 3, AA)


def _draw_axes(canvas, angles, stale):
    x1, y1, x2, _ = AXES_CARD
    bar_x1, bar_x2 = x1 + 20, x2 - 20
    bar_w = bar_x2 - bar_x1
    row_y = y1 + 52

    for i, label in enumerate(AXIS_LABELS):
        y = row_y + i * 70
        value = None if angles is None else int(angles[i])

        _text(canvas, label, (bar_x1, y + 14), 0.5, MUTED, 1)
        txt = "---" if value is None else f"{value:03d}"
        _text(canvas, txt, (bar_x2, y + 18), 0.8,
              MUTED if (stale or value is None) else INK, 1, FONT_BOLD,
              align="right")

        by1, by2 = y + 30, y + 42
        _rounded_rect(canvas, (bar_x1, by1), (bar_x2, by2), BLUE_100, 6)

        if value is not None:
            fill = max(12, int(bar_w * min(max(value, 0), AXIS_MAX_DEG) / AXIS_MAX_DEG))
            color = BLUE_200 if stale else BLUE_600
            _rounded_rect(canvas, (bar_x1, by1), (bar_x1 + fill, by2), color, 6)
            kx, ky = bar_x1 + fill, (by1 + by2) // 2
            cv2.circle(canvas, (kx, ky), 9, WHITE, -1, AA)
            cv2.circle(canvas, (kx, ky), 9, BLUE_200 if stale else BLUE_700, 2, AA)


def _draw_link(canvas, state):
    x1, y1, x2, _ = LINK_CARD
    lx, rx = x1 + 20, x2 - 20

    rows = [("Destino", state.target)]
    rows.append(("Estado", "SINCRONIZADO" if state.in_sync else "AGUARDANDO"))
    dbg = state.debug
    if dbg is not None:
        rows.append(("hand_scale", f"{dbg['hand_scale']:.3f}"))
        rows.append(("wrist Z", f"{dbg['wrist_z']:.3f}"))
        rows.append(("hand Z", f"{dbg['hand_z']:.3f}"))
    else:
        rows += [("hand_scale", "---"), ("wrist Z", "---"), ("hand Z", "---")]

    for i, (label, value) in enumerate(rows):
        y = y1 + 58 + i * 26
        _text(canvas, label, (lx, y), 0.5, MUTED, 1)
        highlight = label == "Estado" and state.in_sync
        _text(canvas, value, (rx, y), 0.5, BLUE_700 if highlight else INK, 1,
              align="right")


def _draw_packet(canvas, packet):
    x1, y1, _, _ = PACKET_CARD
    _text(canvas, packet or "---", (x1 + 170, y1 + 30), 0.7, BLUE_700, 1, FONT_BOLD)


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------
def render_dashboard(static_layer, frame, state: UiState):
    """Devolve um canvas 1280x720 pronto para cv2.imshow."""
    canvas = static_layer.copy()

    _draw_header_status(canvas, state)
    _draw_camera(canvas, frame, state.hand_detected)
    _draw_gripper(canvas, state.gripper_closed)
    _draw_axes(canvas, state.angles, stale=not state.hand_detected)
    _draw_link(canvas, state)
    _draw_packet(canvas, state.last_packet)

    return canvas
