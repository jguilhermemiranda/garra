import logging
import math
import os
import queue
import socket
import threading
import time

import cv2
import mediapipe as mp


# ============================================================
# CONFIGURAÇÃO
# ============================================================

ESP32_IP = "192.168.4.1"
ESP32_PORT = 80
SOCKET_TIMEOUT_S = 1.0

NUM_AXES = 3

# Ordem dos servos:
#
# 0 = Roll
# 1 = Cotovelo
# 2 = Ombro
#
# Deve ser igual à ordem dos AXIS_PINS no ESP32.

AXIS_SMOOTHING_ALPHA = 0.25


# ============================================================
# CALIBRAÇÃO
# ============================================================

HAND_SCALE_NEAR = 0.35
HAND_SCALE_FAR = 0.12

SHOULDER_Z_NEAR = -0.20
SHOULDER_Z_FAR = 0.20


# ============================================================
# CORES
# ============================================================

# OpenCV usa BGR, não RGB.

COLOR_BLUE = (255, 0, 0)
COLOR_WHITE = (255, 255, 255)


# ============================================================
# LOG
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

log = logging.getLogger("hand_control")


# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)


# ============================================================
# CÂMERA
# ============================================================

if os.name == "nt":
    cap = cv2.VideoCapture(
        0,
        cv2.CAP_DSHOW,
    )
else:
    cap = cv2.VideoCapture(
        0,
        cv2.CAP_V4L2,
    )

if not cap.isOpened():
    raise RuntimeError(
        "Não foi possível abrir a câmera."
    )

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280,
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720,
)


# ============================================================
# ESTADO DA GARRA
# ============================================================

FINGER_TIP_MCP = [
    (8, 5),
    (12, 9),
    (16, 13),
    (20, 17),
]


def detect_gripper_state(hand_landmarks):
    """
    Retorna:

        False = mão aberta
        True  = mão fechada
        None  = estado ambíguo
    """

    fingers_up = []

    for tip, mcp in FINGER_TIP_MCP:

        tip_y = (
            hand_landmarks.landmark[tip].y
        )

        mcp_y = (
            hand_landmarks.landmark[mcp].y
        )

        fingers_up.append(
            tip_y < mcp_y
        )

    if all(fingers_up):
        return False

    if not any(fingers_up):
        return True

    return None


# ============================================================
# GEOMETRIA
# ============================================================

def distance_2d(a, b):
    return math.hypot(
        a.x - b.x,
        a.y - b.y,
    )


def clamp(
    value,
    minimum,
    maximum,
):
    return max(
        minimum,
        min(
            maximum,
            value,
        ),
    )


def map_range(
    value,
    in_min,
    in_max,
    out_min,
    out_max,
):
    if in_max == in_min:
        return out_min

    ratio = (
        (value - in_min)
        / (in_max - in_min)
    )

    return (
        out_min
        + ratio
        * (out_max - out_min)
    )


# ============================================================
# SUAVIZAÇÃO
# ============================================================

_smoothed_angles = None


def smooth_angles(raw_angles):
    """
    Aplica filtro EMA e sempre retorna inteiros.
    """

    global _smoothed_angles

    if _smoothed_angles is None:

        _smoothed_angles = [
            float(angle)
            for angle in raw_angles
        ]

    else:

        for i in range(
            len(raw_angles)
        ):

            _smoothed_angles[i] = (
                AXIS_SMOOTHING_ALPHA
                * raw_angles[i]
                +
                (
                    1.0
                    - AXIS_SMOOTHING_ALPHA
                )
                * _smoothed_angles[i]
            )

    return [
        int(
            round(
                clamp(
                    angle,
                    0.0,
                    180.0,
                )
            )
        )
        for angle in _smoothed_angles
    ]


# ============================================================
# CÁLCULO DOS EIXOS
# ============================================================

def compute_axis_angles(
    hand_landmarks,
):
    """
    Calcula:

        eixo 0 = Roll
        eixo 1 = Cotovelo
        eixo 2 = Ombro
    """

    wrist = (
        hand_landmarks.landmark[0]
    )

    index_mcp = (
        hand_landmarks.landmark[5]
    )

    middle_mcp = (
        hand_landmarks.landmark[9]
    )

    pinky_mcp = (
        hand_landmarks.landmark[17]
    )


    # ========================================================
    # EIXO 0 — ROLL
    # ========================================================

    dx = (
        pinky_mcp.x
        - index_mcp.x
    )

    dy = (
        pinky_mcp.y
        - index_mcp.y
    )

    roll = math.degrees(
        math.atan2(
            dy,
            dx,
        )
    )

    axis_roll = roll + 90.0

    axis_roll = clamp(
        axis_roll,
        0.0,
        180.0,
    )


    # ========================================================
    # EIXO 1 — COTOVELO
    # ========================================================

    axis_elbow = (
        1.0 - wrist.y
    ) * 180.0

    axis_elbow = clamp(
        axis_elbow,
        0.0,
        180.0,
    )


    # ========================================================
    # EIXO 2 — OMBRO
    # ========================================================

    wrist_z = wrist.z

    middle_z = middle_mcp.z

    hand_z = (
        wrist_z
        + middle_z
    ) / 2.0

    axis_shoulder = map_range(
        hand_z,
        SHOULDER_Z_NEAR,
        SHOULDER_Z_FAR,
        180.0,
        0.0,
    )

    axis_shoulder = clamp(
        axis_shoulder,
        0.0,
        180.0,
    )


    # ========================================================
    # SUAVIZAÇÃO
    # ========================================================

    raw_angles = [
        axis_roll,
        axis_elbow,
        axis_shoulder,
    ]

    smoothed_angles = (
        smooth_angles(
            raw_angles
        )
    )


    # ========================================================
    # DEBUG
    # ========================================================

    debug = {
        "roll_raw": roll,
        "wrist_y": wrist.y,
        "wrist_z": wrist_z,
        "middle_z": middle_z,
        "hand_z": hand_z,
        "hand_scale": distance_2d(
            wrist,
            middle_mcp,
        ),
    }


    return (
        smoothed_angles,
        debug,
    )


# ============================================================
# DESENHO COMPLETO DA MÃO
# ============================================================

def draw_hand(
    img,
    hand_landmarks,
):
    """
    Desenha TODOS os 21 landmarks do MediaPipe
    e TODAS as conexões da mão.

    Pontos:
        Branco

    Esqueleto/conexões:
        Azul
    """

    landmark_style = mp_draw.DrawingSpec(
        color=COLOR_WHITE,
        thickness=2,
        circle_radius=5,
    )

    connection_style = mp_draw.DrawingSpec(
        color=COLOR_BLUE,
        thickness=3,
        circle_radius=2,
    )

    mp_draw.draw_landmarks(
        image=img,
        landmark_list=hand_landmarks,
        connections=mp_hands.HAND_CONNECTIONS,
        landmark_drawing_spec=landmark_style,
        connection_drawing_spec=connection_style,
    )


# ============================================================
# TRANSPORTE ESP32
# ============================================================

class Esp32Sender:

    def __init__(
        self,
        ip,
        port,
    ):

        self._ip = ip
        self._port = port

        self._queue = queue.Queue(
            maxsize=1
        )

        self._stop = (
            threading.Event()
        )

        self._thread = (
            threading.Thread(
                target=self._run,
                daemon=True,
            )
        )

        self._thread.start()


    def send_state(
        self,
        packet,
    ):

        try:
            self._queue.get_nowait()

        except queue.Empty:
            pass

        try:
            self._queue.put_nowait(
                packet
            )

        except queue.Full:
            pass


    def _run(self):

        while not self._stop.is_set():

            try:

                packet = (
                    self._queue.get(
                        timeout=0.2
                    )
                )

            except queue.Empty:

                continue

            self._send_once(
                packet
            )


    def _send_once(
        self,
        packet,
    ):

        try:

            with socket.socket(
                socket.AF_INET,
                socket.SOCK_STREAM,
            ) as s:

                s.settimeout(
                    SOCKET_TIMEOUT_S
                )

                s.connect(
                    (
                        self._ip,
                        self._port,
                    )
                )

                s.sendall(
                    packet.encode()
                )

                response = s.recv(
                    64
                )

                log.info(
                    "ESP32 -> %s | pacote=%s",
                    response.decode(
                        errors="replace"
                    ),
                    packet,
                )

        except (
            ConnectionRefusedError,
            socket.timeout,
            OSError,
        ) as error:

            log.warning(
                "Falha de transporte: %s | estado=%s",
                error,
                packet,
            )


    def stop(self):

        self._stop.set()

        self._thread.join(
            timeout=1.0
        )


# ============================================================
# PROTOCOLO
# ============================================================

def build_packet(
    gripper_closed,
    axis_angles,
):
    """
    Protocolo:

        1 char = garra
        3 chars = eixo 0
        3 chars = eixo 1
        3 chars = eixo 2

    Total: 10 caracteres.
    """

    if len(axis_angles) != NUM_AXES:

        raise ValueError(
            f"Esperado {NUM_AXES} eixos."
        )


    if any(
        not 0 <= int(angle) <= 180
        for angle in axis_angles
    ):

        raise ValueError(
            "Ângulo fora da faixa 0-180."
        )


    gripper_char = (
        "1"
        if gripper_closed
        else "0"
    )


    angle_fields = "".join(
        f"{int(angle):03d}"
        for angle in axis_angles
    )


    return (
        gripper_char
        + angle_fields
    )


# ============================================================
# INICIALIZAÇÃO
# ============================================================

sender = Esp32Sender(
    ESP32_IP,
    ESP32_PORT,
)

last_gripper_closed = None
last_packet = None
last_send_time = 0.0

SEND_INTERVAL_S = 0.05


# ============================================================
# LOOP PRINCIPAL
# ============================================================

try:

    while True:

        success, img = (
            cap.read()
        )


        if not success:

            log.warning(
                "Falha ao capturar frame."
            )

            continue


        # ----------------------------------------------------
        # ESPELHAR CÂMERA
        # ----------------------------------------------------

        img = cv2.flip(
            img,
            1,
        )


        # ----------------------------------------------------
        # MEDIA PIPE
        # ----------------------------------------------------

        img_rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB,
        )

        results = hands.process(
            img_rgb
        )


        # ----------------------------------------------------
        # DETECÇÃO
        # ----------------------------------------------------

        if results.multi_hand_landmarks:

            hand_lms = (
                results.multi_hand_landmarks[0]
            )


            # ------------------------------------------------
            # ESQUELETO COMPLETO
            # ------------------------------------------------

            draw_hand(
                img,
                hand_lms,
            )


            # ------------------------------------------------
            # GARRA
            # ------------------------------------------------

            gripper_state = (
                detect_gripper_state(
                    hand_lms
                )
            )


            if gripper_state is not None:

                last_gripper_closed = (
                    gripper_state
                )


            # ------------------------------------------------
            # EIXOS
            # ------------------------------------------------

            (
                axis_angles,
                debug,
            ) = compute_axis_angles(
                hand_lms
            )


            axis_roll = int(
                axis_angles[0]
            )

            axis_elbow = int(
                axis_angles[1]
            )

            axis_shoulder = int(
                axis_angles[2]
            )


            # ------------------------------------------------
            # PACOTE
            # ------------------------------------------------

            if (
                last_gripper_closed
                is not None
            ):

                packet = (
                    build_packet(
                        last_gripper_closed,
                        axis_angles,
                    )
                )


                now = time.monotonic()


                if (
                    now
                    - last_send_time
                    >= SEND_INTERVAL_S
                ):

                    sender.send_state(
                        packet
                    )

                    last_send_time = now

                    last_packet = (
                        packet
                    )


            # ------------------------------------------------
            # LABEL GARRA
            # ------------------------------------------------

            if (
                last_gripper_closed
                is True
            ):

                gripper_label = (
                    "Garra: FECHADA"
                )

            elif (
                last_gripper_closed
                is False
            ):

                gripper_label = (
                    "Garra: ABERTA"
                )

            else:

                gripper_label = (
                    "Garra: aguardando"
                )


            # ------------------------------------------------
            # INFORMAÇÕES NA TELA
            # ------------------------------------------------

            overlay_lines = [

                gripper_label,

                f"Roll:       {axis_roll:03d}",

                f"Cotovelo:   {axis_elbow:03d}",

                f"Ombro:      {axis_shoulder:03d}",

                "",

                (
                    f"hand_scale: "
                    f"{debug['hand_scale']:.3f}"
                ),

                (
                    f"wrist Z:    "
                    f"{debug['wrist_z']:.3f}"
                ),

                (
                    f"middle Z:   "
                    f"{debug['middle_z']:.3f}"
                ),

                (
                    f"hand Z:     "
                    f"{debug['hand_z']:.3f}"
                ),

                "",

                (
                    f"packet: "
                    f"{last_packet or '---'}"
                ),
            ]


            for i, line in enumerate(
                overlay_lines
            ):

                y = (
                    35
                    + i * 27
                )


                cv2.putText(
                    img,
                    line,
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    COLOR_WHITE,
                    2,
                    cv2.LINE_AA,
                )


        else:

            cv2.putText(
                img,
                "Nenhuma mao detectada",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                COLOR_WHITE,
                2,
                cv2.LINE_AA,
            )


        # ----------------------------------------------------
        # JANELA
        # ----------------------------------------------------

        display = cv2.resize(
            img,
            (1000, 700),
        )


        cv2.imshow(
            "Hand Control",
            display,
        )


        key = (
            cv2.waitKey(1)
            & 0xFF
        )


        if key == ord("q"):
            break


finally:

    log.info(
        "Encerrando controle..."
    )

    sender.stop()

    cap.release()

    hands.close()

    cv2.destroyAllWindows()