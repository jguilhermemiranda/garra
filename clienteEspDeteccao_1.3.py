import logging
import math
import os
import queue
import socket
import threading
import time
 
import cv2
import mediapipe as mp
 
ESP32_IP = "192.168.4.1"
ESP32_PORT = 80
SOCKET_TIMEOUT_S = 1.0
 
NUM_AXES = 3
 
AXIS_SMOOTHING_ALPHA = 0.25
 
 
DEADBAND_DEG = 2
 
MIN_SEND_INTERVAL_S = 0.05
 
RETRY_INTERVAL_S = 0.5
 
 
HAND_SCALE_NEAR = 0.35
HAND_SCALE_FAR = 0.12
 
SHOULDER_Z_NEAR = -0.20
SHOULDER_Z_FAR = 0.20
 
 
COLOR_BLUE = (255, 0, 0)
COLOR_WHITE = (255, 255, 255)
 
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
 
log = logging.getLogger("hand_control")
 
 
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
 
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
 
 
if os.name == "nt":
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
else:
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
 
if not cap.isOpened():
    raise RuntimeError("Não foi possível abrir a câmera.")
 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
 
 
FINGER_TIP_MCP = [
    (8, 5),
    (12, 9),
    (16, 13),
    (20, 17),
]
 
 
def detect_gripper_state(hand_landmarks):
 
    fingers_up = []
 
    for tip, mcp in FINGER_TIP_MCP:
        tip_y = hand_landmarks.landmark[tip].y
        mcp_y = hand_landmarks.landmark[mcp].y
        fingers_up.append(tip_y < mcp_y)
 
    if all(fingers_up):
        return False
 
    if not any(fingers_up):
        return True
 
    return None
 
 
def distance_2d(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)
 
 
def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))
 
 
def map_range(value, in_min, in_max, out_min, out_max):
    if in_max == in_min:
        return out_min
 
    ratio = (value - in_min) / (in_max - in_min)
 
    return out_min + ratio * (out_max - out_min)
 
 
_smoothed_angles = None
 
 
def smooth_angles(raw_angles):
 
    global _smoothed_angles
 
    if _smoothed_angles is None:
 
        _smoothed_angles = [float(angle) for angle in raw_angles]
 
    else:
 
        for i in range(len(raw_angles)):
 
            _smoothed_angles[i] = (
                AXIS_SMOOTHING_ALPHA * raw_angles[i]
                + (1.0 - AXIS_SMOOTHING_ALPHA) * _smoothed_angles[i]
            )
 
    return [
        int(round(clamp(angle, 0.0, 180.0)))
        for angle in _smoothed_angles
    ]
 
 
def compute_axis_angles(hand_landmarks):
 
    wrist = hand_landmarks.landmark[0]
    index_mcp = hand_landmarks.landmark[5]
    middle_mcp = hand_landmarks.landmark[9]
    pinky_mcp = hand_landmarks.landmark[17]
 
    dx = pinky_mcp.x - index_mcp.x
    dy = pinky_mcp.y - index_mcp.y
 
    roll = math.degrees(math.atan2(dy, dx))
 
    axis_roll = clamp(roll + 90.0, 0.0, 180.0)
 
    axis_elbow = clamp((1.0 - wrist.y) * 180.0, 0.0, 180.0)
 
    wrist_z = wrist.z
    middle_z = middle_mcp.z
    hand_z = (wrist_z + middle_z) / 2.0
 
    axis_shoulder = clamp(
        map_range(
            hand_z,
            SHOULDER_Z_NEAR,
            SHOULDER_Z_FAR,
            180.0,
            0.0,
        ),
        0.0,
        180.0,
    )
 
    smoothed_angles = smooth_angles(
        [axis_roll, axis_elbow, axis_shoulder]
    )
 
    debug = {
        "roll_raw": roll,
        "wrist_y": wrist.y,
        "wrist_z": wrist_z,
        "middle_z": middle_z,
        "hand_z": hand_z,
        "hand_scale": distance_2d(wrist, middle_mcp),
    }
 
    return smoothed_angles, debug
 
 
def draw_hand(img, hand_landmarks):
 
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
 
 
def build_packet(gripper_closed, axis_angles):
 
    if len(axis_angles) != NUM_AXES:
        raise ValueError(f"Esperado {NUM_AXES} eixos.")
 
    if any(not 0 <= int(angle) <= 180 for angle in axis_angles):
        raise ValueError("Ângulo fora da faixa 0-180.")
 
    gripper_char = "1" if gripper_closed else "0"
 
    angle_fields = "".join(f"{int(angle):03d}" for angle in axis_angles)
 
    return gripper_char + angle_fields
 
 
class Esp32Sender:
 
    def __init__(self, ip, port):
 
        self._ip = ip
        self._port = port
 
        self._queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._lock = threading.Lock()
 
        self._confirmed = None
        self._rejected = None
        self._attempted = None
        self._last_attempt_time = 0.0
 
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
 
    @staticmethod
    def _differs(a, b):
 
        if b is None:
            return True
 
        if a[0] != b[0]:
            return True
 
        return any(
            abs(x - y) >= DEADBAND_DEG
            for x, y in zip(a[1], b[1])
        )
 
    def submit(self, gripper_closed, axis_angles):
 
        state = (bool(gripper_closed), tuple(int(a) for a in axis_angles))
        now = time.monotonic()
 
        with self._lock:
 
            if not self._differs(state, self._confirmed):
                return False
 
            if (
                self._rejected is not None
                and not self._differs(state, self._rejected)
            ):
                return False
 
            if now - self._last_attempt_time < MIN_SEND_INTERVAL_S:
                return False
 
            if (
                self._attempted is not None
                and not self._differs(state, self._attempted)
                and now - self._last_attempt_time < RETRY_INTERVAL_S
            ):
                return False
 
            self._attempted = state
            self._last_attempt_time = now
 
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
 
        try:
            self._queue.put_nowait(state)
        except queue.Full:
            pass
 
        return True
 
    @property
    def in_sync(self):
        with self._lock:
            return (
                self._confirmed is not None
                and self._attempted is not None
                and not self._differs(self._attempted, self._confirmed)
            )
 
    def _run(self):
 
        while not self._stop.is_set():
 
            try:
                state = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
 
            self._send_once(state)
 
    def _send_once(self, state):
 
        packet = build_packet(state[0], state[1])
 
        try:
 
            with socket.socket(
                socket.AF_INET,
                socket.SOCK_STREAM,
            ) as s:
 
                s.settimeout(SOCKET_TIMEOUT_S)
                s.connect((self._ip, self._port))
                s.sendall(packet.encode())
 
                response = s.recv(64).decode(errors="replace").strip()
 
        except (ConnectionRefusedError, socket.timeout, OSError) as error:
 
            log.warning(
                "Falha de transporte: %s | pacote=%s (retry em %.1fs)",
                error,
                packet,
                RETRY_INTERVAL_S,
            )
            return
 
        if response.startswith("OK"):
 
            with self._lock:
                self._confirmed = state
                self._rejected = None
 
            log.info("ESP32 -> %s | pacote=%s", response, packet)
 
        elif response.startswith("ERR"):
 
            with self._lock:
                self._rejected = state
 
            log.error(
                "ESP32 rejeitou o pacote %s: %s (não será reenviado)",
                packet,
                response,
            )
 
        else:
 
            log.warning(
                "Resposta inesperada %r | pacote=%s (retry em %.1fs)",
                response,
                packet,
                RETRY_INTERVAL_S,
            )
 
    def stop(self):
 
        self._stop.set()
        self._thread.join(timeout=1.0)
 
 
sender = Esp32Sender(ESP32_IP, ESP32_PORT)
 
last_gripper_closed = None
last_packet = None
 
 
try:
 
    while True:
 
        success, img = cap.read()
 
        if not success:
            log.warning("Falha ao capturar frame.")
            continue
 
        img = cv2.flip(img, 1)
 
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
        results = hands.process(img_rgb)
 
        if results.multi_hand_landmarks:
 
            hand_lms = results.multi_hand_landmarks[0]
 
            draw_hand(img, hand_lms)
 
            gripper_state = detect_gripper_state(hand_lms)
 
            if gripper_state is not None:
                last_gripper_closed = gripper_state
 
            axis_angles, debug = compute_axis_angles(hand_lms)
 
            axis_roll = int(axis_angles[0])
            axis_elbow = int(axis_angles[1])
            axis_shoulder = int(axis_angles[2])
 
            if last_gripper_closed is not None:
 
                if sender.submit(last_gripper_closed, axis_angles):
 
                    last_packet = build_packet(
                        last_gripper_closed,
                        axis_angles,
                    )
 
            if last_gripper_closed is True:
                gripper_label = "Garra: FECHADA"
            elif last_gripper_closed is False:
                gripper_label = "Garra: ABERTA"
            else:
                gripper_label = "Garra: aguardando"
 
            overlay_lines = [
 
                gripper_label,
 
                f"Roll:       {axis_roll:03d}",
                f"Cotovelo:   {axis_elbow:03d}",
                f"Ombro:      {axis_shoulder:03d}",
 
                "",
 
                f"hand_scale: {debug['hand_scale']:.3f}",
                f"wrist Z:    {debug['wrist_z']:.3f}",
                f"middle Z:   {debug['middle_z']:.3f}",
                f"hand Z:     {debug['hand_z']:.3f}",
 
                "",
 
                f"ultimo envio: {last_packet or '---'}",
                f"sincronizado: {'SIM' if sender.in_sync else 'NAO'}",
            ]
 
            for i, line in enumerate(overlay_lines):
 
                cv2.putText(
                    img,
                    line,
                    (10, 35 + i * 27),
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
 
        display = cv2.resize(img, (1000, 700))
 
        cv2.imshow("Hand Control", display)
 
        key = cv2.waitKey(1) & 0xFF
 
        if key == ord("q"):
            break
 
 
finally:
 
    log.info("Encerrando controle...")
 
    sender.stop()
 
    cap.release()
 
    hands.close()
 
    cv2.destroyAllWindows()
 
