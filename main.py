import network
import socket
from machine import Pin, PWM


# ============================================================
# CONFIGURAÇÃO
# ============================================================

# ------------------------------------------------------------
# GARRA
# ------------------------------------------------------------

GRIPPER_PIN = 25

GRIPPER_DUTY_CLOSED = 26
GRIPPER_DUTY_OPEN = 80


# ------------------------------------------------------------
# EIXOS
# ------------------------------------------------------------
#
# Ordem:
#
#   eixo 0 = Roll
#   eixo 1 = Cotovelo
#   eixo 2 = Ombro
#
# Essa ordem precisa ser igual à ordem usada pelo Python.
#

AXIS_PINS = [
    26,  # eixo 0 - Roll
    27,  # eixo 1 - Cotovelo
    14,  # eixo 2 - Ombro
]


# ------------------------------------------------------------
# CALIBRAÇÃO DOS SERVOS
# ------------------------------------------------------------

AXIS_DUTY_AT_0 = [
    26,  # eixo 0
    26,  # eixo 1
    26,  # eixo 2
]

AXIS_DUTY_AT_180 = [
    128,  # eixo 0
    128,  # eixo 1
    128,  # eixo 2
]


# ------------------------------------------------------------
# LIMITES MECÂNICOS
# ------------------------------------------------------------

AXIS_MIN_ANGLE = [
    0,
    0,
    0,
]

AXIS_MAX_ANGLE = [
    180,
    180,
    180,
]


# ============================================================
# PROTOCOLO
# ============================================================

NUM_AXES = len(AXIS_PINS)

ANGLE_FIELD_WIDTH = 3

# 1 caractere da garra
# +
# 3 caracteres por eixo

PACKET_LEN = (
    1
    + NUM_AXES * ANGLE_FIELD_WIDTH
)


# ============================================================
# VALIDAÇÃO DA CONFIGURAÇÃO
# ============================================================

assert len(AXIS_DUTY_AT_0) == NUM_AXES, (
    "AXIS_DUTY_AT_0 fora de sincronia com AXIS_PINS"
)

assert len(AXIS_DUTY_AT_180) == NUM_AXES, (
    "AXIS_DUTY_AT_180 fora de sincronia com AXIS_PINS"
)

assert len(AXIS_MIN_ANGLE) == NUM_AXES, (
    "AXIS_MIN_ANGLE fora de sincronia com AXIS_PINS"
)

assert len(AXIS_MAX_ANGLE) == NUM_AXES, (
    "AXIS_MAX_ANGLE fora de sincronia com AXIS_PINS"
)


for i in range(NUM_AXES):

    assert AXIS_PINS[i] is not None, (
        "AXIS_PINS possui placeholder no eixo %d" % i
    )

    assert AXIS_DUTY_AT_0[i] is not None, (
        "AXIS_DUTY_AT_0 possui placeholder no eixo %d" % i
    )

    assert AXIS_DUTY_AT_180[i] is not None, (
        "AXIS_DUTY_AT_180 possui placeholder no eixo %d" % i
    )

    assert AXIS_MIN_ANGLE[i] is not None, (
        "AXIS_MIN_ANGLE possui placeholder no eixo %d" % i
    )

    assert AXIS_MAX_ANGLE[i] is not None, (
        "AXIS_MAX_ANGLE possui placeholder no eixo %d" % i
    )


# ============================================================
# REDE
# ============================================================

SOCKET_RECV_TIMEOUT_S = 2.0

WIFI_SSID = "DUMMY"
WIFI_PASSWORD = "12345678"

TCP_PORT = 80


# ============================================================
# ESTADO ATUAL
# ============================================================

last_gripper_state = None

last_angles = [90, 90, 90]


# ============================================================
# PWM DOS SERVOS
# ============================================================

gripper = PWM(
    Pin(
        GRIPPER_PIN,
        mode=Pin.OUT,
    )
)

gripper.freq(50)


axes = []

for pin in AXIS_PINS:

    pwm = PWM(
        Pin(
            pin,
            mode=Pin.OUT,
        )
    )

    pwm.freq(50)

    axes.append(pwm)


# ============================================================
# CONVERSÃO ÂNGULO -> DUTY
# ============================================================

def angle_to_duty(
    angle,
    duty_at_0,
    duty_at_180,
):
    """
    Converte 0-180 graus para o duty usado pelo PWM.
    """

    return int(
        duty_at_0
        + (
            angle / 180
        ) * (
            duty_at_180
            - duty_at_0
        )
    )


# ============================================================
# VALIDAÇÃO DO PACOTE
# ============================================================

def validate_packet(raw):
    """
    Retorna:

        (True, gripper_state, angles)

    ou:

        (False, erro, None)

    Pacotes inválidos nunca movimentam os servos.
    """

    # --------------------------------------------------------
    # TAMANHO
    # --------------------------------------------------------

    if len(raw) != PACKET_LEN:

        return (
            False,
            "ERR PACKET_LEN",
            None,
        )


    # --------------------------------------------------------
    # GARRA
    # --------------------------------------------------------

    if raw[0] not in ("0", "1"):

        return (
            False,
            "ERR GRIPPER_CHAR",
            None,
        )

    gripper_state = raw[0]


    # --------------------------------------------------------
    # EIXOS
    # --------------------------------------------------------

    angles = []

    for i in range(NUM_AXES):

        start = (
            1
            + i * ANGLE_FIELD_WIDTH
        )

        end = (
            start
            + ANGLE_FIELD_WIDTH
        )

        chunk = raw[start:end]


        # --------------------------------------------
        # SOMENTE NÚMEROS
        # --------------------------------------------

        if not chunk.isdigit():

            return (
                False,
                "ERR AXIS_%d_CHAR" % i,
                None,
            )


        angle = int(chunk)


        # --------------------------------------------
        # LIMITE PADRÃO DO SERVO
        # --------------------------------------------

        if angle > 180:

            return (
                False,
                "ERR AXIS_%d_RANGE" % i,
                None,
            )


        # --------------------------------------------
        # LIMITE MECÂNICO
        # --------------------------------------------

        if (
            angle < AXIS_MIN_ANGLE[i]
            or
            angle > AXIS_MAX_ANGLE[i]
        ):

            return (
                False,
                "ERR AXIS_%d_LIMIT" % i,
                None,
            )


        angles.append(angle)


    return (
        True,
        gripper_state,
        angles,
    )


# ============================================================
# APLICAÇÃO DO ESTADO
# ============================================================

def apply_state(
    gripper_state,
    angles,
):
    global last_gripper_state
    global last_angles


    # --------------------------------------------------------
    # GARRA
    # --------------------------------------------------------

    if gripper_state == "1":

        gripper.duty(
            GRIPPER_DUTY_CLOSED
        )

    else:

        gripper.duty(
            GRIPPER_DUTY_OPEN
        )


    # --------------------------------------------------------
    # EIXOS
    # --------------------------------------------------------

    for i, angle in enumerate(angles):

        duty = angle_to_duty(
            angle,
            AXIS_DUTY_AT_0[i],
            AXIS_DUTY_AT_180[i],
        )

        axes[i].duty(duty)


    # --------------------------------------------------------
    # MEMÓRIA DO ESTADO
    # --------------------------------------------------------

    last_gripper_state = gripper_state

    last_angles = list(angles)


# ============================================================
# ESTADO INICIAL
# ============================================================

# Coloca os três eixos em 90° inicialmente.

for i in range(NUM_AXES):

    duty = angle_to_duty(
        last_angles[i],
        AXIS_DUTY_AT_0[i],
        AXIS_DUTY_AT_180[i],
    )

    axes[i].duty(duty)


# Garra inicialmente aberta.

last_gripper_state = "0"

gripper.duty(
    GRIPPER_DUTY_OPEN
)


# ============================================================
# ACCESS POINT
# ============================================================

ap = network.WLAN(
    network.AP_IF
)


if ap.active():

    ap.active(False)


ap.active(True)

ap.config(
    essid=WIFI_SSID,
    password=WIFI_PASSWORD,
)

ap.config(
    authmode=network.AUTH_WPA_WPA2_PSK
)


print()
print("========================================")
print("       CONTROLE DE GARRA / BRACO")
print("========================================")
print()

print(
    "Ponto de Acesso criado!"
)

print(
    "SSID:",
    ap.config("essid")
)

print(
    "Endereco IP:",
    ap.ifconfig()[0]
)

print(
    "Porta TCP:",
    TCP_PORT
)

print(
    "Numero de eixos:",
    NUM_AXES
)

print(
    "Tamanho do pacote:",
    PACKET_LEN
)

print(
    "Estado inicial:",
    last_angles
)

print()


# ============================================================
# SERVIDOR TCP
# ============================================================

addr = socket.getaddrinfo(
    "0.0.0.0",
    TCP_PORT,
)[0][-1]


server = socket.socket()

server.setsockopt(
    socket.SOL_SOCKET,
    socket.SO_REUSEADDR,
    1,
)


server.bind(addr)

server.listen(1)


print(
    "Servidor TCP escutando na porta",
    TCP_PORT,
)

print(
    "Aguardando conexao..."
)

print()


# ============================================================
# LOOP PRINCIPAL
# ============================================================

try:

    while True:

        conn = None

        try:

            # ------------------------------------------------
            # AGUARDA CLIENTE
            # ------------------------------------------------

            conn, client_addr = (
                server.accept()
            )

            print(
                "Conexao recebida de",
                client_addr,
            )


            conn.settimeout(
                SOCKET_RECV_TIMEOUT_S
            )


            # ------------------------------------------------
            # RECEBE PACOTE
            # ------------------------------------------------

            try:

                data = conn.recv(64)

            except OSError:

                print(
                    "Timeout aguardando dados."
                )

                conn.close()

                continue


            # ------------------------------------------------
            # NENHUM DADO
            # ------------------------------------------------

            if not data:

                print(
                    "Cliente fechou conexao sem enviar dados."
                )

                conn.close()

                continue


            # ------------------------------------------------
            # DECODIFICA
            # ------------------------------------------------

            try:

                raw = data.decode().strip()

            except Exception:

                response = (
                    "ERR DECODE"
                )

                conn.send(response.encode())

                conn.close()

                continue


            print(
                "Recebido:",
                repr(raw),
            )


            # ------------------------------------------------
            # VALIDA
            # ------------------------------------------------

            ok, a, b = (
                validate_packet(raw)
            )


            # ------------------------------------------------
            # PACOTE VÁLIDO
            # ------------------------------------------------

            if ok:

                gripper_state = a
                angles = b


                # --------------------------------------------
                # MOVIMENTA
                # --------------------------------------------

                apply_state(
                    gripper_state,
                    angles,
                )


                response = (
                    "OK "
                    + raw
                )


                print(
                    "Pacote aplicado:",
                    raw,
                )

                print(
                    "  Garra:",
                    (
                        "FECHADA"
                        if gripper_state == "1"
                        else "ABERTA"
                    ),
                )

                print(
                    "  Roll:",
                    angles[0],
                )

                print(
                    "  Cotovelo:",
                    angles[1],
                )

                print(
                    "  Ombro:",
                    angles[2],
                )


            # ------------------------------------------------
            # PACOTE INVÁLIDO
            # ------------------------------------------------

            else:

                response = a

                print(
                    "Pacote rejeitado:",
                    repr(raw),
                    "->",
                    a,
                )


            # ------------------------------------------------
            # RESPOSTA
            # ------------------------------------------------

            try:

                conn.send(
                    response.encode()
                )

            except OSError:

                pass


        except OSError as error:

            print(
                "Erro no servidor:",
                error,
            )


        finally:

            if conn is not None:

                try:

                    conn.close()

                except OSError:

                    pass


except KeyboardInterrupt:

    print()
    print(
        "Encerrando servidor..."
    )


finally:

    try:

        server.close()

    except Exception:

        pass


    try:

        for pwm in axes:

            pwm.deinit()

        gripper.deinit()

    except Exception:

        pass


    ap.active(False)

    print(
        "Ponto de Acesso desligado."
    )