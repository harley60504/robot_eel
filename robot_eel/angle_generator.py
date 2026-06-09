import math
from dataclasses import dataclass

# =============================
# Servo Config
# =============================
SERVO_COUNT = 6
MIN_DEG = 0
MAX_DEG = 240
servoDefaultAngles = [120] * SERVO_COUNT

# =============================
# MuJoCo / RL gait presets
# =============================
ANGLE_MODE = "CPG"

AJOINT_DEG = 15.0
FREQUENCY_HZ = 1.0
LAMBDA = 1.6275
BODY_LENGTH = 1.0


@dataclass(frozen=True)
class GaitPreset:
    key: str
    label: str
    ajoint: float
    frequency: float
    lambda_: float
    body_length: float
    amp_scales: tuple[float, ...]
    phase_lags: tuple[float, ...]
    joint_bias_deg: tuple[float, ...]


RL_VXHARD_AMP_SCALES = (1.24, 1.08, 1.0, 1.05, 1.1, 1.2)
RL_VXHARD_PHASE_LAGS = (0.614439, 0.614439, 0.614439, 0.614439, 0.614439)

TURN_SOFT_BIAS_DEG = tuple(math.degrees(value) for value in (0.08, 0.10, 0.12, 0.14, 0.16, 0.18))
TURN_STRONG_BIAS_DEG = tuple(math.degrees(value) for value in (0.12, 0.15, 0.18, 0.21, 0.24, 0.27))

GAIT_PRESETS = {
    "straight_rl": GaitPreset(
        key="straight_rl",
        label="Straight RL",
        ajoint=AJOINT_DEG,
        frequency=FREQUENCY_HZ,
        lambda_=LAMBDA,
        body_length=BODY_LENGTH,
        amp_scales=RL_VXHARD_AMP_SCALES,
        phase_lags=RL_VXHARD_PHASE_LAGS,
        joint_bias_deg=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ),
    "left_turn_rl": GaitPreset(
        key="left_turn_rl",
        label="Left Turn RL",
        ajoint=AJOINT_DEG,
        frequency=FREQUENCY_HZ,
        lambda_=LAMBDA,
        body_length=BODY_LENGTH,
        amp_scales=RL_VXHARD_AMP_SCALES,
        phase_lags=RL_VXHARD_PHASE_LAGS,
        joint_bias_deg=TURN_SOFT_BIAS_DEG,
    ),
    "left_spin_rl": GaitPreset(
        key="left_spin_rl",
        label="Left Strong RL",
        ajoint=AJOINT_DEG,
        frequency=FREQUENCY_HZ,
        lambda_=LAMBDA,
        body_length=BODY_LENGTH,
        amp_scales=RL_VXHARD_AMP_SCALES,
        phase_lags=RL_VXHARD_PHASE_LAGS,
        joint_bias_deg=TURN_STRONG_BIAS_DEG,
    ),
    "right_turn_rl": GaitPreset(
        key="right_turn_rl",
        label="Right Turn RL",
        ajoint=AJOINT_DEG,
        frequency=FREQUENCY_HZ,
        lambda_=LAMBDA,
        body_length=BODY_LENGTH,
        amp_scales=RL_VXHARD_AMP_SCALES,
        phase_lags=RL_VXHARD_PHASE_LAGS,
        joint_bias_deg=tuple(-value for value in TURN_SOFT_BIAS_DEG),
    ),
    "right_spin_rl": GaitPreset(
        key="right_spin_rl",
        label="Right Strong RL",
        ajoint=AJOINT_DEG,
        frequency=FREQUENCY_HZ,
        lambda_=LAMBDA,
        body_length=BODY_LENGTH,
        amp_scales=RL_VXHARD_AMP_SCALES,
        phase_lags=RL_VXHARD_PHASE_LAGS,
        joint_bias_deg=tuple(-value for value in TURN_STRONG_BIAS_DEG),
    ),
}

current_gait_key = "straight_rl"


def _gait() -> GaitPreset:
    return GAIT_PRESETS[current_gait_key]

# =============================
# SIN Params
# =============================
SIN_BASE = 0.0
SIN_AMP = AJOINT_DEG
SIN_FREQ = FREQUENCY_HZ

# =============================
# CPG Params
# =============================
ONBOARD_FEEDBACK_GAIN = 1.0


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def phase_offset(j):
    return -sum(_gait().phase_lags[:j])


def target_angle(j, theta):
    gait = _gait()
    out_deg = (
        gait.ajoint
        * gait.amp_scales[j]
        * math.cos(theta + phase_offset(j))
        + gait.joint_bias_deg[j]
    )
    return clamp(round(servoDefaultAngles[j] + out_deg, 1), MIN_DEG, MAX_DEG)


def init_generator():
    # Kept for controller.py compatibility.
    pass


def generate_angles_sin(t):
    gait = _gait()
    theta = 2.0 * math.pi * SIN_FREQ * t
    angles = []
    for j in range(SERVO_COUNT):
        out_deg = (
            SIN_BASE
            + SIN_AMP
            * gait.amp_scales[j]
            * math.sin(theta + phase_offset(j))
            + gait.joint_bias_deg[j]
        )
        angles.append(clamp(round(servoDefaultAngles[j] + out_deg, 1), MIN_DEG, MAX_DEG))
    return angles


def generate_angles_cpg(t, dt):
    theta = 2.0 * math.pi * _gait().frequency * t
    return [target_angle(j, theta) for j in range(SERVO_COUNT)]


def generate_angles(t, dt):
    mode = ANGLE_MODE.upper()

    if mode == "SIN":
        return generate_angles_sin(t)

    if mode == "CPG":
        return generate_angles_cpg(t, dt)

    raise ValueError(f"Unknown ANGLE_MODE: {ANGLE_MODE}")


def list_gaits():
    return [
        {
            "key": gait.key,
            "label": gait.label,
            "ajoint": gait.ajoint,
            "frequency": gait.frequency,
            "lambda": gait.lambda_,
            "amp_scales": list(gait.amp_scales),
            "phase_lags": list(gait.phase_lags),
            "joint_bias_deg": list(gait.joint_bias_deg),
        }
        for gait in GAIT_PRESETS.values()
    ]


def set_gait(key):
    global current_gait_key
    if key not in GAIT_PRESETS:
        raise ValueError(f"Unknown gait preset: {key}")
    current_gait_key = key


def current_gait():
    return _gait()


def generate_cpg_params(t, dt):
    """Return Flutter-compatible set_param fields for on-board CPG mode."""
    gait = _gait()
    payload = {
        "Ajoint": round(gait.ajoint, 4),
        "frequency": round(gait.frequency, 4),
        "lambda": round(gait.lambda_, 4),
        "L": round(gait.body_length, 4),
        "ampScales": [round(value, 6) for value in gait.amp_scales],
        "phaseLags": [round(value, 6) for value in gait.phase_lags],
        "jointBiasDeg": [round(value, 6) for value in gait.joint_bias_deg],
        "paused": False,
        "feedback": round(ONBOARD_FEEDBACK_GAIN, 4),
    }
    return payload
