import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

# =============================
# Servo Config
# =============================
SERVO_COUNT = 6
MIN_DEG = 0
MAX_DEG = 240
servoDefaultAngles = [120] * SERVO_COUNT

# =============================
# MuJoCo / RL gait JSON
# =============================
ANGLE_MODE = "CPG"
BODY_LENGTH = 1.0
CPG_ALPHA = 4.0
CPG_K_COUPLE = 0.35
LAST_GAIT_FILE = Path(__file__).resolve().parent / ".last_gait.json"


@dataclass(frozen=True)
class GaitPreset:
    key: str
    label: str
    ajoint: float
    frequency: float
    lambda_: float
    body_length: float
    alpha: float
    k_couple: float
    amp_scales: tuple[float, ...]
    phase_lags: tuple[float, ...]
    joint_bias_deg: tuple[float, ...]


GAIT_PRESETS = {}
current_gait_key = None
cpg_r = [0.25] * SERVO_COUNT
cpg_theta = [0.0] * SERVO_COUNT


def _repo_root():
    return Path(__file__).resolve().parents[2]


def _json_gait_dirs():
    dirs = []
    env_dir = os.environ.get("ROBOT_EEL_RL_GAIT_DIR", "").strip()
    if env_dir:
        dirs.append(Path(env_dir).expanduser())
    dirs.append(Path(__file__).resolve().parent / "json")
    return dirs


def _as_tuple(values, count, field):
    if not isinstance(values, list) or len(values) != count:
        raise ValueError(f"{field} must contain {count} values")
    return tuple(float(value) for value in values)


def _required(mapping, field, source):
    if not isinstance(mapping, dict) or field not in mapping:
        raise ValueError(f"{source}.{field} is required")
    return mapping[field]


def _json_to_gait_preset(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    source = data.get("source") if isinstance(data.get("source"), dict) else {}
    env_config = source.get("env_config") if isinstance(source.get("env_config"), dict) else {}

    name = str(_required(data, "name", "json"))
    ajoint = float(_required(data, "ajoint", "json"))
    frequency = float(_required(data, "freq", "json"))
    lambda_ = float(_required(data, "wavelength", "json"))
    body_length = float(env_config.get("body_length", data.get("body_length", BODY_LENGTH)))
    alpha = float(env_config.get("alpha", data.get("alpha", CPG_ALPHA)))
    k_couple = float(env_config.get("k_couple", data.get("k_couple", CPG_K_COUPLE)))
    amp_scales = _as_tuple(_required(data, "amp_scales", "json"), SERVO_COUNT, "amp_scales")
    phase_lags = _as_tuple(_required(data, "phase_lags", "json"), SERVO_COUNT - 1, "phase_lags")
    joint_bias_rad = _as_tuple(_required(data, "joint_bias", "json"), SERVO_COUNT, "joint_bias")

    return GaitPreset(
        key=name,
        label=f"RL {name}",
        ajoint=ajoint,
        frequency=frequency,
        lambda_=lambda_,
        body_length=body_length,
        alpha=alpha,
        k_couple=k_couple,
        amp_scales=amp_scales,
        phase_lags=phase_lags,
        joint_bias_deg=tuple(math.degrees(value) for value in joint_bias_rad),
    )


def load_json_gait(path):
    preset = _json_to_gait_preset(path)
    GAIT_PRESETS[preset.key] = preset
    return preset


def _save_last_gait(key, source=None):
    data = {"gait": key}
    if source:
        data["source"] = str(source)
    LAST_GAIT_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _restore_last_gait():
    global current_gait_key
    if not LAST_GAIT_FILE.exists():
        return

    try:
        data = json.loads(LAST_GAIT_FILE.read_text(encoding="utf-8"))
        key = str(data.get("gait") or "")
        source = str(data.get("source") or "")

        if key in GAIT_PRESETS:
            current_gait_key = key
            return

        if source:
            path = Path(source).expanduser()
            if path.exists():
                preset = load_json_gait(path)
                current_gait_key = preset.key
                return
    except Exception as exc:
        print(f"[angle_generator] skip last gait file {LAST_GAIT_FILE}: {exc}")


def reload_json_gaits():
    global current_gait_key
    loaded = []
    seen = set()
    GAIT_PRESETS.clear()
    current_gait_key = None
    for folder in _json_gait_dirs():
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.json")):
            try:
                preset = _json_to_gait_preset(path)
                if preset.key in seen:
                    continue
                GAIT_PRESETS[preset.key] = preset
                if current_gait_key is None:
                    current_gait_key = preset.key
                seen.add(preset.key)
                loaded.append(preset.key)
            except Exception as exc:
                print(f"[angle_generator] skip gait json {path}: {exc}")
    _restore_last_gait()
    return loaded


reload_json_gaits()


def _gait() -> GaitPreset:
    if current_gait_key is None:
        raise ValueError("No gait JSON loaded")
    return GAIT_PRESETS[current_gait_key]

# =============================
# SIN Params
# =============================
SIN_BASE = 0.0

def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def wrap_pi(x):
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def phase_offsets(gait=None):
    gait = gait or _gait()
    offsets = [0.0] * SERVO_COUNT
    for j in range(1, SERVO_COUNT):
        offsets[j] = offsets[j - 1] - gait.phase_lags[j - 1]
    return offsets


def phase_offset(j):
    return phase_offsets()[j]


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
    global cpg_r, cpg_theta
    cpg_r = [0.25] * SERVO_COUNT
    cpg_theta = phase_offsets()


def generate_angles_sin(t):
    gait = _gait()
    theta = 2.0 * math.pi * gait.frequency * t
    angles = []
    for j in range(SERVO_COUNT):
        out_deg = (
            SIN_BASE
            + gait.ajoint
            * gait.amp_scales[j]
            * math.sin(theta + phase_offset(j))
            + gait.joint_bias_deg[j]
        )
        angles.append(clamp(round(servoDefaultAngles[j] + out_deg, 1), MIN_DEG, MAX_DEG))
    return angles


def generate_angles_cpg(t, dt):
    global cpg_r, cpg_theta
    gait = _gait()
    dt = max(0.0, float(dt))
    offsets = phase_offsets(gait)
    old_r = list(cpg_r)
    old_theta = list(cpg_theta)
    omega = 2.0 * math.pi * gait.frequency
    mu_targets = [value * value for value in gait.amp_scales]

    dr = [gait.alpha * (mu_targets[j] - old_r[j] * old_r[j]) * old_r[j] for j in range(SERVO_COUNT)]
    dtheta = [omega] * SERVO_COUNT
    for j in range(SERVO_COUNT):
        if j - 1 >= 0:
            desired_l = offsets[j - 1] - offsets[j]
            err_l = wrap_pi((old_theta[j - 1] - old_theta[j]) - desired_l)
            dtheta[j] += gait.k_couple * math.sin(err_l)
        if j + 1 < SERVO_COUNT:
            desired_r = offsets[j + 1] - offsets[j]
            err_r = wrap_pi((old_theta[j + 1] - old_theta[j]) - desired_r)
            dtheta[j] += gait.k_couple * math.sin(err_r)

    cpg_r = [max(0.0, old_r[j] + dr[j] * dt) for j in range(SERVO_COUNT)]
    cpg_theta = [wrap_pi(old_theta[j] + dtheta[j] * dt) for j in range(SERVO_COUNT)]
    return [
        clamp(round(servoDefaultAngles[j] + gait.ajoint * cpg_r[j] * math.cos(cpg_theta[j]) + gait.joint_bias_deg[j], 1), MIN_DEG, MAX_DEG)
        for j in range(SERVO_COUNT)
    ]


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
            "L": gait.body_length,
            "alpha": gait.alpha,
            "k_couple": gait.k_couple,
            "amp_scales": list(gait.amp_scales),
            "phase_lags": list(gait.phase_lags),
            "joint_bias_deg": list(gait.joint_bias_deg),
        }
        for gait in GAIT_PRESETS.values()
    ]


def set_gait(key):
    global current_gait_key
    source = None
    possible_path = Path(key).expanduser()
    if possible_path.exists():
        source = possible_path.resolve()
        key = load_json_gait(possible_path).key
    if key not in GAIT_PRESETS:
        raise ValueError(f"Unknown gait preset: {key}")
    current_gait_key = key
    init_generator()
    _save_last_gait(key, source)


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
        "alpha": round(gait.alpha, 4),
        "kCouple": round(gait.k_couple, 4),
        "ampScales": [round(value, 6) for value in gait.amp_scales],
        "phaseLags": [round(value, 6) for value in gait.phase_lags],
        "jointBiasDeg": [round(value, 6) for value in gait.joint_bias_deg],
        "paused": False,
    }
    return payload
