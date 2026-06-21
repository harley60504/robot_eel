# RL Gait JSON

This backend can load fixed gait JSON files exported from the MuJoCo PPO turning pipeline.

## JSON Folder

Put exported RL gait JSON files here:

```text
Release/python_backend/json/
```

On startup, `angle_generator.py` loads JSON files in this priority order:

1. `ROBOT_EEL_RL_GAIT_DIR` environment variable, if set
2. `Release/python_backend/json/`
3. `mujoco_simulation/outputs/json/rl_gaits/`

If two files have the same gait `name`, the first one found is used.

## Select A Gait

The backend exposes loaded gaits through:

```http
GET /gaits
```

To switch gait:

```http
POST /set_gait
Content-Type: application/json

{"gait": "ppo_turn_right_a20_y07_run01"}
```

`set_gait()` also accepts a direct JSON file path, for example:

```json
{"gait": "C:/Users/ytyla/Documents/GitHub/robot_eel/Release/python_backend/json/ppo_turn_right_a20_y07_run01.json"}
```

## Parameter Mapping

The JSON fields map to ESP32 CPG params like this:

| JSON field | Backend / ESP32 field | Unit |
| --- | --- | --- |
| `ajoint` | `Ajoint` | degrees |
| `freq` | `frequency` | Hz |
| `wavelength` | `lambda` | body-length scale |
| `amp_scales` | `ampScales` | unitless |
| `phase_lags` | `phaseLags` | radians |
| `joint_bias` | `jointBiasDeg` | JSON radians -> degrees |

The Hopf CPG constants are matched with the MuJoCo/ESP32 implementation:

```text
r init = 0.25
theta init = phase offsets
alpha = 4.0
k_couple = 0.35
mu = ampScales^2
output = Ajoint * r * cos(theta) + jointBiasDeg
```

## Practical Use

For real-fish CPG output, use backend `output_mode = "cpg"`. The backend sends `set_param` to ESP32, and ESP32 runs the CPG locally.

If the backend is already running, restart it after replacing files in `Release/python_backend/json/`, or call `set_gait()` with a direct JSON path to load one file immediately.
