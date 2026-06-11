# RL Gait Training Guide

This guide collects the commands for training PPO gaits, exporting trained policies to gait JSON files, and checking the result in the MuJoCo GUI.

Run all commands from the MuJoCo simulation folder:

```powershell
cd C:\Users\ytyla\Documents\GitHub\robot_eel\mujoco_simulation
```

Install required packages first if this is a new Python environment:

```powershell
python -m pip install mujoco numpy matplotlib gymnasium stable-baselines3
```

## 1. File outputs and what they mean

| Output | Meaning | Used by |
|---|---|---|
| `outputs/ppo_free_swim_shape.zip` | PPO model for straight free-swim shape tuning | `export_rl_gait_json.py` |
| `outputs/ppo_turn_left_shape_bias.zip` | PPO model for left turning gait training | `export_turning_rl_gait_json.py` |
| `outputs/ppo_turn_right_shape_bias.zip` | PPO model for right turning gait training | `export_turning_rl_gait_json.py` |
| `outputs/rl_gaits/rl_straight.json` | fixed straight RL gait JSON | `gait_gui.py`, `view_gait.py` |
| `outputs/rl_gaits/rl_turn_left.json` | fixed left-turn RL gait JSON | `gait_gui.py`, `view_gait.py`, `measure_turning.py` |
| `outputs/rl_gaits/rl_turn_right.json` | fixed right-turn RL gait JSON | `gait_gui.py`, `view_gait.py`, `measure_turning.py` |

`gait_gui.py` reads both `gaits/*.json` and `outputs/rl_gaits/*.json`. RL gaits are shown first in the Fixed Gait list.

## 2. Straight free-swim PPO

Straight free-swim RL trains only CPG shape parameters:

```text
6 amp_scales + 5 phase_lags = 11 action values
```

It does not train `joint_bias`, so it is mainly for straight swimming.

### Train straight PPO

```powershell
python train_free_swim_rl.py --timesteps 500000 --output outputs/ppo_free_swim_shape
```

### Continue straight PPO training

```powershell
python train_free_swim_rl.py --load-model outputs/ppo_free_swim_shape.zip --timesteps 300000 --output outputs/ppo_free_swim_shape
```

### Export straight PPO to gait JSON

```powershell
python export_rl_gait_json.py --model outputs/ppo_free_swim_shape.zip --output outputs/rl_gaits/rl_straight.json
```

### Open GUI and test

```powershell
python gait_gui.py
```

In the GUI, select `Fixed Gait`, press `Refresh`, then choose `[RL] rl_straight`.

## 3. Turning PPO

Turning RL trains CPG shape and static joint offset:

```text
6 amp_scales + 5 phase_lags + 6 joint_bias = 17 action values
```

This is the version used to train `rl_turn_left.json` and `rl_turn_right.json`.

The turning reward encourages:

```text
+ useful swimming speed
- yaw-rate error
- optional turn-radius error
- wrong turn direction
- lateral slip
- energy use
- action jumping
- joint_bias jumping
```

### Train left-turn PPO

```powershell
python train_turning_rl.py --turn-direction left --target-yaw-rate 0.45 --timesteps 500000 --output outputs/ppo_turn_left_shape_bias
```

### Continue left-turn PPO training

```powershell
python train_turning_rl.py --turn-direction left --target-yaw-rate 0.45 --load-model outputs/ppo_turn_left_shape_bias.zip --timesteps 300000 --output outputs/ppo_turn_left_shape_bias
```

### Export left-turn PPO to gait JSON

```powershell
python export_turning_rl_gait_json.py --turn-direction left --target-yaw-rate 0.45 --model outputs/ppo_turn_left_shape_bias.zip --output outputs/rl_gaits/rl_turn_left.json
```

### Measure left-turn gait

```powershell
python measure_turning.py --gait outputs/rl_gaits/rl_turn_left.json
```

### Train right-turn PPO

```powershell
python train_turning_rl.py --turn-direction right --target-yaw-rate 0.45 --timesteps 500000 --output outputs/ppo_turn_right_shape_bias
```

### Continue right-turn PPO training

```powershell
python train_turning_rl.py --turn-direction right --target-yaw-rate 0.45 --load-model outputs/ppo_turn_right_shape_bias.zip --timesteps 300000 --output outputs/ppo_turn_right_shape_bias
```

### Export right-turn PPO to gait JSON

```powershell
python export_turning_rl_gait_json.py --turn-direction right --target-yaw-rate 0.45 --model outputs/ppo_turn_right_shape_bias.zip --output outputs/rl_gaits/rl_turn_right.json
```

### Measure right-turn gait

```powershell
python measure_turning.py --gait outputs/rl_gaits/rl_turn_right.json
```

## 4. Common training parameters

These parameters can be used with `train_free_swim_rl.py` and `train_turning_rl.py` unless noted.

| Parameter | Meaning | Default / common value | What to change |
|---|---|---|---|
| `--timesteps` | Number of PPO training steps for this run | `100000` straight, `150000` turning default; recommended `300000~500000` | Increase if reward is still improving |
| `--output` | Output model path without `.zip` | `outputs/ppo_*` | Change to keep multiple experiments |
| `--load-model` | Existing `.zip` model to continue training | none | Use this to resume training |
| `--episode-seconds` | Simulation seconds per episode | straight `8`, turning `10` | Longer episodes give more stable metrics but train slower |
| `--warmup-seconds` | Seconds with no reward at episode start | `2` | Increase if CPG startup is unstable; decrease if collecting reward too late |
| `--freq` | CPG frequency in Hz | `1.0` | Higher can swim faster but may become unstable |
| `--wavelength` | CPG wavelength parameter | `1.6275` | Change if body wave looks too compressed or too stretched |
| `--ajoint` | Base joint amplitude in degrees | `15` | Larger gives stronger motion but can hit limits or waste energy |
| `--amp-scale-lows` | Lower bound for 6 amplitude scale actions | script default | Narrow if RL explores too aggressively |
| `--amp-scale-highs` | Upper bound for 6 amplitude scale actions | script default | Increase if motion is too weak; decrease if unstable |
| `--phase-lag-lows` | Lower bound for 5 adjacent-joint phase lags | script default | Narrow if phase becomes unrealistic |
| `--phase-lag-highs` | Upper bound for 5 adjacent-joint phase lags | script default | Increase only if the wave needs more delay |

Example of changing amplitude and frequency:

```powershell
python train_free_swim_rl.py --freq 1.2 --ajoint 18 --timesteps 300000 --output outputs/ppo_free_swim_f12_a18
```

Example of narrower action bounds:

```powershell
python train_free_swim_rl.py --amp-scale-lows 1.10,0.95,0.90,0.95,1.00,1.05 --amp-scale-highs 1.30,1.15,1.10,1.15,1.25,1.35 --timesteps 300000 --output outputs/ppo_free_swim_narrow
```

## 5. Turning-only parameters

These parameters are for `train_turning_rl.py` and `export_turning_rl_gait_json.py`.

| Parameter | Meaning | Default / common value | What to change |
|---|---|---|---|
| `--turn-direction` | Which direction to train or export | `left` or `right` | Use `left` for left turn, `right` for right turn |
| `--target-yaw-rate` | Target absolute yaw rate in rad/s | `0.45` | Increase for tighter/faster turning; decrease for wider turning |
| `--target-radius` | Optional target turn radius in meters | none | Use if you care more about radius than yaw rate |
| `--joint-bias-low` | Minimum learned joint bias in radians | `-0.30` | More negative allows stronger right-side bias but may destabilize |
| `--joint-bias-high` | Maximum learned joint bias in radians | `0.30` | More positive allows stronger left-side bias but may destabilize |
| `--speed-weight` | Reward weight for speed | `0.60` | Increase if it turns but barely moves forward |
| `--yaw-rate-weight` | Reward weight for matching yaw rate | `1.20` | Increase if turn rate is far from target |
| `--radius-weight` | Reward weight for matching radius | `0.00`, or `0.40` when `--target-radius` is used | Increase when training a fixed radius |
| `--turn-direction-weight` | Penalty for turning the wrong direction | `0.30` | Increase if it sometimes turns the wrong way |
| `--lateral-speed-weight` | Penalty for side slip | `0.05` | Increase if it slides sideways too much |
| `--energy-weight` | Penalty for large control output | `0.02` | Increase if the gait is too aggressive |
| `--smoothness-weight` | Penalty for action jumps | `0.02` | Increase if action changes are unstable |
| `--bias-smoothness-weight` | Penalty for joint_bias jumps | `0.02` | Increase if learned bias is noisy |

### Target yaw rate examples

Wider and gentler turn:

```powershell
python train_turning_rl.py --turn-direction left --target-yaw-rate 0.30 --timesteps 500000 --output outputs/ppo_turn_left_yaw030
```

Tighter and stronger turn:

```powershell
python train_turning_rl.py --turn-direction left --target-yaw-rate 0.65 --timesteps 500000 --output outputs/ppo_turn_left_yaw065
```

If `--target-yaw-rate` is too high, the policy may learn unstable bias, hit boundaries, or lose forward motion.

### Target radius examples

Train left turn for about 0.8 m radius:

```powershell
python train_turning_rl.py --turn-direction left --target-yaw-rate 0.45 --target-radius 0.8 --timesteps 500000 --output outputs/ppo_turn_left_r08
```

Export it:

```powershell
python export_turning_rl_gait_json.py --turn-direction left --target-yaw-rate 0.45 --target-radius 0.8 --model outputs/ppo_turn_left_r08.zip --output outputs/rl_gaits/rl_turn_left_r08.json
```

## 6. Export parameters

These parameters are for `export_rl_gait_json.py` and `export_turning_rl_gait_json.py`.

| Parameter | Meaning | Default / common value | What to change |
|---|---|---|---|
| `--model` | Trained PPO `.zip` file | `outputs/ppo_*.zip` | Must match the model you trained |
| `--output` | Output gait JSON path | `outputs/rl_gaits/*.json` | Change name to keep multiple versions |
| `--name` | Name stored inside the JSON | derived from file/direction | Change for GUI display |
| `--samples` | Number of steady-state actions collected before export | `300` | Increase for smoother average |
| `--max-episodes` | Maximum rollout episodes during export | `20` | Increase if export cannot collect enough samples |
| `--strategy` | How to collapse actions into one fixed gait | `mean` | Use `best-step` if mean is too weak; use `last` for final policy action |
| `--stochastic` | Use stochastic policy output during export | off | Usually keep off; deterministic is more repeatable |
| `--round` | Decimal places in JSON | `6` | Keep default unless file readability matters |

Recommended export strategy:

```powershell
python export_turning_rl_gait_json.py --turn-direction left --model outputs/ppo_turn_left_shape_bias.zip --output outputs/rl_gaits/rl_turn_left.json --strategy mean --samples 300
```

If the averaged gait becomes too weak, try:

```powershell
python export_turning_rl_gait_json.py --turn-direction left --model outputs/ppo_turn_left_shape_bias.zip --output outputs/rl_gaits/rl_turn_left_best.json --strategy best-step
```

## 7. How to judge whether training is healthy

During training, focus on these log fields:

| Log field | Good sign | Warning sign |
|---|---|---|
| `rollout/ep_rew_mean` | Slowly increases, or becomes less negative | Flat for a long time or suddenly drops |
| `rollout/ep_len_mean` | Near full episode length | Much shorter than expected, often means out-of-bounds termination |
| `train/approx_kl` | Around `0.001~0.03` | Larger than `0.1` often means unstable updates |
| `train/clip_fraction` | Usually below `0.2` | Very high for a long time can mean updates are too aggressive |
| `train/value_loss` | Can fluctuate, but not exploding | Keeps increasing without bound |
| `train/explained_variance` | Higher is better, often `0.5~1.0` | Near zero or negative for a long time |

Do not judge PPO only from `train/loss`. PPO loss can move up and down and still be normal.

## 8. Recommended workflow

For straight gait:

```powershell
python train_free_swim_rl.py --timesteps 500000 --output outputs/ppo_free_swim_shape
python export_rl_gait_json.py --model outputs/ppo_free_swim_shape.zip --output outputs/rl_gaits/rl_straight.json
python gait_gui.py
```

For left and right turning gaits:

```powershell
python train_turning_rl.py --turn-direction left --target-yaw-rate 0.45 --timesteps 500000 --output outputs/ppo_turn_left_shape_bias
python train_turning_rl.py --turn-direction right --target-yaw-rate 0.45 --timesteps 500000 --output outputs/ppo_turn_right_shape_bias

python export_turning_rl_gait_json.py --turn-direction left --model outputs/ppo_turn_left_shape_bias.zip --output outputs/rl_gaits/rl_turn_left.json
python export_turning_rl_gait_json.py --turn-direction right --model outputs/ppo_turn_right_shape_bias.zip --output outputs/rl_gaits/rl_turn_right.json

python measure_turning.py --gait outputs/rl_gaits/rl_turn_left.json
python measure_turning.py --gait outputs/rl_gaits/rl_turn_right.json
python gait_gui.py
```

## 9. What not to change first

Avoid changing these until the basic workflow works:

- XML model path.
- MuJoCo timestep.
- CPG implementation.
- Reward weights all at once.
- Very large `joint_bias` bounds such as `-0.8~0.8`.

Change one group of parameters at a time, then export and measure the resulting JSON.
