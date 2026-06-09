from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from rl_tethered_env import EelTetheredRLEnv, TetheredConfig


def parse_float_list(value: str, expected_len: int, name: str) -> tuple[float, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != expected_len:
        raise argparse.ArgumentTypeError(f"{name} needs {expected_len} comma-separated values")
    return tuple(float(part) for part in parts)


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on the tethered eel thrust environment.")
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--output", type=Path, default=Path("outputs/ppo_tethered_eel"))
    parser.add_argument("--freq-min", type=float, default=None)
    parser.add_argument("--freq-max", type=float, default=None)
    parser.add_argument("--fixed-frequency", type=float, default=None)
    parser.add_argument("--wavelength-min", type=float, default=None)
    parser.add_argument("--wavelength-max", type=float, default=None)
    parser.add_argument("--fixed-wavelength", type=float, default=None)
    parser.add_argument("--frequency-weight", type=float, default=None)
    parser.add_argument("--episode-seconds", type=float, default=None)
    parser.add_argument("--warmup-seconds", type=float, default=None)
    parser.add_argument("--reward-mode", choices=("maximize_fx", "target_force"), default=None)
    parser.add_argument("--target-force", type=float, default=None)
    parser.add_argument("--target-metric", choices=("fx", "resultant"), default=None)
    parser.add_argument("--target-error-weight", type=float, default=None)
    parser.add_argument("--forward-force-weight", type=float, default=None)
    parser.add_argument("--lateral-force-weight", type=float, default=None)
    parser.add_argument("--energy-weight", type=float, default=None)
    parser.add_argument("--action-smoothness-weight", type=float, default=None)
    parser.add_argument("--raw-actions", action="store_true")
    parser.add_argument("--per-joint-action", action="store_true")
    parser.add_argument("--amp-scale-min", type=float, default=None)
    parser.add_argument("--amp-scale-max", type=float, default=None)
    parser.add_argument("--phase-lag-min", type=float, default=None)
    parser.add_argument("--phase-lag-max", type=float, default=None)
    parser.add_argument("--amp-scale-lows", type=lambda value: parse_float_list(value, 6, "amp-scale-lows"), default=None)
    parser.add_argument("--amp-scale-highs", type=lambda value: parse_float_list(value, 6, "amp-scale-highs"), default=None)
    parser.add_argument("--phase-lag-lows", type=lambda value: parse_float_list(value, 5, "phase-lag-lows"), default=None)
    parser.add_argument("--phase-lag-highs", type=lambda value: parse_float_list(value, 5, "phase-lag-highs"), default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    cfg = TetheredConfig()
    if args.freq_min is not None:
        cfg.frequency_min = args.freq_min
    if args.freq_max is not None:
        cfg.frequency_max = args.freq_max
    if args.fixed_frequency is not None:
        cfg.fixed_frequency = args.fixed_frequency
    if args.wavelength_min is not None:
        cfg.wavelength_min = args.wavelength_min
    if args.wavelength_max is not None:
        cfg.wavelength_max = args.wavelength_max
    if args.fixed_wavelength is not None:
        cfg.fixed_wavelength = args.fixed_wavelength
    if args.frequency_weight is not None:
        cfg.frequency_weight = args.frequency_weight
    if args.episode_seconds is not None:
        cfg.episode_seconds = args.episode_seconds
    if args.warmup_seconds is not None:
        cfg.warmup_seconds = args.warmup_seconds
    if args.reward_mode is not None:
        cfg.reward_mode = args.reward_mode
    if args.target_force is not None:
        cfg.target_force = args.target_force
    if args.target_metric is not None:
        cfg.target_metric = args.target_metric
    if args.target_error_weight is not None:
        cfg.target_error_weight = args.target_error_weight
    if args.forward_force_weight is not None:
        cfg.forward_force_weight = args.forward_force_weight
    if args.lateral_force_weight is not None:
        cfg.lateral_force_weight = args.lateral_force_weight
    if args.energy_weight is not None:
        cfg.energy_weight = args.energy_weight
    if args.action_smoothness_weight is not None:
        cfg.action_smoothness_weight = args.action_smoothness_weight
    if args.raw_actions:
        cfg.normalized_actions = False
    if args.per_joint_action:
        cfg.per_joint_action = True
    if args.amp_scale_min is not None:
        cfg.amp_scale_min = args.amp_scale_min
    if args.amp_scale_max is not None:
        cfg.amp_scale_max = args.amp_scale_max
    if args.phase_lag_min is not None:
        cfg.phase_lag_min = args.phase_lag_min
    if args.phase_lag_max is not None:
        cfg.phase_lag_max = args.phase_lag_max
    if args.amp_scale_lows is not None:
        cfg.amp_scale_lows = args.amp_scale_lows
    if args.amp_scale_highs is not None:
        cfg.amp_scale_highs = args.amp_scale_highs
    if args.phase_lag_lows is not None:
        cfg.phase_lag_lows = args.phase_lag_lows
    if args.phase_lag_highs is not None:
        cfg.phase_lag_highs = args.phase_lag_highs

    env = Monitor(EelTetheredRLEnv(cfg))
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=1024,
        batch_size=256,
        gamma=0.98,
        learning_rate=1e-4,
        ent_coef=0.005,
    )
    model.learn(total_timesteps=args.timesteps)
    model.save(args.output)
    print(f"saved policy to {args.output}.zip")


if __name__ == "__main__":
    main()
