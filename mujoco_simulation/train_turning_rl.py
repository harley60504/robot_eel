from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from hopf_cpg import degrees_to_radians
from rl_turning_env import EelTurningRLEnv, TurningConfig, direction_sign
from train_free_swim_rl import parse_float_list


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on open-loop turning gait reward.")
    parser.add_argument("--timesteps", type=int, default=150_000)
    parser.add_argument("--output", type=Path, default=Path("outputs/ppo_turn_left_shape_bias"))
    parser.add_argument("--load-model", type=Path, default=None)
    parser.add_argument("--turn-direction", choices=("left", "right"), default="left")
    parser.add_argument("--target-yaw-rate", type=float, default=0.45, help="Target absolute yaw rate in rad/s.")
    parser.add_argument("--target-radius", type=float, default=None, help="Optional target absolute turn radius in meters.")
    parser.add_argument("--episode-seconds", type=float, default=None)
    parser.add_argument("--warmup-seconds", type=float, default=None)
    parser.add_argument("--freq", type=float, default=None)
    parser.add_argument("--wavelength", type=float, default=None)
    parser.add_argument("--ajoint", type=float, default=None, help="Base joint angle amplitude in degrees.")
    parser.add_argument("--amp-scale-lows", type=lambda value: parse_float_list(value, 6, "amp-scale-lows"), default=None)
    parser.add_argument("--amp-scale-highs", type=lambda value: parse_float_list(value, 6, "amp-scale-highs"), default=None)
    parser.add_argument("--phase-lag-lows", type=lambda value: parse_float_list(value, 5, "phase-lag-lows"), default=None)
    parser.add_argument("--phase-lag-highs", type=lambda value: parse_float_list(value, 5, "phase-lag-highs"), default=None)
    parser.add_argument("--joint-bias-low", type=float, default=None, help="Minimum learned joint bias in radians.")
    parser.add_argument("--joint-bias-high", type=float, default=None, help="Maximum learned joint bias in radians.")
    parser.add_argument("--speed-weight", type=float, default=None)
    parser.add_argument("--yaw-rate-weight", type=float, default=None)
    parser.add_argument("--radius-weight", type=float, default=None)
    parser.add_argument("--turn-direction-weight", type=float, default=None)
    parser.add_argument("--lateral-speed-weight", type=float, default=None)
    parser.add_argument("--energy-weight", type=float, default=None)
    parser.add_argument("--smoothness-weight", type=float, default=None)
    parser.add_argument("--bias-smoothness-weight", type=float, default=None)
    return parser.parse_args()


def config_from_args(args) -> TurningConfig:
    cfg = TurningConfig()
    cfg.turn_direction = args.turn_direction
    cfg.target_yaw_rate = abs(float(args.target_yaw_rate))
    if args.target_radius is not None:
        cfg.target_radius = abs(float(args.target_radius))
        if args.radius_weight is None:
            cfg.radius_weight = 0.40
    if args.episode_seconds is not None:
        cfg.episode_seconds = args.episode_seconds
    if args.warmup_seconds is not None:
        cfg.warmup_seconds = args.warmup_seconds
    if args.freq is not None:
        cfg.fixed_frequency = args.freq
    if args.wavelength is not None:
        cfg.fixed_wavelength = args.wavelength
    if args.ajoint is not None:
        cfg.fixed_ajoint = degrees_to_radians(args.ajoint)
    if args.amp_scale_lows is not None:
        cfg.amp_scale_lows = args.amp_scale_lows
    if args.amp_scale_highs is not None:
        cfg.amp_scale_highs = args.amp_scale_highs
    if args.phase_lag_lows is not None:
        cfg.phase_lag_lows = args.phase_lag_lows
    if args.phase_lag_highs is not None:
        cfg.phase_lag_highs = args.phase_lag_highs
    if args.joint_bias_low is not None:
        cfg.joint_bias_low = args.joint_bias_low
    if args.joint_bias_high is not None:
        cfg.joint_bias_high = args.joint_bias_high
    if cfg.joint_bias_low > cfg.joint_bias_high:
        raise ValueError("joint-bias-low cannot be greater than joint-bias-high")

    # If the user trains only one direction, bias bounds can stay symmetric.
    # The reward's signed target yaw rate decides which side is useful.
    if args.speed_weight is not None:
        cfg.speed_weight = args.speed_weight
    if args.yaw_rate_weight is not None:
        cfg.yaw_rate_weight = args.yaw_rate_weight
    if args.radius_weight is not None:
        cfg.radius_weight = args.radius_weight
    if args.turn_direction_weight is not None:
        cfg.turn_direction_weight = args.turn_direction_weight
    if args.lateral_speed_weight is not None:
        cfg.lateral_speed_weight = args.lateral_speed_weight
    if args.energy_weight is not None:
        cfg.energy_weight = args.energy_weight
    if args.smoothness_weight is not None:
        cfg.smoothness_weight = args.smoothness_weight
    if args.bias_smoothness_weight is not None:
        cfg.bias_smoothness_weight = args.bias_smoothness_weight

    # Validate direction spelling early.
    direction_sign(cfg.turn_direction)
    return cfg


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    env = Monitor(EelTurningRLEnv(config_from_args(args)))
    if args.load_model is None:
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
        reset_num_timesteps = True
    else:
        model = PPO.load(args.load_model, env=env)
        model.verbose = 1
        reset_num_timesteps = False
    model.learn(total_timesteps=args.timesteps, reset_num_timesteps=reset_num_timesteps)
    model.save(args.output)
    print(f"saved turning policy to {args.output}.zip")


if __name__ == "__main__":
    main()
