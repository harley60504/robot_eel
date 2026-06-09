from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from rl_tethered_env import EelTetheredRLEnv, TetheredConfig


def parse_float_list(value: str, expected_len: int, name: str) -> tuple[float, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != expected_len:
        raise argparse.ArgumentTypeError(f"{name} needs {expected_len} comma-separated values")
    return tuple(float(part) for part in parts)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained tethered eel PPO policy.")
    parser.add_argument("--model", type=Path, default=Path("outputs/ppo_hopf_tethered_eel.zip"))
    parser.add_argument("--episodes", type=int, default=5)
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
    env = EelTetheredRLEnv(cfg)
    model = PPO.load(args.model, env=env)

    episode_rewards = []
    force_x = []
    force_y = []
    force_metric = []
    energy = []
    actions = []
    physical_actions = []

    for _ in range(args.episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            if info.get("steady_state", True):
                force_x.append(info["force_x"])
                force_y.append(info["force_y"])
                force_metric.append(info["force_metric"])
                energy.append(info["energy_proxy"])
                actions.append(action)
                physical_actions.append(info["physical_action"])
        episode_rewards.append(total_reward)

    actions_arr = np.asarray(actions, dtype=np.float64)
    physical_actions_arr = np.asarray(physical_actions, dtype=np.float64)
    print("Tethered PPO policy evaluation")
    print(f"  model={args.model}")
    print(f"  mean episode reward={np.mean(episode_rewards):.3f}")
    print("  steady-state metrics only")
    print(f"  mean Fx={np.mean(force_x):.4f} N")
    print(f"  mean Fy={np.mean(force_y):.4f} N")
    print(f"  mean |Fy|={np.mean(np.abs(force_y)):.4f} N")
    print(f"  mean force metric={np.mean(force_metric):.4f} N")
    print(f"  mean energy proxy={np.mean(energy):.4f}")
    if args.per_joint_action:
        if args.fixed_frequency is None:
            print("  mean raw action [-1, 1] [frequency, amp_scale_1..6, phase_lag_1..5]=")
        else:
            print("  mean raw action [-1, 1] [amp_scale_1..6, phase_lag_1..5]=")
    else:
        if args.fixed_frequency is None:
            print("  mean raw action [-1, 1] [frequency, wavelength]=")
        else:
            print("  mean raw action [-1, 1] [wavelength]=")
    print(" ", np.mean(actions_arr, axis=0))
    if args.per_joint_action:
        if args.fixed_frequency is None:
            print("  mean physical action [frequency, amp_scale_1..6, phase_lag_1..5]=")
        else:
            print("  mean physical action [amp_scale_1..6, phase_lag_1..5]=")
    else:
        if args.fixed_frequency is None:
            print("  mean physical action [frequency, wavelength]=")
        else:
            print("  mean physical action [wavelength]=")
    print(" ", np.mean(physical_actions_arr, axis=0))


if __name__ == "__main__":
    main()
