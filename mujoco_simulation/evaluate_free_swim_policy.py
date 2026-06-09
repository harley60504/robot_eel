from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from rl_free_swim_env import EelFreeSwimRLEnv
from train_free_swim_rl import config_from_args, parse_float_list


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a free-swim PPO policy.")
    parser.add_argument("--model", type=Path, default=Path("outputs/ppo_free_swim_shape.zip"))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--episode-seconds", type=float, default=None)
    parser.add_argument("--warmup-seconds", type=float, default=None)
    parser.add_argument("--freq", type=float, default=None)
    parser.add_argument("--wavelength", type=float, default=None)
    parser.add_argument("--ajoint", type=float, default=None)
    parser.add_argument("--amp-scale-lows", type=lambda value: parse_float_list(value, 6, "amp-scale-lows"), default=None)
    parser.add_argument("--amp-scale-highs", type=lambda value: parse_float_list(value, 6, "amp-scale-highs"), default=None)
    parser.add_argument("--phase-lag-lows", type=lambda value: parse_float_list(value, 5, "phase-lag-lows"), default=None)
    parser.add_argument("--phase-lag-highs", type=lambda value: parse_float_list(value, 5, "phase-lag-highs"), default=None)
    parser.add_argument("--lateral-velocity-weight", type=float, default=None)
    parser.add_argument("--lateral-position-weight", type=float, default=None)
    parser.add_argument("--yaw-weight", type=float, default=None)
    parser.add_argument("--yaw-rate-weight", type=float, default=None)
    parser.add_argument("--energy-weight", type=float, default=None)
    parser.add_argument("--smoothness-weight", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    env = EelFreeSwimRLEnv(config_from_args(args))
    model = PPO.load(args.model, env=env)

    rewards = []
    vx_values = []
    vy_values = []
    y_values = []
    yaw_values = []
    energy_values = []
    actions = []

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
                vx_values.append(info["velocity_x"])
                vy_values.append(info["velocity_y"])
                y_values.append(info["y"])
                yaw_values.append(info["yaw"])
                energy_values.append(info["energy_proxy"])
                actions.append(info["physical_action"])
        rewards.append(total_reward)

    actions_arr = np.asarray(actions, dtype=np.float64)
    print("Free-swim PPO policy evaluation")
    print(f"  model={args.model}")
    print(f"  mean episode reward={np.mean(rewards):.4f}")
    print("  steady-state metrics only")
    print(f"  mean vx={np.mean(vx_values):.6f} m/s")
    print(f"  mean vy={np.mean(vy_values):.6f} m/s")
    print(f"  mean |y|={np.mean(np.abs(y_values)):.6f} m")
    print(f"  mean |yaw|={np.mean(np.abs(yaw_values)):.6f} rad")
    print(f"  mean energy proxy={np.mean(energy_values):.6f}")
    print("  mean physical action [amp_scale_1..6, phase_lag_1..5]=")
    print(" ", np.mean(actions_arr, axis=0))


if __name__ == "__main__":
    main()
