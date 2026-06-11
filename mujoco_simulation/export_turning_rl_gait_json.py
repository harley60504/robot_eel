from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from hopf_cpg import degrees_to_radians
from rl_turning_env import EelTurningRLEnv, TurningConfig, direction_sign
from train_free_swim_rl import parse_float_list


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a trained turning PPO policy as a fixed Hopf CPG turning gait JSON."
    )
    parser.add_argument("--model", type=Path, default=Path("outputs/ppo_turn_left_shape_bias.zip"))
    parser.add_argument("--output", type=Path, default=Path("gaits/rl_turn_left.json"))
    parser.add_argument("--name", default=None, help="Name stored in the gait JSON. Default derives from turn direction.")
    parser.add_argument("--turn-direction", choices=("left", "right"), default="left")
    parser.add_argument("--target-yaw-rate", type=float, default=0.45, help="Target absolute yaw rate in rad/s.")
    parser.add_argument("--target-radius", type=float, default=None, help="Optional target absolute turn radius in meters.")
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--max-episodes", type=int, default=20)
    parser.add_argument("--strategy", choices=("mean", "last", "best-step"), default="mean")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--round", type=int, default=6)

    parser.add_argument("--episode-seconds", type=float, default=None)
    parser.add_argument("--warmup-seconds", type=float, default=None)
    parser.add_argument("--freq", type=float, default=None)
    parser.add_argument("--wavelength", type=float, default=None)
    parser.add_argument("--ajoint", type=float, default=None, help="Base joint angle amplitude in degrees.")
    parser.add_argument("--amp-scale-lows", type=lambda value: parse_float_list(value, 6, "amp-scale-lows"), default=None)
    parser.add_argument("--amp-scale-highs", type=lambda value: parse_float_list(value, 6, "amp-scale-highs"), default=None)
    parser.add_argument("--phase-lag-lows", type=lambda value: parse_float_list(value, 5, "phase-lag-lows"), default=None)
    parser.add_argument("--phase-lag-highs", type=lambda value: parse_float_list(value, 5, "phase-lag-highs"), default=None)
    parser.add_argument("--joint-bias-low", type=float, default=None)
    parser.add_argument("--joint-bias-high", type=float, default=None)
    return parser.parse_args()


def config_from_args(args) -> TurningConfig:
    cfg = TurningConfig()
    cfg.turn_direction = args.turn_direction
    cfg.target_yaw_rate = abs(float(args.target_yaw_rate))
    if args.target_radius is not None:
        cfg.target_radius = abs(float(args.target_radius))
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
    direction_sign(cfg.turn_direction)
    return cfg


def round_list(values: np.ndarray, digits: int) -> list[float]:
    return [round(float(value), digits) for value in values]


def summarize_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))]
    if not values:
        return None
    return float(np.mean(values))


def select_action(actions: np.ndarray, rewards: np.ndarray, strategy: str) -> np.ndarray:
    if strategy == "mean":
        return np.mean(actions, axis=0)
    if strategy == "last":
        return actions[-1]
    if strategy == "best-step":
        return actions[int(np.argmax(rewards))]
    raise ValueError(f"unknown export strategy: {strategy}")


def main():
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")

    cfg = config_from_args(args)
    env = EelTurningRLEnv(cfg)
    model = PPO.load(args.model, env=env)

    obs, _ = env.reset()
    collected_actions: list[np.ndarray] = []
    collected_rewards: list[float] = []
    collected_infos: list[dict[str, Any]] = []
    episodes = 1

    while len(collected_actions) < args.samples and episodes <= args.max_episodes:
        action, _ = model.predict(obs, deterministic=not args.stochastic)
        obs, reward, terminated, truncated, info = env.step(action)

        if info.get("steady_state", False):
            collected_actions.append(np.asarray(info["physical_action"], dtype=np.float64))
            collected_rewards.append(float(reward))
            collected_infos.append(dict(info))

        if terminated or truncated:
            obs, _ = env.reset()
            episodes += 1

    if not collected_actions:
        raise RuntimeError(
            "No steady-state turning actions were collected. Increase --max-episodes, "
            "shorten --warmup-seconds, or check that the model can finish rollouts."
        )
    if len(collected_actions) < args.samples:
        print(
            f"warning: collected only {len(collected_actions)} steady-state samples "
            f"before hitting --max-episodes={args.max_episodes}"
        )

    actions = np.asarray(collected_actions, dtype=np.float64)
    rewards = np.asarray(collected_rewards, dtype=np.float64)
    selected = select_action(actions, rewards, args.strategy)

    amp_scales = selected[:6]
    phase_lags = selected[6:11]
    joint_bias = selected[11:17]
    name = args.name or f"rl_turn_{args.turn_direction}"

    gait = {
        "name": name,
        "ajoint": round(float(np.degrees(cfg.fixed_ajoint)), args.round),
        "freq": round(float(cfg.fixed_frequency), args.round),
        "wavelength": round(float(cfg.fixed_wavelength), args.round),
        "amp_scales": round_list(amp_scales, args.round),
        "phase_lags": round_list(phase_lags, args.round),
        "joint_bias": round_list(joint_bias, args.round),
        "source": {
            "type": "ppo_turning_policy_export",
            "model": str(args.model),
            "strategy": args.strategy,
            "turn_direction": args.turn_direction,
            "target_yaw_rate": round(float(env.signed_target_yaw_rate), args.round),
            "target_radius": cfg.target_radius,
            "deterministic": not args.stochastic,
            "steady_state_samples": len(collected_actions),
            "episodes_used": episodes,
            "env_config": {
                key: (str(value) if key == "xml_path" else value)
                for key, value in asdict(cfg).items()
            },
            "metrics_mean": {
                "reward": round(float(np.mean(rewards)), args.round),
                "speed": round(summarize_metric(collected_infos, "speed") or 0.0, args.round),
                "yaw_rate": round(summarize_metric(collected_infos, "yaw_rate") or 0.0, args.round),
                "turn_radius": round(summarize_metric(collected_infos, "turn_radius") or 0.0, args.round),
                "energy_proxy": round(summarize_metric(collected_infos, "energy_proxy") or 0.0, args.round),
            },
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(gait, indent=2) + "\n", encoding="utf-8")

    print(f"saved turning gait JSON to {args.output}")
    print(json.dumps(gait, indent=2))


if __name__ == "__main__":
    main()
