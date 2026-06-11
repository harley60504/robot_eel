from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from hopf_cpg import degrees_to_radians
from rl_free_swim_env import EelFreeSwimRLEnv, FreeSwimConfig
from train_free_swim_rl import parse_float_list


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a trained free-swim PPO policy as a fixed Hopf CPG gait JSON."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("outputs/ppo_free_swim_shape.zip"),
        help="Path to the trained PPO .zip model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("gaits/rl_straight.json"),
        help="Output gait JSON path.",
    )
    parser.add_argument("--name", default="rl_straight", help="Name stored in the gait JSON.")
    parser.add_argument(
        "--samples",
        type=int,
        default=300,
        help="Number of steady-state policy actions to collect before export.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=20,
        help="Safety limit for rollout episodes while collecting samples.",
    )
    parser.add_argument(
        "--strategy",
        choices=("mean", "last", "best-step"),
        default="mean",
        help=(
            "How to collapse the policy actions into one fixed gait: "
            "mean averages steady-state actions, last uses the final collected action, "
            "best-step uses the action from the highest-reward steady-state step."
        ),
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy sampling instead of deterministic PPO prediction.",
    )
    parser.add_argument("--round", type=int, default=6, help="Decimal places in the exported JSON.")

    # Keep these compatible with train_free_swim_rl.py so the export env can match
    # the env used during training.
    parser.add_argument("--episode-seconds", type=float, default=None)
    parser.add_argument("--warmup-seconds", type=float, default=None)
    parser.add_argument("--freq", type=float, default=None)
    parser.add_argument("--wavelength", type=float, default=None)
    parser.add_argument("--ajoint", type=float, default=None, help="Base joint angle amplitude in degrees.")
    parser.add_argument("--amp-scale-lows", type=lambda value: parse_float_list(value, 6, "amp-scale-lows"), default=None)
    parser.add_argument("--amp-scale-highs", type=lambda value: parse_float_list(value, 6, "amp-scale-highs"), default=None)
    parser.add_argument("--phase-lag-lows", type=lambda value: parse_float_list(value, 5, "phase-lag-lows"), default=None)
    parser.add_argument("--phase-lag-highs", type=lambda value: parse_float_list(value, 5, "phase-lag-highs"), default=None)
    return parser.parse_args()


def config_from_args(args) -> FreeSwimConfig:
    cfg = FreeSwimConfig()
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
    return cfg


def round_list(values: np.ndarray, digits: int) -> list[float]:
    return [round(float(value), digits) for value in values]


def summarize_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if key in row]
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
    env = EelFreeSwimRLEnv(cfg)
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
            "No steady-state PPO actions were collected. Increase --max-episodes, "
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
    phase_lags = selected[6:]

    gait = {
        "name": args.name,
        "ajoint": round(float(np.degrees(cfg.fixed_ajoint)), args.round),
        "freq": round(float(cfg.fixed_frequency), args.round),
        "wavelength": round(float(cfg.fixed_wavelength), args.round),
        "amp_scales": round_list(amp_scales, args.round),
        "phase_lags": round_list(phase_lags, args.round),
        "joint_bias": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "source": {
            "type": "ppo_policy_export",
            "model": str(args.model),
            "strategy": args.strategy,
            "deterministic": not args.stochastic,
            "steady_state_samples": len(collected_actions),
            "episodes_used": episodes,
            "env_config": {
                key: (str(value) if key == "xml_path" else value)
                for key, value in asdict(cfg).items()
            },
            "metrics_mean": {
                "reward": round(float(np.mean(rewards)), args.round),
                "velocity_x": round(summarize_metric(collected_infos, "velocity_x") or 0.0, args.round),
                "velocity_y": round(summarize_metric(collected_infos, "velocity_y") or 0.0, args.round),
                "energy_proxy": round(summarize_metric(collected_infos, "energy_proxy") or 0.0, args.round),
            },
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(gait, indent=2) + "\n", encoding="utf-8")

    print(f"saved gait JSON to {args.output}")
    print(json.dumps(gait, indent=2))


if __name__ == "__main__":
    main()
