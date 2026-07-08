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
    parser.add_argument("--model", type=Path, default=Path("outputs/zips/ppo_turn_left_shape_bias.zip"))
    parser.add_argument("--output", type=Path, default=Path("outputs/json/rl_gaits/rl_turn_left.json"))
    parser.add_argument("--name", default=None, help="Name stored in the gait JSON. Default derives from turn direction.")
    parser.add_argument("--turn-direction", choices=("left", "right"), default="left")
    parser.add_argument("--target-yaw-rate", type=float, default=0.45, help="Target absolute yaw rate in rad/s.")
    parser.add_argument("--target-radius", type=float, default=None, help="Optional target absolute turn radius in meters.")
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--max-episodes", type=int, default=20)
    parser.add_argument(
        "--strategy",
        choices=("mean", "last", "best-step", "top-5%", "top-10%", "top-20%"),
        default="mean",
    )
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--round", type=int, default=6)

    parser.add_argument("--episode-seconds", type=float, default=None)
    parser.add_argument("--warmup-seconds", type=float, default=None)
    parser.add_argument("--freq", type=float, default=None)
    parser.add_argument("--wavelength", type=float, default=None)
    parser.add_argument("--ajoint", type=float, default=None, help="Base joint angle amplitude in degrees.")
    parser.add_argument("--action-mode", choices=("bias_only", "bias_tail2_amp", "bias_tail3_amp"), default=None)
    parser.add_argument("--fixed-amp-scales", type=lambda value: parse_float_list(value, 6, "fixed-amp-scales"), default=None)
    parser.add_argument("--amp-scale-lows", type=lambda value: parse_float_list(value, 6, "amp-scale-lows"), default=None)
    parser.add_argument("--amp-scale-highs", type=lambda value: parse_float_list(value, 6, "amp-scale-highs"), default=None)
    parser.add_argument("--phase-lag-lows", type=lambda value: parse_float_list(value, 5, "phase-lag-lows"), default=None)
    parser.add_argument("--phase-lag-highs", type=lambda value: parse_float_list(value, 5, "phase-lag-highs"), default=None)
    parser.add_argument("--joint-bias-low", type=float, default=None)
    parser.add_argument("--joint-bias-high", type=float, default=None)
    parser.add_argument("--tail-amp-multiplier-low", type=float, default=None)
    parser.add_argument("--tail-amp-multiplier-high", type=float, default=None)
    parser.add_argument("--boundary-x-min", type=float, default=None)
    parser.add_argument("--boundary-x-max", type=float, default=None)
    parser.add_argument("--boundary-y", type=float, default=None)
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
    if args.action_mode is not None:
        cfg.action_mode = args.action_mode
    if args.fixed_amp_scales is not None:
        cfg.fixed_amp_scales = tuple(args.fixed_amp_scales)
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
    if args.tail_amp_multiplier_low is not None:
        cfg.tail_amp_multiplier_low = args.tail_amp_multiplier_low
    if args.tail_amp_multiplier_high is not None:
        cfg.tail_amp_multiplier_high = args.tail_amp_multiplier_high
    if args.boundary_x_min is not None:
        cfg.boundary_x_min = args.boundary_x_min
    if args.boundary_x_max is not None:
        cfg.boundary_x_max = args.boundary_x_max
    if args.boundary_y is not None:
        cfg.boundary_y = abs(args.boundary_y)
    direction_sign(cfg.turn_direction)
    return cfg


def round_list(values: np.ndarray, digits: int) -> list[float]:
    return [round(float(value), digits) for value in values]


def summarize_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = []
    for row in rows:
        if key not in row or row[key] is None:
            continue
        try:
            value = float(row[key])
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    if not values:
        return None
    return float(np.mean(values))


def _top_k_fraction(strategy: str) -> float | None:
    if not strategy.startswith("top-") or not strategy.endswith("%"):
        return None
    try:
        percent = float(strategy.removeprefix("top-").removesuffix("%"))
    except ValueError:
        return None
    if percent <= 0.0 or percent > 100.0:
        raise ValueError("top-k mean strategy percent must be > 0 and <= 100")
    return percent / 100.0


def select_action(actions: np.ndarray, rewards: np.ndarray, strategy: str) -> np.ndarray:
    if strategy == "mean":
        return np.mean(actions, axis=0)
    if strategy == "last":
        return actions[-1]
    if strategy == "best-step":
        return actions[int(np.argmax(rewards))]
    top_fraction = _top_k_fraction(strategy)
    if top_fraction is not None:
        count = max(1, int(np.ceil(len(rewards) * top_fraction)))
        top_indices = np.argsort(rewards)[-count:]
        return np.mean(actions[top_indices], axis=0)
    raise ValueError(f"unknown export strategy: {strategy}")


def selected_action_summary(rewards: np.ndarray, strategy: str) -> dict[str, float | int]:
    if strategy == "mean":
        count = len(rewards)
        selected_rewards = rewards
    elif strategy == "last":
        count = 1
        selected_rewards = rewards[-1:]
    elif strategy == "best-step":
        count = 1
        selected_rewards = rewards[[int(np.argmax(rewards))]]
    else:
        top_fraction = _top_k_fraction(strategy)
        if top_fraction is None:
            raise ValueError(f"unknown export strategy: {strategy}")
        count = max(1, int(np.ceil(len(rewards) * top_fraction)))
        selected_rewards = rewards[np.argsort(rewards)[-count:]]
    return {
        "selected_action_count": int(count),
        "selected_reward_mean": float(np.mean(selected_rewards)),
        "selected_reward_min": float(np.min(selected_rewards)),
        "selected_reward_max": float(np.max(selected_rewards)),
    }


def split_policy_action(selected: np.ndarray, cfg: TurningConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = np.asarray(selected, dtype=np.float64)
    if selected.shape[0] == 6:
        return (
            np.asarray(cfg.fixed_amp_scales, dtype=np.float64),
            np.asarray(cfg.fixed_phase_lags, dtype=np.float64),
            selected[:6],
        )
    if cfg.action_mode in {"bias_tail2_amp", "bias_tail3_amp"} and selected.shape[0] in {8, 9}:
        amp_scales = np.asarray(cfg.fixed_amp_scales, dtype=np.float64).copy()
        tail_indices = (4, 5) if cfg.action_mode == "bias_tail2_amp" else (3, 4, 5)
        if selected.shape[0] != 6 + len(tail_indices):
            raise ValueError(
                f"action_mode {cfg.action_mode} expects action size {6 + len(tail_indices)}, "
                f"got {selected.shape[0]}"
            )
        for offset, joint_index in enumerate(tail_indices):
            amp_scales[joint_index] *= selected[6 + offset]
        return (
            amp_scales,
            np.asarray(cfg.fixed_phase_lags, dtype=np.float64),
            selected[:6],
        )
    if selected.shape[0] >= 17:
        return selected[:6], selected[6:11], selected[11:17]
    raise ValueError(
        f"unsupported policy action size {selected.shape[0]}; expected 6 for bias-only or 17 for shape+bias"
    )


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
    selection_summary = selected_action_summary(rewards, args.strategy)

    amp_scales, phase_lags, joint_bias = split_policy_action(selected, cfg)
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
            "strategy_selection": {
                key: round(value, args.round) if isinstance(value, float) else value
                for key, value in selection_summary.items()
            },
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
                "step_reward_best": round(float(np.max(rewards)), args.round),
                "speed": round(summarize_metric(collected_infos, "speed") or 0.0, args.round),
                "body_speed": round(summarize_metric(collected_infos, "body_speed") or 0.0, args.round),
                "yaw_rate": round(summarize_metric(collected_infos, "yaw_rate") or 0.0, args.round),
                "body_yaw_rate": round(summarize_metric(collected_infos, "body_yaw_rate") or 0.0, args.round),
                "turn_radius": round(summarize_metric(collected_infos, "turn_radius") or 0.0, args.round),
                "signed_turn_radius": round(summarize_metric(collected_infos, "signed_turn_radius") or 0.0, args.round),
                "signed_target_radius": round(summarize_metric(collected_infos, "signed_target_radius") or 0.0, args.round),
                "correct_turn_direction_rate": round(summarize_metric(collected_infos, "correct_turn_direction") or 0.0, args.round),
            },
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(gait, indent=2) + "\n", encoding="utf-8")

    print(f"saved turning gait JSON to {args.output}")
    print(json.dumps(gait, indent=2))


if __name__ == "__main__":
    main()
