from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from rl_turning_env import EelTurningRLEnv, TurningConfig, direction_sign


def round_list(values: np.ndarray | list[float] | tuple[float, ...], digits: int) -> list[float]:
    return [round(float(value), digits) for value in values]


def summarize_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values: list[float] = []
    for row in rows:
        if key not in row:
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


def _filtered_mean_drop_fraction(strategy: str) -> float | None:
    if not strategy.startswith("filtered-mean-") or not strategy.endswith("%"):
        return None
    try:
        percent = float(strategy.removeprefix("filtered-mean-").removesuffix("%"))
    except ValueError:
        return None
    if percent < 0.0 or percent >= 100.0:
        raise ValueError("filtered mean drop percent must be >= 0 and < 100")
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
    drop_fraction = _filtered_mean_drop_fraction(strategy)
    if drop_fraction is not None:
        cutoff = float(np.quantile(rewards, drop_fraction))
        keep = rewards >= cutoff
        if not np.any(keep):
            return np.mean(actions, axis=0)
        return np.mean(actions[keep], axis=0)
    raise ValueError(f"unknown export strategy: {strategy}")


def selected_action_summary(rewards: np.ndarray, strategy: str) -> dict[str, Any]:
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
        drop_fraction = _filtered_mean_drop_fraction(strategy)
        if top_fraction is not None:
            count = max(1, int(np.ceil(len(rewards) * top_fraction)))
            selected_rewards = rewards[np.argsort(rewards)[-count:]]
        elif drop_fraction is not None:
            cutoff = float(np.quantile(rewards, drop_fraction))
            selected_rewards = rewards[rewards >= cutoff]
            if selected_rewards.size == 0:
                selected_rewards = rewards
            count = int(selected_rewards.size)
        else:
            raise ValueError(f"unknown export strategy: {strategy}")
    return {
        "selected_action_count": int(count),
        "selected_reward_mean": float(np.mean(selected_rewards)),
        "selected_reward_min": float(np.min(selected_rewards)),
        "selected_reward_max": float(np.max(selected_rewards)),
    }


def _split_policy_action(selected: np.ndarray, cfg: TurningConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map an exported physical action to gait arrays.

    The current turning environment is bias-only: action_dim == 6 and the fixed
    traveling wave comes from TurningConfig.  Older shape+bias policies exported
    17 values: 6 amp scales, 5 phase lags, 6 joint bias.  Supporting both keeps
    old models usable.
    """
    selected = np.asarray(selected, dtype=np.float64)
    if selected.shape[0] == 6:
        amp_scales = np.asarray(cfg.fixed_amp_scales, dtype=np.float64)
        phase_lags = np.asarray(cfg.fixed_phase_lags, dtype=np.float64)
        joint_bias = selected[:6]
        return amp_scales, phase_lags, joint_bias
    if selected.shape[0] >= 17:
        amp_scales = selected[:6]
        phase_lags = selected[6:11]
        joint_bias = selected[11:17]
        return amp_scales, phase_lags, joint_bias
    raise ValueError(
        f"unsupported policy action size {selected.shape[0]}; expected 6 for bias-only or 17 for shape+bias"
    )


def export_turning_policy_to_gait(
    model_path: Path,
    cfg: TurningConfig,
    *,
    samples: int = 300,
    max_episodes: int = 20,
    strategy: str = "best-step",
    stochastic: bool = False,
    name: str | None = None,
    digits: int = 6,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Roll out a trained PPO turning policy and return a fixed gait JSON dict.

    Returns (gait, diagnostics).  The gait can be written directly to disk and
    opened by view_gait.py or plot_fixed_gait_trajectories.py.
    """
    if samples <= 0:
        raise ValueError("samples must be positive")
    if max_episodes <= 0:
        raise ValueError("max_episodes must be positive")
    direction_sign(cfg.turn_direction)

    from stable_baselines3 import PPO

    model_path = Path(model_path).expanduser().resolve()
    env = EelTurningRLEnv(cfg)
    model = PPO.load(model_path, env=env)

    obs, _ = env.reset()
    collected_actions: list[np.ndarray] = []
    collected_rewards: list[float] = []
    collected_infos: list[dict[str, Any]] = []
    episodes = 1

    while len(collected_actions) < samples and episodes <= max_episodes:
        action, _ = model.predict(obs, deterministic=not stochastic)
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
            "No steady-state turning actions were collected. Increase max_episodes, "
            "shorten warmup_seconds, or check that the model can finish rollouts."
        )

    actions = np.asarray(collected_actions, dtype=np.float64)
    rewards = np.asarray(collected_rewards, dtype=np.float64)
    selected = select_action(actions, rewards, strategy)
    selection_summary = selected_action_summary(rewards, strategy)
    amp_scales, phase_lags, joint_bias = _split_policy_action(selected, cfg)

    gait_name = name or f"rl_turn_{cfg.turn_direction}_policy"
    metrics_mean = {
        "reward": round(float(np.mean(rewards)), digits),
        "step_reward_best": round(float(np.max(rewards)), digits),
        "speed": round(summarize_metric(collected_infos, "speed") or 0.0, digits),
        "body_speed": round(summarize_metric(collected_infos, "body_speed") or 0.0, digits),
        "yaw_rate": round(summarize_metric(collected_infos, "yaw_rate") or 0.0, digits),
        "body_yaw_rate": round(summarize_metric(collected_infos, "body_yaw_rate") or 0.0, digits),
        "turn_radius": round(summarize_metric(collected_infos, "turn_radius") or 0.0, digits),
        "signed_turn_radius": round(summarize_metric(collected_infos, "signed_turn_radius") or 0.0, digits),
        "signed_target_radius": round(summarize_metric(collected_infos, "signed_target_radius") or 0.0, digits),
        "correct_turn_direction_rate": round(summarize_metric(collected_infos, "correct_turn_direction") or 0.0, digits),
    }

    gait: dict[str, Any] = {
        "name": gait_name,
        "ajoint": round(float(np.degrees(cfg.fixed_ajoint)), digits),
        "freq": round(float(cfg.fixed_frequency), digits),
        "wavelength": round(float(cfg.fixed_wavelength), digits),
        "amp_scales": round_list(amp_scales, digits),
        "phase_lags": round_list(phase_lags, digits),
        "joint_bias": round_list(joint_bias, digits),
        "source": {
            "type": "ppo_turning_policy_export_bias_only_compatible",
            "model": str(model_path),
            "strategy": strategy,
            "strategy_selection": {
                key: round(value, digits) if isinstance(value, float) else value
                for key, value in selection_summary.items()
            },
            "turn_direction": cfg.turn_direction,
            "target_yaw_rate": round(float(env.signed_target_yaw_rate), digits),
            "target_radius": cfg.target_radius,
            "deterministic": not stochastic,
            "steady_state_samples": len(collected_actions),
            "episodes_used": episodes,
            "env_config": {
                key: (str(value) if key == "xml_path" else value) for key, value in asdict(cfg).items()
            },
            "metrics_mean": metrics_mean,
        },
    }
    diagnostics = {
        "samples_collected": len(collected_actions),
        "episodes_used": episodes,
        "strategy": strategy,
        "strategy_selection": {
            key: round(value, digits) if isinstance(value, float) else value
            for key, value in selection_summary.items()
        },
        "metrics_mean": metrics_mean,
        "selected_action_size": int(selected.shape[0]),
    }
    return gait, diagnostics


def write_gait_json(path: Path, gait: dict[str, Any]) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(gait, indent=2) + "\n", encoding="utf-8")
    return path
