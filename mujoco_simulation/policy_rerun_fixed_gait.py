from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_fitted_gait_curves import (
    add_fitted_radius_metrics,
    add_fitted_yaw_rate_metrics,
    add_sim_metric_box,
    draw_rotated_tank,
    fitted_curve,
    rotate_sim_xy,
    sim_metric_text,
    trajectory_metrics,
)
from rl_policy_exporter import write_gait_json
from rl_turning_env import EelTurningRLEnv, TurningConfig, direction_sign

#載入訓練
def rollout_policy_with_actions(model_zip: Path, cfg: TurningConfig) -> tuple[np.ndarray, float]:
    from stable_baselines3 import PPO

    env = EelTurningRLEnv(cfg)
    model = PPO.load(Path(model_zip), env=env)
    obs, _ = env.reset()
    records: list[list[float]] = []
    total_reward = 0.0

    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        physical_action = np.asarray(info.get("physical_action", []), dtype=np.float64)
        records.append(
            [
                float(env.data.time),
                float(info.get("x", np.nan)),
                float(info.get("y", np.nan)),
                float(info.get("yaw", np.nan)),
                float(reward),
                float(info.get("yaw_rate", np.nan)),
                float(info.get("turn_radius", np.nan)),
                1.0 if info.get("steady_state", False) else 0.0,
                *[float(value) for value in physical_action],
            ]
        )
        total_reward += float(reward)

    arr = np.asarray(records, dtype=np.float64)
    if arr.shape[0] < 2:
        raise RuntimeError("policy rerun produced too few points")
    if arr.shape[1] <= 8:
        raise RuntimeError("policy rerun did not record physical_action columns")
    return arr, total_reward


def steady_actions_and_rewards(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    steady = arr[:, 7] > 0.5
    actions = arr[:, 8:]
    rewards = arr[:, 4]
    if np.any(steady):
        return actions[steady], rewards[steady]
    return actions, rewards


def mean_action_gait(#把訓練後8s的參數平均變固定參數
    *,
    name: str,
    cfg: TurningConfig,
    model_zip: Path,
    arr: np.ndarray,
    policy_csv: Path,
    source_extra: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    actions, rewards = steady_actions_and_rewards(arr)
    joint_bias = np.mean(actions, axis=0)
    if joint_bias.shape[0] != 6:
        raise RuntimeError(f"mean fixed gait expects 6 bias actions, got {joint_bias.shape[0]}")

    diagnostics = {
        "steady_action_count": int(actions.shape[0]),
        "steady_reward_mean": float(np.mean(rewards)),
        "steady_reward_min": float(np.min(rewards)),
        "steady_reward_max": float(np.max(rewards)),
    }
    source: dict[str, Any] = {
        "type": "best_policy_rerun_mean_fixed_gait",
        "model": str(Path(model_zip)),
        "strategy": "policy-rerun-mean",
        "turn_direction": cfg.turn_direction,
        "target_yaw_rate": float(direction_sign(cfg.turn_direction) * abs(float(cfg.target_yaw_rate))),
        "target_radius": cfg.target_radius,
        "policy_rerun_csv": str(policy_csv),
        **diagnostics,
        "env_config": {key: (str(value) if key == "xml_path" else value) for key, value in asdict(cfg).items()},
    }
    if source_extra:
        source.update(source_extra)

    gait = {
        "name": name,
        "ajoint": float(np.degrees(cfg.fixed_ajoint)),
        "freq": float(cfg.fixed_frequency),
        "wavelength": float(cfg.fixed_wavelength),
        "amp_scales": [float(value) for value in cfg.fixed_amp_scales],
        "phase_lags": [float(value) for value in cfg.fixed_phase_lags],
        "joint_bias": [float(value) for value in joint_bias],
        "source": source,
    }
    return gait, diagnostics


def write_policy_rerun_outputs(
    *,
    name: str,
    cfg: TurningConfig,
    model_zip: Path,
    out_dir: Path,
    arr: np.ndarray,
    total_reward: float,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}_policy_rerun_trajectory.csv"
    png_path = out_dir / f"sim_{name}_policy_rerun_fitted_rotated.png"
    summary_path = out_dir / f"{name}_policy_rerun_summary.json"

    action_headers = [f"action_{idx}" for idx in range(max(0, arr.shape[1] - 8))]
    header = ",".join(["time", "x", "y", "yaw", "reward", "yaw_rate", "turn_radius", "steady_state", *action_headers])
    np.savetxt(csv_path, arr, delimiter=",", header=header, comments="")

    xy = rotate_sim_xy(arr[:, 1:3])
    curve, fit = fitted_curve(xy)
    metrics = trajectory_metrics(arr[:, :4], xy)
    target_yaw = None
    if cfg.yaw_rate_weight != 0.0:
        target_yaw = direction_sign(cfg.turn_direction) * abs(float(cfg.target_yaw_rate))
    target_radius = None if cfg.radius_weight == 0.0 else cfg.target_radius
    metrics.update(add_fitted_yaw_rate_metrics(fit, metrics, target_yaw, cfg.turn_direction))
    metrics.update(add_fitted_radius_metrics(fit, target_radius))
    metrics.update(
        {
            "turn_direction": cfg.turn_direction,
            "yaw_rate_reward_weight": float(cfg.yaw_rate_weight),
            "radius_reward_weight": float(cfg.radius_weight),
            "episode_reward": float(total_reward),
            "mean_step_reward": float(np.nanmean(arr[:, 4])),
            "mean_env_yaw_rate": float(np.nanmean(arr[:, 5])),
            "mean_env_turn_radius": float(np.nanmean(arr[:, 6])),
        }
    )

    fig, ax = plt.subplots(figsize=(4.8, 8.0), dpi=170)
    draw_rotated_tank(ax)
    color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
    ax.plot(curve[:, 0], curve[:, 1], color=color, linewidth=3.0)
    ax.scatter([xy[0, 0]], [xy[0, 1]], s=34, color=color, edgecolor="black", zorder=4)
    ax.scatter([xy[-1, 0]], [xy[-1, 1]], s=52, marker="x", color=color, linewidth=2.2, zorder=4)
    ax.set_title(f"{name} policy rerun")
    add_sim_metric_box(ax, sim_metric_text(name, fit, metrics))
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)

    summary = {
        "name": name,
        "model_zip": str(model_zip),
        "trajectory_csv": str(csv_path),
        "policy_rerun_png": str(png_path),
        "deterministic": True,
        **fit,
        **metrics,
        "env_config": {key: (str(value) if key == "xml_path" else value) for key, value in asdict(cfg).items()},
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "policy_rerun_csv": str(csv_path),
        "policy_rerun_png": str(png_path),
        "policy_rerun_summary": str(summary_path),
    }


def write_mean_fixed_gait_from_best_policy(#寫josn檔
    *,
    name: str,
    cfg: TurningConfig,
    model_zip: Path,
    gait_path: Path,
    policy_out_dir: Path,
    source_extra: dict[str, Any] | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    arr, total_reward = rollout_policy_with_actions(model_zip, cfg)
    policy_outputs = write_policy_rerun_outputs(
        name=name,
        cfg=cfg,
        model_zip=model_zip,
        out_dir=policy_out_dir,
        arr=arr,
        total_reward=total_reward,
    )
    gait, diagnostics = mean_action_gait(
        name=name,
        cfg=cfg,
        model_zip=model_zip,
        arr=arr,
        policy_csv=Path(policy_outputs["policy_rerun_csv"]),
        source_extra=source_extra,
    )
    write_gait_json(gait_path, gait)
    return {"gait_json": str(gait_path), **policy_outputs}, diagnostics
