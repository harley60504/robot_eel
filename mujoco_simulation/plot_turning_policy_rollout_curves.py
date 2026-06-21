from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from hopf_cpg import degrees_to_radians
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
from rl_turning_env import EelTurningRLEnv, TurningConfig, direction_sign


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "csv_png" / "policy_rollout_curves"


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def cfg_from_gait_json(gait_path: Path) -> TurningConfig:
    gait = json.loads(Path(gait_path).read_text(encoding="utf-8"))
    source = gait.get("source") if isinstance(gait.get("source"), dict) else {}
    env_config = source.get("env_config") if isinstance(source.get("env_config"), dict) else {}
    cfg = TurningConfig()
    cfg.turn_direction = str(source.get("turn_direction") or env_config.get("turn_direction") or cfg.turn_direction)
    target_yaw = safe_float(source.get("target_yaw_rate"))
    if target_yaw is not None:
        cfg.target_yaw_rate = abs(target_yaw)
    target_radius = safe_float(source.get("target_radius"))
    if target_radius is not None:
        cfg.target_radius = abs(target_radius)
    for attr in (
        "episode_seconds",
        "warmup_seconds",
        "control_dt",
        "fixed_frequency",
        "fixed_wavelength",
        "yaw_rate_weight",
        "radius_weight",
        "boundary_x_min",
        "boundary_x_max",
        "boundary_y",
        "joint_bias_low",
        "joint_bias_high",
        "tail_amp_multiplier_low",
        "tail_amp_multiplier_high",
    ):
        value = safe_float(env_config.get(attr))
        if value is not None:
            setattr(cfg, attr, value)
    action_mode = env_config.get("action_mode")
    if isinstance(action_mode, str) and action_mode:
        cfg.action_mode = action_mode
    ajoint = safe_float(env_config.get("fixed_ajoint"))
    if ajoint is not None:
        cfg.fixed_ajoint = ajoint
    elif safe_float(gait.get("ajoint")) is not None:
        cfg.fixed_ajoint = degrees_to_radians(float(gait["ajoint"]))
    if isinstance(env_config.get("fixed_amp_scales"), list):
        cfg.fixed_amp_scales = tuple(float(v) for v in env_config["fixed_amp_scales"])
    if isinstance(env_config.get("fixed_phase_lags"), list):
        cfg.fixed_phase_lags = tuple(float(v) for v in env_config["fixed_phase_lags"])
    return cfg


def cfg_from_summary_row(row: dict) -> TurningConfig:
    gait_json = Path(row["gait_json"])
    return cfg_from_gait_json(gait_json)


def rollout_policy(model_zip: Path, cfg: TurningConfig, deterministic: bool = True) -> tuple[np.ndarray, list[dict]]:
    env = EelTurningRLEnv(cfg)
    model = PPO.load(Path(model_zip), env=env)
    obs, _ = env.reset()
    records: list[list[float]] = []
    infos: list[dict] = []
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        records.append(
            [
                float(env.data.time),
                float(info.get("x", np.nan)),
                float(info.get("y", np.nan)),
                float(info.get("yaw", np.nan)),
                float(reward),
                float(info.get("yaw_rate", np.nan)),
                float(info.get("turn_radius", np.nan)),
            ]
        )
        infos.append(info)
    return np.asarray(records, dtype=np.float64), infos


def write_rollout_outputs(
    *,
    name: str,
    model_zip: Path,
    cfg: TurningConfig,
    out_dir: Path,
    deterministic: bool,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    arr, infos = rollout_policy(model_zip, cfg, deterministic=deterministic)
    if arr.shape[0] < 2:
        raise RuntimeError(f"{name} policy rollout produced too few points")

    csv_path = out_dir / f"{name}_policy_rollout.csv"
    np.savetxt(
        csv_path,
        arr,
        delimiter=",",
        header="time,x,y,yaw,reward,yaw_rate,turn_radius",
        comments="",
    )

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
            "yaw_rate_reward_weight": cfg.yaw_rate_weight,
            "radius_reward_weight": cfg.radius_weight,
            "mean_step_reward": float(np.nanmean(arr[:, 4])),
            "episode_reward": float(np.nansum(arr[:, 4])),
            "mean_env_yaw_rate": float(np.nanmean(arr[:, 5])),
            "mean_env_turn_radius": float(np.nanmean(arr[:, 6])),
            "mean_body_yaw_rate": float(np.nanmean([info.get("body_yaw_rate", np.nan) for info in infos])),
            "mean_body_speed": float(np.nanmean([info.get("body_speed", np.nan) for info in infos])),
        }
    )

    fig, ax = plt.subplots(figsize=(4.8, 8.0), dpi=170)
    draw_rotated_tank(ax)
    color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
    ax.plot(curve[:, 0], curve[:, 1], color=color, linewidth=3.0)
    ax.scatter([xy[0, 0]], [xy[0, 1]], s=34, color=color, edgecolor="black", zorder=4)
    ax.scatter([xy[-1, 0]], [xy[-1, 1]], s=52, marker="x", color=color, linewidth=2.2, zorder=4)
    ax.set_title(f"{name} policy rollout")
    add_sim_metric_box(ax, sim_metric_text(name, fit, metrics))
    fig.tight_layout()
    png_path = out_dir / f"sim_{name}_policy_rollout_fitted_rotated.png"
    fig.savefig(png_path)
    plt.close(fig)

    summary = {
        "name": name,
        "model_zip": str(model_zip),
        "trajectory_csv": str(csv_path),
        "policy_rollout_png": str(png_path),
        "deterministic": deterministic,
        **fit,
        **metrics,
        "env_config": {key: str(value) if key == "xml_path" else value for key, value in asdict(cfg).items()},
    }
    summary_path = out_dir / f"{name}_policy_rollout_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def rows_from_summary_csv(path: Path, limit: int | None) -> list[dict]:
    rows = [row for row in csv.DictReader(Path(path).open(encoding="utf-8")) if row.get("status") == "done"]
    return rows[:limit] if limit is not None else rows


def best_model_zip_from_eval_dir(row: dict) -> Path | None:
    eval_reward_png = row.get("eval_reward_png")
    if not eval_reward_png:
        return None
    eval_dir = Path(eval_reward_png).with_suffix("")
    if eval_dir.name.endswith("_eval_reward"):
        eval_dir = eval_dir.with_name(eval_dir.name[: -len("_eval_reward")] + "_eval")
    best_zip = eval_dir / "best_model" / "best_model.zip"
    return best_zip if best_zip.exists() else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PPO policy rollout curves, before fixed-gait export averaging.")
    parser.add_argument("--summary-csv", type=Path, default=None, help="Batch summary CSV with model_zip/gait_json rows.")
    parser.add_argument("--model-zip", type=Path, default=None, help="One PPO .zip to plot.")
    parser.add_argument("--gait-json", type=Path, default=None, help="Gait JSON from the same run, used for config metadata.")
    parser.add_argument("--name", default=None)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--use-best-model", action="store_true", help="Use EvalCallback best_model.zip when available.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    rows: list[dict]
    if args.summary_csv is not None:
        rows = rows_from_summary_csv(args.summary_csv, args.limit)
    elif args.model_zip is not None and args.gait_json is not None:
        rows = [
            {
                "name": args.name or Path(args.model_zip).stem,
                "model_zip": str(args.model_zip),
                "gait_json": str(args.gait_json),
            }
        ]
    else:
        raise SystemExit("Provide --summary-csv or both --model-zip and --gait-json")

    summaries = []
    for row in rows:
        cfg = cfg_from_summary_row(row)
        model_zip = Path(row["model_zip"])
        model_source = "final"
        if args.use_best_model:
            best_zip = best_model_zip_from_eval_dir(row)
            if best_zip is not None:
                model_zip = best_zip
                model_source = "best_eval"
        summary = write_rollout_outputs(
            name=row.get("name") or model_zip.stem,
            model_zip=model_zip,
            cfg=cfg,
            out_dir=out_dir,
            deterministic=not args.stochastic,
        )
        summary["model_source"] = model_source
        summaries.append(summary)
        fitted = summary.get("fitted_yaw_rate_rad_s")
        target = summary.get("target_yaw_rate_rad_s")
        print(f"{summary['name']}: policy_yaw_fit={fitted} target={target} png={summary['policy_rollout_png']}")

    (out_dir / "policy_rollout_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
