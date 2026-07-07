from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from hopf_cpg import degrees_to_radians
from plot_fitted_gait_curves import add_sim_metric_box, draw_rotated_tank, rotate_sim_xy, sim_metric_text, trajectory_metrics
from plot_fixed_gait_trajectories import build_trajectory_summary, draw_environment, plot_one, run_gait, summarize
from plot_paper_reward_summary import plot_groups
from rl_free_swim_env import EelFreeSwimRLEnv, FreeSwimConfig
from rl_training_plots import default_eval_log_dir, default_plot_path
from sim_config import DEFAULT_START_X, DEFAULT_START_Y, EEL_MODEL_XML
from train_free_swim_rl import config_from_args, fit_straight_line_curve, parse_float_list


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
ZIP_DIR = OUTPUTS / "zips"
JSON_DIR = OUTPUTS / "json" / "rl_gaits"
CSV_PNG_DIR = OUTPUTS / "csv_png"
POLICY_RERUN_DIR = CSV_PNG_DIR / "policy_rerun_best_once"
TRAJ_DIR = CSV_PNG_DIR / "fixed_gait_trajectories_mean"
FIT_DIR = CSV_PNG_DIR / "fitted_curve_comparison_mean"
LOG_DIR = OUTPUTS / "batch_logs"
SUMMARY_CSV = LOG_DIR / "free_swim_paper10_summary.csv"
STATUS_PATH = LOG_DIR / "free_swim_paper10_status.json"
PAPER_REWARD_DIR = OUTPUTS / "csv_png" / "paper10_free_swim_reward_summary"
ORGANIZED_DIR = OUTPUTS / "paper_free_swim_10_extracted_figures"

PREFIX = "paper10_ppo_free_swim_freq_phase"
RUNS = 10
TIMESTEPS = 200_000
EVAL_FREQ = 5_000
EVAL_EPISODES = 5

SUMMARY_FIELDS = [
    "name",
    "run_idx",
    "status",
    "started_at",
    "finished_at",
    "model_zip",
    "best_model_zip",
    "eval_reward_csv",
    "eval_reward_png",
    "eval_debug_csv",
    "eval_best_policy_png",
    "eval_best_policy_trajectory_csv",
    "eval_best_policy_summary",
    "gait_json",
    "policy_rerun_csv",
    "policy_rerun_png",
    "policy_rerun_summary",
    "trajectory_csv",
    "trajectory_png",
    "trajectory_summary",
    "fitted_png",
    "fitted_summary",
    "export_diagnostics",
    "eval_log_dir",
    "train_log",
    "eval_summary_json",
    "mean_episode_reward",
    "mean_vx",
    "mean_vy",
    "mean_abs_y",
    "mean_abs_yaw",
    "mean_energy_proxy",
    "mean_frequency",
    "mean_phase_lag_1",
    "mean_phase_lag_2",
    "mean_phase_lag_3",
    "mean_phase_lag_4",
    "mean_phase_lag_5",
    "error",
]


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text).strip("_")


def configure_output_names(prefix: str, organized_dir: Path | None = None) -> None:
    global PREFIX, SUMMARY_CSV, STATUS_PATH, PAPER_REWARD_DIR, ORGANIZED_DIR
    PREFIX = safe_name(prefix)
    tag = "paper10"
    if PREFIX.startswith("paper40"):
        tag = "paper40"
    elif PREFIX.startswith("paper"):
        tag = PREFIX.split("_", 1)[0]
    else:
        tag = PREFIX
    SUMMARY_CSV = LOG_DIR / f"free_swim_{tag}_summary.csv"
    STATUS_PATH = LOG_DIR / f"free_swim_{tag}_status.json"
    PAPER_REWARD_DIR = OUTPUTS / "csv_png" / f"{tag}_free_swim_reward_summary"
    ORGANIZED_DIR = organized_dir or (OUTPUTS / f"paper_free_swim_{tag.removeprefix('paper')}_extracted_figures")


def write_status(payload: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_summary(row: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    exists = SUMMARY_CSV.exists()
    with SUMMARY_CSV.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def safe_child(path: Path, parent: Path) -> Path:
    resolved = path.resolve()
    parent_resolved = parent.resolve()
    if resolved != parent_resolved and parent_resolved not in resolved.parents:
        raise ValueError(f"{resolved} is outside {parent_resolved}")
    return resolved


def organize_outputs(*, move: bool = False) -> list[dict[str, str | int]]:
    out_dir = safe_child(ORGANIZED_DIR, OUTPUTS)
    out_dir.mkdir(parents=True, exist_ok=True)
    action = shutil.move if move else shutil.copy2
    categories: list[tuple[str, Path, tuple[str, ...]]] = [
        ("average_reward_summary", PAPER_REWARD_DIR, ("*",)),
        ("batch_logs_and_status", LOG_DIR, (SUMMARY_CSV.name, STATUS_PATH.name, f"{PREFIX}_*.train.log", f"{PREFIX}_*.eval_summary.json")),
        ("gait_jsons", OUTPUTS / "json" / "rl_gaits", (f"{PREFIX}_*.json",)),
        ("individual_rewards", OUTPUTS / "csv_png", (f"{PREFIX}_*_eval_reward.png", f"{PREFIX}_*_eval_reward.csv", f"{PREFIX}_*_eval_debug.csv")),
        ("fixed_gait_trajectories_after_mean", OUTPUTS / "csv_png" / "fixed_gait_trajectories_mean", (f"{PREFIX}_*_trajectory.png", f"{PREFIX}_*_trajectory.csv", f"{PREFIX}_*_summary.json")),
        (
            "training_eval_best_policy_curves",
            OUTPUTS / "csv_png" / "eval_best_policy_curves",
            (
                f"sim_{PREFIX}_*_eval_best_policy_fitted_rotated.png",
                f"{PREFIX}_*_eval_best_policy_trajectory.csv",
                f"{PREFIX}_*_eval_best_policy_summary.json",
            ),
        ),
        ("policy_rerun_before_fixed_mean", OUTPUTS / "csv_png" / "policy_rerun_best_once", (f"sim_{PREFIX}_*_policy_rerun_fitted_rotated.png", f"{PREFIX}_*_policy_rerun_trajectory.csv", f"{PREFIX}_*_policy_rerun_summary.json")),
        ("fixed_gait_fitted_rotated_after_mean", OUTPUTS / "csv_png" / "fitted_curve_comparison_mean", (f"sim_{PREFIX}_*_fitted_rotated.png", f"{PREFIX}_*_fitted_summary.json")),
        ("swim_recording_metadata", OUTPUTS / "paper_swim_recordings", (f"{PREFIX}_*_swim_18s_summary.json", f"{PREFIX}_*_swim_18s_snapshots.csv")),
    ]

    rows = []
    for category, source, patterns in categories:
        dest = safe_child(out_dir / category, out_dir)
        dest.mkdir(parents=True, exist_ok=True)
        copied = 0
        if source.exists():
            for pattern in patterns:
                for path in sorted(source.glob(pattern)):
                    if not path.is_file():
                        continue
                    target = dest / path.name
                    if target.exists():
                        target.unlink()
                    action(str(path), str(target))
                    copied += 1
        rows.append({"category": category, "files": copied, "path": str(dest)})

    with (out_dir / "organized_counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["category", "files", "path"])
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "last_organize_operations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["category", "files", "path"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def build_train_cmd(args: argparse.Namespace, name: str, output_base: Path, eval_log_dir: Path, plot_output: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "train_free_swim_rl.py"),
        "--timesteps",
        str(args.timesteps),
        "--eval-freq",
        str(args.eval_freq),
        "--eval-episodes",
        str(args.eval_episodes),
        "--output",
        str(output_base),
        "--eval-log-dir",
        str(eval_log_dir),
        "--plot-output",
        str(plot_output),
    ]
    if args.no_plot:
        cmd.append("--no-plot")
    passthrough = [
        ("episode_seconds", "--episode-seconds"),
        ("warmup_seconds", "--warmup-seconds"),
        ("freq", "--freq"),
        ("freq_low", "--freq-low"),
        ("freq_high", "--freq-high"),
        ("wavelength", "--wavelength"),
        ("ajoint", "--ajoint"),
        ("start_x", "--start-x"),
        ("start_y", "--start-y"),
        ("phase_lag_low", "--phase-lag-low"),
        ("phase_lag_high", "--phase-lag-high"),
        ("target_speed", "--target-speed"),
        ("speed_error_weight", "--speed-error-weight"),
        ("energy_weight", "--energy-weight"),
        ("reward_average_seconds", "--reward-average-seconds"),
        ("boundary_x_min", "--boundary-x-min"),
        ("boundary_x_max", "--boundary-x-max"),
        ("boundary_y", "--boundary-y"),
    ]
    for attr, flag in passthrough:
        value = getattr(args, attr)
        if value is not None:
            cmd.extend([flag, str(value)])
    if args.fixed_amp_scales is not None:
        cmd.extend(["--fixed-amp-scales", ",".join(str(value) for value in args.fixed_amp_scales)])
    return cmd


def rollout_policy_with_actions(model_path: Path, cfg: FreeSwimConfig) -> tuple[np.ndarray, float]:
    env = EelFreeSwimRLEnv(cfg)
    model = PPO.load(model_path, env=env)
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
                float(info.get("velocity_x", np.nan)),
                float(info.get("velocity_y", np.nan)),
                1.0 if info.get("steady_state", False) else 0.0,
                *[float(value) for value in physical_action],
            ]
        )
        total_reward += float(reward)
    arr = np.asarray(records, dtype=np.float64)
    if arr.shape[0] < 2 or arr.shape[1] < 14:
        raise RuntimeError("free-swim policy rerun produced too few rows or action columns")
    return arr, total_reward


def steady_actions_and_rewards(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    steady = arr[:, 7] > 0.5
    actions = arr[:, 8:]
    rewards = arr[:, 4]
    if np.any(steady):
        return actions[steady], rewards[steady]
    return actions, rewards


def write_policy_rerun_outputs(name: str, cfg: FreeSwimConfig, model_path: Path, arr: np.ndarray, total_reward: float) -> dict:
    POLICY_RERUN_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = POLICY_RERUN_DIR / f"{name}_policy_rerun_trajectory.csv"
    png_path = POLICY_RERUN_DIR / f"sim_{name}_policy_rerun_fitted_rotated.png"
    summary_path = POLICY_RERUN_DIR / f"{name}_policy_rerun_summary.json"
    header = ",".join(
        [
            "time",
            "x",
            "y",
            "yaw",
            "reward",
            "velocity_x",
            "velocity_y",
            "steady_state",
            "frequency",
            "phase_lag_1",
            "phase_lag_2",
            "phase_lag_3",
            "phase_lag_4",
            "phase_lag_5",
        ]
    )
    np.savetxt(csv_path, arr, delimiter=",", header=header, comments="")

    xy = rotate_sim_xy(arr[:, 1:3])
    curve, fit = fit_straight_line_curve(xy)
    metrics = trajectory_metrics(arr[:, :4], xy)
    metrics.update(
        {
            "episode_reward": float(total_reward),
            "mean_step_reward": float(np.nanmean(arr[:, 4])),
            "mean_env_velocity_x": float(np.nanmean(arr[:, 5])),
            "mean_env_velocity_y": float(np.nanmean(arr[:, 6])),
        }
    )

    fig, ax = plt.subplots(figsize=(4.8, 8.0), dpi=170)
    draw_rotated_tank(ax)
    color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
    ax.plot(curve[:, 0], curve[:, 1], color=color, linewidth=3.0)
    ax.scatter([xy[0, 0]], [xy[0, 1]], s=34, color=color, edgecolor="black", zorder=4)
    ax.scatter([xy[-1, 0]], [xy[-1, 1]], s=52, marker="x", color=color, linewidth=2.2, zorder=4)
    ax.set_title(f"{name} policy rerun")
    add_sim_metric_box(ax, sim_metric_text("straight", fit, metrics))
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)

    summary = {
        "name": name,
        "model_zip": str(model_path),
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


def write_mean_gait_json(name: str, cfg: FreeSwimConfig, model_path: Path, arr: np.ndarray, policy_csv: Path) -> tuple[Path, dict]:
    actions, rewards = steady_actions_and_rewards(arr)
    mean_action = np.mean(actions, axis=0)
    gait_path = JSON_DIR / f"{name}.json"
    diagnostics = {
        "steady_action_count": int(actions.shape[0]),
        "steady_reward_mean": float(np.mean(rewards)),
        "steady_reward_min": float(np.min(rewards)),
        "steady_reward_max": float(np.max(rewards)),
    }
    gait = {
        "name": name,
        "ajoint": float(np.degrees(cfg.fixed_ajoint)),
        "freq": float(mean_action[0]),
        "wavelength": float(cfg.fixed_wavelength),
        "amp_scales": [float(value) for value in cfg.fixed_amp_scales],
        "phase_lags": [float(value) for value in mean_action[1:6]],
        "joint_bias": [0.0] * 6,
        "source": {
            "type": "free_swim_best_policy_rerun_mean_fixed_gait",
            "model": str(model_path),
            "strategy": "policy-rerun-mean",
            "policy_rerun_csv": str(policy_csv),
            **diagnostics,
            "env_config": {key: (str(value) if key == "xml_path" else value) for key, value in asdict(cfg).items()},
        },
    }
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    gait_path.write_text(json.dumps(gait, indent=2) + "\n", encoding="utf-8")
    return gait_path, diagnostics


def write_fixed_gait_trajectory(name: str, gait_path: Path, cfg: FreeSwimConfig) -> dict:
    TRAJ_DIR.mkdir(parents=True, exist_ok=True)
    gait, arr, hit_wall = run_gait(Path(EEL_MODEL_XML), gait_path, 30.0, cfg.start_x, cfg.start_y)
    if arr.shape[0] < 2:
        raise RuntimeError(f"{name} fixed gait trajectory has too few points")
    summary = summarize(arr, 0.0)
    csv_path = TRAJ_DIR / f"{name}_trajectory.csv"
    png_path = TRAJ_DIR / f"{name}_trajectory.png"
    summary_path = TRAJ_DIR / f"{name}_summary.json"
    np.savetxt(csv_path, arr, delimiter=",", header="time,x,y,yaw", comments="")

    fig, ax = plt.subplots(figsize=(7, 5), dpi=170)
    draw_environment(ax, cfg.start_x, cfg.start_y)
    plot_one(ax, name, arr, summary)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"{name} fixed gait trajectory")
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)

    row = build_trajectory_summary(name, gait, gait_path, arr, summary, hit_wall)
    row["trajectory_csv"] = str(csv_path)
    row["trajectory_png"] = str(png_path)
    summary_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    return {
        "trajectory_csv": str(csv_path),
        "trajectory_png": str(png_path),
        "trajectory_summary": str(summary_path),
    }


def write_fixed_gait_fitted(name: str, trajectory_csv: Path) -> dict:
    FIT_DIR.mkdir(parents=True, exist_ok=True)
    arr = np.loadtxt(trajectory_csv, delimiter=",", skiprows=1)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    xy = rotate_sim_xy(arr[:, 1:3])
    curve, fit = fit_straight_line_curve(xy)
    metrics = trajectory_metrics(arr[:, :4], xy)
    fig, ax = plt.subplots(figsize=(4.8, 8.0), dpi=170)
    draw_rotated_tank(ax)
    color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
    ax.plot(curve[:, 0], curve[:, 1], color=color, linewidth=3.0)
    ax.scatter([xy[0, 0]], [xy[0, 1]], s=34, color=color, edgecolor="black", zorder=4)
    ax.scatter([xy[-1, 0]], [xy[-1, 1]], s=52, marker="x", color=color, linewidth=2.2, zorder=4)
    ax.set_title(f"{name} fitted curve")
    add_sim_metric_box(ax, sim_metric_text("straight", fit, metrics))
    fig.tight_layout()
    png_path = FIT_DIR / f"sim_{name}_fitted_rotated.png"
    fig.savefig(png_path)
    plt.close(fig)
    summary_path = FIT_DIR / f"{name}_fitted_summary.json"
    summary_path.write_text(json.dumps({"name": name, **fit, **metrics, "fit_png": str(png_path)}, indent=2), encoding="utf-8")
    return {"fitted_png": str(png_path), "fitted_summary": str(summary_path)}


def evaluate_policy(model_path: Path, cfg: FreeSwimConfig, episodes: int) -> dict:
    env = EelFreeSwimRLEnv(cfg)
    model = PPO.load(model_path, env=env)
    rewards: list[float] = []
    vx_values: list[float] = []
    vy_values: list[float] = []
    y_values: list[float] = []
    yaw_values: list[float] = []
    energy_values: list[float] = []
    actions: list[np.ndarray] = []

    for _ in range(episodes):
        obs, _ = env.reset()
        terminated = truncated = False
        total_reward = 0.0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            if info.get("steady_state"):
                vx_values.append(float(info["velocity_x"]))
                vy_values.append(float(info["velocity_y"]))
                y_values.append(float(info["y"]))
                yaw_values.append(float(info["yaw"]))
                energy_values.append(float(info["energy_proxy"]))
                actions.append(np.asarray(info["physical_action"], dtype=np.float64))
        rewards.append(total_reward)

    actions_arr = np.asarray(actions, dtype=np.float64)
    mean_action = np.mean(actions_arr, axis=0) if actions_arr.size else np.full(6, np.nan)
    return {
        "mean_episode_reward": float(np.mean(rewards)) if rewards else float("nan"),
        "mean_vx": float(np.mean(vx_values)) if vx_values else float("nan"),
        "mean_vy": float(np.mean(vy_values)) if vy_values else float("nan"),
        "mean_abs_y": float(np.mean(np.abs(y_values))) if y_values else float("nan"),
        "mean_abs_yaw": float(np.mean(np.abs(yaw_values))) if yaw_values else float("nan"),
        "mean_energy_proxy": float(np.mean(energy_values)) if energy_values else float("nan"),
        "mean_frequency": float(mean_action[0]),
        "mean_phase_lag_1": float(mean_action[1]),
        "mean_phase_lag_2": float(mean_action[2]),
        "mean_phase_lag_3": float(mean_action[3]),
        "mean_phase_lag_4": float(mean_action[4]),
        "mean_phase_lag_5": float(mean_action[5]),
    }


def run_one(args: argparse.Namespace, run_idx: int, cfg: FreeSwimConfig) -> None:
    name = f"{PREFIX}_run{run_idx:02d}"
    output_base = ZIP_DIR / name
    eval_log_dir = default_eval_log_dir(output_base)
    plot_output = default_plot_path(output_base)
    train_log = LOG_DIR / f"{name}.train.log"
    started_at = now_text()
    row = {
        "name": name,
        "run_idx": run_idx,
        "status": "running",
        "started_at": started_at,
        "model_zip": str(output_base.with_suffix(".zip")),
        "best_model_zip": str(eval_log_dir / "best_model" / "best_model.zip"),
        "eval_reward_csv": str(plot_output.with_suffix(".csv")),
        "eval_reward_png": str(plot_output),
        "eval_debug_csv": str(plot_output.with_name(f"{name}_eval_debug.csv")),
        "eval_best_policy_png": str(OUTPUTS / "csv_png" / "eval_best_policy_curves" / f"sim_{name}_eval_best_policy_fitted_rotated.png"),
        "eval_best_policy_trajectory_csv": str(OUTPUTS / "csv_png" / "eval_best_policy_curves" / f"{name}_eval_best_policy_trajectory.csv"),
        "eval_best_policy_summary": str(OUTPUTS / "csv_png" / "eval_best_policy_curves" / f"{name}_eval_best_policy_summary.json"),
        "gait_json": str(JSON_DIR / f"{name}.json"),
        "policy_rerun_csv": str(POLICY_RERUN_DIR / f"{name}_policy_rerun_trajectory.csv"),
        "policy_rerun_png": str(POLICY_RERUN_DIR / f"sim_{name}_policy_rerun_fitted_rotated.png"),
        "policy_rerun_summary": str(POLICY_RERUN_DIR / f"{name}_policy_rerun_summary.json"),
        "trajectory_csv": str(TRAJ_DIR / f"{name}_trajectory.csv"),
        "trajectory_png": str(TRAJ_DIR / f"{name}_trajectory.png"),
        "trajectory_summary": str(TRAJ_DIR / f"{name}_summary.json"),
        "fitted_png": str(FIT_DIR / f"sim_{name}_fitted_rotated.png"),
        "fitted_summary": str(FIT_DIR / f"{name}_fitted_summary.json"),
        "eval_log_dir": str(eval_log_dir),
        "train_log": str(train_log),
        "eval_summary_json": str(LOG_DIR / f"{name}.eval_summary.json"),
    }
    write_status({"status": "running", "run_idx": run_idx, "total_runs": args.runs, "current": row, "updated_at": started_at})
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        cmd = build_train_cmd(args, name, output_base, eval_log_dir, plot_output)
        with train_log.open("w", encoding="utf-8") as handle:
            handle.write(f"# started {started_at}\n")
            handle.write(" ".join(cmd) + "\n\n")
            handle.flush()
            subprocess.run(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
            handle.write(f"\n# finished {now_text()}\n")

        model_path = Path(row["best_model_zip"]) if Path(row["best_model_zip"]).exists() else Path(row["model_zip"])
        policy_arr, policy_reward = rollout_policy_with_actions(model_path, cfg)
        row.update(write_policy_rerun_outputs(name, cfg, model_path, policy_arr, policy_reward))
        gait_path, diagnostics = write_mean_gait_json(
            name,
            cfg,
            model_path,
            policy_arr,
            Path(row["policy_rerun_csv"]),
        )
        row["gait_json"] = str(gait_path)
        row["export_diagnostics"] = json.dumps(diagnostics, sort_keys=True)
        row.update(write_fixed_gait_trajectory(name, gait_path, cfg))
        row.update(write_fixed_gait_fitted(name, Path(row["trajectory_csv"])))
        eval_summary = evaluate_policy(model_path, cfg, args.eval_episodes)
        row.update(eval_summary)
        Path(row["eval_summary_json"]).write_text(json.dumps({"model": str(model_path), **eval_summary}, indent=2), encoding="utf-8")
        row["status"] = "done"
        row["finished_at"] = now_text()
        append_summary(row)
        write_status({"status": "done_one", "last": row, "updated_at": row["finished_at"]})
    except Exception as exc:
        row["status"] = "failed"
        row["finished_at"] = now_text()
        row["error"] = repr(exc)
        append_summary(row)
        write_status({"status": "failed", "failed": row, "updated_at": row["finished_at"]})
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 10 fixed-start free-swim PPO trainings for paper comparison.")
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--start-run", type=int, default=1)
    parser.add_argument("--prefix", default=PREFIX)
    parser.add_argument("--organized-dir", type=Path, default=None)
    parser.add_argument("--timesteps", type=int, default=TIMESTEPS)
    parser.add_argument("--eval-freq", type=int, default=EVAL_FREQ)
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--episode-seconds", type=float, default=None)
    parser.add_argument("--warmup-seconds", type=float, default=None)
    parser.add_argument("--freq", type=float, default=None)
    parser.add_argument("--freq-low", type=float, default=None)
    parser.add_argument("--freq-high", type=float, default=None)
    parser.add_argument("--wavelength", type=float, default=None)
    parser.add_argument("--ajoint", type=float, default=None)
    parser.add_argument("--fixed-amp-scales", type=lambda value: parse_float_list(value, 6, "fixed-amp-scales"), default=None)
    parser.add_argument("--start-x", type=float, default=None)
    parser.add_argument("--start-y", type=float, default=None)
    parser.add_argument("--phase-lag-low", type=float, default=None)
    parser.add_argument("--phase-lag-high", type=float, default=None)
    parser.add_argument("--target-speed", type=float, default=None)
    parser.add_argument("--speed-error-weight", type=float, default=None)
    parser.add_argument("--speed-tolerance", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--energy-weight", type=float, default=None)
    parser.add_argument("--reward-average-seconds", type=float, default=None)
    parser.add_argument("--boundary-x-min", type=float, default=None)
    parser.add_argument("--boundary-x-max", type=float, default=None)
    parser.add_argument("--boundary-y", type=float, default=None)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-reward-plot", action="store_true")
    parser.add_argument("--no-organize", action="store_true")
    return parser.parse_args()


def write_reward_summary_plot() -> Path | None:
    groups: dict[tuple[str, str, str], list[Path]] = {}
    if SUMMARY_CSV.exists():
        with SUMMARY_CSV.open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                path = Path(row.get("eval_reward_csv", ""))
                if path.exists():
                    groups.setdefault(("radius", "straight", "0.0"), []).append(path)
    if not groups:
        return None
    return plot_groups(groups, PAPER_REWARD_DIR, n_bootstrap=10_000, seed=20260624)


def main() -> None:
    args = parse_args()
    configure_output_names(args.prefix, args.organized_dir)
    for directory in (ZIP_DIR, JSON_DIR, CSV_PNG_DIR, POLICY_RERUN_DIR, TRAJ_DIR, FIT_DIR, LOG_DIR, PAPER_REWARD_DIR, ORGANIZED_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    cfg = config_from_args(args)
    write_status(
        {
            "status": "starting",
            "total_runs": args.runs,
            "start_run": args.start_run,
            "settings": {
                "timesteps": args.timesteps,
                "eval_freq": args.eval_freq,
                "eval_episodes": args.eval_episodes,
                "episode_seconds": cfg.episode_seconds,
                "warmup_seconds": cfg.warmup_seconds,
                "start_x": cfg.start_x,
                "start_y": cfg.start_y,
                "frequency_low": cfg.frequency_low,
                "frequency_high": cfg.frequency_high,
                "phase_lag_low": cfg.phase_lag_low,
                "phase_lag_high": cfg.phase_lag_high,
                "target_speed": cfg.target_speed,
                "speed_error_weight": cfg.speed_error_weight,
                "energy_weight": cfg.energy_weight,
                "reward_average_seconds": cfg.reward_average_seconds,
            },
            "updated_at": now_text(),
        }
    )
    if args.start_run < 1 or args.start_run > args.runs:
        raise ValueError("--start-run must be between 1 and --runs")
    for run_idx in range(args.start_run, args.runs + 1):
        run_one(args, run_idx, cfg)
    reward_png = None if args.no_reward_plot else write_reward_summary_plot()
    organized_counts = [] if args.no_organize else organize_outputs(move=False)
    write_status(
        {
            "status": "complete",
            "total_runs": args.runs,
            "summary_csv": str(SUMMARY_CSV),
            "reward_summary_png": None if reward_png is None else str(reward_png),
            "organized_dir": None if args.no_organize else str(ORGANIZED_DIR),
            "organized_counts": organized_counts,
            "finished_at": now_text(),
        }
    )
    print(SUMMARY_CSV)
    if reward_png is not None:
        print(reward_png)
    if not args.no_organize:
        print(ORGANIZED_DIR)


if __name__ == "__main__":
    main()
