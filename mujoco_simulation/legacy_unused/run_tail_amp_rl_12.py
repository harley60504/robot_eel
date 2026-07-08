from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from export_turning_rl_gait_json import config_from_args as export_config_from_args
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
from plot_fixed_gait_trajectories import run_gait, summarize
from plot_turning_policy_rollout_curves import write_rollout_outputs
from rl_training_plots import default_eval_log_dir, default_plot_path
from rl_turning_env import TurningConfig, direction_sign
from sim_config import DEFAULT_START_X, DEFAULT_START_Y, EEL_MODEL_XML
from train_turning_rl import debug_csv_path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
PREFIX = "formal_tailamp_y05_200k_avg1s_eval5k"
LOG_DIR = OUTPUTS / "batch_logs"
SUMMARY_CSV = LOG_DIR / f"{PREFIX}_summary.csv"
STATUS_PATH = LOG_DIR / f"{PREFIX}_status.json"
ZIP_DIR = OUTPUTS / "zips"
JSON_DIR = OUTPUTS / "json" / "rl_gaits"
CSV_PNG_DIR = OUTPUTS / "csv_png" / PREFIX
POLICY_CURVE_DIR = CSV_PNG_DIR / "policy_curve"
FIXED_GAIT_CURVE_DIR = CSV_PNG_DIR / "fixed_gait_curve"

TIMESTEPS = 200_000
EVAL_FREQ = 5_000
EVAL_EPISODES = 5
TARGET_YAW_RATE = 0.5
YAW_WEIGHT = 1.2
RADIUS_WEIGHT = 0.0
REWARD_AVERAGE_SECONDS = 1.0
TAIL_AMP_LOW = 0.9
TAIL_AMP_HIGH = 1.4
RUNS_PER_GROUP = 3

SUMMARY_FIELDS = [
    "name",
    "direction",
    "action_mode",
    "run_idx",
    "status",
    "started_at",
    "finished_at",
    "model_zip",
    "best_model_zip",
    "eval_reward_png",
    "eval_reward_csv",
    "eval_debug_csv",
    "train_log",
    "gait_json",
    "exported_amp_scales",
    "exported_joint_bias",
    "policy_rollout_png",
    "policy_rollout_summary",
    "policy_fitted_yaw_rate_rad_s",
    "policy_fitted_yaw_rate_error_rad_s",
    "policy_radius_m",
    "fixed_gait_png",
    "fixed_gait_summary",
    "fixed_fitted_yaw_rate_rad_s",
    "fixed_fitted_yaw_rate_error_rad_s",
    "fixed_radius_m",
    "error",
]


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


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


def existing_done_names() -> set[str]:
    if not SUMMARY_CSV.exists():
        return set()
    done: set[str] = set()
    with SUMMARY_CSV.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") == "done":
                done.add(row.get("name", ""))
    return done


def build_jobs(runs_per_group: int) -> list[dict]:
    jobs: list[dict] = []
    for direction in ("right", "left"):
        for action_mode in ("bias_only", "bias_tail3_amp"):
            for run_idx in range(1, runs_per_group + 1):
                mode_code = "tail3amp" if action_mode == "bias_tail3_amp" else "biasonly"
                name = f"{PREFIX}_{direction}_{mode_code}_run{run_idx:02d}"
                jobs.append(
                    {
                        "name": name,
                        "direction": direction,
                        "action_mode": action_mode,
                        "run_idx": run_idx,
                    }
                )
    return jobs


def cfg_for_job(job: dict) -> TurningConfig:
    direction_sign(job["direction"])
    cfg = TurningConfig()
    cfg.turn_direction = job["direction"]
    cfg.target_yaw_rate = TARGET_YAW_RATE
    cfg.target_radius = None
    cfg.yaw_rate_weight = YAW_WEIGHT
    cfg.radius_weight = RADIUS_WEIGHT
    cfg.reward_average_seconds = REWARD_AVERAGE_SECONDS
    cfg.action_mode = job["action_mode"]
    cfg.tail_amp_multiplier_low = TAIL_AMP_LOW
    cfg.tail_amp_multiplier_high = TAIL_AMP_HIGH
    return cfg


def train_model(job: dict, cfg: TurningConfig, timesteps: int, eval_freq: int, eval_episodes: int) -> dict:
    name = job["name"]
    output_base = ZIP_DIR / name
    plot_output = CSV_PNG_DIR / f"{name}_eval_reward.png"
    eval_log_dir = CSV_PNG_DIR / f"{name}_eval"
    train_log = LOG_DIR / f"{name}.train.log"
    cmd = [
        sys.executable,
        str(ROOT / "train_turning_rl.py"),
        "--timesteps",
        str(timesteps),
        "--eval-freq",
        str(eval_freq),
        "--eval-episodes",
        str(eval_episodes),
        "--output",
        str(output_base),
        "--turn-direction",
        cfg.turn_direction,
        "--target-yaw-rate",
        str(cfg.target_yaw_rate),
        "--yaw-rate-weight",
        str(cfg.yaw_rate_weight),
        "--radius-weight",
        str(cfg.radius_weight),
        "--reward-average-seconds",
        str(cfg.reward_average_seconds),
        "--action-mode",
        cfg.action_mode,
        "--tail-amp-multiplier-low",
        str(cfg.tail_amp_multiplier_low),
        "--tail-amp-multiplier-high",
        str(cfg.tail_amp_multiplier_high),
        "--plot-output",
        str(plot_output),
        "--eval-log-dir",
        str(eval_log_dir),
    ]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with train_log.open("w", encoding="utf-8") as handle:
        handle.write(f"# started {now_text()}\n")
        handle.write(" ".join(cmd) + "\n\n")
        handle.flush()
        subprocess.run(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
        handle.write(f"\n# finished {now_text()}\n")
    return {
        "model_zip": str(output_base.with_suffix(".zip")),
        "best_model_zip": str(eval_log_dir / "best_model" / "best_model.zip"),
        "eval_reward_png": str(plot_output),
        "eval_reward_csv": str(plot_output.with_suffix(".csv")),
        "eval_debug_csv": str(debug_csv_path(plot_output)),
        "train_log": str(train_log),
    }


def export_gait(job: dict, cfg: TurningConfig, model_zip: Path, best_model_zip: Path) -> dict:
    export_model = best_model_zip if best_model_zip.exists() else model_zip
    gait_path = JSON_DIR / f"{job['name']}.json"
    cmd = [
        sys.executable,
        str(ROOT / "export_turning_rl_gait_json.py"),
        "--model",
        str(export_model),
        "--output",
        str(gait_path),
        "--name",
        job["name"],
        "--turn-direction",
        cfg.turn_direction,
        "--target-yaw-rate",
        str(cfg.target_yaw_rate),
        "--action-mode",
        cfg.action_mode,
        "--tail-amp-multiplier-low",
        str(cfg.tail_amp_multiplier_low),
        "--tail-amp-multiplier-high",
        str(cfg.tail_amp_multiplier_high),
        "--strategy",
        "top-10%",
        "--samples",
        "300",
        "--max-episodes",
        "10",
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    gait = json.loads(gait_path.read_text(encoding="utf-8"))
    return {
        "gait_json": str(gait_path),
        "exported_amp_scales": json.dumps(gait["amp_scales"]),
        "exported_joint_bias": json.dumps(gait["joint_bias"]),
    }


def write_policy_curve(job: dict, cfg: TurningConfig, model_zip: Path) -> dict:
    summary = write_rollout_outputs(
        name=job["name"],
        model_zip=model_zip,
        cfg=cfg,
        out_dir=POLICY_CURVE_DIR,
        deterministic=True,
    )
    summary_path = POLICY_CURVE_DIR / f"{job['name']}_policy_rollout_summary.json"
    return {
        "policy_rollout_png": summary["policy_rollout_png"],
        "policy_rollout_summary": str(summary_path),
        "policy_fitted_yaw_rate_rad_s": summary.get("fitted_yaw_rate_rad_s"),
        "policy_fitted_yaw_rate_error_rad_s": summary.get("fitted_yaw_rate_error_rad_s"),
        "policy_radius_m": summary.get("radius"),
    }


def write_fixed_gait_curve(job: dict, gait_path: Path) -> dict:
    FIXED_GAIT_CURVE_DIR.mkdir(parents=True, exist_ok=True)
    gait, arr, hit_wall = run_gait(
        Path(EEL_MODEL_XML),
        gait_path,
        18.0,
        DEFAULT_START_X,
        DEFAULT_START_Y,
        wall_collision=False,
        stop_on_wall=False,
    )
    name = f"{job['name']}_fixed_gait"
    csv_path = FIXED_GAIT_CURVE_DIR / f"{name}_trajectory.csv"
    np.savetxt(csv_path, arr, delimiter=",", header="time,x,y,yaw", comments="")
    xy = rotate_sim_xy(arr[:, 1:3])
    curve, fit = fitted_curve(xy)
    metrics = trajectory_metrics(arr[:, :4], xy)
    target_yaw = direction_sign(job["direction"]) * TARGET_YAW_RATE
    metrics.update(add_fitted_yaw_rate_metrics(fit, metrics, target_yaw, job["direction"]))
    metrics.update(add_fitted_radius_metrics(fit, None))
    sim_summary = summarize(arr, 2.0)
    metrics.update(
        {
            "turn_direction": job["direction"],
            "yaw_rate_reward_weight": YAW_WEIGHT,
            "radius_reward_weight": RADIUS_WEIGHT,
            "mean_step_reward": 0.0,
            "episode_reward": 0.0,
            "mean_env_yaw_rate": sim_summary["yaw_rate_rad_s"],
            "mean_env_turn_radius": sim_summary["turn_radius_m"],
        }
    )
    fig, ax = plt.subplots(figsize=(4.8, 8.0), dpi=170)
    draw_rotated_tank(ax)
    ax.plot(curve[:, 0], curve[:, 1], linewidth=3.0)
    ax.scatter([xy[0, 0]], [xy[0, 1]], s=34, edgecolor="black", zorder=4)
    ax.scatter([xy[-1, 0]], [xy[-1, 1]], s=52, marker="x", linewidth=2.2, zorder=4)
    ax.set_title(name)
    add_sim_metric_box(ax, sim_metric_text(name, fit, metrics))
    fig.tight_layout()
    png_path = FIXED_GAIT_CURVE_DIR / f"sim_{name}_fitted_rotated.png"
    fig.savefig(png_path)
    plt.close(fig)
    summary = {
        "name": name,
        "gait_json": str(gait_path),
        "trajectory_csv": str(csv_path),
        "fixed_gait_png": str(png_path),
        "hit_wall": bool(hit_wall),
        **sim_summary,
        **fit,
        **metrics,
    }
    summary_path = FIXED_GAIT_CURVE_DIR / f"{name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "fixed_gait_png": str(png_path),
        "fixed_gait_summary": str(summary_path),
        "fixed_fitted_yaw_rate_rad_s": summary.get("fitted_yaw_rate_rad_s"),
        "fixed_fitted_yaw_rate_error_rad_s": summary.get("fitted_yaw_rate_error_rad_s"),
        "fixed_radius_m": summary.get("radius"),
    }


def validate_outputs(row: dict) -> None:
    required = [
        "model_zip",
        "eval_reward_png",
        "eval_reward_csv",
        "eval_debug_csv",
        "gait_json",
        "policy_rollout_png",
        "policy_rollout_summary",
        "fixed_gait_png",
        "fixed_gait_summary",
    ]
    missing = [key for key in required if not row.get(key) or not Path(row[key]).exists()]
    if missing:
        raise RuntimeError(f"missing outputs: {', '.join(missing)}")


def run_one(job: dict, index: int, total: int, timesteps: int, eval_freq: int, eval_episodes: int) -> None:
    started_at = now_text()
    row = {
        "name": job["name"],
        "direction": job["direction"],
        "action_mode": job["action_mode"],
        "run_idx": job["run_idx"],
        "status": "running",
        "started_at": started_at,
    }
    write_status({"status": "running", "index": index, "total": total, "current": row, "updated_at": started_at})
    try:
        cfg = cfg_for_job(job)
        row.update(train_model(job, cfg, timesteps, eval_freq, eval_episodes))
        model_zip = Path(row["model_zip"])
        best_model_zip = Path(row["best_model_zip"])
        curve_model = best_model_zip if best_model_zip.exists() else model_zip
        row.update(export_gait(job, cfg, model_zip, best_model_zip))
        row.update(write_policy_curve(job, cfg, curve_model))
        row.update(write_fixed_gait_curve(job, Path(row["gait_json"])))
        validate_outputs(row)
        row["status"] = "done"
        row["finished_at"] = now_text()
        append_summary(row)
        write_status({"status": "done_one", "index": index, "total": total, "last": row, "updated_at": row["finished_at"]})
    except Exception as exc:
        row["status"] = "failed"
        row["finished_at"] = now_text()
        row["error"] = repr(exc)
        append_summary(row)
        write_status({"status": "failed", "index": index, "total": total, "failed": row, "updated_at": row["finished_at"]})
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 12 formal 200k PPO jobs comparing bias-only and learned tail amp_scale.")
    parser.add_argument("--timesteps", type=int, default=TIMESTEPS)
    parser.add_argument("--eval-freq", type=int, default=EVAL_FREQ)
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--runs-per-group", type=int, default=RUNS_PER_GROUP)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for directory in (ZIP_DIR, JSON_DIR, CSV_PNG_DIR, POLICY_CURVE_DIR, FIXED_GAIT_CURVE_DIR, LOG_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args.runs_per_group)
    done = existing_done_names() if args.resume else set()
    pending = [job for job in jobs if job["name"] not in done]
    write_status(
        {
            "status": "starting",
            "total": len(jobs),
            "pending": len(pending),
            "settings": {
                "timesteps": args.timesteps,
                "eval_freq": args.eval_freq,
                "eval_episodes": args.eval_episodes,
                "runs_per_group": args.runs_per_group,
                "target_yaw_rate": TARGET_YAW_RATE,
                "tail_amp_multiplier_low": TAIL_AMP_LOW,
                "tail_amp_multiplier_high": TAIL_AMP_HIGH,
            },
            "updated_at": now_text(),
        }
    )
    for offset, job in enumerate(pending, start=1):
        run_one(job, offset, len(pending), args.timesteps, args.eval_freq, args.eval_episodes)
    write_status({"status": "complete", "total": len(jobs), "completed": len(pending), "updated_at": now_text()})
    print(SUMMARY_CSV)
    print(STATUS_PATH)


if __name__ == "__main__":
    main()
