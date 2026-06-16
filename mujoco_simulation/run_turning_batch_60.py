from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hopf_cpg import degrees_to_radians
from plot_fitted_gait_curves import (
    add_sim_metric_box,
    draw_rotated_tank,
    fit_sim_trajectory,
    sim_metric_text,
)
from plot_fixed_gait_trajectories import (
    build_trajectory_summary,
    draw_environment,
    plot_one,
    run_gait,
    summarize,
)
from policy_rerun_fixed_gait import write_mean_fixed_gait_from_best_policy
from rl_training_plots import default_eval_log_dir, default_plot_path
from rl_turning_env import TurningConfig, direction_sign
from sim_config import DEFAULT_START_X, DEFAULT_START_Y, EEL_MODEL_XML
from train_turning_rl import debug_csv_path


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
ZIP_DIR = OUTPUTS / "zips"
JSON_DIR = OUTPUTS / "json" / "rl_gaits"
CSV_PNG_DIR = OUTPUTS / "csv_png"
POLICY_RERUN_DIR = CSV_PNG_DIR / "policy_rerun_best_once"
TRAJ_DIR = CSV_PNG_DIR / "fixed_gait_trajectories_mean"
FIT_DIR = CSV_PNG_DIR / "fitted_curve_comparison_mean"
LOG_DIR = OUTPUTS / "batch_logs"
STATUS_PATH = LOG_DIR / "turning_batch_60_status.json"
SUMMARY_CSV = LOG_DIR / "turning_batch_60_summary.csv"

TIMESTEPS = 200_000
EVAL_FREQ = 5_000
EVAL_EPISODES = 5
EXPORT_STRATEGY = "policy-rerun-mean"
AJ0INT_DEG = 20.0
YAW_WEIGHT = 1.2
RADIUS_WEIGHT = 1.2
PLOT_SECONDS = 30.0
PLOT_WARMUP_SECONDS = 0.0


SUMMARY_FIELDS = [
    "name",
    "mode",
    "direction",
    "target",
    "status",
    "started_at",
    "finished_at",
    "model_zip",
    "gait_json",
    "eval_reward_csv",
    "eval_reward_png",
    "eval_debug_csv",
    "best_model_zip",
    "policy_rerun_csv",
    "policy_rerun_png",
    "policy_rerun_summary",
    "trajectory_csv",
    "trajectory_png",
    "trajectory_summary",
    "fitted_png",
    "fitted_summary",
    "train_log",
    "export_diagnostics",
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


def build_jobs() -> list[dict]:
    jobs: list[dict] = []
    for mode in ("radius", "yaw"):
        for direction in ("right", "left"):
            for target in (0.3, 0.5, 0.7):
                target_code = f"{int(round(target * 10)):02d}"
                mode_code = "y" if mode == "yaw" else "r"
                base = f"ppo_turn_{direction}_a20_{mode_code}{target_code}"
                condition = {
                    "mode": mode,
                    "direction": direction,
                    "target": target,
                    "base": base,
                    "runs": [],
                }
                for run_idx in range(1, 6):
                    condition["runs"].append(
                        {
                            "mode": mode,
                            "direction": direction,
                            "target": target,
                            "base": base,
                            "run_idx": run_idx,
                            "name": f"{base}_run{run_idx:02d}",
                        }
                    )
                jobs.append(condition)
    return jobs


def cfg_for_job(job: dict) -> TurningConfig:
    direction_sign(job["direction"])
    cfg = TurningConfig()
    cfg.turn_direction = job["direction"]
    cfg.fixed_ajoint = degrees_to_radians(AJ0INT_DEG)
    if job["mode"] == "yaw":
        cfg.target_yaw_rate = abs(float(job["target"]))
        cfg.target_radius = None
        cfg.yaw_rate_weight = YAW_WEIGHT
        cfg.radius_weight = 0.0
    else:
        cfg.target_yaw_rate = 0.45
        cfg.target_radius = abs(float(job["target"]))
        cfg.yaw_rate_weight = 0.0
        cfg.radius_weight = RADIUS_WEIGHT
    return cfg


def train_model(job: dict, cfg: TurningConfig) -> dict:
    name = job["name"]
    output_base = ZIP_DIR / name
    plot_output = default_plot_path(output_base)
    eval_log_dir = default_eval_log_dir(output_base)
    train_log = LOG_DIR / f"{name}.train.log"
    cmd = [
        sys.executable,
        str(ROOT / "train_turning_rl.py"),
        "--timesteps",
        str(TIMESTEPS),
        "--eval-freq",
        str(EVAL_FREQ),
        "--eval-episodes",
        str(EVAL_EPISODES),
        "--output",
        str(output_base),
        "--turn-direction",
        cfg.turn_direction,
        "--target-yaw-rate",
        str(cfg.target_yaw_rate),
        "--ajoint",
        str(AJ0INT_DEG),
        "--yaw-rate-weight",
        str(cfg.yaw_rate_weight),
        "--radius-weight",
        str(cfg.radius_weight),
        "--plot-output",
        str(plot_output),
        "--eval-log-dir",
        str(eval_log_dir),
    ]
    if cfg.target_radius is not None:
        cmd.extend(["--target-radius", str(cfg.target_radius)])
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
    if not best_model_zip.exists():
        raise FileNotFoundError(f"Eval best model was not found for fixed-gait export: {best_model_zip}")
    export_model = best_model_zip
    gait_path = JSON_DIR / f"{job['name']}.json"
    source_extra = {
        "batch": {
            "script": str(Path(__file__).resolve()),
            "mode": job["mode"],
            "condition_base": job["base"],
            "run_idx": job["run_idx"],
            "timesteps": TIMESTEPS,
            "eval_freq": EVAL_FREQ,
            "eval_episodes": EVAL_EPISODES,
            "config": {key: str(value) if key == "xml_path" else value for key, value in asdict(cfg).items()},
        },
    }
    outputs, diagnostics = write_mean_fixed_gait_from_best_policy(
        name=job["name"],
        cfg=cfg,
        model_zip=export_model,
        gait_path=gait_path,
        policy_out_dir=POLICY_RERUN_DIR,
        source_extra=source_extra,
    )
    return {
        **outputs,
        "best_model_zip": str(best_model_zip),
        "export_diagnostics": json.dumps(diagnostics, sort_keys=True),
    }


def write_trajectory_plot(job: dict, gait_path: Path) -> dict:
    TRAJ_DIR.mkdir(parents=True, exist_ok=True)
    gait, arr, hit_wall = run_gait(Path(EEL_MODEL_XML), gait_path, PLOT_SECONDS, DEFAULT_START_X, DEFAULT_START_Y)
    if arr.shape[0] < 2:
        raise RuntimeError(f"{job['name']} trajectory has too few points")
    summary = summarize(arr, PLOT_WARMUP_SECONDS)
    csv_path = TRAJ_DIR / f"{job['name']}_trajectory.csv"
    png_path = TRAJ_DIR / f"{job['name']}_trajectory.png"
    summary_path = TRAJ_DIR / f"{job['name']}_summary.json"
    np.savetxt(csv_path, arr, delimiter=",", header="time,x,y,yaw", comments="")

    fig, ax = plt.subplots(figsize=(7, 5), dpi=170)
    draw_environment(ax, DEFAULT_START_X, DEFAULT_START_Y)
    plot_one(ax, job["name"], arr, summary)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"{job['name']} trajectory until wall contact")
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)

    row = build_trajectory_summary(job["name"], gait, gait_path, arr, summary, hit_wall)
    row["trajectory_csv"] = str(csv_path)
    row["trajectory_png"] = str(png_path)
    summary_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    return {
        "trajectory_csv": str(csv_path),
        "trajectory_png": str(png_path),
        "trajectory_summary": str(summary_path),
    }


def write_fitted_plot(job: dict, trajectory_csv: Path) -> dict:
    FIT_DIR.mkdir(parents=True, exist_ok=True)
    name, arr, xy, curve, fit, metrics = fit_sim_trajectory(trajectory_csv)
    if arr.shape[0] < 2:
        raise RuntimeError(f"{job['name']} fitted plot has too few points")
    fig, ax = plt.subplots(figsize=(4.8, 8.0), dpi=170)
    draw_rotated_tank(ax)
    color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
    ax.plot(curve[:, 0], curve[:, 1], color=color, linewidth=3.0)
    ax.scatter([xy[0, 0]], [xy[0, 1]], s=34, color=color, edgecolor="black", zorder=4)
    ax.scatter([xy[-1, 0]], [xy[-1, 1]], s=52, marker="x", color=color, linewidth=2.2, zorder=4)
    ax.set_title(f"{name} fitted curve")
    add_sim_metric_box(ax, sim_metric_text(name, fit, metrics))
    fig.tight_layout()
    png_path = FIT_DIR / f"sim_{name}_fitted_rotated.png"
    top_png_path = CSV_PNG_DIR / png_path.name
    fig.savefig(png_path)
    fig.savefig(top_png_path)
    plt.close(fig)
    summary_path = FIT_DIR / f"{name}_fitted_summary.json"
    fitted_summary = {
        "name": name,
        **fit,
        **metrics,
        "fit_png": str(png_path),
        "fit_output_png": str(top_png_path),
    }
    summary_path.write_text(json.dumps(fitted_summary, indent=2), encoding="utf-8")
    return {"fitted_png": str(png_path), "fitted_summary": str(summary_path)}


def validate_outputs(row: dict) -> None:
    required = [
        "model_zip",
        "gait_json",
        "eval_reward_csv",
        "eval_reward_png",
        "eval_debug_csv",
        "trajectory_csv",
        "trajectory_png",
        "trajectory_summary",
        "fitted_png",
        "fitted_summary",
    ]
    missing = [key for key in required if not Path(row[key]).exists()]
    if missing:
        raise RuntimeError(f"missing outputs: {', '.join(missing)}")
    debug_rows = list(csv.DictReader(Path(row["eval_debug_csv"]).open("r", encoding="utf-8")))
    if not debug_rows:
        raise RuntimeError("eval debug csv has no rows")
    fit_data = json.loads(Path(row["fitted_summary"]).read_text(encoding="utf-8"))
    if fit_data.get("duration_s", 0.0) <= 0.0:
        raise RuntimeError("fitted summary duration is not positive")


def run_one(job: dict, condition_index: int, run_count: int) -> None:
    started_at = now_text()
    row = {
        "name": job["name"],
        "mode": job["mode"],
        "direction": job["direction"],
        "target": job["target"],
        "status": "running",
        "started_at": started_at,
    }
    write_status(
        {
            "status": "running",
            "condition_index": condition_index,
            "run_count": run_count,
            "current": row,
            "updated_at": started_at,
        }
    )
    try:
        cfg = cfg_for_job(job)
        row.update(train_model(job, cfg))
        row.update(export_gait(job, cfg, Path(row["model_zip"]), Path(row["best_model_zip"])))
        row.update(write_trajectory_plot(job, Path(row["gait_json"])))
        row.update(write_fitted_plot(job, Path(row["trajectory_csv"])))
        validate_outputs(row)
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


def main() -> None:
    for directory in (ZIP_DIR, JSON_DIR, CSV_PNG_DIR, POLICY_RERUN_DIR, TRAJ_DIR, FIT_DIR, LOG_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    conditions = build_jobs()
    total_runs = sum(len(condition["runs"]) for condition in conditions)
    completed = 0
    write_status(
        {
            "status": "starting",
            "total_conditions": len(conditions),
            "total_runs": total_runs,
            "settings": {
                "timesteps": TIMESTEPS,
                "eval_freq": EVAL_FREQ,
                "strategy": EXPORT_STRATEGY,
                "ajoint_deg": AJ0INT_DEG,
            },
            "updated_at": now_text(),
        }
    )
    for condition_index, condition in enumerate(conditions, start=1):
        runs = condition["runs"]
        run_one(runs[0], condition_index, completed + 1)
        completed += 1
        for job in runs[1:]:
            run_one(job, condition_index, completed + 1)
            completed += 1
    write_status({"status": "complete", "total_runs": total_runs, "finished_at": now_text()})


if __name__ == "__main__":
    main()
