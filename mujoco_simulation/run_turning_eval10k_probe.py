from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

from run_turning_batch_60 import (
    FIT_DIR,
    JSON_DIR,
    LOG_DIR,
    TRAJ_DIR,
    ZIP_DIR,
    cfg_for_job,
    export_gait,
    validate_outputs,
    write_fitted_plot,
    write_trajectory_plot,
)
from rl_training_plots import default_eval_log_dir, default_plot_path
from train_turning_rl import debug_csv_path


TIMESTEPS = 200_000
EVAL_FREQ = 10_000
EVAL_EPISODES = 5
AJ0INT_DEG = 20.0
YAW_WEIGHT = 1.2
STATUS_PATH = LOG_DIR / "turning_eval10k_probe_status.json"
SUMMARY_CSV = LOG_DIR / "turning_eval10k_probe_summary.csv"

FIELDS = [
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
    "trajectory_csv",
    "trajectory_png",
    "trajectory_summary",
    "fitted_png",
    "fitted_summary",
    "train_log",
    "error",
]


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def write_status(payload: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_summary(row: dict) -> None:
    exists = SUMMARY_CSV.exists()
    with SUMMARY_CSV.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in FIELDS})


def jobs() -> list[dict]:
    return [
        {
            "mode": "yaw",
            "direction": "right",
            "target": target,
            "base": f"ppo_turn_right_a20_y{int(target * 10):02d}_e10k_ebest",
            "run_idx": 1,
            "name": f"ppo_turn_right_a20_y{int(target * 10):02d}_e10k_ebest_run01",
        }
        for target in (0.3, 0.5, 0.7)
    ]


def train(job: dict) -> dict:
    output_base = ZIP_DIR / job["name"]
    plot_output = default_plot_path(output_base)
    eval_log_dir = default_eval_log_dir(output_base)
    train_log = LOG_DIR / f"{job['name']}.train.log"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "train_turning_rl.py"),
        "--timesteps",
        str(TIMESTEPS),
        "--eval-freq",
        str(EVAL_FREQ),
        "--eval-episodes",
        str(EVAL_EPISODES),
        "--output",
        str(output_base),
        "--turn-direction",
        "right",
        "--target-yaw-rate",
        str(job["target"]),
        "--ajoint",
        str(AJ0INT_DEG),
        "--yaw-rate-weight",
        str(YAW_WEIGHT),
        "--radius-weight",
        "0.0",
        "--plot-output",
        str(plot_output),
        "--eval-log-dir",
        str(eval_log_dir),
    ]
    with train_log.open("w", encoding="utf-8") as handle:
        handle.write(f"# started {now_text()}\n")
        handle.write(" ".join(cmd) + "\n\n")
        handle.flush()
        subprocess.run(cmd, cwd=Path(__file__).resolve().parent, stdout=handle, stderr=subprocess.STDOUT, check=True)
        handle.write(f"\n# finished {now_text()}\n")
    return {
        "model_zip": str(output_base.with_suffix(".zip")),
        "eval_reward_png": str(plot_output),
        "eval_reward_csv": str(plot_output.with_suffix(".csv")),
        "eval_debug_csv": str(debug_csv_path(plot_output)),
        "train_log": str(train_log),
    }


def run_one(job: dict, index: int, total: int) -> None:
    row = {
        "name": job["name"],
        "mode": job["mode"],
        "direction": job["direction"],
        "target": job["target"],
        "status": "running",
        "started_at": now_text(),
    }
    write_status({"status": "running", "index": index, "total": total, "current": row, "updated_at": now_text()})
    try:
        cfg = cfg_for_job(job)
        row.update(train(job))
        row.update(export_gait(job, cfg, Path(row["model_zip"])))
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
    for directory in (ZIP_DIR, JSON_DIR, TRAJ_DIR, FIT_DIR, LOG_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    all_jobs = jobs()
    write_status({"status": "starting", "total": len(all_jobs), "eval_freq": EVAL_FREQ, "updated_at": now_text()})
    for index, job in enumerate(all_jobs, start=1):
        run_one(job, index, len(all_jobs))
    write_status({"status": "complete", "total": len(all_jobs), "finished_at": now_text()})


if __name__ == "__main__":
    main()
