from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from run_turning_batch_60 import (
    LOG_DIR,
    cfg_for_job,
    export_gait,
    train_model,
    validate_outputs,
    write_fitted_plot,
    write_trajectory_plot,
)


YAW_WEIGHT = 3.0
RUNS_PER_CONDITION = 2
STATUS_PATH = LOG_DIR / "turning_yaw_weight_probe_status.json"
SUMMARY_CSV = LOG_DIR / "turning_yaw_weight_probe_summary.csv"
SUMMARY_FIELDS = [
    "name",
    "mode",
    "direction",
    "target",
    "yaw_weight",
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
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def build_jobs() -> list[dict]:
    jobs: list[dict] = []
    for direction in ("right", "left"):
        for target in (0.3, 0.7):
            target_code = f"{int(round(target * 10)):02d}"
            base = f"ppo_turn_{direction}_a20_y{target_code}_w3"
            for run_idx in range(1, RUNS_PER_CONDITION + 1):
                jobs.append(
                    {
                        "mode": "yaw",
                        "direction": direction,
                        "target": target,
                        "base": base,
                        "run_idx": run_idx,
                        "name": f"{base}_run{run_idx:02d}",
                    }
                )
    return jobs


def run_one(job: dict, index: int, total: int) -> None:
    started_at = now_text()
    row = {
        "name": job["name"],
        "mode": job["mode"],
        "direction": job["direction"],
        "target": job["target"],
        "yaw_weight": YAW_WEIGHT,
        "status": "running",
        "started_at": started_at,
    }
    write_status({"status": "running", "index": index, "total": total, "current": row, "updated_at": started_at})
    try:
        cfg = cfg_for_job(job)
        cfg.yaw_rate_weight = YAW_WEIGHT
        cfg.radius_weight = 0.0
        row.update(train_model(job, cfg))
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
    jobs = build_jobs()
    write_status({"status": "starting", "total": len(jobs), "yaw_weight": YAW_WEIGHT, "updated_at": now_text()})
    for index, job in enumerate(jobs, start=1):
        run_one(job, index, len(jobs))
    write_status({"status": "complete", "total": len(jobs), "finished_at": now_text()})


if __name__ == "__main__":
    main()
