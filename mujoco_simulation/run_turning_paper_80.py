from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import run_turning_batch_60 as batch
from hopf_cpg import degrees_to_radians
from organize_paper_ppo_outputs import organize
from plot_paper_reward_summary import plot_groups, rows_from_summary
from record_fixed_gait_swim import write_recording
from rl_turning_env import TurningConfig, direction_sign


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
PREFIX = "paper80_ppo"
LOG_DIR = OUTPUTS / "batch_logs"
SUMMARY_CSV = LOG_DIR / "turning_paper80_ppo_summary.csv"
STATUS_PATH = LOG_DIR / "turning_paper80_ppo_status.json"
PAPER_REWARD_DIR = OUTPUTS / "csv_png" / f"{PREFIX}_reward_summary"
PAPER_RECORDING_DIR = OUTPUTS / "paper_swim_recordings"
ORGANIZED_DIR = OUTPUTS / "paper_ppo_80_extracted_figures"

TIMESTEPS = batch.TIMESTEPS
EVAL_FREQ = batch.EVAL_FREQ
EVAL_EPISODES = batch.EVAL_EPISODES
RUNS_PER_CONDITION = 10
TARGETS = (0.3, 0.7)
AJ0INT_DEG = batch.AJ0INT_DEG
YAW_WEIGHT = batch.YAW_WEIGHT
RADIUS_WEIGHT = batch.RADIUS_WEIGHT

SUMMARY_FIELDS = [
    *batch.SUMMARY_FIELDS,
    "swim_video_mp4",
    "swim_snapshots_dir",
    "swim_snapshot_count",
    "swim_trajectory_csv",
    "swim_recording_summary",
]


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def configure_batch_globals() -> None:
    batch.LOG_DIR = LOG_DIR
    batch.STATUS_PATH = STATUS_PATH
    batch.SUMMARY_CSV = SUMMARY_CSV
    batch.TIMESTEPS = TIMESTEPS
    batch.EVAL_FREQ = EVAL_FREQ
    batch.EVAL_EPISODES = EVAL_EPISODES


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


def build_jobs(runs_per_condition: int, targets: tuple[float, ...]) -> list[dict]:
    conditions: list[dict] = []
    for mode in ("yaw", "radius"):
        for direction in ("left", "right"):
            for target in targets:
                target_code = f"{int(round(target * 10)):02d}"
                mode_code = "y" if mode == "yaw" else "r"
                base = f"{PREFIX}_turn_{direction}_a20_{mode_code}{target_code}"
                condition = {
                    "mode": mode,
                    "direction": direction,
                    "target": target,
                    "base": base,
                    "runs": [],
                }
                for run_idx in range(1, runs_per_condition + 1):
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
                conditions.append(condition)
    return conditions


def cfg_for_job(job: dict) -> TurningConfig:
    direction_sign(job["direction"])
    cfg = TurningConfig()
    cfg.turn_direction = job["direction"]
    cfg.fixed_ajoint = degrees_to_radians(AJ0INT_DEG)
    target = abs(float(job["target"]))
    if job["mode"] == "yaw":
        cfg.target_yaw_rate = target
        cfg.target_radius = None
        cfg.yaw_rate_weight = YAW_WEIGHT
        cfg.radius_weight = 0.0
    else:
        cfg.target_yaw_rate = 0.45
        cfg.target_radius = target
        cfg.yaw_rate_weight = 0.0
        cfg.radius_weight = RADIUS_WEIGHT
    return cfg


def write_swim_recording(row: dict, job: dict) -> dict:
    out_dir = PAPER_RECORDING_DIR / job["base"] / job["name"]
    summary = write_recording(
        gait_path=Path(row["gait_json"]),
        out_dir=out_dir,
        seconds=18.0,
        fps=60.0,
        width=1920,
        height=1080,
        snapshot_interval=2.0,
        floor_size=(4.0, 3.0),
        camera_distance=2.4,
        camera_elevation=-70.0,
        camera_mode="fixed",
        camera_lookat=(0.60, 0.0, -0.02),
    )
    return {
        "swim_video_mp4": summary["video_mp4"],
        "swim_snapshots_dir": summary["snapshots_dir"],
        "swim_snapshot_count": summary["snapshot_count"],
        "swim_trajectory_csv": summary["trajectory_csv"],
        "swim_recording_summary": str(out_dir / f"{summary['name']}_swim_18s_summary.json"),
    }


def validate_paper_outputs(row: dict, *, require_recording: bool) -> None:
    batch.validate_outputs(row)
    if not require_recording:
        return
    required = [
        "swim_video_mp4",
        "swim_snapshots_dir",
        "swim_trajectory_csv",
        "swim_recording_summary",
    ]
    missing = [key for key in required if not row.get(key) or not Path(row[key]).exists()]
    if missing:
        raise RuntimeError(f"missing paper recording outputs: {', '.join(missing)}")
    if int(row.get("swim_snapshot_count", 0)) != 9:
        raise RuntimeError(f"expected 9 snapshots, got {row.get('swim_snapshot_count')}")


def run_one(job: dict, condition_index: int, run_count: int, *, no_record: bool) -> None:
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
        row.update(batch.train_model(job, cfg))
        row.update(batch.export_gait(job, cfg, Path(row["model_zip"]), Path(row["best_model_zip"])))
        row.update(batch.write_trajectory_plot(job, Path(row["gait_json"])))
        row.update(batch.write_fitted_plot(job, Path(row["trajectory_csv"])))
        if not no_record:
            row.update(write_swim_recording(row, job))
        validate_paper_outputs(row, require_recording=not no_record)
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


def write_reward_summary_plot() -> Path:
    groups: dict[tuple[str, str, str], list[Path]] = {}
    for row in rows_from_summary(SUMMARY_CSV):
        key = (row["mode"], row["direction"], row["target"])
        groups.setdefault(key, []).append(Path(row["eval_reward_csv"]))
    return plot_groups(groups, PAPER_REWARD_DIR, n_bootstrap=10_000, seed=20260619)


def parse_targets(value: str) -> tuple[float, ...]:
    targets = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not targets:
        raise argparse.ArgumentTypeError("at least one target is required")
    return targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper PPO batch for yaw_rate/radius targets 0.3 and 0.7.")
    parser.add_argument("--runs-per-condition", type=int, default=RUNS_PER_CONDITION)
    parser.add_argument("--targets", type=parse_targets, default=TARGETS)
    parser.add_argument("--timesteps", type=int, default=TIMESTEPS)
    parser.add_argument("--eval-freq", type=int, default=EVAL_FREQ)
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--no-record", action="store_true", help="Skip 18s paper videos and snapshots.")
    parser.add_argument("--no-reward-plot", action="store_true", help="Skip final mean reward plot.")
    parser.add_argument("--no-organize", action="store_true", help="Leave outputs in the default scattered folders.")
    parser.add_argument("--include-videos-in-organized", action="store_true", help="Move mp4 files and snapshots too.")
    return parser.parse_args()


def main() -> None:
    global TIMESTEPS, EVAL_FREQ, EVAL_EPISODES
    args = parse_args()
    TIMESTEPS = args.timesteps
    EVAL_FREQ = args.eval_freq
    EVAL_EPISODES = args.eval_episodes
    configure_batch_globals()

    for directory in (
        batch.ZIP_DIR,
        batch.JSON_DIR,
        batch.CSV_PNG_DIR,
        batch.POLICY_RERUN_DIR,
        batch.TRAJ_DIR,
        batch.FIT_DIR,
        LOG_DIR,
        PAPER_REWARD_DIR,
        PAPER_RECORDING_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    conditions = build_jobs(args.runs_per_condition, tuple(args.targets))
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
                "eval_episodes": EVAL_EPISODES,
                "runs_per_condition": args.runs_per_condition,
                "targets": list(args.targets),
                "ajoint_deg": AJ0INT_DEG,
            },
            "updated_at": now_text(),
        }
    )
    for condition_index, condition in enumerate(conditions, start=1):
        for job in condition["runs"]:
            completed += 1
            run_one(job, condition_index, completed, no_record=args.no_record)

    reward_png = None
    if not args.no_reward_plot:
        reward_png = write_reward_summary_plot()
    organized_dir = None if args.no_organize else ORGANIZED_DIR
    write_status(
        {
            "status": "complete",
            "total_runs": total_runs,
            "reward_summary_png": None if reward_png is None else str(reward_png),
            "organized_dir": None if organized_dir is None else str(organized_dir),
            "finished_at": now_text(),
        }
    )
    if not args.no_organize:
        organize(
            prefix=PREFIX,
            out_dir=ORGANIZED_DIR,
            include_videos=args.include_videos_in_organized,
            move=True,
        )
    if reward_png is not None:
        print(reward_png)
    if organized_dir is not None:
        print(organized_dir)


if __name__ == "__main__":
    main()
