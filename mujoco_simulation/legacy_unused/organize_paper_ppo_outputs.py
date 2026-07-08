from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"


@dataclass(frozen=True)
class MoveSet:
    category: str
    source: Path
    patterns: tuple[str, ...]
    recursive: bool = False


def iter_files(source: Path, patterns: tuple[str, ...], recursive: bool) -> list[Path]:
    files: list[Path] = []
    if not source.exists():
        return files
    for pattern in patterns:
        iterator = source.rglob(pattern) if recursive else source.glob(pattern)
        files.extend(path for path in iterator if path.is_file())
    return sorted(set(files))


def safe_child(path: Path, parent: Path) -> Path:
    resolved = path.resolve()
    parent_resolved = parent.resolve()
    if resolved != parent_resolved and parent_resolved not in resolved.parents:
        raise ValueError(f"{resolved} is outside {parent_resolved}")
    return resolved


def organize(*, prefix: str, out_dir: Path, include_videos: bool, move: bool) -> list[dict[str, str | int]]:
    out_dir = safe_child(out_dir, OUTPUTS)
    out_dir.mkdir(parents=True, exist_ok=True)
    action = shutil.move if move else shutil.copy2
    rows = []
    sets = [
        MoveSet("average_reward_summary", OUTPUTS / "csv_png" / f"{prefix}_reward_summary", ("*",)),
        MoveSet("batch_logs_and_status", OUTPUTS / "batch_logs", (f"turning_{prefix}_summary.csv", f"turning_{prefix}_status.json")),
        MoveSet("gait_jsons", OUTPUTS / "json" / "rl_gaits", (f"{prefix}_*.json",)),
        MoveSet(
            "individual_rewards",
            OUTPUTS / "csv_png",
            (f"{prefix}_*_eval_reward.png", f"{prefix}_*_eval_reward.csv", f"{prefix}_*_eval_debug.csv"),
        ),
        MoveSet(
            "fixed_gait_trajectories_after_mean",
            OUTPUTS / "csv_png" / "fixed_gait_trajectories_mean",
            (f"{prefix}_*_trajectory.png", f"{prefix}_*_trajectory.csv", f"{prefix}_*_summary.json"),
        ),
        MoveSet(
            "training_eval_best_policy_curves",
            OUTPUTS / "csv_png" / "eval_best_policy_curves",
            (
                f"sim_{prefix}_*_eval_best_policy_fitted_rotated.png",
                f"{prefix}_*_eval_best_policy_trajectory.csv",
                f"{prefix}_*_eval_best_policy_summary.json",
            ),
        ),
        MoveSet(
            "policy_rerun_before_fixed_mean",
            OUTPUTS / "csv_png" / "policy_rerun_best_once",
            (
                f"sim_{prefix}_*_policy_rerun_fitted_rotated.png",
                f"{prefix}_*_policy_rerun_trajectory.csv",
                f"{prefix}_*_policy_rerun_summary.json",
            ),
        ),
        MoveSet(
            "fixed_gait_fitted_rotated_after_mean",
            OUTPUTS / "csv_png" / "fitted_curve_comparison_mean",
            (f"sim_{prefix}_*_fitted_rotated.png", f"{prefix}_*_fitted_summary.json"),
        ),
        MoveSet(
            "swim_recording_metadata",
            OUTPUTS / "paper_swim_recordings",
            (f"{prefix}_*_swim_18s_summary.json", f"{prefix}_*_swim_18s_snapshots.csv"),
            recursive=True,
        ),
    ]
    if include_videos:
        sets.append(
            MoveSet(
                "swim_recordings",
                OUTPUTS / "paper_swim_recordings",
                (f"{prefix}_*_swim_18s.mp4", f"{prefix}_*_snapshots/*.png"),
                recursive=True,
            )
        )

    for move_set in sets:
        dest = safe_child(out_dir / move_set.category, out_dir)
        dest.mkdir(parents=True, exist_ok=True)
        moved = 0
        for path in iter_files(move_set.source, move_set.patterns, move_set.recursive):
            target = dest / path.name
            action(str(path), str(target))
            moved += 1
        rows.append({"category": move_set.category, "files": moved, "path": str(dest)})

    counts = []
    for directory in sorted(path for path in out_dir.iterdir() if path.is_dir()):
        counts.append(
            {
                "category": directory.name,
                "files": sum(1 for path in directory.iterdir() if path.is_file()),
                "path": str(directory),
            }
        )
    with (out_dir / "organized_counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["category", "files", "path"])
        writer.writeheader()
        writer.writerows(counts)
    with (out_dir / "last_organize_operations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["category", "files", "path"])
        writer.writeheader()
        writer.writerows(rows)
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Move or copy one paper PPO batch into a clean organized folder.")
    parser.add_argument("--prefix", required=True, help="File prefix, for example paper_ppo or paper80_ppo.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--include-videos", action="store_true", help="Also move/copy mp4 files and snapshot PNGs.")
    parser.add_argument("--copy", action="store_true", help="Copy instead of moving.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = organize(prefix=args.prefix, out_dir=args.out_dir, include_videos=args.include_videos, move=not args.copy)
    for row in counts:
        print(f"{row['category']}: {row['files']} files")


if __name__ == "__main__":
    main()
