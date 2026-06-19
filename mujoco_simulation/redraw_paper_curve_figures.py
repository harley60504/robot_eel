from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_fitted_gait_curves import add_sim_metric_box, draw_rotated_tank, fit_sim_trajectory, sim_metric_text
from plot_fixed_gait_trajectories import draw_environment, plot_one, summarize


def redraw_fixed_trajectory(csv_path: Path) -> Path:
    arr = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if arr.ndim == 1:
        arr = arr[None, :]
    name = csv_path.name.removesuffix("_trajectory.csv")
    summary_path = csv_path.with_name(f"{name}_summary.json")
    warmup_seconds = 0.0
    if summary_path.exists():
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            warmup_seconds = float(data.get("warmup_seconds", 0.0) or 0.0)
        except (TypeError, ValueError, json.JSONDecodeError):
            warmup_seconds = 0.0
    summary = summarize(arr, warmup_seconds)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=170)
    draw_environment(ax, float(arr[0, 1]), float(arr[0, 2]))
    plot_one(ax, name, arr, summary)
    ax.relim()
    ax.autoscale_view()
    ax.margins(0.12)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"{name} fixed gait trajectory")
    fig.tight_layout()
    png_path = csv_path.with_suffix(".png")
    fig.savefig(png_path)
    plt.close(fig)
    return png_path


def redraw_fitted_curve(csv_path: Path, out_dir: Path | None = None) -> Path:
    name, arr, xy, curve, fit, metrics = fit_sim_trajectory(csv_path)
    fig, ax = plt.subplots(figsize=(4.8, 8.0), dpi=170)
    draw_rotated_tank(ax)
    color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
    ax.plot(curve[:, 0], curve[:, 1], color=color, linewidth=3.0)
    ax.scatter([xy[0, 0]], [xy[0, 1]], s=34, color=color, edgecolor="black", zorder=4)
    ax.scatter([xy[-1, 0]], [xy[-1, 1]], s=52, marker="x", color=color, linewidth=2.2, zorder=4)
    ax.relim()
    ax.autoscale_view()
    ax.margins(0.12)
    ax.set_title(f"{name} fitted curve")
    add_sim_metric_box(ax, sim_metric_text(name, fit, metrics))
    fig.tight_layout()
    if out_dir is None:
        out_dir = csv_path.parent
    png_path = out_dir / f"sim_{name}_fitted_rotated.png"
    fig.savefig(png_path)
    plt.close(fig)
    return png_path


def redraw_folder(root: Path) -> dict[str, int]:
    counts = {
        "fixed_gait_trajectories_after_mean": 0,
        "fixed_gait_fitted_rotated_after_mean": 0,
        "policy_rerun_before_fixed_mean": 0,
        "training_eval_best_policy_curves": 0,
    }

    fixed_dir = root / "fixed_gait_trajectories_after_mean"
    if fixed_dir.exists():
        for csv_path in sorted(fixed_dir.glob("*_trajectory.csv")):
            if any(part in csv_path.name for part in ("policy_rerun", "eval_best_policy")):
                continue
            redraw_fixed_trajectory(csv_path)
            counts["fixed_gait_trajectories_after_mean"] += 1

    fitted_dir = root / "fixed_gait_fitted_rotated_after_mean"
    if fitted_dir.exists() and fixed_dir.exists():
        for csv_path in sorted(fixed_dir.glob("*_trajectory.csv")):
            if any(part in csv_path.name for part in ("policy_rerun", "eval_best_policy")):
                continue
            redraw_fitted_curve(csv_path, fitted_dir)
            counts["fixed_gait_fitted_rotated_after_mean"] += 1

    policy_dir = root / "policy_rerun_before_fixed_mean"
    if policy_dir.exists():
        for csv_path in sorted(policy_dir.glob("*_policy_rerun_trajectory.csv")):
            redraw_fitted_curve(csv_path, policy_dir)
            counts["policy_rerun_before_fixed_mean"] += 1

    eval_dir = root / "training_eval_best_policy_curves"
    if eval_dir.exists():
        for csv_path in sorted(eval_dir.glob("*_eval_best_policy_trajectory.csv")):
            redraw_fitted_curve(csv_path, eval_dir)
            counts["training_eval_best_policy_curves"] += 1

    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Redraw organized paper PPO trajectory/fitted curve PNGs without tank boundaries.")
    parser.add_argument("--organized-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = redraw_folder(args.organized_dir)
    for key, value in counts.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
