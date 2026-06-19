from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GROUP_ORDER = {
    ("yaw", "left"): 0,
    ("yaw", "right"): 1,
    ("radius", "left"): 2,
    ("radius", "right"): 3,
}

REWARD_NAME_RE = re.compile(
    r"ppo_turn_(?P<direction>left|right)_a\d+_(?P<mode_code>[yr])(?P<target_code>\d+)_run\d+_eval_reward\.csv$"
)


def read_eval_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = list(csv.DictReader(Path(path).open(encoding="utf-8")))
    if not rows:
        raise ValueError(f"empty eval reward csv: {path}")
    timesteps = np.asarray([float(row["timesteps"]) for row in rows], dtype=np.float64)
    rewards = np.asarray([float(row["mean_reward"]) for row in rows], dtype=np.float64)
    return timesteps, rewards


def rows_from_summary(path: Path) -> list[dict]:
    rows = []
    for row in csv.DictReader(Path(path).open(encoding="utf-8")):
        if row.get("status") != "done" or not row.get("eval_reward_csv"):
            continue
        eval_csv = Path(row["eval_reward_csv"])
        if not eval_csv.exists():
            print(f"skip missing eval reward csv: {eval_csv}")
            continue
        rows.append(row)
    return rows


def rows_from_reward_dir(path: Path, target: float | None) -> list[dict]:
    rows = []
    for eval_csv in sorted(Path(path).glob("*_eval_reward.csv")):
        match = REWARD_NAME_RE.match(eval_csv.name)
        if match is None:
            continue
        mode = "yaw" if match.group("mode_code") == "y" else "radius"
        target_value = int(match.group("target_code")) / 10.0
        if target is not None and abs(target_value - target) > 1e-9:
            continue
        rows.append(
            {
                "mode": mode,
                "direction": match.group("direction"),
                "target": f"{target_value:.1f}",
                "eval_reward_csv": str(eval_csv),
            }
        )
    return rows


def common_series(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    loaded = [read_eval_csv(path) for path in paths]
    min_start = max(float(t[0][0]) for t in loaded)
    max_end = min(float(t[0][-1]) for t in loaded)
    reference = loaded[0][0]
    common_steps = reference[(reference >= min_start) & (reference <= max_end)]
    if common_steps.size == 0:
        raise ValueError("evaluation curves have no overlapping timestep range")
    curves = [np.interp(common_steps, timesteps, rewards) for timesteps, rewards in loaded]
    return common_steps, np.vstack(curves)


def bootstrap_ci(curves: np.ndarray, n_bootstrap: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(curves, axis=0)
    if curves.shape[0] < 2:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, curves.shape[0], size=(n_bootstrap, curves.shape[0]))
    boot_means = np.mean(curves[sample_indices], axis=1)
    low, high = np.percentile(boot_means, [2.5, 97.5], axis=0)
    return mean, low, high


def write_group_csv(path: Path, steps: np.ndarray, mean: np.ndarray, low: np.ndarray, high: np.ndarray, curves: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [steps, mean, low, high, *[curves[idx] for idx in range(curves.shape[0])]]
    headers = ["timesteps", "mean_reward", "ci95_low", "ci95_high", *[f"run{idx + 1:02d}" for idx in range(curves.shape[0])]]
    table = np.column_stack(columns)
    np.savetxt(path, table, delimiter=",", header=",".join(headers), comments="")


def group_label(mode: str, direction: str, target: str) -> str:
    if mode == "yaw":
        return f"{direction.capitalize()} yaw_rate {float(target):.1f}"
    return f"{direction.capitalize()} radius {float(target):.1f}"


def plot_groups(groups: dict[tuple[str, str, str], list[Path]], out_dir: Path, n_bootstrap: int, seed: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(groups.items(), key=lambda item: (GROUP_ORDER.get((item[0][0], item[0][1]), 99), item[0][2]))
    if not ordered:
        raise ValueError("no completed eval_reward_csv rows found")

    ncols = 2
    nrows = max(1, int(np.ceil(len(ordered) / ncols)))
    fig_height = 7.5 if nrows == 2 else 3.7 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.4, fig_height), dpi=220, sharex=True, sharey=True)
    axes_flat = np.asarray(axes).ravel()
    summary_rows = []

    for ax, ((mode, direction, target), paths) in zip(axes_flat, ordered):
        steps, curves = common_series(paths)
        mean, low, high = bootstrap_ci(curves, n_bootstrap=n_bootstrap, seed=seed)
        label = group_label(mode, direction, target)
        color = plt.rcParams["axes.prop_cycle"].by_key()["color"][len(summary_rows) % 10]
        ax.fill_between(steps, low, high, color=color, alpha=0.22, linewidth=0)
        ax.plot(steps, mean, color=color, linewidth=2.3, label=f"{label}, n={curves.shape[0]}")
        ax.set_title("Eval mean reward over training steps", fontsize=10)
        ax.set_xlabel("Step")
        ax.set_ylabel("Eval mean reward")
        ax.grid(True, alpha=0.28)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.92)
        ax.ticklabel_format(style="sci", axis="x", scilimits=(6, 6))

        csv_path = out_dir / f"paper_reward_{mode}_{direction}_{str(target).replace('.', 'p')}.csv"
        write_group_csv(csv_path, steps, mean, low, high, curves)
        summary_rows.append((mode, direction, target, len(paths), csv_path))

    for ax in axes_flat[len(ordered) :]:
        ax.axis("off")

    fig.tight_layout()
    png_path = out_dir / "paper_reward_mean_ci95.png"
    fig.savefig(png_path)
    plt.close(fig)

    with (out_dir / "paper_reward_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", "direction", "target", "run_count", "group_csv"])
        writer.writerows(summary_rows)
    return png_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create paper-style mean reward plots with translucent 95% CI bands.")
    parser.add_argument("--summary-csv", type=Path, default=Path("outputs/batch_logs/turning_paper_40_summary.csv"))
    parser.add_argument("--reward-dir", type=Path, default=None, help="Directory containing *_eval_reward.csv files.")
    parser.add_argument("--target", type=float, default=None, help="Optional target filter, for example 0.5.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/csv_png/paper_reward_summary"))
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260618)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groups: dict[tuple[str, str, str], list[Path]] = defaultdict(list)
    source_rows = rows_from_reward_dir(args.reward_dir, args.target) if args.reward_dir is not None else rows_from_summary(args.summary_csv)
    for row in source_rows:
        key = (row["mode"], row["direction"], row["target"])
        groups[key].append(Path(row["eval_reward_csv"]))
    png_path = plot_groups(groups, args.out_dir, args.bootstrap, args.seed)
    print(png_path)


if __name__ == "__main__":
    main()
