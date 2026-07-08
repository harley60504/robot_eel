from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_fixed_gait_trajectories import run_gait, summarize
from sim_config import DEFAULT_START_X, DEFAULT_START_Y, EEL_MODEL_XML


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs" / "tail_amp_scale_sweep"
BASE_GAITS = {
    "left": ROOT / "gaits" / "turn_left.json",
    "right": ROOT / "gaits" / "turn_right.json",
}
MULTIPLIERS = (1.00, 1.10, 1.20, 1.30, 1.40)
SECONDS = 18.0
WARMUP_SECONDS = 2.0


def build_variant(base: dict, direction: str, profile: str, multiplier: float) -> dict:
    gait = json.loads(json.dumps(base))
    scales = [float(value) for value in gait["amp_scales"]]
    if profile == "tail2":
        indices = (4, 5)
    elif profile == "tail3":
        indices = (3, 4, 5)
    else:
        raise ValueError(f"unknown profile: {profile}")

    for idx in indices:
        scales[idx] = round(scales[idx] * multiplier, 4)

    gait["name"] = f"tail_amp_{direction}_{profile}_x{str(multiplier).replace('.', 'p')}"
    gait["amp_scales"] = scales
    gait["source"] = {
        "type": "tail_amp_scale_sweep",
        "base_gait": base.get("name"),
        "direction": direction,
        "profile": profile,
        "tail_multiplier": multiplier,
        "changed_joint_indices_1based": [idx + 1 for idx in indices],
    }
    return gait


def write_gait(path: Path, gait: dict) -> None:
    path.write_text(json.dumps(gait, indent=2), encoding="utf-8")


def finite_radius(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else float("nan")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gait_dir = OUT_DIR / "gaits"
    gait_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    results: list[tuple[dict[str, object], np.ndarray]] = []

    for direction, base_path in BASE_GAITS.items():
        base = json.loads(base_path.read_text(encoding="utf-8"))
        for profile in ("tail2", "tail3"):
            for multiplier in MULTIPLIERS:
                gait = build_variant(base, direction, profile, multiplier)
                gait_path = gait_dir / f"{gait['name']}.json"
                write_gait(gait_path, gait)
                _, arr, hit_wall = run_gait(
                    Path(EEL_MODEL_XML),
                    gait_path,
                    SECONDS,
                    DEFAULT_START_X,
                    DEFAULT_START_Y,
                    wall_collision=False,
                    stop_on_wall=False,
                )
                summary = summarize(arr, WARMUP_SECONDS)
                traj_path = OUT_DIR / f"{gait['name']}_trajectory.csv"
                np.savetxt(traj_path, arr, delimiter=",", header="time,x,y,yaw", comments="")
                row = {
                    "name": gait["name"],
                    "direction": direction,
                    "profile": profile,
                    "tail_multiplier": multiplier,
                    "amp_scales": " ".join(f"{value:.4g}" for value in gait["amp_scales"]),
                    "duration_s": summary["duration_s"],
                    "yaw_change_deg": summary["yaw_change_deg"],
                    "yaw_rate_rad_s": summary["yaw_rate_rad_s"],
                    "abs_yaw_rate_rad_s": abs(float(summary["yaw_rate_rad_s"])),
                    "turn_radius_m": finite_radius(summary["turn_radius_m"]),
                    "speed_m_s": summary["speed_m_s"],
                    "forward_speed_m_s": summary["forward_speed_m_s"],
                    "lateral_speed_m_s": summary["lateral_speed_m_s"],
                    "dx": summary["dx"],
                    "dy": summary["dy"],
                    "hit_wall": hit_wall,
                    "gait_json": str(gait_path),
                    "trajectory_csv": str(traj_path),
                }
                rows.append(row)
                results.append((row, arr))
                print(
                    f"{gait['name']}: yaw_rate={row['yaw_rate_rad_s']:.3f} rad/s, "
                    f"R={row['turn_radius_m']:.3f} m, speed={row['speed_m_s']:.3f} m/s"
                )

    csv_path = OUT_DIR / "tail_amp_scale_sweep_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (OUT_DIR / "tail_amp_scale_sweep_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    plot_metric(rows, "abs_yaw_rate_rad_s", "abs yaw rate (rad/s)", OUT_DIR / "tail_amp_scale_yaw_rate.png")
    plot_metric(rows, "turn_radius_m", "turn radius (m)", OUT_DIR / "tail_amp_scale_turn_radius.png")
    plot_trajectories(results, OUT_DIR / "tail_amp_scale_trajectories.png")
    print(csv_path)


def plot_metric(rows: list[dict[str, object]], metric: str, ylabel: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), dpi=180, sharey=False)
    for ax, direction in zip(axes, ("left", "right")):
        for profile, marker in (("tail2", "o"), ("tail3", "s")):
            subset = [row for row in rows if row["direction"] == direction and row["profile"] == profile]
            x = [float(row["tail_multiplier"]) for row in subset]
            y = [float(row[metric]) for row in subset]
            ax.plot(x, y, marker=marker, linewidth=1.8, label=profile)
        ax.set_title(f"{direction} turn")
        ax.set_xlabel("tail amp_scale multiplier")
        ax.grid(True, alpha=0.25)
        ax.legend()
    axes[0].set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_trajectories(results: list[tuple[dict[str, object], np.ndarray]], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.4), dpi=180)
    axes_by_key = {
        ("left", "tail2"): axes[0, 0],
        ("left", "tail3"): axes[0, 1],
        ("right", "tail2"): axes[1, 0],
        ("right", "tail3"): axes[1, 1],
    }
    for row, arr in results:
        ax = axes_by_key[(row["direction"], row["profile"])]
        label = f"x{float(row['tail_multiplier']):.1f}"
        ax.plot(arr[:, 1], arr[:, 2], linewidth=1.2, label=label)
        ax.scatter([arr[0, 1]], [arr[0, 2]], s=14, color="black", zorder=3)
        ax.scatter([arr[-1, 1]], [arr[-1, 2]], s=22, marker="x", color="black", zorder=3)
    for (direction, profile), ax in axes_by_key.items():
        ax.set_title(f"{direction} {profile}")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
