from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from sim_config import DEFAULT_START_X, DEFAULT_START_Y, EEL_MODEL_XML, RESET_X_MAX, RESET_X_MIN, RESET_Y, TANK_CENTER_X
from hopf_cpg import HopfCPG, HopfCPGParams, amp_scales_to_mu_scales, degrees_to_radians


GAIT_FILES = (
    "straight.json",
    "turn_left.json",
    "turn_right.json",
    "spin_left.json",
    "spin_right.json",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot fixed-gait MuJoCo trajectories.")
    parser.add_argument("--xml", default=EEL_MODEL_XML)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--warmup-seconds", type=float, default=0.0)
    parser.add_argument("--start-x", type=float, default=DEFAULT_START_X)
    parser.add_argument("--start-y", type=float, default=DEFAULT_START_Y)
    parser.add_argument("--gait-dir", type=Path, default=Path("gaits"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/fixed_gait_trajectories_3x1_5"))
    return parser.parse_args()


def is_wall_contact(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    for i in range(data.ncon):
        contact = data.contact[i]
        name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or ""
        name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or ""
        if name1.startswith("wall_") or name2.startswith("wall_"):
            return True
    return False


def run_gait(xml_path: Path, gait_path: Path, seconds: float, start_x: float, start_y: float):
    gait = json.loads(gait_path.read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    model.opt.gravity[:] = (0, 0, 0)
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    base_xml_pos = model.body_pos[base_body_id]
    data.qpos[0] = start_x - base_xml_pos[0]
    data.qpos[1] = start_y - base_xml_pos[1]
    mujoco.mj_forward(model, data)

    params = HopfCPGParams(
        frequency=float(gait["freq"]),
        wavelength=float(gait["wavelength"]),
        ajoint=degrees_to_radians(float(gait["ajoint"])),
        mu_scales=amp_scales_to_mu_scales(tuple(gait["amp_scales"])),
        phase_lags=tuple(gait["phase_lags"]),
        joint_bias=tuple(gait["joint_bias"]),
    )
    cpg = HopfCPG(num_joints=6, params=params)
    records = []
    steps = int(round(seconds / model.opt.timestep))
    hit_wall = False

    for _ in range(steps):
        targets = cpg.step(data.time, model.opt.timestep, params)
        data.ctrl[0:6] = np.clip(targets, -1.2, 1.2)
        mujoco.mj_step(model, data)
        base_pos = data.xpos[base_body_id].copy()
        records.append((float(data.time), float(base_pos[0]), float(base_pos[1]), float(data.qpos[2])))
        if is_wall_contact(model, data):
            hit_wall = True
            break

    arr = np.asarray(records, dtype=np.float64)
    return gait, arr, hit_wall


def summarize(arr: np.ndarray, warmup_seconds: float):
    t = arr[:, 0]
    x = arr[:, 1]
    y = arr[:, 2]
    yaw = np.unwrap(arr[:, 3])
    start_idx = int(np.argmin(np.abs(t - warmup_seconds)))
    end_idx = len(t) - 1
    dt = max(1e-9, float(t[end_idx] - t[start_idx]))
    dx = float(x[end_idx] - x[start_idx])
    dy = float(y[end_idx] - y[start_idx])
    yaw_change = float(yaw[end_idx] - yaw[start_idx])
    speed = float(np.hypot(dx, dy) / dt)
    forward_speed = dx / dt
    lateral_speed = dy / dt
    yaw_rate = yaw_change / dt
    radius = math.inf if abs(yaw_rate) < 1e-9 else speed / abs(yaw_rate)
    return {
        "dx": dx,
        "dy": dy,
        "yaw_change_rad": yaw_change,
        "yaw_change_deg": math.degrees(yaw_change),
        "yaw_rate_rad_s": yaw_rate,
        "speed_m_s": speed,
        "forward_speed_m_s": forward_speed,
        "lateral_speed_m_s": lateral_speed,
        "turn_radius_m": radius,
        "warmup_index": start_idx,
    }


def draw_environment(ax, start_x: float, start_y: float):
    ax.set_xlim(-0.1, 3.1)
    ax.set_ylim(-0.85, 0.85)
    ax.add_patch(plt.Rectangle((-1.5, -0.75), 3.0, 1.5, fill=False, color="#7f1d1d", linewidth=1.4))
    ax.scatter([0.40, 2.60, 2.60, 0.40], [-0.35, -0.35, 0.35, 0.35], s=18, color="#facc15", alpha=0.45)
    ax.scatter([start_x], [start_y], s=28, color="#22c55e", edgecolor="black", zorder=3)


def plot_one(ax, name: str, arr: np.ndarray, summary: dict, color=None):
    x = arr[:, 1]
    y = arr[:, 2]
    line, = ax.plot(x, y, linewidth=1.4, label=name, color=color)
    ax.scatter([x[0]], [y[0]], s=34, marker="o", color=line.get_color(), edgecolor="black", zorder=3)
    ax.scatter([x[-1]], [y[-1]], s=44, marker="x", color=line.get_color(), linewidth=2.0, zorder=3)
    return line


def main():
    args = parse_args()
    root = Path.cwd()
    xml_path = root / args.xml
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for file_name in GAIT_FILES:
        gait, arr, hit_wall = run_gait(xml_path, root / args.gait_dir / file_name, args.seconds, args.start_x, args.start_y)
        summary = summarize(arr, args.warmup_seconds)
        results.append((gait["name"], gait, arr, summary, hit_wall))
        np.savetxt(
            out_dir / f"{gait['name']}_trajectory.csv",
            arr,
            delimiter=",",
            header="time,x,y,yaw",
            comments="",
        )

    fig, ax = plt.subplots(figsize=(9, 5), dpi=170)
    draw_environment(ax, args.start_x, args.start_y)
    for name, _, arr, summary, _ in results:
        plot_one(ax, name, arr, summary)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Fixed gait trajectories in 3m x 1.5m tank")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    combined_png = out_dir / "fixed_gait_trajectories_3x1_5.png"
    fig.savefig(combined_png)
    plt.close(fig)

    summary_rows = []
    for name, _, arr, summary, hit_wall in results:
        fig, ax = plt.subplots(figsize=(7, 5), dpi=170)
        draw_environment(ax, args.start_x, args.start_y)
        plot_one(ax, name, arr, summary)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"{name} trajectory until wall contact")
        ax.text(
            0.02,
            0.98,
            f"time={arr[-1, 0]:.2f}s, hit_wall={hit_wall}\n"
            f"dx={summary['dx']:.3f} m, dy={summary['dy']:.3f} m\n"
            f"yaw={summary['yaw_change_deg']:.1f} deg, rate={summary['yaw_rate_rad_s']:.3f} rad/s\n"
            f"radius={summary['turn_radius_m']:.3f} m",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
            fontsize=8,
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}_trajectory.png")
        plt.close(fig)
        summary_rows.append(
            {
                "name": name,
                "duration_s": float(arr[-1, 0]),
                "hit_wall": hit_wall,
                **{
                    k: (None if isinstance(v, float) and math.isinf(v) else v)
                    for k, v in summary.items()
                    if k != "warmup_index"
                },
            }
        )

    (out_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    print(combined_png)
    print(out_dir / "summary.json")
    for row in summary_rows:
        print(
            f"{row['name']}: dx={row['dx']:.3f} dy={row['dy']:.3f} "
            f"yaw={row['yaw_change_deg']:.1f}deg radius={row['turn_radius_m']:.3f}m"
        )


if __name__ == "__main__":
    main()
