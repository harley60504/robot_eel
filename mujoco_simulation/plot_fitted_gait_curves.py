from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


GAITS = ("straight", "turn_left", "turn_right", "spin_left", "spin_right")

# Current 3 m x 1.5 m tank coordinate convention:
#   MuJoCo x = forward direction, 0.0 m to 3.0 m
#   MuJoCo y = lateral direction, -0.75 m to 0.75 m
# rotate_sim_xy() maps this into plot/camera view:
#   view x = lateral = -sim_y
#   view y = forward = sim_x
TANK_FORWARD_MIN = 0.0
TANK_FORWARD_MAX = 3.0
TANK_LATERAL_HALF = 0.75
START_FORWARD_M = 0.60
START_LATERAL_M = 0.0


def parse_args():
    parser = argparse.ArgumentParser(description="Plot fitted, rotated gait curves for sim and video comparison.")
    parser.add_argument("--sim-dir", type=Path, default=Path("outputs/fixed_gait_trajectories_3x1_5"))
    parser.add_argument("--video-summary", type=Path, default=Path("outputs/video_analysis/clean_v_20260608_141254/tracked_center_summary_cleaned_physical.json"))
    parser.add_argument("--video", type=Path, default=Path("Release/python_backend/recordings/clean_v_20260608_141254.mp4"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/fitted_curve_comparison"))
    parser.add_argument("--sim-only", action="store_true", help="Only plot MuJoCo sim fitted curves. Skip video comparison.")
    return parser.parse_args()


def rotate_sim_xy(xy: np.ndarray) -> np.ndarray:
    # MuJoCo forward is x. Rotate into camera-style view:
    #   view lateral = -sim_y
    #   view forward = sim_x
    return np.column_stack((-xy[:, 1], xy[:, 0]))


def fit_circle(xy: np.ndarray):
    x = xy[:, 0]
    y = xy[:, 1]
    a = np.column_stack((2 * x, 2 * y, np.ones_like(x)))
    b = x * x + y * y
    cx, cy, k = np.linalg.lstsq(a, b, rcond=None)[0]
    radius = float(np.sqrt(max(0.0, k + cx * cx + cy * cy)))
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    rmse = float(np.sqrt(np.mean((dist - radius) ** 2)))
    theta = np.unwrap(np.arctan2(y - cy, x - cx))
    return np.array([float(cx), float(cy)]), radius, rmse, theta


def fit_line_curve(xy: np.ndarray, count: int = 220) -> np.ndarray:
    direction = xy[-1] - xy[0]
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        return np.repeat(xy[:1], count, axis=0)
    unit = direction / norm
    scalar = (xy - xy[0]) @ unit
    s = np.linspace(float(scalar.min()), float(scalar.max()), count)
    return xy[0] + np.outer(s, unit)


def fitted_curve(xy: np.ndarray, count: int = 240, force_circle: bool = False):
    center, radius, rmse, theta = fit_circle(xy)
    span = float(abs(theta[-1] - theta[0]))
    delta = xy[-1] - xy[0]
    nearly_straight = abs(delta[0]) < 0.08 * max(abs(delta[1]), 1e-9) and abs(delta[1]) > 0.5
    if not force_circle and (nearly_straight or radius > 20.0 or span < 0.12):
        curve = fit_line_curve(xy, count=count)
        return curve, {"kind": "line", "radius": None, "rmse": rmse, "arc_deg": 0.0}
    angles = np.linspace(float(theta[0]), float(theta[-1]), count)
    curve = np.column_stack((center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)))
    return curve, {"kind": "circle", "radius": radius, "rmse": rmse, "arc_deg": float(np.degrees(span))}


def draw_rotated_tank(ax):
    # View coordinates after rotate_sim_xy:
    #   horizontal = lateral, -0.75 m to +0.75 m
    #   vertical = forward, 0.0 m to 3.0 m
    ax.add_patch(
        plt.Rectangle(
            (-TANK_LATERAL_HALF, TANK_FORWARD_MIN),
            2.0 * TANK_LATERAL_HALF,
            TANK_FORWARD_MAX - TANK_FORWARD_MIN,
            fill=False,
            color="#7f1d1d",
            linewidth=1.5,
        )
    )
    ax.scatter([START_LATERAL_M], [START_FORWARD_M], s=32, color="#22c55e", edgecolor="black", zorder=4)
    ax.set_xlim(-TANK_LATERAL_HALF - 0.10, TANK_LATERAL_HALF + 0.10)
    ax.set_ylim(TANK_FORWARD_MIN - 0.10, TANK_FORWARD_MAX + 0.10)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.22)
    ax.set_xlabel("lateral (m)")
    ax.set_ylabel("forward (m)")


def plot_sim_curves(sim_dir: Path, out_dir: Path):
    rows = []
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(5.2, 8.2), dpi=170)
    draw_rotated_tank(ax)
    for idx, name in enumerate(GAITS):
        arr = np.loadtxt(sim_dir / f"{name}_trajectory.csv", delimiter=",", skiprows=1)
        xy = rotate_sim_xy(arr[:, 1:3])
        curve, fit = fitted_curve(xy)
        color = colors[idx % len(colors)]
        ax.plot(curve[:, 0], curve[:, 1], color=color, linewidth=2.6, label=name)
        ax.scatter([xy[0, 0]], [xy[0, 1]], s=26, color=color, edgecolor="black", zorder=4)
        ax.scatter([xy[-1, 0]], [xy[-1, 1]], s=44, marker="x", color=color, linewidth=2.0, zorder=4)
        rows.append({"name": name, **fit})
    ax.set_title("MuJoCo fitted curves, rotated to camera view")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "sim_fitted_curves_rotated.png")
    plt.close(fig)

    for idx, name in enumerate(GAITS):
        arr = np.loadtxt(sim_dir / f"{name}_trajectory.csv", delimiter=",", skiprows=1)
        xy = rotate_sim_xy(arr[:, 1:3])
        curve, fit = fitted_curve(xy)
        fig, ax = plt.subplots(figsize=(4.8, 8.0), dpi=170)
        draw_rotated_tank(ax)
        color = colors[idx % len(colors)]
        ax.plot(curve[:, 0], curve[:, 1], color=color, linewidth=3.0)
        ax.scatter([xy[0, 0]], [xy[0, 1]], s=34, color=color, edgecolor="black", zorder=4)
        ax.scatter([xy[-1, 0]], [xy[-1, 1]], s=52, marker="x", color=color, linewidth=2.2, zorder=4)
        radius_text = "line" if fit["radius"] is None else f"R={fit['radius']:.3f} m"
        ax.set_title(f"{name} fitted curve ({radius_text})")
        fig.tight_layout()
        fig.savefig(out_dir / f"sim_{name}_fitted_rotated.png")
        plt.close(fig)

    (out_dir / "sim_fitted_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def draw_video_fit(video_path: Path, summary_path: Path, out_dir: Path):
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    points = np.array([[p[1], p[2]] for p in data["cleaned_points"]], dtype=np.float64)
    curve, fit = fitted_curve(points, count=260, force_circle=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(8 * fps)))
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read {video_path}")

    poly = np.round(curve).astype(np.int32)
    cv2.polylines(frame, [poly], False, (0, 0, 255), 7, cv2.LINE_AA)
    for idx in range(0, len(data["cleaned_points"]), 5):
        t, x, y, _ = data["cleaned_points"][idx]
        cv2.circle(frame, (int(x), int(y)), 9, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(frame, f"{t:.0f}s", (int(x) + 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    out_path = out_dir / "video_spin_left_fit_curve_only.png"
    cv2.imwrite(str(out_path), frame)
    (out_dir / "video_spin_left_fit_summary.json").write_text(json.dumps(fit, indent=2), encoding="utf-8")
    return out_path, fit


def main():
    args = parse_args()
    root = Path.cwd()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_rows = plot_sim_curves(root / args.sim_dir, out_dir)
    print(out_dir / "sim_fitted_curves_rotated.png")
    for row in sim_rows:
        radius = "line" if row["radius"] is None else f"{row['radius']:.3f}m"
        print(f"{row['name']}: {row['kind']} radius={radius} arc={row['arc_deg']:.1f}deg rmse={row['rmse']:.4f}")

    if args.sim_only:
        return

    video_summary = root / args.video_summary
    video_file = root / args.video
    if not video_summary.exists() or not video_file.exists():
        print("video comparison skipped: missing video summary or video file")
        print(f"  video_summary={video_summary}")
        print(f"  video={video_file}")
        return

    video_path, video_fit = draw_video_fit(video_file, video_summary, out_dir)
    print(video_path)
    print(f"video_spin_left: {video_fit['kind']} radius_px={video_fit['radius']:.1f} arc={video_fit['arc_deg']:.1f}deg rmse_px={video_fit['rmse']:.1f}")


if __name__ == "__main__":
    main()
