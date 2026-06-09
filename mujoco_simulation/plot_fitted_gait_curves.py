from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


GAITS = ("straight", "turn_left", "turn_right", "spin_left", "spin_right")

REAL_VIDEO_CONFIGS = (
    {
        "stem": "clean_v_20260608_141203",
        "label": "real_turn_1",
        "kind": "turn",
    },
    {
        "stem": "clean_v_20260608_141254",
        "label": "real_turn_2",
        "kind": "turn",
    },
    {
        "stem": "clean_v_20260607_234118",
        "label": "real_straight",
        "kind": "straight",
    },
)

TANK_FORWARD_MIN = 0.0
TANK_FORWARD_MAX = 3.0
TANK_LATERAL_HALF = 0.75
START_FORWARD_M = 0.60
START_LATERAL_M = 0.0


def parse_args():
    parser = argparse.ArgumentParser(description="Plot fitted sim curves and import all three real videos.")
    parser.add_argument("--sim-dir", type=Path, default=Path("outputs/fixed_gait_trajectories_3x1_5"))
    parser.add_argument("--video-analysis-dir", type=Path, default=Path("outputs/video_analysis"))
    parser.add_argument("--recordings-dir", type=Path, default=Path("../Release/python_backend/recordings"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/fitted_curve_comparison"))
    parser.add_argument("--sim-only", action="store_true", help="Only plot MuJoCo sim fitted curves. Skip real videos.")
    return parser.parse_args()


def resolve_from_cwd(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def rotate_sim_xy(xy: np.ndarray) -> np.ndarray:
    return np.column_stack((-xy[:, 1], xy[:, 0]))


def fit_circle_xy(xy: np.ndarray):
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
    center, radius, rmse, theta = fit_circle_xy(xy)
    span = float(abs(theta[-1] - theta[0]))
    delta = xy[-1] - xy[0]
    nearly_straight = abs(delta[0]) < 0.08 * max(abs(delta[1]), 1e-9) and abs(delta[1]) > 0.5
    if not force_circle and (nearly_straight or radius > 20.0 or span < 0.12):
        curve = fit_line_curve(xy, count=count)
        return curve, {"kind": "line", "radius": None, "rmse": rmse, "arc_deg": 0.0}
    angles = np.linspace(float(theta[0]), float(theta[-1]), count)
    curve = np.column_stack((center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)))
    return curve, {"kind": "circle", "radius": radius, "rmse": rmse, "arc_deg": float(np.degrees(span))}


def trajectory_metrics(arr: np.ndarray, xy_view: np.ndarray) -> dict:
    if arr.shape[0] < 2:
        return {
            "duration_s": 0.0,
            "path_distance_m": 0.0,
            "straight_distance_m": 0.0,
            "mean_speed_m_s": 0.0,
            "forward_displacement_m": 0.0,
            "mean_forward_speed_m_s": 0.0,
            "lateral_drift_m": 0.0,
        }

    t = arr[:, 0].astype(np.float64)
    duration = float(t[-1] - t[0])
    if duration <= 1e-9:
        duration = float(arr.shape[0] - 1)

    step_distance = np.hypot(np.diff(xy_view[:, 0]), np.diff(xy_view[:, 1]))
    path_distance = float(np.sum(step_distance))
    straight_distance = float(np.linalg.norm(xy_view[-1] - xy_view[0]))
    forward_displacement = float(xy_view[-1, 1] - xy_view[0, 1])
    lateral_drift = float(xy_view[-1, 0] - xy_view[0, 0])

    return {
        "duration_s": duration,
        "path_distance_m": path_distance,
        "straight_distance_m": straight_distance,
        "mean_speed_m_s": path_distance / duration,
        "forward_displacement_m": forward_displacement,
        "mean_forward_speed_m_s": forward_displacement / duration,
        "lateral_drift_m": lateral_drift,
    }


def draw_rotated_tank(ax):
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
        metrics = trajectory_metrics(arr, xy)

        color = colors[idx % len(colors)]
        ax.plot(curve[:, 0], curve[:, 1], color=color, linewidth=2.6, label=name)
        ax.scatter([xy[0, 0]], [xy[0, 1]], s=26, color=color, edgecolor="black", zorder=4)
        ax.scatter([xy[-1, 0]], [xy[-1, 1]], s=44, marker="x", color=color, linewidth=2.0, zorder=4)
        rows.append({"name": name, **fit, **metrics})

    ax.set_title("MuJoCo fitted curves, rotated to camera view")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "sim_fitted_curves_rotated.png")
    plt.close(fig)

    for idx, name in enumerate(GAITS):
        arr = np.loadtxt(sim_dir / f"{name}_trajectory.csv", delimiter=",", skiprows=1)
        xy = rotate_sim_xy(arr[:, 1:3])
        curve, fit = fitted_curve(xy)
        metrics = trajectory_metrics(arr, xy)

        fig, ax = plt.subplots(figsize=(4.8, 8.0), dpi=170)
        draw_rotated_tank(ax)
        color = colors[idx % len(colors)]
        ax.plot(curve[:, 0], curve[:, 1], color=color, linewidth=3.0)
        ax.scatter([xy[0, 0]], [xy[0, 1]], s=34, color=color, edgecolor="black", zorder=4)
        ax.scatter([xy[-1, 0]], [xy[-1, 1]], s=52, marker="x", color=color, linewidth=2.2, zorder=4)
        radius_text = "line" if fit["radius"] is None else f"R={fit['radius']:.3f} m"
        speed_text = f"v={metrics['mean_speed_m_s']:.3f} m/s"
        ax.set_title(f"{name} fitted curve ({radius_text}, {speed_text})")
        fig.tight_layout()
        fig.savefig(out_dir / f"sim_{name}_fitted_rotated.png")
        plt.close(fig)

    (out_dir / "sim_fitted_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def points_from_summary(summary_path: Path) -> np.ndarray:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    points = data.get("cleaned_points", [])
    arr = np.asarray(points, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def point_distance_metrics(points: np.ndarray) -> dict:
    if points.shape[0] < 2:
        return {
            "point_count": int(points.shape[0]),
            "duration_s": 0.0,
            "straight_distance_px": 0.0,
            "path_distance_px": 0.0,
            "net_dx_px": 0.0,
            "net_dy_px": 0.0,
        }

    t = points[:, 0]
    x = points[:, 1]
    y = points[:, 2]

    duration = float(t[-1] - t[0])
    net_dx = float(x[-1] - x[0])
    net_dy = float(y[-1] - y[0])
    straight_distance = float(np.hypot(net_dx, net_dy))
    path_distance = float(np.sum(np.hypot(np.diff(x), np.diff(y))))

    return {
        "point_count": int(points.shape[0]),
        "duration_s": duration,
        "straight_distance_px": straight_distance,
        "path_distance_px": path_distance,
        "net_dx_px": net_dx,
        "net_dy_px": net_dy,
    }


def draw_no_points_frame(video_path: Path, out_dir: Path, label: str, kind: str, summary_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()

    if not ok:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    cv2.putText(
        frame,
        f"{label}: no tracked points",
        (40, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        3,
        cv2.LINE_AA,
    )

    stem = video_path.stem
    out_img = out_dir / f"video_{stem}_no_points.png"
    out_json = out_dir / f"video_{stem}_no_points_summary.json"

    summary = {
        "label": label,
        "kind": kind,
        "video": str(video_path),
        "summary": str(summary_path),
        "point_count": 0,
        "status": "no tracked points",
        "straight_distance_px": 0.0,
        "path_distance_px": 0.0,
    }

    cv2.imwrite(str(out_img), frame)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_img, summary


def draw_real_result(video_path: Path, summary_path: Path, out_dir: Path, label: str, kind: str):
    points = points_from_summary(summary_path)
    if points.shape[0] < 2:
        return draw_no_points_frame(video_path, out_dir, label, kind, summary_path)

    xy = points[:, 1:3]
    distance_metrics = point_distance_metrics(points)

    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read {video_path}")

    stem = video_path.stem
    summary = {
        "label": label,
        "kind": kind,
        "video": str(video_path),
        "summary": str(summary_path),
        **distance_metrics,
    }

    poly = np.round(xy).astype(np.int32)

    if kind == "turn":
        curve, fit = fitted_curve(xy, count=260, force_circle=True)
        curve_i = np.round(curve).astype(np.int32)
        cv2.polylines(frame, [curve_i], False, (0, 0, 255), 6, cv2.LINE_AA)
        if len(poly) >= 2:
            cv2.polylines(frame, [poly], False, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, tuple(poly[0]), 9, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(frame, tuple(poly[-1]), 9, (0, 0, 255), -1, cv2.LINE_AA)

        summary.update(
            {
                "fit_kind": fit["kind"],
                "radius_px": fit["radius"],
                "rmse_px": fit["rmse"],
                "arc_deg": fit["arc_deg"],
            }
        )

        radius_text = "nan" if fit["radius"] is None else f"{fit['radius']:.1f} px"
        cv2.putText(frame, f"{label}: R={radius_text}", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
        out_img = out_dir / f"video_{stem}_fit_curve.png"
        out_json = out_dir / f"video_{stem}_fit_summary.json"

    else:
        if len(poly) >= 2:
            cv2.polylines(frame, [poly], False, (0, 255, 255), 4, cv2.LINE_AA)
            start = tuple(poly[0])
            end = tuple(poly[-1])
            cv2.circle(frame, start, 10, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(frame, end, 10, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.line(frame, start, end, (255, 0, 0), 3, cv2.LINE_AA)

        cv2.putText(frame, f"{label}: distance={distance_metrics['straight_distance_px']:.1f} px", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
        out_img = out_dir / f"video_{stem}_straight_distance.png"
        out_json = out_dir / f"video_{stem}_straight_distance_summary.json"

    cv2.imwrite(str(out_img), frame)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_img, summary


def import_real_videos(video_analysis_dir: Path, recordings_dir: Path, out_dir: Path):
    rows = []

    for config in REAL_VIDEO_CONFIGS:
        stem = config["stem"]
        label = config["label"]
        kind = config["kind"]
        summary_path = video_analysis_dir / stem / "tracked_center_summary_cleaned_physical.json"
        video_path = recordings_dir / f"{stem}.mp4"

        if not summary_path.exists() or not video_path.exists():
            print(f"skip real {label}: missing summary or video")
            print(f"  summary={summary_path}")
            print(f"  video={video_path}")
            continue

        out_img, summary = draw_real_result(video_path, summary_path, out_dir, label, kind)
        rows.append(summary)
        print(out_img)

        if summary.get("status") == "no tracked points":
            print(f"{label}: no tracked points; check marker color / threshold / video file")
            continue

        if kind == "turn":
            radius = "nan" if summary.get("radius_px") is None else f"{summary['radius_px']:.3f}px"
            print(
                f"{label}: radius={radius} "
                f"arc={summary.get('arc_deg', 0.0):.1f}deg "
                f"rmse={summary.get('rmse_px', 0.0):.3f}px "
                f"points={summary['point_count']}"
            )
        else:
            print(
                f"{label}: straight_distance={summary['straight_distance_px']:.3f}px "
                f"path_distance={summary['path_distance_px']:.3f}px "
                f"duration={summary['duration_s']:.3f}s "
                f"points={summary['point_count']}"
            )

    (out_dir / "real_video_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def main():
    args = parse_args()

    sim_dir = resolve_from_cwd(args.sim_dir)
    out_dir = resolve_from_cwd(args.out_dir)
    video_analysis_dir = resolve_from_cwd(args.video_analysis_dir)
    recordings_dir = resolve_from_cwd(args.recordings_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    sim_rows = plot_sim_curves(sim_dir, out_dir)
    print(out_dir / "sim_fitted_curves_rotated.png")

    for row in sim_rows:
        radius = "line" if row["radius"] is None else f"{row['radius']:.3f}m"
        print(
            f"{row['name']}: {row['kind']} radius={radius} "
            f"arc={row['arc_deg']:.1f}deg rmse={row['rmse']:.4f} "
            f"speed={row['mean_speed_m_s']:.3f}m/s "
            f"forward_speed={row['mean_forward_speed_m_s']:.3f}m/s"
        )

    straight_row = next((row for row in sim_rows if row["name"] == "straight"), None)
    if straight_row is not None:
        print("\nSim straight swimming speed summary")
        print(f"  mean path speed       = {straight_row['mean_speed_m_s']:.4f} m/s")
        print(f"  mean forward speed    = {straight_row['mean_forward_speed_m_s']:.4f} m/s")
        print(f"  forward displacement  = {straight_row['forward_displacement_m']:.4f} m")
        print(f"  lateral drift         = {straight_row['lateral_drift_m']:.4f} m")
        print(f"  duration              = {straight_row['duration_s']:.4f} s")

    if args.sim_only:
        return

    print("\nReal video summary")
    import_real_videos(video_analysis_dir, recordings_dir, out_dir)


if __name__ == "__main__":
    main()
