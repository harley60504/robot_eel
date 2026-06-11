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
        "stem": "clean_v_20260607_233739",
        "label": "real_straight",
        "kind": "straight",
    },
)

TANK_FORWARD_MIN = 0.0
TANK_FORWARD_MAX = 3.0
TANK_LATERAL_HALF = 0.75
START_FORWARD_M = 0.60
START_LATERAL_M = 0.0
PX_PER_M = 875.0 / 1.5


def parse_args():
    parser = argparse.ArgumentParser(description="Plot fitted sim curves and import selected real videos.")
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


def sim_metric_text(name: str, fit: dict, metrics: dict) -> str:
    if fit["radius"] is None or name == "straight":
        return (
            "R = ∞ m\n"
            f"d_forward = {metrics['forward_displacement_m']:.3f} m\n"
            f"v_forward = {metrics['mean_forward_speed_m_s']:.3f} m/s\n"
            f"lateral drift = {metrics['lateral_drift_m']:.3f} m"
        )
    return (
        f"R = {fit['radius']:.3f} m\n"
        f"v = {metrics['mean_speed_m_s']:.3f} m/s\n"
        f"arc = {fit['arc_deg']:.1f} deg\n"
        f"RMSE = {fit['rmse']:.3f}"
    )


def add_sim_metric_box(ax, text: str) -> None:
    ax.text(
        0.28,
        0.97,
        text,
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.88, edgecolor="#cccccc"),
    )


def sim_trajectory_files(sim_dir: Path) -> list[Path]:
    files = sorted(sim_dir.glob("*_trajectory.csv"))
    fixed_order = {name: idx for idx, name in enumerate(GAITS)}
    return sorted(files, key=lambda p: fixed_order.get(p.name.removesuffix("_trajectory.csv"), 9999))


def plot_sim_curves(sim_dir: Path, out_dir: Path):
    rows = []
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    files = sim_trajectory_files(sim_dir)
    if not files:
        print(f"skip sim curves: no *_trajectory.csv in {sim_dir}")
        return rows

    fig, ax = plt.subplots(figsize=(5.2, 8.2), dpi=170)
    draw_rotated_tank(ax)

    for idx, csv_path in enumerate(files):
        name = csv_path.name.removesuffix("_trajectory.csv")
        arr = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[0] < 2:
            continue
        xy = rotate_sim_xy(arr[:, 1:3])
        curve, fit = fitted_curve(xy)
        metrics = trajectory_metrics(arr, xy)

        color = colors[idx % len(colors)]
        ax.plot(curve[:, 0], curve[:, 1], color=color, linewidth=2.6, label=name)
        ax.scatter([xy[0, 0]], [xy[0, 1]], s=26, color=color, edgecolor="black", zorder=4)
        ax.scatter([xy[-1, 0]], [xy[-1, 1]], s=44, marker="x", color=color, linewidth=2.0, zorder=4)
        rows.append({"name": name, **fit, **metrics})

    if not rows:
        print(f"skip sim curves: trajectory files in {sim_dir} had too few points")
        plt.close(fig)
        return rows

    ax.set_title("MuJoCo fitted curves, rotated to camera view")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "sim_fitted_curves_rotated.png")
    plt.close(fig)

    for idx, csv_path in enumerate(files):
        name = csv_path.name.removesuffix("_trajectory.csv")
        arr = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[0] < 2:
            continue
        xy = rotate_sim_xy(arr[:, 1:3])
        curve, fit = fitted_curve(xy)
        metrics = trajectory_metrics(arr, xy)

        fig, ax = plt.subplots(figsize=(4.8, 8.0), dpi=170)
        draw_rotated_tank(ax)
        color = colors[idx % len(colors)]
        ax.plot(curve[:, 0], curve[:, 1], color=color, linewidth=3.0)
        ax.scatter([xy[0, 0]], [xy[0, 1]], s=34, color=color, edgecolor="black", zorder=4)
        ax.scatter([xy[-1, 0]], [xy[-1, 1]], s=52, marker="x", color=color, linewidth=2.2, zorder=4)
        ax.set_title(f"{name} fitted curve")
        add_sim_metric_box(ax, sim_metric_text(name, fit, metrics))
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


def safe_read_summary(summary_path: Path) -> dict:
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def real_speed_from_summary(summary_data: dict, distance_metrics: dict) -> float | None:
    if summary_data.get("forward_speed_m_s") is not None:
        return float(summary_data["forward_speed_m_s"])
    duration = float(distance_metrics.get("duration_s", 0.0))
    if duration <= 1e-9:
        return None
    return float(distance_metrics.get("path_distance_px", 0.0)) / PX_PER_M / duration


def draw_metric_box(frame, lines: list[str], font_scale: float = 0.9) -> None:
    h, w = frame.shape[:2]
    pad = 14
    line_gap = 12
    thickness = 2
    outline = 4
    font = cv2.FONT_HERSHEY_SIMPLEX

    sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    text_w = max((s[0] for s in sizes), default=0)
    text_h_total = sum(s[1] for s in sizes) + line_gap * max(0, len(lines) - 1)
    box_w = text_w + pad * 2
    box_h = text_h_total + pad * 2

    left = 12
    top = max(10, int(0.03 * h))
    right = left + box_w
    bottom = top + box_h

    left = max(10, left)
    top = max(10, top)
    right = min(w - 10, left + box_w)
    bottom = min(h - 10, top + box_h)

    overlay = frame.copy()
    cv2.rectangle(overlay, (left, top), (right, bottom), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    cv2.rectangle(frame, (left, top), (right, bottom), (200, 200, 200), 2)

    y = top + pad
    for line, size in zip(lines, sizes):
        baseline_y = y + size[1]
        x = right - pad - size[0]
        cv2.putText(frame, line, (x, baseline_y), font, font_scale, (0, 0, 0), outline, cv2.LINE_AA)
        cv2.putText(frame, line, (x, baseline_y), font, font_scale, (40, 40, 40), thickness, cv2.LINE_AA)
        y = baseline_y + line_gap


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
    summary_data = safe_read_summary(summary_path)

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

        radius_m = None if fit["radius"] is None else float(fit["radius"]) / PX_PER_M
        speed_m_s = real_speed_from_summary(summary_data, distance_metrics)
        summary.update(
            {
                "fit_kind": fit["kind"],
                "radius_px": fit["radius"],
                "radius_m": radius_m,
                "rmse_px": fit["rmse"],
                "rmse_m": fit["rmse"] / PX_PER_M,
                "arc_deg": fit["arc_deg"],
                "speed_m_s": speed_m_s,
            }
        )

        radius_text = "nan" if radius_m is None else f"{radius_m:.3f} m"
        speed_text = "nan" if speed_m_s is None else f"{speed_m_s:.3f} m/s"
        draw_metric_box(frame, [f"R = {radius_text}", f"v = {speed_text}", f"arc = {fit['arc_deg']:.1f} deg", f"RMSE = {fit['rmse']:.1f}px"])
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

        speed_m_s = real_speed_from_summary(summary_data, distance_metrics)
        forward_m = summary_data.get("forward_distance_m")
        if forward_m is None:
            forward_m = distance_metrics["straight_distance_px"] / PX_PER_M
        summary.update(
            {
                "fit_kind": "line",
                "radius_m": None,
                "forward_distance_m": float(forward_m),
                "forward_speed_m_s": speed_m_s,
                "rmse_px": summary_data.get("rmse_px"),
            }
        )
        speed_text = "nan" if speed_m_s is None else f"{speed_m_s:.3f} m/s"
        draw_metric_box(frame, ["R = ∞ m", f"v = {speed_text}", f"forward = {float(forward_m):.3f} m", f"RMSE = {summary_data.get('rmse_px', 0.0):.1f}px"])
        out_img = out_dir / f"video_{stem}_straight_distance.png"
        out_json = out_dir / f"video_{stem}_straight_distance_summary.json"

    cv2.imwrite(str(out_img), frame)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_img, summary


def dynamic_real_configs(video_analysis_dir: Path, recordings_dir: Path):
    summaries = sorted(video_analysis_dir.glob("*/tracked_center_summary_cleaned_physical.json"))
    configs = []
    for summary_path in summaries:
        data = safe_read_summary(summary_path)
        stem = data.get("video_stem") or summary_path.parent.name
        video_value = data.get("video")
        video_path = Path(video_value) if video_value else recordings_dir / f"{stem}.mp4"
        if not video_path.exists():
            fallback = recordings_dir / f"{stem}.mp4"
            video_path = fallback if fallback.exists() else video_path
        motion_class = str(data.get("motion_class") or data.get("fit_kind") or "").lower()
        kind = "straight" if motion_class in {"line", "straight"} else "turn"
        configs.append({"stem": stem, "label": stem, "kind": kind, "summary_path": summary_path, "video_path": video_path})
    return configs


def import_real_videos(video_analysis_dir: Path, recordings_dir: Path, out_dir: Path):
    rows = []
    dynamic_configs = dynamic_real_configs(video_analysis_dir, recordings_dir)
    if dynamic_configs:
        configs = dynamic_configs
    else:
        configs = []
        for config in REAL_VIDEO_CONFIGS:
            stem = config["stem"]
            configs.append(
                {
                    "stem": stem,
                    "label": config["label"],
                    "kind": config["kind"],
                    "summary_path": video_analysis_dir / stem / "tracked_center_summary_cleaned_physical.json",
                    "video_path": recordings_dir / f"{stem}.mp4",
                }
            )

    for config in configs:
        label = config["label"]
        kind = config["kind"]
        summary_path = config["summary_path"]
        video_path = config["video_path"]

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
            radius = "nan" if summary.get("radius_m") is None else f"{summary['radius_m']:.3f}m"
            speed = "nan" if summary.get("speed_m_s") is None else f"{summary['speed_m_s']:.3f}m/s"
            print(
                f"{label}: radius={radius} "
                f"speed={speed} "
                f"arc={summary.get('arc_deg', 0.0):.1f}deg "
                f"rmse={summary.get('rmse_px', 0.0):.3f}px "
                f"points={summary['point_count']}"
            )
        else:
            speed = "nan" if summary.get("forward_speed_m_s") is None else f"{summary['forward_speed_m_s']:.3f}m/s"
            print(
                f"{label}: R=inf "
                f"speed={speed} "
                f"forward={summary.get('forward_distance_m', 0.0):.3f}m "
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
    if sim_rows:
        print(out_dir / "sim_fitted_curves_rotated.png")

    for row in sim_rows:
        radius = "R=∞m" if row["radius"] is None else f"R={row['radius']:.3f}m"
        if row["radius"] is None or row["name"] == "straight":
            print(
                f"{row['name']}: line R=∞m "
                f"forward_distance={row['forward_displacement_m']:.3f}m "
                f"forward_speed={row['mean_forward_speed_m_s']:.3f}m/s "
                f"lateral_drift={row['lateral_drift_m']:.3f}m "
                f"rmse={row['rmse']:.4f}"
            )
        else:
            print(
                f"{row['name']}: {row['kind']} {radius} "
                f"arc={row['arc_deg']:.1f}deg rmse={row['rmse']:.4f} "
                f"speed={row['mean_speed_m_s']:.3f}m/s"
            )

    straight_row = next((row for row in sim_rows if row["name"] == "straight"), None)
    if straight_row is not None:
        print("\nSim straight swimming speed summary")
        print(f"  mean path speed       = {straight_row['mean_speed_m_s']:.4f} m/s  (includes lateral oscillation)")
        print(f"  mean forward speed    = {straight_row['mean_forward_speed_m_s']:.4f} m/s  (used for straight figure)")
        print(f"  forward displacement  = {straight_row['forward_displacement_m']:.4f} m")
        print(f"  lateral drift         = {straight_row['lateral_drift_m']:.4f} m")
        print(f"  duration              = {straight_row['duration_s']:.4f} s")

    if args.sim_only:
        return

    print("\nReal video summary")
    import_real_videos(video_analysis_dir, recordings_dir, out_dir)


if __name__ == "__main__":
    main()
