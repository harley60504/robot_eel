from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


GAITS = ("straight", "turn_left", "turn_right", "spin_left", "spin_right")
TANK_FORWARD_MIN = 0.0
TANK_FORWARD_MAX = 3.0
TANK_LATERAL_HALF = 0.75
START_FORWARD_M = 0.60
START_LATERAL_M = 0.0
PX_PER_M = 875.0 / 1.5

MIN_FINAL_POINTS = 6
MIN_FINAL_NET_PX = 45.0
MIN_FINAL_PATH_PX = 55.0
MIN_TURN_NET_PX = 60.0
MIN_TURN_ARC_DEG = 35.0
MIN_TURN_HEADING_DEG = 28.0
MIN_TURN_LATERAL_SPAN_PX = 80.0
MIN_TURN_LATERAL_DISP_PX = 70.0
MAX_TURN_STRAIGHTNESS = 0.93
MAX_TURN_RMSE_OVER_RADIUS = 0.75
MIN_TURN_SCORE = 2
STRAIGHT_FORWARD_DOMINANCE_RATIO = 1.8
MAX_STRAIGHT_HEADING_DEG = 45.0
MAX_STRAIGHT_LATERAL_DISP_PX = 85.0


def parse_args():
    parser = argparse.ArgumentParser(description="Plot fitted sim curves and final real-video motion fits.")
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


def safe_read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


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


def fit_line_xy(xy: np.ndarray, count: int = 240):
    if xy.shape[0] < 2:
        curve = np.repeat(xy[:1], count, axis=0) if xy.shape[0] else np.zeros((count, 2), dtype=float)
        return curve, {"kind": "line", "rmse": 0.0, "length": 0.0}
    center = xy.mean(axis=0)
    _, _, vh = np.linalg.svd(xy - center, full_matrices=False)
    direction = vh[0]
    if np.dot(direction, xy[-1] - xy[0]) < 0:
        direction = -direction
    projection = (xy - center) @ direction
    s = np.linspace(float(projection.min()), float(projection.max()), count)
    curve = center + np.outer(s, direction)
    residual = np.linalg.norm((xy - center) - np.outer(projection, direction), axis=1)
    return curve, {
        "kind": "line",
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "length": float(np.linalg.norm(curve[-1] - curve[0])),
        "line_start_px": curve[0].tolist(),
        "line_end_px": curve[-1].tolist(),
    }


def fitted_curve(xy: np.ndarray, count: int = 240, force_circle: bool = False):
    center, radius, rmse, theta = fit_circle_xy(xy)
    span = float(abs(theta[-1] - theta[0]))
    delta = xy[-1] - xy[0]
    nearly_straight = abs(delta[0]) < 0.08 * max(abs(delta[1]), 1e-9) and abs(delta[1]) > 0.5
    if not force_circle and (nearly_straight or radius > 20.0 or span < 0.12):
        curve, line_fit = fit_line_xy(xy, count=count)
        return curve, {"kind": "line", "radius": None, "rmse": line_fit["rmse"], "arc_deg": 0.0}
    angles = np.linspace(float(theta[0]), float(theta[-1]), count)
    curve = np.column_stack((center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)))
    return curve, {"kind": "circle", "center": center.tolist(), "radius": radius, "rmse": rmse, "arc_deg": float(np.degrees(span))}


def path_distance_xy(xy: np.ndarray) -> float:
    if xy.shape[0] < 2:
        return 0.0
    return float(np.sum(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))))


def net_distance_xy(xy: np.ndarray) -> float:
    if xy.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(xy[-1] - xy[0]))


def heading_change_deg(xy: np.ndarray) -> float:
    if xy.shape[0] < 5:
        return 0.0
    k = max(2, min(xy.shape[0] // 4, 6))
    v1 = xy[k] - xy[0]
    v2 = xy[-1] - xy[-1 - k]
    if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
        return 0.0
    cross = float(v1[0] * v2[1] - v1[1] * v2[0])
    dot = float(np.dot(v1, v2))
    return float(abs(np.degrees(np.arctan2(cross, dot))))


def classify_real_points(points: np.ndarray, source_summary: dict) -> dict:
    if source_summary.get("valid_track") is False:
        return {
            "final_motion_class": "invalid_track",
            "final_fit_kind": "invalid_track",
            "final_reason": source_summary.get("invalid_reason") or "source summary marked invalid",
        }
    if points.shape[0] < MIN_FINAL_POINTS:
        return {
            "final_motion_class": "invalid_track",
            "final_fit_kind": "invalid_track",
            "final_reason": "too few cleaned points for final fit",
        }

    xy = points[:, 1:3]
    path_px = path_distance_xy(xy)
    net_px = net_distance_xy(xy)
    if net_px < MIN_FINAL_NET_PX or path_px < MIN_FINAL_PATH_PX:
        return {
            "final_motion_class": "invalid_track",
            "final_fit_kind": "invalid_track",
            "final_reason": "final track is too short",
            "final_path_px": path_px,
            "final_net_px": net_px,
        }

    line_curve, line_fit = fit_line_xy(xy)
    circle_curve, circle_fit = fitted_curve(xy, force_circle=True)
    radius_px = circle_fit.get("radius")
    circle_rmse = float(circle_fit.get("rmse", 1e9))
    arc_deg = float(circle_fit.get("arc_deg", 0.0))
    line_rmse = float(line_fit.get("rmse", 1e9))
    straightness = net_px / max(path_px, 1e-9)
    lateral_span = float(np.ptp(xy[:, 0]))
    lateral_disp = float(abs(xy[-1, 0] - xy[0, 0]))
    forward_disp = float(abs(xy[-1, 1] - xy[0, 1]))
    heading_deg = heading_change_deg(xy)
    rmse_over_r = None if not radius_px or radius_px <= 1e-9 else circle_rmse / radius_px

    forward_dominant = forward_disp >= STRAIGHT_FORWARD_DOMINANCE_RATIO * max(lateral_disp, 1.0)
    body_wiggle_not_turn = lateral_span >= MIN_TURN_LATERAL_SPAN_PX and lateral_disp <= 0.55 * lateral_span
    straight_override = (
        forward_dominant
        and lateral_disp <= MAX_STRAIGHT_LATERAL_DISP_PX
        and heading_deg <= MAX_STRAIGHT_HEADING_DEG
    )

    evidence = []
    if arc_deg >= MIN_TURN_ARC_DEG:
        evidence.append("arc")
    if heading_deg >= MIN_TURN_HEADING_DEG:
        evidence.append("heading")
    if lateral_disp >= MIN_TURN_LATERAL_DISP_PX:
        evidence.append("lateral_disp")
    if lateral_span >= MIN_TURN_LATERAL_SPAN_PX and lateral_disp >= 45.0:
        evidence.append("lateral_span_with_drift")
    if rmse_over_r is not None and rmse_over_r <= MAX_TURN_RMSE_OVER_RADIUS:
        evidence.append("circle_fit")
    if straightness <= MAX_TURN_STRAIGHTNESS and (heading_deg >= 18.0 or lateral_disp >= 45.0):
        evidence.append("not_straight")

    strong_lateral_turn = lateral_disp >= 95.0
    has_directional_turn = heading_deg >= MIN_TURN_HEADING_DEG or lateral_disp >= MIN_TURN_LATERAL_DISP_PX or arc_deg >= 55.0
    is_turn = (
        not straight_override
        and net_px >= MIN_TURN_NET_PX
        and radius_px is not None
        and has_directional_turn
        and (strong_lateral_turn or len(evidence) >= MIN_TURN_SCORE)
    )

    common = {
        "final_path_px": path_px,
        "final_net_px": net_px,
        "final_forward_displacement_px": forward_disp,
        "final_straightness": straightness,
        "final_line_rmse_px": line_rmse,
        "final_circle_rmse_px": circle_rmse,
        "final_circle_rmse_over_radius": rmse_over_r,
        "final_arc_deg": arc_deg,
        "final_heading_change_deg": heading_deg,
        "final_lateral_span_px": lateral_span,
        "final_lateral_displacement_px": lateral_disp,
        "final_forward_dominant": bool(forward_dominant),
        "final_body_wiggle_not_turn": bool(body_wiggle_not_turn),
        "final_straight_override": bool(straight_override),
        "final_turn_evidence": evidence,
        "final_turn_evidence_score": len(evidence),
    }

    if is_turn:
        return {
            **common,
            "final_motion_class": "turn",
            "final_fit_kind": "circle",
            "final_reason": f"turn evidence={','.join(evidence)}",
            "curve": circle_curve,
            "fit": circle_fit,
        }
    reason = "straight override: forward displacement dominates body wiggle" if straight_override else f"not enough turn evidence={','.join(evidence)}"
    return {
        **common,
        "final_motion_class": "straight",
        "final_fit_kind": "line",
        "final_reason": reason,
        "curve": line_curve,
        "fit": line_fit,
    }


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
        plt.close(fig)
        return rows
    ax.set_title("MuJoCo fitted curves, rotated to camera view")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "sim_fitted_curves_rotated.png")
    plt.close(fig)
    (out_dir / "sim_fitted_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def points_from_summary(summary_path: Path) -> np.ndarray:
    data = safe_read_json(summary_path)
    points = data.get("cleaned_points", [])
    arr = np.asarray(points, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def point_distance_metrics(points: np.ndarray) -> dict:
    if points.shape[0] < 2:
        return {"point_count": int(points.shape[0]), "duration_s": 0.0, "straight_distance_px": 0.0, "path_distance_px": 0.0, "net_dx_px": 0.0, "net_dy_px": 0.0}
    t = points[:, 0]
    x = points[:, 1]
    y = points[:, 2]
    duration = float(t[-1] - t[0])
    net_dx = float(x[-1] - x[0])
    net_dy = float(y[-1] - y[0])
    return {
        "point_count": int(points.shape[0]),
        "duration_s": duration,
        "straight_distance_px": float(np.hypot(net_dx, net_dy)),
        "path_distance_px": float(np.sum(np.hypot(np.diff(x), np.diff(y)))),
        "net_dx_px": net_dx,
        "net_dy_px": net_dy,
    }


def real_speed_from_summary(summary_data: dict, distance_metrics: dict) -> float | None:
    for key in ("speed_m_s", "forward_speed_m_s", "mean_speed_m_s"):
        if summary_data.get(key) is not None:
            return float(summary_data[key])
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
    left = 12
    top = max(10, int(0.03 * h))
    right = min(w - 10, left + text_w + pad * 2)
    bottom = min(h - 10, top + text_h_total + pad * 2)
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


def draw_invalid_frame(video_path: Path, out_dir: Path, label: str, summary_path: Path, summary_data: dict, reason: str, point_count: int):
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    draw_metric_box(frame, ["Invalid track", f"points = {point_count}", reason[:36]])
    stem = video_path.stem
    out_img = out_dir / f"video_{stem}_invalid_track.png"
    out_json = out_dir / f"video_{stem}_final_summary.json"
    summary = {
        "label": label,
        "video": str(video_path),
        "source_summary": str(summary_path),
        "final_motion_class": "invalid_track",
        "final_fit_kind": "invalid_track",
        "final_reason": reason,
        "point_count": point_count,
        "source_fit_kind": summary_data.get("fit_kind"),
        "source_valid_track": summary_data.get("valid_track"),
    }
    cv2.imwrite(str(out_img), frame)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_img, summary


def draw_real_result(video_path: Path, summary_path: Path, out_dir: Path, label: str):
    summary_data = safe_read_json(summary_path)
    points = points_from_summary(summary_path)
    decision = classify_real_points(points, summary_data)
    if decision["final_fit_kind"] == "invalid_track":
        return draw_invalid_frame(video_path, out_dir, label, summary_path, summary_data, decision.get("final_reason", "invalid track"), int(points.shape[0]))
    if points.shape[0] < 2:
        return draw_invalid_frame(video_path, out_dir, label, summary_path, summary_data, "not enough points", int(points.shape[0]))

    xy = points[:, 1:3]
    distance_metrics = point_distance_metrics(points)
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read {video_path}")

    stem = video_path.stem
    poly = np.round(xy).astype(np.int32)
    curve = decision["curve"]
    fit = decision["fit"]
    fit_kind = decision["final_fit_kind"]

    if fit_kind == "circle":
        curve_i = np.round(curve).astype(np.int32)
        cv2.polylines(frame, [curve_i], False, (0, 0, 255), 6, cv2.LINE_AA)
        cv2.polylines(frame, [poly], False, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, tuple(poly[0]), 9, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(frame, tuple(poly[-1]), 9, (0, 0, 255), -1, cv2.LINE_AA)
        radius_m = None if fit.get("radius") is None else float(fit["radius"]) / PX_PER_M
        speed_m_s = real_speed_from_summary(summary_data, distance_metrics)
        radius_text = "nan" if radius_m is None else f"{radius_m:.3f} m"
        speed_text = "nan" if speed_m_s is None else f"{speed_m_s:.3f} m/s"
        draw_metric_box(frame, [f"R = {radius_text}", f"v = {speed_text}", f"arc = {fit.get('arc_deg', 0.0):.1f} deg", f"RMSE = {fit.get('rmse', 0.0):.1f}px"])
        out_img = out_dir / f"video_{stem}_fit_curve.png"
    else:
        curve_i = np.round(curve).astype(np.int32)
        cv2.polylines(frame, [curve_i], False, (255, 0, 0), 4, cv2.LINE_AA)
        cv2.polylines(frame, [poly], False, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, tuple(poly[0]), 9, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(frame, tuple(poly[-1]), 9, (0, 0, 255), -1, cv2.LINE_AA)
        speed_m_s = real_speed_from_summary(summary_data, distance_metrics)
        forward_m = summary_data.get("forward_distance_m")
        if forward_m is None:
            forward_m = distance_metrics["straight_distance_px"] / PX_PER_M
        speed_text = "nan" if speed_m_s is None else f"{speed_m_s:.3f} m/s"
        draw_metric_box(frame, ["R = ∞ m", f"v = {speed_text}", f"forward = {float(forward_m):.3f} m", f"RMSE = {fit.get('rmse', 0.0):.1f}px"])
        out_img = out_dir / f"video_{stem}_straight_distance.png"

    summary = {
        "label": label,
        "video": str(video_path),
        "source_summary": str(summary_path),
        "source_fit_kind": summary_data.get("fit_kind"),
        "source_valid_track": summary_data.get("valid_track"),
        **distance_metrics,
        **{k: v for k, v in decision.items() if k not in {"curve", "fit"}},
    }
    if fit_kind == "circle":
        summary.update({
            "radius_px": fit.get("radius"),
            "radius_m": None if fit.get("radius") is None else float(fit["radius"]) / PX_PER_M,
            "rmse_px": fit.get("rmse"),
            "rmse_m": None if fit.get("rmse") is None else float(fit["rmse"]) / PX_PER_M,
            "arc_deg": fit.get("arc_deg"),
            "speed_m_s": real_speed_from_summary(summary_data, distance_metrics),
        })
    else:
        summary.update({
            "radius_m": None,
            "forward_distance_m": summary_data.get("forward_distance_m", distance_metrics["straight_distance_px"] / PX_PER_M),
            "forward_speed_m_s": real_speed_from_summary(summary_data, distance_metrics),
            "rmse_px": fit.get("rmse"),
        })

    out_json = out_dir / f"video_{stem}_final_summary.json"
    cv2.imwrite(str(out_img), frame)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_img, summary


def dynamic_real_configs(video_analysis_dir: Path, recordings_dir: Path):
    summaries = sorted(video_analysis_dir.glob("*/tracked_center_summary_cleaned_physical.json"))
    configs = []
    for summary_path in summaries:
        data = safe_read_json(summary_path)
        stem = data.get("video_stem") or summary_path.parent.name
        video_value = data.get("video")
        video_path = Path(video_value) if video_value else recordings_dir / f"{stem}.mp4"
        if not video_path.exists():
            fallback = recordings_dir / f"{stem}.mp4"
            video_path = fallback if fallback.exists() else video_path
        configs.append({"stem": stem, "label": stem, "summary_path": summary_path, "video_path": video_path})
    return configs


def import_real_videos(video_analysis_dir: Path, recordings_dir: Path, out_dir: Path):
    rows = []
    configs = dynamic_real_configs(video_analysis_dir, recordings_dir)
    if not configs:
        for stem in ("clean_v_20260608_141203", "clean_v_20260608_141254", "clean_v_20260607_233739"):
            configs.append({
                "stem": stem,
                "label": stem,
                "summary_path": video_analysis_dir / stem / "tracked_center_summary_cleaned_physical.json",
                "video_path": recordings_dir / f"{stem}.mp4",
            })
    for config in configs:
        label = config["label"]
        summary_path = config["summary_path"]
        video_path = config["video_path"]
        if not summary_path.exists() or not video_path.exists():
            print(f"skip real {label}: missing summary or video")
            print(f"  summary={summary_path}")
            print(f"  video={video_path}")
            continue
        out_img, summary = draw_real_result(video_path, summary_path, out_dir, label)
        rows.append(summary)
        print(out_img)
        cls = summary.get("final_motion_class")
        if cls == "turn":
            radius = "nan" if summary.get("radius_m") is None else f"{summary['radius_m']:.3f}m"
            speed = "nan" if summary.get("speed_m_s") is None else f"{summary['speed_m_s']:.3f}m/s"
            print(f"{label}: FINAL turn radius={radius} speed={speed} evidence={summary.get('final_turn_evidence')}")
        elif cls == "straight":
            speed = "nan" if summary.get("forward_speed_m_s") is None else f"{summary['forward_speed_m_s']:.3f}m/s"
            print(f"{label}: FINAL straight speed={speed} forward={summary.get('forward_distance_m', 0.0):.3f}m reason={summary.get('final_reason')}")
        else:
            print(f"{label}: FINAL invalid reason={summary.get('final_reason')}")
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
            print(f"{row['name']}: line R=∞m forward_speed={row['mean_forward_speed_m_s']:.3f}m/s")
        else:
            print(f"{row['name']}: {row['kind']} {radius} arc={row['arc_deg']:.1f}deg speed={row['mean_speed_m_s']:.3f}m/s")
    if args.sim_only:
        return
    print("\nReal video summary")
    import_real_videos(video_analysis_dir, recordings_dir, out_dir)


if __name__ == "__main__":
    main()
