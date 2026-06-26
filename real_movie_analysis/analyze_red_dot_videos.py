from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoMeta:
    direction: str
    command_type: str
    command_value: float | None
    command_unit: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track red markers in real_movie videos using only the color half."
    )
    parser.add_argument("--video-dir", type=Path, default=Path("real_movie"))
    parser.add_argument("--out-dir", type=Path, default=Path("real_movie_analysis"))
    parser.add_argument("--videos", nargs="*", default=None, help="Optional video filenames/stems to process.")
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument(
        "--px-per-m",
        type=float,
        default=269.2105609870623,
        help="Pixel-to-meter scale from the lit tank calibration frame.",
    )
    parser.add_argument(
        "--straight-seconds",
        type=float,
        default=12.0,
        help="End time for steady straight-swim speed fitting.",
    )
    parser.add_argument(
        "--straight-start-seconds",
        type=float,
        default=4.0,
        help="Start time for steady straight-swim speed fitting after launch/startup.",
    )
    parser.add_argument(
        "--turn-trim-last-seconds",
        type=float,
        default=3.0,
        help="Trim this many seconds from the end when fitting turn metrics.",
    )
    parser.add_argument(
        "--turn-start-seconds",
        type=float,
        default=2.0,
        help="Start time for fitting turn metrics after launch/startup.",
    )
    return parser.parse_args()


def parse_video_meta(stem: str) -> VideoMeta:
    parts = stem.split("_")
    if stem.startswith("string") or stem.startswith("straight"):
        return VideoMeta("straight", "straight", None, None)
    direction = parts[0] if parts else "unknown"
    token = parts[1] if len(parts) > 1 else ""
    if len(token) >= 2 and token[0] == "r":
        return VideoMeta(direction, "turn_radius", int(token[1:]) / 10.0, "m")
    if len(token) >= 2 and token[0] == "y":
        return VideoMeta(direction, "yaw_rate", int(token[1:]) / 10.0, "rad/s")
    return VideoMeta(direction, "unknown", None, None)


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def color_half(frame: np.ndarray) -> np.ndarray:
    # The recordings are 1696 x 480: left 848 px is color, right 848 px is depth.
    return frame[:, : frame.shape[1] // 2].copy()


def red_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    blue, green, red = cv2.split(frame)

    hsv_red = ((hue <= 12) | (hue >= 166)) & (sat >= 100) & (val >= 55)
    red_dominant = (
        (red.astype(np.int16) > green.astype(np.int16) + 35)
        & (red.astype(np.int16) > blue.astype(np.int16) + 35)
        & (red >= 65)
    )
    mask = (hsv_red & red_dominant).astype(np.uint8) * 255

    # Keep small marker speckles, but remove isolated sensor noise.
    kernel2 = np.ones((2, 2), np.uint8)
    kernel3 = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel3)
    return mask


def detect_red_center(frame: np.ndarray) -> tuple[float, float, int, int] | None:
    mask = red_mask(frame)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    components = []
    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 2 or area > 1600:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if w > 150 or h > 150:
            continue
        aspect = max(w / max(h, 1), h / max(w, 1))
        if aspect > 18:
            continue
        cx, cy = centroids[i]
        components.append((area, float(cx), float(cy), i))

    if not components:
        return None

    total_area = sum(item[0] for item in components)
    x = sum(area * cx for area, cx, _, _ in components) / total_area
    y = sum(area * cy for area, _, cy, _ in components) / total_area
    return float(x), float(y), int(total_area), int(len(components))


def smooth_points(rows: list[dict]) -> None:
    valid_indices = [i for i, row in enumerate(rows) if row["detected"]]
    if len(valid_indices) < 3:
        return
    xy = np.array([[rows[i]["x_px"], rows[i]["y_px"]] for i in valid_indices], dtype=float)
    smoothed = xy.copy()
    for j in range(len(valid_indices)):
        lo = max(0, j - 2)
        hi = min(len(valid_indices), j + 3)
        smoothed[j] = np.median(xy[lo:hi], axis=0)
    for j, i in enumerate(valid_indices):
        rows[i]["x_smooth_px"] = float(smoothed[j, 0])
        rows[i]["y_smooth_px"] = float(smoothed[j, 1])


def draw_label(frame: np.ndarray, text: str, org: tuple[int, int]) -> None:
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def draw_overlay(frame: np.ndarray, rows: list[dict], frame_index: int, title: str) -> np.ndarray:
    overlay = frame.copy()
    pts = []
    for row in rows:
        if not row["detected"] or row["frame_index"] > frame_index:
            continue
        pts.append([int(round(row["x_smooth_px"])), int(round(row["y_smooth_px"]))])
    if len(pts) >= 2:
        cv2.polylines(overlay, [np.array(pts, dtype=np.int32)], False, (0, 255, 255), 2, cv2.LINE_AA)
    if pts:
        cv2.circle(overlay, tuple(pts[-1]), 7, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(overlay, tuple(pts[-1]), 9, (0, 0, 0), 1, cv2.LINE_AA)
    draw_label(overlay, title, (12, 24))
    draw_label(overlay, f"frame {frame_index}", (12, 48))
    return overlay


def draw_fit_overlay(frame: np.ndarray, rows: list[dict], frame_index: int, summary: dict) -> np.ndarray:
    overlay = frame.copy()
    pts = []
    for row in rows:
        if not row["detected"] or row["frame_index"] > frame_index:
            continue
        pts.append([int(round(row["x_smooth_px"])), int(round(row["y_smooth_px"]))])

    if len(pts) >= 2:
        cv2.polylines(overlay, [np.array(pts, dtype=np.int32)], False, (0, 255, 255), 2, cv2.LINE_AA)

    if summary.get("fit_kind") == "circle" and summary.get("radius_px") not in ("", None):
        center = (int(round(summary["circle_center_x_px"])), int(round(summary["circle_center_y_px"])))
        radius = int(round(summary["radius_px"]))
        cv2.circle(overlay, center, radius, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(overlay, center, 4, (0, 255, 0), -1, cv2.LINE_AA)
        draw_label(
            overlay,
            f"fit circle R={summary['measured_radius_m']:.3f} m, yaw={summary['measured_yaw_rate_abs_rad_s']:.3f} rad/s",
            (12, 24),
        )
    elif summary.get("fit_kind") == "straight" and len(pts) >= 2:
        cv2.line(overlay, tuple(pts[0]), tuple(pts[-1]), (0, 255, 0), 2, cv2.LINE_AA)
        draw_label(overlay, f"straight speed={summary['straight_speed_m_s']:.3f} m/s", (12, 24))
    else:
        draw_label(overlay, "fit unavailable", (12, 24))

    if pts:
        cv2.circle(overlay, tuple(pts[-1]), 6, (0, 255, 255), -1, cv2.LINE_AA)
    draw_label(overlay, f"{summary['video_stem']} frame {frame_index}", (12, 48))
    return overlay


def choose_representative_indices(rows: list[dict], sample_count: int) -> list[int]:
    valid = [row["frame_index"] for row in rows if row["detected"]]
    if not valid:
        return []
    positions = np.linspace(0, len(valid) - 1, sample_count)
    chosen = []
    for pos in positions:
        frame_index = valid[int(round(pos))]
        if frame_index not in chosen:
            chosen.append(frame_index)
    return chosen


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "frame_index",
        "time_s",
        "detected",
        "x_px",
        "y_px",
        "x_smooth_px",
        "y_smooth_px",
        "red_area_px",
        "component_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def fit_circle_xy(xy: np.ndarray) -> dict:
    x = xy[:, 0]
    y = xy[:, 1]
    a = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
    b = x * x + y * y
    cx, cy, k = np.linalg.lstsq(a, b, rcond=None)[0]
    radius = float(np.sqrt(max(0.0, k + cx * cx + cy * cy)))
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    rmse = float(np.sqrt(np.mean((dist - radius) ** 2)))
    return {
        "circle_center_x_px": float(cx),
        "circle_center_y_px": float(cy),
        "radius_px": radius,
        "circle_rmse_px": rmse,
    }


def fit_turn_metrics(valid_rows: list[dict], px_per_m: float, direction: str) -> dict:
    if len(valid_rows) < 3:
        return {
            "fit_kind": "invalid",
            "measured_radius_m": "",
            "measured_yaw_rate_rad_s": "",
            "measured_yaw_rate_abs_rad_s": "",
            "circle_rmse_m": "",
            "arc_angle_rad": "",
        }

    t = np.array([row["time_s"] for row in valid_rows], dtype=float)
    xy = np.array([[row["x_smooth_px"], row["y_smooth_px"]] for row in valid_rows], dtype=float)
    circle = fit_circle_xy(xy)
    cx = circle["circle_center_x_px"]
    cy = circle["circle_center_y_px"]
    theta = np.unwrap(np.arctan2(xy[:, 1] - cy, xy[:, 0] - cx))

    if len(t) >= 5:
        yaw_rate = float(np.polyfit(t - t[0], theta, 1)[0])
    else:
        yaw_rate = float((theta[-1] - theta[0]) / max(t[-1] - t[0], 1e-9))

    # Image coordinates have y increasing downward. The signed value is useful for
    # consistency checks, but paper comparisons usually use the commanded magnitude.
    direction_expected_sign = {"left": -1.0, "right": 1.0}.get(direction, 0.0)
    signed_matches_direction = (
        "" if direction_expected_sign == 0.0 else bool(np.sign(yaw_rate) == np.sign(direction_expected_sign))
    )

    return {
        "fit_kind": "circle",
        **circle,
        "measured_radius_m": circle["radius_px"] / px_per_m,
        "circle_rmse_m": circle["circle_rmse_px"] / px_per_m,
        "measured_yaw_rate_rad_s": yaw_rate,
        "measured_yaw_rate_abs_rad_s": abs(yaw_rate),
        "arc_angle_rad": float(abs(theta[-1] - theta[0])),
        "arc_angle_deg": float(abs(np.degrees(theta[-1] - theta[0]))),
        "signed_yaw_direction_matches_filename": signed_matches_direction,
    }


def fit_straight_metrics(
    valid_rows: list[dict],
    px_per_m: float,
    straight_start_seconds: float,
    straight_end_seconds: float,
) -> dict:
    measurement_rows = [
        row for row in valid_rows if straight_start_seconds <= row["time_s"] <= straight_end_seconds
    ]
    if len(measurement_rows) < 2:
        return {"fit_kind": "invalid", "straight_speed_m_s": ""}
    t = np.array([row["time_s"] for row in measurement_rows], dtype=float)
    xy = np.array([[row["x_smooth_px"], row["y_smooth_px"]] for row in measurement_rows], dtype=float)
    displacement_px = float(np.linalg.norm(xy[-1] - xy[0]))
    path_px = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
    duration = max(float(t[-1] - t[0]), 1e-9)
    tracked_net_speed = (displacement_px / px_per_m) / duration
    tracked_path_speed = (path_px / px_per_m) / duration
    x_slope_px_s = float(np.polyfit(t - t[0], xy[:, 0], 1)[0])
    y_slope_px_s = float(np.polyfit(t - t[0], xy[:, 1], 1)[0])
    steady_speed = float(np.hypot(x_slope_px_s, y_slope_px_s) / px_per_m)
    return {
        "fit_kind": "straight",
        "straight_measurement_start_s": float(t[0]),
        "straight_measurement_requested_start_s": float(straight_start_seconds),
        "straight_measurement_seconds": float(straight_end_seconds),
        "straight_measurement_end_s": float(t[-1]),
        "straight_measurement_frame_count": len(measurement_rows),
        "straight_displacement_px": displacement_px,
        "straight_displacement_m": displacement_px / px_per_m,
        "straight_tracked_net_speed_m_s": tracked_net_speed,
        "straight_tracked_path_length_px": path_px,
        "straight_tracked_path_length_m": path_px / px_per_m,
        "straight_tracked_path_speed_m_s": tracked_path_speed,
        "straight_fit_x_slope_px_s": x_slope_px_s,
        "straight_fit_y_slope_px_s": y_slope_px_s,
        "straight_speed_m_s": steady_speed,
        "straight_speed_source": "linear_fit_of_red_marker_center_in_steady_window",
    }


def compare_command(summary: dict) -> dict:
    command_type = summary.get("command_type")
    command_value = summary.get("command_value")
    if command_value is None:
        return {
            "commanded_metric": "",
            "commanded_value": "",
            "measured_value": "",
            "error": "",
            "abs_error": "",
            "percent_error": "",
        }
    if command_type == "turn_radius":
        measured = summary.get("measured_radius_m")
        metric = "R_m"
    elif command_type == "yaw_rate":
        measured = summary.get("measured_yaw_rate_abs_rad_s")
        metric = "yaw_rate_abs_rad_s"
    else:
        measured = None
        metric = ""
    if measured in (None, ""):
        return {
            "commanded_metric": metric,
            "commanded_value": command_value,
            "measured_value": "",
            "error": "",
            "abs_error": "",
            "percent_error": "",
        }
    error = float(measured) - float(command_value)
    return {
        "commanded_metric": metric,
        "commanded_value": float(command_value),
        "measured_value": float(measured),
        "error": error,
        "abs_error": abs(error),
        "percent_error": 100.0 * error / float(command_value) if command_value else "",
    }


def plot_trajectory(path: Path, rows: list[dict], title: str, width: int, height: int) -> None:
    import matplotlib.pyplot as plt

    valid = [row for row in rows if row["detected"]]
    if not valid:
        return
    x = [row["x_smooth_px"] for row in valid]
    y = [row["y_smooth_px"] for row in valid]
    fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=180)
    scatter = ax.scatter(x, y, c=[row["time_s"] for row in valid], s=9, cmap="viridis")
    ax.plot(x, y, color="tab:red", linewidth=0.8, alpha=0.75)
    ax.set_title(title)
    ax.set_xlabel("x in color frame (px)")
    ax.set_ylabel("y in color frame (px)")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(scatter, ax=ax, label="time (s)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def make_contact_sheet(paths: list[Path], out_path: Path, label_height: int = 28) -> None:
    images = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        image = cv2.resize(image, (318, 180))
        canvas = np.zeros((180 + label_height, 318, 3), dtype=np.uint8)
        canvas[:180] = image
        video_label = path.parents[1].name if len(path.parents) > 1 else path.parent.name
        draw_label(canvas, video_label + "/" + path.stem, (6, 199))
        images.append(canvas)
    if not images:
        return
    cols = 5
    rows = int(np.ceil(len(images) / cols))
    blank = np.zeros_like(images[0])
    grid = []
    for r in range(rows):
        row_images = images[r * cols : (r + 1) * cols]
        row_images += [blank.copy() for _ in range(cols - len(row_images))]
        grid.append(np.hstack(row_images))
    cv2.imwrite(str(out_path), np.vstack(grid))


def make_single_video_posture_sheet(paths: list[Path], out_path: Path) -> None:
    images = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        image = cv2.resize(image, (424, 240))
        canvas = np.zeros((270, 424, 3), dtype=np.uint8)
        canvas[:240] = image
        draw_label(canvas, path.stem, (8, 262))
        images.append(canvas)
    if images:
        cv2.imwrite(str(out_path), np.hstack(images))


def write_fit_overlay_video(video_path: Path, out_path: Path, rows: list[dict], summary: dict) -> None:
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    color_width = source_width // 2
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (color_width, source_height),
    )
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        overlay = draw_fit_overlay(color_half(frame), rows, frame_index, summary)
        writer.write(overlay)
        frame_index += 1
    cap.release()
    writer.release()


def analyze_video(
    video_path: Path,
    out_root: Path,
    sample_count: int,
    px_per_m: float,
    straight_start_seconds: float,
    straight_seconds: float,
    turn_trim_last_seconds: float,
    turn_start_seconds: float,
) -> dict:
    out_dir = out_root / video_path.stem
    clean_dir = out_dir / "representative_clean"
    annotated_dir = out_dir / "representative_annotated"
    clean_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    rows: list[dict] = []
    frame_index = 0
    color_width = source_width // 2
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        color = color_half(frame)
        detected = detect_red_center(color)
        time_s = frame_index / fps if fps > 0 else 0.0
        row = {
            "frame_index": frame_index,
            "time_s": time_s,
            "detected": detected is not None,
            "x_px": "",
            "y_px": "",
            "x_smooth_px": "",
            "y_smooth_px": "",
            "red_area_px": 0,
            "component_count": 0,
        }
        if detected is not None:
            x, y, area, components = detected
            row.update(
                {
                    "x_px": x,
                    "y_px": y,
                    "x_smooth_px": x,
                    "y_smooth_px": y,
                    "red_area_px": area,
                    "component_count": components,
                }
            )
        rows.append(row)
        frame_index += 1
    cap.release()
    smooth_points(rows)

    csv_path = out_dir / "red_dot_tracking.csv"
    write_csv(csv_path, rows)

    meta = parse_video_meta(video_path.stem)
    valid_rows = [row for row in rows if row["detected"]]
    title = video_path.stem
    plot_path = out_dir / "red_dot_trajectory.png"
    plot_trajectory(plot_path, rows, title, color_width, source_height)

    chosen = choose_representative_indices(rows, sample_count)
    cap = cv2.VideoCapture(str(video_path))
    clean_paths = []
    annotated_paths = []
    for order, idx in enumerate(chosen, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        color = color_half(frame)
        clean_path = clean_dir / f"{order:02d}_frame_{idx:05d}.png"
        annotated_path = annotated_dir / f"{order:02d}_frame_{idx:05d}.png"
        cv2.imwrite(str(clean_path), color)
        cv2.imwrite(str(annotated_path), draw_overlay(color, rows, idx, title))
        clean_paths.append(clean_path)
        annotated_paths.append(annotated_path)
    cap.release()

    duration_s = frame_count / fps if fps > 0 else 0.0
    detection_rate = len(valid_rows) / len(rows) if rows else 0.0
    if len(valid_rows) >= 2:
        xy = np.array([[row["x_smooth_px"], row["y_smooth_px"]] for row in valid_rows], dtype=float)
        path_px = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
        net_px = float(np.linalg.norm(xy[-1] - xy[0]))
    else:
        path_px = 0.0
        net_px = 0.0

    metric_rows = valid_rows
    metric_time_window = {"metric_start_s": 0.0, "metric_end_s": duration_s}
    if meta.command_type in {"turn_radius", "yaw_rate"}:
        metric_end_s = max(0.0, duration_s - turn_trim_last_seconds)
        metric_rows = [row for row in valid_rows if turn_start_seconds <= row["time_s"] <= metric_end_s]
        metric_time_window = {"metric_start_s": float(turn_start_seconds), "metric_end_s": metric_end_s}

    if meta.command_type in {"turn_radius", "yaw_rate"}:
        motion_metrics = fit_turn_metrics(metric_rows, px_per_m, meta.direction)
        motion_metrics.update(metric_time_window)
        motion_metrics["turn_trim_last_seconds"] = float(turn_trim_last_seconds)
        motion_metrics["turn_start_seconds"] = float(turn_start_seconds)
    else:
        motion_metrics = fit_straight_metrics(
            valid_rows,
            px_per_m,
            straight_start_seconds,
            straight_seconds,
        )

    summary = {
        "video": str(video_path),
        "video_name": video_path.name,
        "video_stem": video_path.stem,
        "direction": meta.direction,
        "command_type": meta.command_type,
        "command_value": meta.command_value,
        "command_unit": meta.command_unit,
        "source_width_px": source_width,
        "source_height_px": source_height,
        "color_frame_width_px": color_width,
        "color_frame_height_px": source_height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_s": duration_s,
        "px_per_m": px_per_m,
        "detected_frame_count": len(valid_rows),
        "detection_rate": detection_rate,
        "path_length_px": path_px,
        "path_length_m": path_px / px_per_m,
        "net_displacement_px": net_px,
        "net_displacement_m": net_px / px_per_m,
        **motion_metrics,
        "tracking_csv": str(csv_path),
        "trajectory_plot": str(plot_path),
        "representative_clean": [str(path) for path in clean_paths],
        "representative_annotated": [str(path) for path in annotated_paths],
    }
    summary.update(compare_command(summary))

    posture_sheet_path = out_dir / "posture_5_contact_sheet.jpg"
    make_single_video_posture_sheet(clean_paths, posture_sheet_path)
    fit_overlay_video_path = out_dir / "fit_overlay_video.mp4"
    write_fit_overlay_video(video_path, fit_overlay_video_path, rows, summary)
    summary["posture_5_contact_sheet"] = str(posture_sheet_path)
    summary["fit_overlay_video"] = str(fit_overlay_video_path)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    video_dir = resolve(args.video_dir)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(video_dir.glob("*.mp4"))
    if args.videos:
        requested = {Path(item).stem for item in args.videos}
        videos = [path for path in videos if path.stem in requested or path.name in args.videos]
    summaries = [
        analyze_video(
            path,
            out_dir,
            args.sample_count,
            args.px_per_m,
            args.straight_start_seconds,
            args.straight_seconds,
            args.turn_trim_last_seconds,
            args.turn_start_seconds,
        )
        for path in videos
    ]

    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "video_name",
            "direction",
            "command_type",
            "command_value",
            "command_unit",
            "fps",
            "frame_count",
            "duration_s",
            "px_per_m",
            "detected_frame_count",
            "detection_rate",
            "fit_kind",
            "measured_radius_m",
            "measured_yaw_rate_rad_s",
            "measured_yaw_rate_abs_rad_s",
            "commanded_metric",
            "commanded_value",
            "measured_value",
            "error",
            "abs_error",
            "percent_error",
            "circle_rmse_m",
            "arc_angle_deg",
            "path_length_px",
            "path_length_m",
            "net_displacement_px",
            "net_displacement_m",
            "straight_speed_m_s",
            "straight_speed_source",
            "straight_tracked_net_speed_m_s",
            "straight_tracked_path_speed_m_s",
            "straight_fit_x_slope_px_s",
            "straight_fit_y_slope_px_s",
            "straight_measurement_start_s",
            "straight_measurement_requested_start_s",
            "straight_measurement_seconds",
            "straight_measurement_end_s",
            "straight_measurement_frame_count",
            "metric_start_s",
            "metric_end_s",
            "turn_trim_last_seconds",
            "turn_start_seconds",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: summary.get(key, "") for key in fieldnames})

    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    with (out_dir / "command_error_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "video_name",
            "direction",
            "commanded_metric",
            "commanded_value",
            "measured_value",
            "error",
            "abs_error",
            "percent_error",
            "measured_radius_m",
            "measured_yaw_rate_abs_rad_s",
            "circle_rmse_m",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            if summary.get("commanded_metric"):
                writer.writerow({key: summary.get(key, "") for key in fieldnames})

    clean_paths = []
    annotated_paths = []
    for summary in summaries:
        clean_paths.extend(Path(path) for path in summary["representative_clean"])
        annotated_paths.extend(Path(path) for path in summary["representative_annotated"])
    make_contact_sheet(clean_paths, out_dir / "representative_clean_contact_sheet.jpg")
    make_contact_sheet(annotated_paths, out_dir / "representative_annotated_contact_sheet.jpg")

    for summary in summaries:
        print(
            f"{summary['video_name']}: fps={summary['fps']:.2f}, "
            f"duration={summary['duration_s']:.2f}s, "
            f"detected={summary['detected_frame_count']}/{summary['frame_count']} "
            f"({summary['detection_rate']:.1%}), "
            f"metric={summary.get('commanded_metric') or summary.get('fit_kind')}"
        )
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
