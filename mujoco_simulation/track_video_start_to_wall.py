from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class ClipConfig:
    key: str
    video: Path
    wall_seconds: float
    fit_kind: str
    roi: tuple[int, int, int, int] = (80, 620, 940, 1240)
    min_y: float = 1000.0


CLIPS = (
    ClipConfig("turn_left_141203", Path("Release/python_backend/recordings/clean_v_20260608_141203.mp4"), 8.0, "circle"),
    ClipConfig("spin_left_141254", Path("Release/python_backend/recordings/clean_v_20260608_141254.mp4"), 8.0, "circle"),
    ClipConfig(
        "straight_233739",
        Path("Release/python_backend/recordings/clean_v_20260607_233739.mp4"),
        15.0,
        "line",
        roi=(80, 0, 940, 1850),
        min_y=80.0,
    ),
)


def parse_args():
    parser = argparse.ArgumentParser(description="Track video center points from start to wall contact.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/video_start_to_wall"))
    return parser.parse_args()


def detect_points(clip: ClipConfig, root: Path) -> list[tuple[float, float, float, int]]:
    if clip.key == "straight_233739":
        return detect_dark_marker_points(clip, root)

    video_path = root / clip.video
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    roi = clip.roi
    raw = []
    prev = None
    sample_step = max(1, int(round(fps / 5)))

    for frame_idx in range(0, int(clip.wall_seconds * fps) + 1, sample_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break
        x0, y0, w, h = roi
        crop = frame[y0 : y0 + h, x0 : x0 + w]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        low_sat = (hsv[:, :, 1] < 75) & (hsv[:, :, 2] > 35)
        dark = hsv[:, :, 2] < 90
        mask = ((low_sat | dark).astype(np.uint8)) * 255
        if roi[1] < 100:
            mask[:70, :] = 0
        else:
            mask[:100, :] = 0
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        candidates = []
        for i in range(1, count):
            area = int(stats[i, cv2.CC_STAT_AREA])
            cx, cy = centroids[i]
            gx, gy = float(x0 + cx), float(y0 + cy)
            if 80 <= area <= 12000 and 120 < gx < 960 and clip.min_y < gy < 1840:
                score = area if prev is None else -np.hypot(gx - prev[0], gy - prev[1]) + 0.015 * area
                candidates.append((float(score), gx, gy, area))
        if candidates:
            candidates.sort(reverse=True)
            _, gx, gy, area = candidates[0]
            prev = (gx, gy)
            raw.append((float(frame_idx / fps), gx, gy, area))

    return clean_points(raw, clip.min_y)


def detect_dark_marker_points(clip: ClipConfig, root: Path) -> list[tuple[float, float, float, int]]:
    cap = cv2.VideoCapture(str(root / clip.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    points = []
    previous = None
    sample_step = max(1, int(round(fps / 5)))

    for frame_idx in range(0, int(clip.wall_seconds * fps) + 1, sample_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break
        blue, green, red = cv2.split(frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dark = hsv[:, :, 2] < 105
        nonblue = (blue.astype(np.int16) - red.astype(np.int16) < 35) & (blue.astype(np.int16) - green.astype(np.int16) < 35)
        mask = ((dark & nonblue).astype(np.uint8)) * 255
        mask[:, 1000:] = 0
        mask[:40, :] = 0
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        candidates = []
        for i in range(1, count):
            area = int(stats[i, cv2.CC_STAT_AREA])
            cx, cy = centroids[i]
            if 20 <= area <= 8000 and 80 < cx < 950 and 40 < cy < 1850:
                if previous is None:
                    score = -abs(cx - 525) - 0.25 * abs(cy - 350) + 0.02 * area
                else:
                    score = -np.hypot(cx - previous[0], cy - previous[1]) + 0.01 * area
                candidates.append((float(score), float(cx), float(cy), area))
        if candidates:
            candidates.sort(reverse=True)
            _, cx, cy, area = candidates[0]
            previous = (cx, cy)
            points.append((float(frame_idx / fps), cx, cy, area))
    return points


def clean_points(points: list[tuple[float, float, float, int]], min_y: float) -> list[tuple[float, float, float, int]]:
    points = [point for point in points if point[2] > min_y]
    if len(points) < 4:
        return points
    xy = np.array([[point[1], point[2]] for point in points], dtype=float)
    keep = [True] * len(points)
    for i in range(1, len(points) - 1):
        previous = xy[i - 1]
        current = xy[i]
        next_point = xy[i + 1]
        if np.linalg.norm(current - previous) > 280 and np.linalg.norm(current - next_point) > 280:
            keep[i] = False
    return [point for point, should_keep in zip(points, keep) if should_keep]


def fit_circle(xy: np.ndarray):
    x = xy[:, 0]
    y = xy[:, 1]
    a = np.column_stack((2 * x, 2 * y, np.ones_like(x)))
    b = x * x + y * y
    cx, cy, k = np.linalg.lstsq(a, b, rcond=None)[0]
    radius = float(np.sqrt(max(0.0, k + cx * cx + cy * cy)))
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    theta = np.unwrap(np.arctan2(y - cy, x - cx))
    angles = np.linspace(float(theta[0]), float(theta[-1]), 280)
    curve = np.column_stack((cx + radius * np.cos(angles), cy + radius * np.sin(angles)))
    return curve, {
        "kind": "circle",
        "center_px": [float(cx), float(cy)],
        "radius_px": radius,
        "arc_deg": float(abs(np.degrees(theta[-1] - theta[0]))),
        "rmse_px": float(np.sqrt(np.mean((dist - radius) ** 2))),
    }


def fit_line(xy: np.ndarray):
    center = xy.mean(axis=0)
    _, _, vh = np.linalg.svd(xy - center, full_matrices=False)
    direction = vh[0]
    if np.dot(direction, xy[-1] - xy[0]) < 0:
        direction = -direction
    projection = (xy - center) @ direction
    scalar = np.linspace(float(projection.min()), float(projection.max()), 280)
    curve = center + np.outer(scalar, direction)
    residual = np.linalg.norm((xy - center) - np.outer(projection, direction), axis=1)
    return curve, {
        "kind": "line",
        "line_start_px": curve[0].tolist(),
        "line_end_px": curve[-1].tolist(),
        "direction_px": direction.tolist(),
        "length_px": float(np.linalg.norm(curve[-1] - curve[0])),
        "rmse_px": float(np.sqrt(np.mean(residual**2))),
    }


def draw_fit(video_path: Path, wall_seconds: float, points, curve: np.ndarray, out_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(wall_seconds * fps)))
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read {video_path}")

    cv2.polylines(frame, [np.round(curve).astype(np.int32)], False, (0, 0, 255), 9, cv2.LINE_AA)
    seen = set()
    for time_s, x, y, _ in points:
        rounded = int(round(time_s))
        if abs(time_s - rounded) < 0.11 and rounded not in seen:
            seen.add(rounded)
            cv2.circle(frame, (int(x), int(y)), 10, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(frame, f"{rounded}s", (int(x) + 12, int(y) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), frame)


def write_points_csv(path: Path, points):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("time_s", "x_px", "y_px", "area_px"))
        writer.writerows(points)


def process_clip(root: Path, out_root: Path, clip: ClipConfig):
    out_dir = out_root / clip.key
    out_dir.mkdir(parents=True, exist_ok=True)
    points = detect_points(clip, root)
    xy = np.array([[point[1], point[2]] for point in points], dtype=float)
    curve, fit = fit_circle(xy) if clip.fit_kind == "circle" else fit_line(xy)
    summary = {
        "clip": clip.key,
        "video": str(clip.video),
        "wall_seconds": clip.wall_seconds,
        "point_count": len(points),
        "start_point_px": [points[0][1], points[0][2]],
        "end_point_px": [points[-1][1], points[-1][2]],
        **fit,
    }
    write_points_csv(out_dir / "points_start_to_wall.csv", points)
    (out_dir / "fit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    draw_fit(root / clip.video, clip.wall_seconds, points, curve, out_dir / "fit_curve_only_start_to_wall.png")
    return summary


def main():
    args = parse_args()
    root = Path.cwd()
    out_root = root / args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    summaries = [process_clip(root, out_root, clip) for clip in CLIPS]
    (out_root / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    for summary in summaries:
        print(summary["clip"], summary["kind"], "points", summary["point_count"], "wall_s", summary["wall_seconds"], "rmse_px", f"{summary['rmse_px']:.2f}")


if __name__ == "__main__":
    main()
