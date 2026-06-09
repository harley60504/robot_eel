from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


DEFAULT_VIDEO = Path("Release/python_backend/recordings/clean_v_20260608_141254.mp4")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Regenerate tracked_center_summary_cleaned_physical.json from a real swim video."
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--sample-hz", type=float, default=5.0)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--min-y", type=float, default=1000.0)
    parser.add_argument("--max-jump", type=float, default=280.0)
    parser.add_argument("--roi", default="80,620,940,1240", help="x,y,w,h crop for tracking.")
    return parser.parse_args()


def resolve_video(path: Path) -> Path:
    candidates = [
        path,
        Path.cwd() / path,
        Path.cwd().parent / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(path)


def parse_roi(value: str) -> tuple[int, int, int, int]:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("--roi must be x,y,w,h")
    return tuple(parts)  # type: ignore[return-value]


def detect_points(
    video_path: Path,
    seconds: float,
    sample_hz: float,
    roi: tuple[int, int, int, int],
    min_y: float,
) -> list[tuple[float, float, float, int]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    sample_step = max(1, int(round(fps / sample_hz)))
    x0, y0, w, h = roi
    previous = None
    points: list[tuple[float, float, float, int]] = []

    for frame_idx in range(0, int(seconds * fps) + 1, sample_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break

        crop = frame[y0 : y0 + h, x0 : x0 + w]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        low_sat = (hsv[:, :, 1] < 75) & (hsv[:, :, 2] > 35)
        dark = hsv[:, :, 2] < 90
        mask = ((low_sat | dark).astype(np.uint8)) * 255
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
            if 80 <= area <= 12000 and 120 < gx < 960 and min_y < gy < 1840:
                score = area if previous is None else -np.hypot(gx - previous[0], gy - previous[1]) + 0.015 * area
                candidates.append((float(score), gx, gy, area))
        if not candidates:
            continue
        candidates.sort(reverse=True)
        _, gx, gy, area = candidates[0]
        previous = (gx, gy)
        points.append((float(frame_idx / fps), gx, gy, area))

    cap.release()
    return points


def clean_physical_points(
    points: list[tuple[float, float, float, int]],
    min_y: float,
    max_jump: float,
) -> list[tuple[float, float, float, int]]:
    points = [point for point in points if point[2] > min_y]
    if len(points) < 4:
        return points
    xy = np.array([[point[1], point[2]] for point in points], dtype=np.float64)
    keep = [True] * len(points)
    for i in range(1, len(points) - 1):
        previous = xy[i - 1]
        current = xy[i]
        next_point = xy[i + 1]
        if np.linalg.norm(current - previous) > max_jump and np.linalg.norm(current - next_point) > max_jump:
            keep[i] = False
    return [point for point, should_keep in zip(points, keep) if should_keep]


def fit_circle(points: list[tuple[float, float, float, int]]) -> dict:
    xy = np.array([[point[1], point[2]] for point in points], dtype=np.float64)
    x = xy[:, 0]
    y = xy[:, 1]
    a = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
    b = x * x + y * y
    cx, cy, k = np.linalg.lstsq(a, b, rcond=None)[0]
    radius = float(np.sqrt(max(0.0, k + cx * cx + cy * cy)))
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    theta = np.unwrap(np.arctan2(y - cy, x - cx))
    return {
        "n": len(points),
        "center_px": [float(cx), float(cy)],
        "radius_px": radius,
        "rmse_px": float(np.sqrt(np.mean((dist - radius) ** 2))),
        "arc_deg": float(abs(np.degrees(theta[-1] - theta[0]))),
        "start_px": [float(x[0]), float(y[0])],
        "end_px": [float(x[-1]), float(y[-1])],
    }


def draw_overlay(video_path: Path, seconds: float, points, fit: dict, out_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(seconds * fps)))
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    if not ok:
        return

    cx, cy = fit["center_px"]
    radius = fit["radius_px"]
    xy = np.array([[point[1], point[2]] for point in points], dtype=np.float64)
    theta = np.unwrap(np.arctan2(xy[:, 1] - cy, xy[:, 0] - cx))
    angles = np.linspace(float(theta[0]), float(theta[-1]), 280)
    curve = np.column_stack((cx + radius * np.cos(angles), cy + radius * np.sin(angles)))
    cv2.polylines(frame, [np.round(curve).astype(np.int32)], False, (0, 0, 255), 8, cv2.LINE_AA)
    for time_s, x, y, _ in points:
        if abs(time_s - round(time_s)) < 0.11:
            cv2.circle(frame, (int(x), int(y)), 9, (0, 255, 255), -1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), frame)
    cap.release()


def main():
    args = parse_args()
    video_path = resolve_video(args.video)
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = Path("outputs/video_analysis") / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_points = detect_points(video_path, args.seconds, args.sample_hz, parse_roi(args.roi), args.min_y)
    cleaned_points = clean_physical_points(raw_points, args.min_y, args.max_jump)
    fit = fit_circle(cleaned_points)

    output = {
        "cleaned_points": [list(point) for point in cleaned_points],
        "circle_fit": fit,
    }
    json_path = out_dir / "tracked_center_summary_cleaned_physical.json"
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    draw_overlay(video_path, args.seconds, cleaned_points, fit, out_dir / "tracked_center_overlay_cleaned_physical.png")

    print(json_path)
    print(f"points={len(cleaned_points)} radius_px={fit['radius_px']:.3f} arc_deg={fit['arc_deg']:.3f} rmse_px={fit['rmse_px']:.3f}")


if __name__ == "__main__":
    main()
