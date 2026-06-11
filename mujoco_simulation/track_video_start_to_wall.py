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

# Keep detections inside the real water region. This is not a post-fit corner classifier;
# it prevents wall labels, tank rim, timestamp, and D-Link overlays from becoming candidates.
FRAME_X_MIN = 125
FRAME_X_MAX_PAD = 90
FRAME_Y_TOP_PAD = 70
FRAME_Y_BOTTOM_PAD = 135
ROI_EDGE_PAD = 18
MAX_STEP_PX = 180.0
SOFT_MAX_STEP_PX = 95.0


def parse_args():
    parser = argparse.ArgumentParser(description="Track video center points from start to wall contact.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/video_start_to_wall"))
    return parser.parse_args()


def valid_frame_bounds(frame_shape, clip: ClipConfig) -> tuple[float, float, float, float]:
    h, w = frame_shape[:2]
    x0, y0, rw, rh = clip.roi
    left = max(float(FRAME_X_MIN), float(x0 + ROI_EDGE_PAD))
    right = min(float(w - FRAME_X_MAX_PAD), float(x0 + rw - ROI_EDGE_PAD))
    top = max(float(clip.min_y), float(y0 + FRAME_Y_TOP_PAD))
    bottom = min(float(h - FRAME_Y_BOTTOM_PAD), float(y0 + rh - ROI_EDGE_PAD))
    return left, top, right, bottom


def inside_bounds(gx: float, gy: float, bounds: tuple[float, float, float, float]) -> bool:
    left, top, right, bottom = bounds
    return left <= gx <= right and top <= gy <= bottom


def component_quality(area: int, width: int, height: int) -> float | None:
    if area < 70 or area > 6500:
        return None
    if width < 5 or height < 5:
        return None
    aspect = max(width / max(height, 1), height / max(width, 1))
    if aspect > 10.0:
        return None
    fill_ratio = area / max(width * height, 1)
    if fill_ratio < 0.10 or fill_ratio > 0.92:
        return None
    return float(area) * (1.0 + 0.25 * min(aspect, 4.0))


def continuity_score(gx: float, gy: float, area: int, quality: float, prev) -> float:
    if prev is None:
        # Start from the lower-middle water region, not the side overlays.
        return quality - 0.35 * abs(gx - 560.0) - 0.18 * abs(gy - 1320.0)
    dist = float(np.hypot(gx - prev[0], gy - prev[1]))
    if dist > MAX_STEP_PX:
        return -1e9
    jump_penalty = 3.0 * max(0.0, dist - SOFT_MAX_STEP_PX)
    return quality + 0.018 * area - 2.15 * dist - jump_penalty


def build_turn_mask(crop: np.ndarray, roi_top: int) -> np.ndarray:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    blue, green, red = cv2.split(crop)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # White/gray eel body plus dark marker/holes, but reject overexposed glare and blue tank.
    nonblue = (blue.astype(np.int16) - red.astype(np.int16) < 42) & (blue.astype(np.int16) - green.astype(np.int16) < 42)
    body = (saturation < 88) & (value > 45) & (value < 248) & nonblue
    dark_marker = (value < 92) & nonblue
    mask = ((body | dark_marker).astype(np.uint8)) * 255

    # The top strip of each ROI is usually glare/tank edge, not the eel.
    mask[: max(70, min(130, 1040 - roi_top)), :] = 0
    kernel3 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel5)
    return mask


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
        bounds = valid_frame_bounds(frame.shape, clip)
        x0, y0, w, h = roi
        crop = frame[y0 : y0 + h, x0 : x0 + w]
        mask = build_turn_mask(crop, y0)

        count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        candidates = []
        for i in range(1, count):
            area = int(stats[i, cv2.CC_STAT_AREA])
            bx = int(stats[i, cv2.CC_STAT_LEFT])
            by = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            quality = component_quality(area, bw, bh)
            if quality is None:
                continue
            if bx <= ROI_EDGE_PAD or by <= ROI_EDGE_PAD or bx + bw >= w - ROI_EDGE_PAD or by + bh >= h - ROI_EDGE_PAD:
                continue
            cx, cy = centroids[i]
            gx, gy = float(x0 + cx), float(y0 + cy)
            if not inside_bounds(gx, gy, bounds):
                continue
            score = continuity_score(gx, gy, area, quality, prev)
            if score <= -1e8:
                continue
            candidates.append((float(score), gx, gy, area))
        if candidates:
            candidates.sort(reverse=True)
            _, gx, gy, area = candidates[0]
            prev = (gx, gy)
            raw.append((float(frame_idx / fps), gx, gy, area))

    cap.release()
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
        bounds = valid_frame_bounds(frame.shape, clip)
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
            bx = int(stats[i, cv2.CC_STAT_LEFT])
            by = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            quality = component_quality(area, bw, bh)
            if quality is None:
                continue
            cx, cy = centroids[i]
            if not inside_bounds(float(cx), float(cy), bounds):
                continue
            if previous is None:
                score = quality - 0.30 * abs(cx - 525) - 0.18 * abs(cy - 350)
            else:
                dist = float(np.hypot(cx - previous[0], cy - previous[1]))
                if dist > MAX_STEP_PX:
                    continue
                score = quality - 2.0 * dist + 0.01 * area
            candidates.append((float(score), float(cx), float(cy), area))
        if candidates:
            candidates.sort(reverse=True)
            _, cx, cy, area = candidates[0]
            previous = (cx, cy)
            points.append((float(frame_idx / fps), cx, cy, area))
    cap.release()
    return clean_points(points, clip.min_y)


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
        if np.linalg.norm(current - previous) > 180 and np.linalg.norm(current - next_point) > 180:
            keep[i] = False
    cleaned = [point for point, should_keep in zip(points, keep) if should_keep]
    if len(cleaned) >= 5:
        xy2 = np.array([[point[1], point[2]] for point in cleaned], dtype=float)
        smoothed = []
        for i, point in enumerate(cleaned):
            lo = max(0, i - 1)
            hi = min(len(cleaned), i + 2)
            mx, my = np.median(xy2[lo:hi], axis=0)
            smoothed.append((point[0], float(mx), float(my), point[3]))
        return smoothed
    return cleaned


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
