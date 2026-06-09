from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


DEFAULT_VIDEO_STEMS = (
    "clean_v_20260608_141203",
    "clean_v_20260608_141254",
    "clean_v_20260607_234118",
)

DEFAULT_RECORDINGS_DIR = Path("../Release/python_backend/recordings")
DEFAULT_OUT_ROOT = Path("outputs/video_analysis")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Track real robot eel videos and export tracked_center_summary_cleaned_physical.json for each video."
    )
    parser.add_argument(
        "--recordings-dir",
        type=Path,
        default=DEFAULT_RECORDINGS_DIR,
        help="Default: ../Release/python_backend/recordings",
    )
    parser.add_argument(
        "--videos",
        nargs="*",
        default=[f"{stem}.mp4" for stem in DEFAULT_VIDEO_STEMS],
        help="Video filenames or paths. Default: the two turning videos plus the straight video.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help="Default: outputs/video_analysis",
    )
    parser.add_argument(
        "--output-name",
        default="tracked_center_summary_cleaned_physical.json",
    )
    parser.add_argument("--min-area", type=float, default=20.0)
    parser.add_argument("--max-area", type=float, default=20000.0)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--preview", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def resolve_video(video: str, recordings_dir: Path) -> Path:
    path = Path(video)
    if path.is_absolute():
        return path
    if path.parent != Path("."):
        return resolve_path(path)
    return (recordings_dir / path).resolve()


def red_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 70, 50], dtype=np.uint8)
    upper1 = np.array([12, 255, 255], dtype=np.uint8)
    lower2 = np.array([168, 70, 50], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)

    mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def find_marker(frame: np.ndarray, min_area: float, max_area: float):
    mask = red_mask(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0.0

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area or area <= best_area:
            continue

        m = cv2.moments(contour)
        if abs(m["m00"]) < 1e-9:
            continue

        cx = float(m["m10"] / m["m00"])
        cy = float(m["m01"] / m["m00"])
        x, y, w, h = cv2.boundingRect(contour)

        best = {
            "x": cx,
            "y": cy,
            "area": area,
            "bbox": [int(x), int(y), int(w), int(h)],
        }
        best_area = area

    return best


def median_smooth(points: list[list[float]], window: int) -> list[list[float]]:
    if len(points) < 3 or window <= 1:
        return points

    if window % 2 == 0:
        window += 1

    half = window // 2
    arr = np.asarray(points, dtype=np.float64)
    out: list[list[float]] = []

    for i in range(len(arr)):
        lo = max(0, i - half)
        hi = min(len(arr), i + half + 1)
        x_med = float(np.median(arr[lo:hi, 1]))
        y_med = float(np.median(arr[lo:hi, 2]))
        out.append([float(arr[i, 0]), x_med, y_med, float(arr[i, 3])])

    return out


def path_distance(points: list[list[float]]) -> float:
    if len(points) < 2:
        return 0.0
    xy = np.asarray([[p[1], p[2]] for p in points], dtype=np.float64)
    return float(np.sum(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))))


def straight_distance(points: list[list[float]]) -> float:
    if len(points) < 2:
        return 0.0
    x0, y0 = points[0][1], points[0][2]
    x1, y1 = points[-1][1], points[-1][2]
    return float(np.hypot(x1 - x0, y1 - y0))


def fit_circle(points: list[list[float]]) -> dict:
    if len(points) < 3:
        return {
            "radius_px": None,
            "arc_deg": 0.0,
            "rmse_px": None,
        }

    xy = np.asarray([[p[1], p[2]] for p in points], dtype=np.float64)
    x = xy[:, 0]
    y = xy[:, 1]

    a = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
    b = x * x + y * y
    cx, cy, k = np.linalg.lstsq(a, b, rcond=None)[0]
    radius = float(np.sqrt(max(0.0, k + cx * cx + cy * cy)))
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    rmse = float(np.sqrt(np.mean((dist - radius) ** 2)))
    theta = np.unwrap(np.arctan2(y - cy, x - cx))
    arc_deg = float(abs(np.degrees(theta[-1] - theta[0])))

    return {
        "center_px": [float(cx), float(cy)],
        "radius_px": radius,
        "arc_deg": arc_deg,
        "rmse_px": rmse,
    }


def track_one_video(
    video_path: Path,
    out_root: Path,
    output_name: str,
    min_area: float,
    max_area: float,
    frame_step: int,
    smooth_window: int,
    preview: bool,
):
    if not video_path.exists():
        print(f"skip missing video: {video_path}")
        return None

    out_dir = out_root / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"skip unreadable video: {video_path}")
        return None

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    raw_points: list[list[float]] = []
    detections = []

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % max(1, frame_step) != 0:
            frame_index += 1
            continue

        marker = find_marker(frame, min_area, max_area)
        t = frame_index / fps

        if marker is not None:
            x = float(marker["x"])
            y = float(marker["y"])
            area = float(marker["area"])

            raw_points.append([float(t), x, y, area])
            detections.append(
                {
                    "frame": int(frame_index),
                    "time_s": float(t),
                    "x": x,
                    "y": y,
                    "area": area,
                    "bbox": marker["bbox"],
                }
            )

            if preview:
                cv2.circle(frame, (int(round(x)), int(round(y))), 8, (0, 255, 255), -1)
                cv2.imshow("tracking", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        frame_index += 1

    cap.release()
    if preview:
        cv2.destroyAllWindows()

    cleaned_points = median_smooth(raw_points, smooth_window)
    circle = fit_circle(cleaned_points)

    result = {
        "video": str(video_path),
        "video_name": video_path.name,
        "video_stem": video_path.stem,
        "fps": fps,
        "frame_count": frame_count,
        "points": raw_points,
        "cleaned_points": cleaned_points,
        "detections": detections,
        "straight_distance_px": straight_distance(cleaned_points),
        "path_distance_px": path_distance(cleaned_points),
        "point_count": len(cleaned_points),
        **circle,
    }

    out_path = out_dir / output_name
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    radius_text = "nan" if result["radius_px"] is None else f"{result['radius_px']:.3f}"
    rmse_text = "nan" if result["rmse_px"] is None else f"{result['rmse_px']:.3f}"
    print(out_path)
    print(
        f"points={len(cleaned_points)} "
        f"straight_distance_px={result['straight_distance_px']:.3f} "
        f"path_distance_px={result['path_distance_px']:.3f} "
        f"radius_px={radius_text} "
        f"arc_deg={result['arc_deg']:.3f} "
        f"rmse_px={rmse_text}"
    )
    return out_path


def main():
    args = parse_args()

    recordings_dir = resolve_path(args.recordings_dir)
    out_root = resolve_path(args.out_root)

    generated = []
    for video in args.videos:
        video_path = resolve_video(video, recordings_dir)
        out_path = track_one_video(
            video_path=video_path,
            out_root=out_root,
            output_name=args.output_name,
            min_area=args.min_area,
            max_area=args.max_area,
            frame_step=args.frame_step,
            smooth_window=args.smooth_window,
            preview=args.preview,
        )
        if out_path is not None:
            generated.append(out_path)

    print(f"\ngenerated {len(generated)} tracking json file(s)")


if __name__ == "__main__":
    main()
