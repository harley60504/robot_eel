from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

DEFAULT_VIDEO_STEMS = (
    "clean_v_20260607_233739",
    "clean_v_20260608_141203",
    "clean_v_20260608_141254",
)
DEFAULT_RECORDINGS_DIR = Path("../Release/python_backend/recordings")
DEFAULT_OUT_ROOT = Path("outputs/video_analysis")


def parse_args():
    p = argparse.ArgumentParser(description="Track real robot eel center points with merged composite+legacy detection.")
    p.add_argument("--recordings-dir", type=Path, default=DEFAULT_RECORDINGS_DIR)
    p.add_argument("--videos", nargs="*", default=[f"{s}.mp4" for s in DEFAULT_VIDEO_STEMS])
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--output-name", default="tracked_center_summary_cleaned_physical.json")
    p.add_argument("--detector", choices=("merged", "composite", "red"), default="merged")
    p.add_argument("--min-area", type=float, default=20.0)
    p.add_argument("--max-area", type=float, default=20000.0)
    p.add_argument("--frame-step", type=int, default=1)
    p.add_argument("--smooth-window", type=int, default=5)
    p.add_argument("--max-jump-px", type=float, default=260.0)
    p.add_argument("--preview", action="store_true")
    return p.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def resolve_video(video: str, recordings_dir: Path) -> Path:
    path = Path(video)
    if path.is_absolute():
        return path
    if path.parent != Path("."):
        return resolve_path(path)
    return (recordings_dir / path).resolve()


def crop_tracking_roi(frame):
    h, w = frame.shape[:2]
    x0, y0 = int(0.05 * w), int(0.16 * h)
    x1, y1 = int(0.93 * w), int(0.94 * h)
    return frame[y0:y1, x0:x1], x0, y0


def red_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, np.array([0, 58, 45]), np.array([16, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([164, 48, 45]), np.array([180, 255, 255]))
    orange = cv2.inRange(hsv, np.array([16, 45, 55]), np.array([42, 255, 255]))
    mask = cv2.bitwise_or(cv2.bitwise_or(red1, red2), orange)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def body_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    white = (hsv[:, :, 2] > 135) & (hsv[:, :, 1] < 120)
    dark = (hsv[:, :, 2] < 95) | (gray < 82)
    red = red_mask(frame) > 0
    mask = (white | dark | red).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    return mask


def components(mask, xoff, yoff, method, min_area=80, max_area=65000):
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    rh, rw = mask.shape[:2]
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        x, y = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        bw, bh = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        if bw < 5 or bh < 5 or bw > 0.82 * rw or bh > 0.82 * rh:
            continue
        cx, cy = cents[i]
        out.append({"x": float(cx + xoff), "y": float(cy + yoff), "area": area, "bbox": [x + xoff, y + yoff, bw, bh], "method": method})
    return out


def red_candidates(frame, min_area, max_area):
    roi, xoff, yoff = crop_tracking_roi(frame)
    cnts, _ = cv2.findContours(red_mask(roi), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        area = float(cv2.contourArea(c))
        if area < min_area or area > max_area:
            continue
        m = cv2.moments(c)
        if abs(m["m00"]) < 1e-9:
            continue
        x, y, w, h = cv2.boundingRect(c)
        out.append({"x": float(m["m10"] / m["m00"] + xoff), "y": float(m["m01"] / m["m00"] + yoff), "area": area, "bbox": [x + xoff, y + yoff, w, h], "method": "red"})
    return out


def composite_candidates(frame):
    roi, xoff, yoff = crop_tracking_roi(frame)
    return components(body_mask(roi), xoff, yoff, "composite")


def choose_candidate(cands, prev, shape, max_jump):
    h, w = shape[:2]
    scored = []
    for c in cands:
        x, y = c["x"], c["y"]
        if y < 0.10 * h or x < 0.02 * w or x > 0.96 * w:
            continue
        score = c["area"] * (1.30 if c["method"] == "composite" else 1.0)
        if prev is None:
            score -= 0.9 * abs(x - 0.50 * w) + 0.3 * abs(y - 0.58 * h)
        else:
            d = float(np.hypot(x - prev[0], y - prev[1]))
            score -= 4.0 * d
            if d > max_jump:
                score -= 2500 + 8.0 * (d - max_jump)
        scored.append((score, c))
    if not scored:
        return None
    scored.sort(reverse=True, key=lambda t: t[0])
    best = dict(scored[0][1])
    best["quality"] = float(scored[0][0])
    return best


def find_marker(frame, detector, min_area, max_area, prev, max_jump):
    cands = []
    if detector in ("merged", "composite"):
        cands.extend(composite_candidates(frame))
    if detector in ("merged", "red"):
        cands.extend(red_candidates(frame, min_area, max_area))
    return choose_candidate(cands, prev, frame.shape, max_jump)


def median_smooth(points, window):
    if len(points) < 3 or window <= 1:
        return points
    if window % 2 == 0:
        window += 1
    half = window // 2
    arr = np.asarray(points, dtype=np.float64)
    out = []
    for i in range(len(arr)):
        lo, hi = max(0, i - half), min(len(arr), i + half + 1)
        out.append([float(arr[i, 0]), float(np.median(arr[lo:hi, 1])), float(np.median(arr[lo:hi, 2])), float(arr[i, 3])])
    return out


def path_distance(points):
    if len(points) < 2:
        return 0.0
    xy = np.asarray([[p[1], p[2]] for p in points], dtype=np.float64)
    return float(np.sum(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))))


def straight_distance(points):
    if len(points) < 2:
        return 0.0
    return float(np.hypot(points[-1][1] - points[0][1], points[-1][2] - points[0][2]))


def fit_circle(points):
    if len(points) < 3:
        return {"radius_px": None, "arc_deg": 0.0, "rmse_px": None}
    xy = np.asarray([[p[1], p[2]] for p in points], dtype=np.float64)
    x, y = xy[:, 0], xy[:, 1]
    a = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
    b = x * x + y * y
    cx, cy, k = np.linalg.lstsq(a, b, rcond=None)[0]
    r = float(np.sqrt(max(0.0, k + cx * cx + cy * cy)))
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    theta = np.unwrap(np.arctan2(y - cy, x - cx))
    return {"center_px": [float(cx), float(cy)], "radius_px": r, "arc_deg": float(abs(np.degrees(theta[-1] - theta[0]))), "rmse_px": float(np.sqrt(np.mean((dist - r) ** 2)))}


def track_one_video(video_path, out_root, args):
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
    points, detections, methods = [], [], {}
    prev = None
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % max(1, args.frame_step) != 0:
            frame_index += 1
            continue
        marker = find_marker(frame, args.detector, args.min_area, args.max_area, prev, args.max_jump_px)
        if marker is not None:
            x, y, area = float(marker["x"]), float(marker["y"]), float(marker["area"])
            method = marker["method"]
            prev = (x, y)
            methods[method] = methods.get(method, 0) + 1
            t = frame_index / fps
            points.append([float(t), x, y, area])
            detections.append({"frame": frame_index, "time_s": float(t), "x": x, "y": y, "area": area, "bbox": marker["bbox"], "method": method, "quality": marker.get("quality", 0.0)})
            if args.preview:
                color = (255, 0, 255) if method == "composite" else (0, 255, 255)
                cv2.circle(frame, (int(round(x)), int(round(y))), 8, color, -1)
                cv2.imshow("tracking", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        frame_index += 1
    cap.release()
    if args.preview:
        cv2.destroyAllWindows()
    cleaned = median_smooth(points, args.smooth_window)
    result = {
        "tracker_version": "merged_composite_red_v2",
        "detector": args.detector,
        "video": str(video_path),
        "video_name": video_path.name,
        "video_stem": video_path.stem,
        "fps": fps,
        "frame_count": frame_count,
        "points": points,
        "cleaned_points": cleaned,
        "detections": detections,
        "method_counts": methods,
        "straight_distance_px": straight_distance(cleaned),
        "path_distance_px": path_distance(cleaned),
        "point_count": len(cleaned),
        **fit_circle(cleaned),
    }
    out_path = out_dir / args.output_name
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    radius = "nan" if result["radius_px"] is None else f"{result['radius_px']:.3f}"
    print(out_path)
    print(f"points={len(cleaned)} methods={methods} radius_px={radius} arc_deg={result['arc_deg']:.3f} rmse_px={result['rmse_px']}")
    return out_path


def main():
    args = parse_args()
    recordings_dir = resolve_path(args.recordings_dir)
    out_root = resolve_path(args.out_root)
    generated = []
    for video in args.videos:
        out_path = track_one_video(resolve_video(video, recordings_dir), out_root, args)
        if out_path is not None:
            generated.append(out_path)
    print(f"\ngenerated {len(generated)} tracking json file(s)")


if __name__ == "__main__":
    main()
