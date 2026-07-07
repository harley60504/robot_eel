from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import cv2
import numpy as np


DEFAULT_PX_PER_M = 269.2105609870623
W, H = 1800, 1050
ML, MR, MT, MB = 150, 90, 100, 125
PW, PH = W - ML - MR, H - MT - MB

# BGR colors for OpenCV. These match the adaptive figures generated on 2026-06-25.
TXT = (35, 40, 45)
AX = (70, 76, 82)
TRAJ = (214, 170, 175)
FIT_RED = (50, 70, 210)
BODY = (172, 150, 92)
BODY_DARK = (125, 104, 62)
BODY_LIGHT = (202, 186, 130)
LED = (35, 58, 230)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw adaptive teal/red real LED posture figures.")
    parser.add_argument("analysis_root", type=Path, help="Folder containing per-video summary.json/red_dot_tracking.csv folders.")
    parser.add_argument("--video-root", type=Path, default=None, help="Folder containing source MP4 files. Defaults to real_movie.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output folder. Defaults beside analysis_root.")
    parser.add_argument("--label", default=None, help="Contact-sheet filename prefix.")
    parser.add_argument("--title-prefix", default="", help="Text prefix before each video name, e.g. old/.")
    parser.add_argument("--px-per-m", type=float, default=DEFAULT_PX_PER_M)
    return parser.parse_args()


def lab(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.72, color=TXT, thickness: int = 2) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def red_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    blue, green, red = cv2.split(frame)
    hsv_red = ((hue <= 12) | (hue >= 166)) & (sat >= 90) & (val >= 50)
    red_dominant = (
        (red.astype(np.int16) > green.astype(np.int16) + 30)
        & (red.astype(np.int16) > blue.astype(np.int16) + 30)
        & (red >= 75)
    )
    return cv2.medianBlur((hsv_red | red_dominant).astype(np.uint8) * 255, 3)


def led_points(frame: np.ndarray) -> list[tuple[float, float, int]]:
    count, _, stats, centroids = cv2.connectedComponentsWithStats(red_mask(frame), 8)
    components = []
    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if 3 <= area <= 700:
            components.append((area, float(centroids[i][0]), float(centroids[i][1])))
    return [(x, y, area) for area, x, y in sorted(components, key=lambda item: item[0], reverse=True)[:6]]


def order_points(points: list[tuple[float, float, int]], velocity: tuple[float, float]) -> list[tuple[float, float]]:
    pts = [(point[0], point[1]) for point in points]
    if len(pts) <= 2:
        return pts

    best = None
    best_length = float("inf")
    for perm in itertools.permutations(range(len(pts))):
        total = sum(math.hypot(pts[a][0] - pts[b][0], pts[a][1] - pts[b][1]) for a, b in zip(perm, perm[1:]))
        if total < best_length:
            best_length = total
            best = perm

    ordered = [pts[i] for i in best]
    vx, vy = velocity
    if (ordered[-1][0] - ordered[0][0]) * vx + (ordered[-1][1] - ordered[0][1]) * vy < 0:
        ordered = list(reversed(ordered))
    return ordered


def load_rows(video_dir: Path) -> list[dict]:
    rows = []
    with (video_dir / "red_dot_tracking.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("detected", "")).lower() != "true":
                continue
            rows.append(
                {
                    "frame": int(float(row["frame_index"])),
                    "time": float(row["time_s"]),
                    "x": float(row.get("x_smooth_px") or row["x_px"]),
                    "y": float(row.get("y_smooth_px") or row["y_px"]),
                }
            )
    return rows


def metric_rows(all_rows: list[dict], summary: dict) -> list[dict]:
    if summary.get("fit_kind") == "circle":
        start = float(summary.get("metric_start_s") or summary.get("turn_start_s") or 0)
        end = float(summary.get("metric_end_s") or all_rows[-1]["time"])
    else:
        start = float(summary.get("straight_measurement_start_s") or 0)
        end = float(summary.get("straight_measurement_end_s") or all_rows[-1]["time"])
    kept = [row for row in all_rows if start <= row["time"] <= end]
    return kept or all_rows


def local_velocity(rows: list[dict], row: dict) -> tuple[float, float]:
    index = min(range(len(rows)), key=lambda i: abs(rows[i]["frame"] - row["frame"]))
    left = max(0, index - 7)
    right = min(len(rows) - 1, index + 7)
    return rows[right]["x"] - rows[left]["x"], rows[right]["y"] - rows[left]["y"]


def pose_count(summary: dict) -> int:
    if summary.get("fit_kind") != "circle":
        return 5
    radius = float(summary.get("measured_radius_m") or 0)
    if radius < 0.32:
        return 3
    if radius < 0.45:
        return 4
    return 5


def pick_circle(rows: list[dict], summary: dict, count: int) -> list[dict]:
    cx = float(summary["circle_center_x_px"])
    cy = float(summary["circle_center_y_px"])
    angles = np.array([math.atan2(-(row["y"] - cy), row["x"] - cx) for row in rows])
    target_degrees = {
        3: [215, 35, -80],
        4: [210, 120, 35, -70],
        5: [220, 140, 55, -70, -20],
    }[count]

    picked = []
    used_indices = []
    for degrees in target_degrees:
        target = math.radians(degrees)
        diff = np.abs(np.angle(np.exp(1j * (angles - target))))
        sorted_indices = np.argsort(diff)
        pick = int(sorted_indices[0])
        for index in sorted_indices:
            index = int(index)
            if all(abs(rows[index]["time"] - rows[used]["time"]) > 1.5 for used in used_indices):
                pick = index
                break
        used_indices.append(pick)
        picked.append(rows[pick])
    return picked


def pick_straight(rows: list[dict], count: int = 5) -> list[dict]:
    return [rows[int(round(i))] for i in np.linspace(0, len(rows) - 1, count)]


def smooth(points, window: int = 11) -> np.ndarray:
    arr = np.array(points, np.float32)
    if len(arr) < 3:
        return arr
    out = []
    half = window // 2
    for i in range(len(arr)):
        out.append(np.mean(arr[max(0, i - half) : min(len(arr), i + half + 1)], axis=0))
    return np.array(out, np.float32)


def dashed(img: np.ndarray, points, color=TRAJ, thickness: int = 3, dash: int = 16, gap: int = 12) -> None:
    pts = np.array(points, float)
    for a, b in zip(pts, pts[1:]):
        vector = b - a
        length = np.linalg.norm(vector)
        if length < 1:
            continue
        unit = vector / length
        t = 0
        while t < length:
            cv2.line(
                img,
                tuple(np.round(a + unit * t).astype(int)),
                tuple(np.round(a + unit * min(length, t + dash)).astype(int)),
                color,
                thickness,
                cv2.LINE_AA,
            )
            t += dash + gap


def transform(rows: list[dict], summary: dict, pose_sets: list[list[tuple[float, float]]], px_per_m: float):
    xs = [row["x"] for row in rows]
    ys = [row["y"] for row in rows]
    for pts in pose_sets:
        for x, y in pts:
            xs.append(x)
            ys.append(y)
    if summary.get("fit_kind") == "circle":
        cx = float(summary["circle_center_x_px"])
        cy = float(summary["circle_center_y_px"])
        radius = float(summary["radius_px"])
        xs += [cx - radius, cx + radius]
        ys += [cy - radius, cy + radius]

    pad = 0.32 * px_per_m
    minx, maxx = min(xs) - pad, max(xs) + pad
    miny, maxy = min(ys) - pad, max(ys) + pad
    scale = min(PW / (maxx - minx), PH / (maxy - miny))
    ox = ML + (PW - (maxx - minx) * scale) / 2 - minx * scale
    oy = MT + (PH - (maxy - miny) * scale) / 2 - miny * scale
    return (lambda x, y: np.array([x * scale + ox, y * scale + oy], float)), scale, (minx, maxx, miny, maxy)


def capsule(p0, p1, width: float, n: int = 12) -> np.ndarray:
    p0 = np.array(p0, float)
    p1 = np.array(p1, float)
    direction = p1 - p0
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return np.round(np.array([p0])).astype(np.int32)
    direction = direction / length
    normal = np.array([-direction[1], direction[0]])
    radius = width / 2
    pts = [p0 + normal * radius, p1 + normal * radius]
    for angle in np.linspace(math.pi / 2, -math.pi / 2, n):
        pts.append(p1 + normal * radius * math.sin(angle) + direction * radius * math.cos(angle))
    pts += [p1 - normal * radius, p0 - normal * radius]
    for angle in np.linspace(-math.pi / 2, math.pi / 2, n):
        pts.append(p0 + normal * radius * math.sin(angle) - direction * radius * math.cos(angle))
    return np.round(np.array(pts)).astype(np.int32)


def draw_eel(img: np.ndarray, points, index: int) -> None:
    pts = [np.array(point, float) for point in points]
    if len(pts) < 2:
        return
    median_spacing = max(16, float(np.median([np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)])))
    width = max(18, min(34, median_spacing * 0.58))
    head_dir = pts[-1] - pts[-2]
    tail_dir = pts[0] - pts[1]
    if np.linalg.norm(head_dir) > 1e-6:
        head_dir = head_dir / np.linalg.norm(head_dir)
    if np.linalg.norm(tail_dir) > 1e-6:
        tail_dir = tail_dir / np.linalg.norm(tail_dir)

    curve = [pts[0] + tail_dir * median_spacing * 0.65] + pts + [pts[-1] + head_dir * median_spacing * 1.05]
    for i in range(len(curve) - 1):
        cv2.fillConvexPoly(img, capsule(curve[i], curve[i + 1], width), BODY, cv2.LINE_AA)
    cv2.polylines(img, [np.round(np.array(curve)).astype(np.int32)], False, BODY_DARK, max(2, int(width * 0.13)), cv2.LINE_AA)

    head_center = pts[-1] + head_dir * median_spacing * 0.55
    head_radius = max(7, int(width * 0.42))
    cv2.circle(img, tuple(np.round(head_center).astype(int)), head_radius, BODY_LIGHT, -1, cv2.LINE_AA)
    cv2.circle(img, tuple(np.round(head_center).astype(int)), head_radius, BODY_DARK, 1, cv2.LINE_AA)

    tail = pts[0] + tail_dir * median_spacing * 1.05
    cv2.line(img, tuple(np.round(pts[0]).astype(int)), tuple(np.round(tail).astype(int)), BODY_DARK, max(2, int(width * 0.10)), cv2.LINE_AA)

    marker_radius = max(2, int(width * 0.105))
    for point in pts:
        center = tuple(np.round(point).astype(int))
        cv2.circle(img, center, marker_radius, LED, -1, cv2.LINE_AA)
        cv2.circle(img, center, marker_radius + 1, (245, 245, 245), 1, cv2.LINE_AA)
    lab(img, str(index), tuple(np.round(pts[-1] + np.array([10, -8])).astype(int)), 0.56, TXT, 2)


def draw_axes(img: np.ndarray, tr, bounds, px_per_m: float) -> None:
    minx, maxx, miny, maxy = bounds
    origin = tr(minx, maxy)
    x_end = tr(maxx, maxy)
    y_end = tr(minx, miny)
    cv2.arrowedLine(img, tuple(np.round(origin).astype(int)), tuple(np.round(x_end).astype(int)), AX, 2, cv2.LINE_AA, tipLength=0.018)
    cv2.arrowedLine(img, tuple(np.round(origin).astype(int)), tuple(np.round(y_end).astype(int)), AX, 2, cv2.LINE_AA, tipLength=0.018)
    lab(img, "x (m)", tuple(np.round(x_end + np.array([-64, 34])).astype(int)), 0.58, AX)
    lab(img, "y (m)", tuple(np.round(y_end + np.array([-82, 20])).astype(int)), 0.58, AX)
    scale_0 = tr(minx + 0.16 * px_per_m, maxy - 0.11 * px_per_m)
    scale_1 = tr(minx + 0.66 * px_per_m, maxy - 0.11 * px_per_m)
    cv2.line(img, tuple(np.round(scale_0).astype(int)), tuple(np.round(scale_1).astype(int)), (35, 40, 45), 4, cv2.LINE_AA)
    lab(img, "0.5 m", tuple(np.round((scale_0 + scale_1) / 2 + np.array([-34, -14])).astype(int)), 0.5)


def resolve_video_path(summary: dict, video_root: Path | None, video_dir: Path) -> Path:
    if summary.get("video"):
        path = Path(summary["video"])
        if path.exists():
            return path
    name = summary.get("video_name") or f"{video_dir.name}.mp4"
    if video_root is not None:
        return video_root / name
    return video_dir.parents[1] / "real_movie" / name


def render(video_dir: Path, video_root: Path | None, out_root: Path, px_per_m: float, title_prefix: str = "") -> Path | None:
    summary = json.loads((video_dir / "summary.json").read_text(encoding="utf-8"))
    fit = summary.get("fit_kind")
    all_rows = load_rows(video_dir)
    if not all_rows:
        return None
    rows = metric_rows(all_rows, summary)
    count = pose_count(summary)
    selected = pick_circle(rows, summary, count) if fit == "circle" else pick_straight(rows, count)

    video_path = resolve_video_path(summary, video_root, video_dir)
    cap = cv2.VideoCapture(str(video_path))
    pose_sets = []
    for row in selected:
        cap.set(cv2.CAP_PROP_POS_FRAMES, row["frame"])
        ok, frame = cap.read()
        if not ok:
            pose_sets.append([])
            continue
        color = frame[:, : frame.shape[1] // 2] if frame.shape[1] >= 1600 else frame
        pose_sets.append(order_points(led_points(color), local_velocity(rows, row)))
    cap.release()

    if any(len(points) < 2 for points in pose_sets):
        return None

    tr, scale, bounds = transform(rows, summary, pose_sets, px_per_m)
    img = np.full((H, W, 3), 255, np.uint8)
    cv2.rectangle(img, (ML, MT), (W - MR, H - MB), (238, 240, 242), 1, cv2.LINE_AA)
    draw_axes(img, tr, bounds, px_per_m)
    dashed(img, smooth([tr(row["x"], row["y"]) for row in rows]))

    if fit == "circle":
        cx = float(summary["circle_center_x_px"])
        cy = float(summary["circle_center_y_px"])
        radius = float(summary["radius_px"])
        center = tr(cx, cy)
        cv2.circle(img, tuple(np.round(center).astype(int)), int(round(radius * scale)), (220, 225, 250), 8, cv2.LINE_AA)
        cv2.circle(img, tuple(np.round(center).astype(int)), int(round(radius * scale)), FIT_RED, 4, cv2.LINE_AA)
        cv2.circle(img, tuple(np.round(center).astype(int)), 5, FIT_RED, -1, cv2.LINE_AA)
        radius_m = float(summary.get("measured_radius_m") or 0)
        yaw_rate = float(summary.get("measured_yaw_rate_abs_rad_s") or abs(float(summary.get("measured_yaw_rate_rad_s") or 0)))
        metric = f"R = {radius_m:.3f} m    yaw_rate = {yaw_rate:.3f} rad/s"
        title = f"{title_prefix}{video_dir.name}: turning, real LED posture ({count} snapshots)"
    else:
        start = tr(rows[0]["x"], rows[0]["y"])
        end = tr(rows[-1]["x"], rows[-1]["y"])
        cv2.line(img, tuple(np.round(start).astype(int)), tuple(np.round(end).astype(int)), (220, 225, 250), 9, cv2.LINE_AA)
        cv2.line(img, tuple(np.round(start).astype(int)), tuple(np.round(end).astype(int)), FIT_RED, 4, cv2.LINE_AA)
        metric = f"v = {float(summary.get('straight_speed_m_s', 0)):.3f} m/s"
        title = f"{title_prefix}{video_dir.name}: straight swimming, real LED posture"

    lab(img, title, (ML, 44), 0.76, TXT, 2)
    lab(img, metric, (ML + 18, 78), 0.72, FIT_RED, 2)
    for index, (_row, points) in enumerate(zip(selected, pose_sets), 1):
        draw_eel(img, [tr(x, y) for x, y in points], index)
    for index, row in enumerate(selected, 1):
        lab(img, f"{index}: {row['time']:.1f}s", (W - 260, 118 + 34 * index), 0.56, TXT, 2)

    out = out_root / f'{title_prefix.replace("/", "_")}{video_dir.name}_relative_pose_adaptive_teal_red.jpg'
    cv2.imwrite(str(out), img)
    return out


def make_contact_sheet(paths: list[Path], out: Path, columns: int = 2) -> None:
    imgs = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is not None:
            imgs.append(cv2.resize(image, (900, 525)))
    if not imgs:
        return
    blank = np.full_like(imgs[0], 255)
    rows = []
    for i in range(0, len(imgs), columns):
        row = imgs[i : i + columns]
        row += [blank.copy() for _ in range(columns - len(row))]
        rows.append(np.hstack(row))
    cv2.imwrite(str(out), np.vstack(rows))


def batch(label: str, analysis_root: Path, video_root: Path | None, out_root: Path, px_per_m: float, title_prefix: str = "") -> list[Path]:
    out_root.mkdir(parents=True, exist_ok=True)
    outputs = []
    turning = []
    straight = []
    for video_dir in sorted(analysis_root.iterdir()):
        if not (video_dir.is_dir() and (video_dir / "summary.json").exists() and (video_dir / "red_dot_tracking.csv").exists()):
            continue
        if video_dir.name.startswith("calibration"):
            continue
        out = render(video_dir, video_root, out_root, px_per_m, title_prefix)
        if out is None:
            continue
        outputs.append(out)
        summary = json.loads((video_dir / "summary.json").read_text(encoding="utf-8"))
        (straight if summary.get("fit_kind") == "straight" else turning).append(out)

    make_contact_sheet(turning, out_root / f"{label}_turning_adaptive_contact_sheet.jpg", 2)
    make_contact_sheet(straight, out_root / f"{label}_straight_adaptive_contact_sheet.jpg", 1)
    make_contact_sheet(outputs, out_root / f"{label}_all_adaptive_contact_sheet.jpg", 2)
    return outputs


def main() -> None:
    args = parse_args()
    analysis_root = args.analysis_root.resolve()
    video_root = args.video_root.resolve() if args.video_root else None
    out_root = args.out_dir.resolve() if args.out_dir else analysis_root.with_name(f"{analysis_root.name}_relative_pose_adaptive_teal_red")
    label = args.label or analysis_root.name

    outputs = batch(label, analysis_root, video_root, out_root, args.px_per_m, args.title_prefix)
    print(json.dumps({"out_dir": str(out_root), "figures": [str(path) for path in outputs]}, indent=2))


if __name__ == "__main__":
    main()
