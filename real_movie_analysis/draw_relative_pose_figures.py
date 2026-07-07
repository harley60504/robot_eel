from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_PX_PER_M = 269.2105609870623
TEAL = "#2C8FA3"
TEAL_DARK = "#17606F"
RED = "#D83A2E"
PATH = "#8F8ACB"
BLACK = "#333333"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw real-video relative LED posture figures.")
    parser.add_argument("analysis_dir", type=Path, help="Folder produced by analyze_red_dot_videos.py.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output folder. Defaults beside analysis_dir.")
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument("--px-per-m", type=float, default=None)
    parser.add_argument("--max-poses", type=int, default=5)
    parser.add_argument("--prefix", default=None)
    return parser.parse_args()


def safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in value.strip())
    return safe.strip("._") or "relative_pose"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_tracking_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out = []
    for row in rows:
        if str(row.get("detected", "")).lower() != "true":
            continue
        try:
            out.append(
                {
                    "frame_index": int(float(row["frame_index"])),
                    "time_s": float(row["time_s"]),
                    "x": float(row.get("x_smooth_px") or row["x_px"]),
                    "y": float(row.get("y_smooth_px") or row["y_px"]),
                }
            )
        except (KeyError, ValueError):
            continue
    return out


def red_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    blue, green, red = cv2.split(frame)
    hsv_red = ((hue <= 12) | (hue >= 166)) & (sat >= 95) & (val >= 45)
    red_dominant = (
        (red.astype(np.int16) > green.astype(np.int16) + 30)
        & (red.astype(np.int16) > blue.astype(np.int16) + 30)
        & (red >= 55)
    )
    mask = (hsv_red & red_dominant).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return mask


def detect_led_points(frame: np.ndarray, expected: int = 7) -> np.ndarray:
    mask = red_mask(frame)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    components = []
    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 2 or area > 220:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if w > 28 or h > 28:
            continue
        cx, cy = centroids[i]
        components.append((area, float(cx), float(cy)))
    if not components:
        return np.empty((0, 2), dtype=float)

    components.sort(reverse=True)
    pts = np.array([[cx, cy] for _, cx, cy in components[:expected]], dtype=float)
    return order_body_points(pts)


def order_body_points(pts: np.ndarray) -> np.ndarray:
    if len(pts) <= 2:
        return pts
    # The real videos are usually side-on and the head LED is the rightmost point.
    # Sorting by x gives a stable head-to-tail order for the paper posture figures.
    return pts[np.argsort(-pts[:, 0])]


def tracking_lookup(rows: list[dict]) -> dict[int, dict]:
    return {row["frame_index"]: row for row in rows}


def frame_index_from_name(path: Path) -> int | None:
    stem = path.stem
    parts = stem.split("_")
    for i, part in enumerate(parts[:-1]):
        if part == "frame" and parts[i + 1].isdigit():
            return int(parts[i + 1])
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    return None


def px_to_m(points: np.ndarray, px_per_m: float) -> np.ndarray:
    out = points.astype(float).copy()
    out[:, 0] = out[:, 0] / px_per_m
    out[:, 1] = -out[:, 1] / px_per_m
    return out


def fit_circle(xy: np.ndarray) -> tuple[np.ndarray, float] | None:
    if len(xy) < 4:
        return None
    x = xy[:, 0]
    y = xy[:, 1]
    a = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
    b = x * x + y * y
    try:
        cx, cy, k = np.linalg.lstsq(a, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    r = math.sqrt(max(0.0, k + cx * cx + cy * cy))
    return np.array([cx, cy], dtype=float), float(r)


def draw_eel(ax, body_xy: np.ndarray, label: str) -> None:
    if len(body_xy) == 0:
        return
    ax.plot(body_xy[:, 0], body_xy[:, 1], color=TEAL_DARK, linewidth=16, alpha=0.92, solid_capstyle="round")
    ax.plot(body_xy[:, 0], body_xy[:, 1], color=TEAL, linewidth=11, alpha=0.98, solid_capstyle="round")
    ax.scatter(body_xy[:, 0], body_xy[:, 1], s=18, color=RED, edgecolor="white", linewidth=0.7, zorder=5)
    ax.scatter(body_xy[0, 0], body_xy[0, 1], s=48, color=TEAL, edgecolor=TEAL_DARK, linewidth=1.4, zorder=6)
    ax.text(body_xy[0, 0], body_xy[0, 1] + 0.045, label, fontsize=10, weight="bold", color=BLACK)


def axis_limits(path_xy: np.ndarray, pose_sets: list[np.ndarray]) -> tuple[tuple[float, float], tuple[float, float]]:
    chunks = [path_xy] + [pts for pts in pose_sets if len(pts)]
    data = np.vstack(chunks)
    xmin, ymin = np.nanmin(data, axis=0)
    xmax, ymax = np.nanmax(data, axis=0)
    pad_x = max(0.25, (xmax - xmin) * 0.18)
    pad_y = max(0.25, (ymax - ymin) * 0.22)
    return (xmin - pad_x, xmax + pad_x), (ymin - pad_y, ymax + pad_y)


def title_metric(summary: dict) -> str:
    if summary.get("fit_kind") == "circle" and summary.get("measured_radius_m") not in ("", None):
        return f"R = {float(summary['measured_radius_m']):.3f} m"
    if summary.get("fit_kind") == "straight" and summary.get("straight_speed_m_s") not in ("", None):
        return f"v = {float(summary['straight_speed_m_s']):.3f} m/s"
    return ""


def draw_video_figure(video_dir: Path, out_dir: Path, sample_count: int, px_per_m_override: float | None, max_poses: int) -> Path | None:
    summary_path = video_dir / "summary.json"
    tracking_path = video_dir / "red_dot_tracking.csv"
    clean_dir = video_dir / "representative_clean"
    if not summary_path.exists() or not tracking_path.exists() or not clean_dir.exists():
        return None

    summary = read_json(summary_path)
    rows = read_tracking_csv(tracking_path)
    if not rows:
        return None
    px_per_m = float(px_per_m_override or summary.get("px_per_m") or DEFAULT_PX_PER_M)
    path_px = np.array([[row["x"], row["y"]] for row in rows], dtype=float)
    path_xy = px_to_m(path_px, px_per_m)

    lookup = tracking_lookup(rows)
    image_paths = sorted(clean_dir.glob("*.png"))[:sample_count]
    pose_sets = []
    pose_times = []
    for image_path in image_paths[:max_poses]:
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        pts_px = detect_led_points(frame)
        frame_index = frame_index_from_name(image_path)
        if frame_index is None or frame_index not in lookup or len(pts_px) < 3:
            continue

        center_px = np.mean(pts_px, axis=0)
        target_px = np.array([lookup[frame_index]["x"], lookup[frame_index]["y"]], dtype=float)
        pts_px = pts_px + (target_px - center_px)
        pose_sets.append(px_to_m(pts_px, px_per_m))
        pose_times.append(float(lookup[frame_index]["time_s"]))

    if not pose_sets:
        return None

    fig, ax = plt.subplots(figsize=(10.6, 6.2), dpi=170)
    ax.plot(path_xy[:, 0], path_xy[:, 1], color=PATH, linewidth=2.0, alpha=0.42, zorder=1)

    if summary.get("fit_kind") == "circle":
        fit = fit_circle(path_xy)
        if fit is not None:
            center, radius = fit
            circle = plt.Circle(center, radius, color=RED, fill=False, linewidth=2.5, alpha=0.95)
            ax.add_patch(circle)
            ax.scatter([center[0]], [center[1]], s=18, color=RED, zorder=3)
    elif summary.get("fit_kind") == "straight":
        start = path_xy[0]
        end = path_xy[-1]
        ax.plot([start[0], end[0]], [start[1], end[1]], color=RED, linewidth=2.6, alpha=0.95, zorder=2)

    for i, pts in enumerate(pose_sets, start=1):
        draw_eel(ax, pts, str(i))

    metric = title_metric(summary)
    video_stem = str(summary.get("video_stem") or video_dir.name)
    ax.set_title(f"{video_stem}: real LED posture", loc="left", fontsize=15, weight="bold", pad=14)
    if metric:
        ax.text(0.01, 0.96, metric, transform=ax.transAxes, color=RED, fontsize=13, weight="bold", va="top")
    ax.text(
        0.90,
        0.93,
        "\n".join(f"{i}: {t:.1f}s" for i, t in enumerate(pose_times, start=1)),
        transform=ax.transAxes,
        fontsize=10,
        weight="bold",
        va="top",
        color=BLACK,
    )

    xlim, ylim = axis_limits(path_xy, pose_sets)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)", fontsize=12, weight="bold")
    ax.set_ylabel("y (m)", fontsize=12, weight="bold")
    ax.grid(True, alpha=0.10)
    for spine in ax.spines.values():
        spine.set_color("#E6E6E6")

    scale_x = xlim[0] + 0.08 * (xlim[1] - xlim[0])
    scale_y = ylim[0] + 0.08 * (ylim[1] - ylim[0])
    ax.plot([scale_x, scale_x + 0.5], [scale_y, scale_y], color=BLACK, linewidth=3, solid_capstyle="round")
    ax.text(scale_x + 0.25, scale_y + 0.045, "0.5 m", ha="center", va="bottom", fontsize=10, weight="bold", color=BLACK)

    fig.tight_layout()
    out_path = out_dir / f"{safe_name(video_stem)}_relative_pose_teal_red_5poses.jpg"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_contact_sheet(paths: list[Path], out_path: Path, columns: int = 2) -> None:
    if not paths:
        return
    images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths]
    images = [image for image in images if image is not None]
    if not images:
        return
    target_w = 1200
    resized = []
    for image in images:
        h, w = image.shape[:2]
        scale = target_w / max(w, 1)
        resized.append(cv2.resize(image, (target_w, int(round(h * scale))), interpolation=cv2.INTER_AREA))
    rows = []
    for i in range(0, len(resized), columns):
        row_imgs = resized[i : i + columns]
        max_h = max(img.shape[0] for img in row_imgs)
        padded = []
        for img in row_imgs:
            if img.shape[0] < max_h:
                pad = np.full((max_h - img.shape[0], img.shape[1], 3), 255, dtype=np.uint8)
                img = np.vstack([img, pad])
            padded.append(img)
        while len(padded) < columns:
            padded.append(np.full((max_h, target_w, 3), 255, dtype=np.uint8))
        rows.append(np.hstack(padded))
    sheet = np.vstack(rows)
    cv2.imwrite(str(out_path), sheet)


def main() -> None:
    args = parse_args()
    analysis_dir = args.analysis_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else analysis_dir.with_name(f"{analysis_dir.name}_relative_pose_teal_red")
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for child in sorted(analysis_dir.iterdir()):
        if child.is_dir():
            out = draw_video_figure(child, out_dir, args.sample_count, args.px_per_m, args.max_poses)
            if out is not None:
                outputs.append(out)

    prefix = safe_name(args.prefix or analysis_dir.name)
    make_contact_sheet(outputs, out_dir / f"{prefix}_relative_pose_teal_red_contact_sheet.jpg")
    print(json.dumps({"out_dir": str(out_dir), "figures": [str(path) for path in outputs]}, indent=2))


if __name__ == "__main__":
    main()
