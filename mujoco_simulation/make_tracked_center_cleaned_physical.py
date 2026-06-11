from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from track_video_start_to_wall import ClipConfig, draw_fit, fit_circle, fit_line


DEFAULT_VIDEO_STEMS = (
    "clean_v_20260607_233739",
    "clean_v_20260608_141203",
    "clean_v_20260608_141254",
)
DEFAULT_RECORDINGS_DIR = Path("../Release/python_backend/recordings")
DEFAULT_OUT_ROOT = Path("outputs/video_analysis")
DEFAULT_OUTPUT_NAME = "tracked_center_summary_cleaned_physical.json"
DEFAULT_PREVIEW_NAME = "tracked_center_overlay_cleaned_physical.png"
DEFAULT_PX_PER_M = 875.0 / 1.5

PROBE_SECONDS = 8.0
LINE_MEASURE_SECONDS = 13.0
SAMPLE_HZ = 5.0
TOP_CANDIDATES_PER_FRAME = 8
MAX_LINK_STEP_PX = 185.0
SOFT_LINK_STEP_PX = 85.0
MAX_SEGMENT_JUMP_PX = 170.0

MIN_PROBE_POINTS = 6
MIN_PROBE_NET_PX = 45.0
MIN_PROBE_PATH_PX = 55.0
MIN_LINE_POINTS = 10
MIN_LINE_NET_PX = 120.0
MIN_LINE_PATH_PX = 160.0

MIN_TURN_NET_PX = 120.0
MIN_TURN_ARC_DEG = 60.0
MAX_TURN_STRAIGHTNESS = 0.78
MAX_TURN_CIRCLE_RMSE_OVER_RADIUS = 0.25
MAX_TURN_CIRCLE_LINE_RMSE_RATIO = 0.90

EDGE_FRACTION_LIMIT = 0.60
BOTTOM_LEFT_X_FRAC = 0.42
BOTTOM_Y_FRAC = 0.72
RIGHT_EDGE_X_FRAC = 0.90
TOP_EDGE_Y_FRAC = 0.06


@dataclass
class Candidate:
    frame_index: int
    time_s: float
    x: float
    y: float
    area: int
    quality: float


def parse_args():
    parser = argparse.ArgumentParser(description="Export video_analysis JSON using robust selected-point tracking.")
    parser.add_argument("--recordings-dir", type=Path, default=DEFAULT_RECORDINGS_DIR)
    parser.add_argument("--videos", nargs="*", default=[f"{stem}.mp4" for stem in DEFAULT_VIDEO_STEMS])
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--preview-name", default=DEFAULT_PREVIEW_NAME)
    parser.add_argument("--px-per-m", type=float, default=DEFAULT_PX_PER_M)
    parser.add_argument("--no-preview", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def resolve_recordings_dir(path: Path) -> Path:
    resolved = resolve_path(path)
    if resolved.exists():
        return resolved
    fallback = resolve_path(Path("Release/python_backend/recordings"))
    return fallback if fallback.exists() else resolved


def resolve_video(video: str, recordings_dir: Path) -> Path:
    path = Path(video)
    if path.is_absolute():
        return path
    if path.parent != Path("."):
        return resolve_path(path)
    return (recordings_dir / path).resolve()


def make_probe_clip(video_path: Path) -> ClipConfig:
    return ClipConfig(
        f"{video_path.stem}_probe_8s",
        video_path,
        PROBE_SECONDS,
        "auto",
        roi=(80, 620, 940, 1240),
        min_y=1000.0,
    )


def make_line_clip(video_path: Path) -> ClipConfig:
    return ClipConfig(
        f"{video_path.stem}_line_13s",
        video_path,
        LINE_MEASURE_SECONDS,
        "line",
        roi=(80, 0, 940, 1850),
        min_y=80.0,
    )


def make_clip(video_path: Path) -> ClipConfig:
    return make_probe_clip(video_path)


def path_distance(points) -> float:
    if len(points) < 2:
        return 0.0
    xy = np.asarray([[p[1], p[2]] for p in points], dtype=np.float64)
    return float(np.sum(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))))


def straight_distance(points) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.hypot(points[-1][1] - points[0][1], points[-1][2] - points[0][2]))


def first_frame_shape(video_path: Path) -> tuple[int, int] | None:
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    h, w = frame.shape[:2]
    return int(h), int(w)


def valid_frame_bounds(frame_shape, clip: ClipConfig) -> tuple[float, float, float, float]:
    h, w = frame_shape[:2]
    x0, y0, rw, rh = clip.roi
    left = max(125.0, float(x0 + 18))
    right = min(float(w - 90), float(x0 + rw - 18))
    top = max(float(clip.min_y), float(y0 + 70))
    bottom = min(float(h - 135), float(y0 + rh - 18))
    return left, top, right, bottom


def is_forbidden_point(x: float, y: float, frame_shape) -> bool:
    h, w = frame_shape[:2]
    if x < BOTTOM_LEFT_X_FRAC * w and y > BOTTOM_Y_FRAC * h:
        return True
    if x > RIGHT_EDGE_X_FRAC * w or y < TOP_EDGE_Y_FRAC * h:
        return True
    return False


def inside_bounds(x: float, y: float, bounds: tuple[float, float, float, float]) -> bool:
    left, top, right, bottom = bounds
    return left <= x <= right and top <= y <= bottom


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


def build_mask(crop: np.ndarray, clip: ClipConfig, roi_top: int) -> np.ndarray:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    blue, green, red = cv2.split(crop)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    nonblue = (blue.astype(np.int16) - red.astype(np.int16) < 42) & (blue.astype(np.int16) - green.astype(np.int16) < 42)
    body = (saturation < 88) & (value > 45) & (value < 248) & nonblue
    dark_marker = (value < 105) & nonblue
    mask = ((body | dark_marker).astype(np.uint8)) * 255

    if clip.fit_kind != "line":
        mask[: max(70, min(130, 1040 - roi_top)), :] = 0
    else:
        mask[:40, :] = 0
        mask[:, 1000:] = 0

    kernel3 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel5)
    return mask


def candidates_for_frame(frame: np.ndarray, frame_index: int, time_s: float, clip: ClipConfig) -> list[Candidate]:
    bounds = valid_frame_bounds(frame.shape, clip)
    x0, y0, rw, rh = clip.roi
    crop = frame[y0 : y0 + rh, x0 : x0 + rw]
    mask = build_mask(crop, clip, y0)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    candidates: list[Candidate] = []
    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bx = int(stats[i, cv2.CC_STAT_LEFT])
        by = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        quality = component_quality(area, bw, bh)
        if quality is None:
            continue
        if bx <= 18 or by <= 18 or bx + bw >= rw - 18 or by + bh >= rh - 18:
            continue
        cx, cy = centroids[i]
        gx, gy = float(x0 + cx), float(y0 + cy)
        if not inside_bounds(gx, gy, bounds):
            continue
        if is_forbidden_point(gx, gy, frame.shape):
            continue
        candidates.append(Candidate(frame_index, time_s, gx, gy, area, float(quality)))
    candidates.sort(key=lambda c: c.quality, reverse=True)
    return candidates[:TOP_CANDIDATES_PER_FRAME]


def transition_score(prev: Candidate, cur: Candidate) -> float | None:
    dist = float(np.hypot(cur.x - prev.x, cur.y - prev.y))
    frame_gap = max(1, cur.frame_index - prev.frame_index)
    if dist > MAX_LINK_STEP_PX * frame_gap:
        return None
    jump_penalty = 2.4 * dist + 4.0 * max(0.0, dist - SOFT_LINK_STEP_PX * frame_gap)
    gap_penalty = 12.0 * (frame_gap - 1)
    return -jump_penalty - gap_penalty


def best_candidate_path(groups: list[list[Candidate]]) -> list[Candidate]:
    flat = [candidate for group in groups for candidate in group]
    if not flat:
        return []

    scores: list[list[float]] = []
    parents: list[list[tuple[int, int] | None]] = []
    for group_index, group in enumerate(groups):
        group_scores = []
        group_parents = []
        for cand in group:
            best_score = cand.quality - 0.12 * abs(cand.x - 560.0) - 0.08 * abs(cand.y - 1320.0)
            best_parent = None
            for prev_group_index in range(max(0, group_index - 4), group_index):
                for prev_candidate_index, prev in enumerate(groups[prev_group_index]):
                    trans = transition_score(prev, cand)
                    if trans is None:
                        continue
                    candidate_score = scores[prev_group_index][prev_candidate_index] + cand.quality + trans
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_parent = (prev_group_index, prev_candidate_index)
            group_scores.append(best_score)
            group_parents.append(best_parent)
        scores.append(group_scores)
        parents.append(group_parents)

    best_group = 0
    best_index = 0
    best_value = -1e18
    for gi, group_scores in enumerate(scores):
        for ci, score in enumerate(group_scores):
            if score > best_value:
                best_value = score
                best_group = gi
                best_index = ci

    path: list[Candidate] = []
    cursor: tuple[int, int] | None = (best_group, best_index)
    while cursor is not None:
        gi, ci = cursor
        path.append(groups[gi][ci])
        cursor = parents[gi][ci]
    path.reverse()
    return path


def split_good_segments(path: list[Candidate], frame_shape, min_y: float) -> list[list[Candidate]]:
    segments: list[list[Candidate]] = []
    current: list[Candidate] = []
    prev: Candidate | None = None
    for cand in path:
        if cand.y <= min_y or is_forbidden_point(cand.x, cand.y, frame_shape):
            if current:
                segments.append(current)
                current = []
            prev = None
            continue
        if prev is not None:
            dist = float(np.hypot(cand.x - prev.x, cand.y - prev.y))
            if dist > MAX_SEGMENT_JUMP_PX:
                if current:
                    segments.append(current)
                current = []
        current.append(cand)
        prev = cand
    if current:
        segments.append(current)
    return segments


def segment_score(segment: list[Candidate]) -> float:
    if len(segment) < 2:
        return -1e9
    points = [(c.time_s, c.x, c.y, c.area) for c in segment]
    net = straight_distance(points)
    path = path_distance(points)
    return 14.0 * len(segment) + 0.10 * path + 0.08 * net


def clean_segment_points(segment: list[Candidate]) -> list[tuple[float, float, float, int]]:
    if not segment:
        return []
    points = [(c.time_s, c.x, c.y, c.area) for c in segment]
    if len(points) < 5:
        return points
    xy = np.asarray([[p[1], p[2]] for p in points], dtype=np.float64)
    smoothed = []
    for i, point in enumerate(points):
        lo = max(0, i - 1)
        hi = min(len(points), i + 2)
        mx, my = np.median(xy[lo:hi], axis=0)
        smoothed.append((point[0], float(mx), float(my), point[3]))
    return smoothed


def detect_points_robust(clip: ClipConfig, root: Path) -> tuple[list[tuple[float, float, float, int]], dict]:
    video_path = root / clip.video
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    sample_step = max(1, int(round(fps / SAMPLE_HZ)))
    groups: list[list[Candidate]] = []
    frame_shape = None
    raw_candidate_count = 0

    for frame_idx in range(0, int(clip.wall_seconds * fps) + 1, sample_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break
        frame_shape = frame.shape
        candidates = candidates_for_frame(frame, frame_idx, float(frame_idx / fps), clip)
        raw_candidate_count += len(candidates)
        if candidates:
            groups.append(candidates)
    cap.release()

    if frame_shape is None or not groups:
        return [], {
            "raw_candidate_count": raw_candidate_count,
            "candidate_frame_count": len(groups),
            "selected_segment_count": 0,
            "tracking_reason": "no candidates after mask/bounds filtering",
        }

    path = best_candidate_path(groups)
    segments = split_good_segments(path, frame_shape, clip.min_y)
    if not segments:
        return [], {
            "raw_candidate_count": raw_candidate_count,
            "candidate_frame_count": len(groups),
            "selected_segment_count": 0,
            "tracking_reason": "all path points were overlay/corner/edge points",
        }

    best_segment = max(segments, key=segment_score)
    points = clean_segment_points(best_segment)
    return points, {
        "raw_candidate_count": raw_candidate_count,
        "candidate_frame_count": len(groups),
        "selected_path_count": len(path),
        "selected_segment_count": len(best_segment),
        "dropped_segment_count": max(0, len(path) - len(best_segment)),
        "tracking_reason": "robust multi-candidate path selected",
    }


def track_quality(points, video_path: Path, mode: str) -> dict:
    point_count = len(points)
    net_px = straight_distance(points)
    path_px = path_distance(points)
    min_points = MIN_PROBE_POINTS if mode == "probe" else MIN_LINE_POINTS
    min_net = MIN_PROBE_NET_PX if mode == "probe" else MIN_LINE_NET_PX
    min_path = MIN_PROBE_PATH_PX if mode == "probe" else MIN_LINE_PATH_PX
    quality = {
        "valid_track": True,
        "invalid_reason": None,
        "quality_point_count": point_count,
        "quality_net_px": net_px,
        "quality_path_px": path_px,
        "quality_edge_fraction": 0.0,
        "quality_corner_like": False,
    }

    if point_count < min_points:
        return {**quality, "valid_track": False, "invalid_reason": f"too few {mode} points"}
    if net_px < min_net or path_px < min_path:
        return {**quality, "valid_track": False, "invalid_reason": f"{mode} track is too short"}

    shape = first_frame_shape(video_path)
    if shape is None:
        return quality
    h, w = shape
    xy = np.asarray([[p[1], p[2]] for p in points], dtype=np.float64)
    median_x = float(np.median(xy[:, 0]))
    median_y = float(np.median(xy[:, 1]))
    bottom_left = median_x < BOTTOM_LEFT_X_FRAC * w and median_y > BOTTOM_Y_FRAC * h
    edge_mask = (
        (xy[:, 0] < 0.12 * w)
        | (xy[:, 0] > RIGHT_EDGE_X_FRAC * w)
        | (xy[:, 1] < TOP_EDGE_Y_FRAC * h)
        | (xy[:, 1] > 0.78 * h)
    )
    edge_fraction = float(np.mean(edge_mask))
    quality.update(
        {
            "quality_median_x_px": median_x,
            "quality_median_y_px": median_y,
            "quality_frame_width_px": w,
            "quality_frame_height_px": h,
            "quality_edge_fraction": edge_fraction,
            "quality_corner_like": bool(bottom_left),
        }
    )
    if bottom_left:
        return {**quality, "valid_track": False, "invalid_reason": f"{mode} track is concentrated in bottom-left overlay/corner"}
    if edge_fraction >= EDGE_FRACTION_LIMIT:
        return {**quality, "valid_track": False, "invalid_reason": f"too many {mode} points are on image edges"}
    return quality


def choose_auto_fit(points, line_curve, line_fit, circle_curve, circle_fit):
    path_px = path_distance(points)
    net_px = straight_distance(points)
    straightness = 0.0 if path_px <= 1e-9 else net_px / path_px
    line_rmse = float(line_fit.get("rmse_px", 1e9) or 1e9)
    circle_rmse = float(circle_fit.get("rmse_px", 1e9) or 1e9)
    radius_px = circle_fit.get("radius_px")
    arc_deg = float(circle_fit.get("arc_deg", 0.0) or 0.0)
    circle_rmse_over_r = None
    if radius_px is not None and float(radius_px) > 1e-9:
        circle_rmse_over_r = circle_rmse / float(radius_px)

    auto = {
        "auto_fit_path_px": path_px,
        "auto_fit_net_px": net_px,
        "auto_fit_straightness": straightness,
        "auto_fit_line_rmse_px": line_rmse,
        "auto_fit_circle_rmse_px": circle_rmse,
        "auto_fit_circle_rmse_over_radius": circle_rmse_over_r,
        "auto_fit_circle_arc_deg": arc_deg,
    }

    circle_is_clear_turn = (
        net_px >= MIN_TURN_NET_PX
        and radius_px is not None
        and arc_deg >= MIN_TURN_ARC_DEG
        and straightness <= MAX_TURN_STRAIGHTNESS
        and circle_rmse_over_r is not None
        and circle_rmse_over_r <= MAX_TURN_CIRCLE_RMSE_OVER_RADIUS
        and circle_rmse <= MAX_TURN_CIRCLE_LINE_RMSE_RATIO * max(line_rmse, 1.0)
    )

    if circle_is_clear_turn:
        circle_fit = {**circle_fit, **auto, "auto_fit_kind": "circle", "auto_fit_reason": "8s probe found a clear circular turn"}
        return circle_curve, circle_fit, "circle"

    line_fit = {**line_fit, **auto, "auto_fit_kind": "line", "auto_fit_reason": "8s probe was valid but not a clear turn; remeasure as straight for 13s"}
    return line_curve, line_fit, "line"


def fit_points(points, requested_fit_kind: str):
    if len(points) < 2:
        return None, {"status": "not_enough_points", "radius_px": None, "arc_deg": 0.0, "rmse_px": None}, requested_fit_kind
    xy = np.asarray([[p[1], p[2]] for p in points], dtype=np.float64)
    if requested_fit_kind == "line" or len(points) < 3:
        curve, fit = fit_line(xy)
        return curve, fit, "line"
    if requested_fit_kind == "circle":
        curve, fit = fit_circle(xy)
        return curve, fit, "circle"
    line_curve, line_fit = fit_line(xy)
    circle_curve, circle_fit = fit_circle(xy)
    return choose_auto_fit(points, line_curve, line_fit, circle_curve, circle_fit)


def add_metric_units(result: dict, fit: dict, fit_kind: str, wall_seconds: float, px_per_m: float) -> dict:
    result["px_per_m"] = float(px_per_m)
    result["path_distance_m"] = result["path_distance_px"] / px_per_m
    result["straight_distance_m"] = result["straight_distance_px"] / px_per_m

    if fit_kind == "circle" and fit.get("radius_px") is not None:
        result["radius_m"] = fit["radius_px"] / px_per_m
        result["rmse_m"] = fit["rmse_px"] / px_per_m
        return result

    if fit_kind == "line" and "line_start_px" in fit and "line_end_px" in fit:
        line_start_y = float(fit["line_start_px"][1])
        line_end_y = float(fit["line_end_px"][1])
        forward_px = abs(line_end_y - line_start_y)
        forward_m = forward_px / px_per_m
        result["forward_distance_px"] = forward_px
        result["forward_distance_m"] = forward_m
        result["forward_speed_m_s"] = None if wall_seconds <= 1e-9 else forward_m / wall_seconds
        result["forward_speed_source"] = "abs(fitted_line_end_y - fitted_line_start_y) / px_per_m / wall_seconds"
        result["line_length_m"] = fit["length_px"] / px_per_m
        result["rmse_m"] = fit["rmse_px"] / px_per_m
    return result


def draw_text_outline(frame, text: str, org: tuple[int, int], scale: float = 0.8, color=(255, 255, 255)) -> None:
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def draw_invalid_preview(video_path: Path, seconds: float, out_path: Path, reason: str, point_count: int) -> None:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(seconds * fps)))
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return
    lines = [
        "Invalid track  REAL",
        f"Real: {point_count} pts",
        "speed invalid",
        "forward invalid",
        reason[:36],
    ]
    x, y = 28, 48
    for line in lines:
        draw_text_outline(frame, line, (x, y), scale=0.78)
        y += 32
    cv2.imwrite(str(out_path), frame)


def annotate_preview(preview_path: Path, result: dict) -> None:
    if not preview_path.exists():
        return
    frame = cv2.imread(str(preview_path))
    if frame is None:
        return

    if result.get("fit_kind") == "invalid_track":
        lines = [
            "Invalid track  REAL",
            f"Real: {result.get('point_count', 0)} pts",
            "speed invalid",
            "forward invalid",
            str(result.get("invalid_reason", "check tracking"))[:36],
        ]
    elif result.get("fit_kind") == "line":
        speed = result.get("forward_speed_m_s")
        speed_text = "nan" if speed is None else f"{speed:.3f} m/s"
        lines = [
            "Straight swim  REAL",
            f"Real: {result.get('point_count', 0)} pts",
            f"speed {speed_text}",
            f"forward {result.get('forward_distance_m', result.get('straight_distance_m', 0.0)):.3f} m",
            f"line RMSE {result.get('rmse_px', 0.0):.1f}px",
        ]
    else:
        lines = [
            f"{result.get('clip_key', 'turn')}  REAL",
            f"Real: {result.get('point_count', 0)} pts",
            f"R {result.get('radius_m', 0.0):.3f} m ({result.get('radius_px', 0.0):.1f}px)",
            f"arc {result.get('arc_deg', 0.0):.1f} deg",
            f"RMSE {result.get('rmse_px', 0.0):.1f}px",
        ]

    x, y = 28, 48
    for line in lines:
        draw_text_outline(frame, line, (x, y), scale=0.78)
        y += 32
    cv2.imwrite(str(preview_path), frame)


def process_video(
    video_path: Path,
    out_root: Path,
    output_name: str = DEFAULT_OUTPUT_NAME,
    preview_name: str = DEFAULT_PREVIEW_NAME,
    write_preview: bool = True,
    px_per_m: float = DEFAULT_PX_PER_M,
):
    out_dir = out_root / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    probe_clip = make_probe_clip(video_path)
    probe_points, probe_tracking = detect_points_robust(probe_clip, Path.cwd())
    probe_quality = track_quality(probe_points, video_path, "probe")

    clip = probe_clip
    points = probe_points
    curve = None
    fit = {"kind": "invalid_track", "status": "not_fit", "rmse_px": None}
    selected_fit_kind = "invalid_track"
    measurement_mode = "invalid_probe"
    final_quality = probe_quality
    tracking_info = {f"probe_{k}": v for k, v in probe_tracking.items()}

    if not probe_quality["valid_track"]:
        final_quality = probe_quality
        measurement_mode = "invalid_probe"
    else:
        probe_curve, probe_fit, probe_fit_kind = fit_points(probe_points, "auto")
        tracking_info.update(
            {
                "probe_fit_kind": probe_fit_kind,
                "probe_auto_fit_reason": probe_fit.get("auto_fit_reason"),
                "probe_auto_fit_straightness": probe_fit.get("auto_fit_straightness"),
                "probe_auto_fit_line_rmse_px": probe_fit.get("auto_fit_line_rmse_px"),
                "probe_auto_fit_circle_rmse_px": probe_fit.get("auto_fit_circle_rmse_px"),
                "probe_auto_fit_circle_rmse_over_radius": probe_fit.get("auto_fit_circle_rmse_over_radius"),
                "probe_auto_fit_circle_arc_deg": probe_fit.get("auto_fit_circle_arc_deg"),
            }
        )
        if probe_fit_kind == "circle":
            curve = probe_curve
            fit = probe_fit
            selected_fit_kind = "circle"
            measurement_mode = "turn_8s_probe"
        else:
            line_clip = make_line_clip(video_path)
            line_points, line_tracking = detect_points_robust(line_clip, Path.cwd())
            line_quality = track_quality(line_points, video_path, "line")
            tracking_info.update({f"line_{k}": v for k, v in line_tracking.items()})
            clip = line_clip
            points = line_points
            final_quality = line_quality
            if not line_quality["valid_track"]:
                measurement_mode = "invalid_line_after_valid_no_turn_probe"
            else:
                curve, fit, selected_fit_kind = fit_points(line_points, "line")
                measurement_mode = "line_13s_after_valid_no_turn_probe"

    preview_path = out_dir / preview_name
    if write_preview and curve is not None:
        draw_fit(video_path, clip.wall_seconds, points, curve, preview_path)
    elif write_preview and selected_fit_kind == "invalid_track":
        draw_invalid_preview(video_path, clip.wall_seconds, preview_path, str(final_quality.get("invalid_reason") or "invalid track"), len(points))

    result = {
        "tracker_version": "robust_multi_candidate_v1_probe8_line13",
        "video": str(video_path),
        "video_name": video_path.name,
        "video_stem": video_path.stem,
        "clip_key": clip.key,
        "requested_fit_kind": "robust_probe_then_measure",
        "fit_kind": selected_fit_kind,
        "measurement_mode": measurement_mode,
        "valid_track": bool(final_quality.get("valid_track", True)),
        "invalid_reason": final_quality.get("invalid_reason"),
        "wall_seconds": clip.wall_seconds,
        "probe_wall_seconds": probe_clip.wall_seconds,
        "probe_point_count": len(probe_points),
        **tracking_info,
        **final_quality,
        "roi": list(clip.roi),
        "min_y": clip.min_y,
        "points": [list(p) for p in points],
        "cleaned_points": [list(p) for p in points],
        "point_count": len(points),
        "straight_distance_px": straight_distance(points),
        "path_distance_px": path_distance(points),
        **fit,
        "preview_image": str(preview_path) if write_preview else None,
    }
    result = add_metric_units(result, fit, selected_fit_kind, clip.wall_seconds, px_per_m)
    if write_preview and preview_path.exists():
        annotate_preview(preview_path, result)

    out_path = out_dir / output_name
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(out_path)
    if selected_fit_kind == "circle":
        radius = "nan" if result["radius_px"] is None else f"{result['radius_px']:.3f}"
        rmse = "nan" if result["rmse_px"] is None else f"{result['rmse_px']:.3f}"
        print(f"fit=circle mode={measurement_mode} points={len(points)} radius_px={radius} arc_deg={result['arc_deg']:.3f} rmse_px={rmse} preview={result['preview_image']}")
    elif selected_fit_kind == "line":
        speed = result.get("forward_speed_m_s")
        speed_text = "nan" if speed is None else f"{speed:.4f}m/s"
        print(
            f"fit=line mode={measurement_mode} points={len(points)} line_length_px={result.get('length_px', 0.0):.3f} "
            f"forward_px={result.get('forward_distance_px', 0.0):.3f} "
            f"speed={speed_text} rmse_px={result.get('rmse_px', 0.0):.3f} preview={result['preview_image']}"
        )
    else:
        print(f"fit=invalid mode={measurement_mode} points={len(points)} reason={result.get('invalid_reason')} preview={result['preview_image']}")
    return out_path


def main():
    args = parse_args()
    recordings_dir = resolve_recordings_dir(args.recordings_dir)
    out_root = resolve_path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    generated = []
    for video in args.videos:
        video_path = resolve_video(video, recordings_dir)
        if not video_path.exists():
            print(f"skip missing video: {video_path}")
            continue
        generated.append(
            process_video(
                video_path,
                out_root,
                args.output_name,
                args.preview_name,
                not args.no_preview,
                args.px_per_m,
            )
        )
    print(f"\ngenerated {len(generated)} tracking json file(s)")


if __name__ == "__main__":
    main()
