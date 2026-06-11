from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from track_video_start_to_wall import ClipConfig, detect_points, draw_fit, fit_circle, fit_line


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
MIN_TURN_NET_PX = 120.0
MIN_TURN_ARC_DEG = 60.0
MAX_TURN_STRAIGHTNESS = 0.78
MAX_TURN_CIRCLE_RMSE_OVER_RADIUS = 0.25
MAX_TURN_CIRCLE_LINE_RMSE_RATIO = 0.90

MIN_VALID_LINE_POINTS = 10
MIN_VALID_LINE_NET_PX = 120.0
MIN_VALID_LINE_PATH_PX = 160.0
MAX_LINE_EDGE_FRACTION = 0.60
BOTTOM_LEFT_X_FRAC = 0.42
BOTTOM_Y_FRAC = 0.72
RIGHT_EDGE_X_FRAC = 0.90
TOP_EDGE_Y_FRAC = 0.06


def parse_args():
    parser = argparse.ArgumentParser(description="Export video_analysis JSON using the legacy start-to-wall tracker.")
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
    xy = np.asarray([[p[1], p[2]] for p in points], dtype=float)
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


def line_track_quality(points, video_path: Path) -> dict:
    point_count = len(points)
    net_px = straight_distance(points)
    path_px = path_distance(points)
    quality = {
        "valid_track": True,
        "invalid_reason": None,
        "valid_line_point_count": point_count,
        "valid_line_net_px": net_px,
        "valid_line_path_px": path_px,
        "valid_line_edge_fraction": 0.0,
        "valid_line_corner_like": False,
    }

    if point_count < MIN_VALID_LINE_POINTS:
        return {**quality, "valid_track": False, "invalid_reason": "too few tracked points"}
    if net_px < MIN_VALID_LINE_NET_PX or path_px < MIN_VALID_LINE_PATH_PX:
        return {**quality, "valid_track": False, "invalid_reason": "track is too short"}

    shape = first_frame_shape(video_path)
    if shape is None:
        return quality
    h, w = shape
    xy = np.asarray([[p[1], p[2]] for p in points], dtype=float)
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
            "valid_line_median_x_px": median_x,
            "valid_line_median_y_px": median_y,
            "valid_line_frame_width_px": w,
            "valid_line_frame_height_px": h,
            "valid_line_edge_fraction": edge_fraction,
            "valid_line_corner_like": bool(bottom_left),
        }
    )
    if bottom_left:
        return {**quality, "valid_track": False, "invalid_reason": "track is concentrated in bottom-left overlay/corner"}
    if edge_fraction >= MAX_LINE_EDGE_FRACTION:
        return {**quality, "valid_track": False, "invalid_reason": "too many points are on image edges"}
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

    line_fit = {**line_fit, **auto, "auto_fit_kind": "line", "auto_fit_reason": "8s probe did not show a clear turn; remeasure as straight for 13s"}
    return line_curve, line_fit, "line"


def fit_points(points, requested_fit_kind: str):
    if len(points) < 2:
        return None, {"status": "not_enough_points", "radius_px": None, "arc_deg": 0.0, "rmse_px": None}, requested_fit_kind
    xy = np.asarray([[p[1], p[2]] for p in points], dtype=float)
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
    probe_points = detect_points(probe_clip, Path.cwd())
    probe_curve, probe_fit, probe_fit_kind = fit_points(probe_points, probe_clip.fit_kind)

    clip = probe_clip
    points = probe_points
    curve = probe_curve
    fit = probe_fit
    selected_fit_kind = probe_fit_kind
    measurement_mode = "turn_8s_probe"
    track_quality = {"valid_track": True, "invalid_reason": None}

    if probe_fit_kind == "line":
        clip = make_line_clip(video_path)
        points = detect_points(clip, Path.cwd())
        track_quality = line_track_quality(points, video_path)
        if not track_quality["valid_track"]:
            curve = None
            fit = {"kind": "invalid_track", "status": "filtered_too_short_or_corner", "rmse_px": None}
            selected_fit_kind = "invalid_track"
            measurement_mode = "invalid_line_after_no_turn"
        else:
            curve, fit, selected_fit_kind = fit_points(points, "line")
            measurement_mode = "line_13s_after_no_turn"

    preview_path = out_dir / preview_name
    if write_preview and curve is not None:
        draw_fit(video_path, clip.wall_seconds, points, curve, preview_path)
    elif write_preview and selected_fit_kind == "invalid_track":
        draw_invalid_preview(video_path, clip.wall_seconds, preview_path, str(track_quality.get("invalid_reason") or "invalid track"), len(points))

    result = {
        "tracker_version": "legacy_start_to_wall_preview_v6_probe8_line13_invalid_filter",
        "video": str(video_path),
        "video_name": video_path.name,
        "video_stem": video_path.stem,
        "clip_key": clip.key,
        "requested_fit_kind": "auto_probe_then_measure",
        "fit_kind": selected_fit_kind,
        "measurement_mode": measurement_mode,
        "valid_track": bool(track_quality.get("valid_track", True)),
        "invalid_reason": track_quality.get("invalid_reason"),
        "wall_seconds": clip.wall_seconds,
        "probe_wall_seconds": probe_clip.wall_seconds,
        "probe_point_count": len(probe_points),
        "probe_fit_kind": probe_fit_kind,
        "probe_auto_fit_reason": probe_fit.get("auto_fit_reason"),
        "probe_auto_fit_straightness": probe_fit.get("auto_fit_straightness"),
        "probe_auto_fit_line_rmse_px": probe_fit.get("auto_fit_line_rmse_px"),
        "probe_auto_fit_circle_rmse_px": probe_fit.get("auto_fit_circle_rmse_px"),
        "probe_auto_fit_circle_rmse_over_radius": probe_fit.get("auto_fit_circle_rmse_over_radius"),
        "probe_auto_fit_circle_arc_deg": probe_fit.get("auto_fit_circle_arc_deg"),
        **track_quality,
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
