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


def make_clip(video_path: Path) -> ClipConfig:
    stem = video_path.stem.lower()
    if "233739" in stem or "straight" in stem:
        return ClipConfig(
            "straight_233739",
            video_path,
            15.0,
            "line",
            roi=(80, 0, 940, 1850),
            min_y=80.0,
        )
    if "141254" in stem or "spin" in stem:
        key = "spin_left_141254"
    else:
        key = "turn_left_141203"
    return ClipConfig(key, video_path, 8.0, "circle", roi=(80, 620, 940, 1240), min_y=1000.0)


def path_distance(points) -> float:
    if len(points) < 2:
        return 0.0
    xy = np.asarray([[p[1], p[2]] for p in points], dtype=float)
    return float(np.sum(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))))


def straight_distance(points) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.hypot(points[-1][1] - points[0][1], points[-1][2] - points[0][2]))


def fit_points(points, fit_kind: str):
    if len(points) < 2:
        return None, {"status": "not_enough_points", "radius_px": None, "arc_deg": 0.0, "rmse_px": None}
    xy = np.asarray([[p[1], p[2]] for p in points], dtype=float)
    if fit_kind == "circle" and len(points) >= 3:
        return fit_circle(xy)
    return fit_line(xy)


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


def annotate_preview(preview_path: Path, result: dict) -> None:
    if not preview_path.exists():
        return
    frame = cv2.imread(str(preview_path))
    if frame is None:
        return

    if result.get("fit_kind") == "line":
        speed = result.get("forward_speed_m_s")
        speed_text = "nan" if speed is None else f"{speed:.3f} m/s"
        lines = [
            "Straight swim  REAL",
            f"Real: {result.get('point_count', 0)} pts",
            f"speed {speed_text}",
            f"forward {result.get('forward_distance_m', 0.0):.3f} m",
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
    clip = make_clip(video_path)
    out_dir = out_root / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    points = detect_points(clip, Path.cwd())
    curve, fit = fit_points(points, clip.fit_kind)
    preview_path = out_dir / preview_name
    if write_preview and curve is not None:
        draw_fit(video_path, clip.wall_seconds, points, curve, preview_path)

    result = {
        "tracker_version": "legacy_start_to_wall_preview_v3",
        "video": str(video_path),
        "video_name": video_path.name,
        "video_stem": video_path.stem,
        "clip_key": clip.key,
        "fit_kind": clip.fit_kind,
        "wall_seconds": clip.wall_seconds,
        "roi": list(clip.roi),
        "min_y": clip.min_y,
        "points": [list(p) for p in points],
        "cleaned_points": [list(p) for p in points],
        "point_count": len(points),
        "straight_distance_px": straight_distance(points),
        "path_distance_px": path_distance(points),
        **fit,
        "preview_image": str(preview_path) if write_preview and curve is not None else None,
    }
    result = add_metric_units(result, fit, clip.fit_kind, clip.wall_seconds, px_per_m)
    if write_preview and curve is not None:
        annotate_preview(preview_path, result)

    out_path = out_dir / output_name
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(out_path)
    if clip.fit_kind == "circle":
        radius = "nan" if result["radius_px"] is None else f"{result['radius_px']:.3f}"
        rmse = "nan" if result["rmse_px"] is None else f"{result['rmse_px']:.3f}"
        print(f"points={len(points)} radius_px={radius} arc_deg={result['arc_deg']:.3f} rmse_px={rmse} preview={result['preview_image']}")
    else:
        speed = result.get("forward_speed_m_s")
        speed_text = "nan" if speed is None else f"{speed:.4f}m/s"
        print(
            f"points={len(points)} line_length_px={result.get('length_px', 0.0):.3f} "
            f"forward_px={result.get('forward_distance_px', 0.0):.3f} "
            f"speed={speed_text} rmse_px={result.get('rmse_px', 0.0):.3f} preview={result['preview_image']}"
        )
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
