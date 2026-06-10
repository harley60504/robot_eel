from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def parse_args():
    parser = argparse.ArgumentParser(description="Export video_analysis JSON using the legacy start-to-wall tracker.")
    parser.add_argument("--recordings-dir", type=Path, default=DEFAULT_RECORDINGS_DIR)
    parser.add_argument("--videos", nargs="*", default=[f"{stem}.mp4" for stem in DEFAULT_VIDEO_STEMS])
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--preview-name", default=DEFAULT_PREVIEW_NAME)
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


def process_video(video_path: Path, out_root: Path, output_name: str, preview_name: str, write_preview: bool):
    clip = make_clip(video_path)
    out_dir = out_root / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    points = detect_points(clip, Path.cwd())
    curve, fit = fit_points(points, clip.fit_kind)
    preview_path = out_dir / preview_name
    if write_preview and curve is not None:
        draw_fit(video_path, clip.wall_seconds, points, curve, preview_path)

    result = {
        "tracker_version": "legacy_start_to_wall_preview_v1",
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

    out_path = out_dir / output_name
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(out_path)
    if clip.fit_kind == "circle":
        radius = "nan" if result["radius_px"] is None else f"{result['radius_px']:.3f}"
        rmse = "nan" if result["rmse_px"] is None else f"{result['rmse_px']:.3f}"
        print(f"points={len(points)} radius_px={radius} arc_deg={result['arc_deg']:.3f} rmse_px={rmse} preview={result['preview_image']}")
    else:
        print(f"points={len(points)} line_length_px={result.get('length_px', 0.0):.3f} rmse_px={result.get('rmse_px', 0.0):.3f} preview={result['preview_image']}")
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
        generated.append(process_video(video_path, out_root, args.output_name, args.preview_name, not args.no_preview))
    print(f"\ngenerated {len(generated)} tracking json file(s)")


if __name__ == "__main__":
    main()
