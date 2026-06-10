from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


PX_PER_M_DEFAULT = 875.0 / 1.5
DEFAULT_STEPS = (
    "plot_fixed_gait_trajectories.py",
    "make_tracked_center_cleaned_physical.py",
    "track_video_start_to_wall.py",
    "plot_fitted_gait_curves.py",
    "make_real_sim_comparison_panels.py",
)


OUTPUT_IMAGES = (
    "straight_real_vs_mujoco.png",
    "turn_left_real_vs_mujoco.png",
    "spin_left_real_vs_mujoco.png",
    "real_vs_mujoco_all.png",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full real-video recognition and MuJoCo comparison workflow, "
            "then export meter-based R/speed metrics."
        )
    )
    parser.add_argument(
        "--px-per-m",
        type=float,
        default=PX_PER_M_DEFAULT,
        help="Pixel-to-meter scale. Default: 875 / 1.5 = 583.333 px/m.",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Do not rerun the analysis scripts; only rebuild metric_summary_m.json from existing outputs.",
    )
    parser.add_argument(
        "--copy-to-pictures",
        type=Path,
        default=None,
        help="Optional directory, e.g. C:/Users/ytyla/Pictures, to copy final PNG panels into.",
    )
    return parser.parse_args()


def run_step(script: str) -> None:
    print(f"\n=== {script} ===")
    subprocess.run([sys.executable, script], check=True)


def load_json(path: Path, fallback: Any):
    if not path.exists():
        print(f"missing optional output: {path}")
        return fallback
    return json.loads(path.read_text(encoding="utf-8"))


def real_metric_row(item: dict[str, Any], px_per_m: float) -> dict[str, Any]:
    row: dict[str, Any] = {
        "clip": item.get("clip"),
        "kind": item.get("kind"),
        "video": item.get("video"),
        "point_count": item.get("point_count"),
        "wall_seconds": item.get("wall_seconds"),
        "px_per_m": px_per_m,
    }

    if item.get("kind") == "circle":
        radius_px = item.get("radius_px")
        rmse_px = item.get("rmse_px")
        row.update(
            {
                "radius_px": radius_px,
                "radius_m": None if radius_px is None else radius_px / px_per_m,
                "arc_deg": item.get("arc_deg"),
                "rmse_px": rmse_px,
                "rmse_m": None if rmse_px is None else rmse_px / px_per_m,
            }
        )
    elif item.get("kind") == "line":
        line_start = item.get("line_start_px") or [None, None]
        line_end = item.get("line_end_px") or [None, None]
        length_px = item.get("length_px")
        rmse_px = item.get("rmse_px")
        wall_seconds = float(item.get("wall_seconds") or 0.0)

        forward_px = None
        if line_start[1] is not None and line_end[1] is not None:
            forward_px = abs(float(line_end[1]) - float(line_start[1]))

        forward_m = None if forward_px is None else forward_px / px_per_m
        row.update(
            {
                "line_start_px": line_start,
                "line_end_px": line_end,
                "forward_distance_px": forward_px,
                "forward_distance_m": forward_m,
                "forward_speed_m_s": None
                if forward_m is None or wall_seconds <= 0.0
                else forward_m / wall_seconds,
                "line_length_px": length_px,
                "line_length_m": None if length_px is None else length_px / px_per_m,
                "rmse_px": rmse_px,
                "rmse_m": None if rmse_px is None else rmse_px / px_per_m,
            }
        )
    return row


def sim_metric_row(item: dict[str, Any], fixed_gait_rows: list[dict[str, Any]]) -> dict[str, Any]:
    name = item.get("name")
    fixed_by_name = {row.get("name"): row for row in fixed_gait_rows}
    fixed_row = fixed_by_name.get(name, {})

    row: dict[str, Any] = {
        "name": name,
        "kind": item.get("kind"),
        "radius_m": item.get("radius"),
        "arc_deg": item.get("arc_deg"),
        "fit_rmse_m": item.get("rmse"),
        "mean_path_speed_m_s": item.get("mean_speed_m_s"),
        "mean_forward_speed_m_s": item.get("mean_forward_speed_m_s"),
        "forward_displacement_m": item.get("forward_displacement_m"),
        "lateral_drift_m": item.get("lateral_drift_m"),
        "duration_s": item.get("duration_s"),
    }

    if fixed_row:
        row.update(
            {
                "fixed_dx_m": fixed_row.get("dx"),
                "fixed_dy_m": fixed_row.get("dy"),
                "fixed_forward_speed_m_s": fixed_row.get("forward_speed_m_s"),
                "fixed_path_speed_m_s": fixed_row.get("speed_m_s"),
                "fixed_hit_wall": fixed_row.get("hit_wall"),
            }
        )
    return row


def build_metric_summary(px_per_m: float) -> Path:
    out_dir = Path("outputs/real_sim_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    real_summary = load_json(Path("outputs/video_start_to_wall/summary.json"), [])
    sim_summary = load_json(Path("outputs/fitted_curve_comparison/sim_fitted_summary.json"), [])
    fixed_gait_summary = load_json(Path("outputs/fixed_gait_trajectories_3x1_5/summary.json"), [])

    metric_summary = {
        "unit_note": "Real-video radius_m = radius_px / px_per_m. MuJoCo radius is already in meters.",
        "px_per_m": px_per_m,
        "real": [real_metric_row(item, px_per_m) for item in real_summary],
        "mujoco": [sim_metric_row(item, fixed_gait_summary) for item in sim_summary],
        "final_images": [str(out_dir / image_name) for image_name in OUTPUT_IMAGES],
    }

    out_path = out_dir / "metric_summary_m.json"
    out_path.write_text(json.dumps(metric_summary, indent=2), encoding="utf-8")
    print(f"\nmetric summary: {out_path}")
    return out_path


def copy_images_to_pictures(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    source_dir = Path("outputs/real_sim_comparison")
    for image_name in OUTPUT_IMAGES:
        source = source_dir / image_name
        if not source.exists():
            print(f"skip missing image: {source}")
            continue
        target = destination / image_name
        shutil.copy2(source, target)
        print(f"copied {source} -> {target}")


def main() -> None:
    args = parse_args()

    if not args.skip_run:
        for script in DEFAULT_STEPS:
            run_step(script)

    build_metric_summary(args.px_per_m)

    if args.copy_to_pictures is not None:
        copy_images_to_pictures(args.copy_to_pictures)

    print("\nDone. Main outputs:")
    print("  outputs/real_sim_comparison/metric_summary_m.json")
    for image_name in OUTPUT_IMAGES:
        print(f"  outputs/real_sim_comparison/{image_name}")


if __name__ == "__main__":
    main()
