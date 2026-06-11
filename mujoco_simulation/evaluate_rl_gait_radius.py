from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from plot_fixed_gait_trajectories import run_gait
from plot_fitted_gait_curves import fitted_curve, rotate_sim_xy, trajectory_metrics
from sim_config import DEFAULT_START_X, DEFAULT_START_Y, EEL_MODEL_XML


REQUIRED_GAIT_FIELDS = ("ajoint", "freq", "wavelength", "amp_scales", "phase_lags", "joint_bias")
ALIASES = {
    "ajoint_deg": "ajoint",
    "freq_hz": "freq",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate RL-generated Hopf gait JSON files by running MuJoCo, "
            "saving trajectory CSVs, and fitting trajectory-circle radius R."
        )
    )
    parser.add_argument(
        "--gait",
        type=Path,
        action="append",
        default=[],
        help="Gait JSON to evaluate. May be passed multiple times.",
    )
    parser.add_argument(
        "--gait-dir",
        type=Path,
        default=Path("outputs/rl_gaits"),
        help="Directory scanned when --gait is not provided.",
    )
    parser.add_argument("--pattern", default="*.json", help="Glob pattern used with --gait-dir.")
    parser.add_argument("--xml", type=Path, default=Path(EEL_MODEL_XML))
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--fit-start-seconds", type=float, default=0.0)
    parser.add_argument("--start-x", type=float, default=DEFAULT_START_X)
    parser.add_argument("--start-y", type=float, default=DEFAULT_START_Y)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/rl_gait_radius_eval"),
        help="Output folder for normalized gait JSON, trajectory CSV, fitted curve CSV, and summaries.",
    )
    parser.add_argument(
        "--mujoco-xy",
        action="store_true",
        help="Fit in raw MuJoCo x-y coordinates instead of the rotated camera-view coordinates used by plot_fitted_gait_curves.py.",
    )
    parser.add_argument(
        "--allow-line",
        action="store_true",
        help="Allow nearly straight trajectories to be classified as a line. Default forces circle fit because this script is for R.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip writing PNG overview plots.",
    )
    return parser.parse_args()


def resolve_from_cwd(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "rl_gait"


def first_dict_with_gait_fields(raw: Any) -> dict[str, Any] | None:
    candidates: list[Any] = [raw]
    if isinstance(raw, dict):
        for key in ("gait", "best_gait", "gait_parameters", "params", "action_parameters"):
            if key in raw:
                candidates.append(raw[key])
        if isinstance(raw.get("source"), dict):
            candidates.append(raw["source"])

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        normalized = dict(candidate)
        for alias, target in ALIASES.items():
            if target not in normalized and alias in normalized:
                normalized[target] = normalized[alias]
        if all(key in normalized for key in REQUIRED_GAIT_FIELDS):
            return normalized
    return None


def load_and_normalize_gait(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    gait = first_dict_with_gait_fields(raw)
    if gait is None:
        raise ValueError(
            f"{path} does not contain a gait with required fields: {', '.join(REQUIRED_GAIT_FIELDS)}"
        )

    name = str(gait.get("name") or path.stem)
    normalized = {
        "name": name,
        "ajoint": float(gait["ajoint"]),
        "freq": float(gait["freq"]),
        "wavelength": float(gait["wavelength"]),
        "amp_scales": [float(value) for value in gait["amp_scales"]],
        "phase_lags": [float(value) for value in gait["phase_lags"]],
        "joint_bias": [float(value) for value in gait["joint_bias"]],
    }

    if len(normalized["amp_scales"]) != 6:
        raise ValueError(f"{path}: amp_scales must have 6 values")
    if len(normalized["phase_lags"]) != 5:
        raise ValueError(f"{path}: phase_lags must have 5 values")
    if len(normalized["joint_bias"]) != 6:
        raise ValueError(f"{path}: joint_bias must have 6 values")

    if isinstance(gait.get("source"), dict):
        normalized["source"] = gait["source"]
    else:
        normalized["source"] = {"type": "rl_or_manual_gait_json", "file": str(path)}

    return normalized


def collect_gait_paths(args: argparse.Namespace) -> list[Path]:
    if args.gait:
        return [resolve_from_cwd(path) for path in args.gait]

    gait_dir = resolve_from_cwd(args.gait_dir)
    paths = sorted(gait_dir.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(
            f"No gait JSON files found in {gait_dir} with pattern {args.pattern}. "
            "Pass --gait path/to/your_rl_gait.json if the file is somewhere else."
        )
    return paths


def fit_rows_after_start(arr: np.ndarray, fit_start_seconds: float) -> np.ndarray:
    if arr.shape[0] < 3 or fit_start_seconds <= 0.0:
        return arr
    mask = arr[:, 0] >= fit_start_seconds
    fit_arr = arr[mask]
    if fit_arr.shape[0] < 3:
        return arr
    return fit_arr


def signed_radius_from_lateral(radius: float | None, lateral_drift_m: float) -> float | None:
    if radius is None:
        return None
    if abs(lateral_drift_m) < 1e-9:
        return float(radius)
    return math.copysign(float(radius), lateral_drift_m)


def direction_from_signed_radius(signed_radius_m: float | None) -> str:
    if signed_radius_m is None:
        return "unknown"
    if signed_radius_m < 0.0:
        return "left"
    if signed_radius_m > 0.0:
        return "right"
    return "straight_or_unknown"


def save_plot(path: Path, name: str, xy: np.ndarray, curve: np.ndarray, fit: dict[str, Any], metrics: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=170)
    ax.plot(xy[:, 0], xy[:, 1], linewidth=1.2, label="trajectory")
    ax.plot(curve[:, 0], curve[:, 1], linewidth=2.2, label="fitted curve")
    ax.scatter([xy[0, 0]], [xy[0, 1]], s=34, marker="o", edgecolor="black", zorder=3, label="start")
    ax.scatter([xy[-1, 0]], [xy[-1, 1]], s=44, marker="x", linewidth=2.0, zorder=3, label="end")
    radius = fit.get("radius")
    radius_text = "R = line/inf" if radius is None else f"R = {float(radius):.4f} m"
    ax.set_title(f"{name} fitted trajectory radius")
    ax.text(
        0.02,
        0.98,
        f"{radius_text}\narc = {float(fit.get('arc_deg', 0.0)):.2f} deg\n"
        f"RMSE = {float(fit.get('rmse', 0.0)):.4f} m\n"
        f"lateral drift = {metrics['lateral_drift_m']:.4f} m",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.86, edgecolor="#cccccc"),
        fontsize=8,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("lateral (m)" if "rotated" in path.stem else "x (m)")
    ax.set_ylabel("forward (m)" if "rotated" in path.stem else "y (m)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def evaluate_one(path: Path, args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    gait = load_and_normalize_gait(path)
    name = safe_name(str(gait.get("name") or path.stem))
    normalized_path = out_dir / f"{name}_normalized_gait.json"
    normalized_path.write_text(json.dumps(gait, indent=2), encoding="utf-8")

    xml_path = resolve_from_cwd(args.xml)
    _, arr, hit_wall = run_gait(xml_path, normalized_path, args.seconds, args.start_x, args.start_y)
    if arr.shape[0] < 3:
        raise RuntimeError(f"{path}: MuJoCo trajectory has fewer than 3 points; cannot fit radius")

    trajectory_csv = out_dir / f"{name}_trajectory.csv"
    np.savetxt(
        trajectory_csv,
        arr,
        delimiter=",",
        header="time,x,y,yaw",
        comments="",
    )

    fit_arr = fit_rows_after_start(arr, args.fit_start_seconds)
    xy = fit_arr[:, 1:3]
    fit_coordinate_frame = "mujoco_xy"
    if not args.mujoco_xy:
        xy = rotate_sim_xy(xy)
        fit_coordinate_frame = "rotated_camera_view"

    curve, fit = fitted_curve(xy, count=260, force_circle=not args.allow_line)
    metrics = trajectory_metrics(fit_arr, xy)
    signed_radius_m = signed_radius_from_lateral(fit["radius"], metrics["lateral_drift_m"])

    fitted_curve_csv = out_dir / f"{name}_fitted_curve.csv"
    np.savetxt(
        fitted_curve_csv,
        curve,
        delimiter=",",
        header="x_fit,y_fit",
        comments="",
    )

    png_path = None
    if not args.no_plot:
        png_path = out_dir / f"{name}_{fit_coordinate_frame}_fit.png"
        save_plot(png_path, name, xy, curve, fit, metrics)

    summary = {
        "name": gait["name"],
        "input_gait_json": str(path),
        "normalized_gait_json": str(normalized_path),
        "trajectory_csv": str(trajectory_csv),
        "fitted_curve_csv": str(fitted_curve_csv),
        "fit_plot_png": None if png_path is None else str(png_path),
        "xml": str(xml_path),
        "seconds_requested": args.seconds,
        "fit_start_seconds": args.fit_start_seconds,
        "hit_wall": bool(hit_wall),
        "fit_coordinate_frame": fit_coordinate_frame,
        "fit_kind": fit["kind"],
        "radius_m": fit["radius"],
        "signed_radius_m": signed_radius_m,
        "turn_direction_from_rotated_lateral_drift": direction_from_signed_radius(signed_radius_m),
        "fit_rmse_m": fit["rmse"],
        "arc_deg": fit["arc_deg"],
        **metrics,
        "gait": gait,
    }

    summary_path = out_dir / f"{name}_radius_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    return summary


def main() -> None:
    args = parse_args()
    out_dir = resolve_from_cwd(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gait_paths = collect_gait_paths(args)
    rows = []
    for gait_path in gait_paths:
        row = evaluate_one(gait_path, args, out_dir)
        rows.append(row)
        radius = row["radius_m"]
        signed = row["signed_radius_m"]
        radius_text = "line/inf" if radius is None else f"{radius:.4f} m"
        signed_text = "nan" if signed is None else f"{signed:.4f} m"
        print(
            f"{row['name']}: R={radius_text}, signed_R={signed_text}, "
            f"direction={row['turn_direction_from_rotated_lateral_drift']}, "
            f"arc={row['arc_deg']:.2f} deg, rmse={row['fit_rmse_m']:.4f} m, "
            f"csv={row['trajectory_csv']}"
        )

    combined = {
        "description": "RL gait JSON -> MuJoCo trajectory CSV -> fitted trajectory-circle radius R.",
        "radius_definition": "Same fitted trajectory-circle radius concept as plot_fitted_gait_curves.py. MuJoCo coordinates are meters.",
        "signed_radius_note": "Default sign uses rotated camera-view lateral drift: negative = left, positive = right.",
        "count": len(rows),
        "results": rows,
    }
    combined_path = out_dir / "rl_gait_radius_summary.json"
    combined_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(f"summary={combined_path}")


if __name__ == "__main__":
    main()
