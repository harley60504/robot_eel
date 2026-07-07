from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from analyze_red_dot_videos import fit_turn_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw time-colored real trajectories with line or circle fits."
    )
    parser.add_argument("analysis_root", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--px-per-m", type=float, default=269.2105609870623)
    parser.add_argument(
        "--preferred-summary",
        type=Path,
        action="append",
        default=[],
        help="Explicit per-video summary.json to add and prefer for its gait panel.",
    )
    return parser.parse_args()


def load_summary(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or "fit_kind" not in data:
        return None
    data["_summary_path"] = str(path)
    return data


def load_track(summary: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    csv_path = Path(summary["tracking_csv"])
    if not csv_path.exists():
        csv_path = Path(summary["_summary_path"]).parent / "red_dot_tracking.csv"

    rows = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("detected", "")).lower() != "true":
                continue
            rows.append(
                (
                    float(row["time_s"]),
                    float(row.get("x_smooth_px") or row["x_px"]),
                    float(row.get("y_smooth_px") or row["y_px"]),
                )
            )

    values = np.asarray(rows, dtype=float)
    if summary["fit_kind"] == "circle":
        start = float(summary.get("metric_start_s", values[0, 0]))
        end = float(summary.get("metric_end_s", values[-1, 0]))
    else:
        start = float(summary.get("straight_measurement_start_s", values[0, 0]))
        end = float(summary.get("straight_measurement_end_s", values[-1, 0]))
    values = values[(values[:, 0] >= start) & (values[:, 0] <= end)]

    px_per_m = float(summary.get("px_per_m") or 269.2105609870623)
    t = values[:, 0]
    x = (values[:, 1] - values[0, 1]) / px_per_m
    y = -(values[:, 2] - values[0, 2]) / px_per_m
    return t, x, y


def short_label(summary: dict) -> str:
    direction = str(summary.get("direction", "")).capitalize()
    command_type = summary.get("command_type")
    value = float(summary.get("command_value", 0.0))
    if command_type == "turn_radius":
        return f"{direction} R {value:.1f} m"
    if command_type == "yaw_rate":
        return f"{direction} yaw {value:.1f} rad/s"
    return "Forward" if direction == "Forward" else direction


def group_key(summary: dict) -> tuple[str, str, float]:
    return (
        str(summary.get("direction", "")),
        str(summary.get("command_type", "")),
        round(float(summary.get("command_value", 0.0)), 3),
    )


def draw_panel(ax, summary: dict, add_colorbar: bool = False) -> None:
    t, x, y = load_track(summary)
    fit_kind = summary["fit_kind"]
    lap_metrics = None
    if fit_kind == "circle":
        px_per_m = float(summary.get("px_per_m") or 269.2105609870623)
        metric_rows = [
            {
                "time_s": float(time),
                "x_smooth_px": float(x_m * px_per_m),
                "y_smooth_px": float(-y_m * px_per_m),
            }
            for time, x_m, y_m in zip(t, x, y)
        ]
        lap_metrics = fit_turn_metrics(metric_rows, px_per_m, str(summary["direction"]))
        if lap_metrics["per_lap_metrics"]:
            complete_end_s = float(lap_metrics["per_lap_metrics"][-1]["end_s"])
            keep = t <= complete_end_s + 1e-9
            t, x, y = t[keep], x[keep], y[keep]
    else:
        raw_points = np.column_stack((x, y))
        _, _, vh = np.linalg.svd(raw_points - raw_points.mean(axis=0), full_matrices=False)
        forward = vh[0]
        if np.dot(forward, raw_points[-1] - raw_points[0]) < 0:
            forward = -forward
        lateral = np.array([-forward[1], forward[0]])
        rotated = np.column_stack((raw_points @ forward, raw_points @ lateral))
        rotated -= rotated[0]
        x, y = rotated[:, 0], rotated[:, 1]
    points = np.column_stack((x, y))
    segments = np.stack((points[:-1], points[1:]), axis=1)
    norm = Normalize(float(t[0]), float(t[-1]))
    trace = LineCollection(segments, cmap="turbo", norm=norm, linewidth=2.2, zorder=3)
    trace.set_array((t[:-1] + t[1:]) / 2)
    ax.add_collection(trace)

    if fit_kind == "circle":
        px_per_m = float(summary.get("px_per_m") or 269.2105609870623)
        angle = np.linspace(0, 2 * np.pi, 500)
        laps = lap_metrics["per_lap_metrics"]
        if laps:
            for index, lap in enumerate(laps):
                cx = float(lap["circle_center_x_px"]) / px_per_m
                cy = -float(lap["circle_center_y_px"]) / px_per_m
                radius = float(lap["radius_m"])
                ax.plot(
                    cx + radius * np.cos(angle),
                    cy + radius * np.sin(angle),
                    color="#d62728",
                    linewidth=2.0,
                    alpha=0.72,
                    label="per-lap fit" if index == 0 else None,
                    zorder=2,
                )
        else:
            cx = float(lap_metrics["circle_center_x_px"]) / px_per_m
            cy = -float(lap_metrics["circle_center_y_px"]) / px_per_m
            radius = float(lap_metrics["measured_radius_m"])
            ax.plot(cx + radius * np.cos(angle), cy + radius * np.sin(angle),
                    color="#d62728", linewidth=2.5, label="global fit", zorder=2)
        radius = float(lap_metrics["measured_radius_m"])
        radius_std = float(lap_metrics["measured_radius_std_m"])
        yaw = float(lap_metrics["measured_yaw_rate_abs_rad_s"])
        yaw_std = float(lap_metrics["measured_yaw_rate_std_rad_s"])
        lap_count = int(lap_metrics["complete_lap_count"])
        if summary["command_type"] == "turn_radius":
            metric = (
                f"target R = {float(summary['command_value']):.2f} m; "
                f"lap mean R = {radius:.3f} +/- {radius_std:.3f} m\n"
                f"|yaw| = {yaw:.3f} +/- {yaw_std:.3f} rad/s (n={lap_count})"
            )
        else:
            metric = (
                f"target |yaw| = {float(summary['command_value']):.2f} rad/s; "
                f"lap mean |yaw| = {yaw:.3f} +/- {yaw_std:.3f} rad/s\n"
                f"R = {radius:.3f} +/- {radius_std:.3f} m (n={lap_count})"
            )
    else:
        center = points.mean(axis=0)
        _, _, vh = np.linalg.svd(points - center, full_matrices=False)
        direction = vh[0]
        span = np.dot(points - center, direction)
        fit = np.vstack((center + direction * span.min(), center + direction * span.max()))
        ax.plot(fit[:, 0], fit[:, 1], color="#d62728", linewidth=2.5,
                label="line fit", zorder=2)
        metric = (
            f"target speed = {float(summary['command_value']):.2f} m/s; "
            f"fit speed = {float(summary['straight_speed_m_s']):.3f} m/s"
        )

    ax.scatter(x[0], y[0], s=28, color="#202020", marker="o", zorder=5)
    ax.scatter(x[-1], y[-1], s=38, color="#202020", marker=">", zorder=5)
    ax.set_title(short_label(summary), loc="left", fontsize=12, fontweight="bold", y=1.13, pad=0)
    ax.text(0.0, 1.012, metric, transform=ax.transAxes, fontsize=8.2, va="bottom", linespacing=1.25)
    if fit_kind == "circle":
        ax.set_aspect("equal", adjustable="box")
    else:
        ax.set_aspect("auto")
        y_pad = max(0.12, float(np.ptp(y)) * 0.8)
        ax.set_ylim(float(np.min(y) - y_pad), float(np.max(y) + y_pad))
    ax.margins(0.13)
    ax.grid(True, color="#d9dee3", linewidth=0.7, alpha=0.75)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="best", frameon=False, fontsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if add_colorbar:
        bar = ax.figure.colorbar(trace, ax=ax, fraction=0.035, pad=0.025)
        bar.set_label("time (s)")


def selection_score(summary: dict) -> tuple[int, float]:
    if summary.get("_preferred"):
        return 0, abs(float(summary.get("percent_error", float("inf"))))
    # The 23:14 left-R0.3 run is the user's final repeat measurement.
    preferred = "20260702_231446" in str(summary.get("video_name", ""))
    key = group_key(summary)
    force_priority = 1 if preferred and key == ("left", "turn_radius", 0.3) else 2
    return force_priority, abs(float(summary.get("percent_error", float("inf"))))


def main() -> None:
    args = parse_args()
    root = args.analysis_root.resolve()
    out_dir = (args.out_dir or root / "time_colored_fitted_trajectories").resolve()
    individual_dir = out_dir / "individual"
    individual_dir.mkdir(parents=True, exist_ok=True)

    summaries = [item for path in root.rglob("summary.json") if (item := load_summary(path))]
    for preferred_path in args.preferred_summary:
        preferred = load_summary(preferred_path.resolve())
        if preferred is None:
            raise SystemExit(f"Invalid preferred summary: {preferred_path}")
        preferred["_preferred"] = True
        summaries = [
            item for item in summaries
            if Path(item["_summary_path"]).resolve() != preferred_path.resolve()
        ]
        summaries.append(preferred)
    if not summaries:
        raise SystemExit(f"No per-video summary.json files found under {root}")

    for summary in summaries:
        fig, ax = plt.subplots(figsize=(8.0, 6.2), constrained_layout=True)
        draw_panel(ax, summary, add_colorbar=True)
        fig.savefig(
            individual_dir / f"{Path(summary['video_name']).stem}_time_fit.png",
            dpi=220,
            bbox_inches="tight",
            pad_inches=0.12,
        )
        plt.close(fig)

    groups: dict[tuple[str, str, float], list[dict]] = {}
    for summary in summaries:
        if summary.get("direction") == "backward":
            continue
        groups.setdefault(group_key(summary), []).append(summary)
    selected = [min(items, key=selection_score) for items in groups.values()]
    order = {"forward": 0, "left": 1, "right": 2}
    type_order = {"straight_speed": 0, "turn_radius": 1, "yaw_rate": 2}
    selected.sort(key=lambda s: (type_order.get(s["command_type"], 9), float(s["command_value"]), order.get(s["direction"], 9)))

    fig, axes = plt.subplots(3, 3, figsize=(17, 14), constrained_layout=True)
    for ax, summary in zip(axes.flat, selected):
        draw_panel(ax, summary, add_colorbar=True)
    for ax in axes.flat[len(selected):]:
        ax.axis("off")
    fig.suptitle("Representative real swimming trajectories and per-lap fits", fontsize=18, fontweight="bold")
    fig.savefig(
        out_dir / "selected_9_time_colored_fitted_trajectories.png",
        dpi=240,
        bbox_inches="tight",
        pad_inches=0.14,
    )
    plt.close(fig)

    manifest = [
        {
            "panel": short_label(summary),
            "video_name": summary["video_name"],
            "summary_path": summary["_summary_path"],
        }
        for summary in selected
    ]
    (out_dir / "selected_9_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(summaries)} individual plots and {len(selected)}-panel montage to {out_dir}")


if __name__ == "__main__":
    main()
