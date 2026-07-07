from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_red_dot_videos import fit_turn_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average unique real trials using complete-lap turn metrics.")
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def read_summary(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or "video_name" not in value or "fit_kind" not in value:
        return None
    value["_summary_path"] = str(path.resolve())
    return value


def tracking_path(summary: dict) -> Path:
    path = Path(str(summary.get("tracking_csv", "")))
    if path.exists():
        return path
    return Path(summary["_summary_path"]).parent / "red_dot_tracking.csv"


def turn_metrics(summary: dict) -> dict:
    rows = []
    start = float(summary.get("metric_start_s", 0.0))
    end = float(summary.get("metric_end_s", float("inf")))
    with tracking_path(summary).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            time_s = float(row["time_s"])
            if str(row.get("detected", "")).lower() != "true" or not start <= time_s <= end:
                continue
            rows.append(
                {
                    "time_s": time_s,
                    "x_smooth_px": float(row.get("x_smooth_px") or row["x_px"]),
                    "y_smooth_px": float(row.get("y_smooth_px") or row["y_px"]),
                }
            )
    return fit_turn_metrics(rows, float(summary["px_per_m"]), str(summary["direction"]))


def condition_label(summary: dict) -> str:
    direction = str(summary["direction"])
    if summary["command_type"] == "straight_speed":
        return "forward_v017"
    token = "r" if summary["command_type"] == "turn_radius" else "y"
    return f"{direction}_{token}{int(round(float(summary['command_value']) * 10)):02d}"


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    unique: dict[str, dict] = {}
    for root in args.roots:
        for path in root.resolve().rglob("summary.json"):
            summary = read_summary(path)
            if summary is not None:
                unique.setdefault(str(summary["video_name"]), summary)

    trials = []
    for video_name, summary in sorted(unique.items()):
        command_type = summary.get("command_type")
        if command_type not in {"straight_speed", "turn_radius", "yaw_rate"}:
            continue
        if summary.get("direction") == "backward":
            continue

        row = {
            "condition": condition_label(summary),
            "video_name": video_name,
            "summary_path": summary["_summary_path"],
            "direction": summary["direction"],
            "command_type": command_type,
            "target": float(summary["command_value"]),
            "metric_method": "straight_fit" if command_type == "straight_speed" else "",
        }
        if command_type == "straight_speed":
            row.update(
                {
                    "complete_lap_count": "",
                    "radius_m": "",
                    "yaw_rate_rad_s": "",
                    "speed_m_s": float(summary["straight_speed_m_s"]),
                    "circle_rmse_m": "",
                }
            )
        else:
            metrics = turn_metrics(summary)
            row.update(
                {
                    "metric_method": metrics["turn_metric_method"],
                    "complete_lap_count": int(metrics["complete_lap_count"]),
                    "radius_m": float(metrics["measured_radius_m"]),
                    "yaw_rate_rad_s": float(metrics["measured_yaw_rate_abs_rad_s"]),
                    "speed_m_s": "",
                    "circle_rmse_m": float(metrics["circle_rmse_m"]),
                }
            )
        trials.append(row)

    trial_fields = list(trials[0])
    with (out_dir / "trial_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=trial_fields)
        writer.writeheader()
        writer.writerows(trials)

    grouped = defaultdict(list)
    for row in trials:
        grouped[row["condition"]].append(row)

    aggregate = []
    for condition, rows in sorted(grouped.items()):
        command_type = rows[0]["command_type"]
        item = {
            "condition": condition,
            "direction": rows[0]["direction"],
            "command_type": command_type,
            "target": rows[0]["target"],
            "trial_count": len(rows),
            "total_complete_laps": sum(int(row["complete_lap_count"] or 0) for row in rows),
        }
        if command_type == "straight_speed":
            values = np.array([float(row["speed_m_s"]) for row in rows])
            item.update({"mean": float(values.mean()), "std": float(values.std()), "unit": "m/s"})
        elif command_type == "turn_radius":
            values = np.array([float(row["radius_m"]) for row in rows])
            item.update({"mean": float(values.mean()), "std": float(values.std()), "unit": "m"})
        else:
            values = np.array([float(row["yaw_rate_rad_s"]) for row in rows])
            item.update({"mean": float(values.mean()), "std": float(values.std()), "unit": "rad/s"})
        item["error"] = item["mean"] - float(item["target"])
        item["percent_error"] = 100.0 * item["error"] / float(item["target"])
        aggregate.append(item)

    fields = list(aggregate[0])
    with (out_dir / "condition_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(aggregate)
    (out_dir / "condition_summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(trials)} trials and {len(aggregate)} condition averages to {out_dir}")


if __name__ == "__main__":
    main()
