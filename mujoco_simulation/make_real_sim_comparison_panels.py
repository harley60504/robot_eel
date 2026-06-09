from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path.cwd()
OUT_DIR = ROOT / "outputs" / "real_sim_comparison"
PX_PER_M = 875 / 1.5


ROWS = (
    {
        "key": "straight",
        "title": "Straight swim",
        "real_image": ROOT / "outputs/video_start_to_wall/straight_233739/fit_curve_only_start_to_wall.png",
        "sim_image": ROOT / "outputs/fitted_curve_comparison/sim_straight_fitted_rotated.png",
        "real_id": "straight_233739",
        "sim_id": "straight",
    },
    {
        "key": "turn_left",
        "title": "Left turn",
        "real_image": ROOT / "outputs/video_start_to_wall/turn_left_141203/fit_curve_only_start_to_wall.png",
        "sim_image": ROOT / "outputs/fitted_curve_comparison/sim_turn_left_fitted_rotated.png",
        "real_id": "turn_left_141203",
        "sim_id": "turn_left",
    },
    {
        "key": "spin_left",
        "title": "Spin left",
        "real_image": ROOT / "outputs/video_start_to_wall/spin_left_141254/fit_curve_only_start_to_wall.png",
        "sim_image": ROOT / "outputs/fitted_curve_comparison/sim_spin_left_fitted_rotated.png",
        "real_id": "spin_left_141254",
        "sim_id": "spin_left",
    },
)


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return image


def fit_cover(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    target_w, target_h = size
    h, w = image.shape[:2]
    scale = max(target_w / w, target_h / h)
    resized = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    h2, w2 = resized.shape[:2]
    x0 = max(0, (w2 - target_w) // 2)
    y0 = max(0, (h2 - target_h) // 2)
    return resized[y0 : y0 + target_h, x0 : x0 + target_w]


def put_text_block(image: np.ndarray, lines: list[str], origin: tuple[int, int], scale=0.82):
    x, y = origin
    for line in lines:
        cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), 4, cv2.LINE_AA)
        cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (245, 245, 245), 2, cv2.LINE_AA)
        y += int(31 * scale / 0.82)


def real_summary_text(real_summary: dict, key: str):
    if key == "straight":
        length_px = real_summary["length_px"]
        distance_m = length_px / PX_PER_M
        speed_m_s = distance_m / 13.0
        return [f"Real: {real_summary['point_count']} pts", f"speed {speed_m_s:.3f} m/s", f"line RMSE {real_summary['rmse_px']:.1f}px"]
    radius_m = real_summary["radius_px"] / PX_PER_M
    return [
        f"Real: {real_summary['point_count']} pts",
        f"R {radius_m:.3f}m",
        f"circle RMSE {real_summary['rmse_px']:.1f}px",
    ]


def sim_summary_text(sim_summary: dict, key: str):
    if key == "straight":
        straight = json.loads((ROOT / "outputs/fixed_gait_trajectories_3x1_5/summary.json").read_text(encoding="utf-8"))
        row = next(item for item in straight if item["name"] == "straight")
        return [f"MuJoCo: {row['duration_s']:.2f}s to wall", f"speed {row['speed_m_s']:.3f} m/s", "line fit"]
    return [
        "MuJoCo",
        f"R {sim_summary['radius']:.3f}m",
        f"fit RMSE {sim_summary['rmse']:.3f}m",
    ]


def make_pair(row: dict, real_summary: dict, sim_summary: dict):
    cell_w, cell_h = 560, 860
    header_h = 108
    gutter = 22
    canvas = np.full((header_h + cell_h, cell_w * 2 + gutter, 3), 245, dtype=np.uint8)

    real_source = read_image(row["real_image"])
    if row["key"] == "straight":
        real_source = cv2.rotate(real_source, cv2.ROTATE_180)
    real = fit_cover(real_source, (cell_w, cell_h))
    sim = fit_cover(read_image(row["sim_image"]), (cell_w, cell_h))
    canvas[header_h:, :cell_w] = real
    canvas[header_h:, cell_w + gutter :] = sim

    cv2.putText(canvas, row["title"], (20, 39), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(canvas, "REAL", (20, 83), cv2.FONT_HERSHEY_SIMPLEX, 0.86, (0, 80, 180), 2, cv2.LINE_AA)
    cv2.putText(canvas, "MUJOCO", (cell_w + gutter + 20, 83), cv2.FONT_HERSHEY_SIMPLEX, 0.86, (130, 40, 0), 2, cv2.LINE_AA)
    put_text_block(canvas, real_summary_text(real_summary, row["key"]), (150, 72), scale=0.62)
    put_text_block(canvas, sim_summary_text(sim_summary, row["key"]), (cell_w + gutter + 165, 72), scale=0.62)
    cv2.line(canvas, (cell_w + gutter // 2, header_h), (cell_w + gutter // 2, header_h + cell_h), (230, 230, 230), 2)
    return canvas


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    real_summaries = json.loads((ROOT / "outputs/video_start_to_wall/summary.json").read_text(encoding="utf-8"))
    sim_summaries = json.loads((ROOT / "outputs/fitted_curve_comparison/sim_fitted_summary.json").read_text(encoding="utf-8"))
    real_by_id = {item["clip"]: item for item in real_summaries}
    sim_by_id = {item["name"]: item for item in sim_summaries}

    panels = []
    comparison_summary = []
    for row in ROWS:
        panel = make_pair(row, real_by_id[row["real_id"]], sim_by_id[row["sim_id"]])
        path = OUT_DIR / f"{row['key']}_real_vs_mujoco.png"
        cv2.imwrite(str(path), panel)
        panels.append(panel)
        comparison_summary.append({"key": row["key"], "real": real_by_id[row["real_id"]], "sim": sim_by_id[row["sim_id"]]})

    gap = np.full((24, panels[0].shape[1], 3), 245, dtype=np.uint8)
    combined = np.vstack([panels[0], gap, panels[1], gap, panels[2]])
    cv2.imwrite(str(OUT_DIR / "real_vs_mujoco_all.png"), combined)
    (OUT_DIR / "comparison_summary.json").write_text(json.dumps(comparison_summary, indent=2), encoding="utf-8")
    print(OUT_DIR / "real_vs_mujoco_all.png")
    for row in ROWS:
        print(OUT_DIR / f"{row['key']}_real_vs_mujoco.png")


if __name__ == "__main__":
    main()
