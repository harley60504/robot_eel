from __future__ import annotations

import json
import math
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np


DEFAULT_PX_PER_M = 875.0 / 1.5
DEFAULT_OUT_ROOT = Path("outputs/gui_video_analysis")


class TextLogger:
    def __init__(self, widget: tk.Text):
        self.widget = widget

    def write(self, message: str) -> None:
        self.widget.after(0, self._append, message)

    def _append(self, message: str) -> None:
        self.widget.insert(tk.END, message)
        self.widget.see(tk.END)


def red_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 70, 50], dtype=np.uint8)
    upper1 = np.array([12, 255, 255], dtype=np.uint8)
    lower2 = np.array([168, 70, 50], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)
    mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def dark_mask(frame: np.ndarray) -> np.ndarray:
    blue, green, red = cv2.split(frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dark = hsv[:, :, 2] < 105
    nonblue = (blue.astype(np.int16) - red.astype(np.int16) < 35) & (blue.astype(np.int16) - green.astype(np.int16) < 35)
    mask = ((dark & nonblue).astype(np.uint8)) * 255
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def find_marker(
    frame: np.ndarray,
    marker_mode: str,
    min_area: float,
    max_area: float,
    previous_xy: tuple[float, float] | None,
):
    mask = red_mask(frame) if marker_mode == "red" else dark_mask(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = -1e18

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue

        m = cv2.moments(contour)
        if abs(m["m00"]) < 1e-9:
            continue

        cx = float(m["m10"] / m["m00"])
        cy = float(m["m01"] / m["m00"])
        x, y, w, h = cv2.boundingRect(contour)

        if previous_xy is None:
            score = area
        else:
            score = -np.hypot(cx - previous_xy[0], cy - previous_xy[1]) + 0.02 * area

        if score > best_score:
            best = {
                "x": cx,
                "y": cy,
                "area": area,
                "bbox": [int(x), int(y), int(w), int(h)],
            }
            best_score = score

    return best


def median_smooth(points: list[list[float]], window: int) -> list[list[float]]:
    if len(points) < 3 or window <= 1:
        return points
    if window % 2 == 0:
        window += 1

    half = window // 2
    arr = np.asarray(points, dtype=np.float64)
    out: list[list[float]] = []

    for i in range(len(arr)):
        lo = max(0, i - half)
        hi = min(len(arr), i + half + 1)
        out.append(
            [
                float(arr[i, 0]),
                float(np.median(arr[lo:hi, 1])),
                float(np.median(arr[lo:hi, 2])),
                float(arr[i, 3]),
            ]
        )
    return out


def path_distance_px(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    xy = points[:, 1:3]
    return float(np.sum(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))))


def fit_circle(points: np.ndarray) -> tuple[np.ndarray, dict]:
    xy = points[:, 1:3].astype(np.float64)
    x = xy[:, 0]
    y = xy[:, 1]
    a = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
    b = x * x + y * y
    cx, cy, k = np.linalg.lstsq(a, b, rcond=None)[0]
    radius = float(np.sqrt(max(0.0, k + cx * cx + cy * cy)))
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    rmse = float(np.sqrt(np.mean((dist - radius) ** 2)))
    theta = np.unwrap(np.arctan2(y - cy, x - cx))
    arc_deg = float(abs(np.degrees(theta[-1] - theta[0])))
    angles = np.linspace(float(theta[0]), float(theta[-1]), 280)
    curve = np.column_stack((cx + radius * np.cos(angles), cy + radius * np.sin(angles)))
    return curve, {
        "fit_kind": "circle",
        "center_px": [float(cx), float(cy)],
        "radius_px": radius,
        "arc_deg": arc_deg,
        "rmse_px": rmse,
    }


def fit_line(points: np.ndarray) -> tuple[np.ndarray, dict]:
    xy = points[:, 1:3].astype(np.float64)
    center = xy.mean(axis=0)
    _, _, vh = np.linalg.svd(xy - center, full_matrices=False)
    direction = vh[0]
    if np.dot(direction, xy[-1] - xy[0]) < 0:
        direction = -direction

    projection = (xy - center) @ direction
    scalar = np.linspace(float(projection.min()), float(projection.max()), 280)
    curve = center + np.outer(scalar, direction)
    residual = np.linalg.norm((xy - center) - np.outer(projection, direction), axis=1)
    return curve, {
        "fit_kind": "line",
        "line_start_px": curve[0].tolist(),
        "line_end_px": curve[-1].tolist(),
        "direction_px": direction.tolist(),
        "length_px": float(np.linalg.norm(curve[-1] - curve[0])),
        "rmse_px": float(np.sqrt(np.mean(residual**2))),
    }


def draw_overlay(
    video_path: Path,
    points: np.ndarray,
    curve: np.ndarray,
    mode: str,
    metrics: dict,
    out_path: Path,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target_frame = max(0, min(frame_count - 1, frame_count // 2)) if frame_count > 0 else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    cap.release()

    if not ok:
        frame = np.full((720, 1280, 3), 245, dtype=np.uint8)

    poly = np.round(points[:, 1:3]).astype(np.int32)
    curve_i = np.round(curve).astype(np.int32)

    if mode == "circle":
        cv2.polylines(frame, [curve_i], False, (0, 0, 255), 6, cv2.LINE_AA)
        title = f"R={metrics['radius_px']:.1f}px / {metrics['radius_m']:.3f}m"
    else:
        cv2.polylines(frame, [curve_i], False, (255, 0, 0), 5, cv2.LINE_AA)
        title = f"forward={metrics['forward_distance_m']:.3f}m"

    if len(poly) >= 2:
        cv2.polylines(frame, [poly], False, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, tuple(poly[0]), 10, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(frame, tuple(poly[-1]), 10, (0, 0, 255), -1, cv2.LINE_AA)

    cv2.putText(frame, title, (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(frame, title, (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), frame)


def analyze_video(
    video_path: Path,
    out_root: Path,
    mode: str,
    marker_mode: str,
    px_per_m: float,
    min_area: float,
    max_area: float,
    frame_step: int,
    smooth_window: int,
    logger: TextLogger | None = None,
) -> dict:
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    out_dir = out_root / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    raw_points: list[list[float]] = []
    detections: list[dict] = []
    previous_xy: tuple[float, float] | None = None

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % max(1, frame_step) != 0:
            frame_idx += 1
            continue

        marker = find_marker(frame, marker_mode, min_area, max_area, previous_xy)
        if marker is not None:
            t = float(frame_idx / fps)
            x = float(marker["x"])
            y = float(marker["y"])
            area = float(marker["area"])
            previous_xy = (x, y)
            raw_points.append([t, x, y, area])
            detections.append(
                {
                    "frame": int(frame_idx),
                    "time_s": t,
                    "x": x,
                    "y": y,
                    "area": area,
                    "bbox": marker["bbox"],
                }
            )

        frame_idx += 1

    cap.release()

    if len(raw_points) < 3:
        raise RuntimeError(f"Only detected {len(raw_points)} point(s). Try changing marker mode or area limits.")

    cleaned = median_smooth(raw_points, smooth_window)
    points = np.asarray(cleaned, dtype=np.float64)
    duration = float(points[-1, 0] - points[0, 0]) if points.shape[0] >= 2 else 0.0
    net_dx = float(points[-1, 1] - points[0, 1])
    net_dy = float(points[-1, 2] - points[0, 2])

    base_metrics = {
        "video": str(video_path),
        "video_name": video_path.name,
        "video_stem": video_path.stem,
        "mode": mode,
        "marker_mode": marker_mode,
        "fps": fps,
        "frame_count": frame_count,
        "point_count": int(points.shape[0]),
        "duration_s": duration,
        "px_per_m": px_per_m,
        "net_dx_px": net_dx,
        "net_dy_px": net_dy,
        "path_distance_px": path_distance_px(points),
        "path_distance_m": path_distance_px(points) / px_per_m,
        "straight_distance_px": float(math.hypot(net_dx, net_dy)),
        "straight_distance_m": float(math.hypot(net_dx, net_dy)) / px_per_m,
        "points": raw_points,
        "cleaned_points": cleaned,
        "detections": detections,
    }

    if mode == "circle":
        curve, fit = fit_circle(points)
        metrics = {
            **base_metrics,
            **fit,
            "radius_m": fit["radius_px"] / px_per_m,
            "rmse_m": fit["rmse_px"] / px_per_m,
        }
    else:
        curve, fit = fit_line(points)
        forward_px = abs(float(fit["line_end_px"][1]) - float(fit["line_start_px"][1]))
        forward_m = forward_px / px_per_m
        metrics = {
            **base_metrics,
            **fit,
            "forward_distance_px": forward_px,
            "forward_distance_m": forward_m,
            "forward_speed_m_s": None if duration <= 1e-9 else forward_m / duration,
            "line_length_m": fit["length_px"] / px_per_m,
            "rmse_m": fit["rmse_px"] / px_per_m,
        }

    json_path = out_dir / "analysis_summary_m.json"
    overlay_path = out_dir / "analysis_overlay.png"
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    draw_overlay(video_path, points, curve, mode, metrics, overlay_path)

    if logger is not None:
        logger.write(f"\nOutput JSON: {json_path}\n")
        logger.write(f"Output overlay: {overlay_path}\n")

    return {"json_path": str(json_path), "overlay_path": str(overlay_path), **metrics}


class RealVideoAnalysisApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Robot Eel Real Video Analysis")
        self.root.geometry("820x600")

        self.video_var = tk.StringVar()
        self.out_var = tk.StringVar(value=str(DEFAULT_OUT_ROOT))
        self.mode_var = tk.StringVar(value="circle")
        self.marker_var = tk.StringVar(value="red")
        self.px_per_m_var = tk.StringVar(value=f"{DEFAULT_PX_PER_M:.6f}")
        self.min_area_var = tk.StringVar(value="20")
        self.max_area_var = tk.StringVar(value="20000")
        self.frame_step_var = tk.StringVar(value="1")
        self.smooth_window_var = tk.StringVar(value="5")

        self._build_layout()

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)

        file_row = ttk.Frame(outer)
        file_row.pack(fill=tk.X, pady=4)
        ttk.Label(file_row, text="MP4 file").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Entry(file_row, textvariable=self.video_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(file_row, text="Browse", command=self.browse_video).pack(side=tk.LEFT, padx=(8, 0))

        out_row = ttk.Frame(outer)
        out_row.pack(fill=tk.X, pady=4)
        ttk.Label(out_row, text="Output root").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Entry(out_row, textvariable=self.out_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(out_row, text="Browse", command=self.browse_output).pack(side=tk.LEFT, padx=(8, 0))

        options = ttk.LabelFrame(outer, text="Analysis options", padding=10)
        options.pack(fill=tk.X, pady=10)

        ttk.Label(options, text="Fit mode").grid(row=0, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Combobox(options, textvariable=self.mode_var, values=("circle", "line"), width=16, state="readonly").grid(row=0, column=1, sticky=tk.W, padx=4, pady=4)
        ttk.Label(options, text="circle = turn R, line = straight forward speed").grid(row=0, column=2, sticky=tk.W, padx=4, pady=4)

        ttk.Label(options, text="Marker mode").grid(row=1, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Combobox(options, textvariable=self.marker_var, values=("red", "dark"), width=16, state="readonly").grid(row=1, column=1, sticky=tk.W, padx=4, pady=4)
        ttk.Label(options, text="red = red marker, dark = black/dark body marker").grid(row=1, column=2, sticky=tk.W, padx=4, pady=4)

        numeric_row = ttk.Frame(options)
        numeric_row.grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=4)
        for label, var, width in (
            ("px_per_m", self.px_per_m_var, 12),
            ("min_area", self.min_area_var, 8),
            ("max_area", self.max_area_var, 8),
            ("frame_step", self.frame_step_var, 6),
            ("smooth_window", self.smooth_window_var, 6),
        ):
            ttk.Label(numeric_row, text=label).pack(side=tk.LEFT, padx=(0, 4))
            ttk.Entry(numeric_row, textvariable=var, width=width).pack(side=tk.LEFT, padx=(0, 12))

        button_row = ttk.Frame(outer)
        button_row.pack(fill=tk.X, pady=8)
        self.run_button = ttk.Button(button_row, text="Analyze selected MP4", command=self.start_analysis)
        self.run_button.pack(side=tk.LEFT)
        ttk.Button(button_row, text="Open output folder", command=self.open_output_folder).pack(side=tk.LEFT, padx=8)

        self.log = tk.Text(outer, height=18, wrap=tk.WORD)
        self.log.pack(fill=tk.BOTH, expand=True)
        self.logger = TextLogger(self.log)

        self.logger.write("Select an MP4, choose circle/line, then click Analyze.\n")
        self.logger.write("For turning videos, use mode=circle to get R(px) and R(m).\n")
        self.logger.write("For straight videos, use mode=line to get forward distance/speed.\n")

    def browse_video(self) -> None:
        filename = filedialog.askopenfilename(
            title="Select MP4 video",
            filetypes=(("MP4 files", "*.mp4"), ("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")),
        )
        if filename:
            self.video_var.set(filename)

    def browse_output(self) -> None:
        dirname = filedialog.askdirectory(title="Select output folder")
        if dirname:
            self.out_var.set(dirname)

    def open_output_folder(self) -> None:
        path = Path(self.out_var.get()).resolve()
        path.mkdir(parents=True, exist_ok=True)
        try:
            import os

            os.startfile(path)  # type: ignore[attr-defined]
        except Exception as exc:
            messagebox.showerror("Open folder failed", str(exc))

    def start_analysis(self) -> None:
        self.run_button.configure(state=tk.DISABLED)
        thread = threading.Thread(target=self._analysis_worker, daemon=True)
        thread.start()

    def _analysis_worker(self) -> None:
        try:
            video_path = Path(self.video_var.get()).expanduser().resolve()
            out_root = Path(self.out_var.get()).expanduser().resolve()
            mode = self.mode_var.get()
            marker_mode = self.marker_var.get()
            px_per_m = float(self.px_per_m_var.get())
            min_area = float(self.min_area_var.get())
            max_area = float(self.max_area_var.get())
            frame_step = int(self.frame_step_var.get())
            smooth_window = int(self.smooth_window_var.get())

            self.logger.write("\n=== Analysis started ===\n")
            self.logger.write(f"video={video_path}\n")
            self.logger.write(f"mode={mode}, marker={marker_mode}, px_per_m={px_per_m:.6f}\n")

            result = analyze_video(
                video_path=video_path,
                out_root=out_root,
                mode=mode,
                marker_mode=marker_mode,
                px_per_m=px_per_m,
                min_area=min_area,
                max_area=max_area,
                frame_step=frame_step,
                smooth_window=smooth_window,
                logger=self.logger,
            )

            self.logger.write("\n=== Result ===\n")
            self.logger.write(f"points={result['point_count']} duration={result['duration_s']:.3f}s\n")
            if mode == "circle":
                self.logger.write(f"R={result['radius_px']:.3f}px = {result['radius_m']:.4f}m\n")
                self.logger.write(f"arc={result['arc_deg']:.3f}deg rmse={result['rmse_px']:.3f}px\n")
            else:
                speed = result.get("forward_speed_m_s")
                speed_text = "nan" if speed is None else f"{speed:.4f}m/s"
                self.logger.write(f"forward_distance={result['forward_distance_m']:.4f}m speed={speed_text}\n")
                self.logger.write(f"line_rmse={result['rmse_px']:.3f}px\n")
            self.logger.write("=== Done ===\n")

        except Exception as exc:
            self.logger.write(f"\nERROR: {exc}\n")
            self.root.after(0, lambda: messagebox.showerror("Analysis failed", str(exc)))
        finally:
            self.root.after(0, lambda: self.run_button.configure(state=tk.NORMAL))


def main() -> None:
    root = tk.Tk()
    RealVideoAnalysisApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
