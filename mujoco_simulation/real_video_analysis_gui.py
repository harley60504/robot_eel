from __future__ import annotations

import json
import math
import subprocess
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np

from make_tracked_center_cleaned_physical import (
    DEFAULT_PREVIEW_NAME,
    DEFAULT_PX_PER_M,
    DEFAULT_RECORDINGS_DIR,
    DEFAULT_VIDEO_STEMS,
    process_video,
    resolve_path,
    resolve_recordings_dir,
    resolve_video,
)
from plot_fixed_gait_trajectories import draw_environment, plot_one, run_gait, summarize
from plot_fitted_gait_curves import (
    add_sim_metric_box,
    draw_rotated_tank,
    fitted_curve,
    rotate_sim_xy,
    sim_metric_text,
    trajectory_metrics,
)
from sim_config import DEFAULT_START_X, DEFAULT_START_Y, EEL_MODEL_XML


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PIPELINE_ROOT = Path("../gui_pipeline_outputs")
REAL_SUBDIR = "real_video_analysis"
SIM_SUBDIR = "fixed_gait_trajectories_3x1_5"
FIT_SUBDIR = "fitted_curve_comparison"


class TextLogger:
    def __init__(self, widget: tk.Text):
        self.widget = widget

    def write(self, message: str) -> None:
        self.widget.after(0, self._append, message)

    def _append(self, message: str) -> None:
        self.widget.insert(tk.END, message)
        self.widget.see(tk.END)


class RealVideoAnalysisApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Robot Eel Real + MuJoCo Gait Pipeline")
        self.root.geometry("980x700")

        self.recordings_var = tk.StringVar(value=str(DEFAULT_RECORDINGS_DIR))
        self.video_var = tk.StringVar()
        self.gait_json_var = tk.StringVar()
        self.out_var = tk.StringVar(value=str(DEFAULT_PIPELINE_ROOT))
        self.px_per_m_var = tk.StringVar(value=f"{DEFAULT_PX_PER_M:.6f}")
        self.preview_var = tk.BooleanVar(value=True)

        self._build_layout()

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)

        rec_row = ttk.Frame(outer)
        rec_row.pack(fill=tk.X, pady=4)
        ttk.Label(rec_row, text="Recordings folder").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Entry(rec_row, textvariable=self.recordings_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(rec_row, text="Browse", command=self.browse_recordings).pack(side=tk.LEFT, padx=(8, 0))

        file_row = ttk.Frame(outer)
        file_row.pack(fill=tk.X, pady=4)
        ttk.Label(file_row, text="Single MP4").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Entry(file_row, textvariable=self.video_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(file_row, text="Browse", command=self.browse_video).pack(side=tk.LEFT, padx=(8, 0))

        json_row = ttk.Frame(outer)
        json_row.pack(fill=tk.X, pady=4)
        ttk.Label(json_row, text="Gait JSON").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Entry(json_row, textvariable=self.gait_json_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(json_row, text="Browse", command=self.browse_gait_json).pack(side=tk.LEFT, padx=(8, 0))

        out_row = ttk.Frame(outer)
        out_row.pack(fill=tk.X, pady=4)
        ttk.Label(out_row, text="Pipeline output root").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Entry(out_row, textvariable=self.out_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(out_row, text="Browse", command=self.browse_output).pack(side=tk.LEFT, padx=(8, 0))

        options = ttk.LabelFrame(outer, text="Options", padding=10)
        options.pack(fill=tk.X, pady=10)
        ttk.Label(options, text="px_per_m").grid(row=0, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Entry(options, textvariable=self.px_per_m_var, width=14).grid(row=0, column=1, sticky=tk.W, padx=4, pady=4)
        ttk.Checkbutton(options, text=f"Write real preview PNG ({DEFAULT_PREVIEW_NAME})", variable=self.preview_var).grid(row=0, column=2, sticky=tk.W, padx=12, pady=4)
        ttk.Label(
            options,
            text="Output is separated under the selected root folder: real_video_analysis, fixed_gait_trajectories_3x1_5, fitted_curve_comparison.",
        ).grid(row=1, column=0, columnspan=3, sticky=tk.W, padx=4, pady=4)
        ttk.Label(
            options,
            text="Select MP4 for real tracking, or select Gait JSON to run MuJoCo and generate the same trajectory/fitted-R analysis figures.",
        ).grid(row=2, column=0, columnspan=3, sticky=tk.W, padx=4, pady=4)

        button_row = ttk.Frame(outer)
        button_row.pack(fill=tk.X, pady=8)
        self.single_button = ttk.Button(button_row, text="Analyze selected MP4", command=self.start_single)
        self.single_button.pack(side=tk.LEFT)
        self.json_button = ttk.Button(button_row, text="Analyze selected JSON", command=self.start_json)
        self.json_button.pack(side=tk.LEFT, padx=6)
        self.real3_button = ttk.Button(button_row, text="Analyze real 3 videos", command=self.start_real3)
        self.real3_button.pack(side=tk.LEFT, padx=6)
        self.sim_button = ttk.Button(button_row, text="Run MuJoCo fixed gait", command=self.start_sim)
        self.sim_button.pack(side=tk.LEFT, padx=6)
        self.curve_button = ttk.Button(button_row, text="Plot fitted curves", command=self.start_curves)
        self.curve_button.pack(side=tk.LEFT, padx=6)
        self.full_button = ttk.Button(button_row, text="Run full pipeline", command=self.start_full)
        self.full_button.pack(side=tk.LEFT, padx=6)
        ttk.Button(button_row, text="Open output root", command=self.open_output_folder).pack(side=tk.LEFT, padx=6)

        self.log = tk.Text(outer, height=24, wrap=tk.WORD)
        self.log.pack(fill=tk.BOTH, expand=True)
        self.logger = TextLogger(self.log)
        self.logger.write("Robot eel gait pipeline GUI ready.\n")
        self.logger.write("Default 3 real videos:\n")
        for stem in DEFAULT_VIDEO_STEMS:
            self.logger.write(f"  - {stem}.mp4\n")
        self.logger.write("\nPipeline folders under output root:\n")
        self.logger.write(f"  {REAL_SUBDIR}/\n")
        self.logger.write(f"  {SIM_SUBDIR}/\n")
        self.logger.write(f"  {FIT_SUBDIR}/\n")
        self.logger.write("\nUse Gait JSON + Analyze selected JSON to run a new RL gait through MuJoCo and fitted-R analysis.\n")

    def browse_recordings(self) -> None:
        dirname = filedialog.askdirectory(title="Select recordings folder")
        if dirname:
            self.recordings_var.set(dirname)

    def browse_video(self) -> None:
        filename = filedialog.askopenfilename(
            title="Select MP4 video",
            filetypes=(("MP4 files", "*.mp4"), ("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")),
        )
        if filename:
            self.video_var.set(filename)

    def browse_gait_json(self) -> None:
        filename = filedialog.askopenfilename(
            title="Select gait JSON",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
        )
        if filename:
            self.gait_json_var.set(filename)

    def browse_output(self) -> None:
        dirname = filedialog.askdirectory(title="Select pipeline output root")
        if dirname:
            self.out_var.set(dirname)

    def output_root(self) -> Path:
        return resolve_path(Path(self.out_var.get()).expanduser())

    def real_out_dir(self) -> Path:
        return self.output_root() / REAL_SUBDIR

    def sim_out_dir(self) -> Path:
        return self.output_root() / SIM_SUBDIR

    def fit_out_dir(self) -> Path:
        return self.output_root() / FIT_SUBDIR

    def recordings_dir(self) -> Path:
        return resolve_recordings_dir(Path(self.recordings_var.get()).expanduser())

    def open_output_folder(self) -> None:
        path = self.output_root()
        path.mkdir(parents=True, exist_ok=True)
        try:
            import os

            os.startfile(path)  # type: ignore[attr-defined]
        except Exception as exc:
            messagebox.showerror("Open folder failed", str(exc))

    def set_buttons(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        for button in (self.single_button, self.json_button, self.real3_button, self.sim_button, self.curve_button, self.full_button):
            button.configure(state=state)

    def start_single(self) -> None:
        video = self.video_var.get().strip()
        if not video:
            messagebox.showerror("Missing MP4", "Please select one MP4 file first.")
            return
        self._start_thread(self._run_real_videos, [Path(video)])

    def start_json(self) -> None:
        gait_json = self.gait_json_var.get().strip()
        if not gait_json:
            messagebox.showerror("Missing JSON", "Please select one gait JSON file first.")
            return
        self._start_thread(self._run_json_gait, Path(gait_json))

    def start_real3(self) -> None:
        recordings_dir = self.recordings_dir()
        videos = [resolve_video(f"{stem}.mp4", recordings_dir) for stem in DEFAULT_VIDEO_STEMS]
        self._start_thread(self._run_real_videos, videos)

    def start_sim(self) -> None:
        self._start_thread(self._run_simulation)

    def start_curves(self) -> None:
        self._start_thread(self._run_fitted_curves)

    def start_full(self) -> None:
        recordings_dir = self.recordings_dir()
        videos = [resolve_video(f"{stem}.mp4", recordings_dir) for stem in DEFAULT_VIDEO_STEMS]
        self._start_thread(self._run_full_pipeline, videos)

    def _start_thread(self, func, *args) -> None:
        self.set_buttons(False)
        thread = threading.Thread(target=self._safe_worker, args=(func, *args), daemon=True)
        thread.start()

    def _safe_worker(self, func, *args) -> None:
        try:
            func(*args)
            self.logger.write("\n=== Done ===\n")
        except Exception as exc:
            self.logger.write(f"\nERROR: {exc}\n")
            self.root.after(0, lambda: messagebox.showerror("Analysis failed", str(exc)))
        finally:
            self.root.after(0, lambda: self.set_buttons(True))

    @staticmethod
    def _safe_name(value: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in value.strip())
        return safe.strip("._") or "selected_gait"

    @staticmethod
    def _json_ready(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _write_json_gait_trajectory_plot(self, png_path: Path, name: str, arr: np.ndarray, summary: dict) -> None:
        fig, ax = plt.subplots(figsize=(7, 5), dpi=170)
        draw_environment(ax, DEFAULT_START_X, DEFAULT_START_Y)
        plot_one(ax, name, arr, summary)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"{name} trajectory from selected JSON")
        radius = summary.get("turn_radius_m")
        radius_text = "inf" if radius is None or not math.isfinite(float(radius)) else f"{float(radius):.3f}"
        ax.text(
            0.02,
            0.98,
            f"time={arr[-1, 0]:.2f}s\n"
            f"dx={summary['dx']:.3f} m, dy={summary['dy']:.3f} m\n"
            f"yaw={summary['yaw_change_deg']:.1f} deg, rate={summary['yaw_rate_rad_s']:.3f} rad/s\n"
            f"radius={radius_text} m",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
            fontsize=8,
        )
        fig.tight_layout()
        fig.savefig(png_path)
        plt.close(fig)

    def _write_json_gait_fitted_plot(self, png_path: Path, name: str, arr: np.ndarray) -> dict:
        xy = rotate_sim_xy(arr[:, 1:3])
        curve, fit = fitted_curve(xy)
        metrics = trajectory_metrics(arr, xy)

        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        color = colors[0] if colors else None
        fig, ax = plt.subplots(figsize=(4.8, 8.0), dpi=170)
        draw_rotated_tank(ax)
        ax.plot(curve[:, 0], curve[:, 1], color=color, linewidth=3.0)
        ax.scatter([xy[0, 0]], [xy[0, 1]], s=34, color=color, edgecolor="black", zorder=4)
        ax.scatter([xy[-1, 0]], [xy[-1, 1]], s=52, marker="x", color=color, linewidth=2.2, zorder=4)
        ax.set_title(f"{name} fitted curve")
        add_sim_metric_box(ax, sim_metric_text(name, fit, metrics))
        fig.tight_layout()
        fig.savefig(png_path)
        plt.close(fig)

        return {"name": name, **fit, **metrics}

    def _run_json_gait(self, gait_path: Path) -> None:
        gait_path = gait_path.expanduser().resolve()
        if not gait_path.exists():
            raise FileNotFoundError(f"Gait JSON not found: {gait_path}")

        sim_out = self.sim_out_dir()
        fit_out = self.fit_out_dir()
        sim_out.mkdir(parents=True, exist_ok=True)
        fit_out.mkdir(parents=True, exist_ok=True)

        self.logger.write("\n=== MuJoCo selected JSON gait ===\n")
        self.logger.write(f"gait_json={gait_path}\n")
        self.logger.write(f"sim_out={sim_out}\n")
        self.logger.write(f"fit_out={fit_out}\n")

        gait, arr, hit_wall = run_gait(Path(EEL_MODEL_XML), gait_path, seconds=30.0, start_x=DEFAULT_START_X, start_y=DEFAULT_START_Y)
        if arr.size == 0:
            raise RuntimeError(f"No MuJoCo trajectory was produced for {gait_path}")

        name = self._safe_name(str(gait.get("name", gait_path.stem)))
        csv_path = sim_out / f"{name}_trajectory.csv"
        np.savetxt(csv_path, arr, delimiter=",", header="time,x,y,yaw", comments="")

        summary = summarize(arr, warmup_seconds=0.0)
        trajectory_png = sim_out / f"{name}_trajectory.png"
        self._write_json_gait_trajectory_plot(trajectory_png, name, arr, summary)

        fixed_summary = {
            "name": name,
            "source_gait_json": str(gait_path),
            "trajectory_csv": str(csv_path),
            "trajectory_png": str(trajectory_png),
            "duration_s": float(arr[-1, 0]),
            "hit_wall": bool(hit_wall),
            **{key: self._json_ready(value) for key, value in summary.items() if key != "warmup_index"},
        }
        fixed_summary_path = sim_out / f"{name}_summary.json"
        fixed_summary_path.write_text(json.dumps(fixed_summary, indent=2), encoding="utf-8")

        fitted_png = fit_out / f"sim_{name}_fitted_rotated.png"
        fitted_summary = self._write_json_gait_fitted_plot(fitted_png, name, arr)
        fitted_summary.update(
            {
                "source_gait_json": str(gait_path),
                "trajectory_csv": str(csv_path),
                "trajectory_png": str(trajectory_png),
                "fit_png": str(fitted_png),
                "hit_wall": bool(hit_wall),
            }
        )
        fitted_summary = {key: self._json_ready(value) for key, value in fitted_summary.items()}
        fitted_summary_path = fit_out / f"{name}_fitted_summary.json"
        fitted_summary_path.write_text(json.dumps(fitted_summary, indent=2), encoding="utf-8")

        radius = fitted_summary.get("radius")
        radius_text = "line/inf" if radius is None else f"{float(radius):.4f}m"
        self.logger.write(f"  trajectory CSV: {csv_path}\n")
        self.logger.write(f"  trajectory plot: {trajectory_png}\n")
        self.logger.write(f"  fitted plot: {fitted_png}\n")
        self.logger.write(f"  fitted summary: {fitted_summary_path}\n")
        self.logger.write(
            f"  R={radius_text}, arc={float(fitted_summary.get('arc_deg') or 0.0):.3f}deg, "
            f"rmse={float(fitted_summary.get('rmse') or 0.0):.4f}m\n"
        )

    def _run_real_videos(self, videos: list[Path]) -> None:
        out_root = self.real_out_dir()
        out_root.mkdir(parents=True, exist_ok=True)
        px_per_m = float(self.px_per_m_var.get())
        write_preview = bool(self.preview_var.get())

        self.logger.write("\n=== Real video tracking ===\n")
        self.logger.write(f"out_root={out_root}\n")
        self.logger.write(f"px_per_m={px_per_m:.6f}\n")

        for video_path in videos:
            video_path = video_path.expanduser().resolve()
            self.logger.write(f"\nVideo: {video_path}\n")
            if not video_path.exists():
                self.logger.write("  SKIP: file not found\n")
                continue
            process_video(video_path=video_path, out_root=out_root, write_preview=write_preview, px_per_m=px_per_m)
            summary_path = out_root / video_path.stem / "tracked_center_summary_cleaned_physical.json"
            result = json.loads(summary_path.read_text(encoding="utf-8"))
            self._log_real_result(result, summary_path)

    def _run_simulation(self) -> None:
        sim_out = self.sim_out_dir()
        sim_out.mkdir(parents=True, exist_ok=True)
        self.logger.write("\n=== MuJoCo fixed gait trajectories ===\n")
        self._run_command([sys.executable, "plot_fixed_gait_trajectories.py", "--out-dir", str(sim_out)])

    def _run_fitted_curves(self) -> None:
        real_out = self.real_out_dir()
        sim_out = self.sim_out_dir()
        fit_out = self.fit_out_dir()
        fit_out.mkdir(parents=True, exist_ok=True)
        self.logger.write("\n=== Fitted gait curves / real-vs-sim comparison ===\n")
        self._run_command(
            [
                sys.executable,
                "plot_fitted_gait_curves.py",
                "--sim-dir",
                str(sim_out),
                "--video-analysis-dir",
                str(real_out),
                "--recordings-dir",
                str(self.recordings_dir()),
                "--out-dir",
                str(fit_out),
            ]
        )

    def _run_full_pipeline(self, videos: list[Path]) -> None:
        self.logger.write("\n=== Full pipeline ===\n")
        self._run_real_videos(videos)
        self._run_simulation()
        self._run_fitted_curves()

    def _run_command(self, cmd: list[str]) -> None:
        self.logger.write("CMD: " + " ".join(cmd) + "\n")
        proc = subprocess.run(cmd, cwd=SCRIPT_DIR, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if proc.stdout:
            self.logger.write(proc.stdout)
        if proc.returncode != 0:
            raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")

    def _log_real_result(self, result: dict, summary_path: Path) -> None:
        self.logger.write(f"  points={result.get('point_count')} fit={result.get('fit_kind')}\n")
        if result.get("fit_kind") == "circle":
            self.logger.write(
                f"  R={result.get('radius_px'):.3f}px = {result.get('radius_m'):.4f}m, "
                f"arc={result.get('arc_deg'):.3f}deg, rmse={result.get('rmse_px'):.3f}px\n"
            )
        else:
            speed = result.get("forward_speed_m_s")
            speed_text = "nan" if speed is None else f"{speed:.4f}m/s"
            self.logger.write(
                f"  forward={result.get('forward_distance_m'):.4f}m, speed={speed_text}, line_rmse={result.get('rmse_px'):.3f}px\n"
            )
            self.logger.write("  speed source: fitted-line vertical forward displacement, not left-right drift.\n")
        self.logger.write(f"  JSON: {summary_path}\n")
        preview = result.get("preview_image")
        if preview:
            self.logger.write(f"  Preview: {preview}\n")


def main() -> None:
    root = tk.Tk()
    RealVideoAnalysisApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
