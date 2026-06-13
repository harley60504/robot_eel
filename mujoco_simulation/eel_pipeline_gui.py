from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np

from hopf_cpg import DEFAULT_AJOINT_DEG, degrees_to_radians
from make_tracked_center_cleaned_physical import DEFAULT_PREVIEW_NAME, DEFAULT_PX_PER_M, process_video
from plot_fixed_gait_trajectories import draw_environment, plot_one, run_gait, summarize
from plot_fitted_gait_curves import (
    add_sim_metric_box,
    draw_rotated_tank,
    fitted_curve,
    rotate_sim_xy,
    sim_metric_text,
    trajectory_metrics,
)
from rl_policy_exporter import export_turning_policy_to_gait, write_gait_json
from rl_turning_env import TurningConfig
from sim_config import DEFAULT_START_X, DEFAULT_START_Y, EEL_MODEL_XML


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PIPELINE_ROOT = SCRIPT_DIR.parent / "gui_pipeline_outputs"
REAL_SUBDIR = "real_video_analysis"
SIM_SUBDIR = "fixed_gait_trajectories_3x1_5"
FIT_SUBDIR = "fitted_curve_comparison"
RL_GAIT_DIR = SCRIPT_DIR / "outputs" / "rl_gaits"


class TextLogger:
    def __init__(self, widget: tk.Text):
        self.widget = widget

    def write(self, message: str) -> None:
        self.widget.after(0, self._append, message)

    def _append(self, message: str) -> None:
        self.widget.configure(state=tk.NORMAL)
        self.widget.insert(tk.END, message)
        self.widget.see(tk.END)
        self.widget.configure(state=tk.DISABLED)


def resolve_gui_path(path: Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else (SCRIPT_DIR / path).resolve()


def safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in value.strip())
    return safe.strip("._") or "robot_eel_output"


class EelPipelineGui:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Robot Eel RL / MuJoCo / Real Video Pipeline")
        self.root.geometry("1120x780")
        self.root.minsize(980, 680)

        self.viewer_process: subprocess.Popen | None = None
        self.viewer_thread: threading.Thread | None = None
        self.worker_running = False

        self.out_var = tk.StringVar(value=str(DEFAULT_PIPELINE_ROOT))

        self.rl_model_var = tk.StringVar()
        self.rl_output_json_var = tk.StringVar(value=str(RL_GAIT_DIR / "rl_turn_right_preview.json"))
        self.rl_turn_direction_var = tk.StringVar(value="right")
        self.rl_target_yaw_rate_var = tk.StringVar(value="0.45")
        self.rl_target_radius_var = tk.StringVar(value="")
        self.rl_strategy_var = tk.StringVar(value="best-step")
        self.rl_samples_var = tk.StringVar(value="300")
        self.rl_max_episodes_var = tk.StringVar(value="20")
        self.rl_train_timesteps_var = tk.StringVar(value="200000")
        self.rl_train_output_var = tk.StringVar(value=str(SCRIPT_DIR / "outputs" / "ppo_turn_right_gui"))
        self.rl_load_model_var = tk.StringVar(value="")
        self.rl_eval_freq_var = tk.StringVar(value="10000")
        self.rl_freq_var = tk.StringVar(value="")
        self.rl_wavelength_var = tk.StringVar(value="")
        self.rl_ajoint_var = tk.StringVar(value="")
        self.rl_bias_low_var = tk.StringVar(value="")
        self.rl_bias_high_var = tk.StringVar(value="")

        self.gait_json_var = tk.StringVar()
        self.video_var = tk.StringVar()
        self.px_per_m_var = tk.StringVar(value=f"{DEFAULT_PX_PER_M:.6f}")
        self.preview_var = tk.BooleanVar(value=True)
        self.near_wall_note_var = tk.StringVar(value="Real MP4 near-wall plot currently uses the processed/cleaned tracked segment from the video tracker.")

        self._build_layout()
        self.logger.write("Robot eel unified pipeline GUI ready.\n")
        self.logger.write("Use the RL tab to export PPO zip to gait JSON, view it, and plot the trajectory until wall contact.\n")
        self.logger.write("Use the Real MP4 tab to process videos and the Compare tab for fitted real-vs-sim curves.\n")

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)

        output_row = ttk.Frame(outer)
        output_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(output_row, text="Pipeline output root").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Entry(output_row, textvariable=self.out_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(output_row, text="Browse", command=self.browse_output).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(output_row, text="Open", command=self.open_output_folder).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(output_row, text="Stop Viewer", command=self.stop_viewer).pack(side=tk.LEFT, padx=(6, 0))

        self.notebook = ttk.Notebook(outer)
        self.notebook.pack(fill=tk.BOTH, expand=False)
        self._build_train_tab()
        self._build_export_tab()
        self._build_sim_tab()
        self._build_real_tab()
        self._build_compare_tab()

        log_frame = ttk.LabelFrame(outer, text="Log", padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.log = tk.Text(log_frame, height=18, wrap=tk.WORD, font=("Consolas", 10))
        self.log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log.configure(yscrollcommand=scrollbar.set, state=tk.DISABLED)
        self.logger = TextLogger(self.log)

    def _build_turning_target_frame(self, parent: ttk.Frame) -> None:
        options = ttk.LabelFrame(parent, text="Turning target", padding=10)
        options.pack(fill=tk.X, pady=(8, 4))
        ttk.Label(options, text="Direction").grid(row=0, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Combobox(options, textvariable=self.rl_turn_direction_var, values=("left", "right"), state="readonly", width=8).grid(row=0, column=1, sticky=tk.W, padx=4, pady=4)
        ttk.Label(options, text="Target yaw rate |rad/s|").grid(row=0, column=2, sticky=tk.W, padx=12, pady=4)
        ttk.Entry(options, textvariable=self.rl_target_yaw_rate_var, width=10).grid(row=0, column=3, sticky=tk.W, padx=4, pady=4)
        ttk.Label(options, text="Target radius m (blank = none)").grid(row=0, column=4, sticky=tk.W, padx=12, pady=4)
        ttk.Entry(options, textvariable=self.rl_target_radius_var, width=10).grid(row=0, column=5, sticky=tk.W, padx=4, pady=4)

    def _build_export_options_frame(self, parent: ttk.Frame) -> None:
        export = ttk.LabelFrame(parent, text="Export options", padding=10)
        export.pack(fill=tk.X, pady=(8, 4))
        ttk.Label(export, text="Output gait JSON").grid(row=0, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Entry(export, textvariable=self.rl_output_json_var).grid(row=0, column=1, columnspan=5, sticky=tk.EW, padx=4, pady=4)
        ttk.Button(export, text="Browse", command=self.browse_rl_output).grid(row=0, column=6, sticky=tk.W, padx=4, pady=4)
        ttk.Button(export, text="Auto name", command=self.auto_rl_output_name).grid(row=0, column=7, sticky=tk.W, padx=4, pady=4)
        ttk.Label(export, text="Strategy").grid(row=1, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Combobox(export, textvariable=self.rl_strategy_var, values=("best-step", "mean", "last"), state="readonly", width=10).grid(row=1, column=1, sticky=tk.W, padx=4, pady=4)
        ttk.Label(export, text="Samples").grid(row=1, column=2, sticky=tk.W, padx=12, pady=4)
        ttk.Entry(export, textvariable=self.rl_samples_var, width=10).grid(row=1, column=3, sticky=tk.W, padx=4, pady=4)
        ttk.Label(export, text="Max episodes").grid(row=1, column=4, sticky=tk.W, padx=12, pady=4)
        ttk.Entry(export, textvariable=self.rl_max_episodes_var, width=10).grid(row=1, column=5, sticky=tk.W, padx=4, pady=4)
        export.columnconfigure(5, weight=1)

    def _build_env_override_frame(self, parent: ttk.Frame) -> None:
        advanced = ttk.LabelFrame(parent, text="Optional env overrides; leave blank to match TurningConfig defaults", padding=10)
        advanced.pack(fill=tk.X, pady=(8, 4))
        ttk.Label(advanced, text="freq").grid(row=0, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Entry(advanced, textvariable=self.rl_freq_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=4, pady=4)
        ttk.Label(advanced, text="wavelength").grid(row=0, column=2, sticky=tk.W, padx=12, pady=4)
        ttk.Entry(advanced, textvariable=self.rl_wavelength_var, width=10).grid(row=0, column=3, sticky=tk.W, padx=4, pady=4)
        ttk.Label(advanced, text="ajoint deg").grid(row=0, column=4, sticky=tk.W, padx=12, pady=4)
        ttk.Entry(advanced, textvariable=self.rl_ajoint_var, width=10).grid(row=0, column=5, sticky=tk.W, padx=4, pady=4)
        ttk.Label(advanced, text="bias low").grid(row=1, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Entry(advanced, textvariable=self.rl_bias_low_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=4, pady=4)
        ttk.Label(advanced, text="bias high").grid(row=1, column=2, sticky=tk.W, padx=12, pady=4)
        ttk.Entry(advanced, textvariable=self.rl_bias_high_var, width=10).grid(row=1, column=3, sticky=tk.W, padx=4, pady=4)

    def _build_train_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="Train RL Turning")

        self._build_turning_target_frame(tab)

        train = ttk.LabelFrame(tab, text="Train PPO directly from GUI", padding=10)
        train.pack(fill=tk.X, pady=(8, 4))
        ttk.Label(train, text="Timesteps").grid(row=0, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Entry(train, textvariable=self.rl_train_timesteps_var, width=12).grid(row=0, column=1, sticky=tk.W, padx=4, pady=4)
        ttk.Label(train, text="Eval freq").grid(row=0, column=2, sticky=tk.W, padx=12, pady=4)
        ttk.Entry(train, textvariable=self.rl_eval_freq_var, width=12).grid(row=0, column=3, sticky=tk.W, padx=4, pady=4)
        ttk.Label(train, text="Output model base").grid(row=1, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Entry(train, textvariable=self.rl_train_output_var).grid(row=1, column=1, columnspan=4, sticky=tk.EW, padx=4, pady=4)
        ttk.Button(train, text="Browse", command=self.browse_rl_train_output).grid(row=1, column=5, sticky=tk.W, padx=4, pady=4)
        ttk.Label(train, text="Load model zip (optional)").grid(row=2, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Entry(train, textvariable=self.rl_load_model_var).grid(row=2, column=1, columnspan=4, sticky=tk.EW, padx=4, pady=4)
        ttk.Button(train, text="Browse", command=self.browse_rl_load_model).grid(row=2, column=5, sticky=tk.W, padx=4, pady=4)
        train.columnconfigure(4, weight=1)

        self._build_env_override_frame(tab)
        self._build_export_options_frame(tab)

        buttons = ttk.Frame(tab)
        buttons.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(buttons, text="Train RL", command=lambda: self.start_rl_train(export=False, view=False, plot=False)).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Train + Export + View", command=lambda: self.start_rl_train(export=True, view=True, plot=False)).pack(side=tk.LEFT, padx=6)
        ttk.Button(buttons, text="Train + Export + Plot", command=lambda: self.start_rl_train(export=True, view=False, plot=True)).pack(side=tk.LEFT, padx=6)
        ttk.Button(buttons, text="Train + Export + View + Plot", command=lambda: self.start_rl_train(export=True, view=True, plot=True)).pack(side=tk.LEFT, padx=6)

        ttk.Label(
            tab,
            text="This tab creates a new PPO .zip first. The optional export fields are only used by the Train + Export buttons.",
            foreground="#555555",
        ).pack(anchor=tk.W, pady=(8, 0))

    def _build_export_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="Export PPO Model")

        row = ttk.Frame(tab)
        row.pack(fill=tk.X, pady=4)
        ttk.Label(row, text="PPO model zip", width=16).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.rl_model_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text="Browse", command=self.browse_rl_model).pack(side=tk.LEFT, padx=(8, 0))

        self._build_turning_target_frame(tab)
        self._build_env_override_frame(tab)
        self._build_export_options_frame(tab)

        export_buttons = ttk.Frame(tab)
        export_buttons.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(export_buttons, text="Export JSON", command=lambda: self.start_rl_export(view=False, plot=False)).pack(side=tk.LEFT)
        ttk.Button(export_buttons, text="Export + View", command=lambda: self.start_rl_export(view=True, plot=False)).pack(side=tk.LEFT, padx=6)
        ttk.Button(export_buttons, text="Export + Plot Trajectory", command=lambda: self.start_rl_export(view=False, plot=True)).pack(side=tk.LEFT, padx=6)
        ttk.Button(export_buttons, text="Export + View + Plot", command=lambda: self.start_rl_export(view=True, plot=True)).pack(side=tk.LEFT, padx=6)

        note = (
            "Use this tab when a PPO .zip already exists. Bias-only policies export joint_bias from the PPO action; "
            "amp_scales and phase_lags come from the fixed TurningConfig wave."
        )
        ttk.Label(tab, text=note, foreground="#555555").pack(anchor=tk.W, pady=(8, 0))

    def _build_sim_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="MuJoCo JSON Gait")

        row = ttk.Frame(tab)
        row.pack(fill=tk.X, pady=4)
        ttk.Label(row, text="Gait JSON file(s)", width=16).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.gait_json_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text="Browse", command=self.browse_gait_jsons).pack(side=tk.LEFT, padx=(8, 0))

        buttons = ttk.Frame(tab)
        buttons.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(buttons, text="View first JSON", command=self.view_first_json).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Plot selected JSON gait(s)", command=self.start_jsons).pack(side=tk.LEFT, padx=6)
        ttk.Button(buttons, text="View + Plot first JSON", command=self.view_and_plot_first_json).pack(side=tk.LEFT, padx=6)

        ttk.Label(
            tab,
            text="Simulation plots stop at wall contact and save CSV / PNG / summary JSON under the pipeline output root.",
            foreground="#555555",
        ).pack(anchor=tk.W, pady=(8, 0))

    def _build_real_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="Real MP4 Tracking")

        row = ttk.Frame(tab)
        row.pack(fill=tk.X, pady=4)
        ttk.Label(row, text="MP4 file(s)", width=16).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.video_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text="Browse", command=self.browse_videos).pack(side=tk.LEFT, padx=(8, 0))

        options = ttk.LabelFrame(tab, text="Tracking options", padding=10)
        options.pack(fill=tk.X, pady=(8, 4))
        ttk.Label(options, text="px_per_m").grid(row=0, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Entry(options, textvariable=self.px_per_m_var, width=14).grid(row=0, column=1, sticky=tk.W, padx=4, pady=4)
        ttk.Checkbutton(options, text=f"Write preview PNG ({DEFAULT_PREVIEW_NAME})", variable=self.preview_var).grid(row=0, column=2, sticky=tk.W, padx=12, pady=4)
        ttk.Label(options, textvariable=self.near_wall_note_var, foreground="#555555").grid(row=1, column=0, columnspan=3, sticky=tk.W, padx=4, pady=4)

        buttons = ttk.Frame(tab)
        buttons.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(buttons, text="Analyze selected MP4(s)", command=self.start_mp4s).pack(side=tk.LEFT)

    def _build_compare_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="Compare / Outputs")

        ttk.Label(
            tab,
            text="Run this after real MP4 tracking and/or MuJoCo JSON trajectory plotting. It reads current output folders and writes fitted curve comparisons.",
            foreground="#555555",
        ).pack(anchor=tk.W, pady=(0, 8))

        buttons = ttk.Frame(tab)
        buttons.pack(fill=tk.X, pady=4)
        ttk.Button(buttons, text="Run selected MP4(s) + JSON gait(s)", command=self.start_selected_pipeline).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Plot fitted curves from current outputs", command=self.start_curves).pack(side=tk.LEFT, padx=6)
        ttk.Button(buttons, text="Open output root", command=self.open_output_folder).pack(side=tk.LEFT, padx=6)

        folders = (
            f"Output folders:\n"
            f"  {REAL_SUBDIR}/\n"
            f"  {SIM_SUBDIR}/\n"
            f"  {FIT_SUBDIR}/\n"
            f"  outputs/rl_gaits/ for exported PPO gait JSONs"
        )
        ttk.Label(tab, text=folders).pack(anchor=tk.W, pady=(10, 0))

    @staticmethod
    def _join_paths(paths: tuple[str, ...] | list[str]) -> str:
        return "; ".join(str(path) for path in paths)

    @staticmethod
    def _paths_from_var(value: str) -> list[Path]:
        return [Path(part.strip()).expanduser() for part in value.split(";") if part.strip()]

    def _parse_optional_float(self, text: str, label: str) -> float | None:
        text = text.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError as exc:
            raise ValueError(f"{label} must be a number or blank") from exc

    def browse_output(self) -> None:
        dirname = filedialog.askdirectory(title="Select pipeline output root")
        if dirname:
            self.out_var.set(dirname)

    def browse_rl_model(self) -> None:
        filename = filedialog.askopenfilename(
            title="Select PPO model zip",
            filetypes=(("PPO zip", "*.zip"), ("All files", "*.*")),
        )
        if filename:
            self.rl_model_var.set(filename)
            if not self.rl_output_json_var.get().strip() or self.rl_output_json_var.get().endswith("rl_turn_right_preview.json"):
                self.auto_rl_output_name()

    def browse_rl_output(self) -> None:
        filename = filedialog.asksaveasfilename(
            title="Save exported gait JSON",
            initialdir=str(RL_GAIT_DIR),
            defaultextension=".json",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
        )
        if filename:
            self.rl_output_json_var.set(filename)

    def browse_rl_train_output(self) -> None:
        filename = filedialog.asksaveasfilename(
            title="Select PPO training output base",
            initialdir=str(SCRIPT_DIR / "outputs"),
            filetypes=(("Stable-Baselines model base", "*"), ("Zip files", "*.zip"), ("All files", "*.*")),
        )
        if filename:
            # train_turning_rl.py accepts the base path; Stable-Baselines writes .zip.
            if filename.lower().endswith(".zip"):
                filename = filename[:-4]
            self.rl_train_output_var.set(filename)

    def browse_rl_load_model(self) -> None:
        filename = filedialog.askopenfilename(
            title="Select PPO model zip to continue training",
            initialdir=str(SCRIPT_DIR / "outputs"),
            filetypes=(("PPO zip", "*.zip"), ("All files", "*.*")),
        )
        if filename:
            self.rl_load_model_var.set(filename)

    def browse_gait_jsons(self) -> None:
        filenames = filedialog.askopenfilenames(
            title="Select gait JSON file(s)",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
        )
        if filenames:
            self.gait_json_var.set(self._join_paths(list(filenames)))

    def browse_videos(self) -> None:
        filenames = filedialog.askopenfilenames(
            title="Select MP4 video file(s)",
            filetypes=(("MP4 files", "*.mp4"), ("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")),
        )
        if filenames:
            self.video_var.set(self._join_paths(list(filenames)))

    def output_root(self) -> Path:
        return resolve_gui_path(Path(self.out_var.get()))

    def real_out_dir(self) -> Path:
        return self.output_root() / REAL_SUBDIR

    def sim_out_dir(self) -> Path:
        return self.output_root() / SIM_SUBDIR

    def fit_out_dir(self) -> Path:
        return self.output_root() / FIT_SUBDIR

    def selected_videos(self) -> list[Path]:
        return [resolve_gui_path(path) for path in self._paths_from_var(self.video_var.get())]

    def selected_gait_jsons(self) -> list[Path]:
        return [resolve_gui_path(path) for path in self._paths_from_var(self.gait_json_var.get())]

    def open_output_folder(self) -> None:
        path = self.output_root()
        path.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:
            messagebox.showerror("Open folder failed", str(exc))

    def auto_rl_output_name(self) -> None:
        model_text = self.rl_model_var.get().strip()
        model_stem = Path(model_text).stem if model_text else "ppo_turning_policy"
        direction = self.rl_turn_direction_var.get().strip() or "turn"
        yaw = safe_name(self.rl_target_yaw_rate_var.get().strip().replace(".", "p") or "yaw")
        radius = self.rl_target_radius_var.get().strip()
        radius_part = f"_r{safe_name(radius.replace('.', 'p'))}" if radius else ""
        name = safe_name(f"{model_stem}_{direction}_yaw{yaw}{radius_part}_{self.rl_strategy_var.get()}")
        self.rl_output_json_var.set(str(RL_GAIT_DIR / f"{name}.json"))

    def make_turning_config_from_gui(self) -> TurningConfig:
        cfg = TurningConfig()
        cfg.turn_direction = self.rl_turn_direction_var.get().strip()
        cfg.target_yaw_rate = abs(float(self.rl_target_yaw_rate_var.get()))
        radius = self._parse_optional_float(self.rl_target_radius_var.get(), "target radius")
        if radius is not None:
            cfg.target_radius = abs(radius)
            cfg.radius_weight = 0.40

        freq = self._parse_optional_float(self.rl_freq_var.get(), "freq")
        if freq is not None:
            cfg.fixed_frequency = freq
        wavelength = self._parse_optional_float(self.rl_wavelength_var.get(), "wavelength")
        if wavelength is not None:
            cfg.fixed_wavelength = wavelength
        ajoint = self._parse_optional_float(self.rl_ajoint_var.get(), "ajoint")
        if ajoint is not None:
            cfg.fixed_ajoint = degrees_to_radians(ajoint)
        low = self._parse_optional_float(self.rl_bias_low_var.get(), "bias low")
        if low is not None:
            cfg.joint_bias_low = low
        high = self._parse_optional_float(self.rl_bias_high_var.get(), "bias high")
        if high is not None:
            cfg.joint_bias_high = high
        if cfg.joint_bias_low > cfg.joint_bias_high:
            raise ValueError("bias low cannot be greater than bias high")
        return cfg

    def set_busy(self, busy: bool) -> None:
        self.worker_running = busy

    def _start_thread(self, func, *args) -> None:
        if self.worker_running:
            messagebox.showwarning("Busy", "A pipeline task is already running.")
            return
        self.set_busy(True)
        thread = threading.Thread(target=self._safe_worker, args=(func, *args), daemon=True)
        thread.start()

    def _safe_worker(self, func, *args) -> None:
        try:
            func(*args)
            self.logger.write("\n=== Done ===\n")
        except Exception as exc:
            self.logger.write(f"\nERROR: {exc}\n")
            self.root.after(0, lambda: messagebox.showerror("Pipeline failed", str(exc)))
        finally:
            self.root.after(0, lambda: self.set_busy(False))

    def _training_output_zip_path(self) -> Path:
        output_base = resolve_gui_path(Path(self.rl_train_output_var.get()))
        return output_base if output_base.suffix.lower() == ".zip" else output_base.with_suffix(".zip")

    def _build_train_command(self) -> list[str]:
        output_base = resolve_gui_path(Path(self.rl_train_output_var.get()))
        if output_base.suffix.lower() == ".zip":
            output_base = output_base.with_suffix("")

        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "train_turning_rl.py"),
            "--timesteps",
            str(int(self.rl_train_timesteps_var.get())),
            "--output",
            str(output_base),
            "--turn-direction",
            self.rl_turn_direction_var.get().strip(),
            "--target-yaw-rate",
            str(abs(float(self.rl_target_yaw_rate_var.get()))),
            "--eval-freq",
            str(int(self.rl_eval_freq_var.get())),
        ]

        radius = self._parse_optional_float(self.rl_target_radius_var.get(), "target radius")
        if radius is not None:
            cmd += ["--target-radius", str(abs(radius))]

        load_model = self.rl_load_model_var.get().strip()
        if load_model:
            cmd += ["--load-model", str(resolve_gui_path(Path(load_model)))]

        optional_args = [
            ("--freq", self.rl_freq_var.get(), "freq"),
            ("--wavelength", self.rl_wavelength_var.get(), "wavelength"),
            ("--ajoint", self.rl_ajoint_var.get(), "ajoint"),
            ("--joint-bias-low", self.rl_bias_low_var.get(), "bias low"),
            ("--joint-bias-high", self.rl_bias_high_var.get(), "bias high"),
        ]
        for flag, text, label in optional_args:
            value = self._parse_optional_float(text, label)
            if value is not None:
                cmd += [flag, str(value)]
        return cmd

    def start_rl_train(self, *, export: bool, view: bool, plot: bool) -> None:
        self._start_thread(self._run_rl_train, export, view, plot)

    def _run_rl_train(self, export: bool, view: bool, plot: bool) -> None:
        cmd = self._build_train_command()
        self.logger.write("\n=== RL PPO training ===\n")
        self.logger.write("CMD: " + " ".join(cmd) + "\n")
        self._run_command_stream(cmd)

        model_zip = self._training_output_zip_path()
        self.logger.write(f"training output model: {model_zip}\n")
        if not model_zip.exists():
            raise FileNotFoundError(f"Training finished but model zip was not found: {model_zip}")

        self.root.after(0, lambda: self.rl_model_var.set(str(model_zip)))
        if export:
            self._run_rl_export_with_model(model_zip, view=view, plot=plot)

    def _run_command_stream(self, cmd: list[str]) -> None:
        proc = subprocess.Popen(
            cmd,
            cwd=SCRIPT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            self.logger.write(line)
        return_code = proc.wait()
        if return_code != 0:
            raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(cmd)}")

    def start_rl_export(self, *, view: bool, plot: bool) -> None:
        self._start_thread(self._run_rl_export, view, plot)

    def _run_rl_export(self, view: bool, plot: bool) -> None:
        model_path = resolve_gui_path(Path(self.rl_model_var.get()))
        self._run_rl_export_with_model(model_path, view=view, plot=plot)

    def _run_rl_export_with_model(self, model_path: Path, *, view: bool, plot: bool) -> None:
        model_path = Path(model_path).expanduser().resolve()
        if model_path.is_dir():
            raise IsADirectoryError(f"PPO model must be a .zip file, not a folder: {model_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"PPO model zip not found: {model_path}")
        output_path = resolve_gui_path(Path(self.rl_output_json_var.get()))
        cfg = self.make_turning_config_from_gui()
        samples = int(self.rl_samples_var.get())
        max_episodes = int(self.rl_max_episodes_var.get())
        strategy = self.rl_strategy_var.get()

        self.logger.write("\n=== RL turning policy export ===\n")
        self.logger.write(f"model={model_path}\n")
        self.logger.write(f"turn_direction={cfg.turn_direction}, target_yaw_rate={cfg.target_yaw_rate}\n")
        self.logger.write(f"target_radius={cfg.target_radius}\n")
        self.logger.write(f"strategy={strategy}, samples={samples}, max_episodes={max_episodes}\n")

        gait_name = output_path.stem
        gait, diag = export_turning_policy_to_gait(
            model_path,
            cfg,
            samples=samples,
            max_episodes=max_episodes,
            strategy=strategy,
            stochastic=False,
            name=gait_name,
        )
        write_gait_json(output_path, gait)
        self.logger.write(f"saved gait JSON: {output_path}\n")
        self.logger.write("diagnostics: " + json.dumps(diag, indent=2) + "\n")
        self.root.after(0, lambda: self._append_gait_path_to_selection(output_path))

        if plot:
            sim_out = self.sim_out_dir()
            fit_out = self.fit_out_dir()
            sim_out.mkdir(parents=True, exist_ok=True)
            fit_out.mkdir(parents=True, exist_ok=True)
            self._run_one_json_gait(output_path, set(), sim_out, fit_out)
        if view:
            self.root.after(0, lambda: self._launch_viewer(output_path))

    def _append_gait_path_to_selection(self, path: Path) -> None:
        paths = self.selected_gait_jsons()
        if path not in paths:
            paths.insert(0, path)
        self.gait_json_var.set(self._join_paths([str(p) for p in paths]))

    def view_first_json(self) -> None:
        paths = self.selected_gait_jsons()
        if not paths:
            messagebox.showerror("Missing JSON", "Please select a gait JSON first.")
            return
        self._launch_viewer(paths[0])

    def view_and_plot_first_json(self) -> None:
        self.view_first_json()
        paths = self.selected_gait_jsons()
        if paths:
            self._start_thread(self._run_json_gaits, [paths[0]])

    def _launch_viewer(self, gait_path: Path) -> None:
        gait_path = Path(gait_path).expanduser().resolve()
        if not gait_path.exists():
            messagebox.showerror("Missing JSON", f"Gait JSON not found: {gait_path}")
            return
        self.stop_viewer(silent=True)
        cmd = [sys.executable, str(SCRIPT_DIR / "view_gait.py"), str(gait_path), "--print-contacts"]
        self.logger.write("\n=== Launch viewer ===\n")
        self.logger.write("CMD: " + " ".join(cmd) + "\n")
        try:
            self.viewer_process = subprocess.Popen(
                cmd,
                cwd=SCRIPT_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            messagebox.showerror("Launch viewer failed", str(exc))
            return
        self.viewer_thread = threading.Thread(target=self._read_viewer_output, daemon=True)
        self.viewer_thread.start()

    def _read_viewer_output(self) -> None:
        proc = self.viewer_process
        if proc is None or proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                self.logger.write(line)
        except ValueError:
            return

    def stop_viewer(self, silent: bool = False) -> None:
        proc = self.viewer_process
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        self.viewer_process = None
        if not silent:
            self.logger.write("\nStopped viewer.\n")

    def start_jsons(self) -> None:
        gait_jsons = self.selected_gait_jsons()
        if not gait_jsons:
            messagebox.showerror("Missing JSON", "Please select one or more gait JSON files first.")
            return
        self._start_thread(self._run_json_gaits, gait_jsons)

    @staticmethod
    def _unique_name(base_name: str, used_names: set[str], sim_out: Path, fit_out: Path) -> str:
        base_name = safe_name(base_name)
        name = base_name
        idx = 2
        while (
            name in used_names
            or (sim_out / f"{name}_trajectory.csv").exists()
            or (fit_out / f"{name}_fitted_summary.json").exists()
        ):
            name = f"{base_name}_{idx}"
            idx += 1
        used_names.add(name)
        return name

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
        ax.set_title(f"{name} trajectory until wall contact")
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

    def _run_json_gaits(self, gait_paths: list[Path]) -> None:
        sim_out = self.sim_out_dir()
        fit_out = self.fit_out_dir()
        sim_out.mkdir(parents=True, exist_ok=True)
        fit_out.mkdir(parents=True, exist_ok=True)

        self.logger.write("\n=== MuJoCo selected JSON gait(s) ===\n")
        self.logger.write(f"count={len(gait_paths)}\n")
        self.logger.write(f"sim_out={sim_out}\n")
        self.logger.write(f"fit_out={fit_out}\n")

        used_names: set[str] = set()
        for gait_path in gait_paths:
            self._run_one_json_gait(gait_path, used_names, sim_out, fit_out)

    def _run_one_json_gait(self, gait_path: Path, used_names: set[str], sim_out: Path, fit_out: Path) -> dict:
        gait_path = Path(gait_path).expanduser().resolve()
        if not gait_path.exists():
            raise FileNotFoundError(f"Gait JSON not found: {gait_path}")

        self.logger.write(f"\nGait JSON: {gait_path}\n")
        gait, arr, hit_wall = run_gait(Path(EEL_MODEL_XML), gait_path, seconds=30.0, start_x=DEFAULT_START_X, start_y=DEFAULT_START_Y)
        if arr.size == 0:
            raise RuntimeError(f"No MuJoCo trajectory was produced for {gait_path}")

        name = self._unique_name(gait_path.stem, used_names, sim_out, fit_out)
        csv_path = sim_out / f"{name}_trajectory.csv"
        np.savetxt(csv_path, arr, delimiter=",", header="time,x,y,yaw", comments="")

        summary = summarize(arr, warmup_seconds=0.0)
        trajectory_png = sim_out / f"{name}_trajectory.png"
        self._write_json_gait_trajectory_plot(trajectory_png, name, arr, summary)

        fixed_summary = {
            "name": name,
            "gait_name_in_json": gait.get("name"),
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
                "gait_name_in_json": gait.get("name"),
                "source_gait_json": str(gait_path),
                "trajectory_csv": str(csv_path),
                "trajectory_png": str(trajectory_png),
                "trajectory_summary_json": str(fixed_summary_path),
                "fit_png": str(fitted_png),
                "hit_wall": bool(hit_wall),
            }
        )
        fitted_summary = {key: self._json_ready(value) for key, value in fitted_summary.items()}
        fitted_summary_path = fit_out / f"{name}_fitted_summary.json"
        fitted_summary_path.write_text(json.dumps(fitted_summary, indent=2), encoding="utf-8")

        radius = fitted_summary.get("radius")
        radius_text = "line/inf" if radius is None else f"{float(radius):.4f}m"
        self.logger.write(f"  output base name: {name}\n")
        self.logger.write(f"  trajectory CSV: {csv_path}\n")
        self.logger.write(f"  trajectory plot: {trajectory_png}\n")
        self.logger.write(f"  fitted plot: {fitted_png}\n")
        self.logger.write(f"  fitted summary: {fitted_summary_path}\n")
        self.logger.write(
            f"  hit_wall={hit_wall}, R={radius_text}, arc={float(fitted_summary.get('arc_deg') or 0.0):.3f}deg, "
            f"rmse={float(fitted_summary.get('rmse') or 0.0):.4f}m\n"
        )
        return fitted_summary

    def start_mp4s(self) -> None:
        videos = self.selected_videos()
        if not videos:
            messagebox.showerror("Missing MP4", "Please select one or more MP4 files first.")
            return
        self._start_thread(self._run_real_videos, videos)

    def _run_real_videos(self, videos: list[Path]) -> None:
        out_root = self.real_out_dir()
        out_root.mkdir(parents=True, exist_ok=True)
        px_per_m = float(self.px_per_m_var.get())
        write_preview = bool(self.preview_var.get())

        self.logger.write("\n=== Real video tracking ===\n")
        self.logger.write(f"count={len(videos)}\n")
        self.logger.write(f"out_root={out_root}\n")
        self.logger.write(f"px_per_m={px_per_m:.6f}\n")

        for video_path in videos:
            video_path = Path(video_path).expanduser().resolve()
            self.logger.write(f"\nVideo: {video_path}\n")
            if not video_path.exists():
                self.logger.write("  SKIP: file not found\n")
                continue
            process_video(video_path=video_path, out_root=out_root, write_preview=write_preview, px_per_m=px_per_m)
            summary_path = out_root / video_path.stem / "tracked_center_summary_cleaned_physical.json"
            result = json.loads(summary_path.read_text(encoding="utf-8"))
            self._log_real_result(result, summary_path)

    def start_selected_pipeline(self) -> None:
        videos = self.selected_videos()
        gait_jsons = self.selected_gait_jsons()
        if not videos and not gait_jsons:
            messagebox.showerror("Missing input", "Please select MP4 files and/or gait JSON files first.")
            return
        self._start_thread(self._run_selected_pipeline, videos, gait_jsons)

    def _run_selected_pipeline(self, videos: list[Path], gait_jsons: list[Path]) -> None:
        self.logger.write("\n=== Selected file pipeline ===\n")
        if videos:
            self._run_real_videos(videos)
        else:
            self.logger.write("No MP4 files selected; skipping real video tracking.\n")

        if gait_jsons:
            self._run_json_gaits(gait_jsons)
        else:
            self.logger.write("No gait JSON files selected; skipping MuJoCo JSON gait analysis.\n")

    def start_curves(self) -> None:
        self._start_thread(self._run_fitted_curves)

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
                str(real_out),
                "--out-dir",
                str(fit_out),
            ]
        )

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
    EelPipelineGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
