from __future__ import annotations

import json
import csv
import math
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np

from hopf_cpg import DEFAULT_AJOINT_DEG, degrees_to_radians
from plot_fixed_gait_trajectories import draw_environment, extract_gait_target_info, plot_one, run_gait, summarize
from plot_fitted_gait_curves import (
    add_fitted_radius_metrics,
    add_fitted_yaw_rate_metrics,
    add_sim_metric_box,
    draw_rotated_tank,
    fitted_curve,
    rotate_sim_xy,
    sim_metric_text,
    trajectory_metrics,
)
from rl_turning_env import TurningConfig
from sim_config import DEFAULT_START_X, DEFAULT_START_Y, EEL_MODEL_XML
from plot_turning_policy_rollout_curves import write_rollout_outputs
from policy_rerun_fixed_gait import write_mean_fixed_gait_from_best_policy
from rl_free_swim_env import FreeSwimConfig
import run_free_swim_paper_10 as free_swim_batch
from run_free_swim_paper_10 import (
    rollout_policy_with_actions as rollout_free_swim_policy_with_actions,
    write_fixed_gait_fitted as write_free_swim_fixed_gait_fitted,
    write_fixed_gait_trajectory as write_free_swim_fixed_gait_trajectory,
    write_mean_gait_json as write_free_swim_mean_gait_json,
    write_policy_rerun_outputs as write_free_swim_policy_rerun_outputs,
)


SCRIPT_DIR = Path(__file__).resolve().parent
SIM_SUBDIR = "fixed_gait_trajectories_mean"
FIT_SUBDIR = "fitted_curve_comparison_mean"
POLICY_ROLLOUT_SUBDIR = "policy_rerun_best_once"
OUTPUTS_DIR = SCRIPT_DIR / "outputs"
DEFAULT_PIPELINE_ROOT = OUTPUTS_DIR
ZIP_DIR = OUTPUTS_DIR / "zips"
CSV_PNG_DIR = OUTPUTS_DIR / "csv_png"
JSON_DIR = OUTPUTS_DIR / "json"
RL_GAIT_DIR = JSON_DIR / "rl_gaits"


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


def model_zip_path(path: Path) -> Path:
    path = Path(path)
    return path if path.name.lower().endswith(".zip") else Path(f"{path}.zip")


def model_save_path(path: Path) -> Path:
    return model_zip_path(path)


class EelPipelineGui:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Robot Eel RL Train / View")
        self.root.geometry("1120x780")
        self.root.minsize(980, 680)

        self.viewer_process: subprocess.Popen | None = None
        self.viewer_thread: threading.Thread | None = None
        self.worker_running = False

        self.out_var = tk.StringVar(value=str(DEFAULT_PIPELINE_ROOT))

        self.rl_train_mode_var = tk.StringVar(value="straight")
        self.rl_target_speed_var = tk.StringVar(value="0.17")
        self.rl_frequency_low_var = tk.StringVar(value="1.0")
        self.rl_frequency_high_var = tk.StringVar(value="1.2")
        self.rl_phase_lag_low_var = tk.StringVar(value="0.5")
        self.rl_phase_lag_high_var = tk.StringVar(value="0.8")
        self.rl_model_var = tk.StringVar()
        self.rl_output_json_var = tk.StringVar(value=str(RL_GAIT_DIR / "rl_turn_right_preview.json"))
        self.rl_turn_direction_var = tk.StringVar(value="right")
        self.rl_target_yaw_rate_var = tk.StringVar(value="0.45")
        self.rl_target_radius_var = tk.StringVar(value="0.4")
        self.rl_use_yaw_reward_var = tk.BooleanVar(value=True)
        self.rl_use_radius_reward_var = tk.BooleanVar(value=True)
        self.rl_yaw_reward_weight_var = tk.StringVar(value="1.20")
        self.rl_radius_reward_weight_var = tk.StringVar(value="1.20")
        self.rl_run_name_var = tk.StringVar(value="eel_train_gui")
        self.rl_run_count_var = tk.StringVar(value="1")
        self.rl_strategy_var = tk.StringVar(value="policy-rerun-mean")
        self.rl_samples_var = tk.StringVar(value="300")
        self.rl_max_episodes_var = tk.StringVar(value="20")
        self.rl_train_timesteps_var = tk.StringVar(value="200000")
        self.rl_train_output_var = tk.StringVar(value=str(ZIP_DIR / "eel_train_gui.zip"))
        self.rl_load_model_var = tk.StringVar(value="")
        self.rl_eval_freq_var = tk.StringVar(value="5000")
        self.rl_freq_var = tk.StringVar(value="")
        self.rl_wavelength_var = tk.StringVar(value="")
        self.rl_ajoint_var = tk.StringVar(value="")
        self.rl_bias_low_var = tk.StringVar(value="-0.35")
        self.rl_bias_high_var = tk.StringVar(value="0.35")
        self.rl_reward_average_seconds_var = tk.StringVar(value="")
        self.rl_boundary_x_min_var = tk.StringVar(value="")
        self.rl_boundary_x_max_var = tk.StringVar(value="")
        self.rl_boundary_y_var = tk.StringVar(value="")

        self.gait_json_var = tk.StringVar()
        self.sim_start_x_var = tk.StringVar(value=f"{DEFAULT_START_X:.3f}")
        self.sim_start_y_var = tk.StringVar(value=f"{DEFAULT_START_Y:.3f}")
        self.view_mode_var = tk.StringVar(value="return_policy")
        self.view_model_var = tk.StringVar()
        self.view_gait_var = tk.StringVar()
        self.view_summary_var = tk.StringVar()

        for path in (ZIP_DIR, RL_GAIT_DIR):
            path.mkdir(parents=True, exist_ok=True)

        self._build_layout()
        self.logger.write("Robot eel training GUI ready.\n")
        self.logger.write("Choose straight speed training or turning yaw/radius training, then run one or more PPO jobs.\n")

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)

        paned = ttk.PanedWindow(outer, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left_pane = ttk.Frame(paned)
        right_pane = ttk.Frame(paned)
        paned.add(left_pane, weight=3)
        paned.add(right_pane, weight=2)

        output_row = ttk.Frame(left_pane)
        output_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(output_row, text="Output root").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Entry(output_row, textvariable=self.out_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(output_row, text="Browse", command=self.browse_output).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(output_row, text="Open", command=self.open_output_folder).pack(side=tk.LEFT, padx=(6, 0))

        control_canvas = tk.Canvas(left_pane, highlightthickness=0)
        control_scrollbar = ttk.Scrollbar(left_pane, orient=tk.VERTICAL, command=control_canvas.yview)
        control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        control_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        control_canvas.configure(yscrollcommand=control_scrollbar.set)

        controls = ttk.Frame(control_canvas)
        controls_window = control_canvas.create_window((0, 0), window=controls, anchor=tk.NW)

        def _update_scroll_region(_event=None) -> None:
            control_canvas.configure(scrollregion=control_canvas.bbox("all"))

        def _fit_controls_width(event) -> None:
            control_canvas.itemconfigure(controls_window, width=event.width)

        controls.bind("<Configure>", _update_scroll_region)
        control_canvas.bind("<Configure>", _fit_controls_width)
        control_canvas.bind_all("<MouseWheel>", lambda event: control_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units"))

        self._build_train_tab(controls)

        log_frame = ttk.LabelFrame(right_pane, text="Log", padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log = tk.Text(log_frame, width=56, wrap=tk.WORD, font=("Consolas", 10))
        self.log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log.configure(yscrollcommand=scrollbar.set, state=tk.DISABLED)
        self.logger = TextLogger(self.log)

    def _build_mode_frame(self, parent: ttk.Frame) -> None:
        options = ttk.LabelFrame(parent, text="Train mode", padding=10)
        options.pack(fill=tk.X, pady=(8, 4))
        ttk.Radiobutton(options, text="Straight speed", variable=self.rl_train_mode_var, value="straight").grid(row=0, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Radiobutton(options, text="Turning yaw/radius", variable=self.rl_train_mode_var, value="turning").grid(row=0, column=1, sticky=tk.W, padx=16, pady=4)

    def _build_straight_target_frame(self, parent: ttk.Frame) -> None:
        options = ttk.LabelFrame(parent, text="Straight speed target", padding=10)
        options.pack(fill=tk.X, pady=(8, 4))
        ttk.Label(options, text="Target speed m/s (+forward, -backward)").grid(row=0, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Entry(options, textvariable=self.rl_target_speed_var, width=12).grid(row=0, column=1, sticky=tk.W, padx=4, pady=4)

    def _build_turning_target_frame(self, parent: ttk.Frame) -> None:
        options = ttk.LabelFrame(parent, text="Turning target", padding=10)
        options.pack(fill=tk.X, pady=(8, 4))
        ttk.Label(options, text="Direction").grid(row=0, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Combobox(options, textvariable=self.rl_turn_direction_var, values=("left", "right"), state="readonly", width=8).grid(row=0, column=1, sticky=tk.W, padx=4, pady=4)
        ttk.Label(options, text="Target yaw rate |rad/s|").grid(row=0, column=2, sticky=tk.W, padx=12, pady=4)
        ttk.Entry(options, textvariable=self.rl_target_yaw_rate_var, width=10).grid(row=0, column=3, sticky=tk.W, padx=4, pady=4)
        ttk.Label(options, text="Target radius m").grid(row=0, column=4, sticky=tk.W, padx=12, pady=4)
        ttk.Entry(options, textvariable=self.rl_target_radius_var, width=10).grid(row=0, column=5, sticky=tk.W, padx=4, pady=4)
        ttk.Label(options, text="Yaw weight").grid(row=1, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Entry(options, textvariable=self.rl_yaw_reward_weight_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=4, pady=4)
        ttk.Label(options, text="Radius weight").grid(row=1, column=2, sticky=tk.W, padx=12, pady=4)
        ttk.Entry(options, textvariable=self.rl_radius_reward_weight_var, width=10).grid(row=1, column=3, sticky=tk.W, padx=4, pady=4)

    def _build_run_options_frame(self, parent: ttk.Frame) -> None:
        run = ttk.LabelFrame(parent, text="Run output", padding=10)
        run.pack(fill=tk.X, pady=(8, 4))
        ttk.Label(run, text="File name").grid(row=0, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Entry(run, textvariable=self.rl_run_name_var, width=26).grid(row=0, column=1, sticky=tk.W, padx=4, pady=4)
        ttk.Label(run, text="Runs").grid(row=0, column=2, sticky=tk.W, padx=12, pady=4)
        ttk.Entry(run, textvariable=self.rl_run_count_var, width=8).grid(row=0, column=3, sticky=tk.W, padx=4, pady=4)
        ttk.Button(run, text="Auto name", command=self.auto_rl_output_name).grid(row=0, column=4, sticky=tk.W, padx=12, pady=4)
        ttk.Label(run, text="Output model base").grid(row=1, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Entry(run, textvariable=self.rl_train_output_var).grid(row=1, column=1, columnspan=3, sticky=tk.EW, padx=4, pady=4)
        ttk.Button(run, text="Browse", command=self.browse_rl_train_output).grid(row=1, column=4, sticky=tk.W, padx=12, pady=4)
        run.columnconfigure(3, weight=1)

    def _build_env_override_frame(self, parent: ttk.Frame) -> None:
        advanced = ttk.LabelFrame(parent, text="Bounds and shared training parameters", padding=10)
        advanced.pack(fill=tk.X, pady=(8, 4))
        ttk.Label(advanced, text="Straight freq low").grid(row=0, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Entry(advanced, textvariable=self.rl_frequency_low_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=4, pady=4)
        ttk.Label(advanced, text="Straight freq high").grid(row=0, column=2, sticky=tk.W, padx=12, pady=4)
        ttk.Entry(advanced, textvariable=self.rl_frequency_high_var, width=10).grid(row=0, column=3, sticky=tk.W, padx=4, pady=4)
        ttk.Label(advanced, text="Straight phase_lag low").grid(row=1, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Entry(advanced, textvariable=self.rl_phase_lag_low_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=4, pady=4)
        ttk.Label(advanced, text="Straight phase_lag high").grid(row=1, column=2, sticky=tk.W, padx=12, pady=4)
        ttk.Entry(advanced, textvariable=self.rl_phase_lag_high_var, width=10).grid(row=1, column=3, sticky=tk.W, padx=4, pady=4)
        ttk.Label(advanced, text="Turning bias low").grid(row=2, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Entry(advanced, textvariable=self.rl_bias_low_var, width=10).grid(row=2, column=1, sticky=tk.W, padx=4, pady=4)
        ttk.Label(advanced, text="Turning bias high").grid(row=2, column=2, sticky=tk.W, padx=12, pady=4)
        ttk.Entry(advanced, textvariable=self.rl_bias_high_var, width=10).grid(row=2, column=3, sticky=tk.W, padx=4, pady=4)
        ttk.Label(advanced, text="Reward avg seconds").grid(row=3, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Entry(advanced, textvariable=self.rl_reward_average_seconds_var, width=10).grid(row=3, column=1, sticky=tk.W, padx=4, pady=4)

    def _build_train_tab(self, parent: ttk.Frame) -> None:
        tab = ttk.Frame(parent, padding=10)
        tab.pack(fill=tk.X, expand=False)

        self._build_mode_frame(tab)
        self._build_straight_target_frame(tab)
        self._build_turning_target_frame(tab)

        train = ttk.LabelFrame(tab, text="Train PPO directly from GUI", padding=10)
        train.pack(fill=tk.X, pady=(8, 4))
        ttk.Label(train, text="Timesteps").grid(row=0, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Entry(train, textvariable=self.rl_train_timesteps_var, width=12).grid(row=0, column=1, sticky=tk.W, padx=4, pady=4)
        ttk.Label(train, text="Eval freq").grid(row=0, column=2, sticky=tk.W, padx=12, pady=4)
        ttk.Entry(train, textvariable=self.rl_eval_freq_var, width=12).grid(row=0, column=3, sticky=tk.W, padx=4, pady=4)
        ttk.Label(train, text="Load model zip (optional)").grid(row=1, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Entry(train, textvariable=self.rl_load_model_var).grid(row=1, column=1, columnspan=4, sticky=tk.EW, padx=4, pady=4)
        ttk.Button(train, text="Browse", command=self.browse_rl_load_model).grid(row=1, column=5, sticky=tk.W, padx=4, pady=4)
        train.columnconfigure(4, weight=1)

        self._build_env_override_frame(tab)
        self._build_run_options_frame(tab)

        buttons = ttk.Frame(tab)
        buttons.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(buttons, text="Train", command=lambda: self.start_rl_train(export=False, view=False, plot=False)).pack(side=tk.LEFT)

        self._build_view_frame(tab)

    def _build_view_frame(self, parent: ttk.Frame) -> None:
        view = ttk.LabelFrame(parent, text="View trained result", padding=10)
        view.pack(fill=tk.X, pady=(10, 4))
        ttk.Radiobutton(view, text="Return policy", variable=self.view_mode_var, value="return_policy").grid(row=0, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Radiobutton(view, text="Fixed gait", variable=self.view_mode_var, value="fixed_gait").grid(row=0, column=1, sticky=tk.W, padx=16, pady=4)
        ttk.Button(view, text="View", command=self.view_selected_result).grid(row=0, column=2, sticky=tk.W, padx=12, pady=4)
        ttk.Button(view, text="Stop View", command=self.stop_viewer).grid(row=0, column=3, sticky=tk.W, padx=4, pady=4)

        ttk.Label(view, text="Return policy .zip").grid(row=1, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Entry(view, textvariable=self.view_model_var).grid(row=1, column=1, columnspan=2, sticky=tk.EW, padx=4, pady=4)
        ttk.Button(view, text="Browse", command=self.browse_view_model).grid(row=1, column=3, sticky=tk.W, padx=4, pady=4)
        ttk.Label(view, text="Policy summary JSON").grid(row=2, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Entry(view, textvariable=self.view_summary_var).grid(row=2, column=1, columnspan=2, sticky=tk.EW, padx=4, pady=4)
        ttk.Button(view, text="Browse", command=self.browse_view_summary).grid(row=2, column=3, sticky=tk.W, padx=4, pady=4)
        ttk.Label(view, text="Fixed gait JSON").grid(row=3, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Entry(view, textvariable=self.view_gait_var).grid(row=3, column=1, columnspan=2, sticky=tk.EW, padx=4, pady=4)
        ttk.Button(view, text="Browse", command=self.browse_view_gait).grid(row=3, column=3, sticky=tk.W, padx=4, pady=4)
        view.columnconfigure(2, weight=1)

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

    def _sim_start_xy(self) -> tuple[float, float]:
        try:
            return float(self.sim_start_x_var.get()), float(self.sim_start_y_var.get())
        except ValueError as exc:
            raise ValueError("MuJoCo start x/y must be numbers") from exc

    def run_count(self) -> int:
        try:
            count = int(self.rl_run_count_var.get())
        except ValueError as exc:
            raise ValueError("Runs must be an integer") from exc
        if count < 1:
            raise ValueError("Runs must be at least 1")
        return count

    def run_base_name(self) -> str:
        text = self.rl_run_name_var.get().strip()
        if text:
            return safe_name(Path(text).stem)
        output_text = self.rl_output_json_var.get().strip()
        if output_text:
            return safe_name(Path(output_text).stem)
        return "ppo_turning_policy"

    def numbered_name(self, base_name: str, index: int, count: int) -> str:
        return base_name if count == 1 else f"{base_name}_run{index:02d}"

    def browse_output(self) -> None:
        dirname = filedialog.askdirectory(title="Select pipeline output root")
        if dirname:
            self.out_var.set(dirname)

    def browse_rl_model(self) -> None:
        filename = filedialog.askopenfilename(
            title="Select PPO model zip",
            initialdir=str(ZIP_DIR),
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
            self.rl_run_name_var.set(safe_name(Path(filename).stem))

    def browse_rl_train_output(self) -> None:
        filename = filedialog.asksaveasfilename(
            title="Select PPO training output base",
            initialdir=str(ZIP_DIR),
            filetypes=(("Stable-Baselines model base", "*"), ("Zip files", "*.zip"), ("All files", "*.*")),
        )
        if filename:
            if not filename.lower().endswith(".zip"):
                filename = f"{filename}.zip"
            self.rl_train_output_var.set(filename)
            self.rl_run_name_var.set(safe_name(Path(filename).name.removesuffix(".zip")))

    def browse_rl_load_model(self) -> None:
        filename = filedialog.askopenfilename(
            title="Select PPO model zip to continue training",
            initialdir=str(ZIP_DIR),
            filetypes=(("PPO zip", "*.zip"), ("All files", "*.*")),
        )
        if filename:
            self.rl_load_model_var.set(filename)

    def browse_view_model(self) -> None:
        filename = filedialog.askopenfilename(
            title="Select return policy PPO model zip",
            initialdir=str(ZIP_DIR),
            filetypes=(("PPO zip", "*.zip"), ("All files", "*.*")),
        )
        if filename:
            self.view_model_var.set(filename)

    def browse_view_summary(self) -> None:
        filename = filedialog.askopenfilename(
            title="Select policy summary JSON",
            initialdir=str(OUTPUTS_DIR),
            filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
        )
        if filename:
            self.view_summary_var.set(filename)

    def browse_view_gait(self) -> None:
        filename = filedialog.askopenfilename(
            title="Select fixed gait JSON",
            initialdir=str(RL_GAIT_DIR),
            filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
        )
        if filename:
            self.view_gait_var.set(filename)

    def output_root(self) -> Path:
        return resolve_gui_path(Path(self.out_var.get()))

    def sim_out_dir(self, run_name: str | None = None) -> Path:
        return CSV_PNG_DIR / SIM_SUBDIR if run_name is None else self.organized_run_dir(run_name) / "06_fixed_gait_trajectory"

    def fit_out_dir(self, run_name: str | None = None) -> Path:
        return CSV_PNG_DIR / FIT_SUBDIR if run_name is None else self.organized_run_dir(run_name) / "07_fixed_gait_fitted"

    def policy_rollout_out_dir(self, run_name: str | None = None) -> Path:
        return CSV_PNG_DIR / POLICY_ROLLOUT_SUBDIR if run_name is None else self.organized_run_dir(run_name) / "04_policy_rerun_mean"

    def eval_best_policy_out_dir(self, run_name: str | None = None) -> Path:
        return CSV_PNG_DIR / "eval_best_policy_curves" if run_name is None else self.organized_run_dir(run_name) / "03_return_policy_trajectory"

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
        mode = self.rl_train_mode_var.get()
        if mode == "straight":
            speed = safe_name(self.rl_target_speed_var.get().strip().replace("-", "neg") or "speed")
            name = safe_name(f"straight_speed_{speed}")
        else:
            direction = self.rl_turn_direction_var.get().strip() or "turn"
            yaw = safe_name(self.rl_target_yaw_rate_var.get().strip() or "yaw")
            radius = self.rl_target_radius_var.get().strip()
            radius_part = f"_r{safe_name(radius)}" if radius else ""
            name = safe_name(f"turn_{direction}_yaw{yaw}{radius_part}")
        self.rl_run_name_var.set(name)
        self.rl_train_output_var.set(str(ZIP_DIR / f"{name}.zip"))
        self.rl_output_json_var.set(str(RL_GAIT_DIR / f"{name}.json"))

    def make_turning_config_from_gui(self) -> TurningConfig:
        if not self.rl_use_yaw_reward_var.get() and not self.rl_use_radius_reward_var.get():
            raise ValueError("Reward must use at least one of yaw_rate or R")
        cfg = TurningConfig()
        cfg.turn_direction = self.rl_turn_direction_var.get().strip()
        cfg.target_yaw_rate = abs(float(self.rl_target_yaw_rate_var.get()))
        cfg.yaw_rate_weight = (
            abs(float(self.rl_yaw_reward_weight_var.get())) if self.rl_use_yaw_reward_var.get() else 0.0
        )
        radius = self._parse_optional_float(self.rl_target_radius_var.get(), "target radius")
        if radius is not None:
            cfg.target_radius = abs(radius)
            cfg.radius_weight = (
                abs(float(self.rl_radius_reward_weight_var.get())) if self.rl_use_radius_reward_var.get() else 0.0
            )
        elif self.rl_use_radius_reward_var.get():
            raise ValueError("Target radius is required when Reward R is enabled")
        else:
            cfg.radius_weight = 0.0

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
        avg_seconds = self._parse_optional_float(self.rl_reward_average_seconds_var.get(), "avg seconds")
        if avg_seconds is not None:
            cfg.reward_average_seconds = avg_seconds
        boundary_x_min = self._parse_optional_float(self.rl_boundary_x_min_var.get(), "boundary x min")
        if boundary_x_min is not None:
            cfg.boundary_x_min = boundary_x_min
        boundary_x_max = self._parse_optional_float(self.rl_boundary_x_max_var.get(), "boundary x max")
        if boundary_x_max is not None:
            cfg.boundary_x_max = boundary_x_max
        boundary_y = self._parse_optional_float(self.rl_boundary_y_var.get(), "boundary y")
        if boundary_y is not None:
            cfg.boundary_y = abs(boundary_y)
        if cfg.boundary_x_min >= cfg.boundary_x_max:
            raise ValueError("boundary x min must be less than boundary x max")
        return cfg

    def make_free_swim_config_from_gui(self) -> FreeSwimConfig:
        cfg = FreeSwimConfig()
        cfg.target_speed = float(self.rl_target_speed_var.get())
        cfg.frequency_low = float(self.rl_frequency_low_var.get())
        cfg.frequency_high = float(self.rl_frequency_high_var.get())
        cfg.phase_lag_low = float(self.rl_phase_lag_low_var.get())
        cfg.phase_lag_high = float(self.rl_phase_lag_high_var.get())
        if cfg.frequency_low > cfg.frequency_high:
            raise ValueError("Straight freq low cannot be greater than freq high")
        if cfg.phase_lag_low > cfg.phase_lag_high:
            raise ValueError("Straight phase_lag low cannot be greater than phase_lag high")
        avg_seconds = self._parse_optional_float(self.rl_reward_average_seconds_var.get(), "avg seconds")
        if avg_seconds is not None:
            cfg.reward_average_seconds = avg_seconds
        return cfg

    def organized_run_dir(self, run_name: str) -> Path:
        return self.output_root() / safe_name(run_name)

    def _copy_file_if_exists(self, source: Path, dest_dir: Path) -> bool:
        source = Path(source)
        if not source.is_file():
            return False
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest_dir / source.name)
        return True

    def _copy_dir_if_exists(self, source: Path, dest_dir: Path) -> int:
        source = Path(source)
        if not source.is_dir():
            return 0
        target = dest_dir / source.name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
        return sum(1 for path in target.rglob("*") if path.is_file())

    def _move_file_if_exists(self, source: Path, dest_dir: Path) -> bool:
        source = Path(source)
        if not source.is_file():
            return False
        dest_dir.mkdir(parents=True, exist_ok=True)
        target = dest_dir / source.name
        if source.resolve() == target.resolve():
            return True
        if target.exists():
            target.unlink()
        shutil.move(str(source), str(target))
        return True

    def _move_dir_contents_if_exists(self, source: Path, dest_dir: Path) -> int:
        source = Path(source)
        if not source.is_dir():
            return 0
        dest_dir.mkdir(parents=True, exist_ok=True)
        moved = 0
        for path in sorted(source.iterdir()):
            target = dest_dir / path.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(path), str(target))
            moved += sum(1 for item in target.rglob("*") if item.is_file()) if target.is_dir() else 1
        try:
            source.rmdir()
        except OSError:
            pass
        return moved

    def organize_run_outputs(self, run_name: str, model_zip: Path, gait_json: Path | None = None) -> None:
        run_name = safe_name(run_name)
        root = self.organized_run_dir(run_name)
        reward_dir = root / "02_reward"
        categories: list[tuple[str, Path, tuple[str, ...]]] = [
            ("01_model", ZIP_DIR, (f"{run_name}.zip",)),
            ("02_reward", reward_dir, (f"{run_name}_eval_reward.png", f"{run_name}_eval_reward.csv", f"{run_name}_eval_debug.csv")),
            (
                "03_return_policy_trajectory",
                reward_dir / "eval_best_policy_curves",
                (
                    f"sim_{run_name}_eval_best_policy_fitted_rotated.png",
                    f"{run_name}_eval_best_policy_trajectory.csv",
                    f"{run_name}_eval_best_policy_summary.json",
                ),
            ),
            (
                "04_policy_rerun_mean",
                self.policy_rollout_out_dir(run_name),
                (
                    f"sim_{run_name}_policy_rerun_fitted_rotated.png",
                    f"{run_name}_policy_rerun_trajectory.csv",
                    f"{run_name}_policy_rerun_summary.json",
                    f"sim_{run_name}_policy_rollout_fitted_rotated.png",
                    f"{run_name}_policy_rollout_trajectory.csv",
                    f"{run_name}_policy_rollout_summary.json",
                ),
            ),
            ("05_fixed_gait_json", RL_GAIT_DIR, (f"{run_name}.json",)),
            ("06_fixed_gait_trajectory", self.sim_out_dir(run_name), (f"{run_name}_trajectory.png", f"{run_name}_trajectory.csv", f"{run_name}_summary.json")),
            ("07_fixed_gait_fitted", self.fit_out_dir(run_name), (f"sim_{run_name}_fitted_rotated.png", f"{run_name}_fitted_summary.json")),
        ]

        rows: list[dict[str, str | int]] = []
        for category, source_dir, patterns in categories:
            dest_dir = root / category
            copied = 0
            for pattern in patterns:
                for path in sorted(source_dir.glob(pattern)):
                    keep_original = category in {"01_model", "05_fixed_gait_json"}
                    moved = self._copy_file_if_exists(path, dest_dir) if keep_original else self._move_file_if_exists(path, dest_dir)
                    if moved:
                        copied += 1
            rows.append({"category": category, "files": copied, "path": str(dest_dir)})

        copied_eval_files = self._move_dir_contents_if_exists(root / "08_eval_log_dir" / f"{run_name}_eval", root / "08_eval_log_dir")
        rows.append({"category": "08_eval_log_dir", "files": copied_eval_files, "path": str(root / "08_eval_log_dir")})
        if self._copy_file_if_exists(model_zip, root / "01_model"):
            rows[0]["files"] = int(rows[0]["files"]) + 1
        if gait_json is not None and self._copy_file_if_exists(gait_json, root / "05_fixed_gait_json"):
            rows[4]["files"] = int(rows[4]["files"]) + 1

        root.mkdir(parents=True, exist_ok=True)
        manifest_path = root / "organized_outputs_manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["category", "files", "path"])
            writer.writeheader()
            writer.writerows(rows)
        self.logger.write(f"organized output folder: {root}\n")

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
        return model_zip_path(output_base)

    def _training_output_zip_for_base(self, output_base: Path) -> Path:
        return model_zip_path(output_base)

    def _best_model_zip_for_model(self, model_path: Path) -> Path | None:
        model_path = Path(model_path).expanduser().resolve()
        stem = model_path.stem if model_path.suffix.lower() == ".zip" else model_path.name
        candidates = [
            self.organized_run_dir(stem) / "08_eval_log_dir" / "best_model" / "best_model.zip",
            self.organized_run_dir(stem) / "08_eval_log_dir" / f"{stem}_eval" / "best_model" / "best_model.zip",
            CSV_PNG_DIR / f"{stem}_eval" / "best_model" / "best_model.zip",
        ]
        for best_zip in candidates:
            if best_zip.exists():
                return best_zip
        return None

    def _eval_best_policy_summary_for_model(self, model_path: Path) -> Path | None:
        model_path = Path(model_path).expanduser().resolve()
        stem = model_path.stem if model_path.suffix.lower() == ".zip" else model_path.name
        candidates = [
            self.eval_best_policy_out_dir(stem) / f"{stem}_eval_best_policy_summary.json",
            self.organized_run_dir(stem) / "02_reward" / "eval_best_policy_curves" / f"{stem}_eval_best_policy_summary.json",
            self.eval_best_policy_out_dir() / f"{stem}_eval_best_policy_summary.json",
        ]
        for summary_path in candidates:
            if summary_path.exists():
                return summary_path
        return None

    def _policy_summary_for_run(self, run_name: str) -> Path | None:
        candidates = [
            self.policy_rollout_out_dir(run_name) / f"{run_name}_policy_rerun_summary.json",
            self.policy_rollout_out_dir(run_name) / f"{run_name}_policy_rollout_summary.json",
            self.eval_best_policy_out_dir(run_name) / f"{run_name}_eval_best_policy_summary.json",
            self.organized_run_dir(run_name) / "02_reward" / "eval_best_policy_curves" / f"{run_name}_eval_best_policy_summary.json",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def _set_latest_view_paths(self, run_name: str, model_zip: Path, gait_json: Path | None) -> None:
        summary_path = self._policy_summary_for_run(run_name)
        self.root.after(0, lambda path=model_zip: self.view_model_var.set(str(path)))
        if summary_path is not None:
            self.root.after(0, lambda path=summary_path: self.view_summary_var.set(str(path)))
        if gait_json is not None:
            self.root.after(0, lambda path=gait_json: self.view_gait_var.set(str(path)))

    def _build_train_command(self, output_base: Path | None = None) -> list[str]:
        mode = self.rl_train_mode_var.get().strip()
        if mode not in {"straight", "turning"}:
            raise ValueError("Train mode must be straight or turning")
        output_base = resolve_gui_path(Path(output_base if output_base is not None else self.rl_train_output_var.get()))
        output_base = model_save_path(output_base)
        run_name = output_base.name.removesuffix(".zip")
        run_root = self.organized_run_dir(run_name)
        reward_dir = run_root / "02_reward"
        eval_log_dir = run_root / "08_eval_log_dir" / f"{run_name}_eval"
        plot_output = reward_dir / f"{run_name}_eval_reward.png"
        reward_dir.mkdir(parents=True, exist_ok=True)
        eval_log_dir.parent.mkdir(parents=True, exist_ok=True)

        if mode == "straight":
            freq_low = float(self.rl_frequency_low_var.get())
            freq_high = float(self.rl_frequency_high_var.get())
            phase_low = float(self.rl_phase_lag_low_var.get())
            phase_high = float(self.rl_phase_lag_high_var.get())
            if freq_low > freq_high:
                raise ValueError("Straight freq low cannot be greater than freq high")
            if phase_low > phase_high:
                raise ValueError("Straight phase_lag low cannot be greater than phase_lag high")
            cmd = [
                sys.executable,
                str(SCRIPT_DIR / "train_free_swim_rl.py"),
                "--timesteps",
                str(int(self.rl_train_timesteps_var.get())),
                "--output",
                str(output_base),
                "--target-speed",
                str(float(self.rl_target_speed_var.get())),
                "--freq-low",
                str(freq_low),
                "--freq-high",
                str(freq_high),
                "--phase-lag-low",
                str(phase_low),
                "--phase-lag-high",
                str(phase_high),
                "--eval-freq",
                str(int(self.rl_eval_freq_var.get())),
                "--eval-log-dir",
                str(eval_log_dir),
                "--plot-output",
                str(plot_output),
            ]
            load_model = self.rl_load_model_var.get().strip()
            if load_model:
                cmd += ["--load-model", str(resolve_gui_path(Path(load_model)))]
            avg_seconds = self._parse_optional_float(self.rl_reward_average_seconds_var.get(), "avg seconds")
            if avg_seconds is not None:
                cmd += ["--reward-average-seconds", str(avg_seconds)]
            return cmd

        if not self.rl_use_yaw_reward_var.get() and not self.rl_use_radius_reward_var.get():
            raise ValueError("Reward must use at least one of yaw_rate or R")
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
            "--eval-log-dir",
            str(eval_log_dir),
            "--plot-output",
            str(plot_output),
        ]

        radius = self._parse_optional_float(self.rl_target_radius_var.get(), "target radius")
        if radius is not None:
            cmd += ["--target-radius", str(abs(radius))]
        elif self.rl_use_radius_reward_var.get():
            raise ValueError("Target radius is required when Reward R is enabled")

        cmd += [
            "--yaw-rate-weight",
            str(abs(float(self.rl_yaw_reward_weight_var.get())) if self.rl_use_yaw_reward_var.get() else 0.0),
            "--radius-weight",
            str(abs(float(self.rl_radius_reward_weight_var.get())) if self.rl_use_radius_reward_var.get() else 0.0),
        ]

        load_model = self.rl_load_model_var.get().strip()
        if load_model:
            cmd += ["--load-model", str(resolve_gui_path(Path(load_model)))]

        optional_args = [
            ("--freq", self.rl_freq_var.get(), "freq"),
            ("--wavelength", self.rl_wavelength_var.get(), "wavelength"),
            ("--ajoint", self.rl_ajoint_var.get(), "ajoint"),
            ("--joint-bias-low", self.rl_bias_low_var.get(), "bias low"),
            ("--joint-bias-high", self.rl_bias_high_var.get(), "bias high"),
            ("--reward-average-seconds", self.rl_reward_average_seconds_var.get(), "avg seconds"),
            ("--boundary-x-min", self.rl_boundary_x_min_var.get(), "boundary x min"),
            ("--boundary-x-max", self.rl_boundary_x_max_var.get(), "boundary x max"),
            ("--boundary-y", self.rl_boundary_y_var.get(), "boundary y"),
        ]
        for flag, text, label in optional_args:
            value = self._parse_optional_float(text, label)
            if value is not None:
                cmd += [flag, str(value)]
        return cmd

    def start_rl_train(self, *, export: bool, view: bool, plot: bool) -> None:
        self._start_thread(self._run_rl_train, export, view, plot)

    def _run_rl_train(self, export: bool, view: bool, plot: bool) -> None:
        count = self.run_count()
        base_name = self.run_base_name()
        mode = self.rl_train_mode_var.get().strip()
        output_template = resolve_gui_path(Path(self.rl_train_output_var.get()))
        output_template = model_zip_path(output_template)
        last_model_zip: Path | None = None
        last_output_json: Path | None = None
        for index in range(1, count + 1):
            run_name = self.numbered_name(base_name, index, count)
            output_base = output_template if count == 1 else output_template.with_name(f"{run_name}.zip")
            output_json = RL_GAIT_DIR / f"{run_name}.json"
            cmd = self._build_train_command(output_base)
            self.logger.write(f"\n=== RL PPO training {index}/{count}: {run_name} ===\n")
            self.logger.write("CMD: " + " ".join(cmd) + "\n")
            self._run_command_stream(cmd)

            model_zip = self._training_output_zip_for_base(output_base)
            self.logger.write(f"training output model: {model_zip}\n")
            if not model_zip.exists():
                raise FileNotFoundError(f"Training finished but model zip was not found: {model_zip}")

            last_model_zip = model_zip
            last_output_json = output_json
            if mode == "straight":
                self._run_free_swim_post_train_outputs(run_name, model_zip, output_json)
            else:
                self._run_rl_export_with_model(model_zip, view=False, plot=True, output_path=output_json)
            self.organize_run_outputs(run_name, model_zip, output_json)
            self._set_latest_view_paths(run_name, model_zip, output_json)

        if last_model_zip is not None:
            self.root.after(0, lambda path=last_model_zip: self.rl_model_var.set(str(path)))
        if last_output_json is not None:
            self.root.after(0, lambda path=last_output_json: self.rl_output_json_var.set(str(path)))

    def _run_free_swim_post_train_outputs(self, run_name: str, model_zip: Path, output_json: Path) -> None:
        cfg = self.make_free_swim_config_from_gui()
        best_model = self._best_model_zip_for_model(model_zip)
        export_model = best_model or model_zip
        run_root = self.organized_run_dir(run_name)
        free_swim_batch.POLICY_RERUN_DIR = self.policy_rollout_out_dir(run_name)
        free_swim_batch.TRAJ_DIR = self.sim_out_dir(run_name)
        free_swim_batch.FIT_DIR = self.fit_out_dir(run_name)
        free_swim_batch.JSON_DIR = run_root / "05_fixed_gait_json"
        self.logger.write("\n=== Free-swim policy rerun and fixed gait ===\n")
        self.logger.write(f"selected_model={model_zip}\n")
        self.logger.write(f"fixed_gait_model={export_model}\n")
        self.logger.write(f"target_speed={cfg.target_speed}\n")
        arr, total_reward = rollout_free_swim_policy_with_actions(export_model, cfg)
        policy_outputs = write_free_swim_policy_rerun_outputs(run_name, cfg, export_model, arr, total_reward)
        gait_path, diag = write_free_swim_mean_gait_json(run_name, cfg, export_model, arr, Path(policy_outputs["policy_rerun_csv"]))
        if gait_path != output_json:
            output_json.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(gait_path, output_json)
            gait_path = output_json
        trajectory_outputs = write_free_swim_fixed_gait_trajectory(run_name, gait_path, cfg)
        fitted_outputs = write_free_swim_fixed_gait_fitted(run_name, Path(trajectory_outputs["trajectory_csv"]))
        self.logger.write(f"saved gait JSON: {gait_path}\n")
        self.logger.write(f"policy rerun plot: {policy_outputs.get('policy_rerun_png')}\n")
        self.logger.write(f"fixed gait plot: {trajectory_outputs.get('trajectory_png')}\n")
        self.logger.write(f"fixed gait fitted plot: {fitted_outputs.get('fitted_png')}\n")
        self.logger.write("diagnostics: " + json.dumps(diag, indent=2) + "\n")
        self.root.after(0, lambda path=gait_path: self._append_gait_path_to_selection(path))

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
        count = self.run_count()
        base_name = self.run_base_name()
        if count == 1:
            output_json = RL_GAIT_DIR / f"{base_name}.json"
            self._run_rl_export_with_model(model_path, view=view, plot=plot, output_path=output_json)
            self.root.after(0, lambda path=output_json: self.rl_output_json_var.set(str(path)))
            return
        last_output_json: Path | None = None
        for index in range(1, count + 1):
            run_name = self.numbered_name(base_name, index, count)
            output_json = RL_GAIT_DIR / f"{run_name}.json"
            self._run_rl_export_with_model(model_path, view=view and index == count, plot=plot, output_path=output_json)
            last_output_json = output_json
        if last_output_json is not None:
            self.root.after(0, lambda path=last_output_json: self.rl_output_json_var.set(str(path)))

    def _run_rl_export_with_model(self, model_path: Path, *, view: bool, plot: bool, output_path: Path | None = None) -> None:
        model_path = Path(model_path).expanduser().resolve()
        if model_path.is_dir():
            raise IsADirectoryError(f"PPO model must be a .zip file, not a folder: {model_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"PPO model zip not found: {model_path}")
        output_path = resolve_gui_path(Path(output_path if output_path is not None else self.rl_output_json_var.get()))
        cfg = self.make_turning_config_from_gui()
        best_model = self._best_model_zip_for_model(model_path)
        export_model = best_model or model_path
        model_source = "eval_best_model" if best_model is not None else "selected_model"

        self.logger.write("\n=== RL turning policy export ===\n")
        self.logger.write(f"selected_model={model_path}\n")
        self.logger.write(f"fixed_gait_model={export_model}\n")
        self.logger.write(f"model_source={model_source}\n")
        self.logger.write(f"turn_direction={cfg.turn_direction}, target_yaw_rate={cfg.target_yaw_rate}\n")
        self.logger.write(f"target_radius={cfg.target_radius}\n")
        self.logger.write(f"train_boundary=({cfg.boundary_x_min}, {cfg.boundary_x_max}, +/-{cfg.boundary_y})\n")
        self.logger.write("strategy=policy-rerun-mean\n")

        gait_name = output_path.stem
        run_root = self.organized_run_dir(gait_name)
        outputs, diag = write_mean_fixed_gait_from_best_policy(
            name=gait_name,
            cfg=cfg,
            model_zip=export_model,
            gait_path=output_path,
            policy_out_dir=self.policy_rollout_out_dir(gait_name),
            source_extra={
                "gui": {
                    "selected_model": str(model_path),
                    "model_source": model_source,
                }
            },
        )
        self.logger.write(f"saved gait JSON: {output_path}\n")
        self.logger.write(f"policy rerun CSV: {outputs.get('policy_rerun_csv')}\n")
        self.logger.write(f"policy rerun plot: {outputs.get('policy_rerun_png')}\n")
        self.logger.write("diagnostics: " + json.dumps(diag, indent=2) + "\n")
        self.root.after(0, lambda: self._append_gait_path_to_selection(output_path))

        if plot:
            sim_out = self.sim_out_dir(gait_name)
            fit_out = self.fit_out_dir(gait_name)
            sim_out.mkdir(parents=True, exist_ok=True)
            fit_out.mkdir(parents=True, exist_ok=True)
            start_x, start_y = self._sim_start_xy()
            self.logger.write("\n=== Fixed gait curve from exported JSON ===\n")
            self.logger.write(f"start_x={start_x:.3f}, start_y={start_y:.3f}\n")
            self._run_one_json_gait(output_path, set(), sim_out, fit_out, start_x=start_x, start_y=start_y)
        if view:
            self.root.after(0, lambda: self._launch_viewer(output_path))

    def _run_policy_rollout_plot(self, model_path: Path, cfg: TurningConfig, name: str) -> dict:
        eval_best_summary = self._eval_best_policy_summary_for_model(model_path)
        if eval_best_summary is not None:
            summary = json.loads(eval_best_summary.read_text(encoding="utf-8"))
            self.logger.write("\n=== Eval best policy curve ===\n")
            self.logger.write("model_source=eval_best_episode\n")
            self.logger.write(f"summary={eval_best_summary}\n")
            self.logger.write(f"trajectory CSV: {summary.get('trajectory_csv')}\n")
            self.logger.write(f"fitted plot: {summary.get('eval_best_policy_png')}\n")
            fitted_yaw = summary.get("fitted_yaw_rate_rad_s")
            yaw_err = summary.get("fitted_yaw_rate_error_rad_s")
            fitted_yaw_text = "nan" if fitted_yaw is None else f"{float(fitted_yaw):.3f}rad/s"
            yaw_err_text = "nan" if yaw_err is None else f"{float(yaw_err):.3f}rad/s"
            self.logger.write(f"  fitted_yaw_rate={fitted_yaw_text}, yaw_err={yaw_err_text}\n")
            return summary

        best_model = self._best_model_zip_for_model(model_path)
        plot_model = best_model or model_path
        model_source = "best_eval" if best_model is not None else "selected_model"
        out_dir = self.policy_rollout_out_dir(name)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.logger.write("\n=== PPO policy rollout curve ===\n")
        self.logger.write(f"model_source={model_source}\n")
        self.logger.write(f"policy_model={plot_model}\n")
        self.logger.write(f"out_dir={out_dir}\n")
        summary = write_rollout_outputs(
            name=name,
            model_zip=plot_model,
            cfg=cfg,
            out_dir=out_dir,
            deterministic=True,
        )
        summary["model_source"] = model_source
        fitted_yaw = summary.get("fitted_yaw_rate_rad_s")
        yaw_err = summary.get("fitted_yaw_rate_error_rad_s")
        fitted_yaw_text = "nan" if fitted_yaw is None else f"{float(fitted_yaw):.3f}rad/s"
        yaw_err_text = "nan" if yaw_err is None else f"{float(yaw_err):.3f}rad/s"
        self.logger.write(f"  policy rollout CSV: {summary.get('trajectory_csv')}\n")
        self.logger.write(f"  policy rollout plot: {summary.get('policy_rollout_png')}\n")
        self.logger.write(f"  fitted_yaw_rate={fitted_yaw_text}, yaw_err={yaw_err_text}\n")
        return summary

    def _append_gait_path_to_selection(self, path: Path) -> None:
        paths = self.selected_gait_jsons()
        if path not in paths:
            paths.insert(0, path)
        self.gait_json_var.set(self._join_paths([str(p) for p in paths]))

    def view_selected_result(self) -> None:
        if self.view_mode_var.get() == "return_policy":
            model_path = self.view_model_var.get().strip() or self.rl_model_var.get().strip()
            if not model_path:
                messagebox.showerror("Missing model", "Please select a return policy .zip first.")
                return
            summary_text = self.view_summary_var.get().strip()
            self._launch_return_policy_viewer(Path(model_path), Path(summary_text) if summary_text else None)
            return

        gait_path = self.view_gait_var.get().strip() or self.rl_output_json_var.get().strip()
        if not gait_path:
            messagebox.showerror("Missing fixed gait", "Please select a fixed gait JSON first.")
            return
        self._launch_viewer(Path(gait_path), start_x=0.0, start_y=0.0)

    def view_first_json(self) -> None:
        paths = self.selected_gait_jsons()
        if not paths:
            messagebox.showerror("Missing JSON", "Please select a gait JSON first.")
            return
        try:
            start_x, start_y = self._sim_start_xy()
        except ValueError as exc:
            messagebox.showerror("Invalid start position", str(exc))
            return
        self._launch_viewer(paths[0], start_x=start_x, start_y=start_y)

    def view_and_plot_first_json(self) -> None:
        self.view_first_json()
        paths = self.selected_gait_jsons()
        if paths:
            self._start_thread(self._run_json_gaits, [paths[0]])

    def _launch_viewer(self, gait_path: Path, *, start_x: float = 0.0, start_y: float = 0.0) -> None:
        gait_path = Path(gait_path).expanduser().resolve()
        if not gait_path.exists():
            messagebox.showerror("Missing JSON", f"Gait JSON not found: {gait_path}")
            return
        self.stop_viewer(silent=True)
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "view_gait.py"),
            str(gait_path),
            "--start-x",
            str(start_x),
            "--start-y",
            str(start_y),
            "--camera-mode",
            "follow",
        ]
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

    def _launch_return_policy_viewer(self, model_path: Path, summary_path: Path | None) -> None:
        model_path = Path(model_path).expanduser().resolve()
        if not model_path.exists():
            messagebox.showerror("Missing model", f"PPO model zip not found: {model_path}")
            return
        if summary_path is not None:
            summary_path = Path(summary_path).expanduser().resolve()
            if not summary_path.exists():
                messagebox.showerror("Missing summary", f"Policy summary JSON not found: {summary_path}")
                return
        self.stop_viewer(silent=True)
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "view_return_policy.py"),
            str(model_path),
            "--mode",
            "auto" if summary_path is not None else self.rl_train_mode_var.get().strip(),
            "--camera-mode",
            "follow",
        ]
        if summary_path is not None:
            cmd += ["--summary", str(summary_path)]
        if self.rl_train_mode_var.get().strip() == "straight":
            cmd += ["--target-speed", str(float(self.rl_target_speed_var.get()))]
        else:
            cmd += [
                "--turn-direction",
                self.rl_turn_direction_var.get().strip(),
                "--target-yaw-rate",
                str(abs(float(self.rl_target_yaw_rate_var.get()))),
            ]
            radius = self._parse_optional_float(self.rl_target_radius_var.get(), "target radius")
            if radius is not None:
                cmd += ["--target-radius", str(abs(radius))]
        self.logger.write("\n=== Launch return policy viewer ===\n")
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

    def _write_json_gait_trajectory_plot(
        self,
        png_path: Path,
        name: str,
        arr: np.ndarray,
        summary: dict,
        *,
        start_x: float,
        start_y: float,
    ) -> None:
        fig, ax = plt.subplots(figsize=(7, 5), dpi=170)
        draw_environment(ax, start_x, start_y)
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

    def _write_json_gait_fitted_plot(
        self,
        png_path: Path,
        name: str,
        arr: np.ndarray,
        target_yaw_rate: float | None = None,
        turn_direction: str | None = None,
        target_radius_m: float | None = None,
    ) -> dict:
        xy = rotate_sim_xy(arr[:, 1:3])
        curve, fit = fitted_curve(xy)
        metrics = trajectory_metrics(arr, xy)
        metrics.update(add_fitted_yaw_rate_metrics(fit, metrics, target_yaw_rate, turn_direction))
        metrics.update(add_fitted_radius_metrics(fit, target_radius_m))

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
        start_x, start_y = self._sim_start_xy()

        self.logger.write("\n=== MuJoCo selected JSON gait(s) ===\n")
        self.logger.write(f"count={len(gait_paths)}\n")
        self.logger.write(f"start_x={start_x:.3f}, start_y={start_y:.3f}\n")
        self.logger.write(f"sim_out={sim_out}\n")
        self.logger.write(f"fit_out={fit_out}\n")

        used_names: set[str] = set()
        for gait_path in gait_paths:
            self._run_one_json_gait(gait_path, used_names, sim_out, fit_out, start_x=start_x, start_y=start_y)

    def _run_one_json_gait(
        self,
        gait_path: Path,
        used_names: set[str],
        sim_out: Path,
        fit_out: Path,
        *,
        start_x: float,
        start_y: float,
    ) -> dict:
        gait_path = Path(gait_path).expanduser().resolve()
        if not gait_path.exists():
            raise FileNotFoundError(f"Gait JSON not found: {gait_path}")

        self.logger.write(f"\nGait JSON: {gait_path}\n")
        gait, arr, hit_wall = run_gait(Path(EEL_MODEL_XML), gait_path, seconds=30.0, start_x=start_x, start_y=start_y)
        if arr.size == 0:
            raise RuntimeError(f"No MuJoCo trajectory was produced for {gait_path}")

        name = self._unique_name(gait_path.stem, used_names, sim_out, fit_out)
        csv_path = sim_out / f"{name}_trajectory.csv"
        np.savetxt(csv_path, arr, delimiter=",", header="time,x,y,yaw", comments="")

        summary = summarize(arr, warmup_seconds=0.0)
        trajectory_png = sim_out / f"{name}_trajectory.png"
        self._write_json_gait_trajectory_plot(trajectory_png, name, arr, summary, start_x=start_x, start_y=start_y)

        target_info = extract_gait_target_info(gait)
        fixed_summary = {
            "name": name,
            "gait_name_in_json": gait.get("name"),
            "source_gait_json": str(gait_path),
            "trajectory_csv": str(csv_path),
            "trajectory_png": str(trajectory_png),
            "start_x_m": float(start_x),
            "start_y_m": float(start_y),
            "duration_s": float(arr[-1, 0] - arr[0, 0]) if arr.shape[0] >= 2 else 0.0,
            "hit_wall": bool(hit_wall),
            **target_info,
            **{key: self._json_ready(value) for key, value in summary.items() if key != "warmup_index"},
        }
        fixed_summary_path = sim_out / f"{name}_summary.json"
        fixed_summary_path.write_text(json.dumps(fixed_summary, indent=2), encoding="utf-8")

        fitted_png = fit_out / f"sim_{name}_fitted_rotated.png"
        yaw_reward_weight = target_info.get("yaw_rate_reward_weight")
        target_yaw_for_plot = None if yaw_reward_weight == 0.0 else target_info.get("target_yaw_rate_rad_s")
        radius_reward_weight = target_info.get("radius_reward_weight")
        target_radius_for_plot = None if radius_reward_weight == 0.0 else target_info.get("target_radius_m")
        fitted_summary = self._write_json_gait_fitted_plot(
            fitted_png,
            name,
            arr,
            target_yaw_rate=target_yaw_for_plot,
            turn_direction=target_info.get("turn_direction"),
            target_radius_m=target_radius_for_plot,
        )
        curve_output_png = fitted_png
        fitted_summary.update(
            {
                "gait_name_in_json": gait.get("name"),
                "source_gait_json": str(gait_path),
                "trajectory_csv": str(csv_path),
                "trajectory_png": str(trajectory_png),
                "trajectory_summary_json": str(fixed_summary_path),
                "fit_png": str(fitted_png),
                "fit_output_png": str(curve_output_png),
                "hit_wall": bool(hit_wall),
                **target_info,
            }
        )
        fitted_summary = {key: self._json_ready(value) for key, value in fitted_summary.items()}
        fitted_summary_path = fit_out / f"{name}_fitted_summary.json"
        fitted_summary_path.write_text(json.dumps(fitted_summary, indent=2), encoding="utf-8")

        radius = fitted_summary.get("radius")
        radius_text = "line/inf" if radius is None else f"{float(radius):.4f}m"
        target_yaw = fitted_summary.get("target_yaw_rate_rad_s")
        fitted_yaw = fitted_summary.get("fitted_yaw_rate_rad_s")
        yaw_err = fitted_summary.get("fitted_yaw_rate_error_rad_s")
        target_radius = fitted_summary.get("target_radius_m")
        radius_err = fitted_summary.get("fitted_radius_error_m")
        target_text = "none" if target_yaw is None else f"{float(target_yaw):.3f}rad/s"
        fitted_text = "nan" if fitted_yaw is None else f"{float(fitted_yaw):.3f}rad/s"
        err_text = "nan" if yaw_err is None else f"{float(yaw_err):.3f}rad/s"
        radius_target_text = "none" if target_radius is None else f"{float(target_radius):.3f}m"
        radius_err_text = "nan" if radius_err is None else f"{float(radius_err):.3f}m"
        self.logger.write(f"  output base name: {name}\n")
        self.logger.write(f"  trajectory CSV: {csv_path}\n")
        self.logger.write(f"  trajectory plot: {trajectory_png}\n")
        self.logger.write(f"  fitted plot: {fitted_png}\n")
        self.logger.write(f"  fitted output plot: {curve_output_png}\n")
        self.logger.write(f"  fitted summary: {fitted_summary_path}\n")
        self.logger.write(
            f"  hit_wall={hit_wall}, R={radius_text}, fitted_yaw_rate={fitted_text}, "
            f"target={target_text}, err={err_text}, "
            f"R_target={radius_target_text}, R_err={radius_err_text}, "
            f"arc={float(fitted_summary.get('arc_deg') or 0.0):.3f}deg, "
            f"rmse={float(fitted_summary.get('rmse') or 0.0):.4f}m\n"
        )
        return fitted_summary

def main() -> None:
    root = tk.Tk()
    EelPipelineGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
