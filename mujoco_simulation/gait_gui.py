from __future__ import annotations
import json
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk


ROOT = Path(__file__).resolve().parent
GAIT_DIRS = (
    ROOT / "gaits",
    ROOT / "outputs" / "rl_gaits",
)
REQUIRED_GAIT_FIELDS = ("ajoint", "freq", "wavelength", "amp_scales", "phase_lags", "joint_bias")


def gait_source_label(path: Path, data: dict) -> str:
    source = data.get("source")
    if isinstance(source, dict):
        source_type = str(source.get("type", "")).lower()
        if "turning" in source_type:
            return "RL turning"
        if "ppo" in source_type or path.stem.startswith("rl_"):
            return "RL"
    if path.stem.startswith("rl_turn"):
        return "RL turning"
    if path.stem.startswith("rl_"):
        return "RL"
    return "preset"


def gait_sort_key(item: dict) -> tuple[int, str, str]:
    path = item["file"]
    label = item["source_label"]
    priority = 0 if label.startswith("RL") else 1
    return (priority, path.stem.lower(), str(path).lower())


class GaitGui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Eel Gait Viewer")
        self.geometry("820x520")
        self.minsize(760, 480)
        self.process: subprocess.Popen | None = None
        self.output_queue: queue.Queue[str] = queue.Queue()
        self.reader_thread: threading.Thread | None = None
        self.metrics_window: tk.Toplevel | None = None
        self.metrics_text: tk.Text | None = None
        self.gaits = self.load_gaits()
        self.selected_gait = tk.StringVar(value=self.gaits[0]["file"].name if self.gaits else "")
        self.mode = tk.StringVar(value="rectangle")
        self.step_control_mode = tk.StringVar(value="hopf")
        self.step_period = tk.StringVar(value="3.0")
        self.step_after_wavelength = tk.StringVar(value="3.0")
        self.step_after_amp = tk.StringVar(value="0.55")
        self.step_alpha = tk.StringVar(value="4.0")
        self.step_k_couple = tk.StringVar(value="0.35")
        self.step_k_anchor = tk.StringVar(value="0.10")

        self.create_widgets()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(500, self.poll_process)
        self.after(100, self.poll_output)

    def load_gaits(self) -> list[dict]:
        gaits = []
        seen_paths: set[Path] = set()
        for gait_dir in GAIT_DIRS:
            if not gait_dir.exists():
                continue
            for path in sorted(gait_dir.glob("*.json")):
                resolved = path.resolve()
                if resolved in seen_paths:
                    continue
                seen_paths.add(resolved)
                try:
                    with path.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                except (OSError, json.JSONDecodeError):
                    continue
                missing = [key for key in REQUIRED_GAIT_FIELDS if key not in data]
                if missing:
                    continue
                source_label = gait_source_label(path, data)
                try:
                    relative = path.relative_to(ROOT)
                except ValueError:
                    relative = path
                gaits.append(
                    {
                        "file": path,
                        "relative": relative,
                        "data": data,
                        "source_label": source_label,
                    }
                )
        return sorted(gaits, key=gait_sort_key)

    def gait_display_text(self, gait: dict) -> str:
        data = gait["data"]
        name = data.get("name", gait["file"].stem)
        return f"[{gait['source_label']}] {name}  ({gait['relative']})"

    def create_widgets(self):
        outer = ttk.Frame(self, padding=16)
        outer.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(outer)
        top.pack(fill=tk.X)

        title = ttk.Label(top, text="Eel Gait Viewer", font=("Segoe UI", 16, "bold"))
        title.pack(side=tk.LEFT)

        mode_box = ttk.Frame(top)
        mode_box.pack(side=tk.RIGHT)
        ttk.Radiobutton(mode_box, text="Rectangle Course", variable=self.mode, value="rectangle", command=self.update_mode).pack(side=tk.LEFT)
        ttk.Radiobutton(mode_box, text="Fixed Gait", variable=self.mode, value="gait", command=self.update_mode).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Radiobutton(mode_box, text="CPG Step Test", variable=self.mode, value="step", command=self.update_mode).pack(side=tk.LEFT, padx=(8, 0))

        controls = ttk.Frame(outer)
        controls.pack(fill=tk.X, pady=(12, 0))

        self.open_button = ttk.Button(controls, text="Open Viewer", command=self.open_viewer)
        self.open_button.pack(side=tk.LEFT)

        self.stop_button = ttk.Button(controls, text="Stop Viewer", command=self.stop_viewer, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(8, 0))

        self.refresh_button = ttk.Button(controls, text="Refresh", command=self.refresh_gaits)
        self.refresh_button.pack(side=tk.LEFT, padx=(8, 0))

        self.metrics_button = ttk.Button(controls, text="Metrics Window", command=self.show_metrics_window)
        self.metrics_button.pack(side=tk.LEFT, padx=(8, 0))

        self.status = ttk.Label(controls, text="Ready")
        self.status.pack(side=tk.RIGHT)

        self.step_options = ttk.LabelFrame(outer, text="CPG Step Test", padding=10)
        self.step_options.pack(fill=tk.X, pady=(12, 0))

        ttk.Label(self.step_options, text="Control").pack(side=tk.LEFT)
        self.step_control_combo = ttk.Combobox(
            self.step_options,
            textvariable=self.step_control_mode,
            values=("hopf", "sin"),
            state="readonly",
            width=8,
        )
        self.step_control_combo.pack(side=tk.LEFT, padx=(6, 14))

        ttk.Label(self.step_options, text="Period").pack(side=tk.LEFT)
        self.step_period_entry = ttk.Entry(self.step_options, textvariable=self.step_period, width=6)
        self.step_period_entry.pack(side=tk.LEFT, padx=(6, 14))

        ttk.Label(self.step_options, text="After lambda").pack(side=tk.LEFT)
        self.step_wavelength_entry = ttk.Entry(self.step_options, textvariable=self.step_after_wavelength, width=7)
        self.step_wavelength_entry.pack(side=tk.LEFT, padx=(6, 14))

        ttk.Label(self.step_options, text="After amp").pack(side=tk.LEFT)
        self.step_amp_entry = ttk.Entry(self.step_options, textvariable=self.step_after_amp, width=7)
        self.step_amp_entry.pack(side=tk.LEFT, padx=(6, 0))

        step_options_2 = ttk.Frame(self.step_options)
        step_options_2.pack(fill=tk.X, pady=(8, 0))

        ttk.Label(step_options_2, text="alpha").pack(side=tk.LEFT)
        ttk.Entry(step_options_2, textvariable=self.step_alpha, width=7).pack(side=tk.LEFT, padx=(6, 14))

        ttk.Label(step_options_2, text="k_couple").pack(side=tk.LEFT)
        ttk.Entry(step_options_2, textvariable=self.step_k_couple, width=7).pack(side=tk.LEFT, padx=(6, 14))

        ttk.Label(step_options_2, text="k_anchor").pack(side=tk.LEFT)
        ttk.Entry(step_options_2, textvariable=self.step_k_anchor, width=7).pack(side=tk.LEFT, padx=(6, 0))

        body = ttk.Frame(outer)
        body.pack(fill=tk.BOTH, expand=True, pady=(14, 10))

        left = ttk.LabelFrame(body, text="Fixed Gaits", padding=10)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Label(
            left,
            text="Reads: gaits/*.json and outputs/rl_gaits/*.json. RL gaits are listed first.",
            foreground="#555555",
        ).pack(anchor=tk.W, pady=(0, 6))

        self.listbox = tk.Listbox(left, height=10, exportselection=False)
        self.listbox.pack(fill=tk.BOTH, expand=True)
        for gait in self.gaits:
            self.listbox.insert(tk.END, self.gait_display_text(gait))
        if self.gaits:
            self.listbox.selection_set(0)
        self.listbox.bind("<<ListboxSelect>>", self.on_select)

        right = ttk.LabelFrame(body, text="Selected", padding=10)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(12, 0))

        self.info = tk.Text(right, height=10, width=26, wrap=tk.WORD)
        self.info.pack(fill=tk.BOTH, expand=True)
        self.info.configure(state=tk.DISABLED)

        self.update_info()
        self.update_mode()

    def selected_index(self) -> int | None:
        selection = self.listbox.curselection()
        if not selection:
            return None
        return int(selection[0])

    def selected_file(self) -> Path | None:
        idx = self.selected_index()
        if idx is None or idx >= len(self.gaits):
            return None
        return self.gaits[idx]["file"]

    def on_select(self, _event=None):
        self.update_info()

    def update_info(self):
        idx = self.selected_index()
        self.info.configure(state=tk.NORMAL)
        self.info.delete("1.0", tk.END)
        if self.mode.get() == "rectangle":
            self.info.insert(tk.END, "mode: rectangle course\n\n")
            self.info.insert(tk.END, "This opens the waypoint controller:\n")
            self.info.insert(tk.END, "view_rectangle_course.py\n\n")
            self.info.insert(tk.END, "It will swim around the 3 m x 1.5 m rectangle with wall collision.")
        elif self.mode.get() == "step":
            self.info.insert(tk.END, "mode: CPG step test\n\n")
            self.info.insert(tk.END, "This opens:\n")
            self.info.insert(tk.END, "view_cpg_step_change.py\n\n")
            self.info.insert(tk.END, "It repeatedly switches wavelength and amplitude so Hopf and direct sin can be compared.")
        elif idx is None or idx >= len(self.gaits):
            self.info.insert(tk.END, "No gait selected.")
        else:
            gait = self.gaits[idx]
            data = gait["data"]
            self.info.insert(tk.END, f"source: {gait['source_label']}\n")
            self.info.insert(tk.END, f"name: {data.get('name', gait['file'].stem)}\n")
            self.info.insert(tk.END, f"file: {gait['relative']}\n\n")
            self.info.insert(tk.END, f"freq: {data.get('freq')}\n")
            self.info.insert(tk.END, f"ajoint: {data.get('ajoint')} deg\n")
            self.info.insert(tk.END, f"wavelength: {data.get('wavelength')}\n\n")
            self.info.insert(tk.END, "amp_scales:\n")
            self.info.insert(tk.END, ", ".join(str(v) for v in data.get("amp_scales", [])))
            self.info.insert(tk.END, "\n\n")
            self.info.insert(tk.END, "phase_lags:\n")
            self.info.insert(tk.END, ", ".join(str(v) for v in data.get("phase_lags", [])))
            self.info.insert(tk.END, "\n\n")
            self.info.insert(tk.END, "joint_bias:\n")
            self.info.insert(tk.END, ", ".join(str(v) for v in data.get("joint_bias", [])))
            source = data.get("source")
            if isinstance(source, dict):
                self.info.insert(tk.END, "\n\nexport source:\n")
                self.info.insert(tk.END, str(source.get("type", "unknown")))
                if "model" in source:
                    self.info.insert(tk.END, f"\nmodel: {source['model']}")
            self.info.insert(tk.END, "\n\nFixed gait will run once until first wall contact, then print average R in Metrics Window.")
        self.info.configure(state=tk.DISABLED)

    def update_mode(self):
        fixed_enabled = self.mode.get() == "gait"
        self.listbox.configure(state=tk.NORMAL if fixed_enabled else tk.DISABLED)
        self.refresh_button.configure(state=tk.NORMAL if fixed_enabled else tk.DISABLED)
        step_enabled = self.mode.get() == "step"
        self.set_children_state(self.step_options, tk.NORMAL if step_enabled else tk.DISABLED)
        self.step_control_combo.configure(state="readonly" if step_enabled else tk.DISABLED)
        self.update_info()

    def set_children_state(self, widget, state):
        for child in widget.winfo_children():
            try:
                child.configure(state=state)
            except tk.TclError:
                pass
            self.set_children_state(child, state)

    def open_viewer(self):
        if self.process is not None and self.process.poll() is None:
            self.stop_viewer(silent=True)

        if self.mode.get() == "rectangle":
            cmd = [sys.executable, str(ROOT / "view_rectangle_course.py"), "--print-contacts"]
            label = "rectangle course"
        elif self.mode.get() == "step":
            try:
                period = float(self.step_period.get())
                after_wavelength = float(self.step_after_wavelength.get())
                after_amp = float(self.step_after_amp.get())
                alpha = float(self.step_alpha.get())
                k_couple = float(self.step_k_couple.get())
                k_anchor = float(self.step_k_anchor.get())
            except ValueError:
                messagebox.showwarning("Bad value", "Step-test values must be numbers.")
                return
            cmd = [
                sys.executable,
                str(ROOT / "view_cpg_step_change.py"),
                "--control-mode",
                self.step_control_mode.get(),
                "--repeat",
                "--switch-period",
                str(period),
                "--after-wavelength",
                str(after_wavelength),
                "--after-amp-scale",
                str(after_amp),
                "--alpha",
                str(alpha),
                "--k-couple",
                str(k_couple),
                "--k-anchor",
                str(k_anchor),
            ]
            label = f"CPG step test ({self.step_control_mode.get()})"
        else:
            idx = self.selected_index()
            if idx is None:
                messagebox.showwarning("No gait", "Please select a gait first.")
                return
            gait_path = self.gaits[idx]["file"]
            cmd = [
                sys.executable,
                str(ROOT / "view_gait.py"),
                str(gait_path),
                "--print-contacts",
            ]
            label = gait_path.stem
        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            messagebox.showerror("Launch failed", str(exc))
            return

        self.clear_metrics()
        self.append_metric(f"Running: {label}\n")
        self.append_metric("------------------------------------------------------------\n")
        self.reader_thread = threading.Thread(target=self.read_process_output, daemon=True)
        self.reader_thread.start()
        self.status.configure(text=f"Running: {label}")
        self.open_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)

    def read_process_output(self):
        process = self.process
        if process is None or process.stdout is None:
            return
        try:
            for line in process.stdout:
                self.output_queue.put(line)
        except ValueError:
            return

    def stop_viewer(self, silent: bool = False):
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None
        if not silent:
            self.append_metric("\nStopped viewer.\n")
        self.status.configure(text="Stopped" if not silent else "Switching...")
        self.open_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)

    def poll_process(self):
        if self.process is not None and self.process.poll() is not None:
            self.process = None
            self.status.configure(text="Viewer closed")
            self.open_button.configure(state=tk.NORMAL)
            self.stop_button.configure(state=tk.DISABLED)
        self.after(500, self.poll_process)

    def poll_output(self):
        while True:
            try:
                line = self.output_queue.get_nowait()
            except queue.Empty:
                break
            self.append_metric(line)
        self.after(100, self.poll_output)

    def show_metrics_window(self):
        if self.metrics_window is not None and self.metrics_window.winfo_exists():
            self.metrics_window.lift()
            return

        self.metrics_window = tk.Toplevel(self)
        self.metrics_window.title("Eel Metrics")
        self.metrics_window.geometry("760x420")
        self.metrics_window.minsize(560, 320)

        frame = ttk.Frame(self.metrics_window, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        self.metrics_text = tk.Text(frame, wrap=tk.NONE, font=("Consolas", 10))
        self.metrics_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.metrics_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.metrics_text.configure(yscrollcommand=scrollbar.set)
        self.metrics_text.configure(state=tk.DISABLED)

    def clear_metrics(self):
        self.show_metrics_window()
        if self.metrics_text is None:
            return
        self.metrics_text.configure(state=tk.NORMAL)
        self.metrics_text.delete("1.0", tk.END)
        self.metrics_text.configure(state=tk.DISABLED)

    def append_metric(self, text: str):
        self.show_metrics_window()
        if self.metrics_text is None:
            return
        self.metrics_text.configure(state=tk.NORMAL)
        self.metrics_text.insert(tk.END, text)
        self.metrics_text.see(tk.END)
        self.metrics_text.configure(state=tk.DISABLED)

    def refresh_gaits(self):
        selected_file = self.selected_file()
        self.gaits = self.load_gaits()
        self.listbox.delete(0, tk.END)
        for gait in self.gaits:
            self.listbox.insert(tk.END, self.gait_display_text(gait))
        if self.gaits:
            selected_index = 0
            if selected_file is not None:
                for idx, gait in enumerate(self.gaits):
                    if gait["file"] == selected_file:
                        selected_index = idx
                        break
            self.listbox.selection_set(selected_index)
            self.listbox.activate(selected_index)
        self.update_info()

    def on_close(self):
        self.stop_viewer()
        self.destroy()


def main():
    app = GaitGui()
    app.mainloop()


if __name__ == "__main__":
    main()
