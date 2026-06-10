from __future__ import annotations

import json
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from make_tracked_center_cleaned_physical import (
    DEFAULT_OUT_ROOT,
    DEFAULT_PREVIEW_NAME,
    DEFAULT_PX_PER_M,
    DEFAULT_RECORDINGS_DIR,
    DEFAULT_VIDEO_STEMS,
    process_video,
    resolve_path,
    resolve_recordings_dir,
    resolve_video,
)


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
        self.root.title("Robot Eel Real Video Analysis - Legacy Tracker")
        self.root.geometry("900x640")

        self.recordings_var = tk.StringVar(value=str(DEFAULT_RECORDINGS_DIR))
        self.video_var = tk.StringVar()
        self.out_var = tk.StringVar(value=str(DEFAULT_OUT_ROOT))
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

        out_row = ttk.Frame(outer)
        out_row.pack(fill=tk.X, pady=4)
        ttk.Label(out_row, text="Output root").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Entry(out_row, textvariable=self.out_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(out_row, text="Browse", command=self.browse_output).pack(side=tk.LEFT, padx=(8, 0))

        options = ttk.LabelFrame(outer, text="Analysis options", padding=10)
        options.pack(fill=tk.X, pady=10)
        ttk.Label(options, text="px_per_m").grid(row=0, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Entry(options, textvariable=self.px_per_m_var, width=14).grid(row=0, column=1, sticky=tk.W, padx=4, pady=4)
        ttk.Checkbutton(options, text=f"Write preview PNG ({DEFAULT_PREVIEW_NAME})", variable=self.preview_var).grid(row=0, column=2, sticky=tk.W, padx=12, pady=4)
        ttk.Label(
            options,
            text="Uses the same legacy start-to-wall tracker that produced the correct R values. Straight speed uses fitted-line forward displacement only.",
        ).grid(row=1, column=0, columnspan=3, sticky=tk.W, padx=4, pady=4)

        button_row = ttk.Frame(outer)
        button_row.pack(fill=tk.X, pady=8)
        self.single_button = ttk.Button(button_row, text="Analyze selected MP4", command=self.start_single)
        self.single_button.pack(side=tk.LEFT)
        self.batch_button = ttk.Button(button_row, text="Analyze default 3 videos", command=self.start_batch)
        self.batch_button.pack(side=tk.LEFT, padx=8)
        ttk.Button(button_row, text="Open output folder", command=self.open_output_folder).pack(side=tk.LEFT, padx=8)

        self.log = tk.Text(outer, height=22, wrap=tk.WORD)
        self.log.pack(fill=tk.BOTH, expand=True)
        self.logger = TextLogger(self.log)
        self.logger.write("Legacy real-video tracker GUI ready.\n")
        self.logger.write("Default 3 videos:\n")
        for stem in DEFAULT_VIDEO_STEMS:
            self.logger.write(f"  - {stem}.mp4\n")
        self.logger.write("\nOutputs per video:\n")
        self.logger.write("  tracked_center_summary_cleaned_physical.json\n")
        self.logger.write("  tracked_center_overlay_cleaned_physical.png\n")

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

    def browse_output(self) -> None:
        dirname = filedialog.askdirectory(title="Select output folder")
        if dirname:
            self.out_var.set(dirname)

    def open_output_folder(self) -> None:
        path = Path(self.out_var.get()).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        try:
            import os

            os.startfile(path)  # type: ignore[attr-defined]
        except Exception as exc:
            messagebox.showerror("Open folder failed", str(exc))

    def set_buttons(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        self.single_button.configure(state=state)
        self.batch_button.configure(state=state)

    def start_single(self) -> None:
        video = self.video_var.get().strip()
        if not video:
            messagebox.showerror("Missing MP4", "Please select one MP4 file first.")
            return
        self.set_buttons(False)
        thread = threading.Thread(target=self._worker, args=([Path(video)],), daemon=True)
        thread.start()

    def start_batch(self) -> None:
        self.set_buttons(False)
        recordings_dir = resolve_recordings_dir(Path(self.recordings_var.get()).expanduser())
        videos = [resolve_video(f"{stem}.mp4", recordings_dir) for stem in DEFAULT_VIDEO_STEMS]
        thread = threading.Thread(target=self._worker, args=(videos,), daemon=True)
        thread.start()

    def _worker(self, videos: list[Path]) -> None:
        try:
            out_root = resolve_path(Path(self.out_var.get()).expanduser())
            out_root.mkdir(parents=True, exist_ok=True)
            px_per_m = float(self.px_per_m_var.get())
            write_preview = bool(self.preview_var.get())

            self.logger.write("\n=== Analysis started ===\n")
            self.logger.write(f"out_root={out_root}\n")
            self.logger.write(f"px_per_m={px_per_m:.6f}\n")

            for video_path in videos:
                video_path = video_path.expanduser().resolve()
                self.logger.write(f"\nVideo: {video_path}\n")
                if not video_path.exists():
                    self.logger.write("  SKIP: file not found\n")
                    continue

                process_video(
                    video_path=video_path,
                    out_root=out_root,
                    write_preview=write_preview,
                    px_per_m=px_per_m,
                )
                summary_path = out_root / video_path.stem / "tracked_center_summary_cleaned_physical.json"
                result = json.loads(summary_path.read_text(encoding="utf-8"))
                self._log_result(result)

            self.logger.write("\n=== Done ===\n")
        except Exception as exc:
            self.logger.write(f"\nERROR: {exc}\n")
            self.root.after(0, lambda: messagebox.showerror("Analysis failed", str(exc)))
        finally:
            self.root.after(0, lambda: self.set_buttons(True))

    def _log_result(self, result: dict) -> None:
        self.logger.write(f"  points={result.get('point_count')} fit={result.get('fit_kind')}\n")
        if result.get("fit_kind") == "circle":
            radius_px = result.get("radius_px")
            radius_m = result.get("radius_m")
            self.logger.write(
                f"  R={radius_px:.3f}px = {radius_m:.4f}m, arc={result.get('arc_deg'):.3f}deg, rmse={result.get('rmse_px'):.3f}px\n"
            )
        else:
            speed = result.get("forward_speed_m_s")
            speed_text = "nan" if speed is None else f"{speed:.4f}m/s"
            self.logger.write(
                f"  forward={result.get('forward_distance_m'):.4f}m, speed={speed_text}, line_rmse={result.get('rmse_px'):.3f}px\n"
            )
            self.logger.write("  speed source: fitted-line vertical forward displacement, not left-right drift.\n")
        self.logger.write(f"  JSON: {out_or_blank(result.get('video'), result)}\n")
        preview = result.get("preview_image")
        if preview:
            self.logger.write(f"  Preview: {preview}\n")


def out_or_blank(_video: str | None, result: dict) -> str:
    video_stem = result.get("video_stem")
    if not video_stem:
        return ""
    return str(Path(result.get("preview_image", "")).parent / "tracked_center_summary_cleaned_physical.json")


def main() -> None:
    root = tk.Tk()
    RealVideoAnalysisApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
