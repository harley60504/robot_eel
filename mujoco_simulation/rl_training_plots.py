from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def model_output_stem(output_path: Path) -> str:
    """Return a stable model name even when the user passes a .zip path."""
    output_path = Path(output_path)
    return output_path.stem if output_path.suffix == ".zip" else output_path.name


def default_eval_log_dir(output_path: Path) -> Path:
    output_path = Path(output_path)
    return output_path.parent / f"{model_output_stem(output_path)}_eval"


def default_plot_path(output_path: Path) -> Path:
    output_path = Path(output_path)
    return output_path.parent / f"{model_output_stem(output_path)}_eval_reward.png"


def evaluation_file(eval_log_dir: Path) -> Path:
    return Path(eval_log_dir) / "evaluations.npz"


def load_eval_series(eval_npz: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Stable-Baselines3 EvalCallback results.

    Returns timesteps, mean reward, std reward, min reward, and max reward.
    """
    eval_npz = Path(eval_npz)
    data = np.load(eval_npz)
    timesteps = np.asarray(data["timesteps"], dtype=np.float64)
    results = np.asarray(data["results"], dtype=np.float64)
    if results.ndim == 1:
        results = results[:, None]
    mean_reward = np.mean(results, axis=1)
    std_reward = np.std(results, axis=1)
    min_reward = np.min(results, axis=1)
    max_reward = np.max(results, axis=1)
    return timesteps, mean_reward, std_reward, min_reward, max_reward


def save_eval_csv(eval_npz: Path, csv_path: Path) -> Path:
    timesteps, mean_reward, std_reward, min_reward, max_reward = load_eval_series(eval_npz)
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table = np.column_stack((timesteps, mean_reward, std_reward, min_reward, max_reward))
    np.savetxt(
        csv_path,
        table,
        delimiter=",",
        header="timesteps,mean_reward,std_reward,min_reward,max_reward",
        comments="",
    )
    return csv_path


def _import_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_eval_curve(
    eval_npz: Path,
    output_png: Path,
    *,
    label: str | None = None,
    title: str = "Eval mean reward over training steps",
    save_csv: bool = True,
) -> Path:
    """Plot one EvalCallback evaluations.npz file."""
    eval_npz = Path(eval_npz)
    if not eval_npz.exists():
        raise FileNotFoundError(f"evaluation file not found: {eval_npz}")

    timesteps, mean_reward, std_reward, _, _ = load_eval_series(eval_npz)
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(8, 5))
    curve_label = label or eval_npz.parent.name
    ax.plot(timesteps, mean_reward, label=curve_label)
    ax.fill_between(timesteps, mean_reward - std_reward, mean_reward + std_reward, alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Eval mean reward")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_png, dpi=200)
    plt.close(fig)

    if save_csv:
        save_eval_csv(eval_npz, output_png.with_suffix(".csv"))
    return output_png


def plot_many_eval_curves(
    eval_npz_paths: Iterable[Path],
    output_png: Path,
    *,
    labels: Iterable[str] | None = None,
    title: str = "Eval mean reward over training steps",
    save_csv: bool = True,
) -> Path:
    """Plot several EvalCallback evaluations.npz files on one figure."""
    eval_paths = [Path(path) for path in eval_npz_paths]
    if not eval_paths:
        raise ValueError("at least one evaluation file is required")
    for eval_path in eval_paths:
        if not eval_path.exists():
            raise FileNotFoundError(f"evaluation file not found: {eval_path}")

    if labels is None:
        label_list = [path.parent.name for path in eval_paths]
    else:
        label_list = list(labels)
        if len(label_list) != len(eval_paths):
            raise ValueError("number of labels must match number of evaluation files")

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(9, 5))
    csv_columns = []
    csv_header_parts = []

    for eval_path, label in zip(eval_paths, label_list):
        timesteps, mean_reward, std_reward, _, _ = load_eval_series(eval_path)
        ax.plot(timesteps, mean_reward, label=label)
        ax.fill_between(timesteps, mean_reward - std_reward, mean_reward + std_reward, alpha=0.15)
        if save_csv:
            csv_columns.extend([timesteps, mean_reward, std_reward])
            safe_label = label.replace(",", "_").replace(" ", "_")
            csv_header_parts.extend(
                [
                    f"{safe_label}_timesteps",
                    f"{safe_label}_mean_reward",
                    f"{safe_label}_std_reward",
                ]
            )

    ax.set_title(title)
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Eval mean reward")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_png, dpi=200)
    plt.close(fig)

    if save_csv:
        min_len = min(len(column) for column in csv_columns)
        table = np.column_stack([column[:min_len] for column in csv_columns])
        np.savetxt(
            output_png.with_suffix(".csv"),
            table,
            delimiter=",",
            header=",".join(csv_header_parts),
            comments="",
        )
    return output_png


def try_plot_eval_curve(eval_log_dir: Path, output_png: Path, *, label: str | None = None) -> None:
    """Best-effort plotting so training still succeeds if matplotlib is missing."""
    eval_npz = evaluation_file(eval_log_dir)
    try:
        plot_eval_curve(eval_npz, output_png, label=label)
        print(f"saved eval reward plot to {output_png}")
        print(f"saved eval reward csv to {Path(output_png).with_suffix('.csv')}")
    except Exception as exc:  # pragma: no cover - helper should not break training
        print(f"could not plot eval reward curve: {exc}")
