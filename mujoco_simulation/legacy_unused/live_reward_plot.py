from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                rows.append(
                    {
                        "timesteps": float(row["timesteps"]),
                        "mean_reward": float(row["mean_reward"]),
                        "mean_vx": float(row["mean_vx"]),
                        "mean_frequency": float(row.get("mean_frequency", "nan")),
                    }
                )
            except (KeyError, ValueError):
                continue
    return rows


def write_plot(csv_path: Path, png_path: Path, title: str, target_vx: float | None) -> None:
    rows = read_rows(csv_path)
    if not rows:
        return
    steps = [row["timesteps"] for row in rows]
    rewards = [row["mean_reward"] for row in rows]
    vxs = [row["mean_vx"] for row in rows]

    fig, ax_reward = plt.subplots(figsize=(10, 5.5), dpi=150)
    ax_reward.plot(steps, rewards, color="#1f77b4", linewidth=2.6, label="Eval mean reward")
    ax_reward.set_xlabel("Training steps")
    ax_reward.set_ylabel("Eval mean reward")
    ax_reward.grid(True, alpha=0.28)

    ax_vx = ax_reward.twinx()
    ax_vx.plot(steps, vxs, color="#d62728", linewidth=2.0, alpha=0.78, label="mean vx")
    if target_vx is not None:
        ax_vx.axhline(target_vx, color="#d62728", linestyle="--", linewidth=1.5, alpha=0.45, label=f"target vx = {target_vx:g}")
    ax_vx.set_ylabel("mean vx (m/s)")

    lines = ax_reward.get_lines() + ax_vx.get_lines()
    ax_reward.legend(lines, [line.get_label() for line in lines], loc="best")
    ax_reward.set_title(title)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path)
    plt.close(fig)


def write_html(html_path: Path, png_path: Path, title: str) -> None:
    image_name = png_path.name
    html_path.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f7f7f7; color: #222; }}
    header {{ padding: 10px 16px; background: #fff; border-bottom: 1px solid #ddd; }}
    img {{ display: block; width: min(96vw, 1200px); margin: 16px auto; background: #fff; box-shadow: 0 1px 5px rgba(0,0,0,.12); }}
    .small {{ color: #666; font-size: 13px; }}
  </style>
</head>
<body>
  <header>
    <div>{title}</div>
    <div class="small">Auto refresh every 5 seconds. Last refresh: <span id="time"></span></div>
  </header>
  <img id="plot" src="{image_name}" alt="live reward plot">
  <script>
    const img = document.getElementById("plot");
    const time = document.getElementById("time");
    function refresh() {{
      img.src = "{image_name}?t=" + Date.now();
      time.textContent = new Date().toLocaleTimeString();
    }}
    refresh();
    setInterval(refresh, 5000);
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--status", type=Path, default=None)
    parser.add_argument("--png", type=Path, required=True)
    parser.add_argument("--html", type=Path, default=None)
    parser.add_argument("--title", default="Live eval reward")
    parser.add_argument("--target-vx", type=float, default=None)
    args = parser.parse_args()

    csv_path = args.csv
    title = args.title
    if args.status is not None:
        status = json.loads(args.status.read_text(encoding="utf-8"))
        current = status.get("current") or status.get("failed") or {}
        csv_text = current.get("eval_debug_csv")
        if csv_text:
            csv_path = Path(csv_text)
        name = current.get("name")
        run_idx = current.get("run_idx")
        total_runs = status.get("total_runs")
        if name:
            suffix = f"run {run_idx}/{total_runs}" if run_idx and total_runs else str(name)
            title = f"{args.title} ({suffix})"
    if csv_path is None:
        raise ValueError("Provide --csv or --status with current.eval_debug_csv")

    write_plot(csv_path, args.png, title, args.target_vx)
    if args.html is not None:
        write_html(args.html, args.png, args.title)


if __name__ == "__main__":
    main()
