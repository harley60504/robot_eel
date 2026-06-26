from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor

from hopf_cpg import degrees_to_radians
from plot_fitted_gait_curves import add_sim_metric_box, draw_rotated_tank, rotate_sim_xy, sim_metric_text, trajectory_metrics
from rl_free_swim_env import EelFreeSwimRLEnv, FreeSwimConfig
from rl_training_plots import default_eval_log_dir, default_plot_path, try_plot_eval_curve


DETAILED_EVAL_FIELDS = [
    "timesteps",
    "mean_reward",
    "std_reward",
    "mean_ep_len",
    "mean_vx",
    "mean_vy",
    "mean_abs_y",
    "mean_abs_yaw",
    "mean_energy_proxy",
    "mean_frequency",
    "mean_phase_lag_1",
    "mean_phase_lag_2",
    "mean_phase_lag_3",
    "mean_phase_lag_4",
    "mean_phase_lag_5",
]


def fit_straight_line_curve(xy: np.ndarray, count: int = 240) -> tuple[np.ndarray, dict]:
    center = np.mean(xy, axis=0)
    shifted = xy - center
    if xy.shape[0] < 2 or np.linalg.norm(shifted) < 1e-12:
        curve = np.repeat(xy[:1], count, axis=0)
        return curve, {"kind": "line", "radius": None, "rmse": 0.0, "arc_deg": 0.0}
    _, _, vh = np.linalg.svd(shifted, full_matrices=False)
    direction = vh[0]
    if direction[1] < 0.0:
        direction = -direction
    projection = shifted @ direction
    curve = center + np.outer(np.linspace(float(projection.min()), float(projection.max()), count), direction)
    perpendicular = shifted - np.outer(projection, direction)
    rmse = float(np.sqrt(np.mean(np.sum(perpendicular * perpendicular, axis=1))))
    return curve, {"kind": "line", "radius": None, "rmse": rmse, "arc_deg": 0.0}


class DetailedFreeSwimEvalMetricsCallback(BaseCallback):
    """Write deterministic eval trajectory plots for the best training checkpoint."""

    def __init__(
        self,
        cfg: FreeSwimConfig,
        csv_path: Path,
        eval_freq: int,
        n_eval_episodes: int,
        best_episode_dir: Path | None = None,
        run_name: str | None = None,
    ):
        super().__init__(verbose=0)
        self.cfg = cfg
        self.csv_path = Path(csv_path)
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.best_episode_dir = None if best_episode_dir is None else Path(best_episode_dir)
        self.run_name = run_name or self.csv_path.stem.removesuffix("_eval_debug")
        self.best_mean_reward = -np.inf

    def _on_training_start(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if self.best_episode_dir is not None:
            self.best_episode_dir.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=DETAILED_EVAL_FIELDS)
                writer.writeheader()

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True
        row, best_episode = self._evaluate()
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=DETAILED_EVAL_FIELDS)
            writer.writerow(row)
        mean_reward = float(row.get("mean_reward", float("nan")))
        if self.best_episode_dir is not None and np.isfinite(mean_reward) and mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            self._write_best_episode_outputs(best_episode, row)
        return True

    def _evaluate(self) -> tuple[dict[str, float], dict[str, object]]:
        episode_rewards: list[float] = []
        episode_lengths: list[int] = []
        step_infos: list[dict] = []
        best_episode: dict[str, object] = {"reward": -np.inf, "records": []}

        for _ in range(self.n_eval_episodes):
            env = EelFreeSwimRLEnv(self.cfg)
            obs, _ = env.reset()
            total_reward = 0.0
            length = 0
            terminated = truncated = False
            records: list[list[float]] = []
            while not (terminated or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                length += 1
                physical_action = np.asarray(info.get("physical_action", np.full(6, np.nan)), dtype=np.float64)
                records.append(
                    [
                        float(env.data.time),
                        float(info.get("x", np.nan)),
                        float(info.get("y", np.nan)),
                        float(info.get("yaw", np.nan)),
                        float(reward),
                        float(info.get("velocity_x", np.nan)),
                        float(info.get("velocity_y", np.nan)),
                        *[float(value) for value in physical_action[:6]],
                    ]
                )
                if info.get("steady_state"):
                    step_infos.append(info)
            if total_reward > float(best_episode["reward"]):
                best_episode = {"reward": total_reward, "length": length, "records": records}
            episode_rewards.append(total_reward)
            episode_lengths.append(length)

        def mean_info(key: str, *, abs_value: bool = False) -> float:
            values = []
            for info in step_infos:
                if key not in info:
                    continue
                try:
                    value = float(info[key])
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    values.append(abs(value) if abs_value else value)
            return float(np.mean(values)) if values else float("nan")

        actions = [
            np.asarray(info.get("physical_action"), dtype=np.float64)
            for info in step_infos
            if info.get("physical_action") is not None
        ]
        mean_action = np.mean(np.asarray(actions, dtype=np.float64), axis=0) if actions else np.full(6, np.nan)
        return {
            "timesteps": int(self.num_timesteps),
            "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else float("nan"),
            "std_reward": float(np.std(episode_rewards)) if episode_rewards else float("nan"),
            "mean_ep_len": float(np.mean(episode_lengths)) if episode_lengths else float("nan"),
            "mean_vx": mean_info("velocity_x"),
            "mean_vy": mean_info("velocity_y"),
            "mean_abs_y": mean_info("y", abs_value=True),
            "mean_abs_yaw": mean_info("yaw", abs_value=True),
            "mean_energy_proxy": mean_info("energy_proxy"),
            "mean_frequency": float(mean_action[0]),
            "mean_phase_lag_1": float(mean_action[1]),
            "mean_phase_lag_2": float(mean_action[2]),
            "mean_phase_lag_3": float(mean_action[3]),
            "mean_phase_lag_4": float(mean_action[4]),
            "mean_phase_lag_5": float(mean_action[5]),
        }, best_episode

    def _write_best_episode_outputs(self, best_episode: dict[str, object], row: dict[str, float]) -> None:
        records = np.asarray(best_episode.get("records", []), dtype=np.float64)
        if records.ndim != 2 or records.shape[0] < 2:
            return
        assert self.best_episode_dir is not None
        csv_path = self.best_episode_dir / f"{self.run_name}_eval_best_policy_trajectory.csv"
        png_path = self.best_episode_dir / f"sim_{self.run_name}_eval_best_policy_fitted_rotated.png"
        summary_path = self.best_episode_dir / f"{self.run_name}_eval_best_policy_summary.json"
        try:
            np.savetxt(
                csv_path,
                records,
                delimiter=",",
                header="time,x,y,yaw,reward,velocity_x,velocity_y,frequency,phase_lag_1,phase_lag_2,phase_lag_3,phase_lag_4,phase_lag_5",
                comments="",
            )
        except PermissionError:
            suffix = f"_step{int(row['timesteps'])}"
            csv_path = self.best_episode_dir / f"{self.run_name}_eval_best_policy_trajectory{suffix}.csv"
            png_path = self.best_episode_dir / f"sim_{self.run_name}_eval_best_policy_fitted_rotated{suffix}.png"
            summary_path = self.best_episode_dir / f"{self.run_name}_eval_best_policy_summary{suffix}.json"
            np.savetxt(
                csv_path,
                records,
                delimiter=",",
                header="time,x,y,yaw,reward,velocity_x,velocity_y,frequency,phase_lag_1,phase_lag_2,phase_lag_3,phase_lag_4,phase_lag_5",
                comments="",
            )

        arr = records[:, :4]
        xy = rotate_sim_xy(arr[:, 1:3])
        curve, fit = fit_straight_line_curve(xy)
        metrics = trajectory_metrics(arr, xy)
        fig, ax = plt.subplots(figsize=(4.8, 8.0), dpi=170)
        draw_rotated_tank(ax)
        color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
        ax.plot(curve[:, 0], curve[:, 1], color=color, linewidth=3.0)
        ax.scatter([xy[0, 0]], [xy[0, 1]], s=34, color=color, edgecolor="black", zorder=4)
        ax.scatter([xy[-1, 0]], [xy[-1, 1]], s=52, marker="x", color=color, linewidth=2.2, zorder=4)
        ax.set_title(f"{self.run_name} eval best policy")
        add_sim_metric_box(ax, sim_metric_text("straight", fit, metrics))
        fig.tight_layout()
        fig.savefig(png_path)
        plt.close(fig)

        summary = {
            "name": self.run_name,
            "trajectory_csv": str(csv_path),
            "eval_best_policy_png": str(png_path),
            "eval_best_selection": "highest_mean_reward",
            "eval_mean_reward": float(row["mean_reward"]),
            "eval_episode_reward": float(best_episode.get("reward", float("nan"))),
            **fit,
            **metrics,
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_float_list(value: str, expected_len: int, name: str) -> tuple[float, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != expected_len:
        raise argparse.ArgumentTypeError(f"{name} needs {expected_len} comma-separated values")
    return tuple(float(part) for part in parts)


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on free-swim forward velocity.")
    parser.add_argument("--timesteps", type=int, default=150_000)
    parser.add_argument("--output", type=Path, default=Path("outputs/zips/ppo_free_swim_freq_phase"))
    parser.add_argument("--load-model", type=Path, default=None)
    parser.add_argument("--episode-seconds", type=float, default=None)
    parser.add_argument("--warmup-seconds", type=float, default=None)
    parser.add_argument("--freq", type=float, default=None, help="Initial/default fixed frequency value; action bounds decide learned frequency.")
    parser.add_argument("--freq-low", type=float, default=None)
    parser.add_argument("--freq-high", type=float, default=None)
    parser.add_argument("--wavelength", type=float, default=None)
    parser.add_argument("--ajoint", type=float, default=None, help="Base joint angle amplitude in degrees.")
    parser.add_argument("--fixed-amp-scales", type=lambda value: parse_float_list(value, 6, "fixed-amp-scales"), default=None)
    parser.add_argument("--phase-lag-low", type=float, default=None)
    parser.add_argument("--phase-lag-high", type=float, default=None)
    parser.add_argument("--target-speed", type=float, default=None)
    parser.add_argument("--speed-tolerance", type=float, default=None)
    parser.add_argument("--energy-weight", type=float, default=None)
    parser.add_argument("--reward-average-seconds", type=float, default=None)
    parser.add_argument("--boundary-x-min", type=float, default=None)
    parser.add_argument("--boundary-x-max", type=float, default=None)
    parser.add_argument("--boundary-y", type=float, default=None)
    parser.add_argument("--eval-freq", type=int, default=10_000, help="Evaluate every N training steps. Use 0 to disable.")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Episodes per evaluation.")
    parser.add_argument("--eval-log-dir", type=Path, default=None, help="Directory for evaluations.npz.")
    parser.add_argument("--plot-output", type=Path, default=None, help="PNG path for eval reward curve.")
    parser.add_argument("--no-plot", action="store_true", help="Do not create a PNG/CSV plot after training.")
    return parser.parse_args()


def config_from_args(args) -> FreeSwimConfig:
    cfg = FreeSwimConfig()
    if args.episode_seconds is not None:
        cfg.episode_seconds = args.episode_seconds
    if args.warmup_seconds is not None:
        cfg.warmup_seconds = args.warmup_seconds
    if args.freq is not None:
        cfg.fixed_frequency = args.freq
    if args.freq_low is not None:
        cfg.frequency_low = args.freq_low
    if args.freq_high is not None:
        cfg.frequency_high = args.freq_high
    if args.wavelength is not None:
        cfg.fixed_wavelength = args.wavelength
    if args.ajoint is not None:
        cfg.fixed_ajoint = degrees_to_radians(args.ajoint)
    if args.fixed_amp_scales is not None:
        cfg.fixed_amp_scales = args.fixed_amp_scales
    if args.phase_lag_low is not None:
        cfg.phase_lag_low = args.phase_lag_low
    if args.phase_lag_high is not None:
        cfg.phase_lag_high = args.phase_lag_high
    if args.target_speed is not None:
        cfg.target_speed = args.target_speed
    if args.speed_tolerance is not None:
        cfg.speed_tolerance = args.speed_tolerance
    if args.energy_weight is not None:
        cfg.energy_weight = args.energy_weight
    if args.reward_average_seconds is not None:
        cfg.reward_average_seconds = args.reward_average_seconds
    if args.boundary_x_min is not None:
        cfg.boundary_x_min = args.boundary_x_min
    if args.boundary_x_max is not None:
        cfg.boundary_x_max = args.boundary_x_max
    if args.boundary_y is not None:
        cfg.boundary_y = abs(args.boundary_y)
    return cfg


def debug_csv_path(plot_output: Path) -> Path:
    plot_output = Path(plot_output)
    stem = plot_output.stem
    if stem.endswith("_eval_reward"):
        stem = stem[: -len("_eval_reward")] + "_eval_debug"
    else:
        stem = f"{stem}_debug"
    return plot_output.with_name(f"{stem}.csv")


def eval_best_policy_dir(plot_output: Path) -> Path:
    return Path(plot_output).parent / "eval_best_policy_curves"


def make_eval_callback(args, cfg: FreeSwimConfig) -> tuple[BaseCallback | None, Path | None, Path | None, Path | None]:
    if args.eval_freq <= 0:
        return None, None, None, None
    eval_log_dir = args.eval_log_dir or default_eval_log_dir(args.output)
    plot_output = args.plot_output or default_plot_path(args.output)
    debug_output = debug_csv_path(plot_output)
    eval_log_dir.mkdir(parents=True, exist_ok=True)
    eval_env = Monitor(EelFreeSwimRLEnv(cfg))
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(eval_log_dir / "best_model"),
        log_path=str(eval_log_dir),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )
    debug_callback = DetailedFreeSwimEvalMetricsCallback(
        cfg,
        debug_output,
        args.eval_freq,
        args.eval_episodes,
        best_episode_dir=eval_best_policy_dir(plot_output),
        run_name=args.output.stem,
    )
    return CallbackList([eval_callback, debug_callback]), eval_log_dir, plot_output, debug_output


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cfg = config_from_args(args)
    env = Monitor(EelFreeSwimRLEnv(cfg))
    callback, eval_log_dir, plot_output, debug_output = make_eval_callback(args, cfg)
    if args.load_model is None:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            n_steps=1024,
            batch_size=256,
            gamma=0.98,
            learning_rate=1e-4,
            ent_coef=0.005,
        )
        reset_num_timesteps = True
    else:
        model = PPO.load(args.load_model, env=env)
        model.verbose = 1
        reset_num_timesteps = False
    model.learn(total_timesteps=args.timesteps, reset_num_timesteps=reset_num_timesteps, callback=callback)
    model.save(args.output)
    print(f"saved policy to {args.output}.zip")
    if callback is not None and eval_log_dir is not None and plot_output is not None and not args.no_plot:
        try_plot_eval_curve(eval_log_dir, plot_output, label=args.output.name)
    if debug_output is not None and debug_output.exists():
        print(f"saved eval debug csv to {debug_output}")


if __name__ == "__main__":
    main()
