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
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from hopf_cpg import degrees_to_radians
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
from rl_turning_env import EelTurningRLEnv, TurningConfig, direction_sign
from rl_robot_imu_env import EelTurningRobotImuEnv, TurningRobotImuConfig
from rl_training_plots import default_eval_log_dir, default_plot_path, try_plot_eval_curve
from train_free_swim_rl import parse_float_list


DETAILED_EVAL_FIELDS = [
    "timesteps",
    "mean_reward",
    "std_reward",
    "mean_ep_len",
    "yaw_rate_weight",
    "radius_weight",
    "correct_turn_direction_rate",
    "mean_yaw_rate",
    "mean_body_yaw_rate",
    "mean_trajectory_yaw_rate",
    "mean_reward_yaw_rate_value",
    "mean_target_yaw_rate",
    "mean_yaw_rate_error",
    "mean_turn_radius",
    "mean_signed_turn_radius",
    "mean_signed_target_radius",
    "mean_radius_error",
    "mean_speed",
    "mean_body_speed",
    "mean_reward_yaw_rate",
    "mean_reward_radius",
    "mean_episode_reward_yaw_rate",
    "mean_episode_reward_radius",
    "mean_x",
    "mean_y",
]


class DetailedEvalMetricsCallback(BaseCallback):#紀錄訓練
    """Write reward-component eval diagnostics that EvalCallback's npz omits."""

    def __init__(
        self,
        cfg: TurningConfig,
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
        episode_reward_yaw_rate: list[float] = []
        episode_reward_radius: list[float] = []
        step_infos: list[dict] = []
        best_episode: dict[str, object] = {"reward": -np.inf, "records": []}

        for _ in range(self.n_eval_episodes):
            env_cls = EelTurningRobotImuEnv if isinstance(self.cfg, TurningRobotImuConfig) else EelTurningRLEnv
            env = env_cls(self.cfg)
            obs, _ = env.reset()
            total_reward = 0.0
            total_reward_yaw_rate = 0.0
            total_reward_radius = 0.0
            length = 0
            terminated = truncated = False
            records: list[list[float]] = []
            while not (terminated or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                length += 1
                records.append(
                    [
                        float(env.data.time),
                        float(info.get("x", np.nan)),
                        float(info.get("y", np.nan)),
                        float(info.get("yaw", np.nan)),
                        float(reward),
                        float(info.get("yaw_rate", np.nan)),
                        float(info.get("turn_radius", np.nan)),
                    ]
                )
                if info.get("steady_state"):
                    step_infos.append(info)
                    total_reward_yaw_rate += float(info.get("reward_yaw_rate", 0.0))
                    total_reward_radius += float(info.get("reward_radius", 0.0))
            if total_reward > float(best_episode["reward"]):
                best_episode = {"reward": total_reward, "length": length, "records": records}
            episode_rewards.append(total_reward)
            episode_lengths.append(length)
            episode_reward_yaw_rate.append(total_reward_yaw_rate)
            episode_reward_radius.append(total_reward_radius)

        def mean_info(key: str) -> float:
            values = []
            for info in step_infos:
                if key not in info:
                    continue
                try:
                    value = float(info[key])
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    values.append(value)
            return float(np.mean(values)) if values else float("nan")

        return {
            "timesteps": int(self.num_timesteps),
            "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else float("nan"),
            "std_reward": float(np.std(episode_rewards)) if episode_rewards else float("nan"),
            "mean_ep_len": float(np.mean(episode_lengths)) if episode_lengths else float("nan"),
            "yaw_rate_weight": float(self.cfg.yaw_rate_weight),
            "radius_weight": float(self.cfg.radius_weight),
            "correct_turn_direction_rate": mean_info("correct_turn_direction"),
            "mean_yaw_rate": mean_info("yaw_rate"),
            "mean_body_yaw_rate": mean_info("body_yaw_rate"),
            "mean_trajectory_yaw_rate": mean_info("trajectory_yaw_rate"),
            "mean_reward_yaw_rate_value": mean_info("reward_yaw_rate_value"),
            "mean_target_yaw_rate": mean_info("target_yaw_rate"),
            "mean_yaw_rate_error": mean_info("yaw_rate_error"),
            "mean_turn_radius": mean_info("turn_radius"),
            "mean_signed_turn_radius": mean_info("signed_turn_radius"),
            "mean_signed_target_radius": mean_info("signed_target_radius"),
            "mean_radius_error": mean_info("radius_error"),
            "mean_speed": mean_info("speed"),
            "mean_body_speed": mean_info("body_speed"),
            "mean_reward_yaw_rate": mean_info("reward_yaw_rate"),
            "mean_reward_radius": mean_info("reward_radius"),
            "mean_episode_reward_yaw_rate": float(np.mean(episode_reward_yaw_rate)) if episode_reward_yaw_rate else float("nan"),
            "mean_episode_reward_radius": float(np.mean(episode_reward_radius)) if episode_reward_radius else float("nan"),
            "mean_x": mean_info("x"),
            "mean_y": mean_info("y"),
        }, best_episode

    def _write_best_episode_outputs(self, best_episode: dict[str, object], row: dict[str, float]) -> None:
        records = np.asarray(best_episode.get("records", []), dtype=np.float64)
        if records.ndim != 2 or records.shape[0] < 2:
            return
        assert self.best_episode_dir is not None
        csv_path = self.best_episode_dir / f"{self.run_name}_eval_best_policy_trajectory.csv"
        png_path = self.best_episode_dir / f"sim_{self.run_name}_eval_best_policy_fitted_rotated.png"
        summary_path = self.best_episode_dir / f"{self.run_name}_eval_best_policy_summary.json"
        np.savetxt(
            csv_path,
            records,
            delimiter=",",
            header="time,x,y,yaw,reward,yaw_rate,turn_radius",
            comments="",
        )

        xy = rotate_sim_xy(records[:, 1:3])
        curve, fit = fitted_curve(xy)
        metrics = trajectory_metrics(records[:, :4], xy)
        target_yaw = None if self.cfg.yaw_rate_weight == 0.0 else direction_sign(self.cfg.turn_direction) * abs(float(self.cfg.target_yaw_rate))
        target_radius = None if self.cfg.radius_weight == 0.0 else self.cfg.target_radius
        metrics.update(add_fitted_yaw_rate_metrics(fit, metrics, target_yaw, self.cfg.turn_direction))
        metrics.update(add_fitted_radius_metrics(fit, target_radius))
        metrics.update(
            {
                "turn_direction": self.cfg.turn_direction,
                "yaw_rate_reward_weight": float(self.cfg.yaw_rate_weight),
                "radius_reward_weight": float(self.cfg.radius_weight),
                "timesteps": int(row["timesteps"]),
                "eval_best_selection": "highest_mean_reward",
                "eval_mean_reward": float(row["mean_reward"]),
                "eval_episode_reward": float(best_episode.get("reward", float("nan"))),
                "mean_step_reward": float(np.nanmean(records[:, 4])),
                "mean_env_yaw_rate": float(np.nanmean(records[:, 5])),
                "mean_env_turn_radius": float(np.nanmean(records[:, 6])),
            }
        )

        fig, ax = plt.subplots(figsize=(4.8, 8.0), dpi=170)
        draw_rotated_tank(ax)
        color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
        ax.plot(curve[:, 0], curve[:, 1], color=color, linewidth=3.0)
        ax.scatter([xy[0, 0]], [xy[0, 1]], s=34, color=color, edgecolor="black", zorder=4)
        ax.scatter([xy[-1, 0]], [xy[-1, 1]], s=52, marker="x", color=color, linewidth=2.2, zorder=4)
        ax.set_title(f"{self.run_name} eval best policy")
        add_sim_metric_box(ax, sim_metric_text(self.run_name, fit, metrics))
        fig.tight_layout()
        fig.savefig(png_path)
        plt.close(fig)

        summary = {
            "name": self.run_name,
            "trajectory_csv": str(csv_path),
            "eval_best_policy_png": str(png_path),
            **fit,
            **metrics,
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args():#命令行指令
    parser = argparse.ArgumentParser(description="Train PPO on open-loop turning gait reward.")
    parser.add_argument("--timesteps", type=int, default=150_000)
    parser.add_argument("--output", type=Path, default=Path("outputs/zips/ppo_turn_left_shape_bias"))
    parser.add_argument("--load-model", type=Path, default=None)
    parser.add_argument("--turn-direction", choices=("left", "right"), default="left")
    parser.add_argument("--target-yaw-rate", type=float, default=0.45, help="Target absolute yaw rate in rad/s.")
    parser.add_argument("--target-radius", type=float, default=None, help="Optional target absolute turn radius in meters.")
    parser.add_argument("--episode-seconds", type=float, default=None)
    parser.add_argument("--warmup-seconds", type=float, default=None)
    parser.add_argument("--freq", type=float, default=None)
    parser.add_argument("--wavelength", type=float, default=None)
    parser.add_argument("--ajoint", type=float, default=None, help="Base joint angle amplitude in degrees.")
    parser.add_argument("--action-mode", choices=("bias_only", "bias_tail2_amp", "bias_tail3_amp"), default=None)
    parser.add_argument("--fixed-amp-scales", type=lambda value: parse_float_list(value, 6, "fixed-amp-scales"), default=None)
    parser.add_argument("--amp-scale-lows", type=lambda value: parse_float_list(value, 6, "amp-scale-lows"), default=None)
    parser.add_argument("--amp-scale-highs", type=lambda value: parse_float_list(value, 6, "amp-scale-highs"), default=None)
    parser.add_argument("--phase-lag-lows", type=lambda value: parse_float_list(value, 5, "phase-lag-lows"), default=None)
    parser.add_argument("--phase-lag-highs", type=lambda value: parse_float_list(value, 5, "phase-lag-highs"), default=None)
    parser.add_argument("--joint-bias-low", type=float, default=None, help="Minimum learned joint bias in radians.")
    parser.add_argument("--joint-bias-high", type=float, default=None, help="Maximum learned joint bias in radians.")
    parser.add_argument("--tail-amp-multiplier-low", type=float, default=None)
    parser.add_argument("--tail-amp-multiplier-high", type=float, default=None)
    parser.add_argument("--yaw-rate-weight", type=float, default=None)
    parser.add_argument("--yaw-rate-source", choices=("body", "trajectory"), default=None)
    parser.add_argument("--radius-weight", type=float, default=None)
    parser.add_argument("--reward-average-seconds", type=float, default=None)
    parser.add_argument("--root-x-damping-scale", type=float, default=None)
    parser.add_argument("--root-y-damping-scale", type=float, default=None)
    parser.add_argument("--root-yaw-damping-scale", type=float, default=None)
    parser.add_argument("--boundary-x-min", type=float, default=None)
    parser.add_argument("--boundary-x-max", type=float, default=None)
    parser.add_argument("--boundary-y", type=float, default=None)
    parser.add_argument("--eval-freq", type=int, default=10_000, help="Evaluate every N training steps. Use 0 to disable.")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Episodes per evaluation.")
    parser.add_argument("--eval-log-dir", type=Path, default=None, help="Directory for evaluations.npz.")
    parser.add_argument("--plot-output", type=Path, default=None, help="PNG path for eval reward curve.")
    parser.add_argument("--no-plot", action="store_true", help="Do not create a PNG/CSV plot after training.")
    parser.add_argument("--robot-imu", action="store_true", help="Train with robot-deployable target/feedback/error/IMU observations.")
    parser.add_argument("--robot-imu-features", choices=("basic", "servo_qd", "filtered"), default=None)
    parser.add_argument("--n-envs", type=int, default=1, help="Number of parallel environments.")
    parser.add_argument("--vec-env", choices=("dummy", "subproc"), default="dummy", help="Parallel environment backend.")
    parser.add_argument("--servo-feedback-delay-steps", type=int, default=None)
    parser.add_argument("--servo-feedback-noise-std-deg", type=float, default=None)
    parser.add_argument("--servo-feedback-quantization-deg", type=float, default=None)
    parser.add_argument("--imu-delay-steps", type=int, default=None)
    parser.add_argument("--imu-accel-noise-std-g", type=float, default=None)
    parser.add_argument("--imu-gyro-noise-std-rps", type=float, default=None)
    parser.add_argument("--imu-roll-mount-deg", type=float, default=None)
    parser.add_argument("--imu-pitch-mount-deg", type=float, default=None)
    parser.add_argument("--imu-yaw-mount-deg", type=float, default=None)
    parser.add_argument("--imu-random-roll-deg", type=float, default=None)
    parser.add_argument("--imu-random-pitch-deg", type=float, default=None)
    parser.add_argument("--imu-random-yaw-deg", type=float, default=None)
    return parser.parse_args()


def config_from_args(args) -> TurningConfig:#環境檔參數檔
    cfg = TurningRobotImuConfig() if args.robot_imu else TurningConfig()
    cfg.turn_direction = args.turn_direction
    cfg.target_yaw_rate = abs(float(args.target_yaw_rate))
    if args.target_radius is not None:
        cfg.target_radius = abs(float(args.target_radius))
        if args.radius_weight is None:
            cfg.radius_weight = 0.40
    if args.episode_seconds is not None:
        cfg.episode_seconds = args.episode_seconds
    if args.warmup_seconds is not None:
        cfg.warmup_seconds = args.warmup_seconds
    if args.freq is not None:
        cfg.fixed_frequency = args.freq
    if args.wavelength is not None:
        cfg.fixed_wavelength = args.wavelength
    if args.ajoint is not None:
        cfg.fixed_ajoint = degrees_to_radians(args.ajoint)
    if args.action_mode is not None:
        cfg.action_mode = args.action_mode
    if args.fixed_amp_scales is not None:
        cfg.fixed_amp_scales = tuple(args.fixed_amp_scales)
    if args.amp_scale_lows is not None:
        cfg.amp_scale_lows = args.amp_scale_lows
    if args.amp_scale_highs is not None:
        cfg.amp_scale_highs = args.amp_scale_highs
    if args.phase_lag_lows is not None:
        cfg.phase_lag_lows = args.phase_lag_lows
    if args.phase_lag_highs is not None:
        cfg.phase_lag_highs = args.phase_lag_highs
    if args.joint_bias_low is not None:
        cfg.joint_bias_low = args.joint_bias_low
    if args.joint_bias_high is not None:
        cfg.joint_bias_high = args.joint_bias_high
    if args.tail_amp_multiplier_low is not None:
        cfg.tail_amp_multiplier_low = args.tail_amp_multiplier_low
    if args.tail_amp_multiplier_high is not None:
        cfg.tail_amp_multiplier_high = args.tail_amp_multiplier_high
    if cfg.joint_bias_low > cfg.joint_bias_high:
        raise ValueError("joint-bias-low cannot be greater than joint-bias-high")
    if cfg.tail_amp_multiplier_low > cfg.tail_amp_multiplier_high:
        raise ValueError("tail-amp-multiplier-low cannot be greater than tail-amp-multiplier-high")

    # If the user trains only one direction, bias bounds can stay symmetric.
    # The reward's signed target yaw rate decides which side is useful.
    if args.yaw_rate_weight is not None:
        cfg.yaw_rate_weight = args.yaw_rate_weight
    if args.yaw_rate_source is not None:
        cfg.yaw_rate_source = args.yaw_rate_source
    if args.radius_weight is not None:
        cfg.radius_weight = args.radius_weight
    if args.reward_average_seconds is not None:
        cfg.reward_average_seconds = args.reward_average_seconds
    if args.root_x_damping_scale is not None:
        cfg.root_x_damping_scale = args.root_x_damping_scale
    if args.root_y_damping_scale is not None:
        cfg.root_y_damping_scale = args.root_y_damping_scale
    if args.root_yaw_damping_scale is not None:
        cfg.root_yaw_damping_scale = args.root_yaw_damping_scale
    if args.boundary_x_min is not None:
        cfg.boundary_x_min = args.boundary_x_min
    if args.boundary_x_max is not None:
        cfg.boundary_x_max = args.boundary_x_max
    if args.boundary_y is not None:
        cfg.boundary_y = abs(args.boundary_y)
    if args.robot_imu:
        if args.robot_imu_features is not None:
            cfg.robot_imu_features = args.robot_imu_features
        if args.servo_feedback_delay_steps is not None:
            cfg.servo_feedback_delay_steps = args.servo_feedback_delay_steps
        if args.servo_feedback_noise_std_deg is not None:
            cfg.servo_feedback_noise_std_deg = args.servo_feedback_noise_std_deg
        if args.servo_feedback_quantization_deg is not None:
            cfg.servo_feedback_quantization_deg = args.servo_feedback_quantization_deg
        if args.imu_delay_steps is not None:
            cfg.imu_delay_steps = args.imu_delay_steps
        if args.imu_accel_noise_std_g is not None:
            cfg.imu_accel_noise_std_g = args.imu_accel_noise_std_g
        if args.imu_gyro_noise_std_rps is not None:
            cfg.imu_gyro_noise_std_rps = args.imu_gyro_noise_std_rps
        if args.imu_roll_mount_deg is not None:
            cfg.imu_roll_mount_deg = args.imu_roll_mount_deg
        if args.imu_pitch_mount_deg is not None:
            cfg.imu_pitch_mount_deg = args.imu_pitch_mount_deg
        if args.imu_yaw_mount_deg is not None:
            cfg.imu_yaw_mount_deg = args.imu_yaw_mount_deg
        if args.imu_random_roll_deg is not None:
            cfg.imu_random_roll_deg = args.imu_random_roll_deg
        if args.imu_random_pitch_deg is not None:
            cfg.imu_random_pitch_deg = args.imu_random_pitch_deg
        if args.imu_random_yaw_deg is not None:
            cfg.imu_random_yaw_deg = args.imu_random_yaw_deg

    # Validate direction spelling early.
    direction_sign(cfg.turn_direction)
    return cfg


def make_env(args, cfg: TurningConfig):
    env_cls = EelTurningRobotImuEnv if args.robot_imu else EelTurningRLEnv
    return Monitor(env_cls(cfg))


def make_train_env(args, cfg: TurningConfig):
    n_envs = max(1, int(args.n_envs))
    if n_envs == 1:
        return make_env(args, cfg)

    def env_factory():
        return make_env(args, cfg)

    env_fns = [env_factory for _ in range(n_envs)]
    if args.vec_env == "subproc":
        return SubprocVecEnv(env_fns)
    return DummyVecEnv(env_fns)


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


def make_eval_callback(args, cfg: TurningConfig) -> tuple[BaseCallback | None, Path | None, Path | None, Path | None]:
    if args.eval_freq <= 0:
        return None, None, None, None
    eval_log_dir = args.eval_log_dir or default_eval_log_dir(args.output)
    plot_output = args.plot_output or default_plot_path(args.output)
    debug_output = debug_csv_path(plot_output)
    eval_log_dir.mkdir(parents=True, exist_ok=True)
    eval_env = make_env(args, cfg)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(eval_log_dir / "best_model"),
        log_path=str(eval_log_dir),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )
    debug_callback = DetailedEvalMetricsCallback(
        cfg,
        debug_output,
        args.eval_freq,
        args.eval_episodes,
        best_episode_dir=eval_best_policy_dir(plot_output),
        run_name=args.output.stem,
    )
    return CallbackList([eval_callback, debug_callback]), eval_log_dir, plot_output, debug_output


def main():#訓練入口
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cfg = config_from_args(args)
    env = make_train_env(args, cfg)
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
    print(f"saved turning policy to {args.output}.zip")
    if callback is not None and eval_log_dir is not None and plot_output is not None and not args.no_plot:
        try_plot_eval_curve(eval_log_dir, plot_output, label=args.output.name)
    if debug_output is not None and debug_output.exists():
        print(f"saved eval debug csv to {debug_output}")


if __name__ == "__main__":
    main()
