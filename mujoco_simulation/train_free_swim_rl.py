from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from hopf_cpg import degrees_to_radians
from rl_free_swim_env import EelFreeSwimRLEnv, FreeSwimConfig
from rl_training_plots import default_eval_log_dir, default_plot_path, try_plot_eval_curve


def parse_float_list(value: str, expected_len: int, name: str) -> tuple[float, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != expected_len:
        raise argparse.ArgumentTypeError(f"{name} needs {expected_len} comma-separated values")
    return tuple(float(part) for part in parts)


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on free-swim forward velocity.")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--output", type=Path, default=Path("outputs/ppo_free_swim_shape"))
    parser.add_argument("--load-model", type=Path, default=None)
    parser.add_argument("--episode-seconds", type=float, default=None)
    parser.add_argument("--warmup-seconds", type=float, default=None)
    parser.add_argument("--freq", type=float, default=None)
    parser.add_argument("--wavelength", type=float, default=None)
    parser.add_argument("--ajoint", type=float, default=None, help="Base joint angle amplitude in degrees.")
    parser.add_argument("--amp-scale-lows", type=lambda value: parse_float_list(value, 6, "amp-scale-lows"), default=None)
    parser.add_argument("--amp-scale-highs", type=lambda value: parse_float_list(value, 6, "amp-scale-highs"), default=None)
    parser.add_argument("--phase-lag-lows", type=lambda value: parse_float_list(value, 5, "phase-lag-lows"), default=None)
    parser.add_argument("--phase-lag-highs", type=lambda value: parse_float_list(value, 5, "phase-lag-highs"), default=None)
    parser.add_argument("--lateral-velocity-weight", type=float, default=None)
    parser.add_argument("--lateral-position-weight", type=float, default=None)
    parser.add_argument("--yaw-weight", type=float, default=None)
    parser.add_argument("--yaw-rate-weight", type=float, default=None)
    parser.add_argument("--energy-weight", type=float, default=None)
    parser.add_argument("--smoothness-weight", type=float, default=None)
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
    if args.wavelength is not None:
        cfg.fixed_wavelength = args.wavelength
    if args.ajoint is not None:
        cfg.fixed_ajoint = degrees_to_radians(args.ajoint)
    if args.amp_scale_lows is not None:
        cfg.amp_scale_lows = args.amp_scale_lows
    if args.amp_scale_highs is not None:
        cfg.amp_scale_highs = args.amp_scale_highs
    if args.phase_lag_lows is not None:
        cfg.phase_lag_lows = args.phase_lag_lows
    if args.phase_lag_highs is not None:
        cfg.phase_lag_highs = args.phase_lag_highs
    if args.lateral_velocity_weight is not None:
        cfg.lateral_velocity_weight = args.lateral_velocity_weight
    if args.lateral_position_weight is not None:
        cfg.lateral_position_weight = args.lateral_position_weight
    if args.yaw_weight is not None:
        cfg.yaw_weight = args.yaw_weight
    if args.yaw_rate_weight is not None:
        cfg.yaw_rate_weight = args.yaw_rate_weight
    if args.energy_weight is not None:
        cfg.energy_weight = args.energy_weight
    if args.smoothness_weight is not None:
        cfg.smoothness_weight = args.smoothness_weight
    return cfg


def make_eval_callback(args, cfg: FreeSwimConfig) -> tuple[EvalCallback | None, Path | None, Path | None]:
    if args.eval_freq <= 0:
        return None, None, None
    eval_log_dir = args.eval_log_dir or default_eval_log_dir(args.output)
    plot_output = args.plot_output or default_plot_path(args.output)
    eval_log_dir.mkdir(parents=True, exist_ok=True)
    eval_env = Monitor(EelFreeSwimRLEnv(cfg))
    callback = EvalCallback(
        eval_env,
        best_model_save_path=str(eval_log_dir / "best_model"),
        log_path=str(eval_log_dir),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )
    return callback, eval_log_dir, plot_output


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cfg = config_from_args(args)
    env = Monitor(EelFreeSwimRLEnv(cfg))
    callback, eval_log_dir, plot_output = make_eval_callback(args, cfg)
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


if __name__ == "__main__":
    main()
