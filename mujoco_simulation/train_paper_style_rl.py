from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from hopf_cpg import DEFAULT_AJOINT_DEG, degrees_to_radians
from rl_tethered_env import EelTetheredRLEnv, TetheredConfig


def parse_float_list(value: str, expected_len: int, name: str) -> tuple[float, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != expected_len:
        raise argparse.ArgumentTypeError(f"{name} needs {expected_len} comma-separated values")
    return tuple(float(part) for part in parts)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paper-style tethered force-target PPO: fixed frequency, fixed wavelength, CPG shape action."
    )
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--output", type=Path, default=Path("outputs/ppo_paper_style_target"))
    parser.add_argument("--target-force", type=float, default=4.8)
    parser.add_argument("--target-metric", choices=("fx", "resultant"), default="fx")
    parser.add_argument("--freq", type=float, default=1.0)
    parser.add_argument("--wavelength", type=float, default=1.6275)
    parser.add_argument("--ajoint", type=float, default=DEFAULT_AJOINT_DEG, help="Base joint angle amplitude in degrees.")
    parser.add_argument("--episode-seconds", type=float, default=4.0)
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument("--lateral-force-weight", type=float, default=0.05)
    parser.add_argument("--energy-weight", type=float, default=0.005)
    parser.add_argument("--smoothness-weight", type=float, default=0.01)
    parser.add_argument("--forward-force-weight", type=float, default=0.02)
    parser.add_argument(
        "--amp-lows",
        type=lambda value: parse_float_list(value, 6, "amp-lows"),
        default=(0.85, 0.85, 0.85, 0.85, 0.85, 0.85),
    )
    parser.add_argument(
        "--amp-highs",
        type=lambda value: parse_float_list(value, 6, "amp-highs"),
        default=(1.30, 1.30, 1.25, 1.25, 1.35, 1.35),
    )
    parser.add_argument(
        "--phase-lows",
        type=lambda value: parse_float_list(value, 5, "phase-lows"),
        default=(0.55, 0.55, 0.55, 0.55, 0.55),
    )
    parser.add_argument(
        "--phase-highs",
        type=lambda value: parse_float_list(value, 5, "phase-highs"),
        default=(0.70, 0.70, 0.70, 0.70, 0.70),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    cfg = TetheredConfig(
        fixed_frequency=args.freq,
        fixed_wavelength=args.wavelength,
        fixed_ajoint=degrees_to_radians(args.ajoint),
        per_joint_action=True,
        reward_mode="target_force",
        target_force=args.target_force,
        target_metric=args.target_metric,
        target_error_weight=1.0,
        forward_force_weight=args.forward_force_weight,
        lateral_force_weight=args.lateral_force_weight,
        energy_weight=args.energy_weight,
        action_smoothness_weight=args.smoothness_weight,
        episode_seconds=args.episode_seconds,
        warmup_seconds=args.warmup_seconds,
        amp_scale_lows=args.amp_lows,
        amp_scale_highs=args.amp_highs,
        phase_lag_lows=args.phase_lows,
        phase_lag_highs=args.phase_highs,
    )

    env = Monitor(EelTetheredRLEnv(cfg))
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
    model.learn(total_timesteps=args.timesteps)
    model.save(args.output)
    print(f"saved policy to {args.output}.zip")


if __name__ == "__main__":
    main()
