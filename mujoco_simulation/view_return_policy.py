from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

from rl_free_swim_env import EelFreeSwimRLEnv, FreeSwimConfig
from rl_turning_env import EelTurningRLEnv, TurningConfig


def _safe_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_summary(path: Path | None) -> dict:
    if path is None:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _apply_common_bounds(cfg) -> None:
    cfg.start_x = 0.0
    cfg.start_y = 0.0
    cfg.boundary_x_min = -10.0
    cfg.boundary_x_max = 10.0
    cfg.boundary_y = 10.0
    cfg.episode_seconds = max(float(getattr(cfg, "episode_seconds", 10.0)), 3600.0)


def _free_cfg(summary: dict, args: argparse.Namespace) -> FreeSwimConfig:
    env_config = summary.get("env_config") if isinstance(summary.get("env_config"), dict) else {}
    cfg = FreeSwimConfig()
    for attr in (
        "fixed_frequency",
        "fixed_wavelength",
        "fixed_ajoint",
        "reward_average_seconds",
        "target_speed",
        "speed_error_weight",
        "energy_weight",
        "frequency_low",
        "frequency_high",
        "phase_lag_low",
        "phase_lag_high",
    ):
        value = _safe_float(env_config.get(attr))
        if value is not None:
            setattr(cfg, attr, value)
    if isinstance(env_config.get("fixed_amp_scales"), list):
        cfg.fixed_amp_scales = tuple(float(v) for v in env_config["fixed_amp_scales"])
    if args.target_speed is not None:
        cfg.target_speed = float(args.target_speed)
    _apply_common_bounds(cfg)
    return cfg


def _turning_cfg(summary: dict, args: argparse.Namespace) -> TurningConfig:
    env_config = summary.get("env_config") if isinstance(summary.get("env_config"), dict) else {}
    cfg = TurningConfig()
    for attr in (
        "fixed_frequency",
        "fixed_wavelength",
        "fixed_ajoint",
        "reward_average_seconds",
        "target_yaw_rate",
        "target_radius",
        "yaw_rate_weight",
        "radius_weight",
        "joint_bias_low",
        "joint_bias_high",
        "tail_amp_multiplier_low",
        "tail_amp_multiplier_high",
    ):
        value = _safe_float(env_config.get(attr))
        if value is not None:
            setattr(cfg, attr, value)
    if isinstance(env_config.get("turn_direction"), str):
        cfg.turn_direction = env_config["turn_direction"]
    if isinstance(env_config.get("action_mode"), str):
        cfg.action_mode = env_config["action_mode"]
    if isinstance(env_config.get("fixed_amp_scales"), list):
        cfg.fixed_amp_scales = tuple(float(v) for v in env_config["fixed_amp_scales"])
    if isinstance(env_config.get("fixed_phase_lags"), list):
        cfg.fixed_phase_lags = tuple(float(v) for v in env_config["fixed_phase_lags"])
    if args.turn_direction:
        cfg.turn_direction = args.turn_direction
    if args.target_yaw_rate is not None:
        cfg.target_yaw_rate = abs(float(args.target_yaw_rate))
    if args.target_radius is not None:
        cfg.target_radius = abs(float(args.target_radius))
    _apply_common_bounds(cfg)
    cfg.wall_collision = False
    return cfg


def infer_mode(summary: dict, requested: str) -> str:
    if requested != "auto":
        return requested
    env_config = summary.get("env_config") if isinstance(summary.get("env_config"), dict) else {}
    if "target_speed" in env_config or "frequency_low" in env_config:
        return "straight"
    return "turning"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View a trained PPO return policy in the shared +/-10 m tank.")
    parser.add_argument("model", type=Path)
    parser.add_argument("--summary", type=Path, default=None, help="policy_rerun_summary or policy_rollout_summary JSON")
    parser.add_argument("--mode", choices=("auto", "straight", "turning"), default="auto")
    parser.add_argument("--target-speed", type=float, default=None)
    parser.add_argument("--turn-direction", choices=("left", "right"), default=None)
    parser.add_argument("--target-yaw-rate", type=float, default=None)
    parser.add_argument("--target-radius", type=float, default=None)
    parser.add_argument("--viewer-fps", type=float, default=60.0)
    parser.add_argument("--camera-mode", choices=("fixed", "follow"), default="follow")
    parser.add_argument("--camera-distance", type=float, default=6.0)
    parser.add_argument("--camera-elevation", type=float, default=-70.0)
    parser.add_argument("--print-hz", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = _load_summary(args.summary)
    mode = infer_mode(summary, args.mode)
    cfg = _free_cfg(summary, args) if mode == "straight" else _turning_cfg(summary, args)
    env = EelFreeSwimRLEnv(cfg) if mode == "straight" else EelTurningRLEnv(cfg)
    model = PPO.load(args.model, env=env)
    obs, _ = env.reset()

    for geom_id in range(env.model.ngeom):
        name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name.startswith("wall_"):
            env.model.geom_contype[geom_id] = 0
            env.model.geom_conaffinity[geom_id] = 0

    print(f"Loaded return policy: {args.model}", flush=True)
    print(f"  mode={mode}", flush=True)
    print("  tank: shared x/y +/-10 m visual pool; wall collision disabled", flush=True)

    base_body_id = env.base_body_id
    last_print = 0.0
    print_period = 1.0 / max(args.print_hz, 1e-6)
    target_fps = max(args.viewer_fps, 1.0)
    frame_dt = 1.0 / target_fps

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = np.array([0.0, 0.0, -0.02])
            viewer.cam.distance = args.camera_distance
            viewer.cam.elevation = args.camera_elevation
            viewer.cam.azimuth = 0

        while viewer.is_running():
            frame_start = time.perf_counter()
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
                info = {}
            base_pos = env.data.xpos[base_body_id]

            now = time.time()
            if now - last_print >= print_period:
                print(
                    f"t={env.data.time:6.2f}s | x={base_pos[0]:8.3f} y={base_pos[1]:8.3f} "
                    f"yaw={env.data.qpos[2]:8.3f} | reward_info={info.get('steady_state', False)}",
                    flush=True,
                )
                last_print = now

            if args.camera_mode == "follow":
                with viewer.lock():
                    viewer.cam.lookat[0] = base_pos[0]
                    viewer.cam.lookat[1] = base_pos[1]
            viewer.sync()

            elapsed = time.perf_counter() - frame_start
            sleep_time = frame_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
