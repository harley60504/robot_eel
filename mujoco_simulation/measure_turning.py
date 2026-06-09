from __future__ import annotations

import argparse
import csv
from pathlib import Path

import mujoco
import numpy as np

from sim_config import DEFAULT_START_X, DEFAULT_START_Y, EEL_MODEL_XML, RESET_X_MAX, RESET_X_MIN, RESET_Y, TANK_CENTER_X
from hopf_cpg import DEFAULT_AJOINT_DEG, HopfCPG, HopfCPGParams, degrees_to_radians
from view_rectangle_course import amp_scales_to_mu_scales, turning_amp_scales


def parse_float_list(value: str, expected_len: int, name: str) -> tuple[float, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != expected_len:
        raise argparse.ArgumentTypeError(f"{name} needs {expected_len} comma-separated values")
    return tuple(float(part) for part in parts)


def parse_args():
    parser = argparse.ArgumentParser(description="Measure free-swim turning gait metrics.")
    parser.add_argument("--xml", default=EEL_MODEL_XML)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--warmup-seconds", type=float, default=2.0)
    parser.add_argument("--ajoint", "--amp", dest="ajoint", type=float, default=DEFAULT_AJOINT_DEG, help="Base joint angle amplitude in degrees.")
    parser.add_argument("--freq", type=float, default=1.0)
    parser.add_argument("--wavelength", type=float, default=1.6275)
    parser.add_argument(
        "--amp-scales",
        type=lambda value: parse_float_list(value, 6, "amp-scales"),
        default=(1.24, 1.08, 1.0, 1.05, 1.1, 1.2),
    )
    parser.add_argument(
        "--phase-lags",
        type=lambda value: parse_float_list(value, 5, "phase-lags"),
        default=(0.614439, 0.614439, 0.614439, 0.614439, 0.614439),
    )
    parser.add_argument(
        "--joint-bias",
        type=lambda value: parse_float_list(value, 6, "joint-bias"),
        default=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        help="Static per-joint steering offset in radians.",
    )
    parser.add_argument(
        "--turn-amp-gain",
        type=float,
        default=0.0,
        help="Increase tail amplitude based on the largest joint-bias magnitude.",
    )
    parser.add_argument(
        "--turn-phase-gain",
        type=float,
        default=0.0,
        help="Increase tail-side phase lags based on the largest joint-bias magnitude.",
    )
    parser.add_argument("--csv", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    model.opt.gravity[:] = (0, 0, 0)

    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    cpg = HopfCPG(num_joints=6)
    steer_proxy = max(abs(value) for value in args.joint_bias) if args.joint_bias else 0.0
    target_amp_scales = turning_amp_scales(args.amp_scales, steer_proxy, args.turn_amp_gain)
    mu_scales = amp_scales_to_mu_scales(target_amp_scales)
    phase_weights = np.array([0.0, 0.2, 0.5, 0.8, 1.0], dtype=np.float64)
    base_phase_lags = np.asarray(args.phase_lags, dtype=np.float64)
    phase_lags = tuple(
        float(value)
        for value in np.clip(
            base_phase_lags + args.turn_phase_gain * steer_proxy * phase_weights,
            0.25,
            1.20,
        )
    )
    ajoint_rad = degrees_to_radians(args.ajoint)
    cpg_params = HopfCPGParams(
        frequency=args.freq,
        wavelength=args.wavelength,
        ajoint=ajoint_rad,
        mu_scales=mu_scales,
        phase_lags=phase_lags,
        joint_bias=args.joint_bias,
    )

    steps = int(round(args.seconds / model.opt.timestep))
    warmup_steps = int(round(args.warmup_seconds / model.opt.timestep))
    records = []

    mujoco.mj_forward(model, data)
    steady_start_pos = None
    steady_start_yaw = None
    steady_start_time = None

    for step in range(steps):
        targets = cpg.step(data.time, model.opt.timestep, cpg_params)
        data.ctrl[0:6] = np.clip(targets, -1.2, 1.2)
        mujoco.mj_step(model, data)

        base_pos = data.xpos[base_body_id].copy()
        yaw = float(data.qpos[2])

        if step == warmup_steps:
            steady_start_pos = base_pos.copy()
            steady_start_yaw = yaw
            steady_start_time = float(data.time)

        if step >= warmup_steps:
            records.append(
                (
                    data.time,
                    base_pos[0],
                    base_pos[1],
                    yaw,
                    data.qvel[0],
                    data.qvel[1],
                    data.qvel[2],
                )
            )

    arr = np.asarray(records, dtype=np.float64)
    end_pos = data.xpos[base_body_id].copy()
    end_yaw = float(data.qpos[2])
    if steady_start_pos is None or steady_start_yaw is None or steady_start_time is None:
        steady_start_pos = data.xpos[base_body_id].copy()
        steady_start_yaw = end_yaw
        steady_start_time = 0.0

    steady_dt = max(1e-9, float(data.time) - steady_start_time)
    dx = float(end_pos[0] - steady_start_pos[0])
    dy = float(end_pos[1] - steady_start_pos[1])
    yaw_unwrapped = np.unwrap(arr[:, 3]) if len(arr) else np.array([steady_start_yaw, end_yaw])
    yaw_change = float(yaw_unwrapped[-1] - yaw_unwrapped[0]) if len(arr) else 0.0
    mean_yaw_rate = yaw_change / steady_dt
    mean_speed = float(np.mean(np.hypot(arr[:, 4], arr[:, 5]))) if len(arr) else 0.0
    turn_radius = np.inf if abs(mean_yaw_rate) < 1e-6 else mean_speed / abs(mean_yaw_rate)

    print("Turning measurement")
    print(
        f"  Hopf CPG: ajoint={args.ajoint:.3f} deg ({ajoint_rad:.3f} rad), freq={args.freq:.3f} Hz, "
        f"wavelength={args.wavelength:.4f}"
    )
    print("  target amp scales:", ", ".join(f"{value:.3f}" for value in target_amp_scales))
    print("  phase lags:", ", ".join(f"{value:.3f}" for value in phase_lags))
    print("  joint bias:", ", ".join(f"{value:.3f}" for value in args.joint_bias))
    print(f"  steady dx={dx: .4f} m, dy={dy: .4f} m")
    print(f"  mean vx={np.mean(arr[:, 4]): .4f} m/s")
    print(f"  mean vy={np.mean(arr[:, 5]): .4f} m/s")
    print(f"  mean speed={mean_speed: .4f} m/s")
    print(f"  yaw change={yaw_change: .4f} rad")
    print(f"  mean yaw rate={mean_yaw_rate: .4f} rad/s")
    print(f"  estimated turn radius={turn_radius: .4f} m")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "x", "y", "yaw", "vx", "vy", "yaw_rate"])
            writer.writerows(records)
        print(f"  wrote {args.csv}")


if __name__ == "__main__":
    main()
