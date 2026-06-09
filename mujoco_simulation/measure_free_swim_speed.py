from __future__ import annotations

import argparse
import csv
from pathlib import Path

import mujoco
import numpy as np

from hopf_cpg import DEFAULT_AJOINT_DEG, HopfCPG, HopfCPGParams, amp_scales_to_mu_scales, degrees_to_radians


def parse_float_list(value: str, expected_len: int, name: str) -> tuple[float, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != expected_len:
        raise argparse.ArgumentTypeError(f"{name} needs {expected_len} comma-separated values")
    return tuple(float(part) for part in parts)


def parse_args():
    parser = argparse.ArgumentParser(description="Measure untethered eel free-swim speed.")
    parser.add_argument("--xml", default="eel.xml")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--warmup-seconds", type=float, default=2.0)
    parser.add_argument("--ajoint", "--amp", dest="ajoint", type=float, default=DEFAULT_AJOINT_DEG, help="Base joint angle amplitude in degrees.")
    parser.add_argument("--freq", type=float, default=1.0)
    parser.add_argument("--wavelength", type=float, default=1.5)
    parser.add_argument(
        "--amp-scales",
        type=lambda value: parse_float_list(value, 6, "amp-scales"),
        default=None,
    )
    parser.add_argument(
        "--phase-lags",
        type=lambda value: parse_float_list(value, 5, "phase-lags"),
        default=None,
    )
    parser.add_argument(
        "--joint-bias",
        type=lambda value: parse_float_list(value, 6, "joint-bias"),
        default=None,
        help="Static per-joint steering offset in radians.",
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
    ajoint_rad = degrees_to_radians(args.ajoint)
    cpg_params = HopfCPGParams(
        frequency=args.freq,
        wavelength=args.wavelength,
        ajoint=ajoint_rad,
        mu_scales=amp_scales_to_mu_scales(args.amp_scales),
        phase_lags=args.phase_lags,
        joint_bias=args.joint_bias,
    )

    steps = int(round(args.seconds / model.opt.timestep))
    warmup_steps = int(round(args.warmup_seconds / model.opt.timestep))
    records = []

    mujoco.mj_forward(model, data)
    start_pos = data.xpos[base_body_id].copy()
    steady_start_pos = None
    steady_start_time = None

    for step in range(steps):
        targets = cpg.step(data.time, model.opt.timestep, cpg_params)
        data.ctrl[0:6] = np.clip(targets, -1.2, 1.2)
        mujoco.mj_step(model, data)
        base_pos = data.xpos[base_body_id].copy()

        if step == warmup_steps:
            steady_start_pos = base_pos.copy()
            steady_start_time = float(data.time)

        if step >= warmup_steps:
            records.append(
                (
                    data.time,
                    base_pos[0],
                    base_pos[1],
                    data.qpos[2],
                    data.qvel[0],
                    data.qvel[1],
                    data.qvel[2],
                )
            )

    arr = np.asarray(records, dtype=np.float64)
    end_pos = data.xpos[base_body_id].copy()
    if steady_start_pos is None or steady_start_time is None:
        steady_start_pos = start_pos
        steady_start_time = 0.0
    steady_dt = max(1e-9, float(data.time) - steady_start_time)
    steady_dx = float(end_pos[0] - steady_start_pos[0])
    steady_dy = float(end_pos[1] - steady_start_pos[1])

    print("Free-swim speed measurement")
    print(
        f"  Hopf CPG: ajoint={args.ajoint:.3f} deg ({ajoint_rad:.3f} rad), freq={args.freq:.3f} Hz, "
        f"wavelength={args.wavelength:.4f}"
    )
    if args.amp_scales is not None:
        print("  amp scales:", ", ".join(f"{value:.3f}" for value in args.amp_scales))
    if args.phase_lags is not None:
        print("  phase lags:", ", ".join(f"{value:.3f}" for value in args.phase_lags))
    if args.joint_bias is not None:
        print("  joint bias:", ", ".join(f"{value:.3f}" for value in args.joint_bias))
    print(f"  samples={len(arr)}, warmup={args.warmup_seconds:.2f}s, total={args.seconds:.2f}s")
    print(f"  start x={start_pos[0]: .4f} y={start_pos[1]: .4f}")
    print(f"  end   x={end_pos[0]: .4f} y={end_pos[1]: .4f}")
    print(f"  steady displacement dx={steady_dx: .4f} m, dy={steady_dy: .4f} m")
    print(f"  mean vx from displacement={steady_dx / steady_dt: .4f} m/s")
    print(f"  mean vy from displacement={steady_dy / steady_dt: .4f} m/s")
    print(f"  mean qvel vx={np.mean(arr[:, 4]): .4f} m/s")
    print(f"  mean qvel vy={np.mean(arr[:, 5]): .4f} m/s")
    print(f"  final yaw={data.qpos[2]: .4f} rad")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "x", "y", "yaw", "vx", "vy", "yaw_rate"])
            writer.writerows(records)
        print(f"  wrote {args.csv}")


if __name__ == "__main__":
    main()
