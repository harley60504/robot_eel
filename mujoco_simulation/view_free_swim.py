from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from hopf_cpg import DEFAULT_AJOINT_DEG, HopfCPG, HopfCPGParams, amp_scales_to_mu_scales, degrees_to_radians


def parse_float_list(value: str, expected_len: int, name: str) -> tuple[float, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != expected_len:
        raise argparse.ArgumentTypeError(f"{name} needs {expected_len} comma-separated values")
    return tuple(float(part) for part in parts)


def parse_args():
    parser = argparse.ArgumentParser(description="View the untethered eel swimming freely.")
    parser.add_argument("--xml", default="eel.xml")
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
    parser.add_argument("--print-hz", type=float, default=2.0)
    parser.add_argument("--reset-x", type=float, default=1.725)
    parser.add_argument("--reset-y", type=float, default=0.90)
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
    last_print = 0.0
    print_period = 1.0 / max(args.print_hz, 1e-6)

    def reset_to_start():
        mujoco.mj_resetData(model, data)
        cpg.reset()
        mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = np.array([0.0, 0.0, -0.02])
            viewer.cam.distance = 1.4
            viewer.cam.elevation = -70
            viewer.cam.azimuth = 0

        while viewer.is_running():
            targets = cpg.step(data.time, model.opt.timestep, cpg_params)
            data.ctrl[0:6] = np.clip(targets, -1.2, 1.2)
            mujoco.mj_step(model, data)
            base_pos = data.xpos[base_body_id]

            if abs(base_pos[0]) > args.reset_x or abs(base_pos[1]) > args.reset_y:
                print(
                    f"reset to start: x={base_pos[0]:.3f}, y={base_pos[1]:.3f}",
                    flush=True,
                )
                reset_to_start()
                base_pos = data.xpos[base_body_id]

            now = time.time()
            if now - last_print >= print_period:
                print(
                    f"t={data.time:6.2f}s | "
                    f"x={base_pos[0]:8.3f} y={base_pos[1]:8.3f} yaw={data.qpos[2]:8.3f} | "
                    f"vx={data.qvel[0]:8.3f} vy={data.qvel[1]:8.3f}",
                    flush=True,
                )
                last_print = now

            with viewer.lock():
                viewer.cam.lookat[0] = base_pos[0]
                viewer.cam.lookat[1] = base_pos[1]
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
