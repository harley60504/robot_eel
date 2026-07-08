from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from sim_config import DEFAULT_START_X, DEFAULT_START_Y, EEL_MODEL_XML, RESET_X_MAX, RESET_X_MIN, RESET_Y, TANK_CENTER_X
from hopf_cpg import DEFAULT_AJOINT_DEG, HopfCPG, HopfCPGParams, amp_scales_to_mu_scales, degrees_to_radians


def parse_float_list(value: str, expected_len: int, name: str) -> tuple[float, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != expected_len:
        raise argparse.ArgumentTypeError(f"{name} needs {expected_len} comma-separated values")
    return tuple(float(part) for part in parts)


def parse_args():
    parser = argparse.ArgumentParser(
        description="View a free-swimming eel while abruptly changing CPG amplitude and wavelength."
    )
    parser.add_argument("--xml", default=EEL_MODEL_XML)
    parser.add_argument("--freq", type=float, default=1.0)
    parser.add_argument("--ajoint", type=float, default=DEFAULT_AJOINT_DEG, help="Base joint angle amplitude in degrees.")
    parser.add_argument("--before-wavelength", type=float, default=1.6275)
    parser.add_argument("--after-wavelength", type=float, default=3.0)
    parser.add_argument("--before-amp-scale", type=float, default=1.0)
    parser.add_argument("--after-amp-scale", type=float, default=0.55)
    parser.add_argument("--switch-time", type=float, default=5.0)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--k-couple", type=float, default=0.35)
    parser.add_argument("--k-anchor", type=float, default=0.10)
    parser.add_argument(
        "--repeat",
        action="store_true",
        help="Keep toggling between before/after parameters after switch-time.",
    )
    parser.add_argument("--switch-period", type=float, default=4.0)
    parser.add_argument(
        "--amp-scales",
        type=lambda value: parse_float_list(value, 6, "amp-scales"),
        default=(1.24, 1.08, 1.0, 1.05, 1.1, 1.2),
    )
    parser.add_argument(
        "--amp-mode",
        choices=("mu", "ajoint"),
        default="mu",
        help="mu changes Hopf target amplitude smoothly; ajoint changes output gain immediately.",
    )
    parser.add_argument(
        "--control-mode",
        choices=("hopf", "sin"),
        default="hopf",
        help="hopf uses oscillator state; sin directly computes joint angles from time.",
    )
    parser.add_argument("--print-hz", type=float, default=2.0)
    parser.add_argument("--viewer-fps", type=float, default=60.0, help="Viewer render FPS used for real-time pacing.")
    parser.add_argument("--start-x", type=float, default=DEFAULT_START_X)
    parser.add_argument("--start-y", type=float, default=DEFAULT_START_Y)
    parser.add_argument("--reset-x-min", type=float, default=RESET_X_MIN)
    parser.add_argument("--reset-x-max", type=float, default=RESET_X_MAX)
    parser.add_argument("--reset-y", type=float, default=RESET_Y)
    return parser.parse_args()


def make_params(args, switched: bool) -> HopfCPGParams:
    amp_scale = args.after_amp_scale if switched else args.before_amp_scale
    wavelength = args.after_wavelength if switched else args.before_wavelength
    base_ajoint = degrees_to_radians(args.ajoint)

    if args.amp_mode == "mu":
        ajoint = base_ajoint
        target_amp_scales = tuple(amp_scale * float(value) for value in args.amp_scales)
        mu_scales = amp_scales_to_mu_scales(target_amp_scales)
    else:
        ajoint = base_ajoint * amp_scale
        mu_scales = amp_scales_to_mu_scales(args.amp_scales)

    return HopfCPGParams(
        frequency=args.freq,
        wavelength=wavelength,
        ajoint=ajoint,
        alpha=args.alpha,
        k_couple=args.k_couple,
        k_anchor=args.k_anchor,
        mu_scales=mu_scales,
    )


def sine_targets(t: float, args, switched: bool) -> np.ndarray:
    amp_scale = args.after_amp_scale if switched else args.before_amp_scale
    wavelength = args.after_wavelength if switched else args.before_wavelength
    phase_lag = 1.0 / max(wavelength, 1e-6)
    amp_scales = np.asarray(args.amp_scales, dtype=np.float64)
    phases = -np.arange(6, dtype=np.float64) * phase_lag
    omega = 2.0 * np.pi * args.freq
    return degrees_to_radians(args.ajoint) * amp_scale * amp_scales * np.cos(omega * t + phases)


def main():
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    model.opt.gravity[:] = (0, 0, 0)
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

    cpg = HopfCPG(num_joints=6)
    last_print = 0.0
    last_switched = None
    print_period = 1.0 / max(args.print_hz, 1e-6)

    def reset_to_start():
        mujoco.mj_resetData(model, data)
        base_xml_pos = model.body_pos[base_body_id]
        data.qpos[0] = args.start_x - base_xml_pos[0]
        data.qpos[1] = args.start_y - base_xml_pos[1]
        cpg.reset()
        mujoco.mj_forward(model, data)

    reset_to_start()

    print(
        "CPG step-change viewer\n"
        f"  switch at t={args.switch_time:.2f}s\n"
        f"  wavelength: {args.before_wavelength:.4f} -> {args.after_wavelength:.4f}\n"
        f"  phase lag:  {1.0 / args.before_wavelength:.4f} -> {1.0 / args.after_wavelength:.4f} rad/joint\n"
        f"  amp scale:  {args.before_amp_scale:.3f} -> {args.after_amp_scale:.3f} ({args.amp_mode} mode)\n"
        f"  Hopf gains: alpha={args.alpha:.3f}, k_couple={args.k_couple:.3f}, k_anchor={args.k_anchor:.3f}\n"
        "  viewer pacing: real-time wall-clock pacing with batched MuJoCo steps",
        flush=True,
    )

    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = np.array([args.start_x, args.start_y, -0.02])
            viewer.cam.distance = 1.4
            viewer.cam.elevation = -70
            viewer.cam.azimuth = 0

        target_fps = max(args.viewer_fps, 1.0)
        frame_dt = 1.0 / target_fps
        last_wall_clock = time.perf_counter()
        base_pos = data.xpos[base_body_id]
        params = make_params(args, False)
        switched = False

        while viewer.is_running():
            frame_start = time.perf_counter()
            wall_dt = min(frame_start - last_wall_clock, 0.05)
            last_wall_clock = frame_start
            target_sim_time = data.time + wall_dt

            while data.time + 1e-12 < target_sim_time:
                if args.repeat and data.time >= args.switch_time:
                    switched = int((data.time - args.switch_time) // max(args.switch_period, model.opt.timestep)) % 2 == 0
                else:
                    switched = data.time >= args.switch_time
                params = make_params(args, switched)
                if args.control_mode == "sin":
                    targets = sine_targets(data.time, args, switched)
                else:
                    targets = cpg.step(data.time, model.opt.timestep, params)
                data.ctrl[0:6] = np.clip(targets, -1.2, 1.2)
                mujoco.mj_step(model, data)
                base_pos = data.xpos[base_body_id]

                if switched != last_switched:
                    state = "AFTER" if switched else "BEFORE"
                    print(f"mode={state} at t={data.time:.2f}s", flush=True)
                    last_switched = switched

                if base_pos[0] < args.reset_x_min or base_pos[0] > args.reset_x_max or abs(base_pos[1]) > args.reset_y:
                    print(
                        f"reset to start: x={base_pos[0]:.3f}, y={base_pos[1]:.3f}",
                        flush=True,
                    )
                    reset_to_start()
                    base_pos = data.xpos[base_body_id]
                    break

            now = time.time()
            if now - last_print >= print_period:
                lag = 1.0 / max(params.wavelength, 1e-6)
                print(
                    f"t={data.time:6.2f}s | "
                    f"{'after ' if switched else 'before'} | "
                    f"lambda={params.wavelength:5.3f} lag={lag:5.3f} | "
                    f"{'r_mean=' + format(np.mean(cpg.r), '5.3f') if args.control_mode == 'hopf' else 'direct-sin'} | "
                    f"x={base_pos[0]:7.3f} y={base_pos[1]:7.3f} yaw={data.qpos[2]:7.3f}",
                    flush=True,
                )
                last_print = now

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
