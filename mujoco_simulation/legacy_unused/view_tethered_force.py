from __future__ import annotations

import argparse
import time
from collections import deque

import mujoco
import mujoco.viewer
import numpy as np

from hopf_cpg import DEFAULT_AJOINT_DEG, HopfCPG, HopfCPGParams, degrees_to_radians


def parse_args():
    parser = argparse.ArgumentParser(description="View tethered eel motion while printing live force estimates.")
    parser.add_argument("--xml", default="eel_tethered.xml")
    parser.add_argument("--ajoint", "--amp", dest="ajoint", type=float, default=DEFAULT_AJOINT_DEG, help="Base joint angle amplitude in degrees.")
    parser.add_argument("--freq", type=float, default=1.0)
    parser.add_argument("--wavelength", type=float, default=1.5)
    parser.add_argument("--fb-phase", type=float, default=0.0)
    parser.add_argument("--fb-amp", type=float, default=0.0)
    parser.add_argument("--print-hz", type=float, default=5.0)
    parser.add_argument("--avg-window", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    model.opt.gravity[:] = (0, 0, 0)

    root_x_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_x")
    root_y_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_y")
    root_x_dof = model.jnt_dofadr[root_x_jid]
    root_y_dof = model.jnt_dofadr[root_y_jid]

    avg_samples = max(1, int(round(args.avg_window / model.opt.timestep)))
    force_window = deque(maxlen=avg_samples)
    last_print = 0.0
    print_period = 1.0 / max(args.print_hz, 1e-6)
    cpg = HopfCPG(num_joints=6)
    ajoint_rad = degrees_to_radians(args.ajoint)
    cpg_params = HopfCPGParams(
        frequency=args.freq,
        wavelength=args.wavelength,
        ajoint=ajoint_rad,
        fb_phase=args.fb_phase,
        fb_amp=args.fb_amp,
    )

    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = np.array([-1.2, 0.0, -0.02])
            viewer.cam.distance = 1.2
            viewer.cam.elevation = -70
            viewer.cam.azimuth = 0

        while viewer.is_running():
            data.ctrl[0:3] = 0.0
            targets = cpg.step(data.time, model.opt.timestep, cpg_params)
            data.ctrl[3:9] = np.clip(targets, -1.2, 1.2)
            mujoco.mj_step(model, data)

            fx = -data.qfrc_actuator[root_x_dof]
            fy = -data.qfrc_actuator[root_y_dof]
            force_window.append((fx, fy))

            now = time.time()
            if now - last_print >= print_period and force_window:
                forces = np.array(force_window, dtype=np.float64)
                avg_fx = float(np.mean(forces[:, 0]))
                avg_fy = float(np.mean(forces[:, 1]))
                mag = float(np.sqrt(fx * fx + fy * fy))
                avg_mag = float(np.mean(np.sqrt(forces[:, 0] ** 2 + forces[:, 1] ** 2)))
                print(
                    f"t={data.time:6.2f}s | "
                    f"Fx={fx:8.3f} N  Fy={fy:8.3f} N  |F|={mag:8.3f} N | "
                    f"avg Fx={avg_fx:8.3f}  avg Fy={avg_fy:8.3f}  avg |F|={avg_mag:8.3f}",
                    flush=True,
                )
                last_print = now

            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
