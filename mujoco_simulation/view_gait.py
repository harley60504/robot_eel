from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from sim_config import DEFAULT_START_X, DEFAULT_START_Y, EEL_MODEL_XML, RESET_X_MAX, RESET_X_MIN, RESET_Y, TANK_CENTER_X
from hopf_cpg import HopfCPG, HopfCPGParams, amp_scales_to_mu_scales, degrees_to_radians


def parse_args():
    parser = argparse.ArgumentParser(description="View a saved Hopf CPG gait JSON.")
    parser.add_argument("gait", type=Path, help="Path to a gait JSON file.")
    parser.add_argument("--xml", default=EEL_MODEL_XML)
    parser.add_argument("--print-hz", type=float, default=2.0)
    parser.add_argument("--start-x", type=float, default=DEFAULT_START_X)
    parser.add_argument("--start-y", type=float, default=DEFAULT_START_Y)
    parser.add_argument("--print-contacts", action="store_true")
    parser.add_argument(
        "--contact-ignore-seconds",
        type=float,
        default=0.5,
        help="Ignore wall contacts during the initial drop/reset transient.",
    )
    return parser.parse_args()


def load_gait(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        gait = json.load(f)

    required = ("ajoint", "freq", "wavelength", "amp_scales", "phase_lags", "joint_bias")
    missing = [key for key in required if key not in gait]
    if missing:
        raise ValueError(f"{path} is missing required fields: {missing}")
    if len(gait["amp_scales"]) != 6:
        raise ValueError("amp_scales must have 6 values")
    if len(gait["phase_lags"]) != 5:
        raise ValueError("phase_lags must have 5 values")
    if len(gait["joint_bias"]) != 6:
        raise ValueError("joint_bias must have 6 values")
    return gait


def main():
    args = parse_args()
    gait = load_gait(args.gait)

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    model.opt.gravity[:] = (0, 0, 0)
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

    wall_geom_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in ("wall_bottom", "wall_top", "wall_left", "wall_right")
    }
    wall_geom_ids.discard(-1)

    cpg = HopfCPG(num_joints=6)
    ajoint_deg = float(gait["ajoint"])
    ajoint_rad = degrees_to_radians(ajoint_deg)
    cpg_params = HopfCPGParams(
        frequency=float(gait["freq"]),
        wavelength=float(gait["wavelength"]),
        ajoint=ajoint_rad,
        mu_scales=amp_scales_to_mu_scales(tuple(float(value) for value in gait["amp_scales"])),
        phase_lags=tuple(float(value) for value in gait["phase_lags"]),
        joint_bias=tuple(float(value) for value in gait["joint_bias"]),
    )
    last_print = 0.0
    print_period = 1.0 / max(args.print_hz, 1e-6)

    print(f"Loaded gait: {gait.get('name', args.gait.stem)}")
    print(f"  file={args.gait}")
    print(f"  ajoint={ajoint_deg:.3f} deg ({cpg_params.ajoint:.3f} rad), freq={cpg_params.frequency}, wavelength={cpg_params.wavelength}")
    print("  joint_bias=", ", ".join(f"{value:.3f}" for value in cpg_params.joint_bias or ()))
    print("  MuJoCo adapter: servo joint axes are axis=\"0 0 -1\" in eel.xml")

    def reset_to_start():
        mujoco.mj_resetData(model, data)
        base_xml_pos = model.body_pos[base_body_id]
        data.qpos[0] = args.start_x - base_xml_pos[0]
        data.qpos[1] = args.start_y - base_xml_pos[1]
        cpg.reset()
        mujoco.mj_forward(model, data)

    def detect_wall_contact() -> tuple[bool, list[str]]:
        examples: list[str] = []
        hit_wall = False
        for i in range(data.ncon):
            contact = data.contact[i]
            if contact.geom1 in wall_geom_ids or contact.geom2 in wall_geom_ids:
                hit_wall = True
                g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or f"geom{contact.geom1}"
                g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or f"geom{contact.geom2}"
                examples.append(f"{g1}<->{g2}")
        return hit_wall, examples

    reset_to_start()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = np.array([args.start_x, args.start_y, -0.02])
            viewer.cam.distance = 1.4
            viewer.cam.elevation = -70
            viewer.cam.azimuth = 0

        while viewer.is_running():
            targets = cpg.step(data.time, model.opt.timestep, cpg_params)
            data.ctrl[0:6] = np.clip(targets, -1.2, 1.2)
            mujoco.mj_step(model, data)
            base_pos = data.xpos[base_body_id]

            hit_wall, contact_examples = detect_wall_contact()
            if hit_wall and data.time >= args.contact_ignore_seconds:
                if args.print_contacts:
                    examples = sorted(set(contact_examples))[:3]
                    print(f"reset to start: wall contact {examples}", flush=True)
                else:
                    print(f"reset to start: wall contact x={base_pos[0]:.3f}, y={base_pos[1]:.3f}", flush=True)
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
