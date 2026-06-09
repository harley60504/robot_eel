from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from sim_config import DEFAULT_START_X, DEFAULT_START_Y, EEL_MODEL_XML, TANK_CENTER_X
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
    parser.add_argument(
        "--metrics-warmup-seconds",
        type=float,
        default=0.5,
        help="Ignore the first seconds when computing average speed, yaw rate, and turn radius.",
    )
    parser.add_argument(
        "--reset-on-wall",
        action="store_true",
        help="Old behavior: reset to start after wall contact instead of ending the run and printing one summary.",
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


def print_turn_summary(records: list[tuple[float, float, float, float, float, float, float]], gait_name: str):
    print("\n================ Turn radius summary ================", flush=True)
    print(f"gait: {gait_name}", flush=True)

    if len(records) < 2:
        print("not enough samples after warmup to estimate R", flush=True)
        print("=====================================================\n", flush=True)
        return

    arr = np.asarray(records, dtype=np.float64)
    t = arr[:, 0]
    x = arr[:, 1]
    y = arr[:, 2]
    yaw = np.unwrap(arr[:, 3])
    vx = arr[:, 4]
    vy = arr[:, 5]
    wz = arr[:, 6]

    duration = float(t[-1] - t[0])
    if duration <= 1e-9:
        print("duration too short to estimate R", flush=True)
        print("=====================================================\n", flush=True)
        return

    dx = float(x[-1] - x[0])
    dy = float(y[-1] - y[0])
    path_distance = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
    straight_distance = float(np.hypot(dx, dy))

    yaw_change = float(yaw[-1] - yaw[0])
    mean_yaw_rate_from_yaw = yaw_change / duration
    mean_yaw_rate_from_qvel = float(np.mean(wz))
    mean_speed = float(np.mean(np.hypot(vx, vy)))
    mean_vx = float(np.mean(vx))
    mean_vy = float(np.mean(vy))

    signed_r_from_yaw = np.inf if abs(mean_yaw_rate_from_yaw) < 1e-9 else mean_speed / mean_yaw_rate_from_yaw
    signed_r_from_qvel = np.inf if abs(mean_yaw_rate_from_qvel) < 1e-9 else mean_speed / mean_yaw_rate_from_qvel

    print(f"samples: {len(records)}", flush=True)
    print(f"duration: {duration:.3f} s", flush=True)
    print(f"start: x={x[0]:.4f}, y={y[0]:.4f}, yaw={yaw[0]:.4f}", flush=True)
    print(f"end:   x={x[-1]:.4f}, y={y[-1]:.4f}, yaw={yaw[-1]:.4f}", flush=True)
    print(f"dx={dx:.4f} m, dy={dy:.4f} m", flush=True)
    print(f"path distance={path_distance:.4f} m, straight distance={straight_distance:.4f} m", flush=True)
    print(f"mean vx={mean_vx:.4f} m/s, mean vy={mean_vy:.4f} m/s", flush=True)
    print(f"mean speed={mean_speed:.4f} m/s", flush=True)
    print(f"yaw change={yaw_change:.4f} rad", flush=True)
    print(f"mean yaw rate from yaw={mean_yaw_rate_from_yaw:.4f} rad/s", flush=True)
    print(f"mean yaw rate from qvel={mean_yaw_rate_from_qvel:.4f} rad/s", flush=True)
    print(f"signed R from yaw={signed_r_from_yaw:.4f} m", flush=True)
    print(f"average R from yaw={abs(signed_r_from_yaw):.4f} m", flush=True)
    print(f"signed R from qvel={signed_r_from_qvel:.4f} m", flush=True)
    print(f"average R from qvel={abs(signed_r_from_qvel):.4f} m", flush=True)
    print("=====================================================\n", flush=True)


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
    metric_records: list[tuple[float, float, float, float, float, float, float]] = []

    gait_name = gait.get("name", args.gait.stem)

    print(f"Loaded gait: {gait_name}", flush=True)
    print(f"  file={args.gait}", flush=True)
    print(f"  ajoint={ajoint_deg:.3f} deg ({cpg_params.ajoint:.3f} rad), freq={cpg_params.frequency}, wavelength={cpg_params.wavelength}", flush=True)
    print("  joint_bias=", ", ".join(f"{value:.3f}" for value in cpg_params.joint_bias or ()), flush=True)
    print("  MuJoCo adapter: servo joint axes are axis=\"0 0 -1\" in eel.xml", flush=True)
    print("  wall behavior: one run until first wall contact, then print average R", flush=True)

    def reset_to_start():
        nonlocal metric_records
        mujoco.mj_resetData(model, data)
        base_xml_pos = model.body_pos[base_body_id]
        data.qpos[0] = args.start_x - base_xml_pos[0]
        data.qpos[1] = args.start_y - base_xml_pos[1]
        cpg.reset()
        metric_records = []
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

            if data.time >= args.metrics_warmup_seconds:
                metric_records.append(
                    (
                        float(data.time),
                        float(base_pos[0]),
                        float(base_pos[1]),
                        float(data.qpos[2]),
                        float(data.qvel[0]),
                        float(data.qvel[1]),
                        float(data.qvel[2]),
                    )
                )

            hit_wall, contact_examples = detect_wall_contact()
            if hit_wall and data.time >= args.contact_ignore_seconds:
                if args.print_contacts:
                    examples = sorted(set(contact_examples))[:3]
                    print(f"wall contact: {examples}", flush=True)
                else:
                    print(f"wall contact: x={base_pos[0]:.3f}, y={base_pos[1]:.3f}", flush=True)

                print_turn_summary(metric_records, gait_name)

                if args.reset_on_wall:
                    print("reset to start", flush=True)
                    reset_to_start()
                    base_pos = data.xpos[base_body_id]
                else:
                    break

            now = time.time()
            if now - last_print >= print_period:
                print(
                    f"t={data.time:6.2f}s | "
                    f"x={base_pos[0]:8.3f} y={base_pos[1]:8.3f} yaw={data.qpos[2]:8.3f} | "
                    f"vx={data.qvel[0]:8.3f} vy={data.qvel[1]:8.3f} wz={data.qvel[2]:8.3f}",
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
