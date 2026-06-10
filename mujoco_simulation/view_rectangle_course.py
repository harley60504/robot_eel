from __future__ import annotations

import argparse
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

from hopf_cpg import DEFAULT_AJOINT_DEG, HopfCPG, HopfCPGParams, degrees_to_radians, wrap_pi
from rectangle_path import RectanglePath
from sim_config import RECTANGLE_CONTROL_SIGN, RECTANGLE_MODEL_XML, RECTANGLE_PATH_CENTER_X, RECTANGLE_PATH_CENTER_Y, RECTANGLE_PATH_HALF_X, RECTANGLE_PATH_HALF_Y, RECTANGLE_WAYPOINTS


def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except OSError:
        sys.stdout = open("NUL", "w", encoding="utf-8")


def parse_float_list(value: str, expected_len: int, name: str) -> tuple[float, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != expected_len:
        raise argparse.ArgumentTypeError(f"{name} needs {expected_len} comma-separated values")
    return tuple(float(part) for part in parts)


def parse_waypoints(value: str) -> np.ndarray:
    points = []
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        xy = [part.strip() for part in item.split(",")]
        if len(xy) != 2:
            raise argparse.ArgumentTypeError("waypoints must look like x,y;x,y;x,y")
        points.append((float(xy[0]), float(xy[1])))
    if len(points) < 2:
        raise argparse.ArgumentTypeError("at least two waypoints are required")
    return np.asarray(points, dtype=np.float64)


def steering_profile(value: float) -> tuple[float, ...]:
    weights = np.array([0.42, 0.52, 0.64, 0.76, 0.88, 1.0], dtype=np.float64)
    return tuple(float(value * weight) for weight in weights)


def turning_amp_scales(base_scales: tuple[float, ...], steer: float, gain: float) -> tuple[float, ...]:
    if gain <= 0.0:
        return base_scales
    base = np.asarray(base_scales, dtype=np.float64)
    tail_weights = np.array([0.0, 0.0, 0.12, 0.28, 0.52, 0.78], dtype=np.float64)
    multiplier = 1.0 + gain * abs(float(steer)) * tail_weights
    return tuple(float(value) for value in np.clip(base * multiplier, 0.2, 1.45))


def amp_scales_to_mu_scales(amp_scales: tuple[float, ...]) -> tuple[float, ...]:
    values = np.asarray(amp_scales, dtype=np.float64)
    return tuple(float(value * value) for value in values)


def soft_limit(value: float, limit: float) -> float:
    limit = abs(float(limit))
    if limit <= 1e-9:
        return 0.0
    return float(limit * np.tanh(float(value) / limit))


def parse_args():
    parser = argparse.ArgumentParser(description="View an eel following a 3 m x 1.5 m rectangle course.")
    parser.add_argument("--xml", default=RECTANGLE_MODEL_XML)
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
        "--waypoints",
        type=parse_waypoints,
        default=parse_waypoints(RECTANGLE_WAYPOINTS),
        help="Semicolon-separated waypoint list, for example: x,y;x,y;x,y",
    )
    parser.add_argument("--controller", choices=("pure_pursuit", "waypoint"), default="pure_pursuit")
    parser.add_argument("--path-half-x", type=float, default=RECTANGLE_PATH_HALF_X)
    parser.add_argument("--path-half-y", type=float, default=RECTANGLE_PATH_HALF_Y)
    parser.add_argument("--path-center-x", type=float, default=RECTANGLE_PATH_CENTER_X)
    parser.add_argument("--path-center-y", type=float, default=RECTANGLE_PATH_CENTER_Y)
    parser.add_argument("--lookahead", type=float, default=0.95)
    parser.add_argument("--reach-radius", type=float, default=0.25)
    parser.add_argument("--steer-gain", type=float, default=0.50)
    parser.add_argument("--max-bias", type=float, default=0.30)
    parser.add_argument(
        "--turn-amp-gain",
        type=float,
        default=0.60,
        help="Increase CPG amplitude target as |steer| grows. 0 disables turning amplitude modulation.",
    )
    parser.add_argument(
        "--steer-time-constant",
        type=float,
        default=0.18,
        help="Seconds for steering low-pass response. Larger values make maximum turns smoother.",
    )
    parser.add_argument(
        "--steer-rate-limit",
        type=float,
        default=1.6,
        help="Maximum steering-bias change per second. Prevents jerky saturation at maximum turn.",
    )
    parser.add_argument("--control-sign", type=float, default=RECTANGLE_CONTROL_SIGN, help="Use -1 when rectangle mode uses the unified eel.xml joint axes.")
    parser.add_argument("--print-hz", type=float, default=2.0)
    parser.add_argument("--viewer-fps", type=float, default=60.0, help="Viewer render FPS used for real-time pacing.")
    parser.add_argument("--follow-camera", action="store_true")
    parser.add_argument("--print-contacts", action="store_true")
    parser.add_argument(
        "--contact-ignore-seconds",
        type=float,
        default=0.5,
        help="Ignore wall contacts during the initial drop/reset transient.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    model.opt.gravity[:] = (0, 0, 0)
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

    cpg = HopfCPG(num_joints=6)
    ajoint_rad = degrees_to_radians(args.ajoint)
    path = RectanglePath(args.path_half_x, args.path_half_y, center_x=args.path_center_x, center_y=args.path_center_y)
    wall_geom_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in ("wall_bottom", "wall_top", "wall_left", "wall_right")
    }
    wall_geom_ids.discard(-1)

    waypoint_index = 0
    laps = 0
    last_path_s = 0.0
    last_print = 0.0
    print_period = 1.0 / max(args.print_hz, 1e-6)
    wall_contact_count = 0
    wall_contact_examples: set[str] = set()
    steer_state = 0.0
    distance = 0.0
    steer = 0.0

    safe_print("Rectangle course viewer", flush=True)
    safe_print("  viewer pacing: real-time wall-clock pacing with batched MuJoCo steps", flush=True)
    safe_print(
        "  turn smoothing: soft steering limit, low-pass, and rate limit enabled",
        flush=True,
    )

    def reset_to_start():
        nonlocal waypoint_index, laps, steer_state, last_path_s
        mujoco.mj_resetData(model, data)
        cpg.reset()
        waypoint_index = 0
        laps = 0
        steer_state = 0.0
        last_path_s = 0.0
        mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = np.array([1.5, 0.0, -0.02])
            viewer.cam.distance = 3.4
            viewer.cam.elevation = -90
            viewer.cam.azimuth = 0

        target_fps = max(args.viewer_fps, 1.0)
        frame_dt = 1.0 / target_fps
        last_wall_clock = time.perf_counter()

        while viewer.is_running():
            frame_start = time.perf_counter()
            wall_dt = min(frame_start - last_wall_clock, 0.05)
            last_wall_clock = frame_start
            target_sim_time = data.time + wall_dt
            base_pos = data.xpos[base_body_id]

            while data.time + 1e-12 < target_sim_time:
                base_pos = data.xpos[base_body_id].copy()

                if args.controller == "pure_pursuit":
                    path_s = path.closest_s(base_pos[:2])
                    if path_s + path.total_length * laps < last_path_s - 0.5 * path.total_length:
                        laps += 1
                    last_path_s = path_s + path.total_length * laps
                    target = path.point_at(path_s + args.lookahead)
                    segment_index, _ = path.progress_info(path_s)
                    waypoint_index = segment_index
                    delta = target - base_pos[:2]
                    distance = float(np.linalg.norm(delta))
                else:
                    waypoint = args.waypoints[waypoint_index]
                    delta = waypoint - base_pos[:2]
                    distance = float(np.linalg.norm(delta))

                    if distance < args.reach_radius:
                        waypoint_index = (waypoint_index + 1) % len(args.waypoints)
                        if waypoint_index == 0:
                            laps += 1
                        waypoint = args.waypoints[waypoint_index]
                        delta = waypoint - base_pos[:2]
                        distance = float(np.linalg.norm(delta))

                desired_yaw = float(np.arctan2(delta[1], delta[0]))
                yaw = float(data.qpos[2])
                heading_error = float(wrap_pi(desired_yaw - yaw))
                raw_steer = -args.steer_gain * heading_error
                target_steer = soft_limit(raw_steer, args.max_bias)

                dt = float(model.opt.timestep)
                tau = max(float(args.steer_time_constant), dt)
                alpha = 1.0 - float(np.exp(-dt / tau))
                filtered_steer = steer_state + alpha * (target_steer - steer_state)
                max_delta = max(float(args.steer_rate_limit), 0.0) * dt
                if max_delta > 0.0:
                    filtered_steer = steer_state + float(np.clip(filtered_steer - steer_state, -max_delta, max_delta))
                steer_state = filtered_steer
                steer = steer_state

                joint_bias = steering_profile(steer)
                target_amp_scales = turning_amp_scales(args.amp_scales, steer, args.turn_amp_gain)
                mu_scales = amp_scales_to_mu_scales(target_amp_scales)

                cpg_params = HopfCPGParams(
                    frequency=args.freq,
                    wavelength=args.wavelength,
                    ajoint=ajoint_rad,
                    mu_scales=mu_scales,
                    phase_lags=args.phase_lags,
                    joint_bias=joint_bias,
                )
                targets = cpg.step(data.time, model.opt.timestep, cpg_params)
                data.ctrl[0:6] = args.control_sign * np.clip(targets, -1.2, 1.2)
                mujoco.mj_step(model, data)

                if args.print_contacts and data.time >= args.contact_ignore_seconds:
                    for i in range(data.ncon):
                        contact = data.contact[i]
                        if contact.geom1 in wall_geom_ids or contact.geom2 in wall_geom_ids:
                            g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or f"geom{contact.geom1}"
                            g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or f"geom{contact.geom2}"
                            wall_contact_count += 1
                            wall_contact_examples.add(f"{g1}<->{g2}")

                base_pos = data.xpos[base_body_id]

            now = time.time()
            if now - last_print >= print_period:
                contact_summary = ""
                if args.print_contacts:
                    examples = sorted(wall_contact_examples)[:3]
                    contact_summary = f" | wall contact events={wall_contact_count} {examples}"
                safe_print(
                    f"t={data.time:6.2f}s | lap={laps} wp={waypoint_index + 1}/{len(args.waypoints)} "
                    f"dist={distance:5.2f} steer={steer:6.3f} | "
                    f"x={base_pos[0]:7.3f} y={base_pos[1]:7.3f} yaw={data.qpos[2]:7.3f}"
                    + contact_summary,
                    flush=True,
                )
                wall_contact_count = 0
                wall_contact_examples.clear()
                last_print = now

            if args.follow_camera:
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
