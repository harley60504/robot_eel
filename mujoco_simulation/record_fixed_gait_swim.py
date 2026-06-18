from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

from hopf_cpg import HopfCPG, HopfCPGParams, amp_scales_to_mu_scales, degrees_to_radians
from plot_fixed_gait_trajectories import set_wall_collision
from sim_config import DEFAULT_START_X, DEFAULT_START_Y, EEL_MODEL_XML


HIDE_PREFIXES = ("wall_", "course_", "corner_marker", "start_marker")


def configure_paper_scene(model: mujoco.MjModel, *, floor_size: tuple[float, float]) -> None:
    set_wall_collision(model, False)
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name.startswith(HIDE_PREFIXES):
            model.geom_rgba[geom_id, 3] = 0.0
            model.geom_contype[geom_id] = 0
            model.geom_conaffinity[geom_id] = 0
        elif name == "floor":
            model.geom_pos[geom_id, 0] = DEFAULT_START_X
            model.geom_pos[geom_id, 1] = DEFAULT_START_Y
            model.geom_size[geom_id, 0] = floor_size[0]
            model.geom_size[geom_id, 1] = floor_size[1]
        elif name == "water_surface":
            model.geom_pos[geom_id, 0] = DEFAULT_START_X
            model.geom_pos[geom_id, 1] = DEFAULT_START_Y
            model.geom_size[geom_id, 0] = max(0.1, floor_size[0] - 0.05)
            model.geom_size[geom_id, 1] = max(0.1, floor_size[1] - 0.05)


def load_gait(path: Path) -> dict:
    gait = json.loads(Path(path).read_text(encoding="utf-8"))
    required = ("freq", "wavelength", "ajoint", "amp_scales", "phase_lags", "joint_bias")
    missing = [key for key in required if key not in gait]
    if missing:
        raise ValueError(f"gait json missing keys: {', '.join(missing)}")
    return gait


def make_cpg_params(gait: dict) -> HopfCPGParams:
    return HopfCPGParams(
        frequency=float(gait["freq"]),
        wavelength=float(gait["wavelength"]),
        ajoint=degrees_to_radians(float(gait["ajoint"])),
        mu_scales=amp_scales_to_mu_scales(tuple(gait["amp_scales"])),
        phase_lags=tuple(gait["phase_lags"]),
        joint_bias=tuple(gait["joint_bias"]),
    )


def bgr_from_renderer(renderer: mujoco.Renderer, data: mujoco.MjData, camera: mujoco.MjvCamera) -> np.ndarray:
    renderer.update_scene(data, camera=camera)
    rgb = renderer.render()
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def write_recording(
    *,
    gait_path: Path,
    out_dir: Path,
    seconds: float,
    fps: float,
    width: int,
    height: int,
    snapshot_interval: float,
    floor_size: tuple[float, float],
    camera_distance: float,
    camera_elevation: float,
    camera_mode: str,
    camera_lookat: tuple[float, float, float],
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    gait = load_gait(gait_path)
    name = str(gait.get("name") or gait_path.stem)

    model = mujoco.MjModel.from_xml_path(str(EEL_MODEL_XML))
    data = mujoco.MjData(model)
    model.opt.gravity[:] = (0, 0, 0)
    configure_paper_scene(model, floor_size=floor_size)

    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    base_xml_pos = model.body_pos[base_body_id]
    data.qpos[0] = DEFAULT_START_X - float(base_xml_pos[0])
    data.qpos[1] = DEFAULT_START_Y - float(base_xml_pos[1])
    mujoco.mj_forward(model, data)

    params = make_cpg_params(gait)
    cpg = HopfCPG(num_joints=6, params=params)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = float(camera_distance)
    camera.elevation = float(camera_elevation)
    camera.azimuth = 0.0
    camera.lookat[:] = np.asarray(camera_lookat, dtype=np.float64)

    video_path = out_dir / f"{name}_swim_18s.mp4"
    snapshots_dir = out_dir / f"{name}_snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer: {video_path}")

    snapshot_times = [
        round((idx + 1) * snapshot_interval, 6)
        for idx in range(int(seconds // snapshot_interval))
    ]
    if len(snapshot_times) != 9:
        snapshot_times = [round((idx + 1) * 2.0, 6) for idx in range(9)]
    snapshot_index = 0
    render_period = 1.0 / float(fps)
    next_render_time = 0.0
    records: list[tuple[float, float, float, float]] = []
    snapshot_rows = []

    with mujoco.Renderer(model, height=int(height), width=int(width)) as renderer:
        while data.time < seconds - 1e-12:
            targets = cpg.step(data.time, model.opt.timestep, params)
            data.ctrl[0:6] = np.clip(targets, -1.2, 1.2)
            mujoco.mj_step(model, data)
            base_pos = data.xpos[base_body_id].copy()
            records.append((float(data.time), float(base_pos[0]), float(base_pos[1]), float(data.qpos[2])))

            if camera_mode == "follow":
                camera.lookat[:] = np.array([float(base_pos[0]), float(base_pos[1]), camera_lookat[2]])
            if data.time + 1e-12 >= next_render_time:
                frame = bgr_from_renderer(renderer, data, camera)
                writer.write(frame)
                while snapshot_index < len(snapshot_times) and data.time + 1e-12 >= snapshot_times[snapshot_index]:
                    snap_path = snapshots_dir / f"{name}_t{snapshot_times[snapshot_index]:04.1f}s.png"
                    cv2.imwrite(str(snap_path), frame)
                    snapshot_rows.append((snapshot_times[snapshot_index], snap_path))
                    snapshot_index += 1
                next_render_time += render_period

        if snapshot_index < len(snapshot_times):
            frame = bgr_from_renderer(renderer, data, camera)
            while snapshot_index < len(snapshot_times):
                snap_path = snapshots_dir / f"{name}_t{snapshot_times[snapshot_index]:04.1f}s.png"
                cv2.imwrite(str(snap_path), frame)
                snapshot_rows.append((snapshot_times[snapshot_index], snap_path))
                snapshot_index += 1

    writer.release()
    if snapshot_index != len(snapshot_times):
        raise RuntimeError(f"wrote {snapshot_index} snapshots, expected {len(snapshot_times)}")

    trajectory_csv = out_dir / f"{name}_swim_18s_trajectory.csv"
    np.savetxt(trajectory_csv, np.asarray(records), delimiter=",", header="time,x,y,yaw", comments="")
    with (out_dir / f"{name}_swim_18s_snapshots.csv").open("w", newline="", encoding="utf-8") as handle:
        writer_csv = csv.writer(handle)
        writer_csv.writerow(["time_s", "snapshot_png"])
        writer_csv.writerows((time_s, str(path)) for time_s, path in snapshot_rows)

    summary = {
        "name": name,
        "gait_json": str(gait_path),
        "video_mp4": str(video_path),
        "snapshots_dir": str(snapshots_dir),
        "snapshot_count": len(snapshot_rows),
        "trajectory_csv": str(trajectory_csv),
        "seconds": seconds,
        "fps": fps,
        "width": width,
        "height": height,
        "wall_collision": False,
        "hidden_visual_prefixes": HIDE_PREFIXES,
        "floor_half_extents_m": floor_size,
        "camera_mode": camera_mode,
        "camera_lookat": camera_lookat,
    }
    summary_path = out_dir / f"{name}_swim_18s_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(video_path)
    print(snapshots_dir)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a paper-ready no-wall fixed-gait swim video and 2s snapshots.")
    parser.add_argument("--gait-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/paper_swim_recordings"))
    parser.add_argument("--seconds", type=float, default=18.0)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--snapshot-interval", type=float, default=2.0)
    parser.add_argument("--floor-half-x", type=float, default=4.0)
    parser.add_argument("--floor-half-y", type=float, default=3.0)
    parser.add_argument("--camera-distance", type=float, default=2.4)
    parser.add_argument("--camera-elevation", type=float, default=-70.0)
    parser.add_argument("--camera-mode", choices=("fixed", "follow"), default="fixed")
    parser.add_argument("--camera-lookat-x", type=float, default=DEFAULT_START_X)
    parser.add_argument("--camera-lookat-y", type=float, default=DEFAULT_START_Y)
    parser.add_argument("--camera-lookat-z", type=float, default=-0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_recording(
        gait_path=args.gait_json,
        out_dir=args.out_dir,
        seconds=args.seconds,
        fps=args.fps,
        width=args.width,
        height=args.height,
        snapshot_interval=args.snapshot_interval,
        floor_size=(args.floor_half_x, args.floor_half_y),
        camera_distance=args.camera_distance,
        camera_elevation=args.camera_elevation,
        camera_mode=args.camera_mode,
        camera_lookat=(args.camera_lookat_x, args.camera_lookat_y, args.camera_lookat_z),
    )


if __name__ == "__main__":
    main()
