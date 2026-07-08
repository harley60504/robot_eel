from __future__ import annotations

import argparse
import csv
from pathlib import Path

import mujoco
import numpy as np

from hopf_cpg import DEFAULT_AJOINT_DEG, HopfCPG, HopfCPGParams, degrees_to_radians


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep Hopf CPG frequency/wavelength on the tethered eel.")
    parser.add_argument("--xml", default="eel_tethered.xml")
    parser.add_argument("--freq-min", type=float, default=0.8)
    parser.add_argument("--freq-max", type=float, default=1.3)
    parser.add_argument("--freq-count", type=int, default=11)
    parser.add_argument("--wavelength-min", type=float, default=1.2)
    parser.add_argument("--wavelength-max", type=float, default=1.8)
    parser.add_argument("--wavelength-count", type=int, default=13)
    parser.add_argument("--ajoint", type=float, default=DEFAULT_AJOINT_DEG, help="Base joint angle amplitude in degrees.")
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument("--fy-weight", type=float, default=0.005)
    parser.add_argument("--csv", type=Path, default=Path("outputs/hopf_param_sweep.csv"))
    parser.add_argument("--top", type=int, default=10)
    return parser.parse_args()


def measure_case(model, freq: float, wavelength: float, ajoint: float, seconds: float, warmup_seconds: float):
    ajoint_rad = degrees_to_radians(ajoint)
    data = mujoco.MjData(model)
    cpg = HopfCPG(
        num_joints=6,
        params=HopfCPGParams(
            frequency=freq,
            wavelength=wavelength,
            ajoint=ajoint_rad,
            fb_phase=0.0,
            fb_amp=0.0,
        ),
    )

    root_x_dof = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_x")]
    root_y_dof = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_y")]

    steps = int(round(seconds / model.opt.timestep))
    warmup_steps = int(round(warmup_seconds / model.opt.timestep))
    fx_values = []
    fy_values = []
    energy_values = []

    for step in range(steps):
        targets = cpg.step(data.time, model.opt.timestep)
        data.ctrl[0:3] = 0.0
        data.ctrl[3:9] = np.clip(targets, -1.2, 1.2)
        mujoco.mj_step(model, data)

        if step >= warmup_steps:
            fx_values.append(-data.qfrc_actuator[root_x_dof])
            fy_values.append(-data.qfrc_actuator[root_y_dof])
            energy_values.append(float(np.mean(np.square(data.ctrl[3:9]))))

    fx = np.asarray(fx_values, dtype=np.float64)
    fy = np.asarray(fy_values, dtype=np.float64)
    return {
        "mean_fx": float(np.mean(fx)),
        "mean_fy": float(np.mean(fy)),
        "abs_mean_fy": float(abs(np.mean(fy))),
        "mean_abs_fy": float(np.mean(np.abs(fy))),
        "mean_force_mag": float(np.mean(np.sqrt(fx**2 + fy**2))),
        "peak_force_mag": float(np.max(np.sqrt(fx**2 + fy**2))),
        "energy_proxy": float(np.mean(energy_values)),
    }


def main():
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(args.xml)
    model.opt.gravity[:] = (0, 0, 0)

    freqs = np.linspace(args.freq_min, args.freq_max, args.freq_count)
    wavelengths = np.linspace(args.wavelength_min, args.wavelength_max, args.wavelength_count)
    rows = []

    total = len(freqs) * len(wavelengths)
    done = 0
    for freq in freqs:
        for wavelength in wavelengths:
            done += 1
            metrics = measure_case(model, float(freq), float(wavelength), args.ajoint, args.seconds, args.warmup_seconds)
            score = metrics["mean_fx"] - args.fy_weight * metrics["abs_mean_fy"]
            row = {
                "frequency": float(freq),
                "wavelength": float(wavelength),
                "ajoint": args.ajoint,
                "score": float(score),
                **metrics,
            }
            rows.append(row)
            print(
                f"[{done:3d}/{total}] f={freq:.3f} wl={wavelength:.3f} "
                f"score={score:.4f} Fx={metrics['mean_fx']:.4f} |mean Fy|={metrics['abs_mean_fy']:.4f}",
                flush=True,
            )

    rows.sort(key=lambda r: r["score"], reverse=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nwrote {args.csv}")
    print(f"top {args.top}:")
    for i, row in enumerate(rows[: args.top], start=1):
        print(
            f"{i:2d}. score={row['score']:.4f} "
            f"Fx={row['mean_fx']:.4f} |mean Fy|={row['abs_mean_fy']:.4f} "
            f"f={row['frequency']:.3f} wl={row['wavelength']:.3f}"
        )


if __name__ == "__main__":
    main()
