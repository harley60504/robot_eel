from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np

from hopf_cpg import DEFAULT_AJOINT_DEG, HopfCPG, HopfCPGParams, amp_scales_to_mu_scales, degrees_to_radians


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune per-joint Hopf CPG amplitude scales and phase lags."
    )
    parser.add_argument("--xml", default="eel_tethered.xml")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/hopf_shape_auto"))
    parser.add_argument("--ajoint", type=float, default=DEFAULT_AJOINT_DEG, help="Base joint angle amplitude in degrees.")
    parser.add_argument("--freq", type=float, default=1.0)
    parser.add_argument("--wavelength", type=float, default=1.6275)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--warmup-seconds", type=float, default=2.0)
    parser.add_argument("--verify-seconds", type=float, default=15.0)
    parser.add_argument("--fy-weight", type=float, default=0.02)
    parser.add_argument("--energy-weight", type=float, default=0.02)
    parser.add_argument("--amp-min", type=float, default=0.65)
    parser.add_argument("--amp-max", type=float, default=1.25)
    parser.add_argument("--phase-min", type=float, default=0.45)
    parser.add_argument("--phase-max", type=float, default=0.85)
    parser.add_argument("--random-starts", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--amp-step", type=float, default=0.08)
    parser.add_argument("--phase-step", type=float, default=0.04)
    parser.add_argument("--top", type=int, default=12)
    return parser.parse_args()


def clamp_vector(values: np.ndarray, lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
    return np.minimum(highs, np.maximum(lows, values))


def score(metrics: dict[str, float], fy_weight: float, energy_weight: float) -> float:
    return (
        metrics["mean_fx"]
        - fy_weight * metrics["abs_mean_fy"]
        - energy_weight * metrics["energy_proxy"]
    )


def measure_case(
    model: mujoco.MjModel,
    *,
    freq: float,
    wavelength: float,
    ajoint: float,
    amp_scales: tuple[float, ...],
    phase_lags: tuple[float, ...],
    seconds: float,
    warmup_seconds: float,
) -> dict[str, float]:
    ajoint_rad = degrees_to_radians(ajoint)
    data = mujoco.MjData(model)
    cpg = HopfCPG(
        num_joints=6,
        params=HopfCPGParams(
            frequency=freq,
            wavelength=wavelength,
            ajoint=ajoint_rad,
            mu_scales=amp_scales_to_mu_scales(amp_scales),
            phase_lags=phase_lags,
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
    force_mag = np.sqrt(fx**2 + fy**2)
    return {
        "mean_fx": float(np.mean(fx)),
        "mean_fy": float(np.mean(fy)),
        "abs_mean_fy": float(abs(np.mean(fy))),
        "mean_abs_fy": float(np.mean(np.abs(fy))),
        "mean_force_mag": float(np.mean(force_mag)),
        "peak_force_mag": float(np.max(force_mag)),
        "energy_proxy": float(np.mean(energy_values)),
    }


def make_row(
    *,
    values: np.ndarray,
    metrics: dict[str, float],
    score_value: float,
    start: int,
    iteration: int,
) -> dict[str, float]:
    row = {
        "start": start,
        "iteration": iteration,
        "score": score_value,
        **{f"amp_scale_{i + 1}": float(values[i]) for i in range(6)},
        **{f"phase_lag_{i + 1}": float(values[6 + i]) for i in range(5)},
        **metrics,
    }
    row["fx_over_abs_fy"] = (
        float(row["mean_fx"] / row["abs_mean_fy"]) if row["abs_mean_fy"] > 1e-9 else float("inf")
    )
    return row


def print_top(rows: list[dict[str, float]], top: int):
    for index, row in enumerate(rows[:top], start=1):
        amps = ",".join(f"{row[f'amp_scale_{i + 1}']:.3f}" for i in range(6))
        phases = ",".join(f"{row[f'phase_lag_{i + 1}']:.3f}" for i in range(5))
        print(
            f"{index:2d}. score={row['score']:.4f} Fx={row['mean_fx']:.4f} "
            f"|mean Fy|={row['abs_mean_fy']:.4f} energy={row['energy_proxy']:.4f} "
            f"amps=[{amps}] phases=[{phases}]"
        )


def write_rows(path: Path, rows: list[dict[str, float]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    model = mujoco.MjModel.from_xml_path(args.xml)
    model.opt.gravity[:] = (0, 0, 0)

    base_phase = 1.0 / max(1e-6, args.wavelength)
    center = np.array([1.0] * 6 + [base_phase] * 5, dtype=np.float64)
    lows = np.array([args.amp_min] * 6 + [args.phase_min] * 5, dtype=np.float64)
    highs = np.array([args.amp_max] * 6 + [args.phase_max] * 5, dtype=np.float64)
    step_sizes = np.array([args.amp_step] * 6 + [args.phase_step] * 5, dtype=np.float64)

    starts = [clamp_vector(center, lows, highs)]
    for _ in range(args.random_starts):
        starts.append(rng.uniform(lows, highs))

    rows: list[dict[str, float]] = []
    for start_index, start_values in enumerate(starts, start=1):
        values = start_values.copy()
        steps = step_sizes.copy()
        metrics = measure_case(
            model,
            freq=args.freq,
            wavelength=args.wavelength,
            ajoint=args.ajoint,
            amp_scales=tuple(values[:6]),
            phase_lags=tuple(values[6:]),
            seconds=args.seconds,
            warmup_seconds=args.warmup_seconds,
        )
        best = make_row(
            values=values,
            metrics=metrics,
            score_value=score(metrics, args.fy_weight, args.energy_weight),
            start=start_index,
            iteration=0,
        )
        rows.append(best)
        print(
            f"\nstart {start_index}: score={best['score']:.4f} "
            f"Fx={best['mean_fx']:.4f} |mean Fy|={best['abs_mean_fy']:.4f}"
        )

        for iteration in range(1, args.iters + 1):
            improved = False
            candidate_best = best
            for dim in range(11):
                for direction in (-1.0, 1.0):
                    candidate = values.copy()
                    candidate[dim] += direction * steps[dim]
                    candidate = clamp_vector(candidate, lows, highs)
                    if np.allclose(candidate, values):
                        continue
                    metrics = measure_case(
                        model,
                        freq=args.freq,
                        wavelength=args.wavelength,
                        ajoint=args.ajoint,
                        amp_scales=tuple(candidate[:6]),
                        phase_lags=tuple(candidate[6:]),
                        seconds=args.seconds,
                        warmup_seconds=args.warmup_seconds,
                    )
                    row = make_row(
                        values=candidate,
                        metrics=metrics,
                        score_value=score(metrics, args.fy_weight, args.energy_weight),
                        start=start_index,
                        iteration=iteration,
                    )
                    rows.append(row)
                    if row["score"] > candidate_best["score"]:
                        candidate_best = row
                        improved = True

            if improved:
                best = candidate_best
                values = np.array(
                    [best[f"amp_scale_{i + 1}"] for i in range(6)]
                    + [best[f"phase_lag_{i + 1}"] for i in range(5)],
                    dtype=np.float64,
                )
                print(
                    f"  iter {iteration}: improve score={best['score']:.4f} "
                    f"Fx={best['mean_fx']:.4f} |mean Fy|={best['abs_mean_fy']:.4f}"
                )
            else:
                steps *= 0.5
                print(f"  iter {iteration}: shrink max_step={np.max(steps):.5f}")

    rows.sort(key=lambda row: row["score"], reverse=True)
    csv_path = args.output_dir / "shape_search.csv"
    write_rows(csv_path, rows)

    best = dict(rows[0])
    best_values = np.array(
        [best[f"amp_scale_{i + 1}"] for i in range(6)]
        + [best[f"phase_lag_{i + 1}"] for i in range(5)],
        dtype=np.float64,
    )
    verify_metrics = measure_case(
        model,
        freq=args.freq,
        wavelength=args.wavelength,
        ajoint=args.ajoint,
        amp_scales=tuple(best_values[:6]),
        phase_lags=tuple(best_values[6:]),
        seconds=args.verify_seconds,
        warmup_seconds=args.warmup_seconds,
    )
    best["verify_score"] = score(verify_metrics, args.fy_weight, args.energy_weight)
    for key, value in verify_metrics.items():
        best[f"verify_{key}"] = float(value)

    amp_arg = ",".join(f"{value:.6g}" for value in best_values[:6])
    phase_arg = ",".join(f"{value:.6g}" for value in best_values[6:])
    result = {
        "xml": args.xml,
        "freq": args.freq,
        "wavelength": args.wavelength,
        "ajoint": args.ajoint,
        "fy_weight": args.fy_weight,
        "energy_weight": args.energy_weight,
        "best": best,
        "files": {"csv": str(csv_path)},
        "commands": {
            "view_tethered": (
                "python view_tethered_force.py "
                f"--ajoint {args.ajoint:.6g} --freq {args.freq:.6g} --wavelength {args.wavelength:.6g} "
                f"--amp-scales {amp_arg} --phase-lags {phase_arg}"
            ),
            "measure_tethered": (
                "python measure_tethered_force.py "
                f"--ajoint {args.ajoint:.6g} --freq {args.freq:.6g} --wavelength {args.wavelength:.6g} "
                f"--amp-scales {amp_arg} --phase-lags {phase_arg}"
            ),
            "measure_free": (
                "python measure_free_swim_speed.py "
                f"--ajoint {args.ajoint:.6g} --freq {args.freq:.6g} --wavelength {args.wavelength:.6g} "
                f"--amp-scales {amp_arg} --phase-lags {phase_arg}"
            ),
        },
    }

    json_path = args.output_dir / "best_hopf_shape.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\nwrote {csv_path}")
    print(f"wrote {json_path}")
    print("top:")
    print_top(rows, args.top)
    print("verified best:")
    print(
        f"  score={best['verify_score']:.4f} Fx={best['verify_mean_fx']:.4f} "
        f"|mean Fy|={best['verify_abs_mean_fy']:.4f} energy={best['verify_energy_proxy']:.4f}"
    )
    print("next commands:")
    print(f"  {result['commands']['view_tethered']}")
    print(f"  {result['commands']['measure_free']}")


if __name__ == "__main__":
    main()
