from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np

from hopf_cpg import DEFAULT_AJOINT_DEG
from sweep_hopf_params import measure_case


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automatically coarse-sweep, fine-sweep, and verify Hopf CPG parameters."
    )
    parser.add_argument("--xml", default="eel_tethered.xml")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/hopf_auto"))
    parser.add_argument("--ajoint", type=float, default=DEFAULT_AJOINT_DEG, help="Base joint angle amplitude in degrees.")
    parser.add_argument("--ajoint-min", type=float, default=None)
    parser.add_argument("--ajoint-max", type=float, default=None)
    parser.add_argument("--ajoint-count", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument("--verify-seconds", type=float, default=8.0)
    parser.add_argument("--fy-weight", type=float, default=0.02)
    parser.add_argument("--dominance-weight", type=float, default=1.0)
    parser.add_argument("--max-fy-over-fx", type=float, default=1.0)
    parser.add_argument("--min-fx", type=float, default=0.5)
    parser.add_argument("--low-fx-weight", type=float, default=3.0)
    parser.add_argument("--top", type=int, default=12)

    parser.add_argument("--coarse-freq-min", type=float, default=0.8)
    parser.add_argument("--coarse-freq-max", type=float, default=1.3)
    parser.add_argument("--coarse-freq-count", type=int, default=11)
    parser.add_argument("--coarse-wavelength-min", type=float, default=1.2)
    parser.add_argument("--coarse-wavelength-max", type=float, default=1.8)
    parser.add_argument("--coarse-wavelength-count", type=int, default=13)

    parser.add_argument("--fine-freq-radius", type=float, default=0.08)
    parser.add_argument("--fine-wavelength-radius", type=float, default=0.14)
    parser.add_argument("--fine-freq-count", type=int, default=17)
    parser.add_argument("--fine-wavelength-count", type=int, default=17)
    parser.add_argument("--optimize", action="store_true", help="Run continuous pattern-search refinement after fine sweep.")
    parser.add_argument("--optimize-starts", type=int, default=5, help="Number of fine-sweep candidates to refine.")
    parser.add_argument("--optimize-iters", type=int, default=8)
    parser.add_argument("--optimize-min-step", type=float, default=0.001)
    parser.add_argument("--optimize-freq-step", type=float, default=0.02)
    parser.add_argument("--optimize-wavelength-step", type=float, default=0.04)
    parser.add_argument("--optimize-ajoint-step", type=float, default=1.0)
    return parser.parse_args()


def score_row(
    row: dict[str, float],
    fy_weight: float,
    dominance_weight: float,
    max_fy_over_fx: float,
    min_fx: float,
    low_fx_weight: float,
) -> float:
    lateral = row["abs_mean_fy"]
    lateral_limit = max_fy_over_fx * max(0.0, row["mean_fx"])
    dominance_error = max(0.0, lateral - lateral_limit)
    low_fx_error = max(0.0, min_fx - row["mean_fx"])
    return (
        row["mean_fx"]
        - fy_weight * lateral
        - dominance_weight * dominance_error
        - low_fx_weight * low_fx_error
    )


def write_rows(path: Path, rows: list[dict[str, float]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_grid(
    *,
    model: mujoco.MjModel,
    label: str,
    freqs: np.ndarray,
    wavelengths: np.ndarray,
    ajoints: np.ndarray,
    seconds: float,
    warmup_seconds: float,
    fy_weight: float,
    dominance_weight: float,
    max_fy_over_fx: float,
    min_fx: float,
    low_fx_weight: float,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    total = len(ajoints) * len(freqs) * len(wavelengths)
    done = 0

    print(f"\n{label}: {len(ajoints)} x {len(freqs)} x {len(wavelengths)} = {total} cases")
    for ajoint in ajoints:
        for freq in freqs:
            for wavelength in wavelengths:
                done += 1
                metrics = measure_case(
                    model,
                    float(freq),
                    float(wavelength),
                    float(ajoint),
                    seconds,
                    warmup_seconds,
                )
                mean_abs_fy = metrics["mean_abs_fy"]
                row = {
                    "frequency": float(freq),
                    "wavelength": float(wavelength),
                    "ajoint": float(ajoint),
                    "score": float(
                        score_row(
                            metrics,
                            fy_weight,
                            dominance_weight,
                            max_fy_over_fx,
                            min_fx,
                            low_fx_weight,
                        )
                    ),
                    **ratio_fields(metrics),
                    **metrics,
                }
                rows.append(row)
                print(
                    f"[{done:3d}/{total}] A={ajoint:.4f} f={freq:.4f} wl={wavelength:.4f} "
                    f"score={row['score']:.4f} Fx={metrics['mean_fx']:.4f} "
                    f"|mean Fy|={metrics['abs_mean_fy']:.4f} "
                    f"mean|Fy|={mean_abs_fy:.4f} Fx/|mean Fy|={row['fx_over_abs_fy']:.3f}",
                    flush=True,
                )

    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows


def print_top(rows: list[dict[str, float]], top: int):
    for i, row in enumerate(rows[:top], start=1):
        print(
            f"{i:2d}. score={row['score']:.4f} "
            f"Fx={row['mean_fx']:.4f} |mean Fy|={row['abs_mean_fy']:.4f} "
            f"mean|Fy|={row['mean_abs_fy']:.4f} "
            f"Fx/|mean Fy|={row['fx_over_abs_fy']:.3f} "
            f"A={row['ajoint']:.4f} f={row['frequency']:.4f} wl={row['wavelength']:.4f}"
        )


def ratio_fields(metrics: dict[str, float]) -> dict[str, float]:
    fx = metrics["mean_fx"]
    abs_fy = metrics["abs_mean_fy"]
    return {
        "fx_over_abs_fy": float(fx / abs_fy) if abs_fy > 1e-9 else float("inf"),
        "abs_fy_over_fx": float(abs_fy / fx) if fx > 1e-9 else float("inf"),
    }


def clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def continuous_refine(
    *,
    model: mujoco.MjModel,
    starts: list[dict[str, float]],
    args: argparse.Namespace,
    freq_bounds: tuple[float, float],
    wavelength_bounds: tuple[float, float],
    ajoint_bounds: tuple[float, float],
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    directions = [
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    ]

    for start_index, start in enumerate(starts, start=1):
        freq = float(start["frequency"])
        wavelength = float(start["wavelength"])
        ajoint = float(start["ajoint"])
        steps = np.array(
            [
                args.optimize_freq_step,
                args.optimize_wavelength_step,
                args.optimize_ajoint_step,
            ],
            dtype=np.float64,
        )

        metrics = measure_case(model, freq, wavelength, ajoint, args.seconds, args.warmup_seconds)
        best = {
            "start": start_index,
            "iteration": 0,
            "frequency": freq,
            "wavelength": wavelength,
            "ajoint": ajoint,
            "score": float(
                score_row(
                    metrics,
                    args.fy_weight,
                    args.dominance_weight,
                    args.max_fy_over_fx,
                    args.min_fx,
                    args.low_fx_weight,
                )
            ),
            **ratio_fields(metrics),
            **metrics,
        }
        rows.append(best)

        print(
            f"\ncontinuous start {start_index}: "
            f"A={ajoint:.4f} f={freq:.4f} wl={wavelength:.4f} score={best['score']:.4f}"
        )

        for iteration in range(1, args.optimize_iters + 1):
            improved = False
            candidate_best = best
            for df, dw, da in directions:
                cand_freq = clamp(freq + df * steps[0], *freq_bounds)
                cand_wavelength = clamp(wavelength + dw * steps[1], *wavelength_bounds)
                cand_ajoint = clamp(ajoint + da * steps[2], *ajoint_bounds)
                if (
                    cand_freq == freq
                    and cand_wavelength == wavelength
                    and cand_ajoint == ajoint
                ):
                    continue

                metrics = measure_case(
                    model,
                    cand_freq,
                    cand_wavelength,
                    cand_ajoint,
                    args.seconds,
                    args.warmup_seconds,
                )
                row = {
                    "start": start_index,
                    "iteration": iteration,
                    "frequency": cand_freq,
                    "wavelength": cand_wavelength,
                    "ajoint": cand_ajoint,
                    "score": float(
                        score_row(
                            metrics,
                            args.fy_weight,
                            args.dominance_weight,
                            args.max_fy_over_fx,
                            args.min_fx,
                            args.low_fx_weight,
                        )
                    ),
                    **ratio_fields(metrics),
                    **metrics,
                }
                rows.append(row)
                if row["score"] > candidate_best["score"]:
                    candidate_best = row
                    improved = True

            if improved:
                best = candidate_best
                freq = float(best["frequency"])
                wavelength = float(best["wavelength"])
                ajoint = float(best["ajoint"])
                print(
                    f"  iter {iteration}: improve score={best['score']:.4f} "
                    f"Fx={best['mean_fx']:.4f} |mean Fy|={best['abs_mean_fy']:.4f} "
                    f"A={ajoint:.4f} f={freq:.4f} wl={wavelength:.4f}"
                )
            else:
                steps *= 0.5
                print(
                    f"  iter {iteration}: shrink steps "
                    f"df={steps[0]:.5f} dw={steps[1]:.5f} dA={steps[2]:.5f}"
                )
                if np.max(steps) < args.optimize_min_step:
                    break

    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(args.xml)
    model.opt.gravity[:] = (0, 0, 0)

    if args.ajoint_min is None or args.ajoint_max is None:
        coarse_ajoints = np.array([args.ajoint], dtype=np.float64)
    else:
        coarse_ajoints = np.linspace(args.ajoint_min, args.ajoint_max, args.ajoint_count)

    coarse_freqs = np.linspace(args.coarse_freq_min, args.coarse_freq_max, args.coarse_freq_count)
    coarse_wavelengths = np.linspace(
        args.coarse_wavelength_min,
        args.coarse_wavelength_max,
        args.coarse_wavelength_count,
    )
    coarse_rows = run_grid(
        model=model,
        label="coarse sweep",
        freqs=coarse_freqs,
        wavelengths=coarse_wavelengths,
        ajoints=coarse_ajoints,
        seconds=args.seconds,
        warmup_seconds=args.warmup_seconds,
        fy_weight=args.fy_weight,
        dominance_weight=args.dominance_weight,
        max_fy_over_fx=args.max_fy_over_fx,
        min_fx=args.min_fx,
        low_fx_weight=args.low_fx_weight,
    )
    coarse_csv = args.output_dir / "coarse.csv"
    write_rows(coarse_csv, coarse_rows)
    print(f"\nwrote {coarse_csv}")
    print("coarse top:")
    print_top(coarse_rows, args.top)

    coarse_best = coarse_rows[0]
    if args.ajoint_min is None or args.ajoint_max is None:
        fine_ajoints = np.array([coarse_best["ajoint"]], dtype=np.float64)
    else:
        coarse_ajoint_step = (args.ajoint_max - args.ajoint_min) / max(1, args.ajoint_count - 1)
        fine_ajoint_min = max(args.ajoint_min, coarse_best["ajoint"] - coarse_ajoint_step)
        fine_ajoint_max = min(args.ajoint_max, coarse_best["ajoint"] + coarse_ajoint_step)
        fine_ajoints = np.linspace(fine_ajoint_min, fine_ajoint_max, args.fine_freq_count)

    fine_freq_min = max(args.coarse_freq_min, coarse_best["frequency"] - args.fine_freq_radius)
    fine_freq_max = min(args.coarse_freq_max, coarse_best["frequency"] + args.fine_freq_radius)
    fine_wavelength_min = max(
        args.coarse_wavelength_min,
        coarse_best["wavelength"] - args.fine_wavelength_radius,
    )
    fine_wavelength_max = min(
        args.coarse_wavelength_max,
        coarse_best["wavelength"] + args.fine_wavelength_radius,
    )

    fine_freqs = np.linspace(fine_freq_min, fine_freq_max, args.fine_freq_count)
    fine_wavelengths = np.linspace(
        fine_wavelength_min,
        fine_wavelength_max,
        args.fine_wavelength_count,
    )
    fine_rows = run_grid(
        model=model,
        label="fine sweep",
        freqs=fine_freqs,
        wavelengths=fine_wavelengths,
        ajoints=fine_ajoints,
        seconds=args.seconds,
        warmup_seconds=args.warmup_seconds,
        fy_weight=args.fy_weight,
        dominance_weight=args.dominance_weight,
        max_fy_over_fx=args.max_fy_over_fx,
        min_fx=args.min_fx,
        low_fx_weight=args.low_fx_weight,
    )
    fine_csv = args.output_dir / "fine.csv"
    write_rows(fine_csv, fine_rows)
    print(f"\nwrote {fine_csv}")
    print("fine top:")
    print_top(fine_rows, args.top)

    best_source = "fine"
    best_rows = fine_rows
    continuous_csv = None
    if args.optimize:
        if args.ajoint_min is None or args.ajoint_max is None:
            ajoint_bounds = (args.ajoint, args.ajoint)
        else:
            ajoint_bounds = (args.ajoint_min, args.ajoint_max)
        continuous_rows = continuous_refine(
            model=model,
            starts=fine_rows[: args.optimize_starts],
            args=args,
            freq_bounds=(args.coarse_freq_min, args.coarse_freq_max),
            wavelength_bounds=(args.coarse_wavelength_min, args.coarse_wavelength_max),
            ajoint_bounds=ajoint_bounds,
        )
        continuous_csv = args.output_dir / "continuous.csv"
        write_rows(continuous_csv, continuous_rows)
        print(f"\nwrote {continuous_csv}")
        print("continuous top:")
        print_top(continuous_rows, args.top)
        best_source = "continuous"
        best_rows = continuous_rows

    best = dict(best_rows[0])
    verify_metrics = measure_case(
        model,
        best["frequency"],
        best["wavelength"],
        best["ajoint"],
        args.verify_seconds,
        args.warmup_seconds,
    )
    best["verify_score"] = float(
        score_row(
            verify_metrics,
            args.fy_weight,
            args.dominance_weight,
            args.max_fy_over_fx,
            args.min_fx,
            args.low_fx_weight,
        )
    )
    best["verify_fx_over_abs_fy"] = (
        float(verify_metrics["mean_fx"] / verify_metrics["abs_mean_fy"])
        if verify_metrics["abs_mean_fy"] > 1e-9
        else float("inf")
    )
    best["verify_abs_fy_over_fx"] = (
        float(verify_metrics["abs_mean_fy"] / verify_metrics["mean_fx"])
        if verify_metrics["mean_fx"] > 1e-9
        else float("inf")
    )
    for key, value in verify_metrics.items():
        best[f"verify_{key}"] = float(value)

    result = {
        "xml": args.xml,
        "fy_weight": args.fy_weight,
        "dominance_weight": args.dominance_weight,
        "max_fy_over_fx": args.max_fy_over_fx,
        "min_fx": args.min_fx,
        "low_fx_weight": args.low_fx_weight,
        "warmup_seconds": args.warmup_seconds,
        "sweep_seconds": args.seconds,
        "verify_seconds": args.verify_seconds,
        "best_source": best_source,
        "best": best,
        "files": {
            "coarse_csv": str(coarse_csv),
            "fine_csv": str(fine_csv),
            "continuous_csv": str(continuous_csv) if continuous_csv is not None else None,
        },
        "commands": {
            "view": (
                "python view_tethered_force.py "
                f"--ajoint {best['ajoint']:.6g} "
                f"--freq {best['frequency']:.6g} "
                f"--wavelength {best['wavelength']:.6g}"
            ),
            "measure": (
                "python measure_tethered_force.py "
                f"--ajoint {best['ajoint']:.6g} "
                f"--freq {best['frequency']:.6g} "
                f"--wavelength {best['wavelength']:.6g}"
            ),
        },
    }

    best_json = args.output_dir / "best_hopf_params.json"
    with best_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\nwrote {best_json}")
    print("verified best:")
    print(
        f"  score={best['verify_score']:.4f} "
        f"Fx={best['verify_mean_fx']:.4f} "
        f"|mean Fy|={best['verify_abs_mean_fy']:.4f} "
        f"mean|Fy|={best['verify_mean_abs_fy']:.4f} "
        f"Fx/|mean Fy|={best['verify_fx_over_abs_fy']:.3f} "
        f"A={best['ajoint']:.4f} f={best['frequency']:.4f} wl={best['wavelength']:.4f}"
    )
    print("next commands:")
    print(f"  {result['commands']['view']}")
    print(f"  {result['commands']['measure']}")


if __name__ == "__main__":
    main()
