# Tail Amplitude Output Index

This folder is a curated copy of tail-amplitude related outputs. Original output
folders are left untouched.

## fixed_sweep

Fixed gait tail-amplitude multiplier sweep.

Source script:

```text
C:\robot_eel\mujoco_simulation\run_tail_amp_scale_sweep.py
```

Original source folder:

```text
C:\robot_eel\mujoco_simulation\outputs\tail_amp_scale_sweep
```

Contents:

- `gaits/`: generated fixed-gait JSON files for `tail2` and `tail3`, multipliers
  `1.0` to `1.4`, left and right turns.
- `trajectories/`: MuJoCo trajectory CSV files used to redraw fitted trajectory
  figures.
- `summaries/`: sweep metric summaries, including yaw rate and fitted radius
  summary tables.
- `figures/`: generated sweep plots.

Use this group for the thesis section that discusses fixed gait tail-amplitude
effect before PPO training.

## ppo_formal

Formal PPO comparison between `bias-only` and `tail3-amp`.

Source script:

```text
C:\robot_eel\mujoco_simulation\run_tail_amp_rl_12.py
```

Main run settings:

- target yaw rate: `0.5 rad/s`
- timesteps: `200k`
- eval frequency: `5k`
- eval episodes: `5`
- runs: left/right x bias-only/tail3-amp x 3 runs

Contents:

- `gait_json/avg1s_eval5k/`: current formal exported gait JSON files used for
  the thesis discussion.
- `gait_json/old_10k_reward/`: older exported JSON files from the earlier run;
  keep only for comparison/history.
- `model_zips/`: trained PPO model zip files.
- `summaries/`: summary CSV/status files and aggregate CSV outputs.
- `logs/`: training and runner logs.
- `policy_curve/`: policy rollout trajectory/fitted-curve outputs.
- `fixed_gait_curve/`: exported fixed-gait rerun trajectory/fitted-curve outputs.
- `eval_best_policy_curves/`: best-evaluation policy trajectories from training.
- `figures/`: aggregate formal PPO figures.

Use `gait_json/avg1s_eval5k/` and the aggregate figures for the current thesis.
The `old_10k_reward/` folder is not the main result.

## thesis_figures

Final thesis-ready figures copied from:

```text
C:\robot_eel\thesis_figures\current\chapter4_results
```

These are the figures currently used by the LaTeX thesis or prepared for direct
inclusion.

## Quick Paths

Current thesis tail-amplitude gait JSON:

```text
C:\robot_eel\mujoco_simulation\outputs\tail_amp_organized\ppo_formal\gait_json\avg1s_eval5k
```

Fixed gait multiplier JSON:

```text
C:\robot_eel\mujoco_simulation\outputs\tail_amp_organized\fixed_sweep\gaits
```

Thesis-ready tail-amplitude figures:

```text
C:\robot_eel\mujoco_simulation\outputs\tail_amp_organized\thesis_figures
```
