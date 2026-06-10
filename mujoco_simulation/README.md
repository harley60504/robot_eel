# Robot Eel MuJoCo Simulation

This folder contains the MuJoCo simulation, fixed gait viewers, rectangle-course controller, measurement scripts, PPO/RL training entrypoints, and real-video comparison scripts for the robot eel.

## Environment

All model XML files use the same `3 m x 1.5 m` tank:

- `environment_3x1_5.xml`

Fixed gait and CPG step tests use:

- `eel.xml`
- the same physical coordinate system as the working MuJoCo model
- viewer scripts can be used for camera/view comparison without rotating the model physics

Rectangle course tests use:

- `eel_rectangle.xml`
- `view_rectangle_course.py`
- an internal tracking rectangle at `x=+/-1.10`, `y=+/-0.35` inside the same `3 m x 1.5 m` tank

Tethered force/RL tests use:

- `eel_tethered.xml`
- same `3 m x 1.5 m` tank

## Common Commands

Open the GUI:

```powershell
python gait_gui.py
```

Run all fixed gaits:

```powershell
python plot_fixed_gait_trajectories.py
```

## Automated real-video / MuJoCo comparison workflow

Run the full workflow from inside this folder:

```powershell
cd C:\Users\ytyla\Documents\GitHub\robot_eel\mujoco_simulation
python run_real_sim_analysis.py
```

This command runs the full chain:

1. `plot_fixed_gait_trajectories.py`
2. `make_tracked_center_cleaned_physical.py`
3. `track_video_start_to_wall.py`
4. `plot_fitted_gait_curves.py`
5. `make_real_sim_comparison_panels.py`

The main outputs are:

```text
outputs/real_sim_comparison/metric_summary_m.json
outputs/real_sim_comparison/straight_real_vs_mujoco.png
outputs/real_sim_comparison/turn_left_real_vs_mujoco.png
outputs/real_sim_comparison/spin_left_real_vs_mujoco.png
outputs/real_sim_comparison/real_vs_mujoco_all.png
```

`metric_summary_m.json` is the metric file to use for report/PPT values. It converts real-video radius from pixels to meters using:

```text
radius_m = radius_px / px_per_m
px_per_m = 875 / 1.5 = 583.333 px/m
```

The `R` shown on the comparison panels is the fitted trajectory-circle radius, not the body yaw-rate radius. MuJoCo `R` is already in meters. Real-video `R` is detected in pixels first and then converted to meters.

To rebuild only the meter summary from existing outputs:

```powershell
python run_real_sim_analysis.py --skip-run
```

To copy final comparison PNGs to your Pictures folder:

```powershell
python run_real_sim_analysis.py --copy-to-pictures C:\Users\ytyla\Pictures
```

## Manual real-video comparison commands

If you want to run each step manually:

```powershell
python make_tracked_center_cleaned_physical.py
python plot_fitted_gait_curves.py
python track_video_start_to_wall.py
python make_real_sim_comparison_panels.py
```

`make_tracked_center_cleaned_physical.py` regenerates:

```text
outputs/video_analysis/<video_name>/tracked_center_summary_cleaned_physical.json
```

from the real videos under:

```text
..\Release\python_backend\recordings
```

Run rectangle-course measurement:

```powershell
python measure_rectangle_course.py
```

Train free-swim PPO:

```powershell
python train_free_swim_rl.py
```
