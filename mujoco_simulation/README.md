# Robot Eel MuJoCo Simulation

This folder contains the MuJoCo simulation, fixed gait viewers, rectangle-course controller, measurement scripts, and PPO/RL training entrypoints for the robot eel.

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

Rebuild fitted trajectory curves and real-vs-MuJoCo comparison panels:

```powershell
python make_tracked_center_cleaned_physical.py
python plot_fitted_gait_curves.py
python track_video_start_to_wall.py
python make_real_sim_comparison_panels.py
```

The `R` shown on those comparison panels is the fitted trajectory-circle radius, not the body yaw-rate radius.
`make_tracked_center_cleaned_physical.py` regenerates `outputs/video_analysis/<video_name>/tracked_center_summary_cleaned_physical.json` from the real videos. By default it processes both `turn_left` and `spin_left`.

Run rectangle-course measurement:

```powershell
python measure_rectangle_course.py
```

Train free-swim PPO:

```powershell
python train_free_swim_rl.py
```
