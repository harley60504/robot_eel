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
- an internal tracking rectangle at `x=±1.10`, `y=±0.35` inside the same `3 m x 1.5 m` tank

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

Run rectangle-course measurement:

```powershell
python measure_rectangle_course.py
```

Train free-swim PPO:

```powershell
python train_free_swim_rl.py
```
