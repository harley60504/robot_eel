# Robot Eel

This repository contains the robot eel desktop release, ESP32 firmware, and MuJoCo simulation package.

## Folder Layout

- `Release/` - Windows desktop build and its Python backend.
- `robot_eel/` - ESP32 camera-board and control-board firmware, plus Python controller utilities.
- `mujoco_simulation/` - MuJoCo models, gait GUI, measurement scripts, and PPO/RL training code.

## Quick Start: Desktop App

Run the packaged Windows app:

```powershell
cd Release
.\flutter_esp_control.exe
```

The app uses the Python backend in `Release/python_backend`. If the backend does not start automatically, run it manually:

```powershell
cd Release\python_backend
python -m pip install -r requirements.txt
.\run_backend.bat
```

The backend serves the local API at:

```text
http://127.0.0.1:8765/
```

## MuJoCo Simulation

Install the Python packages used by the simulation:

```powershell
python -m pip install mujoco numpy matplotlib
```

For PPO/RL training, also install:

```powershell
python -m pip install gymnasium stable-baselines3
```

Open the gait GUI:

```powershell
cd mujoco_simulation
python gait_gui.py
```

Useful GUI modes:

- `Rectangle Course` - follows the 3 m x 1.5 m pool rectangle.
- `Fixed Gait` - previews a selected saved gait from the short-side middle start.
- `CPG Step Test` - compares Hopf CPG and direct sine response to parameter changes.

Run the rectangle-course validation:

```powershell
cd mujoco_simulation
python measure_rectangle_course.py --seconds 60
```

Expected healthy result:

```text
out_of_bounds=False
wall_contact_steps=0
```

Plot fixed-gait trajectories:

```powershell
cd mujoco_simulation
python plot_fixed_gait_trajectories.py
```

Train free-swim PPO:

```powershell
cd mujoco_simulation
python train_free_swim_rl.py
```

Export a trained PPO policy to a fixed gait JSON:

```powershell
cd mujoco_simulation
python export_rl_gait_json.py --model outputs/ppo_free_swim_shape.zip --output gaits/rl_straight.json
```

The exporter rolls out the trained policy, converts PPO actions back to physical CPG parameters, and saves a `gaits/*.json` file with `amp_scales` and `phase_lags`. The exported file can be opened from the GUI with `Fixed Gait`. The current RL action does not include `joint_bias`, so exported PPO gaits are straight-swim presets by default.

## Firmware

The firmware files are under `robot_eel/`:

- `robot_eel/camera/` - ESP32 camera board, Wi-Fi, HTTP/WebSocket, camera stream, CSV download, and UART bridge.
- `robot_eel/control/` - ESP32 control board, CPG gait generation, servo control, and UART communication.

The camera board is the main app-facing board. It relays commands to the control board over UART.

## Notes

- MuJoCo models use the `3 m x 1.5 m` tank in `mujoco_simulation/environment_3x1_5.xml`.
- Fixed gait and CPG step tests start from the short-side middle area at `x=-1.10, y=0.00`.
- Rectangle tracking uses an internal route at `x=+/-1.10`, `y=+/-0.35`.
- The repository includes recorded videos under `Release/python_backend/recordings/` and `robot_eel/recordings/`; some files are larger than GitHub's recommended 50 MB size but below the 100 MB hard limit.
