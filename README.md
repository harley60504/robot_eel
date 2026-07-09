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

The simulation environment was tested with:

```text
Python 3.13.7
MuJoCo 3.4.0
Gymnasium 1.2.2
Stable-Baselines3 2.7.1
```

Install the Python packages from the repository root:

```powershell
python -m pip install -r requirements.txt
```

Open the current training and viewing GUI:

```powershell
cd mujoco_simulation
python eel_pipeline_gui.py
```

View a saved fixed gait directly in MuJoCo:

```powershell
cd mujoco_simulation
python view_gait.py gaits/straight.json
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

Train turning PPO policies:

```powershell
cd mujoco_simulation
python train_turning_rl.py --turn-direction left --target-yaw-rate 0.45 --output outputs/zips/ppo_turn_left_shape_bias
python train_turning_rl.py --turn-direction right --target-yaw-rate 0.45 --output outputs/zips/ppo_turn_right_shape_bias
```

Run the free-swim paper batch pipeline:

```powershell
cd mujoco_simulation
python run_free_swim_paper_10.py
```

For detailed RL training, export, parameter, and GUI-testing commands, see:

```text
mujoco_simulation/RL_GAIT_TRAINING_GUIDE.md
```

Older rectangle-course, tethered-swim, and legacy exporter scripts are kept under:

```text
mujoco_simulation/legacy_unused/
```

The default fixed gaits are in `mujoco_simulation/gaits/`. RL and batch outputs are written under `mujoco_simulation/outputs/`.

## Firmware

The firmware files are under `robot_eel/`:

- `robot_eel/camera/` - ESP32 camera board, Wi-Fi, HTTP/WebSocket, camera stream, CSV download, and UART bridge.
- `robot_eel/control/` - ESP32 control board, CPG gait generation, servo control, and UART communication.

The camera board is the main app-facing board. It relays commands to the control board over UART.

## Notes

- MuJoCo models use the `3 m x 1.5 m` tank in `mujoco_simulation/environment_3x1_5.xml`.
- Current free-swim, turning, and fixed-gait scripts use `mujoco_simulation/eel.xml`.
- The default simulation start position is `x=0.0, y=0.0`.
- The repository includes recorded videos under `Release/python_backend/recordings/` and `robot_eel/recordings/`; some files are larger than GitHub's recommended 50 MB size but below the 100 MB hard limit.

