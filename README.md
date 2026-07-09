# Robot Eel

This repository contains the robot eel desktop release, ESP32 firmware, MuJoCo simulation and PPO/RL training tools, mobile Flutter app sources, real-video analysis scripts, and 3D-print assets.

## Folder Layout

- `Release/` - Packaged Windows desktop app plus the desktop Python backend.
- `robot_eel/` - ESP32 camera-board and control-board firmware.
- `mujoco_simulation/` - MuJoCo model, current RL train/view GUI, gait JSON files, plotting scripts, and PPO/RL training code.
- `flutter_esp_control_mobile/` - Flutter mobile app project and its own mobile-side Python backend.
- `real_movie_analysis/` - Real-video tracking and trajectory analysis scripts.
- `real_movie/` - Real experiment video/data files.
- `3D_print/` - STL assets for the eel body.

## Python Setup

Install the shared Python dependencies from the repository root:

```powershell
python -m pip install -r requirements.txt
```

The root `requirements.txt` is shared by the packaged desktop backend and the MuJoCo simulation tools. It currently pins the tested desktop/backend, MuJoCo, RL, plotting, OpenCV, and RealSense packages.

Tested MuJoCo/RL environment:

```text
Python 3.13.7
MuJoCo 3.4.0
Gymnasium 1.2.2
Stable-Baselines3 2.7.1
```

## Desktop App

Run the packaged Windows app from the repository root:

```powershell
cd Release
.\flutter_esp_control.exe
```

The app uses the Python backend in `Release/python_backend`. If the backend does not start automatically, start it manually from the repository root:

```powershell
python -m pip install -r requirements.txt
cd Release\python_backend
.\run_backend.bat
```

The backend serves the local API at:

```text
http://127.0.0.1:8765/
```

Backend details, camera modes, and GoPro/RealSense notes are in:

```text
Release/python_backend/README.md
```

## MuJoCo Simulation

Run these commands from the repository root. Install dependencies first if needed:

```powershell
python -m pip install -r requirements.txt
cd mujoco_simulation
```

Open the current training/export/plot/view GUI:

```powershell
python eel_pipeline_gui.py
```

View a saved fixed gait directly in MuJoCo:

```powershell
python view_gait.py gaits/straight.json
```

Plot fixed-gait trajectories:

```powershell
python plot_fixed_gait_trajectories.py
```

Train PPO policies:

```powershell
python train_free_swim_rl.py
python train_turning_rl.py --turn-direction left --target-yaw-rate 0.45 --output outputs/zips/ppo_turn_left_shape_bias
python train_turning_rl.py --turn-direction right --target-yaw-rate 0.45 --output outputs/zips/ppo_turn_right_shape_bias
```

Run the free-swim paper batch pipeline:

```powershell
python run_free_swim_paper_10.py
```

For detailed RL training, export, parameter, and GUI-testing commands, see:

```text
mujoco_simulation/RL_GAIT_TRAINING_GUIDE.md
```

Current default fixed gaits are in `mujoco_simulation/gaits/`. RL and batch outputs are written under `mujoco_simulation/outputs/`.

Older rectangle-course, tethered-swim, and legacy exporter/viewer scripts are kept under:

```text
mujoco_simulation/legacy_unused/
```

## Firmware

The ESP32 firmware files are under `robot_eel/`:

- `robot_eel/camera/` - ESP32 camera board, Wi-Fi, HTTP/WebSocket API, camera stream, CSV download, UART bridge, and app-facing control endpoints.
- `robot_eel/control/` - ESP32 control board, CPG gait generation, servo control, UART receive/transmit packets, and servo telemetry.
- `robot_eel/cpg_standalone/` - Standalone CPG-related firmware experiments.

The camera board is the app-facing board. It relays control commands to the control board over UART and receives servo telemetry back from the control board.

Shared packet/header files include:

```text
ControlParamsPacket.h
ServoTargetPacket.h
ServoCenterPacket.h
ServoStatusPacket.h
UartPacketChecksum.h
```

## Other Tools

- Mobile app work is in `flutter_esp_control_mobile/`; see that folder's README for mobile-specific setup.
- Real-video analysis scripts and notes are in `real_movie_analysis/`.
- 3D-print STL files are in `3D_print/eel_stl/`.

## Notes

- The current MuJoCo model entry point is `mujoco_simulation/eel.xml`.
- `eel.xml` includes `mujoco_simulation/environment_3x1_5.xml`; the file name is legacy, but the current XML defines a shared `20 m x 20 m` visual tank with `x/y = -10 ~ +10 m`.
- The default simulation start position is `x=0.0, y=0.0`.
- The repository includes recorded videos under `Release/python_backend/recordings/` and `robot_eel/recordings/`; these files may be large.
