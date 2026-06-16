# Robot Eel Python Backend

This folder is the Python sidecar used by the Flutter desktop app.

Flutter starts this backend automatically on desktop when the Python page
`Start` button is pressed and `http://127.0.0.1:8765/` is not already running.

## Install

```bat
cd python_backend
python -m pip install -r requirements.txt
```

## Run Manually

```bat
run_backend.bat
```

## GoPro Wi-Fi Camera

This release backend defaults to GoPro Wi-Fi preview mode.

- GoPro API: `http://10.5.5.9:8080`
- Preview stream: `udp://0.0.0.0:8554`
- Preview/recording output: `recordings\clean_v_YYYYMMDD_HHMMSS.mp4`

Before starting preview or recording, connect Windows to the GoPro Wi-Fi
network. Then run:

```bat
run_backend.bat
```

Check the GoPro connection:

```text
http://127.0.0.1:8765/gopro/state
```

When `/preview/start` or `/recording/start` is called, the backend first sends:

```text
/gopro/camera/stream/start?port=8554
```

and then records the UDP preview stream with OpenCV/FFmpeg.

The standalone preview helper also defaults to GoPro:

```bat
preview_camera.bat
```

## Camera Modes

The backend supports three camera modes for preview and recording:

- `rtsp`: original Wi-Fi RTSP camera.
- `opengopro_preview`: GoPro Wi-Fi OpenGoPro preview stream.
- `realsense_d435i_color`: Intel RealSense D435i color stream only.
- `realsense_d435i_depth`: Intel RealSense D435i aligned depth colormap only.
- `realsense_d435i_color_depth`: Intel RealSense D435i color + aligned depth stream.

Switch mode while preview and recording are stopped:

```text
POST http://127.0.0.1:8765/settings/camera_mode
{"camera_mode":"rtsp"}

POST http://127.0.0.1:8765/settings/camera_mode
{"camera_mode":"opengopro_preview"}

POST http://127.0.0.1:8765/settings/camera_mode
{"camera_mode":"realsense_d435i_color"}

POST http://127.0.0.1:8765/settings/camera_mode
{"camera_mode":"realsense_d435i_depth"}

POST http://127.0.0.1:8765/settings/camera_mode
{"camera_mode":"realsense_d435i_color_depth"}
```

For D435i, install the Intel RealSense Python package:

```bat
python -m pip install pyrealsense2 numpy
```

The `camera_mode_d435i_color.bat`, `camera_mode_d435i_depth.bat`, and
`camera_mode_d435i_color_depth.bat` files switch between the three D435i
outputs. The combined mode records a side-by-side frame: color on the left,
aligned depth colormap on the right.

The D435i stream size is `848 x 480 @ 30 FPS`. Color-only and depth-only output
are `848 x 480`; combined color+depth output is `1696 x 480`. RealSense frames
are not stretched to `1920 x 1080`, so the preview keeps the camera's real
aspect ratio. The source name shown by the backend is:

```text
realsense://d435i/color_depth
```

## Build Note

When distributing the Flutter desktop build, keep this `python_backend` folder
next to the built Flutter executable, or package it later as a standalone
`robot_eel_backend.exe`.
