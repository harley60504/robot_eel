# Robot Eel Python Backend

This folder is the Python sidecar used by the Flutter desktop app.

Flutter starts this backend automatically on desktop when the Python page
`Start` button is pressed and `http://127.0.0.1:8765/` is not already running.

ESP32 camera-board ports:

- HTTP `80`: Wi-Fi API, servo CSV download, and cache clear/status.
- WebSocket `81`: camera stream.
- WebSocket `82`: Flutter/manual control commands.
- WebSocket `83`: Python backend control commands.
- WebSocket `84`: servo-status telemetry.

The backend sends control JSON to port `83`. Flutter keeps using port `82` for
manual control, and servo status is received from port `84`, so telemetry does
not compete with control commands.

Camera-board firmware files are split by interface:

- `CameraStreamWs.*`: camera WebSocket stream.
- `ControlWsServer.*`: Flutter/Python control WebSockets.
- `ServoStatusWs.*`: servo telemetry WebSocket and CSV cache.
- `HttpApi.*`: HTTP routes for Wi-Fi and servo-log download.

UART packet headers are mirrored on both ESP32 boards:

- `ControlParamsPacket.h`: CPG/control-mode parameters, header `0xAA`.
- `ServoTargetPacket.h`: direct servo target angles, header `0xAB`.
- `ServoCenterPacket.h`: servo center calibration, optional NVS save, header `0xAC`.
- `ServoStatusPacket.h`: control-board telemetry back to the camera board, header `0xBB`.
- `UartPacketChecksum.h`: shared XOR checksum helper for UART packets.

## Install

```bat
cd ..\..
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

For D435i, the Intel RealSense Python package is included in the repository root requirements:

```bat
cd ..\..
python -m pip install -r requirements.txt
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
