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

## Build Note

When distributing the Flutter desktop build, keep this `python_backend` folder
next to the built Flutter executable, or package it later as a standalone
`robot_eel_backend.exe`.
