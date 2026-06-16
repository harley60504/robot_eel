@echo off
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='SilentlyContinue'; Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8765/recording/stop' | Out-Null; Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8765/preview/stop' | Out-Null; $ErrorActionPreference='Stop'; Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8765/settings/camera_mode' -ContentType 'application/json' -Body '{\"camera_mode\":\"realsense_d435i_color\"}' | ConvertTo-Json"
pause
