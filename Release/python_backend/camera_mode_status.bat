@echo off
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-RestMethod -Uri 'http://127.0.0.1:8765/' | Select-Object camera_mode,recorder_url,preview_running,recording | Format-List"
pause
