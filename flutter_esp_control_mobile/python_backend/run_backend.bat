@echo off
cd /d "%~dp0"
python -m uvicorn controller:app --host 127.0.0.1 --port 8765
