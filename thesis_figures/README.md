# Thesis Figure Generation

這個資料夾集中保存論文目前使用的圖片、原始硬體素材，以及可重新產生部分論文圖片的 Python 腳本。

## Folder Layout

- `current/`: 目前論文正在使用或備份的圖片。
- `current/chapter3_methodology/`: 第三章系統架構、硬體、通訊與 JSON 流程圖。
- `current/chapter4_results/`: 第四章模擬、軌跡、PPO 與 fixed gait 結果圖。
- `current/assets/`: 論文共用素材，例如清華浮水印。
- `source_assets/`: 產圖腳本需要的原始圖片，目前包含 XIAO ESP32S3 Sense 與 XIAO bus servo driver 圖片。
- `scripts/`: 可重新產生論文圖片的 Python 腳本。

## Scripts

### `scripts/make_hardware_diagrams.py`

產生第三章硬體與基礎通訊圖：

- `dual_esp32_architecture.png`
- `hardware_platform.png`
- `pin_wiring_diagram.png`
- `json_message_flow.png`

此腳本會讀取：

- `source_assets/xiao_esp32s3_sense.png`
- `source_assets/xiao_bus_servo_driver.png`

### `scripts/draw_chapter3_flows.py`

產生第三章整體流程與 JSON payload 資料流圖：

- `rl_to_real_robot_flow.png`
- `json_payload_flow.png`

通訊標示原則：

- Flutter 對 Python backend 的管理命令使用 HTTP。
- Python backend 對 XIAO ESP32S3 Sense 的即時控制輸出使用 WebSocket JSON。
- XIAO ESP32S3 Sense 與 ESP32 control board 之間使用 UART packet。
- ESP32 control board 與 bus servo 控制板之間使用 TTL UART / servo bus。
- 相機影像串流為 WS:81 binary JPEG，不屬於 JSON payload。

## Setup On Another PC

建議使用 Python 3.10 以上。

```powershell
cd C:\robot_eel\thesis_figures
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install pillow
```

如果電腦沒有 `py` launcher，可改用：

```powershell
python -m venv .venv
```

## Regenerate Figures

如果 LaTeX 專案仍在目前電腦的預設位置，可以直接執行：

```powershell
cd C:\robot_eel\thesis_figures
.\.venv\Scripts\python.exe .\scripts\make_hardware_diagrams.py
.\.venv\Scripts\python.exe .\scripts\draw_chapter3_flows.py
```

腳本會同時輸出到：

- `C:\Users\Harley\Documents\Codex\2026-06-18\word-latex\work\thesis-latex\figures`
- `C:\robot_eel\thesis_figures\current\chapter3_methodology`

如果換電腦後 LaTeX 專案位置不同，先指定 `THESIS_FIGURES_DIR`：

```powershell
$env:THESIS_FIGURES_DIR = "D:\my-thesis\figures"
.\.venv\Scripts\python.exe .\scripts\make_hardware_diagrams.py
.\.venv\Scripts\python.exe .\scripts\draw_chapter3_flows.py
```

## Compile Thesis After Regeneration

產圖完成後回到 LaTeX 專案編譯：

```powershell
cd C:\Users\Harley\Documents\Codex\2026-06-18\word-latex\work\thesis-latex
xelatex -synctex=1 -interaction=nonstopmode -file-line-error -output-directory=build main.tex
biber .\build\main
xelatex -synctex=1 -interaction=nonstopmode -file-line-error -output-directory=build main.tex
xelatex -synctex=1 -interaction=nonstopmode -file-line-error -output-directory=build main.tex
```

