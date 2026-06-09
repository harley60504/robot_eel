import time
import json
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
from websocket import create_connection
from fastapi import FastAPI, Response
from pydantic import BaseModel

from angle_generator import (
    current_gait,
    generate_angles,
    generate_cpg_params,
    init_generator,
    list_gaits,
    set_gait,
)

# =========================
# RTT / CSV
# =========================
seq_counter = 0
measure_enabled = False
csv_lines = ["seq,rtt_ms"]
recording_stop_event = threading.Event()
recording_thread = None
recording_lock = threading.Lock()
preview_stop_event = threading.Event()
preview_thread = None
preview_lock = threading.Lock()
preview_jpeg = None
preview_fps = 0.0

def save_csv():
    with open("latency.csv", "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines))

# =========================
# State
# =========================
@dataclass
class ControlState:
    running: bool = False
    esp_host: str = "192.168.4.1"
    esp_ws_port: int = 82
    interval_ms: int = 50
    output_mode: str = "cpg"     # "angle" = mode 3 + set_angle, "cpg" = mode 1 + set_param
    angle_mode_id: int = 3
    cpg_mode_id: int = 1
    recorder_url: str = "rtsp://admin:184342@192.168.0.102:554/live/profile.0/video"
    recording: bool = False
    recording_path: str = ""
    preview_running: bool = False

state = ControlState()
worker_thread = None
state_lock = threading.Lock()
OFFSET_MODE_ID = 2

# =========================
# Request Models
# =========================
class HostReq(BaseModel):
    esp_host: str
    esp_ws_port: int = 82

class IntervalReq(BaseModel):
    interval_ms: int

class OutputModeReq(BaseModel):
    output_mode: str

class ModeIdReq(BaseModel):
    mode: int

class GaitReq(BaseModel):
    gait: str

class RecorderUrlReq(BaseModel):
    recorder_url: str

# =========================
# Utils
# =========================
def ws_url():
    return f"ws://{state.esp_host}:{state.esp_ws_port}"

def next_seq():
    global seq_counter
    seq = seq_counter
    seq_counter += 1
    return seq

# =========================
# Send
# =========================
def send_angle(ws, angles):
    payload = {
        "cmd": "set_angle",
        "seq": next_seq(),
        "angles": angles
    }

    ws.send(json.dumps(payload))

def send_angle_rtt(ws, angles):
    seq = next_seq()

    payload = {
        "cmd": "set_angle",
        "seq": seq,
        "angles": angles
    }

    t1 = time.perf_counter_ns()
    ws.send(json.dumps(payload))

    while True:
        msg = ws.recv()
        t2 = time.perf_counter_ns()

        try:
            data = json.loads(msg)
        except Exception:
            continue

        if data.get("type") == "angle_ack" and data.get("seq") == seq:
            rtt = (t2 - t1) / 1e6
            csv_lines.append(f"{seq},{rtt:.2f}")
            print(f"[RTT] {rtt:.2f} ms")
            return

def send_params(ws, params):
    payload = {
        "cmd": "set_param",
        "seq": next_seq(),
        "ts_ms": int(time.time() * 1000),
        **params,
    }
    ws.send(json.dumps(payload))

def send_mode(ws, mode_id):
    params = generate_cpg_params(0.0, 0.0)
    params["mode"] = mode_id
    params["paused"] = False
    send_params(ws, params)

def send_offset_once():
    with state_lock:
        url = ws_url()

    ws = create_connection(url, timeout=0.25)
    try:
        send_params(ws, {
            "mode": OFFSET_MODE_ID,
            "paused": False,
        })
    finally:
        ws.close()

# =========================
# Recording
# =========================
def open_video_writer(path, width, height, fps):
    candidates = ["mp4v", "XVID", "MJPG"]
    for codec in candidates:
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if writer.isOpened():
            return writer, codec
        writer.release()
    return None, None

def recording_loop():
    recordings_dir = Path(__file__).resolve().parent / "recordings"
    recordings_dir.mkdir(exist_ok=True)

    with state_lock:
        url = state.recorder_url

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("[REC] camera open failed:", url)
        with state_lock:
            state.recording = False
            state.recording_path = ""
        return

    ok, frame = cap.read()
    if not ok or frame is None:
        print("[REC] first frame read failed")
        cap.release()
        with state_lock:
            state.recording = False
            state.recording_path = ""
        return

    target_w = 1920
    target_h = 1080
    record_w = target_h
    record_h = target_w
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1 or fps > 120:
        fps = 20.0

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = recordings_dir / f"clean_v_{timestamp}.mp4"
    writer, codec = open_video_writer(path, record_w, record_h, fps)
    if writer is None:
        path = recordings_dir / f"clean_v_{timestamp}.avi"
        writer, codec = open_video_writer(path, record_w, record_h, fps)

    if writer is None:
        print("[REC] VideoWriter open failed:", recordings_dir)
        cap.release()
        with state_lock:
            state.recording = False
            state.recording_path = ""
        return

    with state_lock:
        state.recording = True
        state.recording_path = str(path)

    print(f"[REC] start ({codec}): {path}")

    try:
        current = frame
        while not recording_stop_event.is_set():
            temp_frame = cv2.resize(
                current,
                (target_w, target_h),
                interpolation=cv2.INTER_LANCZOS4,
            )
            clean_frame = cv2.rotate(temp_frame, cv2.ROTATE_90_CLOCKWISE)
            writer.write(clean_frame)

            ok, current = cap.read()
            if not ok or current is None:
                print("[REC] frame read failed")
                break
    finally:
        writer.release()
        cap.release()
        with state_lock:
            state.recording = False
        print("[REC] saved:", path)

def make_preview_frame(frame):
    target_w = 1920
    target_h = 1080
    frame = cv2.resize(
        frame,
        (target_w, target_h),
        interpolation=cv2.INTER_LANCZOS4,
    )
    return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

def preview_loop():
    global preview_jpeg, preview_fps

    with state_lock:
        url = state.recorder_url
        state.preview_running = True

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("[PREVIEW] camera open failed:", url)
        with state_lock:
            state.preview_running = False
        return

    last = time.time()
    frames = 0

    try:
        while not preview_stop_event.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[PREVIEW] frame read failed")
                break

            frame = make_preview_frame(frame)
            ok, encoded = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 80],
            )
            if ok:
                with preview_lock:
                    preview_jpeg = encoded.tobytes()

            frames += 1
            now = time.time()
            if now - last >= 1.0:
                preview_fps = frames / (now - last)
                frames = 0
                last = now
    finally:
        cap.release()
        with state_lock:
            state.preview_running = False
        print("[PREVIEW] stopped")

# =========================
# Control Loop
# =========================
def control_loop():
    print("[PY] control loop start")

    try:
        ws = create_connection(ws_url(), timeout=3)
    except Exception as e:
        print("[PY] connect fail:", e)
        return

    init_generator()

    with state_lock:
        output_mode = state.output_mode.lower()
        interval = state.interval_ms
        selected_mode = state.cpg_mode_id if output_mode == "cpg" else state.angle_mode_id
    active_mode = selected_mode

    try:
        send_mode(ws, selected_mode)
        print(f"[PY] set ESP mode={selected_mode} output={output_mode}")
    except Exception as e:
        print("[PY] mode switch fail:", e)
        try:
            ws.close()
        except Exception:
            pass
        return

    t0 = time.time()
    last_time = t0

    while True:
        with state_lock:
            if not state.running:
                break
            interval = state.interval_ms
            output_mode = state.output_mode.lower()
            desired_mode = state.cpg_mode_id if output_mode == "cpg" else state.angle_mode_id

        if desired_mode != active_mode:
            send_mode(ws, desired_mode)
            active_mode = desired_mode
            print(f"[PY] set ESP mode={active_mode} output={output_mode}")

        now = time.time()
        t = now - t0
        dt = now - last_time
        last_time = now

        if dt <= 0:
            dt = interval / 1000.0

        try:
            if output_mode == "cpg":
                params = generate_cpg_params(t, dt)
                params["mode"] = active_mode
                send_params(ws, params)
            else:
                angles = generate_angles(t, dt)
                if measure_enabled:
                    send_angle_rtt(ws, angles)
                else:
                    send_angle(ws, angles)
        except Exception as e:
            print("[PY] generate/send fail:", e)
            break

        time.sleep(interval / 1000.0)

    try:
        send_mode(ws, OFFSET_MODE_ID)
        print("[PY] set ESP mode=2 output=offset")
    except Exception as e:
        print("[PY] offset switch fail:", e)

    try:
        ws.close()
    except Exception:
        pass

    print("[PY] control loop stop")

# =========================
# FastAPI
# =========================
app = FastAPI()

@app.get("/")
def root():
    with state_lock:
        return {
            "running": state.running,
            "esp_host": state.esp_host,
            "esp_ws_port": state.esp_ws_port,
            "interval_ms": state.interval_ms,
            "output_mode": state.output_mode,
            "angle_mode_id": state.angle_mode_id,
            "cpg_mode_id": state.cpg_mode_id,
            "gait": current_gait().key,
            "measure_enabled": measure_enabled,
            "recorder_url": state.recorder_url,
            "recording": state.recording,
            "recording_path": state.recording_path,
            "preview_running": state.preview_running,
            "preview_fps": preview_fps,
        }

@app.post("/set_esp_host")
def set_host(req: HostReq):
    with state_lock:
        state.esp_host = req.esp_host
        state.esp_ws_port = req.esp_ws_port
    return {"ok": True}

@app.post("/set_interval")
def set_interval(req: IntervalReq):
    if req.interval_ms <= 0:
        return {"ok": False, "error": "interval_ms must be > 0"}

    with state_lock:
        state.interval_ms = req.interval_ms

    return {"ok": True, "interval_ms": req.interval_ms}

@app.post("/set_output_mode")
def set_output_mode(req: OutputModeReq):
    mode = req.output_mode.lower()
    if mode not in ("angle", "cpg"):
        return {"ok": False, "error": "output_mode must be 'angle' or 'cpg'"}

    with state_lock:
        state.output_mode = mode

    return {"ok": True, "output_mode": mode}

@app.post("/set_angle_mode")
def set_angle_mode(req: ModeIdReq):
    if req.mode < 0 or req.mode > 255:
        return {"ok": False, "error": "mode must be 0..255"}

    with state_lock:
        state.angle_mode_id = req.mode

    return {"ok": True, "angle_mode_id": req.mode}

@app.post("/set_cpg_mode")
def set_cpg_mode(req: ModeIdReq):
    if req.mode < 0 or req.mode > 255:
        return {"ok": False, "error": "mode must be 0..255"}

    with state_lock:
        state.cpg_mode_id = req.mode

    return {"ok": True, "cpg_mode_id": req.mode}

@app.get("/gaits")
def gaits():
    return {
        "current": current_gait().key,
        "gaits": list_gaits(),
    }

@app.post("/set_gait")
def set_gait_endpoint(req: GaitReq):
    try:
        set_gait(req.gait)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "gait": current_gait().key}

@app.post("/settings/recorder_url")
def set_recorder_url(req: RecorderUrlReq):
    url = req.recorder_url.strip()
    if not url:
        return {"ok": False, "error": "recorder_url is required"}

    with state_lock:
        if state.recording:
            return {"ok": False, "error": "cannot change recorder_url while recording"}
        state.recorder_url = url

    return {"ok": True, "recorder_url": url}

@app.post("/start")
def start():
    global worker_thread

    with state_lock:
        state.running = True

    if worker_thread is None or not worker_thread.is_alive():
        worker_thread = threading.Thread(target=control_loop, daemon=True)
        worker_thread.start()

    return {"ok": True}

@app.post("/stop")
def stop():
    with state_lock:
        state.running = False

    try:
        send_offset_once()
    except Exception as e:
        print("[PY] stop offset fail:", e)

    if measure_enabled:
        save_csv()

    return {"ok": True}

@app.post("/measure_on")
def measure_on():
    global measure_enabled
    measure_enabled = True
    return {"ok": True}

@app.post("/measure_off")
def measure_off():
    global measure_enabled
    measure_enabled = False
    return {"ok": True}

@app.post("/recording/start")
def recording_start():
    global recording_thread

    with recording_lock:
        with state_lock:
            if state.recording:
                return {"ok": True, "recording": True, "path": state.recording_path}

        recording_stop_event.clear()
        recording_thread = threading.Thread(target=recording_loop, daemon=True)
        recording_thread.start()

    return {"ok": True}

@app.post("/recording/stop")
def recording_stop():
    recording_stop_event.set()

    thread = recording_thread
    if thread is not None and thread.is_alive():
        thread.join(timeout=4)

    with state_lock:
        return {
            "ok": True,
            "recording": state.recording,
            "path": state.recording_path,
        }

@app.post("/preview/start")
def preview_start():
    global preview_thread

    with preview_lock:
        with state_lock:
            if state.preview_running:
                return {"ok": True, "preview_running": True}

        preview_stop_event.clear()
        preview_thread = threading.Thread(target=preview_loop, daemon=True)
        preview_thread.start()

    return {"ok": True}

@app.post("/preview/stop")
def preview_stop():
    preview_stop_event.set()
    thread = preview_thread
    if thread is not None and thread.is_alive():
        thread.join(timeout=2)
    return {"ok": True}

@app.get("/preview.jpg")
def preview_image():
    with preview_lock:
        data = preview_jpeg

    if data is None:
        return Response(status_code=204)

    return Response(
        content=data,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )
