import time
import json
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
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
from opengopro_client import OpenGoProHttpClient

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

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
gopro_preview_lock = threading.Lock()
gopro_preview_users = 0
gopro_client = None
preview_record_writer = None
preview_record_codec = None

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
    camera_mode: str = "opengopro_preview"
    gopro_base_url: str = "http://10.5.5.9:8080"
    gopro_preview_port: int = 8554
    recorder_url: str = "udp://0.0.0.0:8554"
    camera_rotate: str = "none"
    camera_width: int = 1920
    camera_height: int = 1080
    realsense_width: int = 848
    realsense_height: int = 480
    realsense_fps: int = 30
    recording: bool = False
    recording_path: str = ""
    preview_running: bool = False

state = ControlState()
worker_thread = None
state_lock = threading.Lock()
OFFSET_MODE_ID = 2
GOPRO_DEFAULT_PREVIEW_PORT = 8554

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

class GoProBaseUrlReq(BaseModel):
    gopro_base_url: str

class CameraModeReq(BaseModel):
    camera_mode: str

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

def use_gopro_preview():
    with state_lock:
        return state.camera_mode.lower() in ("opengopro_preview", "gopro_preview", "preview")

def use_realsense():
    with state_lock:
        return state.camera_mode.lower() in (
            "realsense_d435i",
            "realsense_d435i_color",
            "realsense_d435i_depth",
            "realsense_d435i_color_depth",
            "d435i",
            "realsense",
        )

def gopro_preview_source(port):
    return f"udp://0.0.0.0:{int(port)}"

def default_rtsp_source():
    return "rtsp://admin:184342@192.168.0.102:554/live/profile.0/video"

def get_gopro_client():
    global gopro_client
    with state_lock:
        base_url = state.gopro_base_url
    if gopro_client is None or gopro_client.base_url != base_url.rstrip("/"):
        gopro_client = OpenGoProHttpClient(base_url=base_url)
    return gopro_client

def acquire_camera_source(label):
    global gopro_preview_users
    with state_lock:
        port = state.gopro_preview_port

    if not use_gopro_preview():
        with state_lock:
            return state.recorder_url, False

    url = gopro_preview_source(port)
    with state_lock:
        state.recorder_url = url

    with gopro_preview_lock:
        if gopro_preview_users == 0:
            print(f"[{label}] start GoPro preview stream port={port}")
            get_gopro_client().start_preview_stream(port)
            time.sleep(0.4)
        gopro_preview_users += 1

    return url, True

def release_camera_source(label, used_gopro):
    global gopro_preview_users
    if not used_gopro:
        return

    with gopro_preview_lock:
        gopro_preview_users = max(0, gopro_preview_users - 1)
        if gopro_preview_users == 0:
            try:
                get_gopro_client().stop_preview_stream()
                print(f"[{label}] stopped GoPro preview stream")
            except Exception as e:
                print(f"[{label}] GoPro preview stop warning:", e)

class RealSenseD435iCapture:
    def __init__(self):
        if rs is None:
            raise RuntimeError("pyrealsense2 is not installed")

        with state_lock:
            width = state.realsense_width
            height = state.realsense_height
            fps = state.realsense_fps
            mode = state.camera_mode.lower()

        if mode in ("realsense_d435i_depth",):
            self.output_mode = "depth"
        elif mode in ("realsense_d435i_color",):
            self.output_mode = "color"
        else:
            self.output_mode = "color_depth"

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.align = rs.align(rs.stream.color)
        self.colorizer = rs.colorizer()
        self.colorizer.set_option(rs.option.color_scheme, 0)
        self.colorizer.set_option(rs.option.histogram_equalization_enabled, 1)
        self.colorizer.set_option(rs.option.min_distance, 0.15)
        self.colorizer.set_option(rs.option.max_distance, 4.0)
        self.profile = self.pipeline.start(config)
        self.opened = True
        self.fps = float(fps)

    def isOpened(self):
        return self.opened

    def read(self):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=700)
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                return False, None
            color_image = np.asanyarray(color_frame.get_data())
            depth_color_frame = self.colorizer.colorize(depth_frame)
            depth_image = np.asanyarray(depth_color_frame.get_data())
            if self.output_mode == "color":
                return True, color_image
            if self.output_mode == "depth":
                return True, depth_image
            return True, np.hstack((color_image, depth_image))
        except Exception as e:
            print("[REALSENSE] frame read failed:", e)
            return False, None

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        return 0.0

    def set(self, prop, value):
        return False

    def release(self):
        if self.opened:
            self.pipeline.stop()
            self.opened = False

def process_camera_frame(frame):
    if use_realsense():
        return frame

    with state_lock:
        target_w = state.camera_width
        target_h = state.camera_height
        rotate = state.camera_rotate.lower()

    if target_w > 0 and target_h > 0:
        frame = cv2.resize(
            frame,
            (target_w, target_h),
            interpolation=cv2.INTER_LANCZOS4,
        )

    if rotate in ("cw", "clockwise", "90cw"):
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotate in ("ccw", "counterclockwise", "90ccw"):
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotate in ("180", "flip"):
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame

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

def new_recording_path():
    recordings_dir = Path(__file__).resolve().parent / "recordings"
    recordings_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return recordings_dir / f"clean_v_{timestamp}.mp4"

def open_camera_capture(url):
    if use_realsense():
        return RealSenseD435iCapture()

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    try:
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 700)
    except Exception:
        pass
    return cap

def read_camera_frame(cap, timeout_sec=0.8):
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        ok, frame = cap.read()
        if ok and frame is not None:
            return True, frame
        time.sleep(0.02)
    return False, None

def recording_loop():
    try:
        url, used_gopro = acquire_camera_source("REC")
    except Exception as e:
        print("[REC] GoPro preview start failed:", e)
        with state_lock:
            state.recording = False
            state.recording_path = ""
        return

    cap = open_camera_capture(url)
    if not cap.isOpened():
        print("[REC] camera open failed:", url)
        release_camera_source("REC", used_gopro)
        with state_lock:
            state.recording = False
            state.recording_path = ""
        return

    ok, frame = read_camera_frame(cap)
    if not ok or frame is None:
        print("[REC] first frame read failed")
        cap.release()
        release_camera_source("REC", used_gopro)
        with state_lock:
            state.recording = False
            state.recording_path = ""
        return

    first_clean_frame = process_camera_frame(frame)
    record_h, record_w = first_clean_frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1 or fps > 120:
        fps = 20.0

    path = new_recording_path()
    writer, codec = open_video_writer(path, record_w, record_h, fps)
    if writer is None:
        path = path.with_suffix(".avi")
        writer, codec = open_video_writer(path, record_w, record_h, fps)

    if writer is None:
        print("[REC] VideoWriter open failed:", path.parent)
        cap.release()
        release_camera_source("REC", used_gopro)
        with state_lock:
            state.recording = False
            state.recording_path = ""
        return

    with state_lock:
        state.recording = True
        state.recording_path = str(path)

    print(f"[REC] start ({codec}): {path}")

    try:
        current = first_clean_frame
        while not recording_stop_event.is_set():
            writer.write(current)

            ok, raw_frame = read_camera_frame(cap)
            if not ok or raw_frame is None:
                print("[REC] frame read failed")
                break
            current = process_camera_frame(raw_frame)
    finally:
        writer.release()
        cap.release()
        release_camera_source("REC", used_gopro)
        with state_lock:
            state.recording = False
        print("[REC] saved:", path)

def make_preview_frame(frame):
    return process_camera_frame(frame)

def preview_loop():
    global preview_jpeg, preview_fps, preview_record_writer, preview_record_codec

    try:
        url, used_gopro = acquire_camera_source("PREVIEW")
    except Exception as e:
        print("[PREVIEW] GoPro preview start failed:", e)
        with state_lock:
            state.preview_running = False
        return

    with state_lock:
        state.preview_running = True

    cap = open_camera_capture(url)
    if not cap.isOpened():
        print("[PREVIEW] camera open failed:", url)
        release_camera_source("PREVIEW", used_gopro)
        with state_lock:
            state.preview_running = False
        return

    last = time.time()
    frames = 0
    record_fps = cap.get(cv2.CAP_PROP_FPS)
    if record_fps <= 1 or record_fps > 120:
        record_fps = 20.0

    try:
        while not preview_stop_event.is_set():
            ok, frame = read_camera_frame(cap)
            if not ok or frame is None:
                print("[PREVIEW] frame read failed; reopening camera source")
                cap.release()
                time.sleep(0.2)
                cap = open_camera_capture(url)
                if not cap.isOpened():
                    print("[PREVIEW] camera reopen failed:", url)
                    break
                continue

            frame = make_preview_frame(frame)

            with state_lock:
                should_record = state.recording
                record_path = state.recording_path

            if should_record:
                if preview_record_writer is None:
                    record_h, record_w = frame.shape[:2]
                    path = Path(record_path) if record_path else new_recording_path()
                    writer, codec = open_video_writer(path, record_w, record_h, record_fps)
                    if writer is None:
                        path = path.with_suffix(".avi")
                        writer, codec = open_video_writer(path, record_w, record_h, record_fps)
                    if writer is None:
                        print("[PREVIEW REC] VideoWriter open failed:", path.parent)
                        with state_lock:
                            state.recording = False
                            state.recording_path = ""
                    else:
                        preview_record_writer = writer
                        preview_record_codec = codec
                        with state_lock:
                            state.recording_path = str(path)
                        print(f"[PREVIEW REC] start ({codec}): {path}")

                if preview_record_writer is not None:
                    preview_record_writer.write(frame)
            elif preview_record_writer is not None:
                preview_record_writer.release()
                print("[PREVIEW REC] saved:", record_path)
                preview_record_writer = None
                preview_record_codec = None

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
        if preview_record_writer is not None:
            preview_record_writer.release()
            preview_record_writer = None
            preview_record_codec = None
        cap.release()
        release_camera_source("PREVIEW", used_gopro)
        with state_lock:
            state.preview_running = False
            state.recording = False
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
            "camera_mode": state.camera_mode,
            "gopro_base_url": state.gopro_base_url,
            "gopro_preview_port": state.gopro_preview_port,
            "recorder_url": state.recorder_url,
            "camera_rotate": state.camera_rotate,
            "camera_width": state.camera_width,
            "camera_height": state.camera_height,
            "realsense_width": state.realsense_width,
            "realsense_height": state.realsense_height,
            "realsense_fps": state.realsense_fps,
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
        if state.recording or state.preview_running:
            return {"ok": False, "error": "cannot change recorder_url while camera is active"}
        if state.camera_mode.lower() in ("opengopro_preview", "gopro_preview", "preview"):
            url = gopro_preview_source(state.gopro_preview_port)
        elif state.camera_mode.lower() in (
            "realsense_d435i",
            "realsense_d435i_color",
            "realsense_d435i_depth",
            "realsense_d435i_color_depth",
            "d435i",
            "realsense",
        ):
            output = state.camera_mode.lower().replace("realsense_d435i_", "")
            if output in ("realsense_d435i", "d435i", "realsense"):
                output = "color_depth"
            url = f"realsense://d435i/{output}"
        state.recorder_url = url

    return {"ok": True, "recorder_url": url}

@app.post("/settings/camera_mode")
def set_camera_mode(req: CameraModeReq):
    mode = req.camera_mode.strip().lower()
    aliases = {
        "wifi": "rtsp",
        "wifi_rtsp": "rtsp",
        "rtsp": "rtsp",
        "gopro": "opengopro_preview",
        "gopro_preview": "opengopro_preview",
        "opengopro": "opengopro_preview",
        "opengopro_preview": "opengopro_preview",
        "d435i": "realsense_d435i_color_depth",
        "realsense": "realsense_d435i_color_depth",
        "realsense_d435i": "realsense_d435i_color_depth",
        "d435i_color": "realsense_d435i_color",
        "realsense_color": "realsense_d435i_color",
        "realsense_d435i_color": "realsense_d435i_color",
        "d435i_depth": "realsense_d435i_depth",
        "realsense_depth": "realsense_d435i_depth",
        "realsense_d435i_depth": "realsense_d435i_depth",
        "d435i_color_depth": "realsense_d435i_color_depth",
        "realsense_color_depth": "realsense_d435i_color_depth",
        "realsense_d435i_color_depth": "realsense_d435i_color_depth",
    }
    if mode not in aliases:
        return {
            "ok": False,
            "error": "camera_mode must be rtsp, opengopro_preview, realsense_d435i_color, realsense_d435i_depth, or realsense_d435i_color_depth",
        }

    normalized = aliases[mode]
    preview_stop_event.set()
    recording_stop_event.set()
    thread = preview_thread
    if thread is not None and thread.is_alive():
        thread.join(timeout=3)

    with state_lock:
        state.recording = False
        state.preview_running = False
        state.camera_mode = normalized
        if normalized == "rtsp":
            if (
                not state.recorder_url
                or state.recorder_url.startswith("udp://")
                or state.recorder_url.startswith("realsense://")
            ):
                state.recorder_url = default_rtsp_source()
        elif normalized == "opengopro_preview":
            state.recorder_url = gopro_preview_source(state.gopro_preview_port)
        elif normalized.startswith("realsense_d435i"):
            state.recorder_url = f"realsense://d435i/{normalized.replace('realsense_d435i_', '')}"

    return {
        "ok": True,
        "camera_mode": normalized,
        "recorder_url": state.recorder_url,
    }

@app.post("/settings/gopro_base_url")
def set_gopro_base_url(req: GoProBaseUrlReq):
    global gopro_client
    base_url = req.gopro_base_url.strip().rstrip("/")
    if not base_url:
        return {"ok": False, "error": "gopro_base_url is required"}

    with state_lock:
        if state.recording or state.preview_running:
            return {"ok": False, "error": "cannot change GoPro URL while camera is active"}
        state.gopro_base_url = base_url
    gopro_client = None

    return {"ok": True, "gopro_base_url": base_url}

@app.get("/gopro/state")
def gopro_state():
    try:
        return {"ok": True, "state": get_gopro_client().state()}
    except Exception as e:
        return {"ok": False, "error": str(e)}

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
            if state.preview_running:
                path = new_recording_path()
                state.recording = True
                state.recording_path = str(path)
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
        if state.preview_running and state.recording:
            state.recording = False
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
    with state_lock:
        state.preview_running = False
        state.recording = False
    thread = preview_thread
    if thread is not None and thread.is_alive():
        thread.join(timeout=3)
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
