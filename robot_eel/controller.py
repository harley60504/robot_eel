import time
import json
import threading
from dataclasses import dataclass

from websocket import create_connection
from fastapi import FastAPI
from pydantic import BaseModel

from angle_generator import generate_angles, generate_cpg_params, init_generator

# =========================
# RTT / CSV
# =========================
seq_counter = 0
measure_enabled = False
csv_lines = ["seq,rtt_ms"]

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
    output_mode: str = "angle"   # "angle" = mode 3 + set_angle, "cpg" = mode 4 + set_param
    angle_mode_id: int = 3
    cpg_mode_id: int = 4

state = ControlState()
worker_thread = None
state_lock = threading.Lock()

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
            "measure_enabled": measure_enabled
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
