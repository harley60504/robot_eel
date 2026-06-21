from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math
import os

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get(
    "THESIS_FIGURES_DIR",
    r"C:\Users\Harley\Documents\Codex\2026-06-18\word-latex\work\thesis-latex\figures",
))
ROBOT = ROOT / "current" / "chapter3_methodology"
FONT = r"C:\Windows\Fonts\kaiu.ttf"
OUT.mkdir(parents=True, exist_ok=True)
ROBOT.mkdir(parents=True, exist_ok=True)


def font(size):
    return ImageFont.truetype(FONT, size)


TITLE = font(38)
HEAD = font(28)
BODY = font(22)
SMALL = font(17)

COL = {
    "sim": "#E8F1FF",
    "pc": "#F4ECFF",
    "esp": "#EAF7EA",
    "servo": "#FFF2E2",
    "data": "#F2F2F2",
    "blue": "#2E6FCA",
    "green": "#2C9B55",
    "orange": "#E57A1F",
    "purple": "#7E4DA8",
    "gray": "#777777",
}


def rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def wrap(draw, text, fnt, max_width):
    lines = []
    for para in text.split("\n"):
        line = ""
        for ch in para:
            test = line + ch
            if not line or draw.textbbox((0, 0), test, font=fnt)[2] <= max_width:
                line = test
            else:
                lines.append(line)
                line = ch
        if line:
            lines.append(line)
    return lines


def box(draw, xy, title, body, fill, stroke):
    x1, y1, x2, y2 = xy
    draw.rounded_rectangle(xy, radius=16, fill=rgb(fill), outline=rgb(stroke), width=3)
    draw.text(((x1 + x2) / 2, y1 + 18), title, font=HEAD, fill="#111111", anchor="mt")
    y = y1 + 62
    for line in wrap(draw, body, BODY, x2 - x1 - 28):
        draw.text((x1 + 16, y), line, font=BODY, fill="#222222")
        y += 30


def arrow(draw, start, end, color, label=None, offset=(0, 0), width=5):
    x1, y1 = start
    x2, y2 = end
    c = rgb(color)
    draw.line((x1, y1, x2, y2), fill=c, width=width)
    angle = math.atan2(y2 - y1, x2 - x1)
    size = 16
    pts = [
        (x2, y2),
        (x2 - size * math.cos(angle - math.pi / 6), y2 - size * math.sin(angle - math.pi / 6)),
        (x2 - size * math.cos(angle + math.pi / 6), y2 - size * math.sin(angle + math.pi / 6)),
    ]
    draw.polygon(pts, fill=c)
    if label:
        mx = (x1 + x2) / 2 + offset[0]
        my = (y1 + y2) / 2 + offset[1]
        bb = draw.textbbox((mx, my), label, font=SMALL, anchor="mm")
        draw.rounded_rectangle((bb[0] - 8, bb[1] - 4, bb[2] + 8, bb[3] + 4), radius=6, fill="white", outline=c, width=1)
        draw.text((mx, my), label, font=SMALL, fill=c, anchor="mm")


def save(image, name):
    path = OUT / name
    image.save(path)
    image.save(ROBOT / name)
    print(path)


def draw_rl_flow():
    w, h = 1000, 1260
    image = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(image)
    d.text((w / 2, 28), "從強化學習到實機開游的系統流程", font=TITLE, fill="#111111", anchor="mt")
    x1, x2 = 170, 830
    nodes = [
        ("MuJoCo / PPO 訓練", "建立仿生鰻魚模型，學習轉向策略", COL["sim"], COL["blue"]),
        ("Policy rerun 與固定 gait", "驗證策略軌跡，匯出 gait JSON", COL["sim"], COL["blue"]),
        ("Flutter 儀表板", "選擇 Wi-Fi、gait 與輸出模式", COL["pc"], COL["purple"]),
        ("Python backend", "載入 gait，產生 CPG 參數或連續角度", COL["pc"], COL["purple"]),
        ("XIAO ESP32S3 Sense", "接收 WebSocket JSON，轉成 UART packet", COL["esp"], COL["green"]),
        ("ESP32 控制端", "執行 CPG / angle mode，輸出 servo 命令", COL["esp"], COL["green"]),
        ("Bus servo 與機器鰻魚", "驅動六顆伺服，實體開游", COL["servo"], COL["orange"]),
        ("狀態與影像回傳", "servo status、控制參數與相機影像回到 Flutter", COL["data"], COL["gray"]),
    ]
    y = 105
    boxes = []
    for title, body, fill, stroke in nodes:
        xy = (x1, y, x2, y + 105)
        box(d, xy, title, body, fill, stroke)
        boxes.append(xy)
        y += 140
    labels = ["訓練結果", "匯出 / 選擇 gait", "HTTP 管理", "WebSocket JSON", "UART packet", "servo bus", "status / image"]
    colors = [COL["blue"], COL["blue"], COL["purple"], COL["purple"], COL["green"], COL["orange"], COL["gray"]]
    for i in range(len(boxes) - 1):
        xmid = w // 2
        arrow(d, (xmid, boxes[i][3]), (xmid, boxes[i + 1][1]), colors[i], labels[i], offset=(115, 0))
    d.text((70, 1225), "註：Python backend 以 WebSocket 傳送控制 JSON；camera board 與 control board 之間使用 UART binary packet。", font=SMALL, fill="#333333")
    save(image, "rl_to_real_robot_flow.png")


def draw_json_flow():
    w, h = 1100, 760
    image = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(image)
    d.text((w / 2, 24), "控制通訊與 JSON 訊息流程", font=TITLE, fill="#111111", anchor="mt")

    lanes = [
        ("Flutter", 105, COL["pc"], COL["purple"], "操作與監控"),
        ("Python", 310, COL["pc"], COL["purple"], "backend"),
        ("XIAO S3", 540, COL["esp"], COL["green"], "JSON bridge"),
        ("Control", 760, COL["esp"], COL["green"], "CPG / angle"),
        ("Servo", 975, COL["servo"], COL["orange"], "伺服驅動"),
    ]
    top, lane_bottom = 90, 610
    for title, x, fill, stroke, body in lanes:
        d.line((x, top + 95, x, lane_bottom), fill="#DDDDDD", width=2)
        box(d, (x - 82, top, x + 82, top + 92), title, body, fill, stroke)

    rows = [
        (230, "HTTP 管理", 105, 540, "Wi-Fi / host / start / gait"),
        (315, "HTTP 管理", 105, 310, "output_mode / recording"),
        (400, "WS 控制 JSON", 310, 540, "set_param / set_angle"),
        (485, "UART 封包", 540, 760, "ControlPacket / AnglePacket"),
        (570, "servo bus", 760, 975, "target position"),
        (650, "狀態回傳", 975, 540, "servo_status / angle_ack"),
    ]
    for y, row_label, x1, x2, label in rows:
        d.text((20, y), row_label, font=SMALL, fill="#333333", anchor="lm")
        color = COL["purple"] if "HTTP" in row_label or "WS" in row_label else (COL["green"] if "UART" in row_label else COL["gray"])
        start = (x1 + 84 if x1 < x2 else x1 - 84, y)
        end = (x2 - 84 if x1 < x2 else x2 + 84, y)
        arrow(d, start, end, color, label, offset=(0, -24), width=4)

    arrow(d, (540, 700), (105, 700), COL["blue"], "WS:81 binary JPEG camera frames", offset=(0, -24), width=4)
    d.text((20, 728), "JSON 欄位細節整理於後續表格；圖中只保留資料流與通訊協定。", font=SMALL, fill="#333333")
    save(image, "json_payload_flow.png")


if __name__ == "__main__":
    draw_rl_flow()
    draw_json_flow()
