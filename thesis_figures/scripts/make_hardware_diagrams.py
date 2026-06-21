from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import os

ROOT = Path(__file__).resolve().parents[1]
THESIS_OUT = Path(os.environ.get(
    "THESIS_FIGURES_DIR",
    r"C:\Users\Harley\Documents\Codex\2026-06-18\word-latex\work\thesis-latex\figures",
))
ROBOT_OUT = ROOT / "current" / "chapter3_methodology"
SOURCE = ROOT / "source_assets"
THESIS_OUT.mkdir(parents=True, exist_ok=True)
ROBOT_OUT.mkdir(parents=True, exist_ok=True)

FONT_REG = r"C:\Windows\Fonts\msjh.ttc"
FONT_BOLD = r"C:\Windows\Fonts\msjhbd.ttc"


def font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size)


def save_trimmed(img, name, padding=35):
    bg = Image.new(img.mode, img.size, "white")
    diff = Image.new("L", img.size, 0)
    px_img = img.load()
    px_diff = diff.load()
    for y in range(img.height):
        for x in range(img.width):
            if px_img[x, y] != (255, 255, 255):
                px_diff[x, y] = 255
    bbox = diff.getbbox()
    if bbox:
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img.width, x2 + padding)
        y2 = min(img.height, y2 + padding)
        img = img.crop((x1, y1, x2, y2))
    for out_dir in (THESIS_OUT, ROBOT_OUT):
        img.save(out_dir / name, dpi=(300, 300))


def contain(src, size, bg=(255, 255, 255)):
    src = src.convert("RGB")
    src.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, bg)
    x = (size[0] - src.width) // 2
    y = (size[1] - src.height) // 2
    canvas.paste(src, (x, y))
    return canvas


def center(draw, box, text, ft, fill=(25, 25, 25)):
    x1, y1, x2, y2 = box
    bb = draw.multiline_textbbox((0, 0), text, font=ft, spacing=8, align="center")
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    draw.multiline_text(
        (x1 + (x2 - x1 - w) / 2, y1 + (y2 - y1 - h) / 2),
        text,
        font=ft,
        fill=fill,
        spacing=8,
        align="center",
    )


def box(draw, rect, fill, outline, width=5, radius=22):
    draw.rounded_rectangle(rect, radius=radius, fill=fill, outline=outline, width=width)


def arrow_head(draw, start, end, color, size=18):
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = max((dx * dx + dy * dy) ** 0.5, 1)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    p1 = end
    p2 = (end[0] - ux * size + px * size * 0.55, end[1] - uy * size + py * size * 0.55)
    p3 = (end[0] - ux * size - px * size * 0.55, end[1] - uy * size - py * size * 0.55)
    draw.polygon([p1, p2, p3], fill=color)


def line_arrow(draw, start, end, color, width=6, label=None, label_shift=-34):
    draw.line([start, end], fill=color, width=width)
    arrow_head(draw, start, end, color)
    if label:
        ft = font(28, True)
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        bb = draw.textbbox((0, 0), label, font=ft)
        draw.text((mx - (bb[2] - bb[0]) / 2, my + label_shift), label, font=ft, fill=color)


def poly_arrow(draw, pts, color, width=6, label=None, label_at=None):
    draw.line(pts, fill=color, width=width, joint="curve")
    arrow_head(draw, pts[-2], pts[-1], color)
    if label and label_at:
        ft = font(26, True)
        r = 20
        draw.ellipse((label_at[0] - r, label_at[1] - r, label_at[0] + r, label_at[1] + r), fill="white", outline=color, width=4)
        bb = draw.textbbox((0, 0), label, font=ft)
        draw.text((label_at[0] - (bb[2] - bb[0]) / 2, label_at[1] - (bb[3] - bb[1]) / 2 - 2), label, font=ft, fill=color)


def architecture():
    w, h = 1800, 850
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)

    blue = ((238, 245, 255), (32, 84, 180))
    purple = ((247, 241, 255), (115, 40, 130))
    green = ((238, 252, 239), (38, 145, 54))
    orange = ((255, 246, 232), (212, 106, 18))
    gray = ((248, 248, 248), (105, 105, 105))

    app = (70, 120, 375, 300)
    cam = (585, 120, 900, 300)
    ctl = (1185, 120, 1490, 300)
    servo = (1185, 540, 1490, 720)
    body = (585, 540, 900, 720)

    for rect, col in [(app, blue), (cam, purple), (ctl, green), (servo, orange), (body, gray)]:
        box(draw, rect, *col)

    center(draw, app, "Flutter\n儀表板", font(34, True))
    center(draw, cam, "XIAO ESP32S3\nSense", font(34, True))
    center(draw, ctl, "ESP32\n控制端", font(34, True))
    center(draw, servo, "Bus Servo\n控制板", font(34, True))
    center(draw, body, "仿生機器鰻魚\n本體", font(34, True))

    line_arrow(draw, (375, 170), (585, 170), blue[1], label="Wi-Fi / HTTP")
    line_arrow(draw, (585, 250), (375, 250), blue[1], label="video / status", label_shift=18)
    line_arrow(draw, (900, 170), (1185, 170), orange[1], label="UART D9 / D10")
    line_arrow(draw, (1185, 250), (900, 250), green[1], label="status", label_shift=18)
    line_arrow(draw, (1338, 300), (1338, 540), orange[1], label="UART D6 / D7")
    line_arrow(draw, (1185, 630), (900, 630), gray[1], label="servo bus")

    save_trimmed(img, "dual_esp32_architecture.png")


def pin_diagram():
    w, h = 1900, 980
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)

    purple = ((247, 241, 255), (115, 40, 130))
    green = ((238, 252, 239), (38, 145, 54))
    orange = ((255, 246, 232), (212, 106, 18))
    gray = (90, 90, 90)

    cam = (80, 80, 500, 700)
    ctl = (740, 80, 1160, 700)
    bus = (1410, 80, 1810, 620)
    for rect, col in [(cam, purple), (ctl, green), (bus, orange)]:
        box(draw, rect, *col)

    center(draw, (cam[0], 100, cam[2], 160), "XIAO ESP32S3 Sense", font(30, True))
    center(draw, (ctl[0], 100, ctl[2], 160), "ESP32 控制端", font(30, True))
    center(draw, (bus[0], 100, bus[2], 160), "Bus Servo 控制板", font(30, True))

    def pin(rect, label, col):
        fill, outline = col
        box(draw, rect, fill, outline, width=3, radius=10)
        center(draw, rect, label, font(25))
        return ((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2)

    cam_d9 = pin((330, 205, 475, 260), "D9", purple)
    cam_d10 = pin((330, 320, 475, 375), "D10", purple)
    cam_g = pin((330, 595, 475, 650), "GND", purple)

    ctl_d9 = pin((975, 205, 1120, 260), "D9", green)
    ctl_d10 = pin((975, 320, 1120, 375), "D10", green)
    ctl_d6 = pin((975, 420, 1120, 475), "D6", green)
    ctl_d7 = pin((975, 540, 1120, 595), "D7", green)
    ctl_g = pin((780, 595, 925, 650), "GND", green)

    bus_rx = pin((1450, 205, 1595, 260), "RX", orange)
    bus_tx = pin((1450, 330, 1595, 385), "TX", orange)
    bus_g = pin((1450, 455, 1595, 510), "GND", orange)
    pin((1450, 530, 1595, 585), "VM/V+", orange)

    # signal paths; use fixed corridors so lines do not cross text.
    line_arrow(draw, (475, cam_d9[1]), (975, ctl_d9[1]), orange[1], label="1")
    line_arrow(draw, (975, ctl_d10[1]), (475, cam_d10[1]), orange[1], label="2", label_shift=18)
    poly_arrow(draw, [(1120, ctl_d6[1]), (1240, ctl_d6[1]), (1240, 160), (1450, bus_rx[1])], orange[1], label="3", label_at=(1240, 305))
    poly_arrow(draw, [(1450, bus_tx[1]), (1320, bus_tx[1]), (1320, 705), (1120, ctl_d7[1])], orange[1], label="4", label_at=(1320, 535))

    line_arrow(draw, (475, cam_g[1]), (780, ctl_g[1]), gray, label="G", label_shift=18)

    # Servo daisy chain, drawn below the bus controller.
    draw.line([(1610, 620), (1610, 770)], fill=gray, width=5)
    draw.line([(1370, 770), (1850, 770)], fill=gray, width=5)
    for i, x in enumerate([1370, 1466, 1562, 1658, 1754, 1850], 1):
        draw.line([(x, 770), (x, 815)], fill=gray, width=4)
        rect = (x - 38, 815, x + 38, 875)
        box(draw, rect, (248, 248, 248), gray, width=3, radius=8)
        center(draw, rect, f"S{i}", font(23))

    save_trimmed(img, "pin_wiring_diagram.png")


def hardware_platform():
    xiao_path = SOURCE / "xiao_esp32s3_sense.png"
    servo_path = SOURCE / "xiao_bus_servo_driver.png"
    if not xiao_path.exists() or not servo_path.exists():
        return

    w, h = 1900, 820
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)

    purple = ((247, 241, 255), (115, 40, 130))
    green = ((238, 252, 239), (38, 145, 54))
    orange = ((255, 246, 232), (212, 106, 18))
    gray = (95, 95, 95)

    left = (60, 70, 590, 620)
    mid = (685, 70, 1215, 620)
    right = (1310, 70, 1840, 620)
    box(draw, left, *purple)
    box(draw, mid, *green)
    box(draw, right, *orange)

    center(draw, (left[0], left[1] + 20, left[2], left[1] + 82), "XIAO ESP32S3 Sense", font(31, True))
    center(draw, (mid[0], mid[1] + 20, mid[2], mid[1] + 82), "XIAO ESP32S3 Sense", font(31, True))
    center(draw, (right[0], right[1] + 20, right[2], right[1] + 82), "XIAO Bus Servo Driver", font(31, True))

    xiao_img = Image.open(xiao_path)
    servo_img = Image.open(servo_path)
    xiao_panel = contain(xiao_img, (390, 285))
    servo_panel = contain(servo_img, (390, 285))
    img.paste(xiao_panel, (130, 185))
    img.paste(xiao_panel.copy(), (755, 185))
    img.paste(servo_panel, (1380, 185))

    center(draw, (left[0] + 25, 505, left[2] - 25, 590), "camera board\n影像、Wi-Fi 與指令轉送", font(24))
    center(draw, (mid[0] + 25, 505, mid[2] - 25, 590), "control board\nCPG 與伺服命令產生", font(24))
    center(draw, (right[0] + 25, 505, right[2] - 25, 590), "servo driver\nTTL UART 至 bus servo", font(24))

    line_arrow(draw, (590, 345), (685, 345), purple[1], label="UART")
    line_arrow(draw, (1215, 345), (1310, 345), orange[1], label="UART")

    draw.line([(1575, 620), (1575, 690)], fill=gray, width=5)
    draw.line([(1365, 690), (1785, 690)], fill=gray, width=5)
    for i, x in enumerate([1365, 1449, 1533, 1617, 1701, 1785], 1):
        draw.line([(x, 690), (x, 720)], fill=gray, width=4)
        rect = (x - 32, 720, x + 32, 772)
        box(draw, rect, (248, 248, 248), gray, width=3, radius=8)
        center(draw, rect, f"S{i}", font(20))

    save_trimmed(img, "hardware_platform.png")


def json_flow():
    w, h = 1900, 900
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)

    blue = ((238, 245, 255), (32, 84, 180))
    purple = ((247, 241, 255), (115, 40, 130))
    green = ((238, 252, 239), (38, 145, 54))
    orange = ((255, 246, 232), (212, 106, 18))
    gray = ((248, 248, 248), (95, 95, 95))

    top_y = 80
    boxes = [
        ((80, top_y, 360, top_y + 170), "Flutter UI\n操作介面", blue),
        ((470, top_y, 750, top_y + 170), "JSON encode\ncmd / params", purple),
        ((860, top_y, 1140, top_y + 170), "WebSocket\n控制通道", gray),
        ((1250, top_y, 1530, top_y + 170), "ESP32\nJSON decode", green),
        ((1640, top_y, 1860, top_y + 170), "CPG / Servo\n控制", orange),
    ]
    for rect, text, col in boxes:
        box(draw, rect, *col)
        center(draw, rect, text, font(30, True))

    line_arrow(draw, (360, top_y + 85), (470, top_y + 85), blue[1])
    line_arrow(draw, (750, top_y + 85), (860, top_y + 85), purple[1])
    line_arrow(draw, (1140, top_y + 85), (1250, top_y + 85), gray[1])
    line_arrow(draw, (1530, top_y + 85), (1640, top_y + 85), green[1])

    # Return path.
    y = 470
    return_boxes = [
        ((1640, y, 1860, y + 150), "servo\n狀態", orange),
        ((1250, y, 1530, y + 150), "JSON encode\ntype / data", green),
        ((860, y, 1140, y + 150), "WebSocket\n回傳", gray),
        ((470, y, 750, y + 150), "JSON decode\n更新資料", purple),
        ((80, y, 360, y + 150), "Flutter UI\n表格 / 狀態", blue),
    ]
    for rect, text, col in return_boxes:
        box(draw, rect, *col)
        center(draw, rect, text, font(29, True))

    line_arrow(draw, (1640, y + 75), (1530, y + 75), orange[1], label=None)
    line_arrow(draw, (1250, y + 75), (1140, y + 75), green[1], label=None)
    line_arrow(draw, (860, y + 75), (750, y + 75), gray[1], label=None)
    line_arrow(draw, (470, y + 75), (360, y + 75), purple[1], label=None)

    # Example payloads.
    box(draw, (450, 290, 770, 390), (250, 250, 250), (150, 150, 150), width=2, radius=12)
    center(draw, (450, 290, 770, 390), '{"cmd": "set_param",\n "frequency": 1.0}', font(22))
    box(draw, (1160, 290, 1510, 390), (250, 250, 250), (150, 150, 150), width=2, radius=12)
    center(draw, (1160, 290, 1510, 390), '{"type": "servo_status",\n "target": [...]}', font(22))

    save_trimmed(img, "json_message_flow.png")


architecture()
hardware_platform()
pin_diagram()
json_flow()
print(THESIS_OUT)
print(ROBOT_OUT)
