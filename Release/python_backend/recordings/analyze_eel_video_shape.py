import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


def angle_wrap_deg(value):
    while value <= -180:
        value += 360
    while value > 180:
        value -= 360
    return value


def build_background(cap, sample_count=45):
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        return None
    idxs = np.linspace(0, max(0, total - 1), min(sample_count, total)).astype(int)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.resize(frame, (0, 0), fx=0.5, fy=0.5))
    if not frames:
        return None
    small_bg = np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    return cv2.resize(small_bg, (w, h), interpolation=cv2.INTER_LINEAR)


def foreground_mask(frame, background):
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    if background is not None:
        bg_lab = cv2.cvtColor(background, cv2.COLOR_BGR2LAB)
        diff = cv2.absdiff(lab, bg_lab)
        diff_score = (
            diff[:, :, 0].astype(np.uint16)
            + diff[:, :, 1].astype(np.uint16)
            + diff[:, :, 2].astype(np.uint16)
        )
        motion = diff_score > 48
    else:
        motion = np.zeros((h, w), dtype=bool)

    # The white silicone body is the most stable visual cue. The electronics are
    # darker, but only count them when they are also moving relative to the tub.
    bright_white = (hsv[:, :, 2] > 175) & (hsv[:, :, 1] < 88)
    dark_robot = (hsv[:, :, 2] < 68) & motion
    colored_robot = (motion & (hsv[:, :, 1] > 58) & (hsv[:, :, 2] > 65))

    mask = bright_white | dark_robot | colored_robot

    # Ignore timestamp, tub rim, and the top cloth/glare area that is not the robot.
    mask[: int(0.20 * h), :] = False
    mask[:, : int(0.04 * w)] = False
    mask[:, int(0.96 * w) :] = False
    mask[int(0.92 * h) :, :] = False

    mask = mask.astype(np.uint8) * 255
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    return mask


def pick_component(mask, prev_center=None):
    num, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    candidates = []
    h, w = mask.shape[:2]
    for label in range(1, num):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area < 160:
            continue
        if bw > 0.65 * w or bh > 0.55 * h:
            continue
        cx, cy = cents[label]
        pts = np.column_stack(np.where(labels == label))[:, ::-1].astype(np.float32)
        if len(pts) < 12:
            continue
        cov = np.cov((pts - pts.mean(axis=0)).T)
        vals, _ = np.linalg.eigh(cov)
        vals = np.sort(vals)[::-1]
        elongation = float(vals[0] / max(vals[1], 1e-6))
        dist_penalty = 0.0
        if prev_center is not None:
            dist_penalty = math.hypot(cx - prev_center[0], cy - prev_center[1]) * 1.8
        score = area * min(elongation, 15.0) - dist_penalty
        candidates.append((score, label, (cx, cy), area, elongation, (x, y, bw, bh)))
    if not candidates:
        return None
    candidates.sort(reverse=True, key=lambda item: item[0])
    return candidates[0]


def analyze_component(labels, label, previous_angle=None, bins=7):
    pts = np.column_stack(np.where(labels == label))[:, ::-1].astype(np.float64)
    center = pts.mean(axis=0)
    centered = pts - center
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vec = vecs[:, order[0]]
    angle = math.degrees(math.atan2(vec[1], vec[0]))
    if previous_angle is not None and abs(angle_wrap_deg(angle - previous_angle)) > 90:
        vec = -vec
        angle = angle_wrap_deg(angle + 180)

    tangent = vec / (np.linalg.norm(vec) + 1e-9)
    normal = np.array([-tangent[1], tangent[0]])
    s = centered @ tangent
    n = centered @ normal
    span = float(s.max() - s.min())
    width = float(np.percentile(n, 95) - np.percentile(n, 5))

    centers = []
    if span > 1:
        edges = np.linspace(s.min(), s.max(), bins + 1)
        for i in range(bins):
            take = (s >= edges[i]) & (s <= edges[i + 1])
            if take.sum() < 8:
                centers.append((float("nan"), float("nan")))
                continue
            ss = float(np.median(s[take]))
            nn = float(np.median(n[take]))
            xy = center + ss * tangent + nn * normal
            centers.append((float(xy[0]), float(xy[1])))

    curvature_px = 0.0
    chord_angle = float("nan")
    start_segment_angle = float("nan")
    end_segment_angle = float("nan")
    bend_angle = float("nan")
    valid = np.array([(x, y) for x, y in centers if not math.isnan(x)], dtype=np.float64)
    if len(valid) >= 3:
        line_start = valid[0]
        line_end = valid[-1]
        base = line_end - line_start
        base_len = np.linalg.norm(base)
        if base_len > 1:
            dists = np.abs(base[0] * (valid[:, 1] - line_start[1]) - base[1] * (valid[:, 0] - line_start[0])) / base_len
            curvature_px = float(np.max(dists))
            chord_angle = angle_wrap_deg(math.degrees(math.atan2(base[1], base[0])))
            start_vec = valid[min(2, len(valid) - 1)] - valid[0]
            end_vec = valid[-1] - valid[max(0, len(valid) - 3)]
            if np.linalg.norm(start_vec) > 1 and np.linalg.norm(end_vec) > 1:
                start_segment_angle = angle_wrap_deg(math.degrees(math.atan2(start_vec[1], start_vec[0])))
                end_segment_angle = angle_wrap_deg(math.degrees(math.atan2(end_vec[1], end_vec[0])))
                bend_angle = angle_wrap_deg(end_segment_angle - start_segment_angle)

    return {
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "axis_angle_deg": float(angle_wrap_deg(angle)),
        "chord_angle_deg": chord_angle,
        "start_segment_angle_deg": start_segment_angle,
        "end_segment_angle_deg": end_segment_angle,
        "body_bend_angle_deg": bend_angle,
        "length_px": span,
        "width_px": width,
        "elongation": float(vals[0] / max(vals[1], 1e-6)),
        "curvature_px": curvature_px,
        "centerline": centers,
    }


def draw_debug(frame, mask, component_mask, metrics, out_path):
    overlay = frame.copy()
    overlay[mask > 0] = (0.65 * overlay[mask > 0] + np.array([0, 180, 255]) * 0.35).astype(np.uint8)
    overlay[component_mask > 0] = (0.45 * overlay[component_mask > 0] + np.array([0, 255, 0]) * 0.55).astype(np.uint8)

    pts = [(int(x), int(y)) for x, y in metrics["centerline"] if not math.isnan(x)]
    for p in pts:
        cv2.circle(overlay, p, 6, (0, 0, 255), -1)
    for a, b in zip(pts, pts[1:]):
        cv2.line(overlay, a, b, (0, 0, 255), 3)

    cx, cy = int(metrics["center_x"]), int(metrics["center_y"])
    angle = math.radians(metrics["axis_angle_deg"])
    dx, dy = math.cos(angle) * 90, math.sin(angle) * 90
    cv2.arrowedLine(overlay, (cx, cy), (int(cx + dx), int(cy + dy)), (255, 0, 0), 4)
    cv2.putText(
        overlay,
        f"angle={metrics['axis_angle_deg']:+.1f} deg curv={metrics['curvature_px']:.1f}px",
        (30, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--debug-every", type=int, default=15)
    args = parser.parse_args()

    video = args.video
    out_dir = args.out_dir or video.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = out_dir / "debug_frames"

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"failed to open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    background = build_background(cap)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    rows = []
    prev_center = None
    prev_angle = None
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % args.stride != 0:
            frame_idx += 1
            continue
        mask = foreground_mask(frame, background)
        picked = pick_component(mask, prev_center)
        row = {
            "frame": frame_idx,
            "time_s": frame_idx / fps,
            "detected": 0,
        }
        if picked is not None:
            _, label, center, area, elongation, bbox = picked
            num, labels, _, _ = cv2.connectedComponentsWithStats(mask, 8)
            metrics = analyze_component(labels, label, prev_angle)
            component_mask = (labels == label).astype(np.uint8) * 255
            confidence = min(1.0, area / 7500.0) * min(1.0, elongation / 5.0)
            row.update(
                {
                    "detected": 1,
                    "confidence": confidence,
                    "area_px": area,
                    "bbox_x": bbox[0],
                    "bbox_y": bbox[1],
                    "bbox_w": bbox[2],
                    "bbox_h": bbox[3],
                    "center_x": metrics["center_x"],
                    "center_y": metrics["center_y"],
                    "axis_angle_deg": metrics["axis_angle_deg"],
                    "chord_angle_deg": metrics["chord_angle_deg"],
                    "start_segment_angle_deg": metrics["start_segment_angle_deg"],
                    "end_segment_angle_deg": metrics["end_segment_angle_deg"],
                    "body_bend_angle_deg": metrics["body_bend_angle_deg"],
                    "length_px": metrics["length_px"],
                    "width_px": metrics["width_px"],
                    "elongation": metrics["elongation"],
                    "curvature_px": metrics["curvature_px"],
                    "curvature_pct_length": metrics["curvature_px"] / max(metrics["length_px"], 1.0),
                }
            )
            for idx, (x, y) in enumerate(metrics["centerline"], start=1):
                row[f"seg{idx}_x"] = x
                row[f"seg{idx}_y"] = y
            if rows and rows[-1].get("detected") == 1:
                dt = row["time_s"] - rows[-1]["time_s"]
                if dt > 0:
                    row["speed_px_s"] = math.hypot(
                        row["center_x"] - rows[-1]["center_x"],
                        row["center_y"] - rows[-1]["center_y"],
                    ) / dt
                    row["yaw_rate_deg_s"] = angle_wrap_deg(
                        row["axis_angle_deg"] - rows[-1]["axis_angle_deg"]
                    ) / dt
            if frame_idx % args.debug_every == 0:
                draw_debug(frame, mask, component_mask, metrics, debug_dir / f"frame_{frame_idx:04d}.jpg")
            prev_center = center
            prev_angle = metrics["axis_angle_deg"]
        else:
            row["confidence"] = 0.0
        rows.append(row)
        frame_idx += 1
    cap.release()

    csv_path = out_dir / f"{video.stem}_shape_metrics.csv"
    fields = sorted({key for row in rows for key in row.keys()})
    ordered = [
        "frame",
        "time_s",
        "detected",
        "confidence",
        "center_x",
        "center_y",
        "axis_angle_deg",
        "chord_angle_deg",
        "start_segment_angle_deg",
        "end_segment_angle_deg",
        "body_bend_angle_deg",
        "yaw_rate_deg_s",
        "speed_px_s",
        "curvature_px",
        "curvature_pct_length",
        "length_px",
        "width_px",
        "elongation",
        "area_px",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
    ]
    fields = ordered + [f for f in fields if f not in ordered]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)

    detected = [r for r in rows if r.get("detected") == 1 and r.get("confidence", 0) >= 0.25]
    summary = {
        "video": str(video),
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "sample_stride": args.stride,
        "samples": len(rows),
        "detected_samples": len(detected),
        "csv": str(csv_path),
        "debug_dir": str(debug_dir),
    }
    if detected:
        angles = np.array([r["axis_angle_deg"] for r in detected], dtype=float)
        curv = np.array([r["curvature_pct_length"] for r in detected], dtype=float)
        bends = np.array(
            [r["body_bend_angle_deg"] for r in detected if not math.isnan(r.get("body_bend_angle_deg", float("nan")))],
            dtype=float,
        )
        centers_x = np.array([r["center_x"] for r in detected], dtype=float)
        centers_y = np.array([r["center_y"] for r in detected], dtype=float)
        metrics_summary = {
            "initial_angle_deg": float(angles[0]),
            "mean_angle_deg": float(np.mean(angles)),
            "median_angle_deg": float(np.median(angles)),
            "angle_range_deg": float(np.max(angles) - np.min(angles)),
            "mean_abs_curvature_pct_length": float(np.mean(np.abs(curv))),
            "center_track_dx_px": float(centers_x[-1] - centers_x[0]),
            "center_track_dy_px": float(centers_y[-1] - centers_y[0]),
        }
        if len(bends):
            metrics_summary.update(
                {
                    "initial_body_bend_angle_deg": float(bends[0]),
                    "mean_abs_body_bend_angle_deg": float(np.mean(np.abs(bends))),
                    "median_abs_body_bend_angle_deg": float(np.median(np.abs(bends))),
                    "max_abs_body_bend_angle_deg": float(np.max(np.abs(bends))),
                }
            )
        summary.update(metrics_summary)
    summary_path = out_dir / f"{video.stem}_shape_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
