import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


def wrap_deg(value):
    while value <= -180:
        value += 360
    while value > 180:
        value -= 360
    return value


def component_clusters(points, max_dist=150):
    if not points:
        return []
    remaining = set(range(len(points)))
    clusters = []
    while remaining:
        seed = remaining.pop()
        cluster = {seed}
        frontier = [seed]
        while frontier:
            i = frontier.pop()
            xi, yi = points[i]["center"]
            nearby = []
            for j in remaining:
                xj, yj = points[j]["center"]
                if math.hypot(xi - xj, yi - yj) <= max_dist:
                    nearby.append(j)
            for j in nearby:
                remaining.remove(j)
                cluster.add(j)
                frontier.append(j)
        clusters.append([points[i] for i in cluster])
    return clusters


def detect_dark_servo_candidates(frame):
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Black servo rectangles: low value, not the timestamp/rim, and not tiny wire specks.
    mask = ((hsv[:, :, 2] < 82) | (gray < 70)).astype(np.uint8) * 255
    mask[: int(0.035 * h), :] = 0
    mask[:, : int(0.07 * w)] = 0
    mask[:, int(0.90 * w) :] = 0
    mask[int(0.92 * h) :, :] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < 18 or area > 4500:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 4 or bh < 4:
            continue
        if bw > 150 or bh > 150:
            continue
        fill = area / float(bw * bh)
        if fill < 0.18:
            continue
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (rw, rh), rect_angle = rect
        long_side = max(rw, rh)
        short_side = max(1.0, min(rw, rh))
        aspect = long_side / short_side
        if aspect > 7.5:
            continue
        candidates.append(
            {
                "center": (float(cx), float(cy)),
                "area": float(area),
                "bbox": (int(x), int(y), int(bw), int(bh)),
                "aspect": float(aspect),
                "rect_angle": float(rect_angle),
                "contour": cnt,
            }
        )
    return mask, candidates


def choose_robot_cluster(candidates, prev_center=None):
    clusters = component_clusters(candidates, max_dist=155)
    scored = []
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        xs = np.array([p["center"][0] for p in cluster])
        ys = np.array([p["center"][1] for p in cluster])
        total_area = sum(p["area"] for p in cluster)
        span = math.hypot(float(xs.max() - xs.min()), float(ys.max() - ys.min()))
        center = (float(xs.mean()), float(ys.mean()))
        edge_penalty = 0.0
        if center[0] < 280:
            edge_penalty += 12000.0
        score = total_area + 18.0 * span + 520.0 * min(len(cluster), 6) - edge_penalty
        if prev_center is not None:
            score -= 3.0 * math.hypot(center[0] - prev_center[0], center[1] - prev_center[1])
        scored.append((score, cluster, center))
    if not scored:
        return [], None
    scored.sort(reverse=True, key=lambda item: item[0])
    return scored[0][1], scored[0][2]


def merge_nearby_candidates(cluster, radius=18):
    remaining = set(range(len(cluster)))
    merged = []
    while remaining:
        seed = remaining.pop()
        group = {seed}
        frontier = [seed]
        while frontier:
            i = frontier.pop()
            xi, yi = cluster[i]["center"]
            nearby = []
            for j in remaining:
                xj, yj = cluster[j]["center"]
                if math.hypot(xi - xj, yi - yj) <= radius:
                    nearby.append(j)
            for j in nearby:
                remaining.remove(j)
                group.add(j)
                frontier.append(j)

        parts = [cluster[i] for i in group]
        total_area = sum(p["area"] for p in parts)
        cx = sum(p["center"][0] * p["area"] for p in parts) / max(total_area, 1.0)
        cy = sum(p["center"][1] * p["area"] for p in parts) / max(total_area, 1.0)
        xs = [p["bbox"][0] for p in parts]
        ys = [p["bbox"][1] for p in parts]
        x2s = [p["bbox"][0] + p["bbox"][2] for p in parts]
        y2s = [p["bbox"][1] + p["bbox"][3] for p in parts]
        x, y = min(xs), min(ys)
        bbox = (x, y, max(x2s) - x, max(y2s) - y)
        merged.append(
            {
                "center": (float(cx), float(cy)),
                "area": float(total_area),
                "bbox": bbox,
                "aspect": 1.0,
                "rect_angle": 0.0,
                "contour": None,
            }
        )
    return merged


def ordered_servo_points(cluster, previous_angle=None):
    cluster = merge_nearby_candidates(cluster)
    if len(cluster) > 6:
        cluster = sorted(cluster, key=lambda p: p["area"], reverse=True)[:6]
    centers = np.array([p["center"] for p in cluster], dtype=np.float64)
    if len(centers) < 2:
        return [], {}

    center = centers.mean(axis=0)
    centered = centers - center
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    vec = vecs[:, np.argsort(vals)[-1]]
    angle = wrap_deg(math.degrees(math.atan2(vec[1], vec[0])))
    if previous_angle is not None and abs(wrap_deg(angle - previous_angle)) > 90:
        vec = -vec
        angle = wrap_deg(angle + 180)

    s = centered @ vec
    order = np.argsort(s)
    ordered = [cluster[i] for i in order]
    ordered_centers = centers[order]

    chord = ordered_centers[-1] - ordered_centers[0]
    chord_angle = wrap_deg(math.degrees(math.atan2(chord[1], chord[0])))
    bend_angle = float("nan")
    if len(ordered_centers) >= 4:
        start_vec = ordered_centers[min(2, len(ordered_centers) - 1)] - ordered_centers[0]
        end_vec = ordered_centers[-1] - ordered_centers[max(0, len(ordered_centers) - 3)]
        if np.linalg.norm(start_vec) > 1 and np.linalg.norm(end_vec) > 1:
            start_angle = wrap_deg(math.degrees(math.atan2(start_vec[1], start_vec[0])))
            end_angle = wrap_deg(math.degrees(math.atan2(end_vec[1], end_vec[0])))
            bend_angle = wrap_deg(end_angle - start_angle)
    else:
        start_angle = float("nan")
        end_angle = float("nan")

    lateral_error = 0.0
    base_len = np.linalg.norm(chord)
    if base_len > 1 and len(ordered_centers) >= 3:
        lateral = np.abs(
            chord[0] * (ordered_centers[:, 1] - ordered_centers[0, 1])
            - chord[1] * (ordered_centers[:, 0] - ordered_centers[0, 0])
        ) / base_len
        lateral_error = float(np.max(lateral))

    metrics = {
        "cluster_center_x": float(center[0]),
        "cluster_center_y": float(center[1]),
        "servo_count": int(len(ordered)),
        "servo_axis_angle_deg": float(angle),
        "servo_chord_angle_deg": float(chord_angle),
        "servo_start_angle_deg": float(start_angle),
        "servo_end_angle_deg": float(end_angle),
        "servo_bend_angle_deg": float(bend_angle),
        "servo_lateral_error_px": lateral_error,
    }
    return ordered, metrics


def draw_debug(frame, candidates, ordered, metrics, path):
    out = frame.copy()
    for cand in candidates:
        x, y, w, h = cand["bbox"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 180, 255), 2)
    pts = [(int(p["center"][0]), int(p["center"][1])) for p in ordered]
    for idx, p in enumerate(pts, start=1):
        cv2.circle(out, p, 8, (0, 255, 0), -1)
        cv2.putText(out, str(idx), (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    for a, b in zip(pts, pts[1:]):
        cv2.line(out, a, b, (0, 0, 255), 3)
    label = (
        f"servos={metrics.get('servo_count', 0)} "
        f"bend={metrics.get('servo_bend_angle_deg', float('nan')):+.1f}deg "
        f"axis={metrics.get('servo_axis_angle_deg', float('nan')):+.1f}deg"
    )
    cv2.putText(out, label, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), out)


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
    debug_dir = out_dir / "servo_debug_frames"

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"failed to open video: {video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    prev_center = None
    prev_angle = None
    rows = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % args.stride != 0:
            frame_idx += 1
            continue
        _, candidates = detect_dark_servo_candidates(frame)
        cluster, cluster_center = choose_robot_cluster(candidates, prev_center)
        ordered, metrics = ordered_servo_points(cluster, prev_angle)
        row = {
            "frame": frame_idx,
            "time_s": frame_idx / fps,
            "detected": 1 if len(ordered) >= 2 else 0,
            "candidate_count": len(candidates),
        }
        if len(ordered) >= 2:
            row.update(metrics)
            confidence = min(1.0, len(ordered) / 6.0) * min(1.0, sum(p["area"] for p in ordered) / 1800.0)
            row["confidence"] = confidence
            for i, p in enumerate(ordered, start=1):
                row[f"servo{i}_x"] = p["center"][0]
                row[f"servo{i}_y"] = p["center"][1]
                row[f"servo{i}_area_px"] = p["area"]
            prev_center = cluster_center
            prev_angle = metrics["servo_axis_angle_deg"]
            if frame_idx % args.debug_every == 0:
                draw_debug(frame, candidates, ordered, metrics, debug_dir / f"frame_{frame_idx:04d}.jpg")
        else:
            row["confidence"] = 0.0
        rows.append(row)
        frame_idx += 1
    cap.release()

    csv_path = out_dir / f"{video.stem}_servo_blocks.csv"
    ordered_fields = [
        "frame",
        "time_s",
        "detected",
        "confidence",
        "servo_count",
        "servo_bend_angle_deg",
        "servo_axis_angle_deg",
        "servo_chord_angle_deg",
        "servo_start_angle_deg",
        "servo_end_angle_deg",
        "servo_lateral_error_px",
        "cluster_center_x",
        "cluster_center_y",
        "candidate_count",
    ]
    fields = ordered_fields + sorted({k for row in rows for k in row.keys()} - set(ordered_fields))
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)

    detected = [r for r in rows if r.get("detected") == 1 and r.get("confidence", 0) >= 0.2]
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
        bends = np.array(
            [r["servo_bend_angle_deg"] for r in detected if not math.isnan(r.get("servo_bend_angle_deg", float("nan")))],
            dtype=float,
        )
        axes = np.array([r["servo_axis_angle_deg"] for r in detected], dtype=float)
        counts = np.array([r.get("servo_count", 0) for r in detected], dtype=float)
        summary.update(
            {
                "initial_servo_count": int(detected[0].get("servo_count", 0)),
                "mean_servo_count": float(np.mean(counts)),
                "initial_servo_axis_angle_deg": float(axes[0]),
                "median_servo_axis_angle_deg": float(np.median(axes)),
                "servo_axis_range_deg": float(np.max(axes) - np.min(axes)),
            }
        )
        if len(bends):
            summary.update(
                {
                    "initial_servo_bend_angle_deg": float(bends[0]),
                    "mean_abs_servo_bend_angle_deg": float(np.mean(np.abs(bends))),
                    "median_abs_servo_bend_angle_deg": float(np.median(np.abs(bends))),
                    "max_abs_servo_bend_angle_deg": float(np.max(np.abs(bends))),
                }
            )
    summary_path = out_dir / f"{video.stem}_servo_blocks_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
