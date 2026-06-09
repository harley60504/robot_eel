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


def blobs_from_mask(mask, min_area, max_area=None):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for cnt in cnts:
        area = float(cv2.contourArea(cnt))
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        m = cv2.moments(cnt)
        if m["m00"] <= 1e-6:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        blobs.append(
            {
                "center": (float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])),
                "area": area,
                "bbox": (int(x), int(y), int(w), int(h)),
                "contour": cnt,
            }
        )
    return blobs


def detect_feature_masks(frame):
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # White outer body: bright, low-saturation material. This also catches some
    # splash, so it is only used together with nearby black/red features.
    white = ((hsv[:, :, 2] > 145) & (hsv[:, :, 1] < 105)).astype(np.uint8) * 255

    # Inner black servo blocks and black wiring. Rectangular/area filters happen
    # after contour extraction.
    black = ((hsv[:, :, 2] < 95) | (gray < 82)).astype(np.uint8) * 255

    # Red/orange stickers or marks on the top of the servo shell.
    red1 = cv2.inRange(hsv, np.array([0, 28, 45]), np.array([18, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([155, 24, 45]), np.array([180, 255, 255]))
    orange = cv2.inRange(hsv, np.array([18, 25, 55]), np.array([45, 255, 255]))
    pink_brown = cv2.inRange(hsv, np.array([145, 18, 45]), np.array([179, 210, 230]))
    red = cv2.bitwise_or(cv2.bitwise_or(red1, red2), cv2.bitwise_or(orange, pink_brown))

    # Remove tub edges, timestamp, and the extreme bottom logo area.
    for mask in (white, black, red):
        mask[: int(0.03 * h), :] = 0
        mask[:, : int(0.055 * w)] = 0
        mask[:, int(0.90 * w) :] = 0
        mask[int(0.94 * h) :, :] = 0

    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
    black = cv2.morphologyEx(black, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    red = cv2.dilate(red, np.ones((5, 5), np.uint8), iterations=1)
    return white, black, red


def count_nearby(blobs, cx, cy, rx, ry):
    count = 0
    area = 0.0
    for b in blobs:
        x, y = b["center"]
        if abs(x - cx) <= rx and abs(y - cy) <= ry:
            count += 1
            area += b["area"]
    return count, area


def candidate_white_components(white, black_blobs, red_blobs, prev_center=None):
    cnts, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in cnts:
        area = float(cv2.contourArea(cnt))
        if area < 650 or area > 65000:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 25 or h < 25:
            continue
        if w > 520 or h > 520:
            continue
        m = cv2.moments(cnt)
        if m["m00"] <= 1e-6:
            continue
        cx, cy = float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])

        black_count, black_area = count_nearby(black_blobs, cx, cy, max(80, w * 0.9), max(80, h * 0.9))
        red_count, red_area = count_nearby(red_blobs, cx, cy, max(95, w * 1.1), max(95, h * 1.1))
        if black_count < 1 or red_count < 1:
            continue

        pts = cnt.reshape(-1, 2).astype(np.float64)
        cov = np.cov((pts - pts.mean(axis=0)).T)
        vals, _ = np.linalg.eigh(cov)
        vals = np.sort(vals)[::-1]
        elong = float(vals[0] / max(vals[1], 1e-6))
        score = area + 1800 * red_count + 1200 * black_count + 500 * min(elong, 8.0)
        if prev_center is not None:
            score -= 2.5 * math.hypot(cx - prev_center[0], cy - prev_center[1])
        candidates.append(
            {
                "score": score,
                "center": (cx, cy),
                "area": area,
                "bbox": (x, y, w, h),
                "black_count": black_count,
                "red_count": red_count,
                "black_area": black_area,
                "red_area": red_area,
                "contour": cnt,
            }
        )
    candidates.sort(reverse=True, key=lambda c: c["score"])
    return candidates


def candidate_feature_clusters(white, black_blobs, red_blobs, prev_center=None):
    seeds = []
    for b in black_blobs:
        x, y, w, h = b["bbox"]
        if w > 130 or h > 130:
            continue
        bx, by = b["center"]
        nearby_red, red_area = count_nearby(red_blobs, bx, by, 95, 95)
        if nearby_red < 1:
            continue
        x0, y0 = max(0, int(bx - 115)), max(0, int(by - 115))
        x1, y1 = min(white.shape[1], int(bx + 115)), min(white.shape[0], int(by + 115))
        white_area = float(np.count_nonzero(white[y0:y1, x0:x1]))
        if white_area < 120:
            continue
        seeds.append({"center": (bx, by), "area": b["area"] + red_area + 0.02 * white_area})

    if not seeds:
        return []

    remaining = set(range(len(seeds)))
    clusters = []
    while remaining:
        seed = remaining.pop()
        group = {seed}
        frontier = [seed]
        while frontier:
            i = frontier.pop()
            xi, yi = seeds[i]["center"]
            close = []
            for j in remaining:
                xj, yj = seeds[j]["center"]
                if math.hypot(xi - xj, yi - yj) <= 170:
                    close.append(j)
            for j in close:
                remaining.remove(j)
                group.add(j)
                frontier.append(j)
        clusters.append([seeds[i] for i in group])

    candidates = []
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        total = sum(p["area"] for p in cluster)
        cx = sum(p["center"][0] * p["area"] for p in cluster) / max(total, 1.0)
        cy = sum(p["center"][1] * p["area"] for p in cluster) / max(total, 1.0)
        xs = [p["center"][0] for p in cluster]
        ys = [p["center"][1] for p in cluster]
        x, y = int(min(xs) - 70), int(min(ys) - 70)
        w, h = int(max(xs) - min(xs) + 140), int(max(ys) - min(ys) + 140)
        black_count, black_area = count_nearby(black_blobs, cx, cy, max(120, w * 0.75), max(120, h * 0.75))
        red_count, red_area = count_nearby(red_blobs, cx, cy, max(130, w * 0.85), max(130, h * 0.85))
        score = total + 1600 * len(cluster) + 850 * red_count + 600 * black_count
        if prev_center is not None:
            score -= 2.0 * math.hypot(cx - prev_center[0], cy - prev_center[1])
        contour = np.array(
            [[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]],
            dtype=np.int32,
        )
        candidates.append(
            {
                "score": score,
                "center": (float(cx), float(cy)),
                "area": float(total),
                "bbox": (int(x), int(y), int(w), int(h)),
                "black_count": int(black_count),
                "red_count": int(red_count),
                "black_area": float(black_area),
                "red_area": float(red_area),
                "contour": contour,
            }
        )
    candidates.sort(reverse=True, key=lambda c: c["score"])
    return candidates


def collect_robot_points(candidate, black_blobs, red_blobs):
    cx, cy = candidate["center"]
    x, y, w, h = candidate["bbox"]
    rx, ry = max(95, w * 1.25), max(95, h * 1.25)
    pts = []
    for source, blobs in (("black", black_blobs), ("red", red_blobs)):
        for b in blobs:
            bx, by = b["center"]
            if abs(bx - cx) <= rx and abs(by - cy) <= ry:
                weight = 2.0 if source == "red" else 1.5
                pts.append((bx, by, weight, source, b["area"]))

    # Also use white contour points, but lightly, so the centerline follows the
    # marked servo chain rather than splash outline.
    contour_pts = candidate["contour"].reshape(-1, 2)
    if len(contour_pts) > 0:
        step = max(1, len(contour_pts) // 80)
        for px, py in contour_pts[::step]:
            pts.append((float(px), float(py), 0.18, "white_edge", 1.0))
    return pts


def selected_feature_blobs(candidate, black_blobs, red_blobs):
    cx, cy = candidate["center"]
    x, y, w, h = candidate["bbox"]
    rx, ry = max(95, w * 0.65), max(95, h * 0.65)
    selected_black = []
    selected_red = []
    for b in black_blobs:
        bx, by = b["center"]
        if abs(bx - cx) <= rx and abs(by - cy) <= ry:
            selected_black.append(b)
    for r in red_blobs:
        rx0, ry0 = r["center"]
        if abs(rx0 - cx) <= rx and abs(ry0 - cy) <= ry:
            selected_red.append(r)
    return selected_black, selected_red


def collapse_feature_points(points, max_groups=6):
    if len(points) <= max_groups:
        return points

    arr = np.array([[p["center"][0], p["center"][1]] for p in points], dtype=np.float64)
    weights = np.array([p.get("weight", 1.0) for p in points], dtype=np.float64)
    center = np.average(arr, axis=0, weights=weights)
    centered = arr - center
    cov = (centered * weights[:, None]).T @ centered / max(weights.sum(), 1e-6)
    vals, vecs = np.linalg.eigh(cov)
    axis = vecs[:, np.argsort(vals)[-1]]
    s = centered @ axis
    order = np.argsort(s)

    groups = []
    for idx in order:
        point = points[int(idx)]
        si = float(s[int(idx)])
        if not groups or abs(si - groups[-1]["s_weighted"] / max(groups[-1]["weight"], 1e-6)) > 24:
            groups.append({"items": [point], "weight": point.get("weight", 1.0), "s_weighted": si * point.get("weight", 1.0)})
        else:
            groups[-1]["items"].append(point)
            groups[-1]["weight"] += point.get("weight", 1.0)
            groups[-1]["s_weighted"] += si * point.get("weight", 1.0)

    collapsed = []
    for group in groups:
        total = sum(p.get("weight", 1.0) for p in group["items"])
        representative = max(group["items"], key=lambda p: p.get("area", 0.0) + p.get("weight", 1.0))
        cx, cy = representative["center"]
        collapsed.append(
            {
                "center": (float(cx), float(cy)),
                "weight": float(total),
                "items": group["items"],
                "bbox": representative.get("bbox"),
                "s": group["s_weighted"] / max(group["weight"], 1e-6),
            }
        )

    if len(collapsed) > max_groups:
        collapsed = select_axis_distributed_groups(collapsed, max_groups)
    return collapsed


def select_axis_distributed_groups(groups, target_count):
    groups = sorted(groups, key=lambda g: g["s"])
    s_vals = np.array([g["s"] for g in groups], dtype=np.float64)
    span = float(s_vals[-1] - s_vals[0])
    if span <= 1:
        return groups[:target_count]

    expected_gap = span / max(target_count - 1, 1)
    # Head/tail shell artifacts usually sit at an extreme with a much larger
    # adjacent gap than the servo pitch. Trim one such extreme before sampling.
    trimmed = groups[:]
    while len(trimmed) > target_count:
        gaps = np.diff([g["s"] for g in trimmed])
        left_gap = gaps[0] if len(gaps) else 0
        right_gap = gaps[-1] if len(gaps) else 0
        left_big = left_gap > expected_gap * 1.55
        right_big = right_gap > expected_gap * 1.55
        if left_big or right_big:
            if left_big and (not right_big or left_gap >= right_gap):
                trimmed = trimmed[1:]
            else:
                trimmed = trimmed[:-1]
            continue
        break

    if len(trimmed) <= target_count:
        return trimmed

    s_vals = np.array([g["s"] for g in trimmed], dtype=np.float64)
    targets = np.linspace(s_vals[0], s_vals[-1], target_count)
    selected = []
    used = set()
    for target in targets:
        best_idx = None
        best_score = float("inf")
        for i, group in enumerate(trimmed):
            if i in used:
                continue
            # Stay near the evenly spaced servo pitch, but still prefer real
            # red/black feature strength when two points are close.
            score = abs(group["s"] - target) - 0.12 * group["weight"]
            if score < best_score:
                best_idx = i
                best_score = score
        if best_idx is not None:
            used.add(best_idx)
            selected.append(trimmed[best_idx])
    return sorted(selected, key=lambda g: g["s"])


def centerline_from_servo_features(candidate, black_blobs, red_blobs, previous_angle=None, expected_count=6):
    selected_black, selected_red = selected_feature_blobs(candidate, black_blobs, red_blobs)
    feature_points = []

    for b in selected_black:
        x, y, w, h = b["bbox"]
        # A servo observation must come from a dark candidate box. Some true
        # servos are partially occluded and appear as small yellow boxes, so do
        # not reject small boxes too aggressively; red marks may only boost a
        # nearby black box, never become a servo center by themselves.
        if w < 4 or h < 4 or w > 80 or h > 80:
            continue
        if b["area"] < 18:
            continue
        nearby_red = 0
        for r in selected_red:
            if math.hypot(b["center"][0] - r["center"][0], b["center"][1] - r["center"][1]) <= 45:
                nearby_red += 1
        area_weight = max(1.0, min(5.5, b["area"] / 45.0)) + nearby_red * 2.2
        # Very large black blobs near the ends are often the head shell, not a
        # servo rectangle. Keep them visible as weak evidence instead of letting
        # them dominate group selection.
        if b["area"] > 900:
            area_weight *= 0.35
        feature_points.append(
            {
                "center": b["center"],
                "weight": area_weight,
                "source": "black",
                "bbox": b["bbox"],
                "area": b["area"],
            }
        )

    if len(feature_points) < 3:
        return None

    servos = collapse_feature_points(feature_points, max_groups=expected_count)
    if len(servos) < 3:
        return None

    arr = np.array([[p["center"][0], p["center"][1]] for p in servos], dtype=np.float64)
    weights = np.array([p["weight"] for p in servos], dtype=np.float64)
    center = np.average(arr, axis=0, weights=weights)
    centered = arr - center
    cov = (centered * weights[:, None]).T @ centered / max(weights.sum(), 1e-6)
    vals, vecs = np.linalg.eigh(cov)
    axis = vecs[:, np.argsort(vals)[-1]]
    angle = wrap_deg(math.degrees(math.atan2(axis[1], axis[0])))
    if previous_angle is not None and abs(wrap_deg(angle - previous_angle)) > 90:
        axis = -axis
        angle = wrap_deg(angle + 180)

    s = centered @ axis
    order = np.argsort(s)
    ordered = arr[order]
    chord = ordered[-1] - ordered[0]
    chord_len = np.linalg.norm(chord)
    chord_angle = float("nan")
    lateral_px = float("nan")
    bend = float("nan")
    if chord_len > 1:
        chord_angle = wrap_deg(math.degrees(math.atan2(chord[1], chord[0])))
        lateral = np.abs(
            chord[0] * (ordered[:, 1] - ordered[0, 1])
            - chord[1] * (ordered[:, 0] - ordered[0, 0])
        ) / chord_len
        lateral_px = float(np.max(lateral))
    if len(ordered) >= 4:
        a = ordered[min(2, len(ordered) - 1)] - ordered[0]
        b = ordered[-1] - ordered[max(0, len(ordered) - 3)]
        if np.linalg.norm(a) > 1 and np.linalg.norm(b) > 1:
            bend = wrap_deg(math.degrees(math.atan2(b[1], b[0])) - math.degrees(math.atan2(a[1], a[0])))

    return {
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "axis_angle_deg": float(angle),
        "chord_angle_deg": float(chord_angle),
        "body_bend_angle_deg": float(bend),
        "lateral_error_px": lateral_px,
        "length_px": float(chord_len),
        "centerline": [(float(x), float(y)) for x, y in ordered],
        "servo_bboxes": [servos[int(i)].get("bbox") for i in order],
        "servo_feature_count": int(len(feature_points)),
        "servo_center_count": int(len(ordered)),
    }


def centerline_from_roi_mask(white, black_blobs, red_blobs, candidate, previous_angle=None, bins=6):
    x, y, w, h = candidate["bbox"]
    pad = 16
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(white.shape[1], x + w + pad)
    y1 = min(white.shape[0], y + h + pad)
    roi = np.zeros_like(white)
    roi[y0:y1, x0:x1] = white[y0:y1, x0:x1]

    selected_black, selected_red = selected_feature_blobs(candidate, black_blobs, red_blobs)
    for b in selected_black:
        bx, by, bw, bh = b["bbox"]
        cv2.rectangle(roi, (bx - 8, by - 8), (bx + bw + 8, by + bh + 8), 255, -1)
    for r in selected_red:
        cv2.circle(roi, (int(r["center"][0]), int(r["center"][1])), 18, 255, -1)

    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, np.ones((17, 17), np.uint8))
    roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    cnts, _ = cv2.findContours(roi[y0:y1, x0:x1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 220:
        return None
    cnt[:, 0, 0] += x0
    cnt[:, 0, 1] += y0
    mask = np.zeros_like(white)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    ys, xs = np.where(mask > 0)
    if len(xs) < 30:
        return None

    pts = np.column_stack([xs, ys]).astype(np.float64)
    center = pts.mean(axis=0)
    centered = pts - center
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    vec = vecs[:, np.argsort(vals)[-1]]
    angle = wrap_deg(math.degrees(math.atan2(vec[1], vec[0])))
    if previous_angle is not None and abs(wrap_deg(angle - previous_angle)) > 90:
        vec = -vec
        angle = wrap_deg(angle + 180)
    normal = np.array([-vec[1], vec[0]])
    s = centered @ vec
    n = centered @ normal
    span = float(s.max() - s.min())
    if span < 12:
        return None

    centers = []
    edges = np.linspace(s.min(), s.max(), bins + 1)
    for i in range(bins):
        take = (s >= edges[i]) & (s <= edges[i + 1])
        if take.sum() < 8:
            centers.append((float("nan"), float("nan")))
            continue
        ss = float(np.median(s[take]))
        nn = float(np.median(n[take]))
        xy = center + ss * vec + nn * normal
        centers.append((float(xy[0]), float(xy[1])))

    valid = np.array([(px, py) for px, py in centers if not math.isnan(px)], dtype=np.float64)
    bend = float("nan")
    chord_angle = float("nan")
    lateral_px = float("nan")
    if len(valid) >= 3:
        chord = valid[-1] - valid[0]
        chord_len = np.linalg.norm(chord)
        if chord_len > 1:
            chord_angle = wrap_deg(math.degrees(math.atan2(chord[1], chord[0])))
            lateral = np.abs(
                chord[0] * (valid[:, 1] - valid[0, 1])
                - chord[1] * (valid[:, 0] - valid[0, 0])
            ) / chord_len
            lateral_px = float(np.max(lateral))
        if len(valid) >= 4:
            a = valid[min(2, len(valid) - 1)] - valid[0]
            b = valid[-1] - valid[max(0, len(valid) - 3)]
            if np.linalg.norm(a) > 1 and np.linalg.norm(b) > 1:
                bend = wrap_deg(math.degrees(math.atan2(b[1], b[0])) - math.degrees(math.atan2(a[1], a[0])))

    return {
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "axis_angle_deg": float(angle),
        "chord_angle_deg": float(chord_angle),
        "body_bend_angle_deg": float(bend),
        "lateral_error_px": lateral_px,
        "length_px": span,
        "centerline": centers,
        "body_mask": mask,
    }


def fit_centerline(points, previous_angle=None, bins=6):
    if len(points) < 3:
        return None
    arr = np.array([[p[0], p[1]] for p in points], dtype=np.float64)
    weights = np.array([p[2] for p in points], dtype=np.float64)
    center = np.average(arr, axis=0, weights=weights)
    centered = arr - center
    cov = (centered * weights[:, None]).T @ centered / max(weights.sum(), 1e-6)
    vals, vecs = np.linalg.eigh(cov)
    vec = vecs[:, np.argsort(vals)[-1]]
    angle = wrap_deg(math.degrees(math.atan2(vec[1], vec[0])))
    if previous_angle is not None and abs(wrap_deg(angle - previous_angle)) > 90:
        vec = -vec
        angle = wrap_deg(angle + 180)
    normal = np.array([-vec[1], vec[0]])
    s = centered @ vec
    n = centered @ normal
    order = np.argsort(s)
    s_sorted = s[order]
    span = float(s_sorted[-1] - s_sorted[0])
    if span < 12:
        return None

    centers = []
    edges = np.linspace(s_sorted[0], s_sorted[-1], bins + 1)
    for i in range(bins):
        take = (s >= edges[i]) & (s <= edges[i + 1])
        if take.sum() < 2:
            centers.append((float("nan"), float("nan")))
            continue
        ss = float(np.average(s[take], weights=weights[take]))
        nn = float(np.average(n[take], weights=weights[take]))
        xy = center + ss * vec + nn * normal
        centers.append((float(xy[0]), float(xy[1])))

    valid = np.array([(x, y) for x, y in centers if not math.isnan(x)], dtype=np.float64)
    bend = float("nan")
    chord_angle = float("nan")
    lateral_px = float("nan")
    if len(valid) >= 3:
        chord = valid[-1] - valid[0]
        chord_len = np.linalg.norm(chord)
        if chord_len > 1:
            chord_angle = wrap_deg(math.degrees(math.atan2(chord[1], chord[0])))
            lateral = np.abs(
                chord[0] * (valid[:, 1] - valid[0, 1])
                - chord[1] * (valid[:, 0] - valid[0, 0])
            ) / chord_len
            lateral_px = float(np.max(lateral))
        if len(valid) >= 4:
            a = valid[min(2, len(valid) - 1)] - valid[0]
            b = valid[-1] - valid[max(0, len(valid) - 3)]
            if np.linalg.norm(a) > 1 and np.linalg.norm(b) > 1:
                bend = wrap_deg(math.degrees(math.atan2(b[1], b[0])) - math.degrees(math.atan2(a[1], a[0])))

    return {
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "axis_angle_deg": float(angle),
        "chord_angle_deg": float(chord_angle),
        "body_bend_angle_deg": float(bend),
        "lateral_error_px": lateral_px,
        "length_px": span,
        "centerline": centers,
    }


def draw_debug(frame, white, black_blobs, red_blobs, candidate, metrics, path):
    out = frame.copy()
    if candidate is not None:
        cv2.drawContours(out, [candidate["contour"]], -1, (0, 255, 0), 3)
        x, y, w, h = candidate["bbox"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    selected_black, selected_red = selected_feature_blobs(candidate, black_blobs, red_blobs) if candidate is not None else ([], [])
    for b in selected_black:
        x, y, w, h = b["bbox"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 190, 255), 1)
    for r in selected_red:
        cv2.circle(out, (int(r["center"][0]), int(r["center"][1])), 5, (0, 0, 255), -1)
    if metrics is not None:
        if "body_mask" in metrics:
            overlay = out.copy()
            overlay[metrics["body_mask"] > 0] = (0, 120, 255)
            out = cv2.addWeighted(overlay, 0.28, out, 0.72, 0)
        pts = [(int(x), int(y)) for x, y in metrics["centerline"] if not math.isnan(x)]
        for bbox in metrics.get("servo_bboxes", []) or []:
            if bbox:
                x, y, w, h = bbox
                cv2.rectangle(out, (x, y), (x + w, y + h), (255, 255, 0), 2)
        for idx, p in enumerate(pts, start=1):
            cv2.circle(out, p, 7, (255, 0, 0), -1)
            cv2.putText(out, str(idx), (p[0] + 7, p[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        for a, b in zip(pts, pts[1:]):
            cv2.line(out, a, b, (0, 0, 255), 3)
        label = (
            f"bend={metrics['body_bend_angle_deg']:+.1f}deg "
            f"axis={metrics['axis_angle_deg']:+.1f}deg "
            f"red={candidate['red_count']} black={candidate['black_count']}"
        )
    else:
        label = "not detected"
    cv2.putText(out, label, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 3, cv2.LINE_AA)
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
    debug_dir = out_dir / "composite_debug_frames"

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"failed to open video: {video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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

        white, black_mask, red_mask = detect_feature_masks(frame)
        black_blobs = blobs_from_mask(black_mask, min_area=18, max_area=3800)
        red_blobs = blobs_from_mask(red_mask, min_area=5, max_area=900)
        candidates = candidate_feature_clusters(white, black_blobs, red_blobs, prev_center)
        if not candidates:
            candidates = candidate_white_components(white, black_blobs, red_blobs, prev_center)
        candidate = candidates[0] if candidates else None
        metrics = None
        row = {
            "frame": frame_idx,
            "time_s": frame_idx / fps,
            "detected": 0,
            "white_candidate_count": len(candidates),
            "black_blob_count": len(black_blobs),
            "red_blob_count": len(red_blobs),
        }
        if candidate is not None:
            metrics = centerline_from_servo_features(candidate, black_blobs, red_blobs, prev_angle)
            if metrics is None:
                metrics = centerline_from_roi_mask(white, black_blobs, red_blobs, candidate, prev_angle)
            if metrics is None:
                points = collect_robot_points(candidate, black_blobs, red_blobs)
                metrics = fit_centerline(points, prev_angle)
            if metrics is not None:
                confidence = min(1.0, candidate["red_count"] / 2.0) * min(1.0, candidate["black_count"] / 3.0)
                row.update(candidate)
                row.pop("contour", None)
                row.update(metrics)
                row.pop("centerline", None)
                row.pop("body_mask", None)
                row["confidence"] = confidence
                row["detected"] = 1
                for i, (x, y) in enumerate(metrics["centerline"], start=1):
                    row[f"line{i}_x"] = x
                    row[f"line{i}_y"] = y
                prev_center = candidate["center"]
                prev_angle = metrics["axis_angle_deg"]
        if frame_idx % args.debug_every == 0:
            draw_debug(frame, white, black_blobs, red_blobs, candidate, metrics, debug_dir / f"frame_{frame_idx:04d}.jpg")
        rows.append(row)
        frame_idx += 1
    cap.release()

    csv_path = out_dir / f"{video.stem}_composite_metrics.csv"
    preferred = [
        "frame",
        "time_s",
        "detected",
        "confidence",
        "axis_angle_deg",
        "chord_angle_deg",
        "body_bend_angle_deg",
        "lateral_error_px",
        "length_px",
        "center_x",
        "center_y",
        "red_count",
        "black_count",
        "area",
        "score",
        "white_candidate_count",
        "black_blob_count",
        "red_blob_count",
    ]
    fields = preferred + sorted({k for row in rows for k in row.keys()} - set(preferred))
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
        bends = np.array(
            [r["body_bend_angle_deg"] for r in detected if not math.isnan(r.get("body_bend_angle_deg", float("nan")))],
            dtype=float,
        )
        axes = np.array([r["axis_angle_deg"] for r in detected], dtype=float)
        summary.update(
            {
                "initial_axis_angle_deg": float(axes[0]),
                "median_axis_angle_deg": float(np.median(axes)),
                "axis_range_deg": float(np.max(axes) - np.min(axes)),
            }
        )
        if len(bends):
            summary.update(
                {
                    "initial_body_bend_angle_deg": float(bends[0]),
                    "mean_abs_body_bend_angle_deg": float(np.mean(np.abs(bends))),
                    "median_abs_body_bend_angle_deg": float(np.median(np.abs(bends))),
                    "max_abs_body_bend_angle_deg": float(np.max(np.abs(bends))),
                }
            )
    summary_path = out_dir / f"{video.stem}_composite_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
