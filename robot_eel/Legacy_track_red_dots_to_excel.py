import cv2
import numpy as np
from pathlib import Path
from scipy.optimize import linear_sum_assignment
import subprocess
import os


class FixedSixTracker:
    def __init__(self, search_radius=100, max_speed=50):
        self.locked_points = []
        self.is_locked = False
        self.search_radius = search_radius
        self.init_counter = 0
        self.velocity = np.zeros((6, 2), dtype=np.float32)
        self.max_speed = max_speed

    def reset(self):
        self.locked_points = []
        self.is_locked = False
        self.init_counter = 0
        self.velocity = np.zeros((6, 2), dtype=np.float32)

    def find_candidates(self, frame):
        h, w = frame.shape[:2]

        mask_roi = np.zeros((h, w), dtype=np.uint8)
        pool_poly = np.array([
            [int(w * 0.1), int(h * 0.05)],
            [int(w * 0.9), int(h * 0.05)],
            [int(w * 0.9), int(h * 0.95)],
            [int(w * 0.1), int(h * 0.95)]
        ], np.int32)
        cv2.fillPoly(mask_roi, [pool_poly], 255)

        masked_frame = cv2.bitwise_and(frame, frame, mask=mask_roi)
        hsv = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)

        lower_a1 = np.array([0, 80, 80])
        upper_a1 = np.array([10, 255, 255])
        lower_a2 = np.array([140, 80, 80])
        upper_a2 = np.array([180, 255, 255])

        mask_std = cv2.bitwise_or(
            cv2.inRange(hsv, lower_a1, upper_a1),
            cv2.inRange(hsv, lower_a2, upper_a2)
        )

        lower_b1 = np.array([0, 30, 200])
        upper_b1 = np.array([20, 255, 255])
        lower_b2 = np.array([130, 30, 200])
        upper_b2 = np.array([180, 255, 255])

        mask_glare = cv2.bitwise_or(
            cv2.inRange(hsv, lower_b1, upper_b1),
            cv2.inRange(hsv, lower_b2, upper_b2)
        )

        mask = cv2.bitwise_or(mask_std, mask_glare)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        pts = []
        for c in cnts:
            area = cv2.contourArea(c)
            if 10 < area < 1200:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    pts.append((
                        int(M["m10"] / M["m00"]),
                        int(M["m01"] / M["m00"]),
                        area
                    ))

        return pts, mask

    def update(self, frame):
        candidates, mask = self.find_candidates(frame)

        if not self.is_locked:
            if len(candidates) == 6:
                self.init_counter += 1
            else:
                self.init_counter = 0

            if self.init_counter >= 5:
                candidates.sort(key=lambda p: p[1])
                self.locked_points = np.array(
                    [(p[0], p[1]) for p in candidates],
                    dtype=np.float32
                )
                self.velocity = np.zeros((6, 2), dtype=np.float32)
                self.is_locked = True
                print("🔒 [成功] 已鎖定目標點位。")

            return self.locked_points, candidates, mask

        predicted_points = self.locked_points + self.velocity

        if not candidates:
            self.velocity *= 0.0
            return self.locked_points, [], mask

        num_c = len(candidates)
        cost_matrix = np.zeros((6, num_c))

        for i in range(6):
            for j in range(num_c):
                cand_pt = np.array([candidates[j][0], candidates[j][1]])
                dist_pred = np.linalg.norm(predicted_points[i] - cand_pt)
                dist_curr = np.linalg.norm(self.locked_points[i] - cand_pt)
                cost_matrix[i, j] = 0.7 * dist_pred + 0.3 * dist_curr

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        new_points = self.locked_points.copy()

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < self.search_radius:
                detected_pt = np.array(
                    [candidates[c][0], candidates[c][1]],
                    dtype=np.float32
                )
                new_points[r] = 0.8 * detected_pt + 0.2 * predicted_points[r]

        actual_velocity = new_points - self.locked_points

        for i in range(6):
            speed = np.linalg.norm(actual_velocity[i])
            if speed > self.max_speed:
                actual_velocity[i] = (actual_velocity[i] / speed) * self.max_speed
                new_points[i] = self.locked_points[i] + actual_velocity[i]

        self.velocity = 0.6 * self.velocity + 0.4 * actual_velocity
        self.locked_points = new_points.copy()

        return self.locked_points, candidates, mask


def compress_to_h264(temp_video_path, final_video_path):
    ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe"

    cmd = [
        ffmpeg_path,
        "-y",
        "-i", str(temp_video_path),
        "-vcodec", "libx264",
        "-crf", "28",
        "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(final_video_path)
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        if result.returncode == 0:
            try:
                if temp_video_path.exists():
                    os.remove(temp_video_path)
                    print(f"🗑️ 已刪除暫存檔：{temp_video_path}")
            except Exception as e:
                print(f"⚠️ 刪除暫存檔失敗：{e}")

            print(f"✅ H.264 壓縮完成：{final_video_path}")
            return True

        print(f"❌ ffmpeg 壓縮失敗，保留暫存檔：{temp_video_path}")
        return False

    except FileNotFoundError:
        print("❌ 找不到 ffmpeg.exe")
        print(f"📁 保留暫存檔：{temp_video_path}")
        return False


def main():
    input_folder = Path("recording_NEW")
    output_video_folder = Path("red_point_NEW")
    output_video_folder.mkdir(exist_ok=True)

    video_files = sorted(list(input_folder.glob("*.mp4")))

    if not video_files:
        print(f"❌ 在 '{input_folder.absolute()}' 找不到任何 .mp4 檔案！")
        return

    tracker = FixedSixTracker(search_radius=100, max_speed=40)

    cv2.namedWindow("Combined_Tracking_Live", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Combined_Tracking_Live", 1080, 960)

    for idx, video_path in enumerate(video_files):
        print(f"🎬 正在處理第 ({idx + 1}/{len(video_files)}) 部影片: {video_path.name}")

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"❌ 無法開啟影片：{video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_width = width * 2
        output_height = height

        final_video_path = output_video_folder / f"{video_path.stem}_combined.mp4"
        temp_video_path = output_video_folder / f"{video_path.stem}_combined_temp.mp4"

        writer = cv2.VideoWriter(
            str(temp_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (output_width, output_height)
        )

        if not writer.isOpened():
            print(f"❌ VideoWriter 開啟失敗：{temp_video_path}")
            cap.release()
            continue

        tracker.reset()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display = frame.copy()
            tracked_pts, raw_pts, debug_mask = tracker.update(display)

            red_color = (0, 0, 255)
            line_color = (0, 255, 0)

            if not tracker.is_locked:
                for pt in raw_pts:
                    cv2.circle(display, (pt[0], pt[1]), 10, (0, 255, 255), 1)
            else:
                for i in range(5):
                    pt1 = (int(tracked_pts[i][0]), int(tracked_pts[i][1]))
                    pt2 = (int(tracked_pts[i + 1][0]), int(tracked_pts[i + 1][1]))
                    cv2.line(display, pt1, pt2, line_color, 2)

                for i, pt in enumerate(tracked_pts):
                    cp = (int(pt[0]), int(pt[1]))
                    cv2.circle(display, cp, 12, red_color, 2)
                    cv2.putText(
                        display,
                        str(i + 1),
                        (cp[0] + 12, cp[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        red_color,
                        2
                    )

            cv2.putText(
                display,
                f"Video: {idx + 1}/{len(video_files)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            mask_bgr = cv2.cvtColor(debug_mask, cv2.COLOR_GRAY2BGR)
            combined_frame = np.hstack((display, mask_bgr))

            writer.write(combined_frame)

            cv2.imshow(
                "Combined_Tracking_Live",
                cv2.resize(combined_frame, (1080, 600))
            )

            if cv2.waitKey(30) & 0xFF == ord("q"):
                print("⏭️ 使用者手動跳過當前影片。")
                break

        cap.release()
        writer.release()

        print("🎞️ 開始使用 ffmpeg 壓縮成 H.264...")
        compress_to_h264(temp_video_path, final_video_path)

    cv2.destroyAllWindows()
    print("✅ 所有影片皆已抗干擾處理並輸出完畢！")


if __name__ == "__main__":
    main()