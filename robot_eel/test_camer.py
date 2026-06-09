import time
from pathlib import Path

import cv2


SCRIPT_DIR = Path(__file__).resolve().parent
SAVE_FOLDER = SCRIPT_DIR / "recordings"
SAVE_FOLDER.mkdir(exist_ok=True)

RTSP_URL = "rtsp://admin:184342@192.168.0.102:554/live/profile.0/video"

TARGET_W = 1920
TARGET_H = 1080
FPS = 20.0

# The displayed frame is rotated 90 degrees clockwise before saving.
RECORD_W = TARGET_H
RECORD_H = TARGET_W


def open_video_writer(timestamp):
    candidates = [
        (SAVE_FOLDER / f"clean_v_{timestamp}.mp4", "mp4v"),
        (SAVE_FOLDER / f"clean_v_{timestamp}.avi", "XVID"),
        (SAVE_FOLDER / f"clean_v_{timestamp}.avi", "MJPG"),
    ]

    for path, codec in candidates:
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*codec),
            FPS,
            (RECORD_W, RECORD_H),
        )
        if writer.isOpened():
            return writer, path, codec
        writer.release()

    return None, None, None


def main():
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"Camera open failed: {RTSP_URL}")
        return

    out = None
    current_filename = None
    is_recording = False

    cv2.namedWindow("CCTV_Clean_Record", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("CCTV_Clean_Record", 540, 960)

    print("--- CCTV recorder ---")
    print(f"Save folder: {SAVE_FOLDER}")
    print("Press 'r' to start/stop recording.")
    print("Press 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Frame read failed.")
                break

            temp_frame = cv2.resize(
                frame,
                (TARGET_W, TARGET_H),
                interpolation=cv2.INTER_LANCZOS4,
            )
            clean_frame = cv2.rotate(temp_frame, cv2.ROTATE_90_CLOCKWISE)

            if is_recording and out is not None:
                out.write(clean_frame)

            display_preview = clean_frame.copy()
            if is_recording:
                cv2.circle(display_preview, (50, 50), 20, (0, 0, 255), -1)
                cv2.putText(
                    display_preview,
                    "REC NOW",
                    (85, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3,
                )

            cv2.imshow("CCTV_Clean_Record", display_preview)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("r"):
                if not is_recording:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    out, current_filename, codec = open_video_writer(timestamp)

                    if out is None:
                        print(f"VideoWriter open failed. Save folder: {SAVE_FOLDER}")
                        continue

                    is_recording = True
                    print(f"Recording started ({codec}): {current_filename}")
                else:
                    is_recording = False
                    if out is not None:
                        out.release()
                    print(f"Recording saved: {current_filename}")
                    out = None
                    current_filename = None

            elif key == ord("q"):
                break
    finally:
        cap.release()
        if out is not None:
            out.release()
            print(f"Recording saved: {current_filename}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
