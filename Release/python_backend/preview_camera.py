import argparse
import time

import cv2

from opengopro_client import OpenGoProHttpClient

DEFAULT_URL = "udp://0.0.0.0:8554"
DEFAULT_GOPRO_BASE_URL = "http://10.5.5.9:8080"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--gopro-base-url", default=DEFAULT_GOPRO_BASE_URL)
    parser.add_argument("--gopro-preview-port", type=int, default=8554)
    parser.add_argument("--no-gopro-start", action="store_true")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--rotate", choices=["none", "cw", "ccw"], default="none")
    args = parser.parse_args()

    gopro = None
    if not args.no_gopro_start:
        gopro = OpenGoProHttpClient(base_url=args.gopro_base_url)
        print(f"Starting GoPro preview stream on port {args.gopro_preview_port}.")
        gopro.start_preview_stream(args.gopro_preview_port)
        time.sleep(0.4)

    cap = cv2.VideoCapture(args.url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"Camera open failed: {args.url}")
        if gopro is not None:
            gopro.stop_preview_stream()
        return

    try:
        cv2.namedWindow("Camera Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera Preview", 960, 540)

        last = time.time()
        frames = 0
        fps = 0.0

        print("Camera preview started.")
        print("Press q or Esc to close.")

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Frame read failed.")
                break

            frame = cv2.resize(
                frame,
                (args.width, args.height),
                interpolation=cv2.INTER_LANCZOS4,
            )

            if args.rotate == "cw":
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif args.rotate == "ccw":
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            frames += 1
            now = time.time()
            if now - last >= 1:
                fps = frames / (now - last)
                frames = 0
                last = now

            cv2.putText(
                frame,
                f"GOPRO PREVIEW  FPS {fps:.1f}",
                (24, 44),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (120, 220, 255),
                2,
            )

            cv2.imshow("Camera Preview", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        if gopro is not None:
            try:
                gopro.stop_preview_stream()
            except Exception as exc:
                print(f"GoPro preview stop warning: {exc}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
