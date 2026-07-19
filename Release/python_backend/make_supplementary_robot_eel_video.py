from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


W, H, FPS = 1920, 1080, 30
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (245, 245, 245)
YELLOW = (70, 220, 255)
CYAN = (255, 220, 80)
GREEN = (120, 240, 160)
ORANGE = (80, 170, 255)
BLACK = (0, 0, 0)


def fit_frame(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(W / w, H / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(
        frame,
        (nw, nh),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    x, y = (W - nw) // 2, (H - nh) // 2
    canvas[y : y + nh, x : x + nw] = resized
    return canvas


def dark_overlay(img: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (W, H), BLACK, -1)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def put_text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float,
    color: tuple[int, int, int] = WHITE,
    thickness: int = 2,
) -> None:
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def put_lines(
    img: np.ndarray,
    lines: list[str],
    x: int,
    y: int,
    scale: float = 0.78,
    line_h: int = 44,
    color: tuple[int, int, int] = WHITE,
) -> None:
    for i, line in enumerate(lines):
        put_text(img, line, (x, y + i * line_h), scale, color, 2)


def first_frame(paths: list[Path]) -> np.ndarray:
    for path in paths:
        cap = cv2.VideoCapture(str(path))
        ok, frame = cap.read()
        cap.release()
        if ok:
            return fit_frame(frame)
    return np.zeros((H, W, 3), dtype=np.uint8)


def card(
    writer: cv2.VideoWriter,
    background: np.ndarray,
    title: str,
    lines: list[str],
    seconds: float = 1.2,
) -> None:
    bg = dark_overlay(background, 0.82)
    for _ in range(int(FPS * seconds)):
        frame = bg.copy()
        put_text(frame, title, (70, 150), 1.65, YELLOW, 3)
        put_lines(frame, lines, 76, 240, scale=0.82, line_h=48)
        writer.write(frame)


def annotate(
    frame: np.ndarray,
    title: str,
    section: str,
    condition: str,
    t: float,
    duration: float,
) -> None:
    cv2.rectangle(frame, (0, 0), (W, 150), BLACK, -1)
    cv2.rectangle(frame, (0, H - 122), (W, H), BLACK, -1)

    put_text(frame, "robot_eel", (36, 42), 0.88, WHITE, 2)
    put_text(frame, title, (36, 92), 1.05, YELLOW, 2)
    put_text(frame, section, (36, 132), 0.68, ORANGE, 2)
    put_text(frame, f"Condition: {condition}", (760, 45), 0.62, WHITE, 2)
    put_text(frame, "Camera: top-down | Playback: 1x real time", (760, 84), 0.62, CYAN, 2)
    put_text(frame, "Pool: 1.5 m x 3.0 m", (760, 123), 0.62, CYAN, 2)
    put_text(frame, "Robot dimensions: length 68.5 cm, width 5.5 cm", (36, H - 72), 0.72, GREEN, 2)
    put_text(frame, f"Time in segment: {t:05.2f} / {duration:05.2f} s", (36, H - 32), 0.68, WHITE, 2)


def write_clip(
    writer: cv2.VideoWriter,
    item: dict,
    section: str,
    max_seconds: float | None = None,
) -> None:
    cap = cv2.VideoCapture(str(item["path"]))
    if not cap.isOpened():
        print(f"SKIP unreadable: {item['path']}")
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_frames = total if max_seconds is None else min(total, int(max_seconds * src_fps))
    shown_duration = max_frames / src_fps if src_fps else 0

    ok, first = cap.read()
    if ok:
        card(
            writer,
            fit_frame(first),
            item["title"],
            [section, f"Condition: {item['condition']}"],
            seconds=0.9,
        )

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    i = 0
    while i < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame = fit_frame(frame)
        annotate(
            frame,
            item["title"],
            section,
            item["condition"],
            i / src_fps if src_fps else 0,
            shown_duration,
        )
        writer.write(frame)
        i += 1
    cap.release()


def make_video(
    output: Path,
    visual_clips: list[dict],
    tracking_clips: list[dict],
    tracking_section: str,
) -> None:
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed: {output}")

    if len(visual_clips) == 1:
        intro_lines = [
            visual_clips[0]["title"],
            tracking_clips[0]["condition"],
            "",
            "Robot dimensions: length 68.5 cm; width 5.5 cm",
            "Swimming pool: 1.5 m x 3.0 m; camera angle: top-down; playback speed: 1x real time",
        ]
        tracking_card_lines = [
            tracking_clips[0]["title"],
            tracking_clips[0]["condition"],
            "Condition: lights-off LED tracking",
        ]
    else:
        intro_lines = [
            "Forward swimming; r03 turning; y07 turning",
            "Physical robot motion and LED tracking measurements",
            "",
            "Robot dimensions: length 68.5 cm; width 5.5 cm",
            "Swimming pool: 1.5 m x 3.0 m; camera angle: top-down; playback speed: 1x real time",
        ]
        tracking_card_lines = [
            tracking_section,
            "Forward speed, r03 turning radius, and y07 yaw rate",
            "Condition: lights-off LED tracking",
        ]

    card(
        writer,
        first_frame([c["path"] for c in visual_clips]),
        "robot_eel",
        intro_lines,
        seconds=4.0,
    )

    for item in visual_clips:
        write_clip(writer, item, "Robot motion", max_seconds=item.get("max_seconds"))

    card(
        writer,
        first_frame([c["path"] for c in tracking_clips]),
        "LED tracking",
        tracking_card_lines,
        seconds=3.0,
    )

    for item in tracking_clips:
        write_clip(writer, item, tracking_section, max_seconds=item["max_seconds"])

    writer.release()
    print(output)


def build_clip_lists(repo_root: Path, backend_recordings: Path, use_mark: bool) -> tuple[list[dict], list[dict]]:
    old_movies = repo_root / "real_movie" / "old"
    analysis = repo_root / "real_movie_analysis" / "2026-07-03_lights_off_reviewer_metrics"

    visual_clips = [
        {
            "path": backend_recordings / "clean_v_20260712_223209_w100_ew08_a20_targetv17_ppo_free_swim_freq_phase_run03.mp4",
            "title": "Forward swimming",
            "condition": "targetv17 forward swimming",
            "max_seconds": 8.0,
        },
        {
            "path": backend_recordings / "clean_v_20260712_223400_paper80_ppo_turn_left_a20_r03_run09.mp4",
            "title": "r03 turning",
            "condition": "r03 left turn, target radius R = 0.3 m",
            "max_seconds": 8.0,
        },
        {
            "path": backend_recordings / "clean_v_20260712_223505_paper80_ppo_turn_left_a20_y07_run02.mp4",
            "title": "y07 turning",
            "condition": "y07 left turn, target yaw rate = 0.7 rad/s",
            "max_seconds": 8.0,
        },
    ]

    if use_mark:
        tracking_clips = [
            {
                "path": analysis / "clean_v_20260702_194718_w100_ew08_a20_targetv17_ppo_free_swim_freq_phase_run03" / "fit_overlay_video.mp4",
                "title": "Forward tracking",
                "condition": "targetv17, measured speed = 0.175 m/s",
                "max_seconds": 6.0,
            },
            {
                "path": analysis / "clean_v_20260703_153405_paper80_ppo_turn_left_a20_r03_run09" / "fit_overlay_video.mp4",
                "title": "r03 tracking",
                "condition": "r03, measured R = 0.341 m, |yaw rate| = 0.436 rad/s",
                "max_seconds": 8.0,
            },
            {
                "path": analysis / "clean_v_20260703_154315_paper80_ppo_turn_left_a20_y07_run02" / "fit_overlay_video.mp4",
                "title": "y07 tracking",
                "condition": "y07, measured |yaw rate| = 0.542 rad/s, fitted R = 0.257 m",
                "max_seconds": 8.0,
            },
        ]
    else:
        tracking_clips = [
            {
                "path": old_movies / "clean_v_20260702_194718_w100_ew08_a20_targetv17_ppo_free_swim_freq_phase_run03.mp4",
                "title": "Forward tracking",
                "condition": "targetv17, measured speed = 0.175 m/s",
                "max_seconds": 6.0,
            },
            {
                "path": old_movies / "clean_v_20260703_153405_paper80_ppo_turn_left_a20_r03_run09.mp4",
                "title": "r03 tracking",
                "condition": "r03, measured R = 0.341 m, |yaw rate| = 0.436 rad/s",
                "max_seconds": 8.0,
            },
            {
                "path": old_movies / "clean_v_20260703_154315_paper80_ppo_turn_left_a20_y07_run02.mp4",
                "title": "y07 tracking",
                "condition": "y07, measured |yaw rate| = 0.542 rad/s, fitted R = 0.257 m",
                "max_seconds": 8.0,
            },
        ]

    return visual_clips, tracking_clips


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["with_mark", "no_mark", "both"], default="both")
    parser.add_argument("--layout", choices=["combined", "split", "both"], default="combined")
    args = parser.parse_args()

    backend_dir = Path(__file__).resolve().parent
    recordings = backend_dir / "recordings"
    repo_root = backend_dir.parents[1]

    variants = ["with_mark", "no_mark"] if args.variant == "both" else [args.variant]
    for variant in variants:
        use_mark = variant == "with_mark"
        visual_clips, tracking_clips = build_clip_lists(repo_root, recordings, use_mark)
        section = "LED tracking with trajectory/fit overlay" if use_mark else "LED tracking"
        if args.layout in ("combined", "both"):
            make_video(
                recordings / f"supplementary_robot_eel_reviewer_video_20260712_{variant}.mp4",
                visual_clips,
                tracking_clips,
                section,
            )
        if args.layout in ("split", "both"):
            labels = ["S1_forward", "S2_r03_turning", "S3_y07_turning"]
            for label, visual_clip, tracking_clip in zip(labels, visual_clips, tracking_clips):
                make_video(
                    recordings / f"supplementary_robot_eel_{label}_{variant}.mp4",
                    [visual_clip],
                    [tracking_clip],
                    section,
                )


if __name__ == "__main__":
    main()
