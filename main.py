from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from src.detector import PPEDetector
from src.visualizer import draw_results

HUMAN_MODEL   = "models/20260324_human.pt"
PPE_MODEL     = "models/ppe_model3.pt"
DEFAULT_VIDEO = "videos/GUNSAN_cam14_20251222_183405.mp4"
OUTPUT_DIR    = Path("outputs")

ALL_VIDEOS = [
    "videos/GUNSAN_cam01_20251222_140441.mp4",
    "videos/GUNSAN_cam09_20251127_195722.mp4",
    "videos/GUNSAN_cam09_20251222_141941.mp4",
    "videos/GUNSAN_cam12_20251222_165429.mp4",
    "videos/GUNSAN_cam13_20251127_200612.mp4",
    "videos/GUNSAN_cam13_20251222_141439.mp4",
    "videos/GUNSAN_cam14_20251222_183405.mp4",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--video", "-v", default=DEFAULT_VIDEO)
    p.add_argument("--all", "-a", action="store_true")
    p.add_argument("--human-model", default=HUMAN_MODEL)
    p.add_argument("--ppe-model", default=PPE_MODEL)
    p.add_argument("--conf", "-c", type=float, default=0.25)
    p.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
    p.add_argument("--save", "-s", action="store_true")
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--scale", type=float, default=1.0, help="Scale factor for resolution (e.g. 0.5)")
    p.add_argument("--skip", type=int, default=1, help="Process every N-th frame")
    return p.parse_args()


def process_video(video_path, detector, *, start=0.0, save=False, show=True, scale=1.0, skip=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open: {video_path}")
        return

    if start > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
        print(f"  Seeking to {start:.1f}s ...")

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    remaining   = max(total - start_frame, 1)

    writer = None
    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / f"{Path(video_path).stem}_ppe.mp4"
        # Output FPS should be adjusted if we are skipping frames but want real-time playback speed, 
        # or kept if we want a "fast-forward" effect. Usually for PPE demo, we want real-time speed.
        out_fps = fps_src / skip
        writer   = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (width, height))
        print(f"  Saving to: {out_path} ({width}x{height} @ {out_fps:.1f}fps)")

    frame_idx = 0
    processed_count = 0
    t0 = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % skip != 0:
            frame_idx += 1
            continue

        if scale != 1.0:
            frame = cv2.resize(frame, (width, height))

        persons, ppe_detections = detector.detect(frame)
        fps_cur   = (processed_count + 1) / max(time.time() - t0, 1e-6)
        annotated = draw_results(frame, persons, ppe_detections, fps=fps_cur)

        if writer:
            writer.write(annotated)
        if show:
            cv2.imshow("PPE Detection  [q=quit]", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        processed_count += 1
        if frame_idx % (50 * skip) == 0 or frame_idx == total:
            print(f"    {frame_idx}/{remaining} ({100*frame_idx//remaining}%)  FPS={fps_cur:.1f}")

    cap.release()
    if writer:
        writer.release()
    print(f"  Done — {processed_count} frames processed in {time.time()-t0:.1f}s")


def main():
    args = parse_args()
    print("\nLoading models ...")
    detector = PPEDetector(args.human_model, args.ppe_model, conf=args.conf)
    print("Ready.\n")

    for vp in (ALL_VIDEOS if args.all else [args.video]):
        print(f"Processing: {vp}")
        process_video(
            vp,
            detector,
            start=args.start,
            save=args.save,
            show=not args.no_show,
            scale=args.scale,
            skip=args.skip,
        )
        print()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
