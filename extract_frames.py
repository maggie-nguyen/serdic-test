import cv2
from pathlib import Path
from ultralytics import YOLO

HUMAN_MODEL   = "models/20260324_human.pt"
VIDEO_DIR     = Path("videos")
OUTPUT_DIR    = Path("frames")
INTERVAL_SEC  = 2.0   
PERSON_CONF   = 0.30  

def main():
    model = YOLO(HUMAN_MODEL)
    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    print(f"Found {len(videos)} videos\n")

    total_saved = 0

    for video_path in videos:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval_frames = max(1, int(fps * INTERVAL_SEC))

        out_dir = OUTPUT_DIR / video_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        skipped = 0
        frame_idx = 0

        print(f"Processing: {video_path.name}  ({total_frames} frames)")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % interval_frames == 0:
                res = model(frame, conf=PERSON_CONF, verbose=False)[0]
                if len(res.boxes) > 0:
                    out_path = out_dir / f"{frame_idx:06d}.jpg"
                    cv2.imwrite(str(out_path), frame)
                    saved += 1
                else:
                    skipped += 1

            frame_idx += 1

        cap.release()
        total_saved += saved
        print(f"  Saved {saved} active frames, dropped {skipped} empty frames")

    print(f"\nExtraction complete. Total active frames: {total_saved}")

if __name__ == "__main__":
    main()
