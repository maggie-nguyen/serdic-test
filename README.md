# Industrial Safety PPE Detection — Serdic AI Assignment

Real-time PPE (Personal Protective Equipment) detection on industrial CCTV footage using a two-stage YOLO pipeline.

---

## Requirements

- Python 3.9+
- macOS / Linux / Windows

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 2. Install dependencies

```bash
pip install -r requirement.txt
```

### 3. Download the PPE detection model (run once)

```bash
python setup_models.py
```

This downloads `keremberke/yolov8m-ppev1` from HuggingFace (~50 MB) and saves it to `models/ppe_detection.pt`.

> The provided human detection model (`models/20260324_human.pt`) must already be present.

---

## Running the Demo

### Basic — live window with default video

```bash
python main.py
```

### Custom video

```bash
python main.py --video videos/GUNSAN_cam01_20251222_140441.mp4
```

### Save output video (no display window)

```bash
python main.py --save --no-show
```

Saves to `outputs/<video_name>_ppe.mp4`.

### Process ALL sample videos and save

```bash
python main.py --all --save --no-show
```

### Full options

```
python main.py --help

options:
  --video VIDEO         Input video path (default: videos/GUNSAN_cam14_20251222_183405.mp4)
  --all                 Process all sample videos
  --human-model PATH    Path to human detection model (default: models/20260324_human.pt)
  --ppe-model PATH      Path to PPE detection model   (default: models/ppe_detection.pt)
  --conf FLOAT          Confidence threshold (default: 0.30)
  --save                Save annotated output to outputs/
  --no-show             Do not display the live window
```

---

## Project Structure

```
serdic-test/
├── main.py                        # Entry point
├── setup_models.py                # One-time model download
├── requirement.txt
├── README.md
├── models/
│   ├── 20260324_human.pt          # Provided human detection model (Stage 1)
│   └── ppe_detection.pt           # Downloaded PPE model (Stage 2)
├── src/
│   ├── detector.py                # Two-stage detection pipeline
│   └── visualizer.py              # Frame annotation & HUD
├── videos/                        # Provided sample videos
└── outputs/                       # Saved annotated videos (created on run)
```

---

## Detection Pipeline

```
Video Frame
    │
    ▼
[Stage 1]  Provided Human Model (20260324_human.pt)
           → Bounding boxes around each worker
    │
    ▼
[Stage 2]  PPE Model (keremberke/yolov8m-ppev1)
           → Detects on full frame: Hardhat, Mask, Safety Vest (+ violation classes)
    │
    ▼
[Association]  Each PPE box is matched to the nearest person box (IoA ≥ 0.15)
    │
    ▼
Annotated Frame  →  Live window  / saved .mp4
```

### Visual indicators

| Colour | Meaning |
|--------|---------|
| 🟢 Green person box  | Worker wearing required PPE |
| 🔴 Red person box    | PPE violation detected |
| ⬜ Grey person box   | Worker detected, no PPE in frame |
| 🟢 Green item box    | Compliant item (Hardhat / Mask / Safety Vest) |
| 🔴 Red item box      | Violation (NO-Hardhat / NO-Mask / NO-Safety Vest) |

---

## Known Limitations

- **Gloves**: The PPE model (`keremberke/yolov8m-ppev1`) does not include a Gloves class. Glove detection requires a specialised model or fine-tuning on labeled glove data.
- **Lighting**: Low-light CCTV footage may reduce accuracy.
- **Occlusion**: Heavily overlapping workers may cause missed or merged detections.
