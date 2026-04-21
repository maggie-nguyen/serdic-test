# Industrial Safety PPE Detection — AI Proof of Concept
Serdic AI Objective: Real-time PPE (Personal Protective Equipment) detection on industrial CCTV footage using a two-stage YOLO architecture.

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirement.txt
```
*(Requires `ultralytics`, `opencv-python`, `torch`)*

### 3. Setup Models

### 3. Download the PPE detection models (run once)

```bash
python setup_models.py
```

This downloads specialized YOLOv8 models from HuggingFace and saves them to the `models/` directory. By default, the system uses `models/ppe_model3.pt` (Hexmon/vyra-yolo-ppe-detection) which supports all required items including **Gloves**.

> The provided human detection model (`models/20260324_human.pt`) must already be present.

---

## 🏃 Running Inference

### Basic Run (Live Window)
```bash
python main.py --video videos/GUNSAN_cam14_20251222_183405.mp4
```

### Process & Save Demo Video (Headless)
Generates an annotated `.mp4` without opening a GUI window.
```bash
python main.py --video videos/GUNSAN_cam01_20251222_140441.mp4 --save --no-show
```
*Saves to `outputs/GUNSAN_cam01_20251222_140441_ppe.mp4`*

### Batch Process All Videos
Processes all `.mp4` files in the `videos/` directory automatically.
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
  --ppe-model PATH      Path to PPE detection model   (default: models/ppe_model3.pt)
  --conf FLOAT          Confidence threshold (default: 0.25)
  --save                Save annotated output to outputs/
  --no-show             Do not display the live window
  --scale FLOAT         Scale factor for resolution (e.g. 0.5 for half size)
  --skip INT            Process every N-th frame (e.g. 2 to skip half the frames)
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
│   ├── ppe_model1.pt              # PPE Model (Hansung - Helmet/Vest/Mask)
│   ├── ppe_model2.pt              # PPE Model (Tanishjain - + Gloves, YOLOv8n)
│   └── ppe_model3.pt              # PPE Model (Hexmon - + Gloves, YOLOv8m)
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
[Stage 2]  PPE Model (Hexmon/vyra-yolo-ppe-detection)
           → Detects on full frame: Hardhat, Mask, Safety Vest, Gloves (+ violation classes)
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
| 🟢 Green item box    | Compliant item (Helmet / Mask / Safety Vest / Gloves) |
| 🔴 Red item box      | Violation (NO-Helmet / NO-Mask / NO-Safety Vest / NO-Gloves) |

---

## Known Limitations

- **Lighting**: Low-light CCTV footage may reduce accuracy.
- **Occlusion**: Heavily overlapping workers may cause missed or merged detections.
- **Temporal Stability**: Processing is per-frame; results may flicker occasionally. Tracking could be added for more stability.
