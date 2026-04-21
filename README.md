# Industrial Safety PPE Detection — AI Proof of Concept
Serdic AI Objective: Real-time PPE (Personal Protective Equipment) analysis on industrial CCTV footage using a **Decoupled Multi-stage YOLO Architecture**.

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

Download the required PPE baseline detection models (Run once):

```bash
python setup_models.py
```

This will automatically securely download `models/ppe_model1.pt` (Hansung YOLOv8) for tracking large PPE (Helmets/Vests).

> **Note:** The `models/20260324_human.pt` (Human locater) and `models/glove-mask.pt` (Specialist YOLO26n Mask/Glove crop model) must already be strictly present in the `models/` directory prior to running.

---

## 🏃 Running Inference

### Basic Run (Live Window)
```bash
python main.py --video videos/GUNSAN_cam14_20251222_183405.mp4 --conf 0.50 --scale 0.5
```

### Fast Proxy Evaluation (Jump through frames)
Test the pipeline quickly without evaluating every single 30fps frame.
```bash
python main.py --video videos/GUNSAN_cam14_20251222_183405.mp4 --conf 0.50 --skip 3 --scale 0.5
```

### Process & Save Demo Video (Headless)
Generates an annotated `.mp4` into the outputs folder.
```bash
python main.py --video videos/GUNSAN_cam14_20251222_183405.mp4 --conf 0.50 --save --no-show --scale 1.0
```
*Saves to `outputs/` directory.*

### Available CLI Options

```text
python main.py --help

options:
  --video VIDEO         Input video path (default: videos/GUNSAN_cam14_20251222_183405.mp4)
  --all                 Process all sample videos
  --human-model PATH    Path to human detection model (default: models/20260324_human.pt)
  --ppe-model PATH      Path to PPE detection model (default: models/ppe_model1.pt)
  --glove-mask PATH     Path to specialist model (default: models/glove-mask.pt)
  --conf FLOAT          Confidence threshold (default: 0.50 - Optimized for CCTV)
  --start FLOAT         Start processing from this time in seconds
  --save                Save the final annotated output video to outputs/
  --no-show             Disable the live preview window (faster processing)
  --scale FLOAT         Resolution scaling factor (e.g. 0.5 for 50% size)
  --skip INT            Process 1 out of N frames (e.g., --skip 3)
```

---

## Project Structure

```text
serdic-test/
├── main.py                        # Execution entry point
├── setup_models.py                # One-time baseline model download
├── fast_check.py                  # Proxy internal evaluation script for unlabelled CCTV footage
├── check_detections.py            # Detailed bounding box JSON extraction script
├── requirement.txt
├── README.md
├── models/
│   ├── 20260324_human.pt          # Human detection model (Stage 1 - YOLO11m)
│   ├── ppe_model1.pt              # Baseline PPE Model (Hansung Helmets/Vests - Stage 2 - YOLOv8)
│   └── glove-mask.pt              # Targeted Specialist Model (Gloves/Mask - Stage 3 - YOLO26n)
├── src/
│   ├── detector.py                # Multi-stage decoupled tracking & inference logic
│   └── visualizer.py              # Frame color-coding & HUD annotations
├── videos/                        # Unlabelled CCTV sample domain videos
└── outputs/                       # Saved annotated structural recordings
```

---

## Detection Pipeline (Decoupled Crop Architecture)

To systematically overcome extreme pixel loss limits on CCTV, this architecture leverages dynamic bounding-box logic to decouple micro and macro objects:

```text
Full Video Frame
    │
    ▼
[Stage 1]  Human Locator Model (YOLO11m)
           → Generates distinct Target Bounding Boxes around each worker
    │
    ▼
[Cropping] The localized worker region is cropped exactly from the frame.
           (Mathematically bypasses IoA association logic entirely)
    │
    ├──► [Stage 2] Baseline Model (YOLOv8) evaluates Crop
    │    → Explicitly Detects: Helmet, Vest
    │
    └──► [Stage 3] Specialist Model (YOLO26n) evaluates Crop simultaneously
         → Explicitly Detects: Face, Mask, Hand, Glove
    │
    ▼
[Logic Gating] Hardcoded heuristic rules (e.g., detecting 'Hand' without 'Glove' explicitly outputs 'no_glove')
    │
    ▼
Annotated Full Frame → Real-time GUI / Saved .mp4
```

### Visual Indicators

| Colour | Meaning |
|--------|---------|
| 🟢 Green person box  | Worker perfectly compliant (No violations) |
| 🔴 Red person box    | Strict PPE Violation registered on worker |
| ⬜ Grey person box   | Worker identified, but no PPE features visibly resolved |
| 🟢 Green item box    | Compliant object localized (Helmet / Safety Vest / Glove / Mask) |
| 🔴 Red item box      | System-triggered Violation marker (`no_glove`, `no-mask`) |

---

## Known Limitations & Improvements

- **Extreme Domain Resolution Limits:** Masks and Gloves often occupy mathematically deficient pixel-density within full HD wide-angle CCTV footage ($<$ 15 pixels), forcing severely deflated confidence scores.
- **Occlusion Dynamics:** Top-down oblique camera angles heavily obscure hands and facial features.
- **Temporal Stability Gap:** Tracking logic is currently strictly per-frame. Future iterations definitively require **ByteTrack** integration to preserve compliant Worker IDs across historically occluded frames.
