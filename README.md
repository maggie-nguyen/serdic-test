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

Downloads the required PPE baseline model (Hansung-Cho YOLOv8) from HuggingFace.

```bash
python setup_models.py
```
> Note: The provided human detection model (`models/20260324_human.pt`) must exist in the `models/` directory prior to running inference.

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

---

## 🏗 System Architecture

The PoC implements a **Two-Stage Detection Pipeline** to maximize efficiency and localized association:

1. **Stage 1 (Localization):** Provided Human Detection Model (`20260324_human.pt`) scans the frame to extract worker bounding boxes.
2. **Stage 2 (Classification):** PPE Model (`ppe_model1.pt`) evaluates the full frame for Safety items (Helmet, Vest) and Violation instances (No Helmet, No Vest).
3. **Association (IoA):** Custom spatial association logic (Intersection over Area ≥ 0.15) maps detected PPE items to the respective worker bounding box, ensuring real-time individual compliance tracking.

### HUD Reference
- 🟢 **Green Bounding Box:** Worker is wearing required PPE.
- 🔴 **Red Bounding Box:** Worker is violating PPE requirements.

---

## 📊 Evaluation & Utilities

### Proxy Calibration (`check_detections.py`)
To ensure optimal performance without labeled ground truth, a proxy evaluation script is included. It measures the `mean confidence distribution` of the baseline model across all frames, determining that `conf=0.40` is the optimal threshold to filter background noise while retaining high-confidence true positives.

### Data Engineering (`extract_frames.py`)
A utility script used to extract sequential frames from CCTV videos, automatically filtering out frames devoid of workers using the Stage 1 Human model. This readies the dataset for domain-specific fine-tuning (Approach A).

---

## ⚠️ Known Limitations
- **Severe Domain Gap (Masks & Gloves):** The current public baseline model is highly effective for large PPE (Helmets, Vests) but fails to reliably detect high-resolution micro-objects (Masks, Gloves) due to the low effective pixel density inherent in top-down CCTV angles. 
- **Proposed Solution:** Utilize the extracted frames to annotate a custom dataset and apply transfer learning (Fine-tuning YOLOv8) to adapt to the CCTV domain.
