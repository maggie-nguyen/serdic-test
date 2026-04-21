# Project Backlog & Decision Log

### [2026-04-21] Decision: Optional Glove-Mask Specialist Model
- **Action**: Added `--glove-mask` path argument in `main.py` and integrated optional glove/mask model inference into `src/detector.py` alongside the main PPE model.
- **Decision**: Run the specialist model on each detected human crop and fuse detections with the existing PPE pipeline.
- **Rationale**: Enables support for classes (`face`, `glove`, `hand`, `mask`) while preserving the primary PPE model behavior. Implemented rules: `hand` without `glove` => `No Gloves`, and `face` without `mask` => `No Mask`.

### [2026-04-21] Decision: Crop-Based PPE Inference & Confidence Display
- **Action**: Refactored `src/detector.py` to perform PPE inference on human crops and updated `src/visualizer.py` to show "Human [conf]" prefix.
- **Decision**: Use human detection crops as input for Stage 2 (PPE detection) instead of full-frame association.
- **Rationale**: Improves PPE association accuracy and potentially detection performance for small items. Added human detection confidence to worker labels as requested by the user.

### [2026-04-20] Decision: Fix Qt/Wayland Display Warnings
- **Action**: Forced `QT_QPA_PLATFORM=xcb` and suppressed `QT_LOGGING_RULES` noise in `main.py`.
- **Decision**: Use XWayland (xcb) as a fallback for OpenCV's Qt backend on Linux systems.
- **Rationale**: Prevented "Could not find the Qt platform plugin 'wayland'" and "Cannot find font directory" warnings from cluttering the terminal output when `--no-show` is not used.

### [2026-04-20] Decision: Revert to Native ROCm & Python 3.12
- **Action**: Removed OpenVINO integration and restored native PyTorch ROCm/CUDA backend.
- **Decision**: Use `conda activate pip12` (Python 3.12) to ensure compatibility with ROCm 6.2.
- **Rationale**: User preference for native PyTorch backend over OpenVINO. Integrated `HSA_OVERRIDE_GFX_VERSION=11.0.1` for Radeon 780M support.

### [2026-04-20] Decision: OpenVINO for AMD GPU Support (REVERTED)
- **Action**: Installed `openvino` and updated `src/detector.py` to support OpenVINO backend.
- **Rationale**: Initial attempt to support AMD Radeon 780M on Python 3.13 where native ROCm was unavailable. This was later replaced by the Python 3.12 ROCm path.

### [2026-04-20] Decision: Select ppe_model3.pt (Hexmon)
- **Action**: Set `ppe_model3.pt` (YOLOv8m) as the default PPE model.
- **Rationale**: Most robust model among candidates that explicitly labels "Gloves" and "No-Gloves," meeting the core project requirements.

### [2026-04-20] Decision: Two-Stage Detection Architecture
- **Action**: Implemented human detection followed by PPE detection with IoA association.
- **Rationale**: Allows for precise worker-level compliance reporting rather than just frame-level detection.

---

## Task Status

### Pending Deliverables
- [ ] **Final Video Generation**: Run `python main.py --all --save --no-show` (recommended with `--scale 0.5 --skip 2` for speed).
- [ ] **PDF Report**: Convert `plans/analysis-report-draft.md` into the final 2-3 page PDF submission.

### Completed Tasks
- [x] **Requirement Analysis**: Identified core PPE items (Helmet, Vest, Gloves, Mask) from `problem-statement.txt`.
- [x] **Baseline Setup**: Installed dependencies and downloaded multiple PPE model candidates via `setup_models.py`.
- [x] **Model Selection**: Selected `ppe_model3.pt` (Hexmon/YOLOv8m) as the default.
- [x] **Pipeline Optimization**: Added `--scale`, `--skip`, and hardware auto-detection (ROCm/CUDA/MPS).
- [x] **Visualizer Enhancement**: Updated `LABEL_MAP` and `COLORS` for all classes (Goggles, Fall Detection).
- [x] **Documentation**: Updated `README.md` and created `AGENTS.md`.
- [x] **Reporting**: Drafted the initial analysis report.
