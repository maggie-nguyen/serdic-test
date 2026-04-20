# Agent Workflow Guide (AGENTS.md)

This document serves as the operational manual for AI agents assisting with the Industrial Safety PPE Detection project.

## 1. Contextual Architecture
The system is a **Two-Stage Pipeline**:
1.  **Stage 1 (src/detector.py)**: Human detection using `models/20260324_human.pt`.
2.  **Stage 2 (src/detector.py)**: PPE detection using `models/ppe_model3.pt`.
3.  **Association**: Spatial overlap (IoA) logic in `PPEDetector._associate`.
4.  **Visualization (src/visualizer.py)**: HUD and bounding box overlays.

## 2. Implementation Standards
- **Hardware Agnostic**: Always use `torch.cuda.is_available()` or `torch.backends.mps.is_available()` for device placement.
- **AMD Support**: For Radeon 780M, use `HSA_OVERRIDE_GFX_VERSION=11.0.1` and ensure Python 3.12 is used for ROCm compatibility.
- **Performance First**: When processing batches, use `--scale` and `--skip` to manage compute load.
- **Consistency**: Labels in `src/visualizer.py` MUST match the keys in the YOLO model's `names` dictionary.

## 3. Decision Logging (`BACKLOG.md`)

Agents MUST log every significant decision and strategic pivot in `BACKLOG.md`.

### When to Log
- **Architectural Changes**: Switching between detection strategies or backends.
- **Hardware/Environment Shifts**: Changing Python versions, driver overrides, or inference backends (e.g., ROCm vs. OpenVINO).
- **Model Selection**: Choosing or updating the primary detection weights.
- **Workflow Updates**: Adding new CLI arguments or critical source code refactors.

### How to Log
- **Order**: Latest to Previous. New entries MUST be prepended to the top of the "Decision Log" section.
- **Timeline**: The timeline flows from bottom (oldest) to top (latest).
- **Immutability**: NEVER delete or modify previous log entries. Only add on top.
- **Structure**: Each entry should include:
    - **Date**: [YYYY-MM-DD]
    - **Action**: What was done.
    - **Decision/Rationale**: Why it was done and the logic behind it.

## 4. Common Workflows

### Running Inferences for Submission
1. Ensure `outputs/` is empty or clear.
2. Run: `python main.py --all --save --no-show --scale 0.5 --skip 2`.
3. Verify output integrity in `outputs/*.mp4`.

## 4. Safety & Compliance
- Do not modify `models/*.pt` files directly.
- The `videos/` directory contains sensitive industrial footage; do not attempt to upload or stream this data outside the local environment.
