from __future__ import annotations

import os
import torch
import numpy as np
from ultralytics import YOLO


class PPEDetector:
    def __init__(self, human_model_path: str, ppe_model_path: str, conf: float = 0.25) -> None:
        # Auto-detect device (CUDA, ROCm, MPS, or CPU)
        if torch.cuda.is_available():
            self.device = "cuda"
            # Check for ROCm (AMD) specifically for logging
            is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
            if is_rocm:
                # AMD Radeon 780M (gfx1101) needs an override for ROCm
                os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.1"
                dev_info = "ROCm/AMD"
            else:
                dev_info = "CUDA/NVIDIA"
            print(f"  Using device        : {self.device} ({dev_info})")
        elif torch.backends.mps.is_available():
            self.device = "mps"
            print(f"  Using device        : {self.device} (Apple Silicon)")
        else:
            self.device = "cpu"
            print(f"  Using device        : {self.device} (Falling back to CPU)")

        print(f"  Loading human model : {human_model_path}")
        self.human_model = YOLO(human_model_path).to(self.device)

        print(f"  Loading PPE model   : {ppe_model_path}")
        self.ppe_model = YOLO(ppe_model_path).to(self.device)
        self.ppe_model.overrides["conf"] = conf
        self.ppe_model.overrides["iou"] = 0.45
        self.ppe_model.overrides["max_det"] = 1000

        self.class_names = self.ppe_model.names
        self.violation_classes = {
            cid for cid, name in self.class_names.items()
            if name.lower().startswith("no-") or name.lower().startswith("no_")
        }
        self.ignore_classes = {
            cid for cid, name in self.class_names.items()
            if name.lower() == "person"
        }
        self.conf = conf

    def detect(self, frame: np.ndarray) -> tuple[list[dict], list[dict]]:
        # Stage 1: human detection
        human_res   = self.human_model(frame, conf=self.conf, verbose=False)[0]
        human_boxes = list(human_res.boxes.xyxy.cpu().numpy()) if len(human_res.boxes) > 0 else []

        # Stage 2: PPE detection on full frame
        ppe_res = self.ppe_model(frame, conf=self.conf, verbose=False)[0]
        ppe_detections: list[dict] = []

        if len(ppe_res.boxes) > 0:
            boxes  = ppe_res.boxes.xyxy.cpu().numpy()
            scores = ppe_res.boxes.conf.cpu().numpy()
            cids   = ppe_res.boxes.cls.cpu().numpy().astype(int)
            for i in range(len(boxes)):
                cid = int(cids[i])
                if cid in self.ignore_classes:
                    continue
                ppe_detections.append({
                    "box":          boxes[i],
                    "conf":         float(scores[i]),
                    "class_name":   self.class_names[cid],
                    "is_violation": cid in self.violation_classes,
                })

        persons = self._associate(human_boxes, ppe_detections)
        return persons, ppe_detections

    def _associate(self, human_boxes, ppe_detections):
        persons = []
        for box in human_boxes:
            person = {"box": box, "ppe": [], "violations": [], "compliant": True}
            for ppe in ppe_detections:
                if self._ioa(box, ppe["box"]) >= 0.15:
                    person["ppe"].append(ppe)
                    if ppe["is_violation"]:
                        person["violations"].append(ppe["class_name"])
                        person["compliant"] = False
            persons.append(person)
        return persons

    @staticmethod
    def _ioa(person_box, ppe_box) -> float:
        px1, py1, px2, py2 = person_box
        ex1, ey1, ex2, ey2 = ppe_box
        ix1, iy1 = max(px1, ex1), max(py1, ey1)
        ix2, iy2 = min(px2, ex2), min(py2, ey2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        ppe_area = max((ex2 - ex1) * (ey2 - ey1), 1e-6)
        return float(inter / ppe_area)
