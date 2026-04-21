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
        human_res = self.human_model(frame, conf=self.conf, verbose=False)[0]
        
        persons: list[dict] = []
        all_ppe_detections: list[dict] = []
        
        if len(human_res.boxes) > 0:
            h_boxes  = human_res.boxes.xyxy.cpu().numpy()
            h_scores = human_res.boxes.conf.cpu().numpy()
            
            for i, h_box in enumerate(h_boxes):
                h_conf = float(h_scores[i])
                # Ensure coordinates are within frame boundaries
                x1, y1, x2, y2 = h_box.astype(int)
                fh, fw = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(fw, x2), min(fh, y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Crop for this specific person
                crop = frame[y1:y2, x1:x2]
                
                # Stage 2: PPE detection on the person crop
                ppe_res = self.ppe_model(crop, conf=self.conf, verbose=False)[0]
                
                person_ppe: list[dict] = []
                person_violations: list[str] = []
                person_compliant = True
                
                if len(ppe_res.boxes) > 0:
                    p_boxes  = ppe_res.boxes.xyxy.cpu().numpy()
                    p_scores = ppe_res.boxes.conf.cpu().numpy()
                    p_cids   = ppe_res.boxes.cls.cpu().numpy().astype(int)
                    
                    for j in range(len(p_boxes)):
                        cid = int(p_cids[j])
                        if cid in self.ignore_classes:
                            continue
                            
                        # Translate crop-relative coordinates back to global frame coordinates
                        px1, py1, px2, py2 = p_boxes[j]
                        global_ppe_box = np.array([px1 + x1, py1 + y1, px2 + x1, py2 + y1])
                        
                        ppe_info = {
                            "box":          global_ppe_box,
                            "conf":         float(p_scores[j]),
                            "class_name":   self.class_names[cid],
                            "is_violation": cid in self.violation_classes,
                        }
                        
                        person_ppe.append(ppe_info)
                        all_ppe_detections.append(ppe_info)
                        
                        if ppe_info["is_violation"]:
                            person_violations.append(ppe_info["class_name"])
                            person_compliant = False
                
                persons.append({
                    "box":        h_box,
                    "conf":       h_conf,
                    "ppe":        person_ppe,
                    "violations": person_violations,
                    "compliant":  person_compliant,
                })
        
        return persons, all_ppe_detections
