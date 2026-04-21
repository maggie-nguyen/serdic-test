from __future__ import annotations

import os
import torch
import numpy as np
from ultralytics import YOLO


class PPEDetector:
    def __init__(
        self,
        human_model_path: str,
        ppe_model_path: str,
        conf: float = 0.25,
        glove_mask_model_path: str | None = None,
    ) -> None:
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

        self.glove_mask_model = None
        self.glove_mask_names: dict[int, str] = {}
        if glove_mask_model_path:
            print(f"  Loading glove-mask  : {glove_mask_model_path}")
            self.glove_mask_model = YOLO(glove_mask_model_path).to(self.device)
            self.glove_mask_model.overrides["conf"] = conf
            self.glove_mask_model.overrides["iou"] = 0.45
            self.glove_mask_model.overrides["max_det"] = 1000
            self.glove_mask_names = self.glove_mask_model.names

        self.class_names = self.ppe_model.names
        
        # Strictly filter for only required classes and their violations
        self.allowed_classes = {
            "hardhat", "no-hardhat", "helmet", "no-helmet",
            "safety vest", "no-safety vest", "vest", "no-vest",
            "mask", "no-mask",
            "gloves", "no-gloves", "glove", "no-glove"
        }

        self.violation_classes = {
            cid for cid, name in self.class_names.items()
            if (name.lower().startswith("no-") or name.lower().startswith("no_"))
            and name.lower() in self.allowed_classes
        }
        self.ignore_classes = {
            cid for cid, name in self.class_names.items()
            if name.lower() not in self.allowed_classes
        }
        self.conf = conf

    def _run_glove_mask(self, crop: np.ndarray, x1: int, y1: int) -> tuple[list[dict], bool, bool, bool, bool]:
        if self.glove_mask_model is None:
            return [], False, False, False, False

        gm_res = self.glove_mask_model(crop, conf=self.conf, verbose=False)[0]
        if len(gm_res.boxes) == 0:
            return [], False, False, False, False

        gm_boxes = gm_res.boxes.xyxy.cpu().numpy()
        gm_scores = gm_res.boxes.conf.cpu().numpy()
        gm_cids = gm_res.boxes.cls.cpu().numpy().astype(int)

        detections: list[dict] = []
        has_face = False
        has_mask = False
        has_hand = False
        has_glove = False
        first_face_box = None
        first_hand_box = None

        for j in range(len(gm_boxes)):
            cid = int(gm_cids[j])
            class_name = str(self.glove_mask_names.get(cid, str(cid))).lower()
            px1, py1, px2, py2 = gm_boxes[j]
            global_box = np.array([px1 + x1, py1 + y1, px2 + x1, py2 + y1])

            if class_name == "face":
                has_face = True
                if first_face_box is None:
                    first_face_box = global_box
            elif class_name == "mask":
                has_mask = True
            elif class_name == "hand":
                has_hand = True
                if first_hand_box is None:
                    first_hand_box = global_box
            elif class_name == "glove":
                has_glove = True

            detections.append(
                {
                    "box": global_box,
                    "conf": float(gm_scores[j]),
                    "class_name": class_name,
                    "is_violation": False,
                }
            )

        # Inference rules requested by user:
        # hand present without glove -> no_glove, face present without mask -> no-mask.
        if has_hand and not has_glove and first_hand_box is not None:
            detections.append(
                {
                    "box": first_hand_box,
                    "conf": 1.0,
                    "class_name": "no_glove",
                    "is_violation": True,
                }
            )
        if has_face and not has_mask and first_face_box is not None:
            detections.append(
                {
                    "box": first_face_box,
                    "conf": 1.0,
                    "class_name": "no-mask",
                    "is_violation": True,
                }
            )

        return detections, has_face, has_mask, has_hand, has_glove

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

                gm_detections, _, _, _, _ = self._run_glove_mask(crop, x1, y1)
                for gm_info in gm_detections:
                    person_ppe.append(gm_info)
                    all_ppe_detections.append(gm_info)
                    if gm_info["is_violation"]:
                        person_violations.append(gm_info["class_name"])
                        person_compliant = False
                
                persons.append({
                    "box":        h_box,
                    "conf":       h_conf,
                    "ppe":        person_ppe,
                    "violations": person_violations,
                    "compliant":  person_compliant,
                })
        
        return persons, all_ppe_detections
