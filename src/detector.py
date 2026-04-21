from __future__ import annotations

import numpy as np
from ultralytics import YOLO


class PPEDetector:
    def __init__(self, human_model_path: str, ppe_model_path: str, conf: float = 0.25) -> None:
        print(f"  Loading human model : {human_model_path}")
        self.human_model = YOLO(human_model_path)

        print(f"  Loading PPE model   : {ppe_model_path}")
        self.ppe_model = YOLO(ppe_model_path)
        self.ppe_model.overrides["iou"]     = 0.45
        self.ppe_model.overrides["max_det"] = 1000

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
            
            # 1. Collect all overlapping PPE for this person
            ppe_for_person = []
            for ppe in ppe_detections:
                if self._ioa(box, ppe["box"]) >= 0.40:
                    ppe_for_person.append(ppe)
                    
            # 2. Conflict Resolution: If a positive class is found, ignore negative class predictors
            detected_classes = [p["class_name"].lower() for p in ppe_for_person]
            has_helmet = any(c in {"helmet", "hardhat"} for c in detected_classes)
            has_vest   = any(c in {"safety vest", "vest"} for c in detected_classes)
            has_mask   = any(c in {"mask"} for c in detected_classes)
            has_glove  = any(c in {"gloves", "glove"} for c in detected_classes)
            
            for ppe in ppe_for_person:
                cname = ppe["class_name"].lower()
                is_false_alarm = False
                
                if has_helmet and cname in {"no-helmet", "no-hardhat"}:
                    is_false_alarm = True
                if has_vest and cname in {"no-safety vest", "no-vest"}:
                    is_false_alarm = True
                if has_mask and cname in {"no-mask"}:
                    is_false_alarm = True
                if has_glove and cname in {"no-gloves", "no-glove"}:
                    is_false_alarm = True
                    
                if not is_false_alarm:
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
        return float((ix2-ix1)*(iy2-iy1) / max((ex2-ex1)*(ey2-ey1), 1e-6))

