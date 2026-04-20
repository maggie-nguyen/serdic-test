from __future__ import annotations

import numpy as np
from ultralytics import YOLO


class PPEDetector:
    def __init__(self, human_model_path: str, ppe_model_path: str, conf: float = 0.25) -> None:
        print(f"  Loading human model : {human_model_path}")
        self.human_model = YOLO(human_model_path)

        print(f"  Loading PPE model   : {ppe_model_path}")
        self.ppe_model = YOLO(ppe_model_path)
        self.ppe_model.overrides["conf"]    = conf
        self.ppe_model.overrides["iou"]     = 0.45
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
        return float((ix2-ix1)*(iy2-iy1) / max((ex2-ex1)*(ey2-ey1), 1e-6))


# ── Approach B: Per-person crop (experimental, slower ~1.5 FPS) ──────────────
# Crops each detected person and runs PPE model at higher effective resolution.
# Improves detection of small items (mask, gloves) but too slow for real-time.
#
# import cv2
# CROP_PADDING = 0.15
# CROP_SIZE    = 640
#
# def _detect_ppe_crop(self, frame, human_boxes, h, w):
#     all_ppe = []
#     for box in human_boxes:
#         crop, ox, oy, sx, sy = self._crop_person(frame, box, h, w)
#         ppe_res = self.ppe_model(crop, conf=self.conf, verbose=False)[0]
#         for det in self._parse_ppe_boxes(ppe_res):
#             cx1, cy1, cx2, cy2 = det["box"]
#             det["box"] = np.array([ox+cx1*sx, oy+cy1*sy, ox+cx2*sx, oy+cy2*sy])
#             all_ppe.append(det)
#     return self._nms_ppe(all_ppe)
#
# def _crop_person(self, frame, box, h, w):
#     x1, y1, x2, y2 = box
#     bw, bh = x2-x1, y2-y1
#     px1 = max(0, int(x1 - bw*CROP_PADDING))
#     py1 = max(0, int(y1 - bh*CROP_PADDING))
#     px2 = min(w, int(x2 + bw*CROP_PADDING))
#     py2 = min(h, int(y2 + bh*CROP_PADDING))
#     crop = frame[py1:py2, px1:px2]
#     cw, ch = crop.shape[1], crop.shape[0]
#     return cv2.resize(crop, (CROP_SIZE, CROP_SIZE)), px1, py1, cw/CROP_SIZE, ch/CROP_SIZE
