from __future__ import annotations

import cv2
import numpy as np

# Map model class names → display names matching assignment requirements
LABEL_MAP = {
    "Hardhat":        "Helmet",
    "No-Hardhat":     "No Helmet",
    "hardhat":        "Helmet",
    "no-hardhat":     "No Helmet",
    "NO-Hardhat":     "No Helmet",
    "Safety Vest":    "Safety Vest",
    "No-Safety Vest": "No Safety Vest",
    "safety vest":    "Safety Vest",
    "no-safety vest": "No Safety Vest",
    "NO-Safety Vest": "No Safety Vest",
    "Mask":           "Mask",
    "No-Mask":        "No Mask",
    "mask":           "Mask",
    "no-mask":        "No Mask",
    "NO-Mask":        "No Mask",
    "Gloves":         "Gloves",
    "glove":          "Gloves",
    "no_glove":       "No Gloves",
    "NO-Gloves":      "No Gloves",
    "Goggles":        "Goggles",
    "NO-Goggles":     "No Goggles",
    "Fall-Detected":  "FALL DETECTED",
}

COLORS: dict[str, tuple[int, int, int]] = {
    "Helmet":          (50,  200,  50),
    "Safety Vest":     (50,  220, 130),
    "Mask":            (30,  180, 180),
    "Gloves":          (80,  200,  80),
    "Goggles":         (200, 200,  50),
    "No Helmet":       (30,   30, 220),
    "No Safety Vest":  (60,   80, 220),
    "No Mask":         (20,   60, 200),
    "No Gloves":       (40,   40, 200),
    "No Goggles":      (0,    0,   255),
    "FALL DETECTED":   (0,    0,   255),
    "safe":            (40,  200,  40),
    "violation":       (30,   30, 220),
    "unknown":         (180, 180, 180),
}

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.48
THICKNESS  = 1


def draw_results(frame, persons, ppe_detections, fps=0.0):
    out = frame.copy()
    for ppe in ppe_detections:
        _draw_ppe_box(out, ppe)
    for person in persons:
        _draw_person_box(out, person)
    _draw_hud(out, persons, fps)
    return out


def _draw_ppe_box(frame, ppe):
    x1, y1, x2, y2 = ppe["box"].astype(int)
    label = LABEL_MAP.get(ppe["class_name"], ppe["class_name"])
    color = COLORS.get(label, (128, 128, 128))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    _put_label(frame, f"{label} {ppe['conf']:.2f}", x1, y1, color)


def _draw_person_box(frame, person):
    x1, y1, x2, y2 = person["box"].astype(int)
    h_conf = person.get("conf", 0.0)
    prefix = f"Human {h_conf:.2f}"
    
    if not person["ppe"]:
        color, status = COLORS["unknown"], f"{prefix} Worker"
    elif person["compliant"]:
        color, status = COLORS["safe"], f"{prefix} SAFE"
    else:
        violations = [LABEL_MAP.get(v, v) for v in person["violations"]]
        color  = COLORS["violation"]
        status = f"{prefix} VIOLATION: {', '.join(violations)}"
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    _put_label(frame, status, x1, y2 + 18, color)


def _draw_hud(frame, persons, fps):
    total      = len(persons)
    safe_count = sum(1 for p in persons if p["ppe"] and p["compliant"])
    viol_count = sum(1 for p in persons if p["violations"])
    lines = [
        f"FPS      : {fps:5.1f}",
        f"Workers  : {total}",
        f"Safe     : {safe_count}",
        f"Violation: {viol_count}",
    ]
    pad, line_h, panel_w = 10, 22, 210
    panel_h = pad * 2 + line_h * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, text in enumerate(lines):
        color = (255, 255, 255)
        if i == 3 and viol_count > 0:
            color = (60, 60, 220)
        elif i == 2 and safe_count > 0:
            color = (60, 200, 60)
        cv2.putText(frame, text, (pad, pad + (i + 1) * line_h), FONT, FONT_SCALE, color, THICKNESS, cv2.LINE_AA)


def _put_label(frame, text, x, y, color, font_scale=FONT_SCALE, thickness=THICKNESS):
    (tw, th), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)
    y_top = max(y - th - baseline - 4, 0)
    cv2.rectangle(frame, (x, y_top), (x + tw + 6, y + baseline), color, -1)
    cv2.putText(frame, text, (x + 3, y - baseline), FONT, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
