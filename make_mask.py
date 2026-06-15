#!/usr/bin/env python3
"""
Simple geometric mask generator.

Edit the SHAPES list below, then run:
  /home/pose/miniconda3/envs/realsense/bin/python3 make_mask.py

Outputs:
  demo_data/realsense_cup/masks/000001.png   (binary mask)
  demo_data/realsense_cup/masks/ob_mask.png  (RGB overlay for preview)
"""

import numpy as np
import cv2

SCENE_DIR = "/home/pose/dipl/FoundationPose/demo_data/"
RGB_FRAME  = f"{SCENE_DIR}/rgb/000001.png"
MASK_OUT   = f"{SCENE_DIR}/masks/000001.png"
VIZ_OUT    = f"{SCENE_DIR}/masks/ob_mask.png"

rgb = cv2.imread(RGB_FRAME)
H, W = rgb.shape[:2]   # 360, 640
mask = np.zeros((H, W), dtype=np.uint8)

# ── define shapes ────────────────────────────────────────────────────────────
# Each entry is a tuple: (shape, params)
#
#   ("rect",    y0, y1, x0, x1)          — pixel coordinates, exclusive end
#   ("rect_frac", y0f, y1f, x0f, x1f)   — fractions of H / W  (0.0 – 1.0)
#   ("circle",  cy, cx, radius)          — pixel coords
#   ("poly",    [(x0,y0), (x1,y1), ...]) — filled polygon, pixel coords

SHAPES = [
    # Bottom half, middle half (remove left/right quarters)
    ("rect_frac", 0.5, 1.0, 0.25, 0.75),
]
# ─────────────────────────────────────────────────────────────────────────────

for shape in SHAPES:
    kind = shape[0]
    if kind == "rect":
        _, y0, y1, x0, x1 = shape
        mask[y0:y1, x0:x1] = 255
    elif kind == "rect_frac":
        _, y0f, y1f, x0f, x1f = shape
        mask[int(y0f*H):int(y1f*H), int(x0f*W):int(x1f*W)] = 255
    elif kind == "circle":
        _, cy, cx, r = shape
        cv2.circle(mask, (cx, cy), r, 255, -1)
    elif kind == "poly":
        pts = np.array(shape[1], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

cv2.imwrite(MASK_OUT, mask)
print(f"Mask saved  → {MASK_OUT}  ({(mask > 0).sum()} white px)")

overlay = rgb.copy()
overlay[mask > 0] = (overlay[mask > 0] * 0.4 + np.array([0, 200, 0]) * 0.6).astype(np.uint8)
cv2.imwrite(VIZ_OUT, overlay)
print(f"Preview saved → {VIZ_OUT}")
