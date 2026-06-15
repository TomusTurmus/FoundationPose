#!/usr/bin/env python3
"""
Interactive mask creator for FoundationPose's first frame.

Run with:  conda run -n realsense python3 create_mask.py
       or: /home/pose/miniconda3/envs/realsense/bin/python3 create_mask.py

Controls (cv2 window):
  - Left-click  : add polygon vertex
  - Right-click : close polygon and save
  - r           : reset polygon
  - d           : switch to depth auto-segmentation
  - q / Esc     : quit without saving
"""

import cv2
import os
import numpy as np

SCENE_DIR  = "/home/pose/dipl/FoundationPose/demo_data/realsense_cup"
RGB_FRAME  = f"{SCENE_DIR}/rgb/000001.png"
DEPTH_FILE = f"{SCENE_DIR}/depth/000001.png"
MASK_OUT   = f"{SCENE_DIR}/masks/000001.png"

os.makedirs(os.path.dirname(MASK_OUT), exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def depth_auto_segment(depth_mm: np.ndarray) -> np.ndarray:
    """Find the largest connected component in a mid-range depth band."""
    h, w = depth_mm.shape
    valid = depth_mm[depth_mm > 0]
    if valid.size == 0:
        return _centre_crop(h, w)

    lo = int(np.percentile(valid, 10))
    hi = int(np.percentile(valid, 60))
    fg = ((depth_mm > lo) & (depth_mm < hi)).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  np.ones((5, 5),   np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(fg)
    if num <= 1:
        return _centre_crop(h, w)

    cx, cy = w // 2, h // 2
    best_id, best_score = 1, -1
    for cid in range(1, num):
        area = stats[cid, cv2.CC_STAT_AREA]
        if area < 500:
            continue
        dx = centroids[cid, 0] - cx
        dy = centroids[cid, 1] - cy
        score = area / (dx*dx + dy*dy + 1)
        if score > best_score:
            best_score, best_id = score, cid

    return (labels == best_id).astype(np.uint8) * 255


def _centre_crop(h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    m[h//4:3*h//4, w//4:3*w//4] = 255
    return m


def _overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    out[mask > 0] = (out[mask > 0] * 0.4 + np.array([0, 200, 0]) * 0.6).astype(np.uint8)
    return out


def _draw_polygon(rgb: np.ndarray, pts: list) -> np.ndarray:
    disp = rgb.copy()
    for p in pts:
        cv2.circle(disp, p, 5, (0, 0, 255), -1)
    if len(pts) > 1:
        cv2.polylines(disp, [np.array(pts, np.int32)], False, (0, 0, 255), 2)
    if len(pts) > 2:
        cv2.line(disp, pts[-1], pts[0], (0, 200, 255), 1)
    cv2.putText(disp, "L-click:add  R-click:finish  r:reset  d:auto  q:quit",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
    return disp


# ── polygon tool ──────────────────────────────────────────────────────────────

def interactive_polygon_mask(rgb: np.ndarray) -> np.ndarray | None:
    WIN = "Draw mask"
    pts = []
    done = [False]
    result = [None]

    def mouse(event, x, y, flags, _):
        if done[0]:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
            cv2.imshow(WIN, _draw_polygon(rgb, pts))
        elif event == cv2.EVENT_RBUTTONDOWN:
            done[0] = True

    h, w = rgb.shape[:2]
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, min(w * 2, 1280), min(h * 2, 800))
    cv2.setMouseCallback(WIN, mouse)
    cv2.imshow(WIN, _draw_polygon(rgb, pts))

    while not done[0]:
        key = cv2.waitKey(20) & 0xFF
        if key == ord('r'):
            pts.clear()
            cv2.imshow(WIN, _draw_polygon(rgb, pts))
        elif key == ord('d'):
            cv2.destroyWindow(WIN)
            return None          # caller will fall through to auto mode
        elif key in (ord('q'), 27):
            cv2.destroyWindow(WIN)
            return None

    cv2.destroyWindow(WIN)
    if len(pts) < 3:
        print("  Need at least 3 points – no mask created.")
        return None
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.array(pts, np.int32)], 255)
    return m


# ── preview & save ────────────────────────────────────────────────────────────

def preview_and_save(rgb: np.ndarray, mask: np.ndarray) -> None:
    px = (mask > 0).sum()
    print(f"  Mask covers {px} px / {mask.size} total ({100*px/mask.size:.1f} %)")
    side = np.hstack([rgb, _overlay(rgb, mask)])
    WIN = "Preview – press any key to save, Esc to discard"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    h, w = side.shape[:2]
    cv2.resizeWindow(WIN, min(w, 1280), min(h, 480))
    cv2.imshow(WIN, side)
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()
    if key == 27:
        print("  Discarded.")
        return
    cv2.imwrite(MASK_OUT, mask)
    print(f"  Saved → {MASK_OUT}")
    print("\nAll data ready. Enter the Docker container and run:")
    print("  python run_realsense_cup.py")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(RGB_FRAME):
        print(f"ERROR: {RGB_FRAME} not found – run prepare_realsense.py first.")
        return

    rgb      = cv2.imread(RGB_FRAME)
    depth_mm = cv2.imread(DEPTH_FILE, cv2.IMREAD_UNCHANGED)

    print(f"Frame: {RGB_FRAME}  ({rgb.shape[1]}×{rgb.shape[0]})")
    print()
    print("  [1] Draw polygon manually")
    print("  [2] Depth auto-segmentation")
    choice = input("Choice [1/2]: ").strip()

    if choice == "1":
        mask = interactive_polygon_mask(rgb)
        if mask is None:
            print("  Falling back to depth auto-segmentation …")
            mask = depth_auto_segment(depth_mm)
    else:
        print("Running depth auto-segmentation …")
        mask = depth_auto_segment(depth_mm)

    preview_and_save(rgb, mask)


if __name__ == "__main__":
    main()
