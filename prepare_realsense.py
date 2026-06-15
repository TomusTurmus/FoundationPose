#!/usr/bin/env python3
"""
Prepare RealSense RGB+depth data for FoundationPose.

Run with:  /home/pose/miniconda3/envs/realsense/bin/python3 prepare_realsense.py

What this script does:
  1. Reads RGB (1280x720 PNG) and depth (640x360 EXR, float32 in metres)
  2. Resizes both to 640x360 and converts depth to uint16 PNG (millimetres)
  3. Writes cam_K.txt with approximate D555 intrinsics at 640x360
  4. Generates a simple cup mesh (truncated cone OBJ) in demo_data/realsense_cup/mesh/
"""

import cv2
import os
import numpy as np
import glob
import math

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

SRC_DIR   = "/home/pose/dipl/realsensePY/.images"
DST_DIR   = "/home/pose/dipl/FoundationPose/demo_data/realsense_cup"
TARGET_W  = 640
TARGET_H  = 360
ZFAR_M    = 3.0     # clip depth beyond 3 m (indoor, cup on table)

# ── Approximate D555 colour intrinsics at 1280×720, then halved to 640×360 ──
# The D555 colour module has roughly 69° H × 42° V FOV at this resolution.
# fx = (W/2) / tan(HFOV/2); update these if you have calibrated values.
FX = 456.5
FY = 456.5
CX = 320.0
CY = 180.0

# ─────────────────────────────────────────────────────────────────────────────

def make_cup_mesh(path: str,
                  r_bottom: float = 0.035,
                  r_top:    float = 0.045,
                  height:   float = 0.090,
                  slices:   int   = 32) -> None:
    """Write a textured truncated-cone OBJ (typical paper / ceramic cup).

    Dimensions in metres:
        r_bottom – base radius
        r_top    – top opening radius
        height   – total height

    The origin is at the base centre.
    """
    verts, norms, uvs, faces = [], [], [], []

    def add_ring(y, r, ny):
        for k in range(slices):
            theta = 2 * math.pi * k / slices
            c, s = math.cos(theta), math.sin(theta)
            verts.append((r * c, y, r * s))
            norms.append((c * ny, 1 - ny, s * ny))   # blend side/cap normal
            uvs.append((k / slices, (y + height * 0.5) / height))

    # Side rings
    add_ring(0,      r_bottom, 0.95)   # bottom ring
    add_ring(height, r_top,    0.95)   # top ring

    n_side = slices
    for k in range(slices):
        k1 = (k + 1) % slices
        # two triangles per quad
        faces.append((k + 1, k + n_side + 1, k1 + n_side + 1))
        faces.append((k + 1, k1 + n_side + 1, k1 + 1))

    # Bottom cap
    cap_b_start = len(verts)
    verts.append((0, 0, 0))
    norms.append((0, -1, 0))
    uvs.append((0.5, 0.0))
    for k in range(slices):
        theta = 2 * math.pi * k / slices
        verts.append((r_bottom * math.cos(theta), 0, r_bottom * math.sin(theta)))
        norms.append((0, -1, 0))
        uvs.append((0.5 + 0.5 * math.cos(theta), 0.5 + 0.5 * math.sin(theta)))
    centre_b = cap_b_start + 1
    for k in range(slices):
        k1 = (k + 1) % slices
        faces.append((centre_b, cap_b_start + 1 + k + 1, cap_b_start + 1 + k1 + 1))

    # Top cap
    cap_t_start = len(verts)
    verts.append((0, height, 0))
    norms.append((0, 1, 0))
    uvs.append((0.5, 1.0))
    for k in range(slices):
        theta = 2 * math.pi * k / slices
        verts.append((r_top * math.cos(theta), height, r_top * math.sin(theta)))
        norms.append((0, 1, 0))
        uvs.append((0.5 + 0.5 * math.cos(theta), 0.5 + 0.5 * math.sin(theta)))
    centre_t = cap_t_start + 1
    for k in range(slices):
        k1 = (k + 1) % slices
        faces.append((centre_t, cap_t_start + 1 + k1 + 1, cap_t_start + 1 + k + 1))

    # Write OBJ
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mtl_name = os.path.splitext(os.path.basename(path))[0] + ".mtl"
    mtl_path = os.path.join(os.path.dirname(path), mtl_name)

    with open(path, "w") as f:
        f.write(f"mtllib {mtl_name}\n")
        f.write("o cup\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for uv in uvs:
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        for n in norms:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        f.write("usemtl cup_material\n")
        for face in faces:
            i0, i1, i2 = face
            f.write(f"f {i0}/{i0}/{i0} {i1}/{i1}/{i1} {i2}/{i2}/{i2}\n")

    with open(mtl_path, "w") as f:
        f.write(f"newmtl cup_material\n")
        f.write("Ka 0.8 0.6 0.5\n")   # ambient (ceramic-ish)
        f.write("Kd 0.8 0.6 0.5\n")   # diffuse
        f.write("Ks 0.3 0.3 0.3\n")   # specular
        f.write("Ns 50\n")


def main():
    for sub in ("rgb", "depth", "masks", "mesh"):
        os.makedirs(f"{DST_DIR}/{sub}", exist_ok=True)

    rgb_files   = sorted(glob.glob(f"{SRC_DIR}/rgb_*.png"))
    depth_files = sorted(glob.glob(f"{SRC_DIR}/depth_*.exr"))

    assert len(rgb_files) > 0,   f"No RGB files found in {SRC_DIR}"
    assert len(rgb_files) == len(depth_files), "RGB / depth count mismatch"

    print(f"Found {len(rgb_files)} frames, converting to {TARGET_W}×{TARGET_H} …")

    for i, (rf, df) in enumerate(zip(rgb_files, depth_files)):
        name = f"{i+1:06d}.png"

        # ── RGB ──────────────────────────────────────────────────────────────
        rgb = cv2.imread(rf)
        if rgb is None:
            raise RuntimeError(f"Cannot read {rf}")
        rgb_small = cv2.resize(rgb, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        cv2.imwrite(f"{DST_DIR}/rgb/{name}", rgb_small)

        # ── Depth ─────────────────────────────────────────────────────────────
        dep = cv2.imread(df, -1)      # float32, metres
        if dep is None:
            raise RuntimeError(f"Cannot read EXR: {df}. Set OPENCV_IO_ENABLE_OPENEXR=1")
        dep = cv2.resize(dep, (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST)
        dep = np.clip(dep, 0, ZFAR_M)                   # remove far noise
        dep_mm = (dep * 1000).astype(np.uint16)          # metres → mm → uint16
        cv2.imwrite(f"{DST_DIR}/depth/{name}", dep_mm)

        print(f"  [{i+1:3d}/{len(rgb_files)}] {os.path.basename(rf)}")

    # ── Camera intrinsics ─────────────────────────────────────────────────────
    K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1.0]])
    np.savetxt(f"{DST_DIR}/cam_K.txt", K)

    import json
    cam_json = {"cam_K": [FX, 0.0, CX, 0.0, FY, CY, 0.0, 0.0, 1.0], "depth_scale": 1.0}
    with open(f"{DST_DIR}/camera.json", "w") as f:
        json.dump(cam_json, f, indent=2)
    print(f"\nSaved cam_K.txt + camera.json (approximate D555 intrinsics at {TARGET_W}×{TARGET_H})")
    print("  → Update with measured values from pyrealsense2 for better accuracy")

    # ── Cup mesh ──────────────────────────────────────────────────────────────
    mesh_path = f"{DST_DIR}/mesh/cup_simple.obj"
    make_cup_mesh(mesh_path)
    print(f"\nSaved cup mesh: {mesh_path}")
    print("  → This is a generic truncated-cone (r_bottom=3.5cm, r_top=4.5cm, h=9cm)")
    print("  → Replace with an accurate CAD model or 3D scan for better results")

    # ── Reminder: create mask ─────────────────────────────────────────────────
    print(f"""
Next step: create the initial mask for the first frame.
  Run:  /home/pose/miniconda3/envs/realsense/bin/python3 create_mask.py

The mask must be saved at:
  {DST_DIR}/masks/000001.png
""")


if __name__ == "__main__":
    main()
