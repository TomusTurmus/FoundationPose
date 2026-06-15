# FoundationPose + SAM-6D on RealSense D555 — Setup Notes

## Overview

This documents the work done to run FoundationPose 6D pose estimation on custom
RealSense D555 data (`~/dipl/realsensePY/.images`) using real 3D models from the
KITchen dataset.  The pipeline has two stages:

1. **SAM-6D** — automatic instance segmentation to produce a per-frame object mask
2. **FoundationPose** — 6D pose estimation running inside its Docker container

---

## Files Created

| File | Purpose |
|---|---|
| `prepare_realsense.py` | Convert raw RealSense frames → FoundationPose format |
| `create_mask.py` | Interactive / depth-based mask creator for frame 0 |
| `run_realsense_cup.py` | FoundationPose inference script (run inside Docker) |
| `demo_data/realsense_cup/` | Converted dataset (rgb, depth, masks, cam_K.txt) |
| `/home/pose/dipl/SAM-6D/SAM-6D/run_realsense_seg.sh` | SAM-6D batch segmentation for all 30 frames |

---

## Key Changes and Decisions

### 1. Data Format Conversion (`prepare_realsense.py`)

The RealSense capture script stores:
- RGB: 1280×720 BGR PNG
- Depth: 640×360 float32 EXR **in metres** (already scaled by depth_scale=0.001)

FoundationPose's `YcbineoatReader` expects:
- RGB + depth at the **same resolution**
- Depth as **uint16 PNG in millimetres**

Fix applied:
- Resize RGB 1280×720 → 640×360 with `INTER_AREA` (keeps depth at native resolution)
- Convert depth: `(float32_metres * 1000).astype(uint16)` → PNG
- Clip depth at 3 m to remove background noise

### 2. Camera Intrinsics

The two streams are **not aligned** (no `rs.align` in the capture script).
`cam_K.txt` uses approximate D555 colour intrinsics halved to 640×360:

```
fx = fy = 456.5,  cx = 320,  cy = 180
```

**For better accuracy**: reconnect the camera and read exact intrinsics:
```python
profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
```

### 3. KITchen Dataset Models (mm → m unit handling)

FoundationPose expects mesh vertices in **metres**.  
SAM-6D expects CAD in **millimetres** (it divides by 1000 internally).  
KITchen PLY files (`/home/pose/dipl/datasets/KITchen/models/obj_XXXXXX.ply`) are in **mm**.

| Tool | Unit needed | What we do |
|---|---|---|
| FoundationPose | metres | `mesh.vertices /= 1000.0` after loading |
| SAM-6D | mm | pass PLY directly, no conversion |

Default object: **obj_000077.ply** (mug, 117×93×81 mm, dark red with handle).

Other cup candidates to try with `--obj_id`:

| ID | Name | Dimensions (mm) |
|---|---|---|
| 77 | mug | 117×93×81 |
| 29 | cup_large | 86×130×86 |
| 80 | FlowerCup | 108×111×76 |
| 9 | green-cup | 78×90×77 |
| 83 | g_cups (orange cyl.) | 86×86×70 |
| 81 | h_cups (blue cyl.) | 91×92×71 |
| 66 | i_cups (green cyl.) | 98×97×72 |
| 53 | j_cups (yellow cyl.) | 103×103×72 |

### 4. Initial Mask

FoundationPose needs a binary mask of the object in frame 0.  
`create_mask.py` provides two modes:
- **Polygon** — click vertices around the object in a matplotlib window
- **Depth auto** — finds the largest connected component in a depth band near the image centre

A mask was auto-generated and saved to `demo_data/realsense_cup/masks/000001.png`
(12 815 px, centred on the counter area near the robot arm at ~318,282 in 640×360).  
**Regenerate if it covers the wrong object** — run `create_mask.py` and choose option 2
or draw manually with option 1.

### 5. `run_realsense_cup.py` — mask applied every frame

The mask is loaded once (frame 0 mask) and passed to `est.register()` on every frame
rather than calling `est.track()` for subsequent frames.  This trades tracking speed
for robustness on a static scene.

---

## Running the Pipeline

### Step 0 — Convert raw data (one-time)

```bash
/home/pose/miniconda3/envs/realsense/bin/python3 prepare_realsense.py
```

Output lands in `demo_data/realsense_cup/{rgb,depth,mesh}/` and `cam_K.txt`.

### Step 1 — Create initial mask (one-time)

```bash
/home/pose/miniconda3/envs/realsense/bin/python3 create_mask.py
# choose [2] for depth auto-segmentation, or [1] to draw manually
```

Saved to `demo_data/realsense_cup/masks/000001.png`.

### Step 2 — SAM-6D segmentation (produces masks for FoundationPose)

SAM-6D runs per-frame instance segmentation and writes binary mask PNGs directly
into `demo_data/realsense_cup/masks/` for FoundationPose to consume.
FoundationPose calls `register()` (full pose init) for frames that have a SAM-6D mask
and `track()` (fast refinement from previous pose) for any frames that don't.

**One-time setup:**
```bash
cd /home/pose/dipl/SAM-6D/SAM-6D
conda env create -f Instance_Segmentation_Model/environment.yml
conda activate sam6d-ism

# The environment.yml pip section is incomplete — install inference deps manually.
# Notes on known failures:
#   - pytorch-lightning==1.8.1 pin fails on Python 3.9 (dep conflict with torchmetrics);
#     use <2.0 instead — it IS needed at runtime by model/detector.py
#   - torchmetrics==0.10.3 pin fails on Python 3.9; use <0.10 instead
#   - ruamel_yaml is referenced as both 'ruamel_yaml' and 'ruamel.yaml'; install ruamel.yaml
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install "pytorch-lightning<2.0" "torchmetrics<0.10"
pip install omegaconf hydra-core hydra-colorlog opencv-python scipy pandas imageio \
            distinctipy pycocotools scikit-image gdown fvcore iopath \
            ruamel.yaml trimesh blenderproc \
            git+https://github.com/facebookresearch/segment-anything.git

cd Instance_Segmentation_Model
python download_sam.py        # SAM ViT-H weights (~2.4 GB)
python download_dinov2.py     # DINOv2 ViT-L weights (~1.2 GB)

# Render 42 object templates (uses blenderproc + KITchen mug PLY)
# blenderproc requires CWD to be the Render directory — cd there first
cd /home/pose/dipl/SAM-6D/SAM-6D/Render
export LD_PRELOAD=/opt/conda/envs/sam6d-ism/lib/libstdc++.so.6
export LD_LIBRARY_PATH=/opt/conda/envs/sam6d-ism/lib:$LD_LIBRARY_PATH
blenderproc run render_custom_templates.py \
    --cad_path /home/pose/dipl/datasets/KITchen/models/obj_000077.ply \
    --output_dir /home/pose/dipl/SAM-6D/SAM-6D/output/obj_000077
```

**Per-run:**
```bash
conda activate sam6d-ism
bash /home/pose/dipl/SAM-6D/SAM-6D/run_realsense_seg.sh
# Results: output/obj_000077/sam6d_results/detection_XXXXXX.json + vis_XXXXXX.png
```

To try a different object: edit `OBJ_ID=77` at the top of `run_realsense_seg.sh`.

### Step 3 — FoundationPose (inside Docker)

```bash
bash /home/pose/dipl/FoundationPose/docker/run_container.sh
# Inside the container:
cd /home/pose/dipl/FoundationPose
python run_realsense_cup.py --obj_id 77 --debug 2
```

**If you get `GLIBCXX_3.4.29 not found` on first run**, the conda env's libstdc++ is
newer than the system one. Fix for the current session:

```bash
export LD_LIBRARY_PATH=/opt/conda/envs/my/lib:$LD_LIBRARY_PATH
```

Or fix permanently (run once inside the container):

```bash
conda install -c conda-forge libstdcxx-ng
```

Switch object with `--obj_id <N>` (see table above).  
Poses saved to `debug_realsense/ob_in_cam/XXXXXX.txt` (4×4 SE3 matrices).  
Visualisations saved to `debug_realsense/track_vis/XXXXXX.png`.

---

## Known Limitations / Gotchas

- **Approximate intrinsics** — update `cam_K.txt` with calibrated values for production use
- **Streams not aligned** — depth and colour cover slightly different FOVs; alignment
  would improve mask-to-depth correspondence
- **Single mask reused** — using frame-0 mask for all frames works for a mostly-static
  scene; for moving objects, pipe SAM-6D output masks per frame into FoundationPose
- **Docker required for FoundationPose** — the conda `foundationpose` env has a Python
  3.8/3.11 mycpp conflict and CUDA 11.7 vs PyTorch cu124 mismatch; always run inside Docker
- **`GLIBCXX_3.4.29` missing on first Docker run** — fix with
  `export LD_LIBRARY_PATH=/opt/conda/envs/my/lib:$LD_LIBRARY_PATH`
  or permanently: `conda install -c conda-forge libstdcxx-ng`
- **FoundationPose VRAM (cudaMalloc error 2)** — reduce iterations:
  `--est_refine_iter 2 --track_refine_iter 1`; mesh is auto-simplified to 50k faces
- **Blender `libembree4` symbol error** — same libstdc++ ABI mismatch; preload the conda one:
  ```bash
  export LD_PRELOAD=/opt/conda/envs/sam6d-ism/lib/libstdc++.so.6
  export LD_LIBRARY_PATH=/opt/conda/envs/sam6d-ism/lib:$LD_LIBRARY_PATH
  ```
  Set these before running `blenderproc`.
- **SAM-6D `proposal.squeeze_()` shape mismatch** — `detector.py:241` uses `squeeze_()` which
  drops all size-1 dims, breaking when N_query=1 (2D result) or tensor is 4D (too many dims);
  replaced with `proposal.reshape(-1, H, W)` which always gives `(N_query, H, W)`
- **SAM-6D `best_det` UnboundLocalError** — `visualize()` crashes when no detections are found
  because `best_det` is never assigned; fixed by initialising `best_det = None` and returning
  plain RGB early if the detection list is empty
- **SAM-6D `rle["size"]` unpacking error** — `mask_to_rle` in `model/utils.py` stores
  `binary_mask.shape` directly; if the mask is 3D (H,W,1) the size has 3 values and
  `rle_to_mask`'s `h, w = rle["size"]` fails; fixed with `np.squeeze(binary_mask)` before encoding
- **SAM-6D imageio deprecation** — `run_inference_custom.py` uses bare `imageio.imread` which
  changes behaviour in v3; patched to `import imageio.v2 as imageio`
- **`get_mask()` returns None for missing files** — `cv2.imread` on a non-existent mask returns
  `None`; calling `.astype(bool)` on it raises `AttributeError`; fixed with a None guard that
  falls back to the frame-0 mask
- **`camera.json` missing** — SAM-6D needs `demo_data/realsense_cup/camera.json` (different
  from `cam_K.txt`). `prepare_realsense.py` now writes it automatically. If missing, create it:
  ```bash
  echo '{"cam_K": [456.5, 0.0, 320.0, 0.0, 456.5, 180.0, 0.0, 0.0, 1.0], "depth_scale": 1.0}' \
      > /home/pose/dipl/FoundationPose/demo_data/realsense_cup/camera.json
  ```
- **`blenderproc` CWD** — must be run from the `Render/` directory; passing a relative script
  path from any other directory raises `RuntimeError: run script does not exist`
- **SAM-6D CWD** — `run_inference_custom.py` resolves `./checkpoints/` relative to CWD;
  `run_realsense_seg.sh` wraps each call in `(cd Instance_Segmentation_Model && ...)` to fix this
- **SAM-6D `pytorch-lightning` is a runtime dep** — `model/detector.py` imports it even
  for inference; install with `pip install "pytorch-lightning<2.0" "torchmetrics<0.10"`
