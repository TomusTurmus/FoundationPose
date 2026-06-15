# Cues for Wiring a New Pose Estimation Pipeline

Lessons learned getting SAM-6D + FoundationPose running on custom RealSense data.
Use this before starting — it will save hours.

---

## 0. Collect this before writing any code

The most time was lost discovering things that could have been stated upfront.
Answer these first — paste the output into context at the start of the session.

**System**
```bash
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader   # GPU + VRAM
nvcc --version                                                     # system CUDA
python -c "import torch; print(torch.__version__, torch.version.cuda)"  # PyTorch CUDA
conda env list                                                     # what envs exist
```

**Per conda env that matters** (run for each):
```bash
conda activate <env>
python --version
pip show torch opencv-python trimesh 2>/dev/null | grep -E "^Name|^Version"
```

**Docker / inference container**
- Container name or image tag
- Which conda env is active inside (`echo $CONDA_DEFAULT_ENV`)
- What's pre-installed (don't reinstall things that already work)

**Camera / capture**
- Depth format: float32 EXR in metres, or uint16 PNG in mm?
- Is `rs.align` used? (aligned streams share intrinsics; unaligned streams don't)
- Actual calibrated intrinsics at the capture resolution — or note they're approximate

**Dataset / scene**
- Which object(s) are in the scene — confirm the model ID before spending time on masks
- Rough depth range of the target object (helps tune auto-mask depth band)
- Mesh dataset units (mm? metres? check with `trimesh.load; print(mesh.extents)`)

---

## 1. Units — settle this first, before writing a single line of code

Every tool has a unit expectation. Get them wrong and the pipeline silently produces
garbage poses with no error.

| What | Unit | Notes |
|---|---|---|
| FoundationPose mesh | **metres** | divide KITchen PLY by 1000 after loading |
| SAM-6D CAD | **mm** | it does `/1000` internally for point clouds |
| Depth PNG (YcbineoatReader) | **mm uint16** | reader does `/1e3` to get metres |
| Depth EXR (RealSense) | **metres float32** | multiply by 1000 → uint16 PNG |
| Camera intrinsics | **pixels** | tied to a specific resolution — halve fx,fy,cx,cy when halving resolution |

Write the unit next to every variable and file path. Check mesh extents after loading:
`mesh.extents` should be in the range of the object's real size.

---

## 2. Data format — know what each reader expects before converting

**YcbineoatReader** (FoundationPose):
- `rgb/XXXXXX.png` — BGR, any resolution
- `depth/XXXXXX.png` — uint16, millimetres
- `masks/XXXXXX.png` — uint8, 0/255
- `cam_K.txt` — 3×3 plain text matrix

**SAM-6D run_inference_custom.py**:
- RGB + depth PNGs (depth read with imageio, cast to int32)
- `camera.json` — `{"cam_K": [fx,0,cx,0,fy,cy,0,0,1], "depth_scale": 1.0}`
  (not the same file as `cam_K.txt`)
- CAD in **mm**
- Templates pre-rendered with blenderproc

Mismatches between these two formats caused most of the debug time.

---

## 3. Environment setup — don't trust environment.yml pip sections

They are frequently incomplete or have version pins that don't resolve on the actual
Python version of the env. Do this instead:

1. `conda env create -f environment.yml` — gets the Python version and conda deps
2. Check Python version: `python --version`
3. Install pip deps manually, grouped by function — test imports after each group:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   pip install <inference deps>     # hydra, omegaconf, cv2, etc.
   pip install <training deps>      # only if you need training
   ```
4. For version pins that fail: drop the pin first, then tighten only if there's a
   real incompatibility. `pytorch-lightning==1.8.1` failed on Python 3.9 but
   `"pytorch-lightning<2.0"` worked fine.
5. Check whether a "training-only" package is actually imported at inference time
   before skipping it. `pytorch_lightning` was imported in `model/detector.py`.

---

## 4. libstdc++ — expect this on any conda env + system Python mismatch

Symptom: `symbol lookup error: libXXX.so: undefined symbol` or `GLIBCXX_3.4.29 not found`

Fix (session):
```bash
export LD_PRELOAD=/opt/conda/envs/<env>/lib/libstdc++.so.6
export LD_LIBRARY_PATH=/opt/conda/envs/<env>/lib:$LD_LIBRARY_PATH
```

Fix (permanent):
```bash
conda install -c conda-forge libstdcxx-ng
```

This affects both Python processes AND subprocesses (Blender). Set the exports before
running blenderproc.

---

## 5. Working directory — always use absolute paths in scripts

Several tools resolve relative paths from CWD, not from the script location:

- `blenderproc run script.py` — script path is relative to CWD, must cd to script dir
- `run_inference_custom.py` — resolves `./checkpoints/` from CWD
- hydra configs use relative `config_path` from the script's directory

Pattern that works in shell scripts:
```bash
(cd "$TOOL_DIR" && python script.py --arg "$ABSOLUTE_PATH")
```
Never pass relative data paths — always compute absolute paths at the top of the script.

---

## 6. GPU memory — check before blaming the code

`cudaMalloc error 2` = OOM, not a bug.

- Run `nvidia-smi` first — check free VRAM and kill other processes
- Reduce iterations: `--est_refine_iter 2 --track_refine_iter 1`
- Simplify mesh if > ~50k faces: `mesh.simplify_quadric_decimation(50_000)`
- `--debug 0` skips visualisation tensors and saves VRAM
- Headless Docker: wrap `cv2.imshow` with `if os.environ.get("DISPLAY")`

---

## 7. Upstream bugs to patch before trusting the output

SAM-6D had several bugs that surface with real data:

| File | Line | Bug | Fix |
|---|---|---|---|
| `model/detector.py` | `squeeze_()` | collapses (1,H,W)→(H,W) when N=1 | `reshape(-1, H, W)` |
| `model/utils.py` | `mask_to_rle` | stores 3D shape if mask is (H,W,1) | `np.squeeze(binary_mask)` before encoding |
| `run_inference_custom.py` | `visualize` | `best_det` unbound when no detections | initialise `best_det = None`, guard before use |
| `run_inference_custom.py` | `import imageio` | breaks in imageio v3 | `import imageio.v2 as imageio` |

Check these in any new version of SAM-6D before running.

---

## 8. Mask pipeline — the glue between segmentation and pose estimation

The connection between SAM-6D and FoundationPose is just a binary PNG mask.
SAM-6D outputs RLE-encoded JSON — you need an extraction step:

```python
from segment_anything.utils.amg import rle_to_mask
best = max(detections, key=lambda d: d["score"])
mask = rle_to_mask(best["segmentation"]).astype(np.uint8) * 255
cv2.imwrite(mask_path, mask)
```

Put this extraction inside your segmentation script so it runs per-frame and deposits
masks directly into `demo_data/.../masks/XXXXXX.png`. FoundationPose then picks them
up automatically via `YcbineoatReader.get_mask(i)`.

Use `est.register()` when a mask is available (full pose init), `est.track()` otherwise
(fast refinement from previous pose). Don't call `register()` every frame unless you
have a per-frame mask — it's much slower than `track()`.

---

## 9. Debugging order

Do these in order — each one gates the next:

1. **Data format**: load one frame manually, print shapes and value ranges
2. **Units**: print `mesh.extents` (should match object real size), print depth percentiles
3. **Mask**: visualise the mask overlay — does it cover the right object?
4. **Camera**: reproject a depth point using K, check it lands near the object in RGB
5. **Templates**: open a few `rgb_N.png` from the template dir — does the object look right?
6. **Single-frame inference**: run on frame 1 only before looping all 30
7. **Pose output**: load the 4×4 matrix, check rotation is orthogonal, translation is plausible

---

## 10. What to document as you go

Every non-obvious fix deserves a line in the README immediately — not after the session.
Record: what failed, why, and the exact fix. Future self will not remember the `squeeze_()`
subtlety or that `camera.json` and `cam_K.txt` are different files.
