# FoundationPose — Build & Run Guide

## Why the included Dockerfile fails

Running `docker build` against `docker/dockerfile` produces **exit code 1** at the Miniconda layer (lines 27-33). The culprit is:

```dockerfile
/opt/conda/bin/conda update -n base -c defaults conda -y
```

Anaconda changed its Terms of Service in 2024. The `defaults` channel (`repo.anaconda.com`) now requires a paid license for commercial/organisational use. Inside a Docker build the channel returns an access or solver error, which exits non-zero and aborts the build.

Additional issues in the same Dockerfile that would cause later failures:

| Issue | Detail |
|---|---|
| Deprecated base image | `nvidia/cudagl` is retired; use `nvidia/cuda` |
| CUDA version mismatch | Base is CUDA 11.3, but pip installs `torch+cu118` (CUDA 11.8) |
| `conda activate` in RUN | Has no effect across layers; env vars aren't forwarded |
| `nodejs` via pip | Not a Python package; must be installed with `apt` or `conda` |
| Python 3.8 | EOL October 2024; many modern packages have dropped it |
| `qt5-default` | Removed from Ubuntu 22.04+; breaks on newer base images |

---

## Option A — Pull the pre-built image (recommended)

NVIDIA publishes a ready-to-use image. This is the fastest path and avoids all build issues.

```bash
docker pull wenbowen123/foundationpose
docker tag  wenbowen123/foundationpose foundationpose
```

---

## Option B — Build from scratch (fixed Dockerfile)

The fixed Dockerfile below replaces the broken original. Key changes:
- Uses the official `nvidia/cuda` base (CUDA 11.8 throughout)
- Replaces `defaults` channel with `conda-forge`
- Activates the conda env with `conda run` instead of `conda activate`
- Removes `nodejs` from the pip install list
- Bumps Python to 3.10

Save this as `docker/dockerfile` (replacing the original) then build:

```bash
cd /path/to/FoundationPose
docker build --network host -t foundationpose -f docker/dockerfile .
```

<details>
<summary>Fixed Dockerfile</summary>

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV TZ=US/Pacific
ENV DEBIAN_FRONTEND=noninteractive
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update --fix-missing && apt-get install -y \
    wget bzip2 ca-certificates curl git vim tmux \
    g++ gcc build-essential cmake checkinstall gfortran \
    libjpeg8-dev libtiff5-dev pkg-config yasm \
    libavcodec-dev libavformat-dev libswscale-dev \
    libxine2-dev libv4l-dev libgtk2.0-dev libtbb-dev \
    libatlas-base-dev libfaac-dev libmp3lame-dev \
    libtheora-dev libvorbis-dev libxvidcore-dev \
    libgoogle-glog-dev libgflags-dev libgphoto2-dev \
    libhdf5-dev doxygen libflann-dev libboost-all-dev \
    proj-data libproj-dev libyaml-cpp-dev \
    cmake-curses-gui libzmq3-dev freeglut3-dev \
    libprotobuf-dev protobuf-compiler \
    libopencore-amrnb-dev libopencore-amrwb-dev x264 v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# pybind11
RUN cd / && git clone https://github.com/pybind/pybind11 && \
    cd pybind11 && git checkout v2.10.0 && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DPYBIND11_INSTALL=ON -DPYBIND11_TEST=OFF && \
    make -j$(nproc) && make install

# Eigen 3.4.0
RUN cd / && wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz && \
    tar xzf eigen-3.4.0.tar.gz && cd eigen-3.4.0 && \
    mkdir build && cd build && cmake .. && make install

# Miniconda — use conda-forge to avoid Anaconda ToS issues
SHELL ["/bin/bash", "--login", "-c"]
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    /bin/bash /miniconda.sh -b -p /opt/conda && \
    rm /miniconda.sh && \
    /opt/conda/bin/conda config --system --set channel_priority strict && \
    /opt/conda/bin/conda config --system --add channels conda-forge && \
    /opt/conda/bin/conda config --system --remove channels defaults 2>/dev/null || true && \
    /opt/conda/bin/conda update -n base conda -y && \
    /opt/conda/bin/conda create -n my python=3.10 -y && \
    /opt/conda/bin/conda clean -afy

ENV PATH=/opt/conda/envs/my/bin:/opt/conda/bin:$PATH
ENV CONDA_DEFAULT_ENV=my

# PyTorch — CUDA 11.8 build
RUN conda run -n my pip install \
    torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

# PyTorch3D (stable)
RUN conda run -n my pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Core dependencies
RUN conda run -n my pip install \
    scipy joblib scikit-learn ruamel.yaml trimesh pyyaml \
    opencv-python imageio open3d transformations warp-lang \
    einops kornia pyrender

# nvdiffrast
RUN cd / && git clone https://github.com/NVlabs/nvdiffrast && \
    conda run -n my pip install /nvdiffrast

# Kaolin
RUN cd / && git clone --recursive https://github.com/NVIDIAGameWorks/kaolin && \
    conda run -n my bash -c "cd /kaolin && FORCE_CUDA=1 pip install -e ."

# Extra dependencies (nodejs via conda, not pip)
RUN conda run -n my conda install -y -c conda-forge nodejs h5py
RUN conda run -n my pip install \
    scikit-image meshcat webdataset omegaconf pypng roma \
    seaborn opencv-contrib-python openpyxl wandb imgaug \
    Ninja xlsxwriter timm albumentations xatlas rtree \
    jupyterlab objaverse ultralytics==8.0.120 pycocotools numba

ENV OPENCV_IO_ENABLE_OPENEXR=1
ENV SHELL=/bin/bash
RUN ln -sf /bin/bash /bin/sh
```

</details>

---

## Running the container

From the repo root (not from inside `docker/`):

```bash
bash docker/run_container.sh
```

The script mounts the repo and your home directory and starts an interactive bash session inside the container.

### First launch only — build C++ extensions

Run this **inside** the container after the first start:

```bash
bash build_all.sh
```

This compiles `mycpp` (required for all modes) and optionally the BundleSDF CUDA ops (model-free / NeRF path only).

### Re-entering a running container

```bash
docker exec -it foundationpose bash
```

---

## Data preparation (required before running demos)

1. Download **network weights** from [Google Drive](https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i) and place under `weights/`
   - Refiner: `2023-10-28-18-33-37`
   - Scorer: `2024-01-11-20-02-45`
2. Download **demo data** from [Google Drive](https://drive.google.com/drive/folders/1pRyFmxYXmAnpku7nGRioZaKrVJtIsroP) and extract under `demo_data/`

---

## Running the model-based demo

```bash
python run_demo.py
```

Results are written to the `debug_dir` set in argparse. The first run is slower due to JIT compilation.

---

## GPU notes

- **RTX 4090 / Ada / Hopper (sm_89+)**: the default image may not include the right CUDA arch. Pull the community image instead:
  ```bash
  docker pull shingarey/foundationpose_custom_cuda121:latest
  docker tag  shingarey/foundationpose_custom_cuda121 foundationpose
  ```
  Then run `bash docker/run_container.sh` as normal.

- **Verify GPU access inside the container**:
  ```bash
  nvidia-smi
  python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
  ```
