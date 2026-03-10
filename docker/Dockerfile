# It is important that cuda supports the video card architectures that are important to us:
# NVIDIA GeForce RTX 5090               - sm_120 (Blackwell) - requires CUDA 12.8+
# NVIDIA GeForce RTX 3060 / RTX 3060 Ti - sm_86
# NVIDIA GeForce RTX 2080 Ti            - sm_75
# NVIDIA A100-SXM4-40GB                 - sm_80
# Build with a version tag
# docker build -t quay.io/logivations/ml_all:LS_mmpret_cuda128 -f docker/Dockerfile .

ARG PYTORCH="2.7.1"
ARG CUDA="12.8"
ARG CUDNN="9"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
ARG GITHUB_ACCESS_TOKEN=none

# fetch the key refer to https://forums.developer.nvidia.com/t/18-04-cuda-docker-image-is-broken/212892/9
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub 32
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;12.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="(dirname(which conda))/../"

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    ninja-build \
    libglib2.0-0 \
    libxrender-dev \
    ca-certificates \
 && update-ca-certificates \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Update certifi
RUN pip install --upgrade certifi
RUN pip install openmim
# Build mmcv from source (no pre-built wheel for cu128/torch2.7).
# mmcv 2.0.x doesn't compile with PyTorch 2.7 — use 2.2.0 instead.
# Exclude sm_120 (Blackwell/RTX5090) — not supported by mmcv CUDA kernels (PyTorch handles it).
RUN TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6" pip install mmcv==2.2.0 --no-cache-dir


# Install MMPretrain
RUN conda clean --all
ARG GITHUB_ACCESS_TOKEN

COPY . /mmpretrain
WORKDIR /mmpretrain


# Worked version mim list -> mmpretrain  1.0.0rc7   /workspace/mmpretrain
RUN mim install --no-cache-dir -e . --trusted-host download.openmmlab.com
RUN pip install future tensorboard scikit-learn onnx onnxruntime
