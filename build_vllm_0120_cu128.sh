#!/bin/bash
set -e

# 版本
VLLM_VERSION=v0.12.0
BUILD_DIR=~/userdata/2026/vllm-cu128-build

# CUDA 12.8
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# RTX 4090 = sm_89
export TORCH_CUDA_ARCH_LIST="8.9"

# 控制编译并行度
export MAX_JOBS=8
export NVCC_THREADS=2

# 检查环境
python -V
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda)"
nvcc --version

# 拉源码
rm -rf "${BUILD_DIR}"
git clone --depth 1 --branch "${VLLM_VERSION}" https://github.com/vllm-project/vllm.git "${BUILD_DIR}"
cd "${BUILD_DIR}"

# 关键：复用当前 torch 2.8.0+cu128，避免换 torch
python use_existing_torch.py

# 编译依赖
pip install -U pip setuptools wheel ninja packaging cmake
pip install -r requirements/build.txt

# 生成 wheel，推荐这种，之后可重复安装
pip wheel . --no-build-isolation -w dist

# 安装本地 wheel，不解析依赖，避免拉 CUDA13 / torch2.11
pip install --force-reinstall --no-deps dist/vllm-*.whl

# 检查
python -c "import vllm, torch; print('vllm:', vllm.__version__); print('torch:', torch.__version__, torch.version.cuda)"