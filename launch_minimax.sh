#!/bin/bash

# ==============================================================================
# sglang 啟動腳本 - MiniMax-M2.5 專用
# ==============================================================================

# 1. 環境配置
echo "[1/4] 正在配置環境變數..."
module load nvhpc-hpcx-cuda12/24.7 || echo "警告: 無法載入 nvhpc 模組"
module load gcc/12.5.0 || echo "警告: 無法載入 gcc 12.5.0 模組，編譯可能失敗"

export CUDA_HOME=/work/HPC_software/LMOD/nvidia/packages/hpc_sdk-24.7/Linux_x86_64/24.7/cuda/12.5
export HF_HOME="/work/$USER/huggingface_cache"
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export SAFETENSORS_FAST_GPU=1

# 強制使用 gcc 而非 nvc (解決 Triton 編譯錯誤 Unknown switch: -Wno-psabi)
export CC=gcc
export CXX=g++

# 2. 啟動虛擬環境
echo "[2/4] 正在啟動虛擬環境..."
VENV_PATH="/work/$USER/antigravity/vllm/.venv"

if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
else
    echo "錯誤: 找不到虛擬環境於 $VENV_PATH，請先執行安裝腳本。"
    exit 1
fi

# 3. 檢查 GPU 狀態 (選用)
echo "[3/4] 檢查 GPU 資源..."
nvidia-smi -L

# 4. 啟動 sglang 伺服器
echo "[4/4] 正在啟動 sglang 伺服器 (MiniMax-M2.5)..."
echo "URL: http://0.0.0.0:8000"

python -m sglang.launch_server \
    --model-path MiniMaxAI/MiniMax-M2.5 \
    --tp-size 4 \
    --trust-remote-code \
    --tool-call-parser minimax-m2 \
    --reasoning-parser minimax-append-think \
    --host 0.0.0.0 \
    --port 8000 \
    --mem-fraction-static 0.85
