#!/bin/bash

# ==============================================================================
# SGLang (Singularity) 啟動腳本 - MiniMax-M2.5
# 使用官方 Docker Image 轉換的 SIF 檔案，解決主機 glibc 過舊問題
# ==============================================================================

# 1. 環境配置
echo "[1/3] 正在配置環境變數..."
# 若找不到鏡像，請執行: singularity pull /work/c00cjz00/docker/sglang_latest.sif docker://lmsysorg/sglang:latest
IMAGE_PATH="/work/c00cjz00/docker/sglang_latest.sif"
MODEL_PATH="MiniMaxAI/MiniMax-M2.5"
CACHE_DIR="/work/$USER/huggingface_cache"

export HF_HOME="$CACHE_DIR"
export SAFETENSORS_FAST_GPU=1
export OMP_NUM_THREADS=1
export CC=gcc

# 2. 檢查 GPU 資源
echo "[2/3] 檢查 GPU 資源..."
nvidia-smi -L

# 3. 執行 Singularity 容器
echo "[3/3] 正在啟動 SGLang Singularity 容器..."
echo "模型: $MODEL_PATH"
echo "URL: http://0.0.0.0:8000"

# 使用 singularity exec 執行 sglang
singularity exec --nv -B /work "$IMAGE_PATH" \
    python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8000 \
    --tp-size 4 \
    --trust-remote-code \
    --tool-call-parser minimax-m2 \
    --reasoning-parser minimax-append-think \
    --mem-fraction-static 0.85

    
