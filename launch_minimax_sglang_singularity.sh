#!/bin/bash

# ==============================================================================
# SGLang (Singularity) 啟動腳本 - MiniMax-M2.7
# 使用官方 Docker Image 轉換的 SIF 檔案，解決主機 glibc 過舊問題
# ==============================================================================

# 1. 環境配置
echo "[1/3] 正在配置環境變數..."
# 若找不到鏡像，請執行: singularity pull /work/c00cjz00/docker/sglang_latest.sif docker://lmsysorg/sglang:latest
IMAGE_PATH="/work/c00cjz00/docker/sglang_latest.sif"
MODEL_PATH="MiniMaxAI/MiniMax-M2.7"
CACHE_DIR="/work/$USER/huggingface_cache"

export HF_HOME="$CACHE_DIR"
export SAFETENSORS_FAST_GPU=1
export OMP_NUM_THREADS=1
export CC=gcc

# 2. 自動偵測可用 GPU 數量並設定 Tensor Parallel (tp-size) 與 Expert Parallel (ep-size)
echo "[2/3] 檢查 GPU 資源..."
nvidia-smi -L

if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    TP_SIZE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
else
    TP_SIZE=$(nvidia-smi -L 2>/dev/null | wc -l)
fi

if [ "$TP_SIZE" -lt 1 ]; then
    TP_SIZE=1
fi

EP_PARAMS=""
if [ "$TP_SIZE" -eq 8 ]; then
    EP_PARAMS="--ep-size 8"
    echo "偵測到 8 張 GPU，自動啟用 Expert Parallel: --ep-size 8"
else
    echo "偵測到可用 GPU 數量: $TP_SIZE (設定 --tp-size $TP_SIZE)"
fi

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
    --tp-size "$TP_SIZE" \
    $EP_PARAMS \
    --trust-remote-code \
    --tool-call-parser minimax-m2 \
    --reasoning-parser minimax-append-think \
    --mem-fraction-static 0.85

    
