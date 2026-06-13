#!/bin/bash

# ==============================================================================
# SGLang (Singularity) 啟動腳本 - MiniMax-M3 專用
# 模型：MiniMaxAI/MiniMax-M3
# 架構：~428B params / 23B activated, 原生多模態 (文字+圖片+影片), 1M context
# 使用 sglang-dev.sif 鏡像，解決主機 glibc 或環境相容性問題
# ==============================================================================

# 1. 環境配置
echo "[1/3] 正在配置環境變數..."
IMAGE_PATH="/work/c00cjz00/containers/sglang-dev.sif"
MODEL_PATH="MiniMaxAI/MiniMax-M3"
CACHE_DIR="/work/c00cjz00/huggingface_cache"

export HF_HOME="$CACHE_DIR"
export SAFETENSORS_FAST_GPU=1
export OMP_NUM_THREADS=1

# 強制使用 gcc (解決 Triton 編譯錯誤)
export CC=gcc
export CXX=g++

# M3 推薦參數 (來自官方文件)
export SGLANG_TEMPERATURE=1.0
export SGLANG_TOP_P=0.95
export SGLANG_TOP_K=40

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

# 3. 檢查並載入 Claude Code 相容性修補檔 (Bind Mount)
PATCH_BINDS=""
PATCH_DIR="/work/c00cjz00/model/sglang_minmax/patched_anthropic"
if [ -d "$PATCH_DIR" ]; then
    echo "[Info] 偵測到 Claude Code 修補檔，將自動進行 Bind Mount..."
    # 針對容器內實際的 /sgl-workspace/sglang/python/sglang 安裝路徑進行 Bind
    PATCH_BINDS="-B $PATCH_DIR/protocol.py:/sgl-workspace/sglang/python/sglang/srt/entrypoints/anthropic/protocol.py -B $PATCH_DIR/serving.py:/sgl-workspace/sglang/python/sglang/srt/entrypoints/anthropic/serving.py"
fi

# 4. 執行 Singularity 容器
echo "[4/4] 正在啟動 SGLang (MiniMax-M3) Singularity 容器..."
echo "模型: $MODEL_PATH"
echo "鏡像: $IMAGE_PATH"
echo "URL: http://0.0.0.0:8000"
echo ""
echo "提示：M3 為多模態模型，支援圖片與影片輸入。"
echo "      首次啟動含 DeepGEMM warmup，約需 5~15 分鐘。"
echo ""

# 使用 singularity exec 執行 sglang
singularity exec --nv -B /work $PATCH_BINDS "$IMAGE_PATH" \
    sglang serve \
    --model-path "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8000 \
    --tp-size "$TP_SIZE" \
    $EP_PARAMS \
    --trust-remote-code \
    --reasoning-parser auto \
    --tool-call-parser auto \
    --mem-fraction-static 0.85
