#!/bin/bash

# ==============================================================================
# sglang 啟動腳本 - MiniMax-M2.7 專用
# ==============================================================================

# 1. 環境配置
echo "[1/4] 正在配置環境變數..."
module load cuda/12.6 || echo "警告: 無法載入 cuda/12.6 模組"
module load gcc/12.2 || echo "警告: 無法載入 gcc/12.2 模組，編譯可能失敗"

export CUDA_HOME=/work/envstack/apps/cuda/12.6
export HF_HOME="/work/$USER/huggingface_cache"
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export SAFETENSORS_FAST_GPU=1

# 強制使用 gcc 而非 nvc (解決 Triton 編譯錯誤 Unknown switch: -Wno-psabi)
export CC=gcc
export CXX=g++

# 2. 啟動虛擬環境 (將 uv 安裝路徑加入 PATH)
echo "[2/4] 正在啟動虛擬環境..."
export PATH="$HOME/.local/bin:$PATH"
VENV_PATH="/work/$USER/model/sglang_minmax/.venv"

if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
else
    echo "錯誤: 找不到虛擬環境於 $VENV_PATH，請先執行安裝腳本。"
    exit 1
fi

# 3. 自動偵測可用 GPU 數量並設定 Tensor Parallel (tp-size) 與 Expert Parallel (ep-size)
echo "[3/4] 檢查 GPU 資源..."
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

# 4. 啟動 sglang 伺服器
echo "[4/4] 正在啟動 sglang 伺服器 (MiniMax-M2.7)..."
echo "URL: http://0.0.0.0:8000"

python -m sglang.launch_server \
    --model-path MiniMaxAI/MiniMax-M2.7 \
    --tp-size "$TP_SIZE" \
    $EP_PARAMS \
    --trust-remote-code \
    --tool-call-parser minimax-m2 \
    --reasoning-parser minimax-append-think \
    --host 0.0.0.0 \
    --port 8000 \
    --mem-fraction-static 0.85
