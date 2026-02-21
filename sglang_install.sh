#!/bin/bash

# 設定錯誤時停止執行
set -e

# 1. 定義路徑
WORK_DIR="/work/$USER/antigravity/vllm"
CACHE_DIR="/work/$USER/huggingface_cache"
VENV_PATH="$WORK_DIR/.venv"

echo "=== 開始安裝 sglang ==="

# 2. 載入必要的編譯與 CUDA 模組
echo "載入模組: nvhpc-hpcx-cuda12/24.7 與 gcc/12.5.0"
module load nvhpc-hpcx-cuda12/24.7 || echo "警告: 無法載入 nvhpc 模組"
module load gcc/12.5.0 || echo "警告: 無法載入 gcc 12.5.0 模組"

# 設定 CUDA_HOME (deep_gemm 需要)
export CUDA_HOME=/work/HPC_software/LMOD/nvidia/packages/hpc_sdk-24.7/Linux_x86_64/24.7/cuda/12.5
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 強制使用 gcc (解決 Triton/JIT 編譯衝突)
export CC=gcc
export CXX=g++

# 3. 建立必要目錄

# 3. 使用 uv 安裝獨立的 Python 3.11 (確保包含 Python.h)
echo "正在透過 uv 下載 Python 3.11..."
uv python install 3.11

# 4. 建立虛擬環境
echo "正在建立虛擬環境於: $VENV_PATH"
if [ -d "$VENV_PATH" ]; then
    echo "刪除舊的虛擬環境..."
    rm -rf "$VENV_PATH"
fi
uv venv --python 3.11 "$VENV_PATH"

# 5. 啟動虛擬環境並安裝 sglang
echo "正在安裝 sglang 及其依賴項..."
source "$VENV_PATH/bin/activate"
uv pip install sglang

# 6. 驗證安裝與 GPU 支援
echo "=== 安裝完成，進行驗證 ==="
python -c "import sglang; print(f'sglang 版本: {sglang.__version__}')"
python -c "import torch; print(f'CUDA 是否可用: {torch.cuda.is_available()}'); print(f'CUDA 版本: {torch.version.cuda}')"

echo ""
echo "=== 使用說明 ==="
echo "請執行以下指令來啟動環境與伺服器："
echo "source $VENV_PATH/bin/activate"
echo "module load nvhpc-hpcx-cuda12/24.7"
echo "module load gcc/12.5.0"
echo "export CC=gcc"
echo "export CUDA_HOME=/work/HPC_software/LMOD/nvidia/packages/hpc_sdk-24.7/Linux_x86_64/24.7/cuda/12.5"
echo "export HF_HOME=$CACHE_DIR"
echo "python -m sglang.launch_server \\"
echo "    --model-path MiniMaxAI/MiniMax-M2.5 \\"
echo "    --tp-size 4 \\"
echo "    --trust-remote-code \\"
echo "    --tool-call-parser minimax-m2 \\"
echo "    --reasoning-parser minimax-append-think \\"
echo "    --host 0.0.0.0 \\"
echo "    --port 8000 \\"
echo "    --mem-fraction-static 0.85"
