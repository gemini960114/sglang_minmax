#!/bin/bash
#SBATCH --job-name=sglang-minimax
#SBATCH --partition=normal2
#SBATCH --account=GOV113021
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gpus-per-node=4
#SBATCH --output=logs/minimax-%j.out
#SBATCH --error=logs/minimax-%j.err

# 建立日誌目錄
mkdir -p logs

# 載入必要的環境模組 (雖然 Singularity 內部有環境，但主機端載入以確保路徑正確)
module load nvhpc-hpcx-cuda12/24.7

# 切換到專案目錄
cd /work/$USER/antigravity/vllm

# 執行 SGLang MiniMax 啟動腳本
echo "Starting SGLang MiniMax-M2.5 Service on $(hostname)..."
bash launch_minimax_sglang_singularity.sh
