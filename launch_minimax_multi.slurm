#!/bin/bash
#SBATCH --job-name=minimax-multi
#SBATCH --partition=normal
#SBATCH --account=GOV113021
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gpus-per-node=4
#SBATCH --output=logs/minimax-multi-%j.out
#SBATCH --error=logs/minimax-multi-%j.err

# 建立日誌目錄
mkdir -p logs

# 載入必要的環境模組
module load nvhpc-hpcx-cuda12/24.7
export CC=gcc

# 切換到專案目錄
cd /work/$USER/antigravity/vllm

# 執行 高併發版 SGLang MiniMax 啟動腳本
echo "Starting High-Concurrency SGLang MiniMax-M2.5 Service on $(hostname)..."
bash launch_minimax_multi_sglang_singularity.sh
