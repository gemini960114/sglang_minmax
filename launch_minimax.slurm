#!/bin/bash
#SBATCH --job-name=sglang-minimax
#SBATCH --partition=8gpus
#SBATCH --account=GOV113021
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:H200:4
#SBATCH --mem=800G
#SBATCH --output=logs/minimax-%j.out
#SBATCH --error=logs/minimax-%j.err

# 建立日誌目錄
mkdir -p logs

# 載入必要的環境模組
module load cuda/12.6
module load gcc/12.2
export CC=gcc

# 切換到專案目錄
cd /work/$USER/model/sglang_minmax

# 執行 SGLang MiniMax 啟動腳本
echo "Starting SGLang MiniMax-M2.7 Service on $(hostname)..."
bash launch_minimax.sh
