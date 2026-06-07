#!/bin/bash
#SBATCH --job-name=sglang-minimax-singularity
#SBATCH --partition=8gpus
#SBATCH --account=GOV113021
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:H200:4
#SBATCH --mem=800G
#SBATCH --output=logs/minimax-singularity-%j.out
#SBATCH --error=logs/minimax-singularity-%j.err

# 建立日誌目錄
mkdir -p logs

# 切換到專案目錄
cd /work/$USER/model/sglang_minmax

# 執行 SGLang MiniMax Singularity 啟動腳本
echo "Starting SGLang MiniMax-M2.7 Singularity Service on $(hostname)..."
bash launch_minimax_sglang_singularity.sh
