#!/bin/bash
#SBATCH --job-name=sglang-minimax-m3-singularity
#SBATCH --partition=8gpus
#SBATCH --account=GOV113021
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:H200:8
#SBATCH --mem=800G
#SBATCH --output=logs/minimax-m3-singularity-%j.out
#SBATCH --error=logs/minimax-m3-singularity-%j.err

# ==============================================================================
# Slurm 任務腳本 - MiniMax-M3 (Singularity 容器版)
# 用法: sbatch launch_minimax_m3_singularity.slurm
# 提交後查看節點: squeue -u $USER
# 查看 log: tail -f logs/minimax-m3-singularity-<jobid>.out
# ==============================================================================

# 建立日誌目錄
mkdir -p /work/$USER/model/sglang_minmax/logs

# 切換到專案目錄
cd /work/$USER/model/sglang_minmax

# 印出節點資訊
echo "========================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $(hostname)"
echo "Start Time : $(date '+%Y-%m-%d %H:%M:%S')"
echo "GPU(s)     : $CUDA_VISIBLE_DEVICES"
echo "========================================"

# 執行 SGLang MiniMax-M3 Singularity 啟動腳本
echo "Starting SGLang MiniMax-M3 Singularity Service on $(hostname)..."
bash launch_minimax_m3_sglang_singularity.sh
