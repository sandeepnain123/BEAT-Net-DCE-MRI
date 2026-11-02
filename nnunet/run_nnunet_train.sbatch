#!/bin/bash
#SBATCH --job-name=nnunet_train
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=logs/nnunet_train_%j.out
#SBATCH --error=logs/nnunet_train_%j.err

set -euo pipefail

NNUNET_BASE="${NNUNET_BASE:-$PWD/nnunet_data}"
export nnUNet_raw="${NNUNET_BASE}/nnUNet_raw"
export nnUNet_preprocessed="${NNUNET_BASE}/nnUNet_preprocessed"
export nnUNet_results="${NNUNET_BASE}/nnUNet_results"
mkdir -p logs

echo "============================================"
echo "  nnU-Net v2 — 3d_fullres Training"
echo "  Fold: 0 (matches 960/240/306 split)"
echo "  Node: $(hostname)  Job: ${SLURM_JOB_ID:-manual}"
echo "  Start: $(date)"
echo "============================================"

# Training fold 0 = exact 960 train / 240 val split
nnUNetv2_train 001 3d_fullres 0 \
    --npz \
    -num_gpus 1

echo ""
echo "============================================"
echo "  Training done: $(date)"
echo "============================================"
