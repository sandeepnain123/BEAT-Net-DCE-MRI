#!/bin/bash
#SBATCH --job-name=nnunet_ychc
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/nnunet_ychc_%j.out
#SBATCH --error=logs/nnunet_ychc_%j.err

set -euo pipefail

NNUNET_BASE="${NNUNET_BASE:-$PWD/nnunet_data}"
export nnUNet_raw="${NNUNET_BASE}/nnUNet_raw"
export nnUNet_preprocessed="${NNUNET_BASE}/nnUNet_preprocessed"
export nnUNet_results="${NNUNET_BASE}/nnUNet_results"
mkdir -p logs

echo "=========================================================================="
echo "  Running nnU-Net v2 Fold 0 Inference on YCHC (N=99 Cases)"
echo "  Node: $(hostname) | Date: $(date)"
echo "=========================================================================="

python3 -u evaluation/eval_nnunet_ychc.py \
    --ychc-dir "${YCHC_DIR:-data/ychc_clean}" \
    --nnunet-results "${nnUNet_results}" \
    --out-json "results/ychc_nnunet_zeroshot.json"

echo "=========================================================================="
echo "  Completed at $(date)"
echo "=========================================================================="
