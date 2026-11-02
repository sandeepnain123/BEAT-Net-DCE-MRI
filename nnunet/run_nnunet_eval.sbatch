#!/bin/bash
#SBATCH --job-name=nnunet_eval
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/nnunet_eval_%j.out
#SBATCH --error=logs/nnunet_eval_%j.err

set -euo pipefail

NNUNET_BASE="${NNUNET_BASE:-$PWD/nnunet_data}"
export nnUNet_raw="${NNUNET_BASE}/nnUNet_raw"
export nnUNet_preprocessed="${NNUNET_BASE}/nnUNet_preprocessed"
export nnUNet_results="${NNUNET_BASE}/nnUNet_results"
mkdir -p logs

echo "=========================================================================="
echo "  Starting nnU-Net v2 Evaluation on 306 Test Patients"
echo "  Node: $(hostname) | Date: $(date)"
echo "=========================================================================="

python3 evaluation/evaluate_nnunet_test.py \
    --nnunet-raw "${nnUNet_raw}" \
    --nnunet-results "${nnUNet_results}" \
    --out-json "results/nnunet_test.json"

echo "=========================================================================="
echo "  nnU-Net Evaluation Complete: $(date)"
echo "=========================================================================="
