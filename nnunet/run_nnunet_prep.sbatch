#!/bin/bash
#SBATCH --job-name=nnunet_prep
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/nnunet_prep_%j.out
#SBATCH --error=logs/nnunet_prep_%j.err

set -euo pipefail

NNUNET_BASE="${NNUNET_BASE:-$PWD/nnunet_data}"
export nnUNet_raw="${NNUNET_BASE}/nnUNet_raw"
export nnUNet_preprocessed="${NNUNET_BASE}/nnUNet_preprocessed"
export nnUNet_results="${NNUNET_BASE}/nnUNet_results"
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results" logs

echo "============================================"
echo "  nnU-Net v2 — Data Prep + Planning"
echo "  Node: $(hostname)  Job: ${SLURM_JOB_ID:-manual}"
echo "  Start: $(date)"
echo "============================================"

# Step 1: ensure nnunetv2 is installed
python3 -c "import nnunetv2" 2>/dev/null || {
    echo "[install] nnunetv2 not found — installing..."
    python3 -m pip install nnunetv2 --quiet
}

# Step 2: convert MAMA-MIA to nnU-Net format
echo ""
echo "=== Step 1: Data conversion ==="
python3 nnunet/prepare_nnunet.py \
    --dataset-root "${MAMAMIA_ROOT:-data/MAMA-MIA}" \
    --split-csv "${SPLIT_CSV:-data/metadata/train_test_splits.csv}" \
    --nnunet-raw "$nnUNet_raw" \
    --nnunet-preprocessed "$nnUNet_preprocessed"

# Step 3: extract fingerprint
echo ""
echo "=== Step 2: Fingerprint extraction ==="
nnUNetv2_extract_fingerprint -d 001 -c 3d_fullres --clean

# Step 4: plan and preprocess
echo ""
echo "=== Step 3: Planning and preprocessing ==="
nnUNetv2_plan_and_preprocess -d 001 -c 3d_fullres --verify_dataset_integrity

echo ""
echo "============================================"
echo "  Preparation done: $(date)"
echo "============================================"
