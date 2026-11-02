# BEAT-Net: Breast DCE-MRI Tumour Segmentation Network

Official PyTorch implementation of **BEAT-Net** for volumetric breast dynamic contrast-enhanced magnetic resonance imaging (DCE-MRI) tumour segmentation.

> **Note**: This repository contains the medical imaging BEAT-Net for breast DCE-MRI lesion segmentation (not the audio BeatNet model).

---

## Overview

BEAT-Net is a 3D deep learning framework specifically engineered for automated breast tumour segmentation from multiphase DCE-MRI volumes. The framework integrates:
1. **Concurrent Spatial and Channel Squeeze-and-Excitation (scSE)** attention blocks throughout decoder skip pathways to suppress non-enhancing glandular tissue and calibrate dynamic lesion boundaries.
2. **Flexible Multiphase Input Architecture** handling both full 5-channel DCE protocols (pre-contrast $P_0$, peak $P_1$, late $P_2$, and subtraction volumes $P_1 - P_0$, $P_2 - P_0$) and adaptive 2-channel protocols.
3. **Adaptive Phase Contrast Kinetic Frame Selection (APCPS)** and early-peak kinetic selection to identify the optimal diagnostic post-contrast temporal phase under variable acquisition timing across multi-center scanners.
4. **Rigorous Pre-Registered Statistical Protocol**: All models evaluated strictly at Epoch 500 (`final_model.pt`) on held-out test cases with zero test leakage, evaluated via two-sided paired Wilcoxon signed-rank tests with Holm-Bonferroni FWER control and 10,000 nonparametric bootstrap confidence intervals (non-inferiority margin $\Delta = 1.0$ Dice point).

---

## Repository Structure

```
BEAT-Net-DCE-MRI/
├── data/
│   ├── metadata/                     # Official split snapshots, inventories, and audits
│   │   ├── train_test_splits.csv     # Official 960/240/306 MMC split
│   │   ├── official_test_ids.json    # 306 held-out test case IDs
│   │   ├── ychc_case_inventory.csv   # YCHC N=99 case audit
│   │   ├── ambl_case_inventory.csv   # AMBL N=37 case audit
│   │   ├── ambl_frame_audit.csv      # AMBL temporal frame metadata
│   │   ├── early_peak_test_selections.json
│   │   ├── ambl_apcps_corrected_selections.json
│   │   └── ychc_frame_selection_counts.json
│   ├── mama_mia_5fold.py             # Stratified 5-fold cross-validation split generator
│   ├── preprocess_cache.py           # Multi-threaded cached preprocessing to memory-mapped arrays
│   ├── preprocess_ychc.py            # YCHC external dataset preprocessing
│   ├── preprocess_ambl.py            # AMBL external dataset preprocessing
│   ├── setup_ychc_clean.py           # Standardized symlink organizer
│   └── populate_and_audit_ychc.py    # Audit and integrity validation
├── eda/
│   └── mmc_temporal_probability05.csv # Per-case MMC test Dice at probability 0.5 vs legacy logit 0.5
├── models/
│   ├── attention_55_model.py         # FlexibleUNet + 3D scSE decoder (BEAT-Net primary architecture)
│   ├── attention_isolated_model.py   # Isolated ablation variants (boundary-full, matched, pointwise, etc.)
│   ├── best_model_no_scse_block.py   # Ablation architecture: No-scSE baseline
│   └── best_model_KG_scSE_block.py   # Ablation architecture: Kernel-Gated scSE
├── training/
│   ├── train_5fold.py                # 5-fold cross-validation trainer (Epoch 500 enforcement)
│   ├── train_phase_randomized_2chan.py # Phase-randomised 2-channel trainer
│   ├── train_isolated_fast.py        # Fast trainer for architectural ablation variants
│   └── train_section3_ablation.py    # Section 3 input channel & pretraining ablations
├── frame_selection/
│   └── run_apcps_correction_eval.py  # Corrected APCPS, early-peak, and whole-breast kinetic selection
├── evaluation/
│   ├── evaluate_single_fold_test.py  # Single fold evaluation on 306 MMC held-out test cases
│   ├── run_2a_2b_ensembles.py        # 5-fold ensemble evaluation and paired comparisons
│   ├── run_update_2a_2b.py           # Refresh script including no-scSE and KG-scSE
│   ├── evaluate_nnunet_test.py       # nnU-Net v2 test set benchmark evaluation
│   ├── evaluate_p0_test.py           # Probability-0.5 threshold re-evaluation script
│   ├── harmonise_mmc_temporal.py     # Probability 0.5 vs logit 0.5 temporal threshold reconciler
│   ├── run_step2c_ychc_zeroshot.py   # Zero-shot evaluation on YCHC (N=99)
│   ├── run_ambl_evaluation.py        # Zero-shot evaluation on AMBL (N=37)
│   ├── eval_nnunet_ychc.py           # nnU-Net zero-shot benchmark on YCHC
│   ├── synthesize_ychc_results.py    # Multi-model YCHC synthesis and paired stats
│   ├── run_unified_complexity_benchmark.py # A100 GPU FLOPs, params, latency, and VRAM benchmark
│   └── p0_correction_evaluator/      # Cryptographically frozen evaluator package (v1.0.0)
├── statistics/
│   ├── compute_section3_stats_e500.py # Pairwise Wilcoxon, bootstrap CIs, and Holm adjustment
│   ├── compute_2a_paired_stats.py     # Section 3 paired analysis (raw3ch vs ref)
│   └── synthesize_5fold_statistics.py # 5-fold CV and test synthesis across architectures
├── nnunet/
│   ├── prepare_nnunet.py             # MAMA-MIA to nnU-Net v2 format dataset converter
│   ├── run_nnunet_prep.sbatch        # Fingerprint, plan, and preprocess Slurm script
│   ├── run_nnunet_train.sbatch       # nnU-Net v2 3d_fullres training Slurm script
│   ├── run_nnunet_eval.sbatch        # nnU-Net v2 MMC test evaluation Slurm script
│   └── run_nnunet_ychc.sbatch        # nnU-Net v2 YCHC inference Slurm script
└── protocols/                        # Pre-registrations, formal amendments, and provenance audits
    ├── PROTOCOL.md                   # Primary frozen execution protocol
    ├── ychc_preregistration.md        # YCHC zero-shot study pre-registration
    ├── ambl_preregistration.md        # AMBL zero-shot study pre-registration
    ├── apcps_correction_amendment.md # Formal amendment for APCPS coordinate correction
    ├── mmc_early_peak_unnormalised_preregistration.md
    ├── lococv_provenance.md          # Multi-center cross-validation provenance audit
    └── traceability_matrix.md        # Complete run-to-checkpoint traceability matrix
```

---

## Mapping to Manuscript Sections and Tables

The table below maps each script in this repository to its corresponding section, experiment, and table in the manuscript:

| Paper Section / Table | Topic & Experiments | Primary Scripts | Output Artifacts |
|---|---|---|---|
| **Table 2** | Datasets & Multi-Center Cohorts (MMC: Duke, I-SPY 1, I-SPY 2, NACT; External: YCHC, AMBL) | [`data/mama_mia_5fold.py`](data/mama_mia_5fold.py)<br/>[`data/preprocess_cache.py`](data/preprocess_cache.py)<br/>[`data/preprocess_ychc.py`](data/preprocess_ychc.py)<br/>[`data/preprocess_ambl.py`](data/preprocess_ambl.py) | `data/metadata/train_test_splits.csv`<br/>`data/metadata/ychc_case_inventory.csv`<br/>`data/metadata/ambl_case_inventory.csv` |
| **Table 3** | 5-Fold Cross-Validation & Held-Out Test Evaluation on MMC ($N=306$) | [`training/train_5fold.py`](training/train_5fold.py)<br/>[`training/train_isolated_fast.py`](training/train_isolated_fast.py)<br/>[`evaluation/evaluate_single_fold_test.py`](evaluation/evaluate_single_fold_test.py)<br/>[`evaluation/run_2a_2b_ensembles.py`](evaluation/run_2a_2b_ensembles.py) | `results/5fold_summary.json`<br/>`results/stats_vs_standard.json` |
| **Table 4** | Section 3 Input Channel and Pretraining Ablation Experiments (Raw 3-Ch, Sub 2-Ch, Random Init) | [`training/train_section3_ablation.py`](training/train_section3_ablation.py)<br/>[`statistics/compute_section3_stats_e500.py`](statistics/compute_section3_stats_e500.py)<br/>[`statistics/compute_2a_paired_stats.py`](statistics/compute_2a_paired_stats.py) | `results/section3_stats_E500.json`<br/>`results/paired_stats_sec3_raw3_vs_ref.json` |
| **Table S1** | Architecture Ablation of scSE Attention Blocks (Standard scSE, Boundary-Full, Boundary-Matched, Pointwise-Deep, No-scSE, KG-scSE, SegResNet 3D) | [`models/attention_55_model.py`](models/attention_55_model.py)<br/>[`models/attention_isolated_model.py`](models/attention_isolated_model.py)<br/>[`models/best_model_no_scse_block.py`](models/best_model_no_scse_block.py)<br/>[`models/best_model_KG_scSE_block.py`](models/best_model_KG_scSE_block.py)<br/>[`evaluation/run_2a_2b_ensembles.py`](evaluation/run_2a_2b_ensembles.py) | `results/ensemble_2a_2b_summary.json`<br/>`results/stats_vs_standard.csv` |
| **Table S2** | Paired Statistical Tests on MMC Held-Out Test Set (Wilcoxon Signed-Rank, 10,000 Bootstrap 95% CIs, Holm-Bonferroni FWER Control) | [`evaluation/run_2a_2b_ensembles.py`](evaluation/run_2a_2b_ensembles.py)<br/>[`statistics/synthesize_5fold_statistics.py`](statistics/synthesize_5fold_statistics.py)<br/>[`statistics/compute_section3_stats_e500.py`](statistics/compute_section3_stats_e500.py) | `results/final_5fold_synthesis.json`<br/>`results/stats_vs_standard.json` |
| **Table S3** | Adaptive Phase Contrast Kinetic Frame Selection (Corrected APCPS, Early-Peak Kinetic, Whole-Breast Kinetic Mean vs Conventional Fixed Timing) | [`frame_selection/run_apcps_correction_eval.py`](frame_selection/run_apcps_correction_eval.py)<br/>[`training/train_phase_randomized_2chan.py`](training/train_phase_randomized_2chan.py)<br/>[`evaluation/harmonise_mmc_temporal.py`](evaluation/harmonise_mmc_temporal.py)<br/>[`evaluation/evaluate_p0_test.py`](evaluation/evaluate_p0_test.py)<br/>[`evaluation/p0_correction_evaluator/evaluate.py`](evaluation/p0_correction_evaluator/evaluate.py) | `results/apcps_correction_evaluation.json`<br/>`eda/mmc_temporal_probability05.csv`<br/>`data/metadata/early_peak_test_selections.json`<br/>`data/metadata/ambl_apcps_corrected_selections.json` |
| **Table S4** | Multi-Center External Zero-Shot Generalization (YCHC $N=99$ and AMBL $N=37$) | [`evaluation/run_step2c_ychc_zeroshot.py`](evaluation/run_step2c_ychc_zeroshot.py)<br/>[`evaluation/run_ambl_evaluation.py`](evaluation/run_ambl_evaluation.py)<br/>[`evaluation/eval_nnunet_ychc.py`](evaluation/eval_nnunet_ychc.py)<br/>[`evaluation/synthesize_ychc_results.py`](evaluation/synthesize_ychc_results.py) | `results/ychc_zeroshot_results.json`<br/>`results/ambl_evaluation_results.json`<br/>`results/ychc_synthesis.json` |
| **Table S5** | Unified Model Complexity, Parameters, FLOPs, Latency & VRAM Benchmarks (A100 GPU) | [`evaluation/run_unified_complexity_benchmark.py`](evaluation/run_unified_complexity_benchmark.py) | `results/unified_complexity_benchmark.json`<br/>`results/unified_complexity_benchmark.csv` |

---

## Methodological Note: Probability 0.5 Thresholding vs Legacy Logit Thresholding

All neural network models in this repository output unnormalized raw logits from their final convolutional projection layer (no built-in sigmoid).

- **Standard / Corrected Thresholding**: The binary segmentation mask is obtained via:
  $$\hat{Y} = \sigma(z) \ge 0.5 \iff z \ge 0.0$$
  where $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the sigmoid activation function. This is enforced across all current evaluators and trainers (including `training/train_section3_ablation.py` line ~250 and `evaluation/harmonise_mmc_temporal.py`).
- **Legacy Logit Thresholding (Historical Note)**: In exploratory early ablation sweeps recorded in Table S2, an unnormalized logit threshold of $z \ge 0.5$ (equivalent to $\sigma(z) \ge 0.622$) was applied by the legacy evaluation routine. As documented in [`eda/mmc_temporal_probability05.csv`](eda/mmc_temporal_probability05.csv) and [`evaluation/harmonise_mmc_temporal.py`](evaluation/harmonise_mmc_temporal.py), the mean absolute difference across the 306 MMC test cases between $\sigma(z) \ge 0.5$ and $z \ge 0.5$ is less than 0.17 Dice percentage points, confirming identical statistical rankings and non-inferiority conclusions across both threshold definitions.

---

## Requirements and Installation

### System Requirements
- Linux (x86_64)
- Python >= 3.10
- CUDA >= 11.8 / 12.0 compatible GPU (NVIDIA RTX 3090, A100, or H100 recommended)

### Environment Setup

1. **Clone the repository:**
   ```bash
   git clone git@github.com:sandeepnain123/BEAT-Net-DCE-MRI.git
   cd BEAT-Net-DCE-MRI
   ```

2. **Create and activate a virtual environment:**
   ```bash
   conda create -n beatnet python=3.10 -y
   conda activate beatnet
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage Guide

### 1. Data Preparation and Preprocessing

#### MMC Dataset
Generate 5-fold cross-validation splits and cache resampled/normalized volumes to memory-mapped files:
```bash
# Generate 5-fold stratified split
python data/mama_mia_5fold.py \
    --data-root /path/to/MAMA-MIA \
    --out-csv data/metadata/train_test_splits.csv

# Multi-threaded cached preprocessing
python data/preprocess_cache.py \
    --data-root /path/to/MAMA-MIA \
    --split-csv data/metadata/train_test_splits.csv \
    --out-dir data/preprocessed_mmap \
    --workers 16
```

#### External Datasets (YCHC and AMBL)
```bash
# Preprocess YCHC dataset (N=99 cases) to RAS 0.8 x 0.8 x 1.0 mm
python data/preprocess_ychc.py \
    --src-dir /path/to/ychc_raw \
    --out-dir data/ychc_clean \
    --workers 8

# Preprocess AMBL dataset (N=37 cases)
python data/preprocess_ambl.py \
    --raw-dir /path/to/ambl_raw \
    --out-dir data/ambl_clean
```

---

### 2. Model Training

#### Standard 5-Fold BEAT-Net Training (Table 3)
Trains 5 folds to Epoch 500, checkpointing strictly at Epoch 500:
```bash
python training/train_5fold.py \
    --fold 0 \
    --data-root /path/to/MAMA-MIA \
    --mmap-dir data/preprocessed_mmap \
    --out-dir results/5fold_5chan/fold_0 \
    --epochs 500 \
    --batch-size 2 \
    --lr 2e-4
```

#### Phase-Randomised 2-Channel Training for Adaptive Phase Selection (Table S3)
```bash
python training/train_phase_randomized_2chan.py \
    --mmap-dir data/preprocessed_mmap \
    --split-csv data/metadata/train_test_splits.csv \
    --out-dir results/phase_randomized_2chan_seed0 \
    --seed 0 \
    --epochs 500
```

#### Section 3 Input Channel and Pretraining Ablation Training (Table 4)
```bash
# Raw 3-Channel ablation (I_0, I_1, I_2)
python training/train_section3_ablation.py \
    --exp 3.1_raw3ch \
    --seed 0 \
    --mmap-dir data/preprocessed_mmap \
    --out-dir results/ablation_section3/3.1_raw3ch_seed0

# Subtraction 2-Channel ablation (I_1 - I_0, I_2 - I_0)
python training/train_section3_ablation.py \
    --exp 3.1_sub2ch \
    --seed 0 \
    --mmap-dir data/preprocessed_mmap \
    --out-dir results/ablation_section3/3.1_sub2ch_seed0

# Random Initialisation ablation (no ImageNet pretraining)
python training/train_section3_ablation.py \
    --exp 3.2_random_init \
    --seed 0 \
    --mmap-dir data/preprocessed_mmap \
    --out-dir results/ablation_section3/3.2_random_init_seed0
```

---

### 3. Model Evaluation

#### Single-Fold and 5-Fold Ensemble Evaluation (Tables 3, S1, S2)
```bash
# Evaluate single fold checkpoint (strictly Epoch 500)
python evaluation/evaluate_single_fold_test.py \
    --ckpt results/5fold_5chan/fold_0/final_model.pt \
    --mmap-dir data/preprocessed_mmap \
    --split-csv data/metadata/train_test_splits.csv \
    --out-json results/eval_fold0_test.json

# 5-fold ensemble evaluation and paired Wilcoxon test
python evaluation/run_2a_2b_ensembles.py \
    --results-dir results \
    --mmap-dir data/preprocessed_mmap \
    --out-json results/ensemble_summary.json
```

#### Adaptive Phase Kinetic Frame Selection Evaluation (Table S3)
```bash
# Corrected APCPS, Early-Peak, and Whole-Breast kinetic selection
python frame_selection/run_apcps_correction_eval.py \
    --mmap-dir data/preprocessed_mmap \
    --ckpt results/phase_randomized_2chan_seed0/final_model.pt \
    --out-json results/apcps_correction_evaluation.json

# Probability 0.5 vs Logit 0.5 temporal threshold reconciliation
python evaluation/harmonise_mmc_temporal.py \
    --mmap-dir data/preprocessed_mmap \
    --ckpt results/phase_randomized_2chan_seed0/final_model.pt \
    --out-csv eda/mmc_temporal_probability05.csv \
    --out-summary eda/threshold_reconciliation_summary.json
```

#### External Zero-Shot Generalization (Table S4)
```bash
# Evaluate on YCHC cohort
python evaluation/run_step2c_ychc_zeroshot.py \
    --ychc-dir data/ychc_clean \
    --ckpt-5chan results/5fold_5chan \
    --ckpt-prand results/phase_randomized_2chan_seed0/final_model.pt \
    --out-json results/ychc_zeroshot_results.json

# Evaluate on AMBL cohort
python evaluation/run_ambl_evaluation.py \
    --ambl-dir data/ambl_clean \
    --ckpt-5chan results/5fold_5chan \
    --out-json results/ambl_evaluation_results.json
```

---

### 4. Statistical Analysis

Compute paired Wilcoxon signed-rank tests ($W, z, p$), non-inferiority margin analysis ($\Delta = 1.0$), 10,000 bootstrap 95% confidence intervals, and Holm-Bonferroni adjusted $p$-values:

```bash
# Section 3 Ablations Statistical Analysis (Table 4)
python statistics/compute_section3_stats_e500.py \
    --results-dir results \
    --out-json results/section3_stats_E500.json

# 5-Fold Architecture Synthesis and Paired Statistics (Tables S1 & S2)
python statistics/synthesize_5fold_statistics.py \
    --results-dir results \
    --out-json results/final_5fold_synthesis.json
```

---

### 5. Unified Computational Complexity Benchmark (Table S5)

Benchmarks total parameter counts, FLOPs (GFLOPs), inference latency (mean and standard deviation over 50 warm-up and 100 timed passes), and peak GPU memory allocation on NVIDIA A100 GPU:
```bash
python evaluation/run_unified_complexity_benchmark.py \
    --device cuda:0 \
    --out-dir results
```

---

## License

This repository is licensed under the [MIT License](LICENSE).
