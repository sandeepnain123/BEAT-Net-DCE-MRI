# Pre-Registration Protocol: APCPS Implementation Audit and Correction (AMBL and YCHC)

**Document**: `apcps_correction_amendment.md`  
**Date**: 2026-10-24  
**Author / Operator**: Neeraj (AI Assistant: Antigravity)  
**Status**: Formally Pre-Registered Before Evaluation  
**Git Commit**: Committed prior to running any corrected inference  

---

## 1. Audit Rationale & Discrepancy Statement

An audit of the APCPS (Adaptive Peak Contrast Phase Selection) implementation across cohorts revealed that the scripts used for YCHC and AMBL deviated from the benchmark MMC reference protocol established in `evaluate_apcps.py` and `evaluate_early_peak_test.py`:

1. **YCHC Previous Implementation (`run_gpu1_ychc.py`, `preprocess_ychc.py`)**:
   - **Intensity Space**: Used z-score normalized volumes (`phases_6c.npy` had `NormalizeIntensityd(nonzero=True)` applied during preprocessing), rather than raw resampled voxel intensities.
   - **Mask $B$**: Used the segmented whole-breast anatomical mask (`Breast_mask.nii > 0.5`), rather than the signal-intensity foreground mask ($P_0 > \text{20th percentile}$).
   - **Consequence**: In normalized z-score space, late-phase wash-out and delayed parenchymal enhancement artificially magnified differences against $P_0$, causing APCPS $q_{95}$ to artifactually select $P_5$ in 96.0% (95/99) of cases.
   - **Status**: The earlier results are labeled **"superseded - implementation deviated from protocol"**.

2. **AMBL Previous Implementation (`run_ambl_evaluation.py`)**:
   - **Subtraction**: Used unclipped difference $P_k - P_0$ without non-negative clamping ($\max(0.0, P_k - P_0)$).
   - **Mask $B$**: Used non-zero mask $P_0 \ne 0$ rather than $P_0 > \text{20th percentile}$.
   - **Status**: The earlier results are labeled **"superseded - implementation deviated from protocol"**.

3. **MMC Reference Protocol (`evaluate_apcps.py`, `evaluate_early_peak_test.py`)**:
   - Evaluates on raw resampled voxel intensities (RAS, $0.8 \times 0.8 \times 1.0\text{ mm}$ grid).
   - Non-negative kinetic enhancement: $\text{sub}_k = \max(0.0, P_k - P_0)$.
   - Foreground tissue mask: $B = (P_0 > \text{percentile}(P_0, 20))$ (with fallback if $\sum B < 100$ to $P_0 > 0$).
   - Percentile $q_{95}$: 95th percentile of $\text{sub}_k[B]$.
   - Whole-breast mean: Mean intensity of $\text{sub}_k[B]$.
   - Early-peak rule: Mean of the top-5% voxels in $\text{sub}_k[B]$; selects the earliest frame $k$ satisfying $\text{top5}(k) \ge 0.90 \times \max_j \text{top5}(j)$.

---

## 2. Audited Candidate DCE Frames

1. **AMBL (TCIA Advanced MRI Breast Lesions, N = 13 Evaluable Malignant Cases)**:
   - Audited via `ambl_frame_audit.csv`.
   - The multi-phase DCE series is `Registered AX Sen Vibrant MultiPhase` (GE Signa HDxt, SeriesNumber 500, ImageType `["ORIGINAL", "PRIMARY", "OTHER"]`).
   - Frame $P_0$: Pre-contrast unenhanced baseline acquisition (index 0).
   - Candidate post-contrast frames: $P_1, P_2, P_3, P_4$ in exact acquisition order.
   - Derived clearance maps (`TRAM`), subtraction series, and auxiliary single-phase series are excluded.

2. **YCHC (Yunnan Cancer Hospital Cohort, N = 99 Complete Cases)**:
   - Frame $P_0$: `P0.nii` (pre-contrast baseline).
   - Candidate post-contrast frames: $P_1, P_2, P_3, P_4, P_5$ in acquisition order.

---

## 3. Mathematical Definitions of Frame Selection Policies

For each patient volume on the canonical $0.8 \times 0.8 \times 1.0\text{ mm}$ RAS grid:
1. Define foreground mask:
   $$B = \{x \in \Omega \mid P_0(x) > \text{Perc}_{20}(P_0)\}$$
2. For each candidate post-contrast frame $k \in \{1, \dots, K\}$:
   $$\Delta_k(x) = \max(0.0, P_k(x) - P_0(x)), \quad x \in B$$
3. Selection Rules:
   - **APCPS $q_{95}$ ($P_{\text{peak}}$)**:
     $$k^* = \arg\max_{k \in \{1..K\}} \text{Perc}_{95}(\Delta_k)$$
   - **Whole-Breast Mean ($P_{\text{WB}}$)**:
     $$k^* = \arg\max_{k \in \{1..K\}} \text{Mean}(\Delta_k)$$
   - **Early-Peak ($P_{\text{early}}$)**:
     $$\text{top5}_k = \text{Mean of highest 5\% values of } \Delta_k$$
     $$k^* = \min \{ k \in \{1..K\} \mid \text{top5}_k \ge 0.90 \times \max_{j} \text{top5}_j \}$$

---

## 4. Models Evaluated Under Corrected Protocol

All checkpoints are strictly fixed at **Epoch 500** (`assert stored_epoch == 500`), with the sole exception of the historical baseline model (`best_modelfinal55.pth`).

1. **Historical 2-Channel Baseline (`best_modelfinal55.pth`)**:
   - Checkpoint: `<models_root>/best_modelfinal55.pth`
   - Inputs evaluated:
     - $[P_0, P_{\text{peak}}]$ (with $P_{\text{peak}}$ chosen by corrected APCPS $q_{95}$)
     - $[P_0, P_1]$ (time-ordered fixed early frame)

2. **Phase-Randomized 2-Channel Network**:
   - Checkpoint: `<project_root>/results/phase_randomized_2chan_seed0/final_model.pt` (Epoch 500).
   - Inputs evaluated:
     - $[P_0, P_{\text{peak}}]$ (APCPS $q_{95}$)
     - $[P_0, P_{\text{WB}}]$ (Whole-breast mean)
     - $[P_0, P_{\text{early}}]$ (Early-peak rule)
     - Fixed $[P_0, P_1]$ (time-ordered fixed early frame)

3. **BEAT Net 5-Channel Ensemble (Reference Model)**:
   - Checkpoints: 5 folds $\times$ `<project_root>/results/5fold_5chan/fold_{0..4}/latest_checkpoint.pt` (Epoch 500).
   - Input: Time-ordered $[P_0, P_1, P_2, P_1 - P_0, P_2 - P_0]$.

---

## 5. Statistical Invariants & Hypothesis Testing (Amendment 2)

- Primary metric: Dice Similarity Coefficient (DSC %).
- Statistical tests:
  - Paired Wilcoxon signed-rank test (`zero_method="wilcox"`), reporting $n_{\text{nonzero}}$, sum of ranks $W$, asymptotic $z$, two-sided $p$, and effect size $r = |z| / \sqrt{n_{\text{nonzero}}}$.
  - Bootstrap 95% Confidence Interval of the paired mean difference (10,000 resamples, seed 42).
  - 1.0-point non-inferiority test (one-sided 95% bootstrap lower bound $> -1.0$).
  - Two comparator baselines:
    1. vs BEAT Net 5-Channel Ensemble
    2. vs Fixed $[P_0, P_1]$
- Invariant rule: Ground truth masks or Dice scores are **NEVER** accessed during frame selection.
- Deliverable: Text-only package `~/for_claude/beatnet_bundle_apcps_fix.tar.gz`.
