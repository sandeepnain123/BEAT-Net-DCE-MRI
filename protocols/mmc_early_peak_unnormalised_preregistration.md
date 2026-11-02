# Pre-Registration Protocol: MMC Early-Peak Selection on Unnormalised Resampled Intensities

**Document**: `mmc_early_peak_unnormalised_preregistration.md`  
**Date**: 2026-10-24  
**Author / Operator**: Neeraj (AI Assistant: Antigravity)  
**Status**: Formally Pre-Registered Before Computation  
**Repository**: `<research_repo>/`  

---

## 1. Rationale & Protocol Harmonization

An audit of MMC phase selection procedures established in `evaluate_early_peak_test.py` showed that early-peak selection had been calculated from cached `image.npy` volumes in `<data_root>/mama_mia_dataset/preprocessed_mmap`, which had z-score normalization applied (`NormalizeIntensityd(nonzero=True)`). 

To ensure complete mathematical harmonization with the benchmark MMC reference protocol defined in `evaluate_apcps.py` and the external cohort amendment `apcps_correction_amendment.md` (Git Commit `cb2e9c8`):
1. Early-peak selections across all $N = 306$ MMC test cases are recomputed on **raw resampled unnormalised** voxel intensities.
2. The canonical spatial grid is $0.8 \times 0.8 \times 1.0\text{ mm}$ RAS orientation.
3. Candidate frames for each test case are strictly the pre-contrast baseline $P_0$ (`_0000.nii`) and the two post-contrast acquisitions $P_1$ (`_0001.nii`) and $P_2$ (`_0002.nii`).

---

## 2. Mathematical Definition of Selection Rule

For each test patient:
1. **Raw Resampling**:
   $$P_0, P_1, P_2 = \text{Spacing}_{0.8 \times 0.8 \times 1.0}(\text{Orientation}_{\text{RAS}}(I_0, I_1, I_2))$$
   with unnormalised voxel intensities.
2. **Foreground Signal Tissue Mask $B$**:
   $$B = \{x \in \Omega \mid P_0(x) > \text{Perc}_{20}(P_0)\}$$
   (with fallback to $P_0 > 0$ if $\sum B < 100$).
3. **Clamped Non-Negative Kinetic Enhancement**:
   $$\Delta_k(x) = \max(0.0, P_k(x) - P_0(x)), \quad x \in B, \quad k \in \{1, 2\}$$
4. **Top-5% Enhancement Statistic**:
   $$\text{top5}_k = \text{Mean of highest 5\% values of } \Delta_k[B]$$
5. **Early-Peak Decision Policy**:
   $$k^* = \min \left\{ k \in \{1, 2\} \;\middle|\; \text{top5}_k \ge 0.90 \times \max_{j \in \{1, 2\}} \text{top5}_j \right\}$$

---

## 3. Strict Quarantining & Ground Rules

1. **Zero Ground Truth Leak**: Ground truth segmentation masks and Dice metrics are NEVER accessed during frame selection.
2. **Inference Invariant**: No new model inference is run for this rule; downstream Dice metrics are looked up directly from pre-existing fixed-frame evaluation records (`test_per_case_apcps_phase_rand.csv` and `test_per_case_apcps_2chan.csv` for $[P_0, P_1]$ and $[P_0, P_2]$).
3. **Reproducibility**: Per-case selection, individual $\text{top5}_1$ and $\text{top5}_2$ values, and the selection ratio $\text{top5}_1 / \max(\text{top5}_1, \text{top5}_2)$ are saved to `mmc_early_peak_unnormalised_selections.csv`.
