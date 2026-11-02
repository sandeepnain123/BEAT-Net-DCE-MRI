# YCHC External Cohort Zero-Shot Pre-Registration Protocol
**Document Version**: 1.0  
**Cohort Role**: EXTERNAL Zero-Shot Test Cohort (Strictly zero training, tuning, or rule modification on YCHC).  
**Cohort Source**: Complete 99 cases located at `<data_root>/sample_yunnan_hospital` and symlinked to `<data_root>/ychc_clean/` (`case_001` through `case_099`).  
**Case Inventory Status**: 99/99 complete cases (all 8 NIfTI files present per patient: `P0.nii`–`P5.nii`, `GT.nii`, `Breast_mask.nii`). Case 100 excluded ("Zenodo download failed; excluded").  
**Ground Truth Label Verification**: Confirmed across all 99 cases that labels are strictly binary $\{0, 1\}$ with tumour = 1 and background = 0. No multi-class or non-standard labels exist.

---

## 1. Preprocessing Protocol (Identical to MMC Test Inference)
All cases are preprocessed strictly following the MAMA-MIA / MMC inference pipeline:
- **Spatial Orientation**: Reoriented to `RAS` coordinate space.
- **Resampling**: Resampled to fixed spacing $(0.8, 0.8, 1.0)$ mm using trilinear/bilinear interpolation for MRI intensity volumes (`P0`–`P5`) and nearest-neighbor interpolation for discrete masks (`GT`, `Breast_mask`).
- **Intensity Normalization**: Non-zero z-score normalization per volume:
  $$\hat{I} = \frac{I - \mu_{\text{nonzero}}}{\sigma_{\text{nonzero}}}$$
- **Field of View**: No anatomical cropping applied (preserves complete resampled volume dimensions).
- **Inference Window**: Sliding window inference with patch size $128 \times 128 \times 64$, spatial overlap ratio 0.5, and inference probability threshold 0.5.
- **Post-Processing**: Zero post-processing in primary reporting (no morphological filtering or component removal).

---

## 2. Dynamic Contrast Frames Definition
- **Frame Temporal Order**:
  - $P0$: Unenhanced baseline pre-contrast frame.
  - $P1, P2, P3, P4, P5$: Post-contrast dynamic frames in strict temporal acquisition order (empirically confirmed by kinetic enhancement curves showing rapid wash-in at $P1$ followed by plateau and wash-out).

---

## 3. Analysis Levels: Primary and Secondary
1. **Primary Analysis**: Computed across the entire full resampled volume without anatomical masking.
2. **Secondary Analysis (Pre-Specified)**: Predicted segmentation mask multiplied by the officially released `Breast_mask` before computing evaluation metrics:
   $$\hat{Y}_{\text{breast}} = \hat{Y}_{\text{pred}} \odot \mathbb{I}(\text{Breast\_mask} > 0.5)$$

---

## 4. Evaluation Metrics
Evaluated on the resampled $(0.8, 0.8, 1.0)$ mm voxel grid using the identical evaluation code as MMC test set:
- **Metrics Computed**:
  - Dice Similarity Coefficient (DSC, %)
  - Intersection over Union (IoU / Jaccard, %)
  - Precision (%)
  - Recall (%)
  - 95th Percentile Hausdorff Distance (HD95, mm)
  - Average Symmetric Surface Distance (ASSD, mm)
- **Code Traceability**: `compute_all_metrics()` in `<project_root>/run_2a_2b_ensembles.py` (Commit `83c0c98`).
- **Boundary Metric Handling**: For empty predictions ($\hat{Y} = \emptyset$) when ground truth exists ($Y \neq \emptyset$), Dice is set to 0.0, IoU to 0.0, Precision to 100.0, Recall to 0.0, and HD95/ASSD are set to the full volume bounding diagonal ($D = \sqrt{\sum (s_i \cdot d_i)^2}$) and explicitly flagged in the output records.

---

## 5. Fixed Model Configurations (No Post-Hoc Tuning)
The following pre-specified configurations will be evaluated strictly once:

### a. Standard scSE 5-Channel Ensemble
- **Architecture**: 3D ResNet-50 UNet with scSE attention blocks.
- **Checkpoints**: 5-fold models loaded strictly from `results/5fold_5chan/fold_{0..4}/latest_checkpoint.pt` (Epoch 500 weights only, `epoch == 500`).
- **Ensemble Rule**: Mean sigmoid probability across the 5 fold models, thresholded at 0.5:
  $$\hat{P} = \frac{1}{5} \sum_{f=0}^4 \sigma(M_f(X)), \quad \hat{Y} = \mathbb{I}(\hat{P} \ge 0.5)$$
- **Input Channels (M1)**: $[P0, P1, P2, P1 - P0, P2 - P0]$ (subtraction channels formed after per-volume intensity z-score normalization, identical to MMC).

### b. SegResNet 5-Channel Ensemble
- **Architecture**: MONAI SegResNet (3D).
- **Checkpoints**: 5-fold models loaded from `results/isolated_block_segresnet/fold_{0..4}/best_model.pt` (Epoch 500 weights).
- **Input Channels (M1)**: Identical 5 channels $[P0, P1, P2, P1 - P0, P2 - P0]$.

### c. Phase-Randomized 2-Channel Model
- **Architecture**: 3D ResNet-50 UNet (2 input channels).
- **Checkpoint**: Loaded from `results/phase_randomized_2chan_seed0/final_model.pt` (Epoch 500 weights only).
- **Configurations Evaluated**:
  1. $[P0, P1]$ (Fixed early arterial phase)
  2. $[P0, P2]$ (Fixed intermediate peak phase)
  3. $[P0, P3]$ (Fixed delayed phase 3)
  4. $[P0, P4]$ (Fixed delayed phase 4)
  5. $[P0, P5]$ (Fixed late wash-out phase)
  6. **APCPS q95**: $[P0, P_{k^*}]$ where $k^* = \arg\max_{k \in \{1..5\}} \text{Percentile}_{95}(\max(0, P_k - P0))$.
  7. **Early-Peak**: $[P0, P_{k^*}]$ where $k^*$ is the earliest post-contrast frame $k \in \{1..5\}$ whose mean enhancement in the top-5% region is $\ge 0.90 \times \max_k$. Pre-specified and validated in `PROTOCOL.md` (Commit `83c0c98`).
  8. **Whole-Breast Mean**: $[P0, P_{k^*}]$ where $k^* = \arg\max_{k \in \{1..5\}} \text{Mean}_{\text{breast}}(\max(0, P_k - P0))$.
- **Per-Case Traceability**: Store the selected frame $k^*$ and enhancement metric values for every frame $k \in \{1..5\}$.

### d. Conventional 2-Channel Fold-4 Model
- **Architecture**: 3D ResNet-50 UNet (2 input channels).
- **Checkpoint**: Loaded from `results/5fold_2chan/fold_4/latest_checkpoint.pt` (Epoch 500 weights, strictly trained on $[I_0, I_2]$).
- **Input Channels**: $[P0, P2]$.

### e. nnU-Net v2 Fold 0 Baseline
- **Architecture**: 3D Full-Resolution nnU-Net (`nnUNetTrainer__nnUNetPlans__3d_fullres`).
- **Checkpoint**: `nnunet_data/nnUNet_results/Dataset001_MAMAMIA/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth` (1,000 epochs).
- **Input Channels**: $[P0, P1, P2]$ fed into `nnUNetv2_predict` with native internal resampling.

---

## 6. Statistical Comparison Protocol (Amendment 2 Rules)
All hypothesis tests are paired and conducted per case:
- **Primary Hypothesis Tests**: Every configuration compared against the reference **(a) Standard scSE 5-ch ensemble**.
- **Within-Model Hypothesis Tests (Model c)**:
  - Early-Peak vs APCPS q95
  - Early-Peak vs Fixed $[P0, P1]$
  - Early-Peak vs Fixed $[P0, P2]$
- **Statistical Tests**:
  - Wilcoxon signed-rank test with `zero_method="wilcox"`.
  - Effective sample size: $n_{\text{nonzero}} = \sum \mathbb{I}(d_i \neq 0)$.
  - Test statistics: $W$, asymptotic standard normal $z$, two-sided $p$-value, and effect size $r = |z| / \sqrt{n_{\text{nonzero}}}$.
  - Non-parametric bootstrap 95% confidence interval of mean paired difference (10,000 resamples, fixed seed 42).
  - 1.0-point non-inferiority: Established iff the lower bound of the one-sided 95% bootstrap CI $> -1.0$ Dice points.
  - Multiplicity control: Holm-Bonferroni correction within each family of tests.

---

## 7. Exclusion of Deprecated Artifacts
The following legacy models and checkpoints are strictly excluded and will NOT be evaluated:
- `best_modelk21.pth`
- Legacy Windows fold checkpoints
- Intermediate P5-only runs from earlier exploratory ablation attempts
