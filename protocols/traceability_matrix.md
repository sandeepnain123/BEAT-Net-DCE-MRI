# Data Provenance and Traceability Matrix
**Project**: BEAT-Net / MAMA-MIA Final Rebuttal Package  
**Protocol Hash**: Commit `83c0c98` in `<home>/3d-medimg-research`  
**Evaluation Standard**: Strict `final_model.pt` evaluated at Epoch 500 (`assert last_logged_epoch == 500`). Zero mid-training test evaluations.

---

## 1. Provenance Table of All Models & Checkpoints

| Model ID | Experiment / Role | Architecture | Input Channels | Checkpoint Path | Epoch | Seed | Git Commit | Output Evaluation File |
| :--- | :--- | :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| `std_s0` | Sec 3.0 Reference | ResNet50-UNet + scSE | 5 | `<project_root>/results/ablation_section3/3.0_ref_seed0/final_model.pt` | 500 | 0 | `83c0c98` | `metrics.json`, `per_case_metrics.csv` |
| `std_s1` | Sec 3.0 Reference | ResNet50-UNet + scSE | 5 | `<project_root>/results/ablation_section3/3.0_ref_seed1/final_model.pt` | 500 | 1 | `83c0c98` | `metrics.json`, `per_case_metrics.csv` |
| `raw3_s0` | Sec 3.1 Raw 3-Ch | ResNet50-UNet + scSE | 3 ($I_0, I_1, I_2$) | `<project_root>/results/ablation_section3/3.1_raw3ch_seed0/final_model.pt` | 500 | 0 | `83c0c98` | `metrics.json`, `per_case_metrics.csv` |
| `raw3_s1` | Sec 3.1 Raw 3-Ch | ResNet50-UNet + scSE | 3 ($I_0, I_1, I_2$) | `<project_root>/results/ablation_section3/3.1_raw3ch_seed1/final_model.pt` | 500 | 1 | `83c0c98` | `metrics.json`, `per_case_metrics.csv` |
| `sub2_s0` | Sec 3.1 Sub 2-Ch | ResNet50-UNet + scSE | 2 ($I_1-I_0, I_2-I_0$) | `<project_root>/results/ablation_section3/3.1_sub2ch_seed0/final_model.pt` | 500 | 0 | `83c0c98` | `metrics.json`, `test_per_case_3.1_sub2ch_seed0.csv` |
| `sub2_s1` | Sec 3.1 Sub 2-Ch | ResNet50-UNet + scSE | 2 ($I_1-I_0, I_2-I_0$) | `<project_root>/results/ablation_section3/3.1_sub2ch_seed1/final_model.pt` | 500 | 1 | `83c0c98` | `metrics.json`, `test_per_case_3.1_sub2ch_seed1.csv` |
| `rand_s0` | Sec 3.2 Random Init | ResNet50-UNet + scSE | 5 | `<project_root>/results/ablation_section3/3.2_randinit_seed0/final_model.pt` | 500 | 0 | `83c0c98` | (In progress - Job 171107) |
| `rand_s1` | Sec 3.2 Random Init | ResNet50-UNet + scSE | 5 | `<project_root>/results/ablation_section3/3.2_randinit_seed1/final_model.pt` | 500 | 1 | `83c0c98` | (In progress - Job 171108) |
| `std_5f_0..4` | 5-Fold Reference | ResNet50-UNet + scSE | 5 | `<project_root>/results/isolated_block_standard/fold_{0..4}/final_model.pt` | 500 | 42 | `83c0c98` | `eval_test_standard_fold_{0..4}.json` |
| `pwd_5f_0..4` | 5-Fold Ablation | Pointwise Deep scSE | 5 | `<project_root>/results/isolated_block_pointwise_deep/fold_{0..4}/final_model.pt` | 500 | 42 | `83c0c98` | `eval_test_pointwise_deep_fold_{0..4}.json` |
| `mat_5f_0..4` | 5-Fold Ablation | Matched 3x3x3 scSE | 5 | `<project_root>/results/isolated_block_boundary_matched/fold_{0..4}/final_model.pt` | 500 | 42 | `83c0c98` | `eval_test_boundary_matched_fold_{0..4}.json` |
| `ful_5f_0..4` | 5-Fold BEAT-Net | Full 3x3x3 scSE | 5 | `<project_root>/results/isolated_block_boundary_full/fold_{0..4}/final_model.pt` | 500 | 42 | `83c0c98` | `eval_test_boundary_full_fold_{0..4}.json` |
| `seg_5f_0..4` | 5-Fold Baseline | SegResNet 3D | 5 | `<project_root>/results/segresnet_5chan/fold_{0..4}/final_model.pt` | 500 | 42 | `83c0c98` | `eval_test_segresnet_fold_{0..4}.json` |
| `no_scse_f3..4` | 2-Fold Ablation | ResNet50-UNet (No scSE) | 5 | `<project_root>/results/isolated_block_no_scse/fold_{3..4}/final_model.pt` | 500 | 42 | `83c0c98` | `eval_test_no_scse_fold_{3..4}.json` |
| `kg_scse_f3..4` | 2-Fold Ablation | ResNet50-UNet (KG-scSE) | 5 | `<project_root>/results/isolated_block_kg_scse/fold_{3..4}/final_model.pt` | 500 | 42 | `83c0c98` | `eval_test_kg_scse_fold_{3..4}.json` |
| `nnunet_f0` | SOTA Baseline | PlainConvUNet 3D | 3 | `<nnunet_data>/nnUNet_results/Dataset001_MAMAMIA/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth` | 1000 | N/A | Default | `nnunet_test.json` |
| `apcps_2ch_f4` | APCPS 2-Channel | AttentionAllDecoderUNet | 2 | `<project_root>/results/5fold_2chan/fold_4/latest_checkpoint.pt` | 500 | 42 | `83c0c98` | `test_per_case_apcps_2chan.csv` |
| `apcps_prand` | APCPS Phase-Rand | AttentionAllDecoderUNet | 2 | `<project_root>/results/phase_randomized_2chan_seed0/final_model.pt` | 500 | 0 | `83c0c98` | `test_per_case_apcps_phase_rand.csv` |

---

## 2. Weight Provenance & Initialization Details

1. **Pretrained Weights Source**:
   - `best_modelfinal55.pth`: Transfer learning initialization used by `standard_scse` and all 5-fold models.
   - Origin: MedicalNet / 3D ResNet-50 pretraining on 23 medical imaging cohorts (Chen et al.).
   - Overlap Audit: Checked against all 306 MAMA-MIA held-out test cases. Overlap count = **0 cases**.
2. **First Convolution Channel Adaptation**:
   - The original ResNet-50 stem conv1 has weights shape `[64, 1, 7, 7, 7]`.
   - **5-Channel Models**: Conv1 weights expanded to `[64, 5, 7, 7, 7]`. Channels 0..2 copy the pretrained weights scaled by $1/3$; channels 3..4 (kinetic subtractions) initialize with the channel difference weights ($W_1 - W_0$ and $W_2 - W_0$).
   - **3-Channel Models**: Conv1 weights expanded to `[64, 3, 7, 7, 7]`, repeating pretrained weights across the 3 temporal phases scaled by $1/3$.
   - **2-Channel Models**: Conv1 weights expanded to `[64, 2, 7, 7, 7]`, repeating pretrained weights across the 2 phases scaled by $1/2$.
   - **Random Initialization Models (3.2)**: Kaiming normal initialization across all layers with zero transfer weights.

---

## 3. Amendment 2 Statistical Protocol Implementation

All paired statistics reported across the project adhere strictly to Amendment 2:
1. **Wilcoxon Signed-Rank Test**:
   - Zero-difference handling: Explicitly set to `zero_method="wilcox"`.
   - Effective sample size: $n_{\text{nonzero}} = \sum_{i=1}^n \mathbb{I}(d_i \neq 0)$.
   - Expected rank sum and variance:
     $$\mu_W = \frac{n_{\text{nonzero}}(n_{\text{nonzero}} + 1)}{4}, \quad \sigma_W = \sqrt{\frac{n_{\text{nonzero}}(n_{\text{nonzero}} + 1)(2n_{\text{nonzero}} + 1)}{24}}$$
   - Asymptotic $z$-statistic: $z = (W - \mu_W) / \sigma_W$.
   - Effect size: $r = |z| / \sqrt{n_{\text{nonzero}}}$.
2. **Bootstrap Confidence Intervals**:
   - Resamples: 10,000 paired resamples over all $n$ cases with fixed seed 42.
   - Two-sided 95% CI: Percentiles $[2.5\%, 97.5\%]$.
   - One-sided 95% Lower Confidence Bound: Percentile $5.0\%$.
3. **Non-Inferiority Criterion (Margin $\Delta = 1.0$ Dice Point)**:
   - Established if and only if the one-sided 95% bootstrap lower bound of $(\text{Candidate} - \text{Reference}) > -1.0$.
4. **Multiple Testing Correction**:
   - Step-down Holm-Bonferroni correction applied across the family of comparator models per metric.
