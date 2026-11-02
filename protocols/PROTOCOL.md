# Project Evaluation Protocol & Checkpoint Rules

**Date:** 2026-10-23  
**Status:** ACTIVE & BINDING  
**Scope:** All BEAT-Net / MAMA-MIA experiments and ablation runs.

---

## 1. Checkpoint Selection Rule: Strictly `final_model.pt` at Epoch 500

1. **Official Metric Checkpoint:**
   All reported test metrics (Dice, IoU, Precision, Recall, HD95 mm, ASSD mm) across all architectures, folds, and ablations must be evaluated **exclusively on `final_model.pt` saved at Epoch 500**.
2. **Assertion of Completion:**
   Before any test evaluation is executed, the evaluation pipeline must programmatically assert that the run's last logged epoch equals 500 (`assert last_logged_epoch == 500`). If not finished, evaluation must refuse to run and log `"NOT FINISHED"`.
3. **Prohibition of `best_model.pt` for Reported Results:**
   Peak/best validation checkpoints (`best_model.pt`) must never be used for primary reported test set results. This completely eliminates peak-picking bias, stochastic fluctuation exploitation, and validation snooping.
4. **Validation Metrics:**
   For cross-validation runs, reported validation metrics must reflect validation performance at Epoch 500, not the peak/best historical epoch.

---

## 2. Strict Test-Set Isolation Rule

1. **No Mid-Training Test Evaluations:**
   The 306 held-out test cases must **never** be loaded, batched, or evaluated during training epochs.
2. **Zero-Knowledge Decision Making:**
   No decision (early stopping, run selection, hyperparameter tuning, checkpoint saving, or learning rate schedule adjustment) may use any test set metric.
3. **Post-Hoc Single Evaluation:**
   Held-out test set evaluation is executed strictly **once** after Epoch 500 has completed and `final_model.pt` has been serialized to disk.
4. **Historical Intermediate Test Logs:**
   Any intermediate test evaluations logged during initial Section 3 runs are quarantined in `results/intermediate_test_log.csv` for audit transparency and will never be reported as results.

---

## 3. Intermediate Checkpoint & Resume Safety

1. Scripts may save a rollover `latest_checkpoint.pt` at regular intervals (e.g. every 25 epochs) containing model, optimizer, and scheduler states strictly for disaster recovery and preemption resumption.
2. If a run resumes from `latest_checkpoint.pt`, it resumes with identical hyperparameters and scheduler states without re-evaluating the test set.

---

## 4. Standardized Model Naming

All logs, scripts, tables, and JSON outputs must strictly adhere to the following taxonomy:
- `standard_scse`: Main reference model (ResNet-50 FlexibleUNet with standard $1\times 1\times 1$ scSE in all 5 decoder stages).
- `pointwise_deep`: Ablation variant with deep pointwise bottleneck.
- `matched_3x3x3`: Ablation variant with parameter-matched $3\times 3\times 3$ bottleneck.
- `full_3x3x3`: Ablation variant with full $3\times 3\times 3$ volumetric spatial attention (never to be called "BEAT-Net").
- `kg_scse`: Knowledge-guided spatial-channel SE ablation.
- `no_scse`: Baseline architecture without squeeze-and-excitation blocks.
- `segresnet_5ch`: 5-channel MONAI SegResNet baseline.

---

## 6. APCPS Pre-Specified Hypothesis & Validation Protocol (Amendment 3)

1. **Pre-Specified Alternative Kinetic Pairing Rule ("Early-Peak Selection"):**
   - For each post-contrast frame $k \in \{1, 2\}$, compute the mean signal enhancement $\Delta S_k = \frac{1}{|\Omega_{5\%}|} \sum_{x \in \Omega_{5\%}} (I_k(x) - I_0(x))$ within the top-5% enhancement region $\Omega_{5\%}$ (identical to standard APCPS P95).
   - Compute the peak enhancement across available post-contrast frames: $M = \max_k \Delta S_k$.
   - Select the **earliest** frame $k^*$ whose enhancement reaches at least 90% of the maximum:
     $$k^* = \min \{ k : \Delta S_k \ge 0.90 \times M \}$$
2. **Internal Validation Selection Rule (Zero Test Leakage):**
   - The selection between standard `APCPS_P95` and `early_peak` is decided **strictly and exclusively** on internal validation data: the 240-case internal validation split of Fold 4 (and across all 5 validation folds) evaluated with the `phase_randomized_2chan` model.
   - No test set data may be consulted or used to guide or modify rule selection.
3. **Post-Validation Single Evaluation:**
   - Only after internal validation comparison is finalized and documented will the selected winner be evaluated once on the 306 MMC test cases, external AMBL (13 malignant cases), and YCHC (when full dynamic series arrives), alongside fixed early $[I_0, I_1]$, fixed late $[I_0, I_2]$, and `APCPS_P95`.
   - Frame selection counts per post-contrast phase ($P_1$ vs $P_2$) will be reported across all cohorts.

---

## 7. Standardized Statistical Analysis Protocol (Amendment 2)

1. **Paired Wilcoxon Signed-Rank Test:**
   - Zero-difference handling: strictly use `zero_method="wilcox"` (discarding zero differences from ranking, reporting exact $n_{\text{nonzero}}$).
   - Test metrics reported: Wilcoxon test statistic $W$, asymptotic/exact standard normal statistic $z$, two-sided $p$-value, and effect size $r = \frac{|z|}{\sqrt{n_{\text{nonzero}}}}$.
2. **Nonparametric Bootstrap Confidence Intervals:**
   - 10,000 resamples across all paired cases to compute the two-sided 95% bootstrap CI of the mean difference ($\text{Candidate} - \text{Reference}$).
3. **Pre-Specified 1.0-Point Non-Inferiority Criterion:**
   - Clinical non-inferiority margin is pre-specified as $\delta = 1.0$ percentage points.
   - Non-inferiority of candidate vs reference is statistically established if and only if the **lower bound of the one-sided 95% bootstrap confidence interval** of $(\text{Candidate} - \text{Reference})$ exceeds $-1.0$ (i.e. $\text{CI}_{\text{lower}}^{95\%} > -1.0$). The exact lower bound and bootstrap $p$-value must be reported.
4. **Multiplicity Correction:**
   - Holm-Bonferroni correction is applied within each family of hypotheses (all candidate models evaluated against `standard_scse` for a given metric).

