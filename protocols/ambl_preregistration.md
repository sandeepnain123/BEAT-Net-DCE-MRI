# Pre-Registration Protocol: TCIA Advanced MRI Breast Lesions (AMBL) External Zero-Shot Evaluation

**Date**: 2026-10-24  
**Author / Operator**: Neeraj (AI Assistant: Antigravity)  
**Cohort Role**: External Zero-Shot Test Cohort ($N = 13$ Evaluable Malignant Cases)  
**Strict Invariants**: No training, fine-tuning, threshold optimization, or rule selection on AMBL. Zero manuscript (.docx) edits. Fully reproducible, traceable per-case outputs.

---

## 1. Cohort Provenance & Source Data

- **Archive Source**: The Cancer Imaging Archive (TCIA) collection: *Standard and Delayed Contrast-Enhanced MRI of Malignant and Benign Breast Lesions with Histological and Clinical Supporting Data* (`ADVANCED-MRI-BREAST-LESIONS`).
- **Archive File**: `tcia_output.rar` (4,558 MB, SHA-256: `341d3d611867c212fa92d3f6688461a2936746961c0e35794aa6ab26f0b0fcba`), transferred from Windows host (`D:\tcia_output\tcia_output.rar`) to Linux cluster `<data_root>/ambl_raw/`.
- **Clinical & Pathology Master File**: Official TCIA metadata spreadsheet `Advanced-MRI-Breast-Lesions-DA-Clinical-Sep2024.xlsx` (SHA-256: `b6771d9d437ce499ce5b8fc65fc4ff4d88e404bfb0d36c5339f4d7f573516dc0`), downloaded directly from `https://www.cancerimagingarchive.net/wp-content/uploads/Advanced-MRI-Breast-Lesions-DA-Clinical-Sep2024.xlsx`.
- **Segmentation Annotations**: Expert DICOM-SEG objects converted via `dcmqi` (`segimage2itkimage`) with full provenance in `meta.json` and `seg_conversion.log`.

---

## 2. Inclusion & Exclusion Criteria (All 33 Available Patients)

In strict adherence to protocol:
1. Malignancy status is established **ONLY** from the TCIA clinical/pathology file (`tumor/benign` columns and `pathology` diagnostic codes).
2. The ground truth tumor mask is established **ONLY** from segments whose `SegmentDescription` is `"Tumor"`. Cases with 0 tumor segments or multiple tumor segments are explicitly cataloged and excluded from the primary single-lesion evaluation cohort.
3. Historical evaluation scripts (`eval_modelfinal55_5chan.py`) and legacy reports (`FINAL_AMBL_EVALUATION_MASTER_REPORT.md`) are entirely disregarded.

### Case Classification Matrix (33 Cases)

| Patient ID | Clinical Tumor/Benign | Clinical Pathology Diagnosis | DICOM-SEG Attributes | Tumor Segments ($N$) | Assigned Cohort Status |
| :--- | :---: | :--- | :--- | :---: | :--- |
| **AMBL-001** | L1: 1; L2: 1 | L1: Intraductal papillary lesion (high risk); L2: ADH/ALH (high risk) | 1: Tumor (Mass); 2: Tumor (Mass) | 2 | `EXCLUDED_MULTIPLE_TUMOR_SEGMENTS` |
| **AMBL-003** | L1: 1 | L1: Invasive Ductal Carcinoma (IDC) (malignant) | 1: Tumor (Mass) | 1 | **`EVALUABLE_MALIGNANT`** |
| **AMBL-004** | L1: 0 | L1: Fibroadenoma (benign) | 1: Benign (Mass) | 0 | `EXCLUDED_BENIGN` |
| **AMBL-005** | L1: 1 | L1: IDC + DCIS (malignant) | 1: Tumor (Mass) | 1 | **`EVALUABLE_MALIGNANT`** |
| **AMBL-007** | L1: 1 | L1: IDC + DCIS (malignant) | 1: Tumor (Mass) | 1 | **`EVALUABLE_MALIGNANT`** |
| **AMBL-008** | L1: 1; L2: 0 | L1: IDC + DCIS (malignant); L2: Not seen in iMRI (benign) | 1: Benign (Mass); 2: Tumor (Mass) | 1 | **`EVALUABLE_MALIGNANT`** |
| **AMBL-009** | L1: 0 | L1: Not seen in iMRI (benign) | 1: Benign (Mass) | 0 | `EXCLUDED_BENIGN` |
| **AMBL-010** | L1: 0 | L1: Fibrocystic Changes (FCC) (benign) | 1: Benign (Mass) | 0 | `EXCLUDED_BENIGN` |
| **AMBL-011** | L1: 1; L2: 0 | L1: Invasive Ductal Carcinoma (IDC) (malignant); L2: Not seen in iMRI (benign) | 1: Benign (Mass); 2: Tumor (Mass) | 1 | **`EVALUABLE_MALIGNANT`** |
| **AMBL-013** | L1: 0; L2: 0 | L1: FCC (benign); L2: FCC (benign) | 1: Benign (Mass); 2: Benign (Mass) | 0 | `EXCLUDED_BENIGN` |
| **AMBL-016** | L1: 1; L2: 0; L3: 0 | L1: Infiltrating Lobular Carcinoma (ILC) (malignant); L2: FCC (benign); L3: Fat necrosis (benign) | 1: Benign (Mass); 2: Benign (Mass); 3: Tumor (Mass) | 1 | **`EVALUABLE_MALIGNANT`** |
| **AMBL-018** | L1: 1; L2: 0; L3: 0 | L1: IDC (malignant); L2: Fibroadenoma (benign); L3: Hemangioma (benign) | 1: Benign (Mass); 2: Benign (Mass); 3: Tumor (Mass) | 1 | **`EVALUABLE_MALIGNANT`** |
| **AMBL-022** | L1: 1 | L1: Invasive Ductal Carcinoma (IDC) (malignant) | 1: Tumor (Mass) | 1 | **`EVALUABLE_MALIGNANT`** |
| **AMBL-023** | L1: 0; L2: 0 | L1: Benign (1-yr follow-up); L2: Benign (1-yr follow-up) | 1: Benign (Mass); 2: Benign (Mass) | 0 | `EXCLUDED_BENIGN` |
| **AMBL-024** | L1: 0; L2: 0; L3: 0 | L1: Not seen in iMRI (benign); L2: Not seen in iMRI (benign); L3: Benign | 1: Benign (Mass); 2: Benign (Mass) | 0 | `EXCLUDED_BENIGN` |
| **AMBL-025** | L1: 0 | L1: Benign (1-yr follow-up) | 1: Benign (Mass) | 0 | `EXCLUDED_BENIGN` |
| **AMBL-026** | L1: 0; L2: 0; L3: 0 | L1: FCC (benign); L2: Benign (1-yr follow-up); L3: Benign (1-yr follow-up) | 1: Benign (Mass); 2: Benign (Mass); 3: Benign (Mass) | 0 | `EXCLUDED_BENIGN` |
| **AMBL-028** | L1: 1 | L1: Invasive Ductal Carcinoma (IDC) (malignant) | 1: Tumor (Mass) | 1 | **`EVALUABLE_MALIGNANT`** |
| **AMBL-029** | L1: 1; L2: 1 | L1: Multifocal IDC (malignant); L2: Multifocal IDC (malignant) | 1: Tumor (Mass); 2: Tumor (Mass) | 2 | `EXCLUDED_MULTIPLE_TUMOR_SEGMENTS` |
| **AMBL-031** | L1: 1; L2: 0 | L1: Invasive Ductal Carcinoma (IDC) (malignant); L2: Not seen in iMRI (benign) | 1: Benign (Mass); 2: Tumor (Mass) | 1 | **`EVALUABLE_MALIGNANT`** |
| **AMBL-032** | L1: 1 | L1: Carcinoma - type unspecified (malignant) | 1: Tumor (Mass) | 1 | **`EVALUABLE_MALIGNANT`** |
| **AMBL-033** | L1: 0 | L1: Benign (1-yr follow-up) | 1: Benign (Mass) | 0 | `EXCLUDED_BENIGN` |
| **AMBL-034** | L1: 0; L2: 0 | L1: Fibroadenoma (benign); L2: FCC (benign) | 1: Benign (Mass); 2: Benign (Mass) | 0 | `EXCLUDED_BENIGN` |
| **AMBL-035** | L1: 0 | L1: Fibroadenoma (benign) | 1: Benign (Mass) | 0 | `EXCLUDED_BENIGN` |
| **AMBL-036** | L1: 1 | L1: Metaplastic carcinoma (malignant) | 1: Benign (Mass) [No Tumor segment] | 0 | `EXCLUDED_MALIGNANT_NO_TUMOR_SEGMENT` |
| **AMBL-038** | L1: 1 | L1: IDC + DCIS (malignant) | 1: Tumor (Mass) | 1 | **`EVALUABLE_MALIGNANT`** |
| **AMBL-042** | L1: 0 | L1: Benign (1-yr follow-up) | 1: Benign (Mass) | 0 | `EXCLUDED_BENIGN` |
| **AMBL-043** | L1: 0 | L1: Benign (1-yr follow-up) | 1: Benign (Mass) | 0 | `EXCLUDED_BENIGN` |
| **AMBL-044** | L1: 0 | L1: Not seen in iMRI (benign) | 1: Benign (Mass) | 0 | `EXCLUDED_BENIGN` |
| **AMBL-045** | L1: 0 | L1: Inflammatory cyst (benign) | 1: Benign (Mass) | 0 | `EXCLUDED_BENIGN` |
| **AMBL-046** | L1: 0; L2: 0 | L1: Fibroadenoma (benign); L2: FCC (benign) | 1: Benign (Mass); 2: Benign (Mass) | 0 | `EXCLUDED_BENIGN` |
| **AMBL-047** | L1: 1 | L1: Invasive Ductal Carcinoma (IDC) (malignant) | 1: Tumor (Mass) | 1 | **`EVALUABLE_MALIGNANT`** |
| **AMBL-049** | L1: 0 | L1: Healing fat necrosis (benign) | 1: Benign (Mass) | 0 | `EXCLUDED_BENIGN` |

**Summary**: Exactly **13 evaluable malignant cases** have a confirmed clinical malignancy paired with exactly one distinct `"Tumor"` segment in DICOM-SEG:
`['AMBL-003', 'AMBL-005', 'AMBL-007', 'AMBL-008', 'AMBL-011', 'AMBL-016', 'AMBL-018', 'AMBL-022', 'AMBL-028', 'AMBL-031', 'AMBL-032', 'AMBL-038', 'AMBL-047']`.

---

## 3. Dynamic Contrast-Enhanced (DCE) Temporal Frame Mapping

- **Acquisition**: GE Signa HDxt 1.5T 4D acquisition (`Registered AX Sen Vibrant MultiPhase`).
- **Temporal Order**: In strict temporal order from pre-contrast to late wash-out:
  - $P_0$ = Pre-contrast baseline unenhanced volume (`T1DCE__0.nii`)
  - $P_1$ = 1st dynamic post-contrast volume (`T1DCE__1.nii`, peak wash-in enhancement)
  - $P_2$ = 2nd dynamic post-contrast volume (`T1DCE__2.nii`)
  - $P_3$ = 3rd dynamic post-contrast volume (`T1DCE__3.nii`)
  - $P_4$ = 4th dynamic post-contrast volume (`T1DCE__4.nii`, delayed clearance)
- **Symlink Directory**: Standardized at `<data_root>/ambl_clean/{case_id}/` containing `P0.nii`–`P4.nii` and `GT.nii`.

---

## 4. Preprocessing & Inference Specifications

- **Orientation**: Reoriented to canonical `RAS` (`nib.as_closest_canonical` / MONAI `Orientationd`).
- **Spatial Grid**: Resampled to $0.8 \times 0.8 \times 1.0\text{ mm}$ voxel spacing (trilinear interpolation for image channels; nearest-neighbor for ground truth mask).
- **Intensity Normalization**: Per-channel z-score computed over non-zero voxels ($I' = (I - \mu_{nz}) / \sigma_{nz}$).
- **Multi-channel Derivations**: Subtraction channels $P_1 - P_0$ and $P_2 - P_0$ computed on normalized volumes.
- **Sliding-Window Inference**: Patch ROI size $128 \times 128 \times 64$, overlap 0.50, batch size 1.
- **Thresholding**: Probability threshold $\tau = 0.50$.
- **Post-processing**: None (primary full-volume evaluation).

---

## 5. Model Configurations Evaluated

All model checkpoints are loaded strictly at **Epoch 500** (`assert stored_epoch == 500`), with the sole exception of the historical baseline model (`best_modelfinal55.pth`).

1. **Configuration (a) — Historical 2-Channel Baseline (`best_modelfinal55.pth`)**:
   - Checkpoint: `<models_root>/best_modelfinal55.pth`
   - Input: 2 channels $[P_0, P_1]$ (pre-contrast + early post-contrast).
2. **Configuration (b) — BEAT Net 5-Channel Ensemble (Reference Model)**:
   - Checkpoints: 5 folds $\times$ `<project_root>/results/5fold_5chan/fold_{0..4}/final_model.pt` (Epoch 500).
   - Ensemble Strategy: Mean predicted probability across all 5 folds.
   - Input: 5 channels $[P_0, P_1, P_2, P_1 - P_0, P_2 - P_0]$.
3. **Configuration (c) — Phase-Randomized 2-Channel Network**:
   - Checkpoint: `<project_root>/results/phase_randomized_2chan_seed0/final_model.pt` (Epoch 500).
   - Inputs evaluated:
     - `fixed_P0_P1`: $[P_0, P_1]$
     - `fixed_P0_P2`: $[P_0, P_2]$
     - `fixed_P0_P3`: $[P_0, P_3]$
     - `fixed_P0_P4`: $[P_0, P_4]$
     - `apcps_q95`: $[P_0, P_{k^*}]$ with $k^*$ chosen via 95th-percentile subtraction intensity over $k \in \{1..4\}$.
     - `early_peak`: $[P_0, P_{k^*}]$ with $k^*$ chosen as earliest $k \in \{1..4\}$ where top-5% mean enhancement is $\ge 0.90 \times \max_k$.
4. **Configuration (d) — Conventional 2-Channel Fold 4 Model**:
   - Checkpoint: `<project_root>/results/5fold_2chan/fold_4/latest_checkpoint.pt` (Epoch 500).
   - Input: 2 channels $[P_0, P_2]$.
5. **Configuration (e) — nnU-Net v2 Fold 0 Baseline**:
   - Checkpoint: `<data_root>/mama_mia_dataset/nnUNet_results/Dataset501_MAMAMIA/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth`.
   - Input: 3 raw channels $[P_0, P_1, P_2]$.

---

## 6. Metrics & Statistical Testing

- **Segmentation Metrics**: Dice Similarity Coefficient (DSC %), Intersection over Union (IoU %), Precision (PPV %), Recall (Sensitivity %), 95% Hausdorff Distance (HD95 mm), and Average Symmetric Surface Distance (ASSD mm).
- **Statistical Testing (Amendment 2 Invariants)**:
  - Reference for hypothesis testing: Configuration (b) BEAT Net 5-Channel Ensemble.
  - Paired Wilcoxon signed-rank test using Pratt/Wilcox zero-handling (`zero_method="wilcox"`).
  - Explicit reporting of $n_{\text{nonzero}}$, sum of ranks $W$, asymptotic normal $z$-statistic, two-sided $p$-value, and effect size $r = |z| / \sqrt{n_{\text{nonzero}}}$.
  - Bootstrap 95% Confidence Interval of the paired mean difference (10,000 resamples, seed 42).
  - 1.0 Dice point non-inferiority test (established iff one-sided 95% bootstrap lower bound $> -1.0$).
  - Holm-Bonferroni correction within the candidate comparison family.
  - Within Configuration (c): Pairwise statistical testing of `early_peak` vs `apcps_q95` and vs `fixed_P0_P1`.

---

## 7. Deliverables & Output Schema

- **Per-Case CSVs**: `results/ambl/ambl_per_case_{config}.csv`
- **Cohort Summary**: `results/ambl/ambl_summary.json`
- **Statistical Comparisons**: `results/ambl/ambl_stats.json`
- **Frame Selection Counts**: `results/ambl/ambl_frame_selection_counts.json`
- **Text-Only Bundle**: `~/for_claude/beatnet_bundle_ambl.tar.gz`
