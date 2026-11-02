# BEAT Net: Phase-randomised training and image-based frame selection for zero-shot breast tumour segmentation across DCE-MRI protocols

Code for the manuscript of the same title (Nain, Tuteja, Juneja, Sachdeva, Kumar; under review).
Note: medical-imaging BEAT Net, not the audio BeatNet model.

## Overview
- 3D U-Net, ResNet-50 encoder, scSE decoder (MONAI FlexibleUNet); 5-channel input [P0, P1, P2, P1−P0, P2−P0], each frame z-scored separately.
- Phase-randomised training: two-channel input [P0, Pt], t ∈ {1,2} drawn per patch.
- Image-only frame selection at a new centre: APCPS (highest 95th-percentile enhancement) and an exploratory early-peak rule (earliest frame within 90% of the maximum top-5% enhancement).
- Trained on the official MAMA-MIA training partition (1,200 cases); zero-shot tests on YCHC (99) and AMBL (13).

## Key results
- YCHC, fixed model: Dice 75.93% with P1 falling to 62.68% with P5.
- APCPS: −7.09 Dice points vs fixed P1 on YCHC; early-peak: 75.26%.
- BEAT Net vs nnU-Net (fold 0): +5.26 points on YCHC, +19.15 on AMBL.

## Important note on the stage-1 checkpoint
The pre-trained checkpoint used to initialise BEAT Net and the two-channel models predates this study, and its training cases could not be verified. MAMA-MIA test results of initialised models are therefore development results. The primary evidence is the zero-shot YCHC and AMBL evaluation and models trained without this checkpoint (nnU-Net, SegResNet, randomly initialised BEAT Net). The checkpoint is not distributed.

## Repository layout
data/ preprocessing, 5-fold split, case lists and frame mappings
models/ BEAT Net and attention variants
training/ five-fold, phase-randomised, attention and input-ablation training
frame_selection/ APCPS, whole-breast and early-peak rules
evaluation/ MMC, YCHC and AMBL evaluation, ensembles, timing benchmark
statistics/ paired Wilcoxon, bootstrap CIs, Holm correction
nnunet/ nnU-Net v2 preparation and run scripts
protocols/ study protocol and pre-registration documents
eda/ per-case MMC Dice at probability 0.5 vs the legacy threshold

## Requirements
Python 3.10, PyTorch, MONAI, nnU-Net v2; see requirements.txt. Tested on NVIDIA A100/H100.

## Data (not included)
MAMA-MIA and AMBL: The Cancer Imaging Archive. YCHC: Zenodo doi:10.5281/zenodo.8068383.
Download under their licences and set paths via the script arguments.

## Usage
Preprocess cached volumes:
```bash
python data/preprocess_cache.py --data-root /path/to/MAMA-MIA --out-dir data/preprocessed_mmap
```

Train five-fold model:
```bash
python training/train_5fold.py --fold 0 --in-channels 5 --data-root /path/to/MAMA-MIA --out-dir results/5fold_5chan/fold_0
```

Train phase-randomised model:
```bash
python training/train_phase_randomized_2chan.py --mmap-dir data/preprocessed_mmap --out-dir results/phase_randomized_2chan_seed0
```

Evaluate on MMC test cohort:
```bash
python evaluation/evaluate_single_fold_test.py --mode standard --fold 0 --ckpt results/5fold_5chan/fold_0/final_model.pt --data-dir data/preprocessed_mmap
```

Evaluate zero-shot on YCHC and AMBL:
```bash
python evaluation/run_step2c_ychc_zeroshot.py --preproc-dir data/ychc_preprocessed --results-dir results/ychc_zeroshot
python evaluation/run_ambl_evaluation.py --preproc-dir data/ambl_preprocessed --out-dir results/ambl
```

Frame selection evaluation:
```bash
python frame_selection/run_apcps_correction_eval.py --ychc-preproc data/ychc_preprocessed --ambl-preproc data/ambl_preprocessed
```

Statistical analysis:
```bash
python statistics/compute_section3_stats_e500.py --results-dir results
python statistics/synthesize_5fold_statistics.py --results-dir results
```

## Evaluation settings
Patches/windows 128×128×64, overlap 0.5, sigmoid probability ≥ 0.5, epoch-500 weights, metrics Dice, precision, recall, HD95 on a 0.8×0.8×1.0 mm grid.
Note: the input-ablation and randomly initialised models in the paper were scored with a legacy threshold equivalent to probability 0.62; the current scripts use probability 0.5.

## Citation
```bibtex
@article{nain2026beatnet,
  title={BEAT Net: Phase-randomised training and image-based frame selection for zero-shot breast tumour segmentation across DCE-MRI protocols},
  author={Nain, Sandeep and Tuteja, ... and Juneja, Keshav and Sachdeva, ... and Kumar, ...},
  journal={Under review},
  year={2026}
}
```

## License
MIT
