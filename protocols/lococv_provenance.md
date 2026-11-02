# LoCoCV (Leave-One-Center-Out Cross Validation) Provenance Report

## 1. Code Traceability & Script Identification
- **Historical Implementation Script:** `<research_repo>/ganeevthegoat.py`
- **Model Architecture:** 3D Attention U-Net with ResNet-50 3D Encoder and scSE attention decoder.
- **Input Channels:** 3 channels ($I_0, I_1, I_2$).

## 2. Checkpoint Selection Rule & Initialization
1. **Pretrained Initialization:**
   - **Code Lines (`ganeevthegoat.py` line 92-96):**
     ```python
     checkpoint = torch.load('<home>/best_modelfinal55.pth', map_location='cpu')
     model.load_state_dict(checkpoint, strict=False)
     ```
   - Weights were initialized from `best_modelfinal55.pth` (pretrained 3D ResNet-50 encoder).

2. **Checkpoint Selection Rule:**
   - **Code Lines (`ganeevthegoat.py` lines 145-155):**
     ```python
     if val_dice > best_val_dice:
         best_val_dice = val_dice
         torch.save(model.state_dict(), f"best_model_center_{held_out_center}.pth")
     ```
   - Saved the model checkpoint that achieved the highest validation Dice on the held-out center during training.
   - Evaluated strictly using validation set early-stopping criteria (`best_model.pth`), NOT fixed epoch-500 final weights.

## 3. Preservation Status on Cluster Filesystem
- **Checkpoints Status:** No trained checkpoint files (`best_model_center_*.pth` or `final_model_center_*.pth`) are preserved on `<home>/` or `<project_root>/`.
- **Per-Case Metrics Status:** No per-case prediction CSV or JSON logs exist for LoCoCV.
- **Reporting Status:** In accordance with Ground Rule 0, since raw checkpoints and per-case predictions cannot be recovered from disk, LoCoCV cannot be reproduced with epoch-500 traceability and is reported as unpreserved historical exploratory work.
