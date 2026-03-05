# Brain Tumor Segmentation with U-Net — Complete Technical Report

## 1) Project Overview

This project builds a **binary semantic segmentation** model for brain tumor MRI images using **PyTorch Lightning** and a **U-Net** architecture.  
The main goal is to predict a pixel-wise tumor mask from an MRI image.

### Core objective
- **Input:** RGB MRI image (`3 x 256 x 256` after preprocessing)
- **Output:** 1-channel tumor probability/logit map (`1 x 256 x 256`)
- **Task type:** Binary segmentation (tumor vs background)

### Main artifacts in this workspace
- `UNet.ipynb`: Notebook workflow with training and experiments.
- `unet.py`: Script-export version of notebook containing two training attempts.
- `brain_tumor_unet_final.pth`: Saved final trained model weights.

---

## 2) High-Level Pipeline (End-to-End)

1. Install dependencies (notably `pytorch-lightning`)
2. Download dataset via `kagglehub`
3. Configure data paths and collect image/mask filenames
4. Split data into train/validation/test
5. Build custom `Dataset` and `DataLoader`
6. Define segmentation metrics (Dice, IoU)
7. Define U-Net model (`LightningModule`)
8. Train with combined loss: `BCEWithLogits + (1 - Dice)`
9. Validate each epoch and track metrics
10. Test on held-out test split
11. Visualize predictions
12. Save final model weights

---

## 3) Code Structure Notes: Two Implementations in One File

`unet.py` contains **two full runs**:

## First attempt (earlier section)
- Smaller U-Net-like model with lighter channel sizes (`16, 32, 64, 256` bottleneck)
- Basic preprocessing
- Adam optimizer only (no LR scheduler)
- Simpler logging setup

## Second attempt (`"# 2nd try"` section, more complete)
- Cleaner, more modular U-Net (`DoubleConv`, `Down`, `Up`, `OutConv`)
- Better preprocessing and augmentation
- Learning-rate scheduler (`ReduceLROnPlateau`)
- Better callbacks (early stop + top-k checkpoints)
- Mixed precision training (`precision="16-mixed"`)
- Improved logging and visualization

The second attempt is the **final production-style pipeline** and the best representation of the project architecture.

---

## 4) Dataset and Data Preparation

## Dataset source
- Downloaded using:
  - `kagglehub.dataset_download("nikhilroxtomar/brain-tumor-segmentation")`

## Path management
- Colab-oriented paths are used in code (e.g., `/content/...`, `/root/.cache/...`).
- Dataset is copied from KaggleHub cache to a working directory.

## File collection
- Image and mask files are read using `os.listdir` + `sorted`.
- In second attempt, file extensions are filtered to image types:
  - `.png`, `.jpg`, `.jpeg`

## Train/Validation/Test split
Two-stage split strategy:

1. `test_size=0.1` → 10% test set
2. Remaining 90% split with `test_size=1/9` for validation

This gives approximately:
- **Train:** 80%
- **Validation:** 10%
- **Test:** 10%

`random_state=42` ensures reproducibility of splits.

---

## 5) Preprocessing and Augmentation

## Image preprocessing
- Read image with OpenCV
- Convert BGR to RGB (second attempt)
- Resize to `256 x 256`
- Normalize pixel values to `[0, 1]`
- Rearrange shape from HWC → CHW for PyTorch

## Mask preprocessing
- Read grayscale mask
- Resize using nearest-neighbor interpolation (`INTER_NEAREST`) to preserve labels
- Normalize to `[0, 1]`
- Binarize (`mask > 0.5`)
- Add channel dimension (`1 x H x W`)

## Training-time augmentation (second attempt)
Applied only if `augment=True`:
- Random horizontal flip
- Random rotation by multiples of 90°
- Random brightness scaling for image only

These augmentations improve generalization by exposing the model to spatial and intensity variations.

---

## 6) Dataloader Design

Three dataloaders are created:
- `train_loader`: shuffled, augmented data
- `val_loader`: non-shuffled, no augmentation
- `test_loader`: non-shuffled, no augmentation

Important parameters:
- `batch_size = 16`
- `num_workers = 2`
- `pin_memory=True` (second attempt)

`pin_memory=True` can speed host-to-GPU transfers.

---

## 7) U-Net Architecture: Detailed Explanation

The final model follows a classic encoder-decoder U-Net with skip connections.

## Building blocks

### 7.1 `DoubleConv`
Each block performs:
1. Conv2d (3x3, padding=1)
2. BatchNorm2d
3. ReLU
4. Conv2d (3x3, padding=1)
5. BatchNorm2d
6. ReLU

This improves representational power versus a single conv layer.

### 7.2 `Down`
- `MaxPool2d(2)` for spatial downsampling
- Followed by `DoubleConv`

This halves spatial resolution and increases features.

### 7.3 `Up`
- Upsampling via `ConvTranspose2d` (or bilinear option, not used)
- Concatenate with corresponding encoder feature map (skip connection)
- Apply `DoubleConv`

Skip connections recover fine spatial details lost in downsampling.

### 7.4 `OutConv`
- Final `1x1` convolution maps features to output channel(s)
- For binary segmentation: output channels = 1

---

## 8) Tensor Shape Flow (Final Model)

With `features = [64, 128, 256, 512]` and input `3x256x256`:

1. `inc`: `3 -> 64`, spatial `256x256`
2. `down1`: `64 -> 128`, spatial `128x128`
3. `down2`: `128 -> 256`, spatial `64x64`
4. `down3`: `256 -> 512`, spatial `32x32`
5. `down4` bottleneck: `512 -> 1024`, spatial `16x16`
6. `up1`: combine bottleneck with encoder level 4 -> `512`, `32x32`
7. `up2`: -> `256`, `64x64`
8. `up3`: -> `128`, `128x128`
9. `up4`: -> `64`, `256x256`
10. `outc`: `64 -> 1`, final logits `1x256x256`

---

## 9) Loss Function

The code uses a **hybrid loss**:

\[
\mathcal{L}_{total} = \mathcal{L}_{BCEWithLogits} + (1 - Dice)
\]

### Why this combination?
- **BCEWithLogitsLoss** handles pixel-wise binary classification robustly and includes sigmoid internally.
- **Dice term** optimizes overlap quality and helps with class imbalance (tumor regions often small).

This is a common segmentation strategy balancing local pixel accuracy and global region overlap.

---

## 10) Evaluation Metrics

## 10.1 Dice Coefficient
Implemented by thresholding sigmoid outputs at 0.5 and computing:

\[
Dice = \frac{2|P \cap G| + \epsilon}{|P| + |G| + \epsilon}
\]

Where:
- `P` = predicted positive pixels
- `G` = ground-truth positive pixels
- `\epsilon = 1e-6` for numerical stability

Higher Dice indicates better overlap (max = 1.0).

## 10.2 IoU (Jaccard Index)

\[
IoU = \frac{|P \cap G| + \epsilon}{|P \cup G| + \epsilon}
\]

IoU is stricter than Dice and is widely used for segmentation benchmarking.

---

## 11) Training and Validation Logic (LightningModule)

## `training_step`
- Forward pass to get logits
- Compute BCE loss
- Compute Dice
- Total loss = BCE + (1 - Dice)
- Log:
  - `train_loss`
  - `train_dice`
  - `train_bce`

## `validation_step`
- Same core computations
- Additional metric: `val_iou`
- Log:
  - `val_loss` (monitored by callbacks/scheduler)
  - `val_dice`
  - `val_iou`

## `test_step`
- Computes and logs:
  - `test_dice`
  - `test_iou`

---

## 12) Optimizer and Scheduler

## Optimizer
- `Adam` with learning rate `LR`

## Scheduler (second attempt)
- `ReduceLROnPlateau`
- Monitors `val_loss`
- If validation loss plateaus, LR is reduced by factor `0.5` after `5` patience epochs

This allows coarse-to-fine optimization and can improve convergence stability.

---

## 13) Training Configuration

Key config in second attempt:
- `IMG_SIZE = 256`
- `BATCH_SIZE = 16`
- `NUM_WORKERS = 2`
- `MAX_EPOCHS = 100`
- `LR = 1e-4`

Trainer settings:
- `accelerator="gpu"`
- `devices=1`
- `precision="16-mixed"`
- `log_every_n_steps=5`

Mixed precision reduces GPU memory usage and can speed training.

---

## 14) Callbacks and Checkpointing

## EarlyStopping
- Monitor: `val_loss`
- Mode: `min`
- Patience: `15`

Stops training if no improvement for many epochs, preventing overfitting and wasted compute.

## ModelCheckpoint
- Monitor: `val_loss`
- Mode: `min`
- Keep top 3 best checkpoints (`save_top_k=3`)
- Filename includes epoch and validation loss

Best checkpoint is used during test: `ckpt_path="best"`.

---

## 15) Inference and Visualization

After training/testing:
- Model is set to eval mode
- Predictions are generated with sigmoid + threshold 0.5
- A grid plot is created for each sample:
  1. Input MRI
  2. Ground-truth mask
  3. Predicted mask

Outputs saved as `predictions.png`.

---

## 16) Model Saving

Final weights are saved with:
- `torch.save(model.state_dict(), "brain_tumor_unet_final.pth")`

This stores parameters only; architecture code must be available to reload.

---

## 17) Code Quality / Practical Notes

## Strengths
- Uses Lightning for clean train/val/test structure
- Hybrid loss suitable for segmentation imbalance
- Includes both Dice and IoU metrics
- Uses checkpointing and early stopping
- Includes visual qualitative evaluation

## Caveats
- Paths are Colab-specific (`/content/...`), so local execution needs path edits
- Data pairing relies on sorted filenames; naming consistency is required
- Threshold at 0.5 is fixed (can be tuned)
- No seed-setting for full deterministic training reproducibility

---

## 18) Suggested Reproducible Run Order

1. Install dependencies (`torch`, `pytorch-lightning`, `opencv-python`, `scikit-learn`, `matplotlib`, `kagglehub`)
2. Download/copy dataset
3. Verify `image_dir` and `mask_dir`
4. Run preprocessing + dataloaders
5. Train with trainer callbacks
6. Test best checkpoint
7. Visualize and save final model

---

## 19) Summary

This project implements a complete tumor-segmentation workflow centered around U-Net and PyTorch Lightning. The final (second) implementation combines robust preprocessing, practical augmentation, hybrid segmentation loss, overlap-based metrics, and stable training controls (early stopping, checkpointing, LR scheduling).  

In short, it is a solid end-to-end medical image segmentation pipeline with both quantitative (Dice/IoU) and qualitative (visual plots) evaluation, and a saved deployable model checkpoint.
