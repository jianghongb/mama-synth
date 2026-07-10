# MAMA-SYNTH Workshop Paper — Outline

## Target
- MICCAI Workshop Paper (Springer LNCS template)
- 8 pages + 2 pages references
- Double-blind (anonymized)
- Deadline: July 15, 2026

---

## Title (Draft)

**Tumor-Aware Conditional Pix2PixHD with Per-Image Normalization for Virtual Contrast Enhancement in Breast MRI**

Alternative:
- "Per-Image Normalized Conditional GAN for Synthesizing Post-Contrast Breast MRI from Pre-Contrast Acquisitions"

---

## Structure

### 1. Introduction (~1 page)

- **Problem**: DCE-MRI requires gadolinium contrast agents → safety concerns (NSF, environmental), accessibility barriers
- **Challenge**: MAMA-SYNTH 2026 — synthesize post-contrast subtraction images from pre-contrast T1w
- **Difficulty**: Large intensity variation across scanners/institutions; tumor enhancement patterns are highly heterogeneous
- **Our approach**: Conditional Pix2PixHD with three key innovations:
  1. Per-image z-score normalization (scanner-invariant)
  2. Predicted tumor mask as conditional input (tumor-aware synthesis)
  3. Multi-scale subtraction consistency (MSSC) loss

### 2. Method (~3 pages)

#### 2.1 Overview / Pipeline Figure

```
Pre-contrast MRI
  ├─→ Breast Segmentation (nnUNet 2D) → breast mask
  ├─→ Tumor Detection (nnUNet 2D, pre-contrast) → predicted tumor mask
  │
  ├─→ Per-image z-score normalization (breast foreground stats)
  │
  └─→ Conditional Generator [3ch: norm_pre, breast_mask, tumor_mask]
        → Pix2PixHD (GlobalGenerator, residual mode)
        → Test-time augmentation (horizontal flip, 2×)
        → De-normalize
        → Soft composite (Gaussian σ=5) + background offset
        → Synthetic post-contrast
```

#### 2.2 Per-Image Z-Score Normalization

- Motivation: Cross-scanner intensity variation dominates MSE error
- Method: Compute mean/std from breast foreground per image → normalize to canonical space
- De-normalization: output × σ + μ → back to original intensity space
- Comparison: LPIPS improved 25% vs global normalization (0.157 → 0.117)

#### 2.3 Tumor-Aware Conditional Input

- Problem: Standard 1-channel GAN doesn't know WHERE to enhance
- Solution: Predict tumor location from pre-contrast using nnUNet → feed as 3rd channel
- Generator explicitly receives spatial prior about enhancement target
- Dice improved from 0.565 (1ch) to 0.624 (3ch conditional)

#### 2.4 Generator Architecture

- Pix2PixHD GlobalGenerator (ngf=64, 4× downsampling, 12 ResBlocks)
- Residual mode: output = input + Δ (preserves anatomy)
- Multi-scale PatchGAN discriminator (2 scales, 3 layers each)
- ~239M parameters

#### 2.5 Training Strategy

- **Data**: Multi-slice DCE-MRI from MAMA-MIA dataset (~7000 training slices)
  - Sources: DUKE, ISPY2, YUNNAN, LA-Breast
  - Peak enhancement GT with ±2 neighbor slices
- **Loss function**:
  - GAN loss (λ=1)
  - Feature matching (λ=10)
  - VGG perceptual (λ=10)
  - Multi-Scale Subtraction Consistency — MSSC (λ=100)
  - Tumor-weighted L1 (λ=20, applied within tumor mask)
- **Augmentation**: intensity scaling [0.8, 1.2], Gaussian noise, horizontal flip
- **Training**: 200 epochs, batch=16, lr=3e-4, Adam, InstanceNorm
- **Breast mask loss**: Loss computed only within breast region (Dataset930 BreastDivider)

#### 2.6 Inference Pipeline

- Breast segmentation: nnUNet 2D (distilled from BreastDivider 3D)
- Tumor detection: nnUNet 2D trained on pre-contrast (Dataset940)
- Horizontal flip TTA (2× average)
- Soft Gaussian composite (σ=5) with background offset (+0.17)
- Total inference: ~3s/case on NVIDIA T4

### 3. Experiments (~2 pages)

#### 3.1 Dataset

| Split | Cases | Sources |
|-------|-------|---------|
| Train | ~7042 slices | DUKE, ISPY2, YUNNAN, LA-Breast |
| Test (internal) | 299 slices | DUKE (55), ISPY2 (219), YUNNAN (25) |
| Test (GC) | 300 cases | Radboud (200), Fleming (100) |

#### 3.2 Ablation Study

| Configuration | MSE ↓ | LPIPS ↓ | SSIM ↑ | Dice ↑ | HD95 ↓ |
|---------------|-------|---------|--------|--------|--------|
| Baseline (global norm, 1ch) | 1.101 | 0.157 | 0.394 | 0.493 | 144.9 |
| + Per-image norm | 1.236 | 0.117 | 0.476 | 0.539 | 125.6 |
| + Tumor weight ×2, MSSC ×2 | 1.193 | 0.118 | 0.476 | 0.565 | 112.1 |
| + 3ch conditional (tumor mask) | 1.160 | 0.117 | 0.492 | 0.614 | 76.0 |
| + Soft composite + hflip TTA | **1.127** | 0.123 | **0.509** | **0.624** | **69.4** |

Key findings:
- Per-image norm: LPIPS -25%, SSIM +21% (scanner invariance)
- Conditional tumor mask: Dice +9%, HD95 -32% (explicit spatial prior)
- MSSC ×2: HD95 -10% (subtraction consistency)

#### 3.3 Comparison with Other Approaches (if GC results available)

#### 3.4 Failure Analysis

- MSE dominated by top 10% outlier cases (contribute 40% of total MSE)
- Main failure modes: under-enhancement of large tumors, intensity shift in high-enhancement cases
- Per-dataset: YUNNAN (MSE 0.036) >> ISPY2 (1.23) >> DUKE (1.57)

### 4. Discussion (~0.5 page)

- Per-image normalization is the single most impactful design choice
- Predicted tumor mask as conditional input significantly improves downstream segmentation metrics
- Remaining MSE gap vs top methods likely due to: (1) limited training data, (2) extreme enhancement outliers
- Limitation: tumor detection accuracy on pre-contrast limits conditional GAN performance
- Future: ensemble multiple models, diffusion-based refinement

### 5. Conclusion (~0.5 page)

- Three-stage pipeline: segmentation → normalization → conditional synthesis
- Per-image norm eliminates scanner bias; tumor mask enables targeted enhancement
- Competitive on tumor-related metrics (SSIM, Dice, HD95) while maintaining perceptual quality

---

## Figures (Draft list)

1. **Fig 1**: Pipeline overview (block diagram)
2. **Fig 2**: Per-image normalization illustration (before/after, cross-scanner examples)
3. **Fig 3**: Qualitative results — good cases (3-4 examples: pre, GT, pred, diff)
4. **Fig 4**: Ablation — bar chart or radar plot of metrics across versions
5. (Optional) **Fig 5**: Failure cases with analysis

---

## Key References

- Pix2PixHD (Wang et al., 2018)
- nnU-Net (Isensee et al., 2021)
- MAMA-MIA dataset (challenge organizers)
- BreastDivider (Rokuss et al., 2025)
- medigan (Obi et al., 2023)
- MSSC loss — multi-scale subtraction consistency (ours, novel)
- Per-image z-score normalization strategy (ours, novel)

---

## Writing Timeline

| Date | Task |
|------|------|
| Jul 10-11 | Method section draft + figures |
| Jul 12-13 | Experiments + results tables |
| Jul 14 | Introduction + discussion + polish |
| Jul 15 | Final check + submit to OpenReview |
