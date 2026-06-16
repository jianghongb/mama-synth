# MAMA-SYNTH Pipeline 流程图

## 1. 数据准备阶段

```
┌─────────────────────────────────────────────────────────────────┐
│                    3D → 2D 预处理 (mask_and_preprocess.py)        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  images/<patient>/<patient>_0000.nii.gz  (3D multi-phase DCE)   │
│  segmentations/<patient>.nii.gz          (3D tumour seg)        │
│       │                                                         │
│       ▼                                                         │
│  ┌────────────────────────────────┐                             │
│  │  Exp4x Dataset932 (4ch, 3D)   │                             │
│  │  输入: P0, P1, d_early, d_late │                             │
│  │  输出: 3 labels (breast/FGT/tumor)                           │
│  └────────────────────────────────┘                             │
│       │                                                         │
│       ▼                                                         │
│  breast_mask_3d (binary: label ∈ {1,2,3})                       │
│       │                                                         │
│       ▼                                                         │
│  所有 phase × breast_mask → 去胸壁                               │
│       │                                                         │
│       ▼                                                         │
│  选 peak phase → 选最大 tumour slice → z-score 归一化            │
│       │                                                         │
│       ▼                                                         │
│  output/mha/                                                    │
│    ├── input/         (2D pre-contrast, masked, z-score)        │
│    ├── ground_truth/  (2D peak-enhancement, masked, z-score)    │
│    ├── mask/          (2D tumour mask)                           │
│    └── breast_mask/   (2D breast mask)                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 2. 训练阶段 (Berzelius GPU)

```
┌─────────────────────────────────────────────────────────────────┐
│                        模型训练                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  pre, gt, tumor_mask, breast_mask = load_batch()                │
│                                                                 │
│  pre ──→ resize 512 ──→ ┌──────────────────┐                   │
│                          │  Pix2PixHD       │                   │
│                          │  GlobalGenerator  │──→ fake           │
│                          │  output = pre + Δ │                   │
│                          └──────────────────┘                   │
│                                                                 │
│  ⚠️ Loss 全部限制在 breast_mask 区域内:                           │
│                                                                 │
│  loss_GAN   = GAN(fake × bmask, gt × bmask)                    │
│  loss_feat  = L1(D_feat(fake × bmask), D_feat(gt × bmask))     │
│  loss_VGG   = VGG(fake × bmask, gt × bmask)                    │
│  loss_MSEC  = MSEC((fake-pre) × bmask, (gt-pre) × bmask)       │
│  loss_tumor = L1(fake × tmask, gt × tmask)                     │
│                                                                 │
│  loss_G = 1×GAN + 10×feat + 10×VGG + 50×MSEC + 10×tumor       │
│                                                                 │
│  参数: 182.4M, ngf=64, 4×downsample, 9×ResBlock                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                     weights/latest_net_G.pth (696 MB)
```

## 3. 推理阶段 (Docker 容器, NVIDIA T4)

```
┌─────────────────────────────────────────────────────────────────┐
│  GC 输入                                                         │
│  /input/images/pre-contrast-dce-mri-slice-breast/<uuid>.mha     │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  Docker 容器 (inference.py)                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ① 读取 pre.mha (任意分辨率, z-score float32)                    │
│       │                                                         │
│       ▼                                                         │
│  ② resize → 512×512                                             │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────────────┐                                       │
│  │  Pix2PixHD Generator  │                                      │
│  │  synthetic = pre + Δ  │                                      │
│  └──────────────────────┘                                       │
│       │                                                         │
│       ▼                                                         │
│  ③ resize → 原始分辨率                                           │
│       │                                                         │
│       ▼                                                         │
│  ④ 写出 output.mha (保留原始 spacing/origin/direction)           │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  模型大小: ~1.5 GB (Generator 696MB + base)                      │
│  推理时间: < 10 min/case on T4                                   │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  /output/images/synthetic-contrast-dce-mri-slice-breast/output.mha
└─────────────────────────────────────────────────────────────────┘
```

## 4. 评估阶段 (GC 平台自动执行)

```
┌─────────────────────────────────────────────────────────────────┐
│  GC 评估系统 (8 个指标, Mean Position 排名)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─ Image Fidelity (1/4) ────────────────────────────────────┐  │
│  │  output.mha ──┐                                           │  │
│  │               ├──→ MSE ↓, LPIPS ↓                         │  │
│  │  GT post.mha ─┘                                           │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─ ROI Realism (1/4) ──────────────────────────────────────┐   │
│  │  output.mha ──┐                                          │   │
│  │  GT post.mha ─┼──→ SSIM_tumor ↑, FRD ↓                  │   │
│  │  tumor_mask ──┘                                          │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─ Classification (1/4) ───────────────────────────────────┐   │
│  │  output.mha ──→ radiomics ──→ AUROC_contrast ↑           │   │
│  │  output.mha ──→ tumor ROI ──→ AUROC_tumor ↑              │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─ Segmentation (1/4) ─────────────────────────────────────┐   │
│  │  output.mha ──→ nnUNet ──→ pred_mask                     │   │
│  │  tumor_mask (GT) ──────────→ Dice ↑, HD95 ↓              │   │
│  │                                                           │   │
│  │  ⚠️ Breast masking 对此最关键:                             │   │
│  │    - 防止胸壁假增强 → 减少 false positive                  │   │
│  │    - v5→v9: Dice +47%, HD95 -47%                          │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                 │
│  最终排名 = 8个指标各自排名的算术平均                               │
└─────────────────────────────────────────────────────────────────┘
```

## 5. Breast Masking 方案对比

| 阶段 | 模型 | 输入 | 说明 |
|------|------|------|------|
| **数据准备** (3D) | Exp4x Dataset932 | 4ch: P0, P1, d_early, d_late | 高质量, 需要多 phase |
| **推理** (2D) | Dataset910 BreastSegNet | 1ch: T1 pre-contrast | 只需 pre, 比赛可用 |

## 6. 版本对比摘要

| 版本 | 数据 | Breast Mask | Intensity Aug | 最佳场景 |
|------|------|:-:|:-:|------|
| v5 | 1074 | ❌ | ❌ | baseline |
| v8 | 1074 | ❌ | ✅ | **image fidelity** (MSE, LPIPS) |
| v9 | 1074 | ✅ | ❌ | **tumor realism** (SSIM, Dice) |
| v10 | 2811 | ✅ | ❌ | 最大数据量 |
| v11 | 2811 | ✅ | ✅ | v10 + aug (待评估) |

## 关键点

- **训练时**: 4ch 3D 模型 (Exp4x) 生成高质量 breast mask，loss 限制在 mask 内
- **推理时**: 纯 Generator 一次前向，无需 breast mask 模型
- **评估时**: GC 平台自动执行，训练阶段的 breast mask 已让模型学会只增强乳房区域
