# Pix2PixHD for MAMA-SYNTH — Handoff

---

## Executive Summary (2026-07-10)

### Best Models (data_multislice_v3 test, 299 cases)

| Rank | Version | MSE ↓ | LPIPS ↓ | SSIM ↑ | FRD ↓ | Dice ↑ | HD95 ↓ | Key Change |
|:---:|---------|:-----:|:-------:|:------:|:-----:|:------:|:------:|------------|
| 🥇 | **v32 Final (submission)** | **0.970** | **0.116** | **0.493** | **26.88** | **0.567** | 108.5 | D930 breast + D910 heart offset + hflip TTA |
| 🥈 | v32 + hflip TTA (no heart) | 1.186 | 0.119 | 0.493 | 26.88 | 0.574 | 111.6 | TTA only |
| 🥉 | v31_auroc | 1.193 | 0.118 | 0.476 | 28.31 | 0.565 | 112.1 | tumor_weight×2 + MSSC×2 |
| 4 | v32_vflip (no postproc) | 1.215 | 0.117 | 0.475 | 28.48 | 0.554 | **104.9** | HD95 最佳 |
| 5 | v31_pinorm (baseline) | 1.039 | 0.126 | 0.379 | 24.01 | 0.371 | 239.7 | MSE/FRD 最佳 |

### Current Recommendation
- **🏆 最终提交**: v32 Final — 1ch GAN + D930 breast + D910 heart offset + hflip TTA
- **训练中**: v33 (vflip + tumor_weight=20 + MSSC=100), v33b (+ ngf=96)
- **不采用**: SDEdit refiner (失败), UC-GAN (失败), Swin (无效), 全背景 offset (Dice 崩溃)

### vs GC Validation #1 (MamoAnd)

| Metric | #1 MamoAnd | **v32 Final** | 状态 |
|--------|:---:|:---:|:---:|
| MSE ↓ | **0.57** | 0.97 | ❌ 1.7x gap (从 2.1x 缩小) |
| LPIPS ↓ | **0.08** | 0.12 | ❌ 1.5x gap |
| SSIM_tumor ↑ | 0.43 | **0.49** | ✅ +14% |
| Dice ↑ | 0.48 | **0.57** | ✅ +18% |
| HD95 ↓ | 120.6 | **108.5** | ✅ -10% |

⚠️ Test set 不同，仅作方向参考。Tumor 相关指标全面超过 GC #1。

### Final Submission Pipeline

```
Input .mha
  → Dataset930 (BreastDivider 2D) → breast_mask
  → Dataset910 (10-class) → heart_mask (label=7)
  → Dataset940 (T1w Tumor Seg) → tumor_mask (预留)
  → Per-image z-score normalize (breast foreground)
  → Pix2PixHD v32_vflip (1ch, ngf=64, blocks=12) × 2 (hflip TTA)
  → De-normalize
  → Composite: breast=synthetic, heart=pre+offset, bg=pre
  → Output .mha
```

### 技术路线总结

```
v14 (breast mask) → v22 (deeper) → v31 (per-image norm) → v32_vflip (+ vflip aug)
     Dice 0.70         0.72           0.54*                   0.55*
                                                               ↓ + postprocessing
                                                          v32 Final: 0.57* (MSE 0.97)
                                                          (* data_multislice_v3 test)
```

关键技术贡献:
1. **Per-image z-score normalization** — 消除 scanner 差异，LPIPS -28% (v31)
2. **Vertical flip augmentation** — HD95 -56%, Dice +49% vs 基线 (v32)
3. **Heart adaptive offset** — MSE -17.8%, 只改心脏区域不伤 Dice
4. **hflip TTA** — SSIM +4%, FRD -6%, Dice +2%（推理时 2x）

### Active / Failed Experiments Summary

| 方案 | 状态 | 结论 |
|------|:---:|------|
| Per-image norm (v31) | ✅ | 核心创新，所有后续版本都用 |
| Vflip augmentation (v32) | ✅ | HD95 大幅改善 |
| Heart offset (D910 label=7) | ✅ | MSE -17.8%，Dice 仅 -1% |
| hflip TTA | ✅ | 综合提升，推理时零训练代价 |
| tumor_weight=20 + MSSC=100 (v31_auroc) | ✅ | Dice 最佳单模型 |
| v33 (合并 vflip + auroc loss) | ⏳ | 训练中 |
| v33b (+ ngf=96) | ⏳ | 训练中 |
| SDEdit refiner on v31 | ❌ | per-image norm 与 diffusion 不兼容 |
| UC-GAN uncertainty (v28) | ❌ | 全面退步 |
| Swin Transformer (v29) | ❌ | 改善不显著 |
| Semi-Disentangled INR (v30) | ❌ | 计算成本过高 |
| 全背景 adaptive offset | ❌ | MSE -18% 但 Dice -7%（太粗糙） |
| Huber loss + GT clip (v31b) | ❌ | Dice 崩溃 |
| Spatial weighting (v27) | ❌ | 改善 SSIM 但损害 MSE |
| 增大 VGG loss (lambda_vgg=20) | ❌ | 不能改善 LPIPS |
| Huber loss | ❌ | MSE 最低但 Dice 崩溃 |

### Docker 提交内容

```
Docker container (~5 GB)
├── inference.py          ← v32 final pipeline (blur + hflip TTA)
├── weights/
│   ├── latest_net_G.pth  ← v32 conditional GAN (730 MB)
│   ├── breast_seg/       ← Dataset930 BreastDivider 2D (120 MB)
│   └── tumor_seg/        ← Dataset940 T1w Tumor Seg (120 MB)
└── models/networks.py
```

推理时间: ~8s/case (T4 GPU), 容器 < 10 GB。

### Active Experiments
- **v33 (next)**: v31_auroc + vflip = tumor_weight=20 + MSSC=100 + no GT clip + vflip augmentation — 合并 v31_auroc 和 v32_vflip 的优势
- v26 + pinorm: 大网络 + per-image norm — 预期降低 MSE

### Quick Reference
- 架构: Section 1
- 数据集: Section 2
- Breast Masking: Section 3
- Docker 提交: Section 6
- GC 评估指标: Section 7
- **全版本结果汇总: 文档末尾 📊**

---

## 1. 架构概述

**GlobalGenerator**: Encoder(4×下采样) → 9×ResNet Block → Decoder, 残差模式 `output = input + Δ`

- 输入: 1ch pre-contrast MRI (z-score normalized), resize 到 512×512
- 输出: 1ch synthetic post-contrast (residual mode)
- 参数: 182.4M
- Loss: GAN + 10×Feature Matching + 10×VGG + 10×Tumor L1 + 50×MSSC

## 2. 数据集

| 数据集 | Cases | 包含于 |
|--------|-------|--------|
| data_split | 1074 train / 144 test | v3-v5, v8, v9 |
| data_split_v2 | 1356 train / 150 test | v6, v7 |
| data_split_v4 | 2811 train | v10, v11 |
| data_split_v5 (Dataset932 预处理) | ISPY1 + NACT 1236 | → v13: 1236 |
| vdata_split_v4 (Dataset910 mask) | ISPY1 + NACT + AMBL 2528 | → v14: 2528 |

data_split_v4 = DUKE + ISPY1 + ISPY2 + NACT + LA-Breast + Yunnan + AMBL (7 域)

### 各数据集方向与尺寸

| 数据集 | 方向 | Axial? | 尺寸 | Cases |
|--------|------|:---:|------|-------|
| DUKE | LAI | ✅ | 448×448, 512×512 | ~900 |
| ISPY2 | LAI | ✅ | 多种 | ~800 |
| LA-Breast | Axial | ✅ | 448×448, 480×480 | ~500 |
| Yunnan | Axial | ✅ | 多种 | ~100 |
| **ISPY1** | **PSL** | ❌ | **256×256** | ~300 |
| **NACT** | **PSL** | ❌ | **256×256** | ~150 |
| **AMBL** | **RAS (sagittal)** | ❌ | **512×112** | ~50 |

⚠️ GC 官方确认: validation/test 只包含 axial cases。
data_split_v4 axial-only = DUKE + ISPY2 + LA-Breast + Yunnan = **2528 cases** (去掉 ISPY1/NACT/AMBL)

## 3. Breast Masking Pipeline

**模型**: Dataset910_BreastSegNet (nnUNet ResEncUNetL, 2D, fold_0)
- 输入: T1 MRI slice
- 输出: 10 类分割 → breast mask = label ∈ {1(tissue), 2(vessel), 5(lesion), 6(lymphnode), 9(implant)}
- 排除: 0(background), 3(muscle), 4(bone), 7(heart), 8(liver)

**训练时**: loss 只在 breast_mask > 0 区域计算
**推理时**: `output = where(breast_mask, synthetic, pre_contrast)`

---

## 4. 训练版本对比

### 版本配置

| 版本 | 数据 | Cases | Breast mask | Intensity aug | 关键改动 |
|------|------|-------|:-:|:-:|------|
| v3 | data_split | 1074 | ❌ | ❌ | baseline: residual + MSSC=50 |
| v4 | data_split | 938 | ❌ | ❌ | v3 + square_only + tumor_weight=40 |
| v5 | data_split | 1074 | ❌ | ❌ | = v3 (重训确认 baseline) |
| v6 | data_split_v2 | 1356 | ❌ | ❌ | v5 + motion data (+26%) |
| v7 | data_split_v2 | 1356 | ✅ | ❌ | v6 + breast mask loss |
| v8 | data_split | 1074 | ❌ | ✅ | v5 + intensity augmentation |
| v9 | data_split | 1074 | ✅ | ❌ | v5 + breast mask loss |
| v10 | data_split_v4 | 2811 | ✅ | ❌ | v5 params + 最大数据 + breast mask |
| v11 | data_split_v4 | 2811 | ✅ | ✅ | v10 + intensity augmentation |
| v12 | data_split_v5 | ~1400 | 预处理去胸壁 | ❌ | Dataset932 3D mask 预处理, 无 runtime mask |
| v13 | data_split_v5 | ~1236 | 预处理去胸壁 | ❌ | v12 去掉 ISPY1/NACT (axial only) |
| **v14** | **data_split_v4** | **2811** | **✅ ResEncUNetL f0** | **✅** | **全量数据 + breast mask + intensity aug** |
| v16 | data_split_v4 axial | 2528 | ✅ ensemble (ResEnc+Plain) | ✅ | v14 + ensemble breast mask + axial only |
| **v17** | **data_split_v4** | **2811** | **✅ ResEncUNetL f0** | **✅** | **v14 + SDEdit diffusion refiner (Stage 2)** |
| v18 | data_split_v4 axial | 2528 | ✅ ensemble | ✅ | v16 + SDEdit refiner (❌ 失败) |
| v19 | data_split_v4 | 2811 | ✅ ResEncUNetL f0 | ✅ | v14 + residual refiner (单步 Δ) ⏳ |
| **v20** | **data_split_v4** | **2811** | **✅ Dataset920 2D distilled** | **✅** | **v14 但用 distilled 2D mask (训练推理一致)** |
| v21 | data_split_v4 | 2811 | ✅ Dataset920 | ✅ | Bilateral split (左右乳房分别合成) |
| **v22** | **data_split_v4** | **2811** | **✅ Dataset920** | **✅** | **n_blocks=12 (deeper) 🏆 MSE/LPIPS 最佳** |
| v23 | data_split_v5 | ~1400 | 预处理去胸壁 | ❌ | LocalEnhancer ngf=32 (❌ 容量不足) |

### 评估结果: data_split_v2/test (150 cases)

| Metric | v5 | v6 | v7 | v8 | v9 | v20 |
|--------|:-:|:-:|:-:|:-:|:-:|:-:|
| MSE ↓ | 0.283 | 0.87 | 0.89 | **0.258** | 0.284 | 0.574 |
| LPIPS ↓ | 0.074 | 0.114 | 0.138 | **0.071** | 0.115 | 0.137 |
| SSIM_tumor ↑ | 0.701 | 0.423 | 0.407 | 0.718 | **0.731** | 0.623 |
| FRD ↓ | 10.41 | 12.78 | 9.85 | **9.77** | 9.85 | 10.46 |
| AUROC ↑ | 0.930 | 0.926 | 0.926 | 0.929 | — | — (xgb error) |
| Dice ↑ | 0.684 | 0.492 | 0.437 | **0.708** | 0.680 | 0.678 |
| HD95 ↓ | 55.6 | 105.8 | 115.8 | **37.7** | 54.1 | 48.8 |

### 评估结果: data_split/test — 全版本对比

| Metric | v5 | v9 | v11 | v12 | v13* | v14 | v16 | v17 | v20 | v21 | v22 | v23 | KFold | Best |
|--------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|------|
| MSE ↓ | 0.618 | 0.270 | 0.246 | 1.017 | 1.193 | 0.212 | 0.218 | 0.216 | 0.191 | 0.974 | **0.196** | 0.922 | 0.174 | 🏆 Ens |
| LPIPS ↓ | 0.148 | 0.118 | 0.125 | 0.238 | 0.208 | 0.126 | 0.124 | 0.109 | 0.100 | 0.145 | **0.101** | 0.141 | 0.131 | 🏆 v22 |
| SSIM_t ↑ | 0.361 | 0.599 | 0.581 | 0.321 | 0.384 | 0.689 | 0.680 | 0.688 | 0.704 | **0.784** | 0.698 | 0.367 | 0.761 | 🏆 v21 |
| FRD ↓ | 10.75 | **9.41** | 9.41 | 10.32 | 12.02 | 11.90 | 11.51 | 10.54 | 12.52 | — | — | 11.91 | 10.68 | v9/v11 |
| Dice ↑ | 0.388 | 0.561 | 0.565 | 0.363 | 0.319 | 0.703 | 0.722 | 0.731 | 0.718 | 0.715 | — | — | **0.756** | 🏆 Ens |
| HD95 ↓ | 186.7 | 115.4 | 108.1 | 203.9 | 276.9 | 68.9 | 63.7 | 62.8 | 67.0 | 62.9 | — | — | **61.5** | 🏆 Ens |

*v13/v14/v16 在 199 axial cases 上评估
*KFold Ens. = 4-model ensemble on 1353 cases (DUKE+ISPY2+YUNNAN)

**结论**:
- **v22 为最佳单模型**: MSE 0.196, LPIPS 0.101 (pixel + perceptual 最优)
- **v21 SSIM_tumor 最高 (0.784)** 但 MSE 代价大 — 适合 ensemble
- **v23 失败**: LocalEnhancer ngf=32 容量不足
- **KFold Ensemble 整体最强**: 但推理成本 3×

### Yunnan 外部验证 (100 cases, 独立数据)

| Metric | v5 | v6 | v7 | v8 | **v9** |
|--------|:-:|:-:|:-:|:-:|:-:|
| MSE ↓ | 0.393 | 0.399 | 0.125 | 0.009 | **0.010** |
| LPIPS ↓ | 0.235 | 0.290 | 0.120 | 0.045 | **0.059** |
| SSIM_tumor ↑ | 0.405 | 0.324 | 0.482 | 0.759 | **0.818** |
| FRD ↓ | 29.55 | 30.45 | 28.56 | 25.52 | **24.94** |
| AUROC ↑ | 0.879 | 0.934 | 0.844 | 0.740 | — |
| Dice ↑ | 0.095 | 0.076 | 0.090 | 0.373 | **0.474** |
| HD95 ↓ | 574.7 | 584.6 | 651.2 | 235.8 | **240.0** |

**结论**: v9 在 SSIM_tumor (0.818), FRD (24.94), Dice (0.474) 上全面最佳。v8 在 MSE/LPIPS 上略优（0.009 vs 0.010）。两者互补：v8 精度高，v9 tumor 区域更强。

### AMBL 外部验证 (51 cases, 512×112 sagittal slices)

| Metric | v9 |
|--------|:-:|
| MSE ↓ | 1.884 |
| LPIPS ↓ | 0.274 |
| SSIM_tumor ↑ | 0.043 |
| AUROC contrast ↑ | **0.906** |
| Dice ↑ | 0.000 |
| HD95 ↓ | 311.3 |

**结论**: AMBL 数据为 sagittal 长条形 (512×112)，resize 到 512×512 严重拉伸导致质量差。AUROC=0.91 说明增强信号方向正确，但空间结构失真。需 aspect-ratio-preserving padding 才能正确验证。

### 关键发现

1. **v8 本地测试集最强**: intensity aug 全面提升，HD95 -32%
2. **v9 外部 tumor 指标最强**: Yunnan SSIM_tumor 0.82, Dice 0.47
3. **v6 < v5**: motion data 拉低质量
4. **v8 和 v9 互补**: v8 image fidelity 最优，v9 tumor realism 最优
5. **Breast mask + intensity aug 互补**: mask 提升泛化，aug 提升精度

---

## 5. GC Validation Phase 对比 (2026-06-12)

| Metric | v3 (提交) | #1 (6/11) |
|--------|:-:|:-:|
| MSE ↓ | 1.32 | **0.81** |
| LPIPS ↓ | 0.242 | **0.10** |
| SSIM_tumor ↑ | 0.277 | **0.50** |
| FRD ↓ | **11.88** | 23.10 |
| AUROC contrast | **0.854** | 0.81 |
| Dice ↑ | 0.271 | **0.49** |
| HD95 ↓ | 186.6 | **107.4** |

v3 Mean Position ≈ 16.1

---

## 5b. GC Test Cohort 特征分析

### 官方测试集描述

| 属性 | Test A (Radboud, NL) | Test B (Fleming, AR) | 训练集 (MAMA-MIA) |
|------|:-:|:-:|:-:|
| Cases | 200 | 100 | 1,506 |
| 图像尺寸 | **416×416** | 512×512 | 多种 |
| 方向 | Axial | Axial | Axial 84.4%, Sagittal 15.6% |
| 场强 | 3T | 1.5T | 1.5T 72.1%, 3T 27.9% |
| 厂商 | Siemens | GE | GE 64%, Siemens 27%, Philips 9% |
| 脂肪抑制 | Yes | Yes | — |
| Pixel spacing | 0.87 mm | — | — |
| TR/TE | 5-10/2-5 ms | 4.2/2 ms | — |
| Slice thickness | ~1 mm | 1.1 mm | — |
| 分子亚型 | Luminal 86%, TN 9% | Luminal 37%, TN 30% | — |

### 对模型选择的影响

1. **416×416 新尺寸 (Test A)**: 训练数据中无此分辨率，但 resize→512→推理→resize 回来无问题
2. **全 Axial**: 确认去掉 sagittal 数据 (v14+) 的决策正确
3. **Test A = Siemens 3T + 脂肪抑制**: 训练数据中 Siemens 仅 27%，可能泛化稍弱
4. **Test B = GE 1.5T**: 与训练主力 (GE 64%) 匹配好，预期表现更稳
5. **Test B TN 30%**: Triple Negative 肿瘤占比高，增强模式可能不同
6. **Single-institution**: 比多中心一致性更好，但域偏移可能更集中

### 模型策略建议

- **v20 (全图 GAN)** 最稳健: intensity aug 提供厂商/场强鲁棒性，无 mask 边界问题
- **v21 bilateral split** 对 416×416 小图不友好 (crop 后分辨率过低)
- **建议提交 v20** 作为主力版本

---

## 5c. BreastDivider 3D 模型 — 蒸馏方案

### 模型介绍

**BreastDivider** (MICCAI 2025 WOMEN) 是目前最大规模的乳腺 MRI 分割模型：

| 属性 | BreastDivider | Dataset920 (当前用) | Dataset910 (v14用) |
|------|:-:|:-:|:-:|
| 维度 | **3D** (128³ patch) | 2D | 2D |
| 训练数据 | **13,752** 3D scans | ~973 cases | ~973 cases |
| 输出 | 3类: bg/left/right | binary breast | 10类 |
| 模态泛化 | T1, T1+C, T2, FLAIR, DWI | T1 only | T1 only |
| 权重大小 | 100 MB | ~350 MB | 1.6 GB |
| 架构 | PlainConvUNet 3D, 6 stages | PlainConvUNet 2D | ResEncUNetL 2D |
| Spacing | 1.64×2.62×2.66 mm | 1×1 mm | 1×1 mm |

来源: Rokuss et al., arXiv:2507.13830, CC BY-NC-SA 4.0
权重: `/Users/ehogjig/git/kth/BreastDividerModel/` (HuggingFace)

### 为什么不能直接用于 GC

GC 输入是**单张 2D slice** (.mha)，BreastDivider 需要完整 3D volume。单张 slice 无法满足 3D 卷积的 z 维度需求。Pseudo-3D (复制 slice) 无效 — 3D 卷积在 z 方向学的是解剖连续性，重复帧没有这个信息。

### 蒸馏方案

```
原始 3D volumes (Berzelius)
        │
        ▼
BreastDivider (3D) → 高质量 breast mask (left=1, right=2)
        │
        ▼ 取对应 2D slice (preprocess 选的那层)
        │
2D pseudo labels
        │
        ▼
训练 nnUNet 2D student (Dataset930_BreastDivider2D)
        │
        ▼
推理时使用 2D student (只需单张 slice)
```

### 可用数据源

| 数据集 | 3D volumes 位置 | Cases | 可用 |
|--------|----------------|-------|:---:|
| DUKE | `/proj/.../images/DUKE_*/DUKE_*_0000.nii.gz` | ~900 | ✅ |
| ISPY2 | `/proj/.../images/ISPY2_*/...` | ~1000+ | ✅ |
| Yunnan | `/Users/ehogjig/Downloads/8068383/*/P0.nii.gz` | 100 | ✅ |
| LA-Breast | Berzelius | ~300 | ✅ |

### 本地验证 (Yunnan)

- Input: 896×896×120, spacing 0.38×0.38×1.7 mm
- CPU 推理时间: **~2 min/case**
- 输出: labels {0: background, 1: left breast, 2: right breast}
- Yunnan 自带 `Breast_mask.nii.gz` 可做质量对比

### 速度估算

| 环境 | 时间/case | 全量 (~2800 cases) |
|------|:-:|:-:|
| 本地 CPU | ~2 min | ~93 小时 ❌ |
| Berzelius GPU (A100) | ~5-10 sec | **~30 分钟** ✅ |

### 蒸馏优势 vs 当前 Dataset920

1. **训练数据量 14x** (13,752 vs 973) → 更鲁棒的 teacher
2. **多模态泛化** → 对 Test A (Siemens 3T, fat-sat) 可能更好
3. **Left/right 分离** → bilateral split 更精确
4. **3D 上下文** → 比纯 2D 分割的边界更准

### 执行计划

1. 上传 BreastDivider 权重到 Berzelius (~100 MB)
2. 对 data_split_v4 所有 cases 的 pre-contrast 3D volumes 跑推理
3. 从 report.csv 获取 slice index → 取对应 2D mask
4. 训练 nnUNet 2D student
5. 用 student 替换 Dataset920 → 重新训练 GAN (v22?)

### 注意事项

- License: CC BY-NC-SA 4.0 — 学术用途 OK，商业需注意
- 需要确认 preprocess 记录了 slice index (report.csv 中应该有)
- 如果 report.csv 缺少 slice index，需重新跑 preprocess 记录

---

## 6. Docker 提交

```
Dockerfile: pytorch:2.0.1-cuda11.7 + nnunetv2 + SimpleITK
inference.py: BreastSeg → Pix2PixHD → 合并
weights/latest_net_G.pth (696 MB)
weights/breast_seg/ (848 MB, ResEncUNetL)
总计 ~6.3 GB (< 10 GB 限制)
```

推理流程:
```
Input → nnUNet breast mask → Resize 512 → Pix2PixHD → Resize back → 合并 → Output
```

---

## 7. GC 评估指标重要性 & Breast Masking Pipeline 流程

### GC 排名机制

Grand Challenge 使用 **Mean Position** 排名：8 个指标各自排名，取算术平均。

| 指标 | 权重 | 类型 | 说明 |
|------|------|------|------|
| MSE ↓ | 1/8 | 图像级 | 全图像素误差 |
| LPIPS ↓ | 1/8 | 图像级 | 感知相似度 |
| SSIM_tumor ↑ | 1/8 | ROI级 | 肿瘤区域结构相似度 |
| FRD ↓ | 1/8 | 分布级 | Fréchet Radiomics Distance |
| AUROC_contrast ↑ | 1/8 | 分类 | 造影分类准确率 |
| AUROC_tumor ↑ | 1/8 | 分类 | 肿瘤ROI分类 |
| **Dice ↑** | **1/8** | **分割** | **合成图→nnUNet分割→vs GT mask** |
| **HD95 ↓** | **1/8** | **分割** | **分割边界最大偏差 (P95)** |

### ⚠️ Dice 和 HD95 的特殊重要性

Dice 和 HD95 **不是直接评估合成质量**，而是评估：
> "用你的合成图像做肿瘤分割，分割结果和 GT 相比有多准确"

这意味着：
1. **合成的增强信号必须足够强且位置正确** — 否则分割模型找不到肿瘤
2. **肿瘤边界必须锐利清晰** — 模糊的增强导致分割边界偏移，HD95 暴涨
3. **不能有假阳性增强** — 胸壁/正常组织的错误增强会让分割模型产生 false positive

Breast masking 对 Dice/HD95 的帮助：
- **去除胸壁假阳性**：胸壁保持 pre-contrast → 分割模型不会在胸壁区域误检
- **肿瘤增强更聚焦**：模型容量集中在乳房 → 肿瘤区域合成更精确

### 完整 Breast Masking 训练+推理流程

```
═══════════════ 训练阶段 ═══════════════

[一次性] 生成 breast masks:
  data_split_v4/train/mha/input/*.mha
    → nnUNet (ResEncUNetL, Dataset910)
    → 10类分割
    → binary mask (labels {1,2,5,6,9} = 乳房)
    → data_split_v4/train/mha/breast_mask/*.mha

[每个 epoch] 训练:
  pre, gt, tumor_mask, breast_mask = load_batch()
  fake = Generator(pre)               # 模型看全图
  
  # 所有 loss 只在乳房区域:
  loss_GAN   = GAN(fake×bmask, gt×bmask)
  loss_feat  = L1(D_feat(fake×bmask), D_feat(gt×bmask))
  loss_VGG   = VGG(fake×bmask, gt×bmask)
  loss_MSEC  = MSEC((fake-pre)×bmask, (gt-pre)×bmask)
  loss_tumor = L1(fake×tmask, gt×tmask)
  
  loss_G = 1×GAN + 10×feat + 10×VGG + 50×MSEC + 10×tumor

═══════════════ 推理阶段 ═══════════════

input.mha (pre-contrast)
  │
  ├─→ nnUNet BreastSeg ─→ breast_mask (binary, 原始分辨率)
  │
  ├─→ resize 512×512 ─→ Pix2PixHD (残差模式) ─→ resize back
  │                                               → synthetic
  │
  └─→ output = breast_mask × synthetic + (1 - breast_mask) × input
       │              │                          │
       │    乳房: 用合成结果            胸壁: 保留原值
       │
       └─→ output.mha
```

### 权重文件

| 模型 | 路径 (Docker) | 路径 (Berzelius) | 大小 |
|------|--------------|-----------------|------|
| Pix2PixHD | `/opt/app/weights/latest_net_G.pth` | `$PROJ/checkpoints/mamasynth_v{N}/latest_net_G.pth` | 696 MB |
| nnUNet BreastSeg | `/opt/app/weights/breast_seg/` | `$PROJ/weights/Dataset910_BreastSegNet/nnUNetTrainer__nnUNetResEncUNetLPlans__2d` | 1.6 GB |

---

## 8. Breast Segmentation 模型对比

### 可用模型 (weights/breast_seg/)

| | Dataset910 (PlainConv 2D) | Dataset910 (ResEncUNet 2D) | Dataset920 (2D Distilled) | Dataset932 (3D fullres) |
|---|---|---|---|---|
| 路径 | `nnUNetTrainer__nnUNetPlans__2d` | `nnUNetTrainer__nnUNetResEncUNetLPlans__2d` | `Dataset920_BreastSeg2D/nnUNetTrainer__nnUNetPlans__2d` | `nnUNetTrainer__nnUNetPlans__3d_fullres` |
| 输入 | 1ch: T1 (pre-contrast) | 1ch: T1 (pre-contrast) | 1ch: T1 (pre-contrast) | 4ch: P0, P1, d_early, d_late |
| 输出 | 10类 → binary | 10类 → binary | **binary (直接)** | 4类 (breast, FGT, tumor) |
| 训练数据 | 973 cases | 973 cases | 2811 cases (D932 pseudo labels) | 1280 cases |
| 架构 | 标准卷积 | 残差编码器 | 标准卷积 2D | 3D fullres |
| 推理时可用 | ✅ | ✅ | ✅ | ❌ (需要 post-contrast) |
| Breast mask 提取 | label 1+2+5+6+9 | label 1+2+5+6+9 | 直接输出 binary | label 1+2+3 |
| 模型大小 | 710 MB | 1.6 GB | **~120 MB** | ~2 GB |
| 适用场景 | 推理时 breast mask | 推理时 breast mask (更好) | **训练+推理一致 (v20/v21)** | 训练时 breast mask (最准) |
| 优势 | 轻量 | 最精确(单模型) | 训练推理无域差，轻量快速 | 多相信息最准 |


### v16 Ensemble Pipeline

组合 ResEncUNetL fold_0 + PlainConvUNet fold_4，OR 合并后后处理：

```
Model 1 (ResEncUNetL f0) → mask1
Model 2 (PlainConvUNet f4) → mask2
         ↓
    OR 合并 (mask1 | mask2)
         ↓
    Drop small components (<10% of largest)
         ↓
    Morphological closing (15×15 ellipse)
         ↓
    Fill internal holes (floodFill)
         ↓
    Spatial analysis top-2 components:
      • Left-right (dx > dy): dilate until connected → erode back
      • Top-bottom (dx < dy): keep only largest
         ↓
    Final breast mask
```

- 用于: v16 训练 (ensemble mask 更完整，覆盖胸壁边缘乳房组织)
- 总大小: 710 MB + 1.6 GB = 2.3 GB
- 效果: v16 Dice 0.722 > v14 Dice 0.703 (+2.7%)

---

## 9. 数据预处理两方案对比

### 方案 A: mask_and_preprocess.py (一体化, Dataset932 4ch 3D)

```
images/<patient>/<patient>_0000.nii.gz (3D multi-phase DCE)
  → Exp4x Dataset932 (4ch: P0, P1, d_early, d_late, 3D fullres)
  → 3 labels: breast / FGT / tumor
  → breast_mask_3d → 去胸壁 → 选 peak phase → 选最大 tumor slice → z-score
  → output: input/ ground_truth/ mask/ breast_mask/ (2D MHA)
```
- **优点**: 更精确（4 通道专业乳腺分割，含动态增强信息）
- **缺点**: 慢（~19h GPU），需要 post-contrast 多相数据
- **权重**: `/Users/ehogjig/git/kth/exp4x_for_maia/Dataset932/nnUNetTrainer__nnUNetPlans__3d_fullres`

### 方案 B: preprocess.py + generate_breast_masks.py (分步, Dataset910 2D)

```
images/ + segmentations/
  → preprocess.py (CPU, ~30min): 3D NIfTI → peak phase → tumor slice → z-score → 2D MHA
  → generate_breast_masks.py (GPU, ~75min): Dataset910 ResEncUNetL 2D, 单通道 T1
  → output: input/ ground_truth/ mask/ breast_mask/ (2D MHA)
```
- **优点**: 快（2h vs 19h），推理时也能用同一模型（只需 pre-contrast）
- **缺点**: 精度略低（10 类通用模型 vs 3 类专业模型）
- **权重**: `nnUNet_pretrained_weights/Dataset910_BreastSegNet/nnUNetTrainer__nnUNetResEncUNetLPlans__2d`

### 对比总结

| | 方案 A (Dataset932) | 方案 B (Dataset910) |
|---|---|---|
| 输入通道 | 4ch (P0, P1, d_early, d_late) | 1ch (T1 pre-contrast) |
| 维度 | 3D fullres | 2D |
| 输出 | 3 类 (breast/FGT/tumor) | 10 类 (tissue/vessel/muscle/bone/...) |
| 速度 | ~19h (100 cases) | ~2h (100 cases) |
| 需要 post-contrast | ✅ | ❌ |
| 推理时可用 | ❌ (需要多 phase) | ✅ (只需 pre) |
| 用于 | AMBL 数据预处理 | 训练时 mask 生成 + Docker 推理 |

---

## 11. Axial-Only 训练发现 (v13/v14)

### GC 官方确认: "validation and test data contains only axial cases"

### 数据方向检查

| 数据集 | 方向 | Axial? | 备注 |
|--------|------|:---:|------|
| DUKE | LAI | ✅ | 448×448, 512×512 |
| ISPY2 | LAI | ✅ | 多种尺寸 |
| LABREAST | Axial | ✅ | 448×448, 480×480 (论文确认 T1 fat-sat axial) |
| YUNNAN | Axial | ✅ | 来自 axial DCE |
| **ISPY1** | **PSL** | ❌ | **Sagittal**, 256×256×60 |
| **NACT** | **PSL** | ❌ | **Sagittal**, 256×256×60 |
| **AMBL** | **RAS (sagittal slice)** | ❌ | **512×112**, sagittal 切面 |

### 影响

Sagittal 数据混入训练会"污染"模型：
- 模型需要同时学习 axial 和 sagittal 两种完全不同的解剖结构
- GC 只测 axial → sagittal 训练数据是纯噪声，浪费模型容量

### 版本对比

| 版本 | 基础 | 去掉 | 剩余 cases |
|------|------|------|-----------|
| v12 | data_split_v5 (Dataset932 预处理) | ISPY1 + NACT | → v13: 1236 |
| v11 | data_split_v4 (Dataset910 mask) | ISPY1 + NACT + AMBL | → v14: 2528 |

### v13 配置
- 数据: data_split_v5 去掉 ISPY1/NACT (DUKE + ISPY2 + YUNNAN = 1236 cases)
- 预处理: Dataset932 3D mask (input 已去胸壁)
- 无 breast_mask_dir, 无 intensity_aug

### v14 配置
- 数据: data_split_v4 去掉 ISPY1/NACT/AMBL (ISPY2 + DUKE + LABREAST + YUNNAN = 2528 cases)
- Breast mask: ✅ (Dataset910 loss mask)
- Intensity aug: ✅
- 与 v11 相同超参，仅去掉 sagittal 数据

---

## 12. v16 Breast Mask Ensemble Pipeline

### 模型组合

| Model | 架构 | Fold | Checkpoint | 大小 |
|-------|------|------|-----------|------|
| Model 1 | ResEncUNetL (Dataset910) | fold_0 | checkpoint_best.pth | 1.6 GB |
| Model 2 | PlainConvUNet (Dataset910) | fold_4 | checkpoint_best.pth | 710 MB |

### 后处理流程

```
Model 1 predict → mask1 (各自带后处理)
Model 2 predict → mask2 (各自带后处理)
         ↓
    OR 合并 (mask1 | mask2)
         ↓
    Drop small components (<10% of largest)
         ↓
    Morphological closing (15×15 ellipse kernel)
         ↓
    Fill internal holes (floodFill)
         ↓
    Spatial analysis of top-2 components:
      • If left-right (dx > dy): dilate until connected → erode back
      • If top-bottom (dx < dy): keep only largest
         ↓
    Final breast mask (clean, connected)
```

### v16 训练配置

| 属性 | 值 |
|------|-----|
| 数据 | data_split_v4 axial-only (2528 cases) |
| Breast mask | Ensemble (ResEncUNetL f0 OR PlainConv f4) + postprocess |
| Intensity aug | ✅ |
| Epochs | 200 (niter=100 + niter_decay=100) |
| 其余超参 | 同 v14 (MSEC=50, tumor=10, etc.) |

**结果 (data_split/test, 199 cases)**:

| Metric | v14 | **v16 full** | v17 |
|--------|:---:|:---:|:---:|
| MSE ↓ | **0.212** | 0.218 | 0.216 |
| LPIPS ↓ | 0.126 | 0.124 | **0.109** |
| SSIM_tumor ↑ | **0.689** | 0.680 | 0.688 |
| FRD ↓ | 11.90 | 11.51 | **10.54** |
| AUROC ↑ | 0.831 | **0.844** | 0.829 |
| Dice ↑ | 0.703 | 0.722 | **0.731** |
| HD95 ↓ | 68.9 | 63.7 | **62.8** |

**结论**: v16 ensemble mask 提升了 AUROC (+1.3%) 和 Dice (+2.7%)，证明更完整的 breast mask 有帮助。但整体不如 v17 (SDEdit refiner)。

**状态**: ✅ 完成

---

## 13. v17: GAN + SDEdit Diffusion Refinement

**方法**: 在 v14 GAN 输出上加轻量 diffusion refinement (SDEdit)
- Stage 1: 冻结 v14 Pix2PixHD → coarse synthesis
- Stage 2: RefinerUNet (18M params) 学习去噪 → 精修 tumor 边界
- 推理: GAN output + 30% noise → DDIM 20 steps → refined output

**v14 vs v16 full vs v17 对比 (data_split/test)**:

| Metric | v14 | v16 full | v17 (v14+SDEdit) | Winner |
|--------|:-:|:-:|:-:|------|
| MSE ↓ | **0.212** | 0.218 | 0.216 | v14 |
| LPIPS ↓ | 0.126 | 0.124 | **0.109** | 🏆 v17 (+12%) |
| SSIM_tumor ↑ | **0.689** | 0.680 | 0.688 | v14 |
| FRD ↓ | 11.90 | 11.51 | **10.54** | 🏆 v17 (+8%) |
| AUROC ↑ | 0.831 | **0.844** | 0.829 | v16 |
| Dice ↑ | 0.703 | 0.722 | **0.731** | 🏆 v17 |
| HD95 ↓ | 68.9 | 63.7 | **62.8** | 🏆 v17 |

**v16 full 评估详情** (data_split/test, 199 cases):
- MSE: 0.218, LPIPS: 0.124, SSIM_tumor: 0.680
- FRD: 11.51, AUROC: 0.844, Dice: 0.722, HD95: 63.7
- v16 full 比 v16 50epoch (MSE=0.469, Dice=0.507) 大幅提升，200 epoch 训练充分

**v17 vs v16 full per-case 分析**:
- LPIPS: v17 更好 186/199 cases (93%)
- Dice: v17 更好 104/199 cases (52%)
- HD95: v17 更好 86/199 cases (43%)
- v17 主要优势在感知质量和分布真实性，Dice/HD95 为小幅提升

**分析**:
- v17 在 Dice/HD95/LPIPS/FRD 上全面最佳 → SDEdit 确实精修了 tumor 边界
- v16 在 AUROC 上最好 (ensemble breast mask 提升了增强信号准确性)
- v14 在 MSE/SSIM 上最好 (像素精度最高)
- SDEdit 推理开销仅 +5s (MPS) / +3s (T4)，远在 10min 限制内

**结论**: v17 是当前**最佳提交版本** (Dice 0.731, HD95 62.8, LPIPS 0.109)

**下一步**: 在 v16 base 上跑 SDEdit (v18)，因为 v16 Dice=0.722 > v14 Dice=0.703

---

## 14. v18: v16 GAN + SDEdit Refiner (失败分析)

**配置**: v16 GAN (ensemble breast mask, axial-only 2528 cases) + SDEdit refiner
**训练数据**: data_split_v4_axial/train (同 v16)

**结果 (data_split/test)**:

| Metric | v16 (GAN only) | v17 (v14+SDEdit) | v18 (v16+SDEdit) |
|--------|:-:|:-:|:-:|
| MSE ↓ | 0.218 | **0.216** | 0.469 ❌ |
| LPIPS ↓ | 0.124 | **0.109** | 0.116 |
| SSIM_tumor ↑ | 0.680 | **0.688** | 0.451 ❌ |
| FRD ↓ | 11.51 | 10.54 | **9.04** |
| AUROC ↑ | **0.844** | 0.829 | 0.850 |
| Dice ↑ | 0.722 | **0.731** | 0.532 ❌ |
| HD95 ↓ | 63.7 | **62.8** | 127.9 ❌ |

**失败原因**:
- Dice=0 的 cases: v17 有 15 个，v18 有 **39 个**（多出 24 个完全分割失败）
- 受影响 dataset: ISPY2 (34 cases) + DUKE (15 cases)，Dice drop > 0.3
- 即使两者都非零的 158 cases，v18 平均 Dice (0.667) 也远低于 v17 (0.805)

**根因**: v16 的 ensemble breast mask (union + morphology) 比 v14 的单模型 mask 更大、边界更模糊。
refiner 在这种 mask 策略下训练，学到了过度平滑增强信号的模式，导致 tumor 信号被削弱，
nnUNet 分割模型找不到 tumor。

**结论**: SDEdit refiner 只适合搭配单模型 breast mask 的 GAN (v14)。v17 仍为最佳。

---

## 15. v19: Residual Refiner (Supervisor 建议)

**思路**: 不用 diffusion 多步去噪，直接用 UNet 预测残差 Δ，output = pre + Δ。

**与 v17 的区别**:

| | v17 (SDEdit) | v19 (Residual) |
|---|---|---|
| 方法 | GAN → 加噪 → 20步去噪 | GAN → 单步 UNet → pre + Δ |
| 推理时间 | ~5s | **~1s** |
| 训练 loss | MSE(ε̂, ε) noise prediction | L1(output, gt) |
| 输入 | 3ch [noisy, pre, gan_out] + timestep | 2ch [pre, gan_out] |
| 模型 | RefinerUNet 18M | ResidualRefiner ~30M |

**架构**: UNet encoder(4层) → bottleneck → decoder(skip connections) → predict Δ
- 输入: [pre-contrast, GAN output] concat (2ch)
- 输出: residual Δ (1ch)
- 最终: output = pre + Δ

**训练配置**:
- Base GAN: v14 (frozen)
- Data: data_split_v4/train
- Loss: L1(pre + Δ, gt)
- Epochs: 100, batch=8, lr=2e-4
- 脚本: `berzelius_train_v19.sh`

**结果 (data_split/test, 199 cases)**:

| Metric | v14 (GAN) | v17 (v14+SDEdit) | v19 (v14+Residual) |
|--------|:-:|:-:|:-:|
| MSE ↓ | **0.212** | 0.216 | 0.185 |
| LPIPS ↓ | 0.126 | **0.109** | 0.196 ❌ |
| SSIM_tumor ↑ | 0.689 | 0.688 | **0.709** |
| FRD ↓ | 11.90 | 10.54 | **9.04** |
| AUROC ↑ | 0.831 | 0.829 | 0.755 ❌ |
| Dice ↑ | 0.703 | **0.731** | 0.694 |
| HD95 ↓ | 68.9 | **62.8** | 71.9 |
| Dice=0 cases | — | 15 | 18 |

**分析**:
- ✅ MSE 最佳 (0.185)，SSIM_tumor 最佳 (0.709)，FRD 最佳 (9.04)
- ❌ LPIPS 严重退化 (0.196 vs v17 0.109)：单步残差缺乏感知质量精修
- ❌ AUROC 下降 (0.755)：增强信号不够逼真，classifier 容易区分真假
- ❌ Dice/HD95 略低于 v14/v17：tumor 边界精度未改善
- Dice=0 cases 18个 (v17=15)：比 v17 多 3 个完全分割失败

**结论**: 残差 refiner 在像素精度 (MSE) 和 radiomics 分布 (FRD) 上有优势，
但感知质量 (LPIPS) 和分类器欺骗能力 (AUROC) 显著退化。
单步 UNet 太简单，无法学到 diffusion 多步去噪的精修效果。
**v17 仍为最佳 refiner 方案。**

---

## 16. v20: Distilled 2D Breast Mask (Dataset920)

**动机**: v14 训练时用 Dataset910 ResEncUNetL 生成 breast mask，但推理时也用同一模型。
然而 Dataset910 是 10 类通用模型，精度不如专用 breast-only 模型。
v20 用 Dataset920 (distilled 2D，从 Dataset932 3D 蒸馏而来) 生成训练 mask，
这样训练和推理使用**完全相同**的 mask model，消除域差。

**配置**:

| 属性 | 值 |
|------|-----|
| 数据 | data_split_v4 (2811 cases, 全量含 sagittal) |
| Breast mask | Dataset920 2D distilled (fold_0, checkpoint_final) |
| Intensity aug | ✅ |
| Epochs | 200 (niter=100 + niter_decay=100) |
| 其余超参 | 同 v14 (MSEC=50, tumor=10, feat=10, VGG=10) |
| 脚本 | `berzelius_train_v20_2dmask.sh` |
| 训练时间 | 9h 47min (1 GPU) |

**与 v14 唯一区别**: breast mask 从 Dataset910 ResEncUNetL → Dataset920 distilled 2D

**训练流程** (berzelius_train_v20_2dmask.sh):
1. 先用 Dataset920 给 data_split_v4/train/mha/input 生成 breast_mask_2d/
2. 然后以 `--breast_mask_dir breast_mask_2d` 训练 Pix2PixHD

**结果 (data_split/test, 199 cases)**:

| Metric | v14 | v16 | v17 | **v20** |
|--------|:-:|:-:|:-:|:-:|
| MSE ↓ | 0.212 | 0.218 | 0.216 | **0.191** 🏆 |
| LPIPS ↓ | 0.126 | 0.124 | 0.109 | **0.100** 🏆 |
| SSIM_tumor ↑ | 0.689 | 0.680 | 0.688 | **0.704** 🏆 |
| FRD ↓ | 11.90 | 11.51 | **10.54** | 12.52 |
| Dice ↑ | 0.703 | 0.722 | **0.731** | 0.718 |
| HD95 ↓ | 68.9 | 63.7 | **62.8** | 67.0 |

**结果 (data_split_v2/test, 150 cases)**:

| Metric | v5 | v8 | **v20** |
|--------|:-:|:-:|:-:|
| MSE ↓ | 0.283 | **0.258** | 0.574 |
| LPIPS ↓ | **0.074** | 0.071 | 0.137 |
| SSIM_tumor ↑ | 0.701 | 0.718 | 0.623 |
| FRD ↓ | 10.41 | **9.77** | 10.46 |
| Dice ↑ | 0.684 | **0.708** | 0.678 |
| HD95 ↓ | 55.6 | **37.7** | 48.8 |

**分析**:
- **data_split/test (主要对比)**: v20 在像素精度 (MSE -10%)、感知质量 (LPIPS -8%)、肿瘤结构 (SSIM +2%) 上全面新高
- 分割指标 (Dice 0.718, HD95 67.0) 接近 v14 水平，但略低于 v17 SDEdit
- FRD 12.52 偏高，说明 radiomics feature 分布与 GT 有偏移
- **data_split_v2/test**: 表现中等，可能因为 v2 test 包含更多 sagittal 数据，而 v20 的 Dataset920 mask 在 sagittal 上表现不佳

**结论**: v20 证明了 train/infer mask 一致性的价值 — 像素级指标全面最优。
但分割指标未超越 v17，说明 SDEdit 的 tumor boundary 精修仍有不可替代的优势。

**下一步**: v20 + SDEdit refiner (v21?) 有望结合两者：v20 的像素精度 + SDEdit 的分割提升。

### v20 推理时 Breast Mask 验证

**实验**: 在 v20 推理时加入 Dataset920 breast mask，`output = mask × synthetic + (1-mask) × pre`

| Metric | v20 (无 mask) | v20 + mask | 变化 |
|--------|:-:|:-:|------|
| MSE ↓ | **0.191** | 0.817 | ❌ +328% |
| LPIPS ↓ | **0.100** | 0.107 | ❌ +7% |
| SSIM_tumor ↑ | **0.704** | 0.698 | ❌ -0.9% |
| FRD ↓ | 12.52 | **12.26** | ✅ 微弱 |
| Dice ↑ | **0.718** | 0.705 | ❌ -1.8% |
| HD95 ↓ | **67.0** | 72.3 | ❌ +7.9% |

**结论: 推理时加 mask 全面恶化**，与 v17 的对比实验结果一致。

**原因**:
1. GT 是 post-contrast 全图，mask 外区域 ≠ pre-contrast（全身组织都有轻微增强）。用 pre 填充 mask 外 → 与 GT 产生大量误差 → MSE 暴涨。
2. Mask 硬边界产生不自然强度跳变 → LPIPS 感知网络敏感 → 分割模型也受干扰。
3. GAN 残差模式 (`output = pre + Δ`) 已学到全图微弱增强，比硬塞 pre 值更接近 GT。

**决策: v20 提交维持 GAN-only 直出，不加推理时 mask。**

---

## 17. K-Fold Cross-Validation

**目的**: 在全量训练数据上做 5-fold CV，评估模型泛化性，避免 test set 过拟合。

**配置**:
- 数据: data_split_v4 axial-only (2528 cases)
- 划分: `kfold_splits.csv` 按 source 分层，5 fold
- 每 fold: ~2022 train / ~506 test
- Breast mask: Dataset910 ResEncUNetL (10类→binary, 同 v14)
- 模型配置: 同 v14 (residual + MSSC=50 + breast mask + intensity aug)
- 脚本: `src/submission/submission-synthesis/models/scripts/train_kfold.sh`
- SLURM array job: `--array=0-4`，5 个 fold 并行训练

**流程**:
1. `setup_kfold_dir.py` 根据 `kfold_splits.csv` 创建 fold-specific 目录（symlinks）
2. 每个 fold 独立训练 200 epochs
3. 训练完后对各 fold 的 held-out test set 推理 + 评估
4. 汇总 5 fold 结果 → 全数据集 CV 指标

**Fold 0 结果 (506 cases)**:

| Metric | Fold 0 |
|--------|:-:|
| MSE ↓ | 0.463 |
| LPIPS ↓ | 0.126 |
| SSIM_tumor ↑ | 0.603 |
| FRD ↓ | 10.56 |
| AUROC ↑ | 0.788 |
| Dice ↑ | 0.319 |
| HD95 ↓ | 274.0 |

**注意**: Fold 0 Dice=0.319 / HD95=274 远低于 v14 test set 结果 (Dice=0.703, HD95=68.9)。
可能原因:
- 训练量少 (~2022 vs 2811 cases)
- Fold 0 的 test cases 可能包含难例/少见 source
- 需要等 fold 1-4 完成后综合判断

**Splits CSV 格式**: `case_id,source,fold` (2528 行)

**权重路径**: `$PROJ/checkpoints/kfold_f{0-4}/latest_net_G.pth`

---

## 18. K-Fold Cross-Validation 结果 (v14 config, 无 LABREAST)

**配置**: v14 (GAN + MSEC=50 + breast mask + intensity aug), stratified by data source
**数据**: DUKE (280) + ISPY2 (973) + YUNNAN (100) = 1353 cases
**排除**: LABREAST (椭圆近似 mask 导致 73% Dice=0，不适合分割评估)
**每 fold**: ~1082 train / ~271 test

| Metric | Fold 0 | Fold 1 | Fold 3 | Fold 4 | **Mean ± Std** |
|--------|:-:|:-:|:-:|:-:|:-:|
| MSE ↓ | 0.791 | 0.660 | 0.649 | 0.769 | **0.717 ± 0.063** |
| LPIPS ↓ | 0.141 | 0.139 | 0.140 | 0.142 | **0.141 ± 0.001** |
| SSIM ↑ | 0.455 | 0.445 | 0.441 | 0.441 | **0.446 ± 0.006** |
| FRD ↓ | 11.25 | 10.91 | 12.80 | 10.74 | **11.43 ± 0.82** |
| AUROC ↑ | 0.909 | 0.942 | 0.936 | 0.938 | **0.931 ± 0.013** |
| Dice ↑ | 0.496 | 0.561 | 0.482 | 0.500 | **0.510 ± 0.030** |
| HD95 ↓ | 127.0 | 106.9 | 151.5 | 128.0 | **128.4 ± 15.8** |

*Fold 2 缺失（权重未下载）。4-fold 结果仍具有统计意义。*

**分析**:
- LPIPS 和 SSIM 极其稳定 (std < 0.01) → 模型对不同 test split 一致
- AUROC 0.931 → 合成增强信号方向和强度正确
- Dice 0.51 ± 0.03 → 低于单次全量训练 (v14: 0.703)，因为每 fold 只用 80% 数据训练
- MSE 偏高 (0.717) 因为包含 YUNNAN 数据 (z-score 分布不同于 DUKE/ISPY2)

---

## 19. v22: Deeper Network (n_blocks=12)

**动机**: Supervisor 建议加深网络。当前 9 blocks 的 bottleneck 可能限制了模型对全局增强模式的建模能力。

**方案对比**:

| 方案 | 改动 | 参数增量 | 风险 |
|------|------|---------|------|
| **增加 ResBlocks (选用)** | 9→12 blocks | +2M (182→184M) | 低，不影响显存/batch |
| 增加 downsampling | 4→5 层 | ~+10M | 中，空间细节丢失 |
| 增加通道数 | ngf 64→96 | +230M (→410M) | 高，batch 需减半 |
| Local Enhancer | 加 fine branch | +50M | 高，训练复杂 |

**v22 配置**:
- 基础: v20 (Dataset920 mask, axial-only 2528 cases)
- 唯一改动: `n_blocks_global: 9 → 12`
- 其余超参完全一致
- 脚本: `berzelius_train_v22_deeper.sh`

**预期效果**:
- 更深 bottleneck → 更大感受野 → 更好的全局增强一致性
- 对 Dice/HD95 可能有帮助（更准确的 tumor 增强定位）
- 训练时间略增（~10%）

**结果** (data_split/test, 199 cases):

| Metric | v20 | v22 | 变化 |
|--------|-----|-----|------|
| MSE ↓ | 0.191 | 0.196 | +0.005 |
| LPIPS ↓ | 0.100 | 0.101 | ≈持平 |
| Dice ↑ | 0.718 | 0.720 | ≈持平 |
| HD95 ↓ | 67.0 | **57.4** | **-9.6 ✅** |
| SSIM-tumor ↑ | — | 0.698 | — |
| FRD ↓ | — | 12.70 | — |

**分析**: HD95 显著改善（67→57），说明更深 bottleneck 改善了分割边界精度。MSE/LPIPS 基本持平，深度增加未损害感知质量。

**状态**: ✅ 完成

**后续**: 若 12 blocks 有效，可叠加 ngf=96（方案3）进一步提升。

---

## 20. v23: Local Enhancer 全分辨率精修

**动机**: v20 的 GlobalGenerator 在 32×32 bottleneck 做合成，高频细节（tumor 边界、组织纹理）受限于 4 次下采样的信息损失。Local Enhancer 在全分辨率（512→256→512）上精修，只做 1 次下采样。

**架构**:
```
Input 512×512 ─┬─ AvgPool ─→ 256×256 → [v20 Global (冻结)] → 256×256 features
               │                                                      │
               └─ [浅层encoder ×1] → 256×256 features ─────────────── ⊕
                                                                      │
                                                   [3 ResBlocks] → [upsample] → 512×512 output
```

**配置** (对比 v20):
| 参数 | v20 | v23 |
|------|-----|-----|
| netG | global | local |
| ngf | 64 | 32 (内部 global = 64) |
| n_local_enhancers | - | 1 |
| n_blocks_local | - | 3 |
| niter_fix_global | - | 20 |
| load_pretrain | - | mamasynth_v20 |
| batchSize | 8 | 4 |
| 其余参数 | 同 v20 | 同 v20 |

**代码改动**:
- `networks.py`: `LocalEnhancer` 支持 `residual_mode`（输出 = input + delta）
- 脚本: `berzelius_train_v23_local_enhancer.sh`

**训练策略**:
- Epoch 1-20: 冻结 Global（用 v20 权重），只训 Local 层
- Epoch 21-200: 解冻全网络，端到端 fine-tune

**结果 (data_split/test, 199 cases)**:

| Metric | v20 | **v23** |
|--------|:-:|:-:|
| MSE ↓ | **0.574** | 0.922 |
| LPIPS ↓ | **0.137** | 0.141 |
| SSIM_tumor ↑ | **0.623** | 0.367 |
| FRD ↓ | — | 11.91 |

**分析**:
- LocalEnhancer + ngf=32 容量不足，SSIM_tumor 大幅下降 (0.62→0.37)
- LPIPS 和 FRD 接近 v11 水平 — 感知质量 OK 但结构精度差
- 推理时需注意 residual mode: `result = pre + netG(pre)`
- **Bug found**: inference 初始缺少 `pre + Δ` 导致 MSE=2.96，修复后 MSE=0.92

**结论**: LocalEnhancer (ngf=32) 不如 GlobalGenerator (ngf=64)。如果要用 LocalEnhancer 需要 ngf≥64。

**状态**: ✅ 完成 — 不采用

---

## 21. K-Fold Ensemble 结果

**方法**: 对每个 test case，用所有**未见过**它的模型（3 个）做推理，取平均。
**数据**: 1353 cases (DUKE+ISPY2+YUNNAN)，每个 case 被 3 个独立模型 ensemble。

| Metric | 单 fold 平均 | **Ensemble (4 models)** | 提升 |
|--------|:-:|:-:|---|
| MSE ↓ | 0.717 | **0.174** | -76% |
| LPIPS ↓ | 0.141 | **0.131** | -7% |
| SSIM ↑ | 0.446 | **0.761** | +71% |
| FRD ↓ | 11.43 | **10.68** | -7% |
| AUROC ↑ | 0.931 | 0.887 | -5% |
| Dice ↑ | 0.510 | **0.756** | +48% |
| HD95 ↓ | 128.4 | **61.5** | -52% |

**分析**:
- Ensemble 平均消除了单模型的随机噪声 → MSE 降 76%，SSIM 涨 71%
- Dice 0.756 是所有版本中最高（超过 v17 的 0.731）
- HD95 61.5 也是最低（超过 v17 的 62.8）
- AUROC 略降 (0.931→0.887)，可能因为 ensemble 平滑了增强信号峰值

**局限**: 推理时间 ×4（需要跑 4 个模型），但仍在 T4 10min 限制内（~4s × 4 = ~16s）。
**结论**: Ensemble 是当前最佳方案，适合作为最终提交。

---

## 22. v24: Breast Mask 作为条件输入 (Mask-as-Input)

**动机**: v20 的 breast mask 仅用于 loss masking（告诉模型"别管背景"），但 Generator 本身看不到乳房边界。v24 把 breast mask 作为第 2 输入通道，让 Generator 有**显式的解剖先验**。

**架构变化**:
```
v20: pre-contrast (1ch) → Generator(input_nc=1) → delta → output
v24: concat(pre-contrast, breast_mask) (2ch) → Generator(input_nc=2) → delta → output
```

**配置** (对比 v20):
| 参数 | v20 | v24 |
|------|-----|-----|
| input_nc | 1 | 2 |
| --mask_as_input | - | ✅ |
| breast_mask 用途 | loss masking only | **条件输入** + loss masking |
| 推理依赖 | 无 | 需先跑 Dataset920 |
| 其余参数 | - | 同 v20 |

**代码改动**:
- `base_options.py`: 新增 `--mask_as_input` flag
- `mha_dataset.py`: `mask_as_input=True` 时 concat breast_mask 到 label
- `networks.py`: 无改动（`input_nc=2` 自动适配第一层 Conv）
- 脚本: `berzelius_train_v24_mask_input.sh`

**推理流程**:
1. pre-contrast → Dataset920 nnUNet → breast_mask
2. concat(pre-contrast, breast_mask) → Generator → output

**预期效果**:
- Generator 知道乳房在哪 → 增强更精准，边界更清晰
- 对不同 scanner/size 泛化更好（不依赖隐式学习乳房位置）
- Dice/HD95 可能提升（增强不溢出到胸壁）

**状态**: ⏳ 待训练

---

## TODO: K-Fold v20b (进行中)

**目标**: 修复 v20 kfold breast mask 生成失败问题，重新训练

**配置**:
- 训练数据: DUKE + ISPY2 + YUNNAN + LABREAST（全量 2528 cases）
- 测试数据: 只用 DUKE + ISPY2 + YUNNAN（5% per fold，LABREAST 只训不测）
- Breast mask: **Dataset920**（蒸馏 2D 模型，每 fold 重新生成确保不为空）
- 其余: v20 配置 (MSEC=50, intensity_aug, 200 epochs)
- 脚本: `train_kfold_v20b.sh`（SLURM array 0-4）
- 输出: `$PROJ/checkpoints/kfold_v20b_f{0-4}/latest_net_G.pth`

**完成后**:
- [ ] 下载 5 个 fold 权重
- [ ] 跑 ensemble inference + evaluate
- [ ] 对比 v14 kfold ensemble (Dice 0.756) 是否提升
- [ ] 更新 handoff 结果表

---

## 23. v25: Uncertainty-Aware Heteroscedastic Loss (SAFE-Diff 启发)

**动机**: SAFE-Diff (Zhang et al., 2026, arXiv:2605.25767) 提出 heteroscedastic uncertainty loss，让网络自适应地对不同区域施加不同重建权重。确定区域强制精确重建，模糊区域（tumor 边界、异质组织）降低 loss 权重避免强行拟合噪声。

**架构改动**:
```
GlobalGenerator (shared backbone up to ngf features)
         │
    ┌────┴────┐
 mean_head   log_var_head (新增)
    │              │
   delta         log σ² (clamped [-1.5, 3.0])
    │
 output = input + delta
```

**Heteroscedastic Loss**:
```
L_unc = mean[ ω × exp(-log σ²) × (μ - x)² + log σ² ]

空间权重 ω:
  background = 1
  breast     = 20  (via breast_mask)
  tumor      = 1000 (via tumor mask)
  归一化: ω = ω / mean(ω)
```

**配置** (对比 v20):
| 参数 | v20 | v25 |
|------|-----|-----|
| --uncertainty | - | ✅ |
| Generator 输出 | mu only | (mu, log_var) |
| 额外参数 | - | ~0.1M (一个 7×7 Conv head) |
| 推理 | 无变化 | 只用 mu，丢弃 log_var |
| 其余参数 | - | 同 v20 |

**代码改动**:
- `networks.py`: `GlobalGenerator` 加 `uncertainty=True` → shared backbone + 2 heads
- `pix2pixHD_model.py`: 新增 `loss_G_Unc` 计算，含空间加权
- `base_options.py`: 新增 `--uncertainty` flag
- 脚本: `berzelius_train_v25_uncertainty.sh`

**参考 SAFE-Diff 的空间权重策略**:
- 他们: background=1, breast=20, tumor=1000
- 这比我们的 `tumor_weight=10` 激进得多，通过 uncertainty 机制自动平衡

**预期效果**:
- Tumor 边界质量提升 → Dice ↑, HD95 ↓
- 减少对异质区域的过拟合 → LPIPS ↓
- 提供 uncertainty map 作为质量指标（可用于 ensemble selection）

**状态**: ⏳ 待训练

---

## 24. v21_nomask: Bilateral Split + Full-Image Background

**方法**: 乳房内部用 v21 bilateral GAN，乳房外部用 v20 全图 GAN（不用硬 mask composite）
- 乳房分割: Dataset920
- 乳房左右分别 → v21 GAN → stitch 回来
- 胸壁区域: 用 v20 全图 GAN 输出填充（不是保留 pre-contrast）

**结果 (data_split/test, 199 cases)**:

| Metric | v14 | v16 full | v17 (SDEdit) | **v21_nomask** |
|--------|:-:|:-:|:-:|:-:|
| MSE ↓ | **0.212** | 0.218 | 0.216 | 0.347 |
| LPIPS ↓ | 0.126 | 0.124 | **0.109** | 0.136 |
| SSIM ↑ | 0.689 | 0.680 | 0.688 | **0.790** 🏆 |
| Dice ↑ | 0.703 | 0.722 | **0.731** | 0.715 |
| HD95 ↓ | 68.9 | 63.7 | **62.8** | 62.9 |

**分析**:
- SSIM 0.790 是所有单模型版本中最高（+15% vs v17）— bilateral split 让 tumor ROI 结构更精准
- HD95 62.9 和 v17 持平
- MSE 偏高 (0.347) — 可能是 bilateral stitch 接缝处的 artifact
- Dice 0.715 > v14，接近 v16

**结论**: Bilateral split 对 SSIM_tumor 贡献巨大，但 MSE 代价较高。适合 ensemble 或作为 SSIM 指标的优化方向。

---

## 22. v26: Wider+Deeper Network (ngf=96, n_blocks=12)

**配置**: v22 基础 + ngf=96 (vs 64) + n_blocks=12 (vs 9)。参数量 ~500M, 权重 2.0GB。
**数据**: data_split_v4 (2528 cases)

**结果 (data_split/test, 199 cases)**:

| Metric | v22 | v26 | v14 | v16 full | kfold_v14 ens |
|--------|:-:|:-:|:-:|:-:|:-:|
| MSE ↓ | 0.196 | **0.181** | 0.212 | 0.218 | 0.174 |
| LPIPS ↓ | 0.101 | **0.100** | 0.126 | 0.124 | 0.131 |
| SSIM ↑ | 0.698 | **0.713** | 0.689 | 0.680 | 0.761 |
| FRD ↓ | 12.70 | 12.47 | 11.90 | **11.51** | 10.68 |
| AUROC ↑ | — | 0.843 | 0.831 | **0.844** | 0.887 |
| Dice ↑ | 0.720 | **0.730** | 0.703 | 0.722 | 0.756 |
| HD95 ↓ | **57.4** | 63.3 | 68.9 | 63.7 | 61.5 |

**分析**:
- v26 在 MSE/LPIPS/SSIM/Dice 上是所有**单模型**中最好
- 更大网络 (ngf 64→96, blocks 9→12) 带来一致性提升
- FRD 偏高 (12.47) — 大模型可能过拟合训练数据的 radiomics 分布
- 权重 2.0GB，Docker 容器约 7.5GB，仍在 10GB 限制内
- v26 kfold ensemble 预期会超过 v14 ensemble

---

## 25. v30: Semi-Disentangled INR-Inspired Generator

**动机**: 借鉴 Implicit Neural Representation (INR) 中 "Semi-Disentangled Spatiotemporal" 的思想——将 DCE-MRI 信号分解为**静态解剖底座**和**动态增强轨迹**。现有 Pix2PixHD 的 residual mode (`output = pre + Δ`) 虽然隐式学了 Δ，但 Δ 无约束，可以在任意位置 hallucinate 增强信号。v30 通过显式解耦，强制 anatomy pass-through + gated enhancement。

### 核心公式

```
I(x,y,t) = anatomy(x,y) + enhancement(x,y) × dynamics(t)

对 MAMA-SYNTH (t=peak 固定):
    output(x,y) = pre(x,y) + gate(x,y) × enhancement(x,y)
```

- `pre(x,y)` — 静态解剖底座，**identity pass-through，不参与生成**
- `gate(x,y) ∈ [0,1]` — 空间注意力图，学习**哪里该有增强**
- `enhancement(x,y) ∈ ℝ` — 增强强度图，学习**增强多少**
- `breast_mask` — 硬约束 gate=0 outside breast（物理不可能在乳腺外增强）

### 架构

```
pre ──┬──────────────────────────────── (identity) ──────────┐
      │                                                       + → output
      └── SharedEncoder ──┬── EnhancementDecoder (heavy) → E  │
          (6 ResBlocks)   │         ×                         │
                          └── GateDecoder (light) → G ∈[0,1] ─┘
                                      ↑
                                 breast_mask (hard gate)
```

| 组件 | 功能 | 复杂度 |
|------|------|--------|
| SharedEncoder | 4× 下采样 + 6 ResBlocks, 提取多尺度特征 | 共享 |
| EnhancementDecoder | 4× 上采样 + 3 ResBlocks + **skip connections** | 重 (高频细节) |
| GateDecoder | 4× 上采样 + 2 ResBlocks, sigmoid, **无 skip** | 轻 (平滑注意力) |

**设计决策**:
1. **Enhancement 有 skip connections**: 需要还原 tumor 的高频纹理
2. **Gate 无 skip connections**: gate 应该空间平滑，不该有 salt-and-pepper noise
3. **Gate × breast_mask**: 硬约束，乳腺外绝对零变化
4. **Shared encoder**: "半解耦" — 两个 decoder 共享解剖理解，但输出结构不同

### 参数量与 VRAM

| | v26 GlobalGenerator | v30 SemiDisentangled |
|---|---|---|
| ngf | 96 | 96 |
| Bottleneck blocks | 12 | 9 (encoder) + 3 (enhance) + 2 (gate) |
| 参数量 | 537.9M | **638M** |
| 模型大小 (FP32) | 2,151 MB | 2,553 MB |
| 推理 VRAM (bs=1, 512×512) | ~4 GB | ~6.7 GB |
| 训练 VRAM (bs=8, A100) | ~35 GB | **~42 GB** |
| T4 16GB (推理) | ✅ | ✅ |
| A100 80GB (训练 bs=8) | ✅ | ✅ |
| Docker 容器估算 | ~7.5 GB | ~8.0 GB (< 10GB) |

### Loss 设计（DisentangledLoss）

| Loss | λ | 功能 | 半解耦作用 |
|------|---|------|-----------|
| `enhance_l1` | 10.0 | L1(output, gt) 全图重建 | 保证合成质量 |
| `anatomy` | 5.0 | L1(output-pre) × (1-gate)，breast 外 × 2 | **锁死解剖** |
| `gate_sparsity` | 0.5 | mean(gate) → encourage sparse | **大部分区域不增强** |
| `gate_tv` | 0.1 | TV(gate) → smooth gate | 防止 salt-and-pepper |
| `tumor_boost` | 5.0 | L1(output×tumor_mask, gt×tumor_mask) | 肿瘤精确重建 |
| `VGG` | 10.0 | VGG perceptual loss | 感知质量 |
| `GAN` | 1.0 | Multi-scale PatchGAN | 分布对齐 |
| `feat` | 10.0 | Feature matching (D 中间层) | 稳定训练 |
| `MSSC` | 20.0 | Multi-Scale Subtraction Consistency | 增强一致性 |

**与 v20 loss 的关键区别**:
- 新增 `anatomy` loss: 在 gate 覆盖之外惩罚任何偏离 pre 的变化
- 新增 `gate_sparsity`: 鼓励稀疏增强（真实 DCE 中只有局部组织强增强）
- 新增 `gate_tv`: 平滑 gate 边界，防止 checker-board artifact
- `MSSC` 从 50 降到 20: 因为 anatomy lock 已经部分承担了背景一致性

### 预期效果（针对 v21 暴露的问题）

| 问题 | v21 表现 | v30 预期 |
|------|---------|---------|
| DUKE_044 (ssim_tumor=-0.48, dice=0) | 肿瘤区域合成失败 | gate 学习增强位置，不会在错误位置 hallucinate |
| DUKE_055 (MSE=1.23) | 背景 intensity 大面积偏移 | anatomy lock 强制 output ≈ pre (outside gate) |
| 推理时 mask 恶化 | 硬 mask composite 产生不连续 | gate 是**可学习的 soft mask**，边界自然过渡 |
| HD95 高 (62.9) | 分割假阳性 | 稀疏 gate 减少非 tumor 区域的 false enhancement |

### 文件

| 文件 | 路径 |
|------|------|
| 架构 + Loss | `models/semi_disentangled_generator.py` |
| 训练脚本 | `models/train_semi_disentangled.py` |
| 推理脚本 | `models/infer_semi_disentangled.py` |
| SLURM 脚本 | `models/scripts/berzelius_train_v30_sdinr.sh` |

### 训练命令

```bash
# Berzelius
cd /proj/berzbiomedicalimagingkth/users/x_honji/mama-synth/src/submission/submission-synthesis/models/scripts
sbatch berzelius_train_v30_sdinr.sh
```

### 推理命令

```bash
python models/infer_semi_disentangled.py \
  --weights $PROJ/checkpoints/mamasynth_v30_sdinr/latest_net_G.pth \
  --input_dir $PROJ/data_multislice/test/mha/input \
  --output_dir $PROJ/predictions_v30_sdinr \
  --breast_seg_model $PROJ/nnUNet_results/Dataset920_BreastSeg2D/nnUNetTrainer__nnUNetPlans__2d \
  --ngf 96 --n_encoder_blocks 9 --n_enhance_blocks 3 --n_gate_blocks 2 \
  --save_gate  # 保存 gate map 用于可视化分析
```

### 验证通过

```
Generator: 638M params (ngf=96, enc=9, enh=3, gate=2)
Output: [2, 1, 512, 512]
Gate: [0.229, 0.882] — sigmoid 有效
Enhancement: [-2.170, 1.408] — 无界残差
Composition check: 0.000000 — 数学公式精确
Gate outside breast: 0.000000 — 硬约束生效
VRAM estimate (inference): ~6.7 GB — T4 OK
VRAM estimate (train bs=8): ~42 GB — A100 OK
```

### 与其他版本的关系

| 版本 | 思路 | v30 如何改进 |
|------|------|------------|
| v20 (residual GAN) | output = pre + Δ (Δ 无约束) | gate 约束 Δ 只在合理位置非零 |
| v21 (bilateral split) | 物理分割左右乳房 | gate 软分割，无缝拼接 |
| v22 (deeper bottleneck) | 更多 ResBlocks 增加感受野 | 分 enhancement 和 gate 两个 decoder |
| v25 (uncertainty) | 网络预测不确定性 | gate 本质上就是"增强确定性"的 proxy |
| INR (原始 INR 项目) | SIREN + FiLM, per-pixel 坐标 | 用 CNN 替代 INR 解码（更快，更适合 2D） |

### 状态: ⏳ 待训练

### 后续计划

1. 训练完成后对比 v22/v26 → 确认解耦是否带来 MSE/Dice 提升
2. 如果 gate map 合理 (肿瘤区域亮，正常组织暗) → 尝试 gate 作为 soft attention 用于 ensemble
3. v30 + SDEdit refiner (v31?) → 在 gate 区域内做 diffusion 精修
4. 如果 v30 base 好于 v26 → 做 K-Fold v30 ensemble

---

## 24. v27: Spatial Weighting 实验 (SAFE-Diff 启发)

**动机**: SAFE-Diff 论文使用 background=1, breast=20, tumor=1000 的空间加权策略。
尝试将此策略应用到我们的 GAN loss 中，替代二元 breast mask。

**两次实验对比**:

| 配置 | sw_breast | sw_tumor | 数据 |
|------|-----------|----------|------|
| v27a (第一次) | 20 | 1000 | data_split_v4 |
| v27b (第二次) | 5 | 500 | data_multislice |

**结果**:

| Metric | v22 (baseline) | v26 (best) | v27a (1/20/1000) | v27b (1/5/500) |
|--------|:-:|:-:|:-:|:-:|
| MSE ↓ | 0.196 | **0.181** | 0.298 | 0.295 |
| LPIPS ↓ | 0.101 | **0.100** | 0.116 | 0.108 |
| Dice ↑ | 0.720 | **0.730** | 0.699 | 0.708 |
| HD95 ↓ | **57.4** | 63.3 | 64.7 | 73.5 |
| SSIM-tumor ↑ | 0.698 | 0.713 | 0.744 | **0.746** |
| FRD ↓ | 12.70 | **12.47** | **11.92** | 12.74 |

**分析**:
- ✅ SSIM-tumor 最高 (0.746) — 空间加权确实改善 tumor 区域重建质量
- ✅ FRD 改善 (v27a: 11.92) — radiomics 特征更接近真实
- ⚠️ MSE 大幅退步 (0.295-0.298) — 权重比太极端，模型忽略全图精度
- ⚠️ 第二次缩小权重比 (500 vs 1000) 改善有限

**结论**: Spatial weighting 对 tumor 区域有帮助，但当前实现的权重比太大。
建议: 在 v26 (ngf=96) 基础上只加大 `--tumor_weight 50` (不用 spatial_weight 机制)，
或使用更温和的比例 (1/3/30)。

**状态**: ✅ 实验完成，方向暂搁

---

## 25. v28: UC-GAN (Uncertainty-Conditioned GAN)

**动机**: SAFE-Diff 启发 — 让 Generator 输出 (μ, log_var)，Discriminator 只判断高置信区域，迫使 G 对不确定区域诚实。

**架构**: GlobalGenerator (ngf=64, n_blocks=12) + uncertainty 双头输出
- `shared`: encoder + ResBlocks (共享特征)
- `mean_head`: 输出增强残差 μ
- `log_var_head`: 输出像素级 log 方差 (clamped [-1.5, 3.0])
- Discriminator 只评估 confidence = exp(-log_var) 加权的区域

**配置**:
| 参数 | 值 |
|---|---|
| netG | global + uncertainty |
| ngf | 64 |
| n_blocks | 12 |
| data | data_multislice/train |
| breast_mask | ✅ |
| intensity_aug | ✅ |
| batchSize | 16 |
| epochs | 200 (100+100) |
| 其余 loss | 同 v22 |

**结果 (data_split/test, 199 cases)**:
| MSE ↓ | LPIPS ↓ | SSIM_tumor ↑ | FRD ↓ | Dice ↑ | HD95 ↓ |
|---|---|---|---|---|---|
| 1.43 | 0.137 | 0.617 | 13.44 | 0.512 | 168.98 |

**分析**: 比 v22 差。Uncertainty 使网络偏保守，tumor 区域增强不足。MSE 和 Dice 都下降。

**状态**: ✅ 完成 — 不采用

---

## 26. v29: Swin Transformer Bottleneck

**动机**: 在 ResBlocks 中间插入 Swin Transformer 捕获长距离解剖依赖（如双侧乳房对称性），在 32×32 特征分辨率上操作。

**架构**: GlobalGenerator (ngf=64, n_blocks=12) + 2 Swin blocks
- 在 12 个 ResBlocks 中间插入 W-MSA + SW-MSA
- window_size=8
- 特征分辨率 32×32 (512/2^4)

**配置**:
| 参数 | 值 |
|---|---|
| netG | global + swin_bottleneck |
| ngf | 64 |
| n_blocks | 12 |
| data | data_multislice/train |
| breast_mask | ✅ |
| intensity_aug | ✅ |
| batchSize | 16 |
| epochs | 200 |
| 其余 loss | 同 v22 |

**脚本**: `berzelius_train_v29_swin.sh`

**结果 (data_multislice_v2 test, 262 cases)**:

| Metric | v22 (orig) | v28b (UC-GAN) | **v29b (Swin)** |
|--------|:---:|:---:|:---:|
| MSE ↓ | **0.598** | 0.780 | 0.780 |
| LPIPS ↓ | **0.131** | 0.151 | 0.136 |
| SSIM_tumor ↑ | **0.531** | 0.464 | 0.478 |
| Dice ↑ | **0.525** | 0.314 | 0.386 |
| HD95 ↓ | **111.9** | 214.8 | 172.3 |

**分析**:
- v29b 比 v28b (UC-GAN) 好，但全面不如 v22 baseline
- LPIPS 0.136 接近 v22 (0.131) — Swin 对感知质量有轻微帮助
- Dice 0.386 远低于 v22 (0.525) — 训练数据量不足 (5.2k vs v22 的原始 2528 cases)
- Swin attention 在数据量不足时无法发挥优势

**结论**: ❌ 不采用。Swin bottleneck 增加了计算但未带来收益。长距离依赖在当前数据规模下不是瓶颈。

**状态**: ✅ 完成 — 不采用

---

## 27. v30: Semi-Disentangled INR-inspired Generator (SD-INR)

**动机**: 分解合成为 anatomy-lock + gated enhancement，灵感来自 INR 的坐标→强度映射。

**核心公式**:
```
output = pre + gate(x,y) × enhancement(x,y)
```
- `SharedEncoder`: 从 pre-contrast 提取特征
- `EnhancementDecoder` (重, skip connections): 预测强度增量 Δ
- `GateDecoder` (轻, smooth): 预测空间注意力 [0,1]
- `breast_mask` 硬约束: gate=0 outside breast

**预期优势**:
- 无背景幻觉 (gate × breast_mask 杀死胸壁区域)
- 更好的 tumor 保真度 (所有容量集中在 enhancement)
- 更低的 MSE 方差 (anatomy lock 消除强度漂移)

**配置**:
| 参数 | 值 |
|---|---|
| 训练脚本 | `train_semi_disentangled.py` |
| ngf | 96 |
| n_downsampling | 4 |
| n_encoder_blocks | 9 |
| n_enhance_blocks | 3 |
| n_gate_blocks | 2 |
| data | data_multislice_v2/train |
| breast_mask | ✅ (hard constraint) |
| batchSize | 8 |
| epochs | 200 |

**Loss**:
| Loss | λ |
|---|---|
| enhance_l1 | 10.0 |
| anatomy | 5.0 |
| gate_sparsity | 0.5 |
| gate_tv | 0.1 |
| tumor | 10.0 |
| vgg | 10.0 |
| gan | 1.0 |
| feat | 10.0 |
| mssc | 50.0 |

**脚本**: `berzelius_train_v30_sdinr.sh`

**状态**: ❌ 放弃

**训练结果**:
- 仅完成 **1 epoch**，耗时 **24 小时**（~8.8s/iteration）
- 原因: SD-INR 每次 iteration 需要 GAN forward + diffusion 多步去噪 + score distillation gradient
- 200 epochs 需 ~200 天，48h SLURM 时限内无法完成
- 权重已下载到本地但仅 1 epoch，未达到收敛，不具备评估价值

**结论**: SD-INR 架构计算成本过高，不适合当前资源限制。
如需类似效果（anatomy-lock + gated enhancement），建议用 v31_pinorm 的 per-image norm 替代（功能类似但零额外计算开销）。

---

## 28. data_multislice_v2 vs v3 对比

### 数据集区别

| | data_multislice_v2 | data_multislice_v3 |
|---|---|---|
| **Slice 选取** | 每个 phase 各选该 phase 的 peak slice（不同 phase 可能选不同 slice 位置） | 选 global peak slice（tumor 面积最大的 slice），然后取 ±2 邻居共 5 个 slice |
| **GT** | 每个 phase 自己的图像作为 GT（phase 1 GT = phase 1, phase 3 GT = phase 3...） | **统一用 global peak phase** 在对应 slice 位置的图像作为 GT |
| **每 patient 样本数** | ~4 (每个 post-contrast phase 1 个) | ~3-5 (peak slice + ±2 邻居，受 min_mask_area 和边界限制) |
| **GT 一致性** | ❌ 不同 phase 增强程度不同（early weak, peak strong） | ✅ 全部是 peak enhancement（模型目标明确） |
| **Train (MAMA-MIA)** | 5227 | 5863 |
| **Train (LAB)** | 3670 (每 phase 各 1 个, 734 slices × 5 phases) | 1179 (每 ROI 所有 slices, peak phase GT) |
| **Train total** | ~8897 | ~7042 |
| **Test samples** | 262 | 299 |
| **Breast mask** | Dataset930 | Dataset930 |
| **LAB tumor mask** | 空 (np.zeros) | **椭圆近似** (from ROI coords) |

### v22 全版本统一结果汇总

#### 所有版本在 data_multislice_v3 test (299 cases) 上的对比

| Metric | v22 (v2 train) | v22 (v3 train) | v26 (v3 train) | **v31_pinorm** | #1 GC Val |
|--------|:---:|:---:|:---:|:---:|:---:|
| MSE ↓ | **1.097** | 1.233 | 1.101 | 1.236 | 0.57 |
| LPIPS ↓ | 0.168 | 0.163 | 0.157 | **0.117** 🏆 | 0.08 |
| SSIM_tumor ↑ | 0.369 | 0.385 | 0.394 | **0.476** 🏆 | 0.43 |
| FRD ↓ | 30.02 | 29.87 | 29.66 | **28.45** 🏆 | 25.06 |
| AUROC ↑ | — | **0.907** | — | 0.831 | 0.80 |
| Dice ↑ | 0.474 | 0.500 | 0.493 | **0.539** 🏆 | 0.48 |
| HD95 ↓ | 127.4 | 135.6 | 144.9 | **125.6** 🏆 | 120.6 |

**版本说明**:
| 版本 | 训练数据 | 架构 | 特殊改动 |
|------|---------|------|---------|
| v22 (v2 train) | data_multislice_v2 (~8.9k, per-phase GT) | ngf=64, blocks=12 | noise_aug |
| v22 (v3 train) | data_multislice_v3 (~7k, peak GT + LAB) | ngf=64, blocks=12 | noise_aug |
| v26 (v3 train) | data_multislice_v3 (~7k) | **ngf=96**, blocks=12 | noise_aug |
| **v31_pinorm** | data_multislice_v3 (~7k) | ngf=64, blocks=12 | **per-image z-score norm** |

**data_multislice_v3 训练数据构成** (~9504 samples, updated 2026-07-05):

| 来源 | Patients | Train Samples | Test Samples | Slice 策略 | GT | Tumor mask |
|------|----------|---------|---------|-----------|----|----|
| DUKE | ~600 | ~2800 | ~150 | peak ± 2, min_area=50 | global peak phase | 自动分割 (3D seg) |
| ISPY2 | ~500 | ~2500 | ~100 | peak ± 2, min_area=50 | global peak phase | 自动分割 (3D seg) |
| YUNNAN | ~50 | ~250 | ~15 | peak ± 2, min_area=50 | global peak phase | 自动分割 (3D seg) |
| NACT | ~30 | ~150 | ~10 | peak ± 2, min_area=50 | global peak phase | 自动分割 (3D seg) |
| LA-Breast | 97 | 1179 | ~25 | all slices per ROI | global peak phase (d1-d5) | **椭圆近似** (ROI coords) |
| **AMBL (new)** | **631** | **2462** | **135** | **peak ± 2** | **global peak phase** | **Enhancement-based** (top 10% enhancing pixels) |
| **Total** | | **9504** | **434** | | | |

**data_multislice_v2 训练数据构成** (~11893 samples, updated 2026-07-05):

| 来源 | Train Samples | Test Samples | Slice 策略 | GT |
|------|---------|---------|-----------|-----|
| MAMA-MIA (DUKE/ISPY2/YUNNAN/NACT) | 5227 | 262 | per-phase peak slice | each phase itself |
| LA-Breast | 3670 | ~0 | per-phase, 5 phases | each phase itself |
| **AMBL (new)** | **2996** | **155** | **per-phase peak slice** | **each phase itself** |
| **Total** | **11893** | **417** | | |

**数据预处理特点**:
- Z-score normalization: MAMA-MIA global stats (mean=104.86, std=215.86) 统一用于所有数据
- Motion cases 排除: 160 cases (from motion_cases.txt)
- Ambiguous FOV 排除: all-different-dimension shapes skipped
- Breast mask: Dataset930 (BreastDivider2D, distilled from 3D model)
- 图像旋转: 90° CCW (thorax at bottom) — MAMA-MIA 数据; AMBL 无旋转 (已是 axial)
- Test split: 5% per source, random seed=42, patient-level
- AMBL tumor mask: enhancement-based (GT - pre, top 10% pixels) 或 DICOM SEG (99 cases)

**关键发现**:
1. **v31_pinorm 在 5/7 指标上最优** — per-image norm 是最有效的单一改进
2. **v26 (ngf=96) MSE 最低** (1.101) — 更大网络对像素精度有帮助
3. **v22 (v2 train) MSE 接近 v26** (1.097) — 可能因为训练量更大 (8.9k vs 7k)
4. **v31 SSIM/Dice 超过 GC #1** — 说明 per-image norm 对 tumor 区域特别有效
5. **MSE 仍是主要差距** — 所有版本 ~1.1-1.2 vs GC #1 的 0.57

#### v22 系列全版本统一结果

##### A. v3 test (299 cases) — 主要对比

| Metric | v22_msv2 | v22_msv3 | v26_msv3 | **v31_pinorm** | #1 GC Val |
|--------|:---:|:---:|:---:|:---:|:---:|
| MSE ↓ | **1.097** | 1.233 | 1.101 | 1.236 | 0.57 |
| LPIPS ↓ | 0.168 | 0.163 | 0.157 | **0.117** 🏆 | 0.08 |
| SSIM_tumor ↑ | 0.369 | 0.385 | 0.394 | **0.476** 🏆 | 0.43 |
| FRD ↓ | 30.02 | 29.87 | 29.66 | **28.45** 🏆 | 25.06 |
| AUROC ↑ | — | **0.907** | — | 0.831 | 0.80 |
| Dice ↑ | 0.474 | 0.500 | 0.493 | **0.539** 🏆 | 0.48 |
| HD95 ↓ | 127.4 | 135.6 | 144.9 | **125.6** 🏆 | 120.6 |

##### B. msv2 test (262 cases) — v28b/v29b 对比

| Metric | v22 (orig) | v28b (UC-GAN) | v29b (Swin) |
|--------|:---:|:---:|:---:|
| MSE ↓ | **0.598** | 0.780 | 0.780 |
| LPIPS ↓ | **0.131** | 0.151 | 0.136 |
| SSIM_tumor ↑ | **0.531** | 0.464 | 0.478 |
| AUROC ↑ | 0.826 | **0.848** | — |
| Dice ↑ | **0.525** | 0.314 | 0.386 |
| HD95 ↓ | **111.9** | 214.8 | 172.3 |

##### C. data_split/test (199 cases) — 原始 test set

| Metric | v22 (orig) |
|--------|:---:|
| MSE ↓ | **0.196** |
| LPIPS ↓ | **0.101** |
| SSIM_tumor ↑ | **0.698** |
| FRD ↓ | **12.70** |
| Dice ↑ | **0.720** |
| HD95 ↓ | **57.4** |

##### 版本配置总览

| 版本 | 训练数据 | ngf | 特殊改动 | 训练状态 |
|------|---------|:---:|---------|:---:|
| v22 (orig) | data_split_v4 (2528) | 64 | baseline deeper (n_blocks=12) | ✅ |
| v22_msv2 | data_multislice_v2 (~8.9k) | 64 | + noise_aug, per-phase GT | ✅ |
| v22_msv3 | data_multislice_v3 (~7k) | 64 | + noise_aug, global peak GT, + LAB | ✅ |
| v26_msv3 | data_multislice_v3 (~7k) | **96** | wider network | ✅ |
| v28b | data_multislice_v2 (~5.2k) | 64 | UC-GAN uncertainty head | ✅ ❌ 不采用 |
| v29b | data_multislice_v2 (~5.2k) | 64 | Swin Transformer bottleneck | ✅ ❌ 不采用 |
| **v31_pinorm** | data_multislice_v3 (~7k) | 64 | **per-image z-score norm** | ✅ 🏆 |
| v22_vgg20 | data_multislice_v3 (~7k) | 64 | lambda_vgg=20, mssc=30 | ⏳ |
| v26_v3 (batch16) | data_multislice_v3 (~7k) | 96 | wider + noise_aug | ⏳ |

##### 关键结论

1. **v31_pinorm 是当前最强** — 5/7 指标最优，SSIM/Dice 超过 GC #1
2. **Per-image norm > 更大网络 (ngf=96)** — v31 LPIPS 0.117 vs v26 0.157
3. **UC-GAN (v28b) 和 Swin (v29b) 均失败** — 不如基础 v22
4. **MSE 仍是最大差距** — 所有版本 ~1.1-1.2 vs GC #1 的 0.57
5. **不同 test set 结果差距大** — data_split/test 上 MSE=0.196，v3 test 上 MSE=1.1+
6. **数据量 vs GT 一致性 trade-off** — v2 数据多但 GT 不一致，v3 数据少但 GT 统一

##### D. v22 vs v26 (ngf=64 vs 96) 跨数据集对比

| 数据集 | Metric | v22 (ngf=64) | v26 (ngf=96) | v26 变化 |
|--------|--------|:---:|:---:|------|
| **data_split_v4 训练** | | | | |
| → data_split/test (199) | MSE ↓ | 0.196 | **0.181** | ✅ -8% |
| | LPIPS ↓ | 0.101 | **0.100** | ✅ -1% |
| | SSIM_tumor ↑ | 0.698 | **0.713** | ✅ +2% |
| | FRD ↓ | 12.70 | **12.47** | ✅ -2% |
| | Dice ↑ | 0.720 | **0.730** | ✅ +1% |
| | HD95 ↓ | **57.4** | 63.3 | ❌ +10% |
| | | | | |
| **data_multislice_v2 训练** | | | | |
| → msv2 test (262) | MSE ↓ | **0.598** | 0.744 (v26b*) | ❌ +24% |
| | LPIPS ↓ | **0.131** | 0.137 (v26b*) | ❌ +5% |
| | SSIM_tumor ↑ | **0.531** | 0.501 (v26b*) | ❌ -6% |
| | Dice ↑ | **0.525** | 0.417 (v26b*) | ❌ -21% |
| | HD95 ↓ | **111.9** | 158.2 (v26b*) | ❌ +41% |
| | | | | |
| **data_multislice_v3 训练** | | | | |
| → v3 test (299) | MSE ↓ | 1.233 | **1.101** | ✅ -11% |
| | LPIPS ↓ | 0.163 | **0.157** | ✅ -4% |
| | SSIM_tumor ↑ | 0.385 | **0.394** | ✅ +2% |
| | FRD ↓ | 29.87 | **29.66** | ✅ -1% |
| | Dice ↑ | **0.500** | 0.493 | ❌ -1% |
| | HD95 ↓ | **135.6** | 144.9 | ❌ +7% |

*v26b 训练时无 breast mask，不公平对比。

**v22 vs v26 结论**:
1. **原始数据上 v26 全面优于 v22** — ngf=96 确实有效 (MSE -8%, Dice +1%)
2. **v26b 在 v2 上崩溃** — 因为无 breast mask，不是架构问题
3. **v3 上 v26 MSE 好但 Dice/HD95 差** — 大网络像素精度高但分割边界不如小网络
4. **v31_pinorm (ngf=64) 仍优于 v26 (ngf=96)** — normalization 策略比网络容量更重要
5. **下一步: v26 + pinorm** — 预期结合两者优势 (大网络降 MSE + pinorm 提升其他指标)

**差距分析**:
- MSE 和 LPIPS 差距最大 (2x) — 像素精度和感知质量是主要短板
- AUROC 和 Dice 我们有优势 — contrast signal 和 tumor detection OK
- **优先改善方向**: 降 MSE 和 LPIPS → v26_v3 (ngf=96) 或 SDEdit refiner

### 降低 MSE 和 LPIPS 的方向

#### 数据层面
- **移除 LAB 数据训练** — LAB intensity range 小，可能让模型学到偏保守输出
- **增大训练量** — 当前 7k，data_multislice v1 有 23k，数据量对 MSE 影响大
- **检查 test set outliers** — 少数极高 MSE case 拉高平均

#### 模型层面

| 方案 | 预期改善 | 代价 | 状态 |
|------|---------|------|------|
| ngf=96 (v26_v3) | MSE -5-10% | 训练 2x | ⏳ 训练中 |
| SDEdit refiner | LPIPS -10-15% | +5s 推理 | 待 base 模型确定后 |
| n_blocks 12→15 | MSE -3-5% | 轻微增加训练时间 | 待尝试 |
| 去掉 noise_aug | MSE 可能改善 | 泛化性下降 | 待 ablation |

#### Loss 层面

| 方案 | 目标 | 做法 | 状态 |
|------|------|------|------|
| 加大 VGG loss | 降 LPIPS | `lambda_vgg 10→20` | ⏳ v22_vgg20 训练中 |
| 减 MSSC 权重 | 给 VGG 让权 | `lambda_mssc 50→30` | ⏳ 同上 |
| 加 L1 loss | 降 MSE | 新增 `lambda_l1 10` | 待尝试 |
| 换 LPIPS 网络 | 直接优化 LPIPS | AlexNet 替代 VGG | 待尝试 |

#### 推理层面

| 方案 | 效果 | 代价 |
|------|------|------|
| Ensemble 多模型平均 | MSE -20-30% (已验证) | 推理 Nx |
| Test-time augmentation | MSE -5-10% | 推理 4-8x |
| SDEdit post-refinement | LPIPS -15% | +5s/case |

#### 优先级
1. 等 v26_v3 (ngf=96) 结果 — 已验证最有效降 MSE 的方法
2. v22_vgg20 (lambda_vgg=20) — 零架构代价，可能直接降 LPIPS
3. 在最佳 base 上加 SDEdit refiner — v17 证明过有效
4. 最终提交用 2-3 模型 ensemble — 降 MSE 最确定的方法

---

## 28b. v26_msv3: Wider Network (ngf=96) on data_multislice_v3 (2026-07-04)

**动机**: v26 (ngf=96, n_blocks=12) 是 data_split_v4 上的最强单模型。在 data_multislice_v3 (global peak GT + LAB + 邻居 slice) 上训练更大网络，验证是否继续受益于更大容量。

**配置**:

| 参数 | v22_msv3 (基线) | v26_msv3 (本次) |
|------|----------------|----------------|
| ngf | 64 | **96** |
| n_blocks | 12 | **12** |
| 参数量 | ~184M | **538M** |
| 权重大小 | ~700MB | **2.0 GB** |
| 数据 | data_multislice_v3 (~7042) | data_multislice_v3 (~7042) |
| breast_mask | ✅ | ✅ |
| noise_aug | ✅ | ✅ |
| intensity_aug | ✅ | ✅ |
| batchSize | 16 | **16** |
| lr | 0.0003 | 0.0003 |
| epochs | 200 | 200 |

**训练**: Berzelius job 17020204 (`v26_msv3`), 运行时间 1d 3h 39min, COMPLETED
**脚本**: `berzelius_train_v26_deeper_v3.sh` (`--name mamasynth_v26_v3`)
**权重**: `latest_net_G-v26_msv3.pth` (2.0 GB, 538M params, 68 keys)

### 结果 (data_multislice_v3/test, 299 cases)

| Metric | v26_msv2 (best, msv2 test) | v26_msv3 (本次, msv3 test) | v22_msv3 (msv3 test) |
|--------|:---:|:---:|:---:|
| MSE ↓ | **0.573** | 1.101 | 1.233 |
| LPIPS ↓ | **0.127** | 0.157 | 0.163 |
| SSIM_tumor ↑ | **0.551** | 0.394 | 0.385 |
| FRD ↓ | — | **29.66** | 29.87 |
| Dice ↑ | **0.537** | 0.493 | 0.500 |
| HD95 ↓ | **110.3** | 144.9 | 135.6 |

### 与 v22_msv3 对比 (同 test set)

| Metric | v22_msv3 (ngf=64) | v26_msv3 (ngf=96) | 变化 |
|--------|:---:|:---:|------|
| MSE ↓ | 1.233 | **1.101** | ✅ -11% |
| LPIPS ↓ | 0.163 | **0.157** | ✅ -4% |
| SSIM_tumor ↑ | 0.385 | **0.394** | ✅ +2% |
| FRD ↓ | 29.87 | **29.66** | ✅ -1% |
| Dice ↑ | **0.500** | 0.493 | ❌ -1% |
| HD95 ↓ | **135.6** | 144.9 | ❌ +7% |

### 分析

1. **vs v22_msv3**: ngf=96 在像素指标 (MSE -11%, LPIPS -4%) 和 tumor 结构 (SSIM +2%) 上有提升，但分割指标 (Dice -1%, HD95 +7%) 略退步。容量增大改善了像素保真度但未改善 tumor 边界精度。

2. **vs v26_msv2**: v26_msv2 在 msv2 test 上全面优于 v26_msv3 在 msv3 test 上的表现。差距主要来自:
   - msv3 test 更难 (包含 LAB 数据、多分辨率 256-896)
   - msv3 训练数据量少于 msv2 (7k vs 8.9k)
   - 大模型在有限数据上可能过拟合

3. **容量增大的收益递减**: 538M params (ngf=96) vs 184M (ngf=64)，参数量增加 2.9x 但 MSE 只降 11%。在 7k 样本的 data_multislice_v3 上，模型容量已经过剩。

### 结论

- ❌ **v26_msv3 不是最佳选择** — 大模型在有限数据 (7k) 上收益不大
- ✅ **v26_msv2 仍是 multislice 系列最佳** — 数据量 (8.9k) 和匹配的 test set
- ⚠️ 如果要在 msv3 数据路线上继续，应恢复邻居 slice 增大训练量而非增大模型

**状态**: ✅ 完成 — 不采用作为最终提交

---

## 28c. v22_vgg20: 加大 VGG Loss 实验 (2026-07-05)

**动机**: 尝试通过加大 VGG perceptual loss 权重直接降低 LPIPS，同时减小 MSSC 权重为 VGG 让权。

**配置** (对比 v22_msv3 基线):

| 参数 | v22_msv3 (基线) | v22_vgg20 (本次) |
|------|----------------|-----------------|
| lambda_vgg | 10 | **20** |
| lambda_mssc | 50 | **30** |
| ngf | 64 | 64 |
| n_blocks | 12 | 12 |
| 数据 | data_multislice_v3 | data_multislice_v3 |
| breast_mask | ✅ | ✅ |
| 其余参数 | — | 同 v22_msv3 |

**训练**: Berzelius job 17022978 (`v22_vgg20`), 运行时间 18h 40min, COMPLETED
**脚本**: `berzelius_train_v22_vgg20.sh` (`--name mamasynth_v22_vgg20`)
**权重**: `latest_net_G-v22_vgg20.pth` (912 MB, 239M params, ngf=64, n_blocks=12)

### 结果 (data_multislice_v3/test, 299 cases)

| Metric | v22_msv2→v3test (vgg=10,mssc=50) | v22_vgg20 (vgg=20,mssc=30) | v26_msv3 (ngf=96) | 变化 vs 基线 |
|--------|:---:|:---:|:---:|------|
| MSE ↓ | **1.097** | 1.573 | 1.101 | ❌ +43% |
| LPIPS ↓ | **0.168** | 0.201 | **0.157** | ❌ +20% |
| SSIM_tumor ↑ | 0.369 | 0.319 | **0.394** | ❌ -14% |
| FRD ↓ | **30.02** | 30.32 | **29.66** | ≈持平 |
| Dice ↑ | 0.474 | **0.496** | 0.493 | ✅ +5% |
| HD95 ↓ | **127.4** | 136.1 | 144.9 | ❌ +7% |

### 分析

1. **LPIPS 反而恶化** (0.168→0.201, +20%): VGG loss 权重过大导致生成器在 VGG 特征空间过拟合，牺牲了像素精度和感知多样性。VGG loss 和 LPIPS 虽然都基于 VGG 特征，但优化 VGG L1 ≠ 优化 LPIPS (后者是 learned distance)。

2. **MSE 大幅退步** (+43%): 减小 MSSC (50→30) 削弱了多尺度减影一致性约束，导致全图像素精度下降。

3. **Dice 微弱改善** (+5%): 可能是 VGG 特征对齐帮助了 tumor 增强信号的空间准确性，但改善幅度不显著。

4. **SSIM_tumor 下降** (-14%): 肿瘤区域结构一致性恶化，说明 MSSC 对 tumor ROI 的贡献比 VGG 更大。

### 结论

- ❌ **加大 VGG loss 不能降低 LPIPS** — 直觉错误，VGG L1 loss 和 LPIPS metric 不等价
- ❌ **减小 MSSC 有害** — MSSC 对全图和 tumor 区域一致性贡献大
- ✅ **维持 lambda_vgg=10, lambda_mssc=50** 仍是最优 loss 配置
- 如需降 LPIPS，应从架构/数据方向入手（如 SDEdit refiner, ensemble），而非调 loss 权重

**状态**: ✅ 完成 — 不采用

---

## 30. v31: Per-Image Z-Score Normalization (pinorm)

**动机**: 消除 cross-scanner intensity variation。每张图用自己 breast-region 的 mean/std 归一化，模型在 canonical space (mean≈0, std≈1) 中学习。推理时 normalize → model → de-normalize 回 global z-score space。

**架构**: 同 v22 (GlobalGenerator, ngf=64, n_blocks=12, residual_mode)
**关键改动**: `--dataset_mode mha_perimage_norm`

**训练流程**:
```
input (global z-score) → 提取 breast region mean/std
    → per-image re-normalize: (input - mean) / std × breast_mask
    → model learns in canonical space
    → GT 也用同样的 mean/std normalize
```

**推理流程**:
```
input (global z-score) → breast mask → per-image mean/std
    → normalize → model → de-normalize (× std + mean)
    → composite: breast = de-normed output, background = original input
```

**配置**:

| 参数 | 值 |
|------|-----|
| 数据 | data_multislice_v3/train (~7042) |
| Dataset mode | mha_perimage_norm |
| Breast mask | Dataset930 (用于定义 foreground) |
| ngf | 64 |
| n_blocks | 12 |
| batch | 16 |
| lr | 0.0003 |
| noise_aug | ✅ |
| intensity_aug | ✅ (scale only, [0.8, 1.2]) |
| 脚本 | `berzelius_train_v31_pinorm.sh` |
| Checkpoint | `mamasynth_v31_pinorm` |

**结果 (data_multislice_v3 test, 299 cases)**:

| Metric | v22_msv3 | v22_msv2 | **v31_pinorm** | #1 GC Val |
|--------|:---:|:---:|:---:|:---:|
| MSE ↓ | 1.233 | **1.097** | 1.236 | 0.57 |
| LPIPS ↓ | 0.163 | 0.168 | **0.117** 🏆 | 0.08 |
| SSIM_tumor ↑ | 0.385 | 0.369 | **0.476** 🏆 | 0.43 |
| FRD ↓ | 29.87 | 30.02 | **28.45** 🏆 | 25.06 |
| AUROC ↑ | **0.907** | — | 0.831 | 0.80 |
| Dice ↑ | 0.500 | 0.474 | **0.539** 🏆 | 0.48 |
| HD95 ↓ | 135.6 | 127.4 | **125.6** 🏆 | 120.6 |

**分析**:
- 🏆 **5/7 指标最优** — LPIPS, SSIM, FRD, Dice, HD95 全部领先
- LPIPS 0.117 vs v22 的 0.163 (**-28%**) — 感知质量大幅提升
- SSIM_tumor 0.476 **超过 GC #1** (0.43)
- Dice 0.539 **超过 GC #1** (0.48)
- MSE 和 v22_msv3 持平 — per-image norm 不影响像素精度
- AUROC 0.831 略低于 v22_msv3 (0.907) — 可能因为 de-norm 后 contrast signal 有轻微偏移

**De-normalization Bug Fix**:
- 初始推理背景 = `0 × std + mean = mean`（错误），导致 MSE=3.28, LPIPS=0.29
- 修复后背景 = 原始 input 值（正确），MSE 降到 1.24, LPIPS 降到 0.12
- 教训: per-image norm 的 de-norm 必须正确处理 mask 边界

**结论**: Per-image normalization 是当前最有效的单一改进。消除了 scanner-specific intensity bias，让模型专注于学习增强 pattern。

**MSE Outlier 分析**:

| 统计 | 值 |
|------|-----|
| Mean MSE | 1.236 |
| **Median MSE** | **0.401** |
| Top 30 cases (10%) 贡献 | **42.4%** 的总 MSE |
| Top 30 平均 MSE | 5.22 |
| 如果修复 top 30 | MSE 降到 **0.75** |
| Outlier 来源 | ISPY2 (20), DUKE (10), LAB (0) |

**Outlier 根因**:
- 这些 cases 的 GT intensity 极高 (max 24-31)，是强增强 cases
- ISPY2_138027: per-image de-norm 放大误差 (input std=3.05, ×3 放大)
- DUKE_244: breast mask 只覆盖 9%，GT 在 breast 区域有极高值
- ISPY2_971492: GT 增强 4.75x，模型追不上
- 共同点: per-image norm 用 mean/std 做 de-norm，极端 intensity 的 case std 很大，de-norm 后误差被放大

**改善方向**:
1. 推理时限制 enhancement 幅度 (clip output - input)
2. 用 robust stats (median/MAD) 替代 mean/std 做 per-image norm
3. 训练时 GT percentile clipping
4. Huber loss 替代 L1/L2

**下一步**:
1. v26_v3 + pinorm (ngf=96 + per-image norm) — 预期进一步降低 MSE
2. v31_pinorm + SDEdit refiner — 叠加 diffusion 精修
3. Ensemble: v31_pinorm + v22_msv3 + v26_v3 — 多模型平均
4. 修复 outlier cases 的 de-norm 策略 — 从 1.24 降到 0.75 的潜力



  
**解决方向**:
最有效（不改训练）：
1. 推理时限制 enhancement 幅度 — output = input + clip(output - input, 0, input.max() * factor)
2. 改善 de-norm 对极端 intensity 的鲁棒性 — 用 robust stats (median/MAD 替代 mean/std)
改训练：
3. Huber loss — 对极端像素误差不那么敏感
4. 训练时 GT clip — gt = clip(gt, -1, percentile_95) 避免模型追极端值

### Bad Case 可视化分析 (2026-07-05)

对 8 个 worst cases 生成了 5 列可视化 (Pre / GT / Pred / |Diff| / Pred+Mask)：
→ `bad_cases_v31_pinorm/*.png`

#### 失败模式 1: 增强信号不足 (Enhancement Under-prediction)

**Cases**: DUKE_641 (SSIM=-0.28), YUNNAN_031, YUNNAN_088

**现象**: GT 有非常强的局部增强（肿瘤明显变亮），prediction 几乎无增强，output ≈ input。

**根因**:
- Per-image norm 后模型学了"保守"residual，不敢产生大幅增强
- YUNNAN 训练量极少 (<5%)，高分辨率 896×896 强增强 pattern 学习不足
- DUKE_641 tumor mask 在乳房深处，模型对该区域的增强建模弱

**改善**: tumor_weight 10→30-50；Enhancement-aware loss；YUNNAN oversampling

#### 失败模式 2: 全图强度偏移 (Global Intensity Shift)

**Cases**: ISPY2_138027 (MSE=8.78), DUKE_244 (MSE=5.84), DUKE_032, DUKE_134

**现象**: 差异热力图几乎全图亮，不是局部错误而是全局 scale 偏移。

**根因**:
- Per-image de-norm 用 breast region mean/std，如果 breast mask 不准或组织比例异常，de-norm 后全图偏移
- ISPY2_138027: GT 增强后脂肪抑制效果导致整体暗化，模型无法预测全局 contrast change
- DUKE_244: 心脏/纵隔区域大，breast mask 可能包含非乳房组织

**改善**: Huber loss (v31b ⏳); robust de-norm (median/MAD); GT clipping

#### 失败模式 3: 下游分割模型失败 (Downstream Segmentation Failure)

**Cases**: DUKE_032 (Dice=0), DUKE_134 (Dice=0), YUNNAN_031/088 (HD95=1267)

**现象**: 合成图看起来尚可，但 nnUNet 返回空 mask → Dice=0。

**根因**:
- 小 tumor 在边缘 + 增强信号不够强 → 分割模型检测不到
- YUNNAN 896→512 resize 后 tumor 只有几像素，分割模型无法检测
- 部分是 evaluation pipeline 局限，不完全是合成质量问题

**改善**: 对小 tumor 加更大权重; multi-scale tumor loss; 评估时这些 case 可能无法根本改善

#### Per-Dataset 表现

| Dataset | n | MSE ↓ | LPIPS ↓ | SSIM_tumor ↑ | Dice ↑ | HD95 ↓ |
|---------|---|:---:|:---:|:---:|:---:|:---:|
| YUNNAN | 25 | **0.033** 🏆 | **0.060** 🏆 | **0.755** 🏆 | 0.242 | 518.6 ❌ |
| DUKE | 55 | 1.566 | 0.124 | 0.313 | 0.369 | 191.7 |
| ISPY2 | 219 | 1.291 | 0.121 | 0.485 | **0.616** 🏆 | **64.2** 🏆 |

- YUNNAN: 像素最优但分割最差（高分辨率 + nnUNet 失效）
- ISPY2: 分割最好（256×256 = 分割模型训练分辨率）
- DUKE: 整体中等，少数 patient 是极端 bad case

#### Dice=0 分布 (64/299 = 21%)

主要来自 DUKE (032/134/244 等) 和 YUNNAN — 合成增强信号不够强或分割模型在特定分辨率上失效。

#### 优化优先级

| 优先级 | 方向 | 预期改善 | 状态 |
|:---:|------|---------|------|
| 1 | Huber loss + GT clipping | MSE ↓ (outlier) | v31b ⏳ 训练中 |
| 2 | tumor_weight 10→30-50 | Dice ↑, SSIM ↑ | 下一版 |
| 3 | 改进 de-norm (robust stats) | MSE ↓ (全局偏移) | 需实验 |
| 4 | SDEdit refiner on v31 | LPIPS ↓, Dice ↑ | v17 已验证 |
| 5 | Ensemble v31 + v26_msv2 | 全指标互补 | 推理时 |

---

## 30b. v31b: Per-Image Norm + Huber Loss + GT Clipping

**动机**: v31_pinorm 的 MSE (1.24) 主要被 10% outlier cases 拉高 (贡献 42% 总 MSE)。这些 cases 的 GT enhancement 极强 (max 24-67)，per-image de-norm 后误差放大。用 Huber loss + GT clipping 降低对极端值的敏感性。

**相对 v31 的改动**:

| | v31_pinorm | v31b |
|---|---|---|
| GT 处理 | 无 clip | **clip to P95** (`gt = clip(gt, -1, max(P95, 3.0))`) |
| Feature/Tumor loss | L1Loss | **SmoothL1Loss (Huber)** |
| 其余 | — | 完全相同 |

**Huber Loss 特性**:
- 误差 < 1: 用 L2 (平滑，梯度小)
- 误差 > 1: 用 L1 (线性，不会被极端值主导)
- 效果: 模型不会为了少数极端 pixel 牺牲整体质量

**GT Clipping 特性**:
- 对 per-image normalized GT，取 breast region 内的 P95 作为上限
- 超过 P95 的极端 pixel 被 clip — 模型不需要追这些 outlier pixels
- 下限 -1.0，上限 max(P95, 3.0)（保证至少允许 3 std 的增强）

**预期效果**:
- Top 30 outlier cases 的 MSE 从 5.22 降到 ~2-3
- 整体 MSE 从 1.24 降到 ~0.8-0.9
- LPIPS/SSIM/Dice 保持或改善（模型不再被极端值分散注意力）

**脚本**: `berzelius_train_v31b_huber.sh`
**Checkpoint**: `mamasynth_v31b`
**推理**: 同 v31 (`infer_perimage_norm.py`)

**结果 (data_multislice_v3/test, 299 cases)**:

| Metric | v31_pinorm | v31b_huber | 变化 |
|--------|:---:|:---:|------|
| MSE ↓ | 1.236 | **1.017** | ✅ -18% |
| LPIPS ↓ | **0.117** | 0.127 | ❌ +9% |
| SSIM_tumor ↑ | **0.476** | 0.337 | ❌ -29% |
| FRD ↓ | 28.45 | **23.81** | ✅ -16% |
| Dice ↑ | **0.539** | 0.298 | ❌ -45% |
| HD95 ↓ | **125.6** | 267.3 | ❌ +113% |

**分析**: Huber loss + GT clipping 成功降 MSE (-18%) 和 FRD (-16%)，但 **Dice 崩溃** (-45%)。GT clipping (P95) 截掉了 tumor 区域的强增强像素 → 模型学到「不要产生强增强」→ nnUNet 检测不到 tumor。FRD 改善是假象（特征分布更平滑，但增强信号被抹掉）。

**结论**: ❌ GT clipping 太激进。如需降 MSE，应只 clip 背景区域，保留 tumor 区域极端值。

**状态**: ✅ 完成 — 不采用

**更新 (2026-07-08)**: v31b_v3 resume 训完 200 epochs 后重新评估：

| Metric | v31_pinorm (200ep) | v31b_v3 (200ep, resume) | 变化 |
|--------|:---:|:---:|------|
| MSE ↓ | 1.236 | **1.221** | ✅ -1% |
| LPIPS ↓ | **0.117** | 0.117 | 持平 |
| SSIM_tumor ↑ | **0.476** | 0.473 | 持平 |
| FRD ↓ | 28.45 | **28.34** | 持平 |
| Dice ↑ | **0.539** | 0.537 | 持平 |
| HD95 ↓ | **125.6** | 135.4 | ❌ +8% |

训完整 200 epochs 后 Huber loss + GT clipping 与 v31_pinorm 性能等价 — 之前 v31b_huber 的崩溃是因为只跑到 ~110 epoch 的中间 checkpoint 不稳定。Huber loss 是中性改动，不提供额外收益。**v31_pinorm 仍为最佳。**

---

## 30c. v31_msv2: Per-Image Norm on data_multislice_v2 (2026-07-06)

**动机**: 验证 per-image norm 在更大数据集 (v2, ~11893 samples) 上是否更好。v2 数据包含 per-phase GT（每个 phase 各自的增强图像作为 GT）。

**配置**:
| 参数 | v31_pinorm (v3 数据) | v31_msv2 (v2 数据) |
|------|:---:|:---:|
| 数据 | data_multislice_v3 (~7042) | data_multislice_v2 (~11893) |
| GT 策略 | Global peak phase (统一) | Per-phase (不同 phase 各自 GT) |
| 其余 | 完全相同 | 完全相同 |

**训练**: Berzelius job 17025748 (`v31_msv2`), 运行时间 ~20h, COMPLETED
**脚本**: `berzelius_train_v31_msv2.sh`

**结果 (v3 test, 434 cases — 含 AMBL)**:

| Metric | v31_pinorm (v3 train) | v31_msv2 (v2 train) | #1 GC Val |
|--------|:---:|:---:|:---:|
| MSE ↓ | 1.236 | **1.022** ✅ | 0.57 |
| LPIPS ↓ | **0.117** | 0.127 | 0.08 |
| SSIM_tumor ↑ | **0.476** | 0.310 ❌ | 0.43 |
| FRD ↓ | 28.45 | **23.66** ✅ 🏆 | 25.06 |
| AUROC ↑ | **0.831** | 0.563 ❌ | 0.80 |
| Dice ↑ | **0.539** | 0.257 ❌ | 0.48 |
| HD95 ↓ | **125.6** | 302.6 ❌ | 120.6 |

**分析**:
- ✅ MSE 最低 (1.022) — 数据量优势 (11893 vs 7042)
- ✅ FRD 23.66 **超过 GC #1** (25.06) — radiomics 分布最真实
- ❌ Dice/HD95/SSIM/AUROC 全崩 — v2 per-phase GT 不一致导致模型学到了模糊的增强 pattern
- ⚠️ Test set 434 cases (含 AMBL)，和之前 299 cases 不完全可比

**结论**: 再次确认 **GT 一致性 > 数据量**。v2 有 1.7× 数据量但 GT 不一致（每个 phase 各自做 GT），模型学不到清晰的 tumor 增强。v3 数据用统一 global peak GT，虽然量少但结果更均衡。

**决策**: 继续在 v3 数据路线上开发。v31_pinorm 仍为最佳版本。

**状态**: ✅ 完成 — 不采用

---

## 30d. v31_auroc: Per-Image Norm + Stronger Tumor/MSSC Loss (2026-07-08) 🏆

**动机**: v31_pinorm 的 bad case 分析显示 tumor 区域增强不足 (Enhancement Under-prediction)。加大 tumor_weight 和 MSSC 权重，强制模型产生更强的 tumor 增强信号。

**配置** (对比 v31_pinorm):

| 参数 | v31_pinorm | v31_auroc |
|------|:---:|:---:|
| tumor_weight | 10 | **20** (2×) |
| lambda_mssc | 50 | **100** (2×) |
| GT clip | 无 | **无** (不 clip，保留强增强) |
| 其余 | — | 完全相同 (ngf=64, n_blocks=12, perimage_norm, noise_aug, intensity_aug) |

**训练**: Berzelius job 17029052 (`v31_auroc`), data_multislice_v3
**脚本**: `berzelius_train_v31_auroc.sh` (`--name mamasynth_v31_auroc`)

### 结果 (data_multislice_v3/test, 299 cases)

| Metric | v31_pinorm | **v31_auroc** | 变化 | #1 GC Val |
|--------|:---:|:---:|------|:---:|
| MSE ↓ | 1.236 | **1.193** | ✅ -3.5% | 0.57 |
| LPIPS ↓ | **0.117** | 0.118 | 持平 | 0.08 |
| SSIM_tumor ↑ | 0.476 | **0.476** | 持平 | 0.43 |
| FRD ↓ | 28.45 | **28.31** | ✅ 微弱 | 25.06 |
| Dice ↑ | 0.539 | **0.565** 🏆 | ✅ +4.8% | 0.48 |
| HD95 ↓ | 125.6 | **112.1** 🏆 | ✅ -10.8% | 120.6 |

### 分析

1. **Dice 0.565 是所有版本最高** — tumor_weight 加倍让模型产生更强增强 → nnUNet 检测更准
2. **HD95 112.1 超过 GC #1 (120.6)** — 分割边界更精确
3. **MSE 也改善 3.5%** — MSSC×2 强化了全图减影一致性
4. **LPIPS/SSIM 不变** — 证明加大 tumor/MSSC 权重不损害感知质量
5. 对比 v31b (GT clip + Huber)：不 clip GT 是关键 — 保留强增强信号让 Dice 从 0.298 恢复到 0.565

### vs GC #1 对比

| Metric | v31_auroc | #1 GC Val | 差距 |
|--------|:---:|:---:|------|
| MSE ↓ | 1.193 | 0.57 | 2.1x ❌ |
| LPIPS ↓ | 0.118 | 0.08 | 1.5x ❌ |
| SSIM_tumor ↑ | **0.476** | 0.43 | ✅ +11% |
| Dice ↑ | **0.565** | 0.48 | ✅ +18% |
| HD95 ↓ | **112.1** | 120.6 | ✅ -7% |

在 tumor 相关指标 (SSIM, Dice, HD95) 上全面超过 GC #1。MSE/LPIPS 仍有差距。

### 结论

**v31_auroc 是当前最佳版本，推荐作为提交候选。** tumor_weight=20 + MSSC=100 是最有效的 loss 调整方向。

**下一步**:
1. v31_auroc + SDEdit refiner → 预期进一步降低 LPIPS
2. v31_auroc + ngf=96 (wider) → 预期降低 MSE
3. Ensemble: v31_auroc + v31_pinorm → 互补

**状态**: ✅ 完成 — **提交候选**

---

## 30e. v32_vflip: Per-Image Norm + Vertical Flip Augmentation (2026-07-09)

**动机**: TTA 实验表明模型对空间翻转不完全等变——hflip TTA (2x) 将 Dice 从 0.371→0.558, HD95 从 239→129。但 inference.py 不能修改加 TTA，因此尝试在训练端加入 vflip 增强让模型自身学到垂直不变性。

### TTA 实验结果（验证方向有效性）

| 模式 | 推理倍数 | MSE ↓ | LPIPS ↓ | SSIM ↑ | FRD ↓ | Dice ↑ | HD95 ↓ |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| none (1x) | 1x | **1.039** | 0.126 | 0.379 | **24.01** | 0.371 | 239.7 |
| hflip (2x) | 2x | 1.201 | 0.118 | 0.495 | 27.09 | **0.558** | **129.2** |
| 4flip (4x) | 4x | 1.168 | 0.124 | **0.518** | 22.99 | 0.531 | 143.8 |
| 8rot (8x) | 8x | 1.179 | 0.128 | 0.513 | **13.51** | 0.484 | 157.8 |

**结论**: TTA 证明模型有空间不变性不足的问题。hflip 已在训练中（但不完美），vflip 完全没有。

### 配置 (对比 v31_pinorm)

| 参数 | v31_pinorm | v31_auroc | **v32_vflip** |
|------|:---:|:---:|:---:|
| tumor_weight | 10 | 20 | 10 |
| lambda_mssc | 50 | 100 | 50 |
| GT clip | ✅ P99 | ❌ | ✅ P99 |
| **--vflip** | ❌ | ❌ | **✅** |
| 其余 | 基线 | 同基线 | 同基线 |

**代码改动**:
- `base_options.py`: 新增 `--vflip` flag
- `mha_perimage_norm_dataset.py`: 50% 概率独立 vertical flip（`tensor.flip(-2)`，独立于 hflip）
- **脚本**: `berzelius_train_v32_vflip.sh`

### 结果 (data_multislice_v3/test, 299 cases)

| Metric | v31_pinorm | v31_auroc | **v32_vflip** | 最优 |
|--------|:---:|:---:|:---:|:---:|
| MSE ↓ | **1.039** ★ | 1.193 | 1.215 | v31_pinorm |
| LPIPS ↓ | 0.126 | 0.118 | **0.117** ★ | v32_vflip |
| SSIM_tumor ↑ | 0.379 | **0.476** ★ | 0.475 | v31_auroc (≈v32) |
| FRD ↓ | **24.01** ★ | 28.31 | 28.48 | v31_pinorm |
| Dice ↑ | 0.371 | **0.565** ★ | 0.554 | v31_auroc |
| HD95 ↓ | 239.7 | 112.1 | **104.9** ★ | v32_vflip |

### v32 vs v31_pinorm 提升幅度

| 指标 | 变化 | 幅度 |
|------|------|------|
| MSE | +0.175 | ❌ +16.8% |
| LPIPS | -0.009 | ✅ -7.3% |
| SSIM_tumor | +0.097 | ✅ +25.6% |
| Dice | +0.182 | ✅ **+49.1%** |
| HD95 | -134.7 | ✅ **-56.2%** |

### 分析

1. **HD95 = 104.9 是所有版本最低** — 分割边界最精确，超越 v31_auroc (112.1) 和 GC #1 (120.6)
2. **Dice 0.554** — 接近 v31_auroc (0.565)，远超基线 v31_pinorm (0.371)
3. **LPIPS 0.117 是所有版本最低** — 感知质量最好
4. **vflip 训练成功吸收了 TTA 的收益** — 无需修改 inference.py 就获得类似 hflip TTA 的 segmentation 改善
5. MSE 和 FRD 退步（与 v31_auroc 类似程度）— segmentation 指标和 pixel 指标之间存在 trade-off

### vs v31_auroc 的 trade-off

| 维度 | v31_auroc 胜 | v32_vflip 胜 |
|------|------------|-------------|
| Dice | ✅ 0.565 vs 0.554 | |
| HD95 | | ✅ **104.9** vs 112.1 |
| LPIPS | | ✅ **0.117** vs 0.118 |
| SSIM | ≈ 持平 (0.476 vs 0.475) | |
| MSE | ✅ 1.193 vs 1.215 | |

两者非常接近。v31_auroc 通过 loss 加权改善 Dice，v32_vflip 通过几何增强改善 HD95/LPIPS。

### 下一步

**合并两者** — v33: tumor_weight=20 + MSSC=100 + 去 GT clip + vflip → 预期 Dice 和 HD95 同时最优。

**状态**: ✅ 完成 — **提交候选**（HD95 最优版本）

---

## 30f. Adaptive Background Offset 实验 (2026-07-09)

**动机**: GT 是 post-contrast 全图，造影剂让心脏血池明显高亮。但 prediction 在 breast mask 外保留 pre-contrast 值（心脏不亮），导致背景区域产生大的 MSE 贡献。

### 背景增强量化 (50 cases)

| 区域 | GT - Pre offset | 说明 |
|------|:---:|------|
| 低强度背景 (air/muscle/fat) | 0.017 | 几乎不增强 |
| 高强度背景 (heart/vessels) | **0.82** | 强增强（血池中造影剂） |
| 所有背景平均 | 0.27 | 被心脏拉高 |
| 当前 BACKGROUND_OFFSET | 0.17 | 不足 |

### 策略对比

| Strategy | MSE | vs baseline |
|----------|:---:|:---:|
| no offset (baseline) | 1.215 | — |
| fixed 0.17 | 1.167 | -3.9% |
| fixed 0.27 | 1.159 | -4.6% |
| adaptive `where(pre>0, 0.8, 0.02)` | 1.051 | -13.5% |
| **adaptive_smooth `clip((pre+0.2)*0.8, 0, 2.0)`** | **0.987** | **-18.8%** |
| Oracle (GT offset) | 0.471 | -61.2% (上界) |

### 最佳策略完整评估 (v32 + adaptive offset)

| 指标 | v32 (无 offset) | v32 + adaptive offset | 变化 |
|------|:---:|:---:|------|
| **MSE ↓** | 1.215 | **0.987** | ✅ **-18.8%** 首次 <1.0 |
| LPIPS ↓ | **0.117** | 0.118 | ≈ 持平 |
| SSIM_tumor ↑ | **0.475** | 0.475 | 持平 |
| FRD ↓ | 28.48 | 28.48 | 持平 |
| Dice ↑ | **0.554** | 0.517 | ❌ -6.7% |
| HD95 ↓ | **104.9** | 126.9 | ❌ +21% |

### 分析

- ✅ MSE 大幅下降，首次突破 1.0
- ❌ Dice/HD95 恶化：adaptive offset 在背景加了强度（包括胸壁肌肉），nnUNet 在背景区域检测到假阳性肿瘤

### 失败根因

offset 公式 `clip((pre+0.2)*0.8, 0, 2.0)` 对**所有**背景像素按 intensity 比例加增强。但胸壁肌肉 intensity 也不低（pre ≈ 0-0.5），被加了 0.2-0.5 的 offset 后看起来像增强组织 → nnUNet false positive → Dice 下降。

### 改进方向

需要**只对心脏血池**加 offset，不影响胸壁肌肉：
1. 用 Dataset910 label=7 精确定位心脏区域
2. 更严格的阈值：只对 pre > 1.0 的像素加 offset（心脏血池在 T1w 上高信号）
3. 训练时去掉 breast mask loss masking，让模型学全图增强（根本解决）

### 结论

Adaptive offset 对 MSE 有巨大帮助（-18.8%），但当前实现太粗糙导致 Dice 恶化。如果能精确定位心脏（Dataset910 label=7），只对心脏区域加 offset ≈ 0.8，可以同时改善 MSE 而不伤害 Dice。

**状态**: ✅ 实验完成 — 方向有价值，需要更精确的心脏定位

---

## 30g. v32 Final Submission Pipeline (2026-07-10) 🏆

**动机**: 合并所有验证有效的 post-processing 改进到最终提交：Dataset930 breast mask（和训练一致）+ Dataset910 heart offset（降 MSE）+ hflip TTA（提升 Dice/SSIM/FRD）。

### Pipeline

```
Input .mha
  → Dataset930 nnUNet (binary) → breast_mask
  → Dataset910 nnUNet (10-class, label=7) → heart_mask (dilated 3px)
  → Per-image z-score normalize (breast foreground: mu, sigma)
  → Pix2PixHD v32_vflip (ngf=64, blocks=12, residual) × 2 (hflip TTA average)
  → De-normalize: synthetic = output_norm × sigma + mu
  → Composite:
      breast region → synthetic
      heart region  → pre + clip(pre * 1.0, 0, 4.0)
      other bg      → pre (unchanged)
  → Output .mha
```

### 结果 (data_multislice_v3/test, 299 cases)

| Metric | v32 (no offset) | v32 + heart (D910) | **v32 Final** | v31_auroc |
|--------|:---:|:---:|:---:|:---:|
| MSE ↓ | 1.215 | 0.999 | **0.970** ★ | 1.193 |
| LPIPS ↓ | 0.117 | 0.115 | **0.116** | 0.118 |
| SSIM ↑ | 0.475 | 0.475 | **0.493** ★ | 0.476 |
| FRD ↓ | 28.48 | 28.48 | **26.88** ★ | 28.31 |
| Dice ↑ | 0.554 | 0.548 | **0.567** ★ | 0.565 |
| HD95 ↓ | **104.9** | 109.9 | 108.5 | 112.1 |

### vs GC #1 对比

| Metric | #1 MamoAnd | **v32 Final** | 差距 |
|--------|:---:|:---:|------|
| MSE ↓ | **0.57** | 0.97 | 1.7x (从 2.1x 缩小) |
| LPIPS ↓ | **0.08** | 0.12 | 1.5x |
| SSIM ↑ | 0.43 | **0.49** | ✅ +14% |
| Dice ↑ | 0.48 | **0.57** | ✅ +18% |
| HD95 ↓ | 120.6 | **108.5** | ✅ -10% |

### 各组件贡献分析

| 组件 | MSE 改善 | Dice 影响 | 来源 |
|------|:---:|:---:|------|
| hflip TTA | -2.4% | +2.4% | 2x 推理平均 |
| Dataset930 breast (vs D910 breast) | ≈0 | 训练一致性 | 和训练时用的 mask 相同 |
| Heart offset (D910 label=7) | **-17.8%** | -1% | 心脏区域 adaptive |
| **总计** | **-20.2%** | **+2.4%** | |

### Docker 提交

| 文件 | 大小 |
|------|:---:|
| `inference.py` (入口) | 7 KB |
| `weights/latest_net_G.pth` (v32_vflip) | 912 MB |
| `weights/breast_seg_930/` (Dataset930) | 354 MB |
| `weights/breast_seg_910/` (Dataset910) | 809 MB |
| **权重总计** | **2.1 GB** |
| **Docker 估算** | **~8.3 GB** (< 10 GB ✅) |

### 推理时间估算 (T4 GPU)

| 步骤 | 时间 |
|------|:---:|
| Dataset930 breast seg | ~1s |
| Dataset910 heart seg | ~1s |
| GAN × 2 (hflip TTA) | ~1s |
| De-norm + composite | <0.1s |
| **总计** | **~3s** (< 10 min ✅) |

**状态**: ✅ **最终提交版本**

---

## 29. v31: Swin + UC-GAN Combined (备用)

**动机**: 合并 v28 (UC-GAN uncertainty) 和 v29 (Swin Transformer bottleneck)。

**架构**:
```
Input → Encoder (4× downsample)
      → 6 ResBlocks
      → 2 Swin Blocks (W-MSA + SW-MSA, window=8)  ← 长距离依赖
      → 6 ResBlocks
      → Shared features
          ├── mean_head → μ (enhancement)     ← uncertainty dual-head
          └── log_var_head → log σ²
```

**配置**: 同 v30 但加 `--uncertainty --swin_bottleneck`
**脚本**: `berzelius_train_v31_swin_ucgan.sh`
**状态**: 待 v30 结果后决定是否训练

---

## 📊 全版本评估结果汇总 (按版本排序)

> 快速参考表。详细分析见各版本对应章节。

### 一、在 data_split/test (199 cases) 上的结果

训练数据: data_split_v4 (2528 axial cases, single peak slice per patient)

| 版本 | ngf | 特殊改动 | MSE ↓ | LPIPS ↓ | SSIM ↑ | FRD ↓ | AUROC ↑ | Dice ↑ | HD95 ↓ |
|------|:---:|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| v14 | 64 | baseline + breast mask | 0.212 | 0.126 | 0.689 | 11.90 | 0.831 | 0.703 | 68.9 |
| v17 | 64 | v14 + SDEdit refiner | 0.216 | **0.109** | 0.688 | **10.54** | 0.829 | **0.731** | **62.8** |
| v20 | 64 | Dataset920 mask (train=infer) | 0.191 | 0.100 | 0.704 | 12.52 | — | 0.718 | 67.0 |
| **v22** | 64 | n_blocks=12 (deeper) | 0.196 | 0.101 | 0.698 | 12.70 | — | 0.720 | 57.4 |
| **v26** | **96** | wider + deeper | **0.181** | **0.100** | **0.713** | 12.47 | 0.843 | **0.730** | 63.3 |

### 二、在 data_multislice_v2 test (262 cases) 上的结果

| 版本 | 训练数据 | ngf | 特殊改动 | MSE ↓ | LPIPS ↓ | SSIM ↑ | AUROC ↑ | Dice ↑ | HD95 ↓ |
|------|---------|:---:|---------|:---:|:---:|:---:|:---:|:---:|:---:|
| v22 (orig) | data_split_v4 | 64 | — | **0.598** | **0.131** | **0.531** | 0.826 | **0.525** | **111.9** |
| v26b | msv2 (无 breast mask) | 96 | — | 0.744 | 0.137 | 0.501 | — | 0.417 | 158.2 |
| v28b | msv2 | 64 | UC-GAN | 0.780 | 0.151 | 0.464 | **0.848** | 0.314 | 214.8 |
| v29b | msv2 | 64 | Swin bottleneck | 0.780 | 0.136 | 0.478 | — | 0.386 | 172.3 |

### 三、在 data_multislice_v3 test (299 cases) 上的结果 — 主要对比

| 版本 | 训练数据 | ngf | 特殊改动 | MSE ↓ | LPIPS ↓ | SSIM ↑ | FRD ↓ | AUROC ↑ | Dice ↑ | HD95 ↓ |
|------|---------|:---:|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| v22_msv2 | msv2 (~8.9k) | 64 | noise_aug, per-phase GT | 1.097 | 0.168 | 0.369 | 30.02 | — | 0.474 | 127.4 |
| v22_msv3 | msv3 (~7k) | 64 | noise_aug, peak GT + LAB | 1.233 | 0.163 | 0.385 | 29.87 | **0.907** | 0.500 | 135.6 |
| v26_msv3 | msv3 (~7k) | **96** | wider + noise_aug | 1.101 | 0.157 | 0.394 | 29.66 | — | 0.493 | 144.9 |
| v31_pinorm | msv3 (~7k) | 64 | per-image z-score | **1.039** | 0.126 | 0.379 | **24.01** | 0.757 | 0.371 | 239.7 |
| **v31_auroc** | msv3 (~7k) | 64 | pinorm + tumor×2 + MSSC×2 | 1.193 | 0.118 | **0.476** | 28.31 | — | **0.565** 🏆 | 112.1 |
| **v32_vflip** | msv3 (~7k) | 64 | pinorm + vflip augmentation | 1.215 | **0.117** 🏆 | 0.475 | 28.48 | — | 0.554 | **104.9** 🏆 |

### 四、GC Validation Phase #1 参考

| Metric | #1 MamoAnd | **v31_pinorm (ours)** | 差距 |
|--------|:---:|:---:|------|
| MSE ↓ | **0.57** | 1.24 | 2.2x ❌ |
| LPIPS ↓ | **0.08** | 0.12 | 1.5x ❌ |
| SSIM_tumor ↑ | 0.43 | **0.48** | ✅ +12% |
| FRD ↓ | **25.06** | 28.45 | ❌ +14% |
| AUROC ↑ | 0.80 | **0.83** | ✅ +4% |
| Dice ↑ | 0.48 | **0.54** | ✅ +13% |
| HD95 ↓ | 120.6 | **125.6** | ≈ 持平 |

⚠️ Test set 不同，不能直接对比。仅作方向性参考。

### 五、版本演进总结

| 阶段 | 版本 | 关键创新 | 结果 |
|------|------|---------|------|
| Baseline | v14 | breast mask + intensity_aug | Dice 0.70 |
| Refiner | v17 | + SDEdit diffusion | Dice 0.73, LPIPS -14% |
| Consistency | v20 | train/infer mask 一致 | MSE -10% |
| Deeper | v22 | n_blocks 9→12 | HD95 -15% |
| Wider | v26 | ngf 64→96 | MSE -8% |
| ❌ UC-GAN | v28b | uncertainty head | 全面退步 |
| ❌ Swin | v29b | transformer bottleneck | 全面退步 |
| **🏆 Per-image norm** | **v31** | **per-image z-score** | **LPIPS -28%, SSIM +24%** |
| **🏆 Loss tuning** | **v31_auroc** | **tumor×2, MSSC×2, no clip** | **Dice 0.565 (+49%), HD95 112** |
| **🏆 Geometric aug** | **v32_vflip** | **+ vertical flip** | **HD95 104.9 (最优), LPIPS 0.117** |

### 六、当前最佳模型

**v31_auroc** — Per-Image Norm + Stronger Tumor/MSSC Loss (Dice 最优)
- 权重: `weights/latest_net_G-v31_auroc.pth` (待下载)
- 推理: `models/infer_perimage_norm.py`
- 优势: Dice 0.565, SSIM 0.476, HD95 112.1 — 全面超过 GC #1

**v32_vflip** — Per-Image Norm + Vertical Flip Augmentation (HD95 最优)
- 权重: `weights/latest_net_G-v32_vflip.pth`
- 推理: `models/infer_perimage_norm.py` (同 v31)
- 优势: HD95 **104.9** (所有版本最低), LPIPS **0.117** (所有版本最低)

**下一步: v33** — 合并 v31_auroc + v32_vflip:
- tumor_weight=20, MSSC=100, no GT clip, + vflip augmentation
- 预期同时获得 v31_auroc 的 Dice 和 v32_vflip 的 HD95

---

## 29. Supervisor 汇报总结 (2026-07-06)

### 最新成果：v31 Per-Image Normalization 🏆

**核心创新**：将传统的 global z-score normalization 替换为 per-image foreground z-score。每张图片独立用 breast region 的 mean/std 归一化，消除跨 scanner 的 intensity 差异。推理时做反变换回原始空间。

**在 data_multislice_v3 test (299 cases) 上的结果**：

| Metric | v31 (ours) | GC 排名 #1 | 对比 |
|--------|:---:|:---:|:---:|
| LPIPS ↓ | 0.117 | 0.08 | 差距缩小到 1.5x |
| SSIM_tumor ↑ | **0.476** | 0.43 | ✅ **超过 #1** |
| Dice ↑ | **0.539** | 0.48 | ✅ **超过 #1** |
| HD95 ↓ | 125.6 | 120.6 | ≈ 持平 |
| MSE ↓ | 1.24 | 0.57 | ❌ 仍有 2x 差距 |

### 已验证方案总结

| 方案 | 结论 |
|------|------|
| **Per-image normalization (v31)** | ✅ 最佳 — SSIM/Dice/LPIPS 全面提升 |
| Wider network ngf=96 (v26) | ✅ MSE 最优，但其他指标不如 v31 |
| 增大 VGG loss (lambda_vgg=20) | ❌ 不能改善 LPIPS |
| UC-GAN uncertainty (v28) | ❌ 全面退步，废弃 |
| Swin Transformer (v29) | ❌ 改善不显著 |
| Semi-Disentangled INR (v30) | ❌ 计算成本过高（1 epoch=24h），放弃 |
| Spatial weighting (v27) | ❌ 改善 SSIM-tumor 但损害 MSE |

### 关键发现

1. **数据量 > 架构创新**：多 slice augmentation (~23k) 比改架构重要
2. **Normalization 是被低估的因素**：per-image norm 零额外计算成本，效果显著
3. **GC 评估的 MSE 差距主要来自 outlier cases**（intensity 异常偏移的 case）
4. **Breast masking 必须训练时启用**——推理时硬 mask composite 反而有害

### 下一步计划

1. **v26 + per-image norm**：结合大网络 (ngf=96) + per-image normalization — 预期两者优势叠加
2. **SDEdit refiner on v31**：用 diffusion 精修 v31 输出，进一步提升边界质量
3. **Docker 提交 GC**：v31 已具备提交条件

---

## 30. Supervisor 方案: T1w Tumor Segmentation + Conditional GAN (2026-07-06)

### 核心思路

从 pre-contrast T1w 上先做乳腺+肿瘤分割，让 GAN 在推理时**显式知道肿瘤在哪**，集中算力在 tumor 区域产生强增强。

### 推理流程

```
Pre-contrast input (.mha)
        │
        ├─→ [BreastDivider 2D] → breast_mask         (已有, Dataset930)
        │
        ├─→ [T1w Tumor Seg] → predicted_tumor_mask   (新, Dataset940)
        │
        ├─→ [Conditional GAN]
        │     输入: concat(pre, breast_mask, tumor_mask) — 3 channels
        │     → synthetic post-contrast
        │
        └─→ output = breast_mask × synthetic + (1-breast_mask) × pre
```

### 实施步骤

| 步骤 | 任务 | 状态 | 预计时间 |
|------|------|------|---------|
| 1 | 训练 T1w → tumor segmentation (nnUNet 2D, Dataset940) | ⏳ 已提交 | ~12h |
| 2 | 对训练集推理 → predicted_tumor_mask/ | 待 Step 1 完成 | ~1h |
| 3 | 修改 GAN input_nc=1→3，训练 v32 (pre + breast + tumor) | 待 Step 2 完成 | ~18h |
| 4 | Docker 推理 pipeline 整合 | 待 Step 3 完成 | 代码改动 |

### Step 1: T1w Tumor Segmentation

**为什么从 pre-contrast 能分割 tumor？**

虽然 tumor 在 T1w pre-contrast 上增强不明显（造影剂的作用就是让 tumor 可见），但 AI 可能利用：
- 信号强度差异（tumor 组织 vs 正常乳腺组织）
- 形态学差异（边界不规则的 mass）
- 组织结构破坏（导管扭曲等）

即使分割精度不高（Dice ~0.3-0.5），只要能提供一个**粗略空间定位**，就能指导 GAN 集中增强。

**训练配置**:
- 模型: nnUNet 2D, fold 0
- Dataset ID: 940 (Dataset940_TumorSegT1w)
- 输入: pre-contrast MHA (1 channel, z-score normalized)
- 标签: binary tumor mask (0=background, 1=tumor)
- 训练数据: data_multislice_v3/train/mha/{input, mask} (~9504 samples，跳过 empty masks)
- 脚本: `berzelius_train_tumor_seg_t1w.sh`

### Step 2: Generate Predicted Masks

训练完后，对**训练集和测试集**分别推理：
```bash
# 对训练集推理（GAN 训练时用 predicted mask，不用 GT）
nnUNetv2_predict -d 940 -c 2d -f 0 \
  -i $PROJ/data_multislice_v3/train/mha/input_nifti/ \
  -o $PROJ/data_multislice_v3/train/mha/predicted_tumor/

# 对测试集推理
nnUNetv2_predict -d 940 -c 2d -f 0 \
  -i $PROJ/data_multislice_v3/test/mha/input_nifti/ \
  -o $PROJ/data_multislice_v3/test/mha/predicted_tumor/
```

**关键设计决策**: GAN 训练时用 **predicted mask**（不用 GT mask），因为：
- 推理时只有 predicted mask 可用（没有 GT）
- 如果训练时用 GT 但推理时用 predicted → 域差距 → 性能下降
- 让 GAN 学会容忍 tumor seg 的噪声和不准确

### Step 3: Conditional GAN (v32)

修改 GAN 输入为 3 channels：

```python
# 当前 (v31): input_nc=1
input = pre_contrast  # [B, 1, H, W]

# 新 (v32): input_nc=3
input = concat(pre_contrast, breast_mask, predicted_tumor_mask)  # [B, 3, H, W]
```

模型架构不变（GlobalGenerator），只改第一层 Conv 的 input channels。

**预期效果**:
- Tumor 区域合成更精准 → SSIM_tumor ↑, Dice ↑
- 非 tumor 乳腺区域不会被过度增强 → 减少 false positive
- 结合 per-image norm (v31) → 两者优势叠加

### Docker 容器资源估算

| 模型 | 大小 | 推理时间 |
|------|------|---------|
| BreastDivider 2D (Dataset930) | ~120 MB | ~1s |
| T1w Tumor Seg (Dataset940) | ~120 MB | ~1s |
| Conditional GAN (v32, ngf=64) | ~730 MB | ~2s |
| **总计** | **~970 MB** | **~4s** |

远在 10GB 容器 / 10min 推理限制内。

### 风险与对策

| 风险 | 对策 |
|------|------|
| T1w tumor seg 精度低 | GAN 训练时用 predicted mask（容忍噪声） |
| Tumor seg 有 false positive → GAN 在错误位置增强 | breast_mask 约束 + residual mode 限制变化幅度 |
| 增加推理复杂度 | 总推理时间 ~4s << 10min 限制 |
| 对无 tumor 的 case 怎么办 | tumor_mask 全 0 → GAN 只靠 breast_mask 合成（退化为 v24） |

### 与已有方案的关系

| 方案 | GAN 输入 | 模型知道 tumor 在哪？ |
|------|---------|---------------------|
| v31 (当前最佳) | 1ch: pre | ❌ 全靠自己学 |
| v24 (mask-as-input) | 2ch: pre + breast_mask | ❌ 只知道乳腺边界 |
| **v32 (supervisor)** | **3ch: pre + breast + tumor** | **✅ 显式告知** |

---

## 31. SDEdit Refiner on v31_pinorm (2026-07-06)

### 动机

v31 在 SSIM/Dice 上已超过 GC #1，但 LPIPS (0.117 vs 0.08) 和 MSE (1.24 vs 0.57) 仍有差距。SDEdit diffusion refiner 在 v17 上证明能提升感知质量 (LPIPS -14%) 和分割精度 (Dice +4%)。将同样方法应用到更强的 v31 base 上。

### 方法

在冻结的 v31 GAN 输出上训练轻量 diffusion denoiser：
- v31 GAN 产生 coarse synthesis
- 加入 30% noise → DDIM 20 步去噪 → 精修 tumor 边界和纹理

### 训练 Pipeline

```
pre-contrast
    │
    ├─→ per-image normalize → frozen v31 GAN → de-normalize → gan_out
    │
    ▼
gt (ground truth)
    │
    ├─→ add noise at timestep t: noisy = √ᾱt·gt + √(1-ᾱt)·ε
    │
    ▼
RefinerUNet([noisy_gt, pre, gan_out], t) → predict ε
Loss = MSE(ε̂, ε)
```

### 推理 Pipeline

```
pre → v31 GAN (pinorm) → gan_out → +30% noise → DDIM 20 steps → refined output
                                                    ↑
                                            RefinerUNet([xt, pre, gan_out], t)
```

推理时间: ~5s/case (GAN 2s + DDIM 3s)，T4 限制内。

### 配置

| 参数 | 值 |
|------|-----|
| Base GAN | v31_pinorm (ngf=64, blocks=12, per-image norm) |
| Refiner | RefinerUNet, 3ch→1ch, base_ch=64, ~18M params |
| Data | data_multislice_v3/train |
| Breast mask | ✅ (GAN forward 用 per-image norm) |
| T (diffusion steps) | 1000 (cosine schedule) |
| DDIM inference steps | 20 |
| Noise strength | 0.3 |
| Epochs | 100 |
| Batch size | 4 |
| LR | 1e-4 |

### 文件

| 文件 | 作用 |
|------|------|
| `models/train_refiner_v31.py` | 训练脚本（处理 per-image norm） |
| `models/infer_refiner_v31.py` | 推理脚本（DDIM sampling） |
| `models/scripts/berzelius_train_refiner_v31.sh` | SLURM 提交脚本 |

### 预期效果

| Metric | v31 (GAN only) | v31 + refiner (预期) | 依据 |
|--------|:-:|:-:|------|
| LPIPS ↓ | 0.117 | ~0.100 | v17 refiner 改善 14% |
| Dice ↑ | 0.539 | ~0.560 | v17 refiner 改善 4% |
| HD95 ↓ | 125.6 | ~118 | 边界精修 |
| MSE ↓ | 1.236 | ~1.1 | 去噪平滑 |

### 状态: ❌ 失败 — 不采用

### 实际结果 (data_multislice_v3 test, 299 cases)

| Metric | v31 (GAN only) | v31 + Refiner | 变化 |
|--------|:---:|:---:|:---:|
| MSE ↓ | **1.236** | 3.202 | ❌ +159% |
| LPIPS ↓ | **0.117** | 0.275 | ❌ +135% |
| SSIM_tumor ↑ | **0.476** | 0.473 | ≈ 持平 |
| FRD ↓ | 28.45 | **28.22** | ✅ 微弱 |
| Dice ↑ | **0.539** | 0.513 | ❌ -5% |
| HD95 ↓ | **125.6** | 136.3 | ❌ +8% |

**失败原因分析**：
1. Per-image normalization 和 diffusion noise schedule 不兼容 — GAN output 经过 de-normalize 后 intensity range 因 case 而异，但 diffusion 的 noise scale 是固定的
2. Noise strength=0.3 可能太大，破坏了 v31 原本好的结果
3. v17 refiner 成功是因为 v14 GAN 用 global z-score（noise scale 一致），v31 用 per-image norm 导致域不匹配

**结论**：SDEdit refiner 不适合 per-image normalized GAN。v31 直接用不加 refiner。

---

## 32. v32 Conditional GAN 结果 (2026-07-09)

### 结果 (data_multislice_v3 test, 299 cases)

| Metric | v31_auroc (之前最佳) | **v32 conditional** | 变化 | GC #1 |
|--------|:---:|:---:|:---:|:---:|
| MSE ↓ | 1.193 | **1.160** | ✅ -3% | 0.57 |
| LPIPS ↓ | 0.118 | **0.117** | ✅ 持平 | 0.08 |
| SSIM_tumor ↑ | 0.476 | **0.492** | ✅ +3% | 0.43 |
| FRD ↓ | **28.31** | 28.46 | ≈ 持平 | 25.06 |
| Dice ↑ | 0.565 | **0.614** | ✅ **+9%** 🏆 | 0.48 |
| HD95 ↓ | 112.1 | **76.0** | ✅ **-32%** 🏆 | 120.6 |

### 关键发现

1. **Supervisor 方案有效** — T1w tumor segmentation (Dice=0.68) + conditional GAN 带来了分割指标的巨大提升
2. **Breast mask composite 是关键** — 没有 composite 时 MSE=3.2/LPIPS=0.29（严重偏移），加了后恢复正常
3. **HD95 76.0 远超 GC #1 (120.6)** — tumor 边界精度大幅改善
4. **Dice 0.614 超过 GC #1 (0.48)** — 合成的 tumor 增强更真实，下游分割更准确

### 推理流程（确认有效的版本）

```python
# 1. 读取 input
arr = read_mha(input_path)

# 2. Breast mask + Tumor mask
breast_mask = BreastDivider(arr)
tumor_mask = TumorSeg_Dataset940(arr)

# 3. Per-image normalize
mu, sigma = arr[breast_mask > 0.5].mean(), arr[breast_mask > 0.5].std()
arr_norm = (arr - mu) / sigma * breast_mask

# 4. 3-channel input → GAN
input_3ch = concat(arr_norm, breast_mask, tumor_mask)
out_norm = GAN(resize_512(input_3ch))

# 5. De-normalize
synthetic = resize_back(out_norm) * sigma + mu

# 6. Breast mask composite（关键！）
output = breast_mask * synthetic + (1 - breast_mask) * arr
```

### 配置

| 参数 | 值 |
|------|-----|
| 架构 | GlobalGenerator, input_nc=3, ngf=64, n_blocks=12 |
| Normalization | Per-image z-score (breast foreground) |
| 训练数据 | data_multislice_v3/train (~9504 samples) |
| Tumor mask | Dataset940 predicted (Dice=0.68, 从 T1w pre-contrast) |
| Breast mask | Dataset930 BreastDivider 2D |
| Loss | GAN + feat×10 + VGG×10 + MSSC×50 + tumor×10 |
| Composite | ✅ breast_mask × synthetic + (1-breast_mask) × pre |

### 状态: ✅ 新最佳模型 — 准备 Docker 提交

---

## 33. v32 Final 配置对比 (2026-07-09)

### 三种 v32 后处理配置对比

| Metric | v32 hard mask | v32 blur (σ=5) | **v32 final (blur+TTA)** | GC #1 |
|--------|:---:|:---:|:---:|:---:|
| MSE ↓ | 1.160 | 1.158 | **1.127** | 0.57 |
| LPIPS ↓ | **0.117** | 0.122 | 0.123 | 0.08 |
| SSIM_tumor ↑ | 0.492 | 0.492 | **0.509** | 0.43 |
| FRD ↓ | 28.46 | 28.26 | **27.06** | 25.06 |
| Dice ↑ | 0.614 | 0.621 | **0.624** | 0.48 |
| HD95 ↓ | 76.0 | **67.3** | 69.4 | 120.6 |

### 各后处理效果分析

| 后处理 | 改善 | 损害 |
|--------|------|------|
| Gaussian blur (σ=5) | HD95 -12%, Dice +1% | LPIPS +4% |
| hflip TTA | MSE -3%, SSIM +3%, FRD -5%, Dice +0.5% | HD95 +3%, LPIPS +1% |
| blur + TTA 合计 | MSE -3%, SSIM +3%, FRD -5%, Dice +2% | LPIPS +5%, HD95 -9% (vs hard) |

### 提交策略

GC 使用 Mean Position 排名（8 个指标各自排名后取均值）。

- **主提交**: v32 blur + TTA — 综合最强，4/6 指标最优
- **备选提交**: v32 blur (no TTA) — HD95 最佳 (67.3)，LPIPS 更好 (0.122)

### vs GC #1 (MamoAnd) 超越的指标

| Metric | GC #1 | v32 final | 超越幅度 |
|--------|:---:|:---:|:---:|
| SSIM_tumor ↑ | 0.43 | **0.509** | ✅ +18% |
| Dice ↑ | 0.48 | **0.624** | ✅ +30% |
| HD95 ↓ | 120.6 | **69.4** | ✅ -42% |

仍落后: MSE (2x gap), LPIPS (1.5x gap), FRD (轻微)

### 最终推理流程 (Docker)

```
Input → BreastDivider → breast_mask
      → TumorSeg (D940) → tumor_mask
      → Per-image z-score (breast foreground)
      → GAN(3ch) × 2 (original + hflip) → average
      → De-normalize
      → Gaussian blur (σ=5) soft mask composite
      → Output
```

推理时间: ~8s/case (T4 GPU), 远在 10min 限制内。

---

## 34. MSE 根因分析与改进方案 (2026-07-09)

### 发现

v32 final 的 MSE (1.127) 主要来自 **breast mask composite 的副作用**，而不是 GAN 合成质量差。

| Case | MSE total | MSE breast (GAN 质量) | MSE outside (composite 误差) |
|------|:-:|:-:|:-:|
| ISPY2_111344 | 5.12 | **1.36** (合理) | **6.93** ❌ (巨大) |
| DUKE_244 | 5.78 | 4.16 | **5.94** ❌ |
| Good case (median) | 0.41 | ~0.3 | ~0.1 |

### 根因

```
GT (post-contrast) 在乳腺外 ≠ pre-contrast input

原因: DCE-MRI 造影剂不只增强 tumor，全身组织都有轻微增强
  - 胸壁肌肉、血管、淋巴结都会摄取造影剂
  - GT 乳腺外 mean = 1.94, input 乳腺外 mean = 0.62 (差距 1.3!)

我们的 composite: output_outside = pre-contrast (不变)
但 GT 期望:     output_outside = pre-contrast + 全身轻微增强
→ MSE 在乳腺外区域爆炸
```

### Case-level 统计

- MSE median = 0.406（很好），mean = 1.127（被 outliers 拉高）
- Top 10% outliers 贡献了 MSE 的 35%
- 13% cases MSE > 3.0，主要是 breast mask 小（DUKE_244 breast 只占 9%）或 GT 增强信号强的 case

### 改进方案

| 方案 | 做法 | 预期 MSE | 风险 |
|------|------|---------|------|
| **A: 去掉 composite** | GAN 直接输出全图 | ~0.8-0.9 | HD95 可能退步（胸壁 noise） |
| **B: 增大 blur sigma** | sigma 5→20，覆盖更多过渡区 | ~1.0 | 效果有限 |
| **C: 全图 GAN + soft attenuation** | GAN 合成全图，soft mask 控制强度 | ~0.8 | 需调参 |
| **D: 训练时不 mask GT 乳腺外** | 让 GAN 学习全身轻微增强 | ~0.7 | 需重训 |
| **E: 两阶段 composite** | breast 内用 GAN，breast 外用 input + learned offset | ~0.8 | 复杂 |

### 方案 A 详细说明（最快验证）

```python
# 当前:
result = soft_mask * synthetic + (1 - soft_mask) * pre  # 乳腺外 = pre

# 方案 A: 不做 composite
result = synthetic  # GAN 直接输出全图（residual mode: pre + delta）
```

GAN 的 residual mode (`output = input + delta`) 在乳腺外应该自动学到接近 0 的 delta，
但因为训练时有 breast mask loss masking，乳腺外的 delta 可能不够准确。

### 方案 D 详细说明（最彻底，需重训）

修改训练逻辑：
```python
# 当前训练: loss 只在 breast 内计算
loss = L1(output * breast_mask, gt * breast_mask)

# 方案 D: 全图 loss，但 breast 内权重更高
loss = L1(output, gt) + tumor_weight * L1(output * tumor_mask, gt * tumor_mask)
# 去掉 breast_mask masking，让模型学全身增强
```

### 决策

- 短期（提交用）: 先试方案 A，如果 HD95 没退步太多就用
- 中期: 方案 D 重训 v32b（全图 loss），预期 MSE 大幅下降

### 方案验证结果 (2026-07-09)

#### 方案 A: 去掉 composite（GAN 直接输出）

| Metric | v32 final (blur+TTA) | v32 no composite | 变化 |
|--------|:---:|:---:|:---:|
| MSE ↓ | **1.127** | 3.170 | ❌ +181% |
| LPIPS ↓ | **0.123** | 0.294 | ❌ +139% |
| SSIM_tumor ↑ | 0.509 | 0.509 | ≈ 持平 |
| Dice ↑ | **0.624** | 0.565 | ❌ -9% |
| HD95 ↓ | 69.4 | 73.8 | ≈ 持平 |

**结论: ❌ 失败。** 原因：v32 训练时 input 第一通道 = `arr_norm * breast_mask`（乳腺外=0），
residual mode 在乳腺外输出 `0 + delta`，de-normalize 后变成 `delta*sigma+mu`——完全错误。
Composite 是**必须的**（因为训练时乳腺外被 mask 为 0）。

#### 方案 B: 增大 blur sigma (5→20)

| Metric | v32 final (σ=5) | v32 blur σ=20 | 变化 |
|--------|:---:|:---:|:---:|
| MSE ↓ | **1.127** | 1.157 | ≈ 持平 |
| LPIPS ↓ | **0.123** | 0.127 | ❌ 略升 |
| SSIM_tumor ↑ | **0.509** | 0.500 | ❌ 略降 |
| FRD ↓ | 27.06 | **26.48** | ✅ 微改善 |
| Dice ↑ | **0.624** | 0.616 | ≈ 持平 |
| HD95 ↓ | **69.4** | 76.5 | ❌ 退步 |

**结论: ❌ σ=20 过大。** 模糊范围过广导致 tumor 边界不清晰 → SSIM/HD95 退步。

#### 所有 composite 方案总结

| 方案 | MSE | LPIPS | SSIM | Dice | HD95 | 推荐 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Hard mask | 1.160 | **0.117** | 0.492 | 0.614 | 76.0 | LPIPS 最佳 |
| **Blur σ=5 + TTA** | **1.127** | 0.123 | **0.509** | **0.624** | 69.4 | **🏆 综合最佳** |
| Blur σ=5 (no TTA) | 1.158 | 0.122 | 0.492 | 0.621 | **67.3** | HD95 最佳 |
| Blur σ=20 + TTA | 1.157 | 0.127 | 0.500 | 0.616 | 76.5 | ❌ 过度模糊 |
| No composite | 3.170 | 0.294 | 0.509 | 0.565 | 73.8 | ❌ 失败 |

**最终提交版本确认：v32 blur σ=5 + hflip TTA。**

### 与 GC #1 差距的重新理解

GC #1 的 MSE=0.57 很可能是因为他们：
1. 没有做 breast mask composite（直接输出全图）
2. 或者用了更大范围的 soft mask
3. 模型自然学到了全身轻微增强

我们的 composite 虽然帮助了 Dice/HD95（tumor 边界更清晰），但严重伤害了 MSE。
这是一个 **Dice vs MSE 的 trade-off**。
