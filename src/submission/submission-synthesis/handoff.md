# Pix2PixHD for MAMA-SYNTH — Handoff

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
| **v16** | **data_split_v4 axial** | **2528** | **✅ ensemble (ResEnc+Plain OR)** | **✅** | **v14 + ensemble mask, 200ep → 🏆 最佳** |

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

| Metric | v5 | v9 | v11 | v12 | v13* | v14 | v16 | v17 (v14+SDEdit) | v20 | KFold Ens. | Best |
|--------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|------|
| MSE ↓ | 0.618 | 0.270 | 0.246 | 1.017 | 1.193 | 0.212 | 0.218 | 0.216 | 0.191 | **0.174** | 🏆 Ensemble |
| LPIPS ↓ | 0.148 | 0.118 | 0.125 | 0.238 | 0.208 | 0.126 | 0.124 | 0.109 | **0.100** | 0.131 | 🏆 v20 |
| SSIM_tumor ↑ | 0.361 | 0.599 | 0.581 | 0.321 | 0.384 | 0.689 | 0.680 | 0.688 | 0.704 | **0.761** | 🏆 Ensemble |
| FRD ↓ | 10.75 | **9.41** | 9.41 | 10.32 | 12.02 | 11.90 | 11.51 | 10.54 | 12.52 | 10.68 | v9/v11 |
| AUROC ↑ | 0.867 | 0.837 | 0.828 | 0.891 | **0.892** | 0.831 | 0.844 | 0.829 | — | 0.887 | v12/v13 |
| Dice ↑ | 0.388 | 0.561 | 0.565 | 0.363 | 0.319 | 0.703 | 0.722 | 0.731 | 0.718 | **0.756** | 🏆 Ensemble |
| HD95 ↓ | 186.7 | 115.4 | 108.1 | 203.9 | 276.9 | 68.9 | 63.7 | 62.8 | 67.0 | **61.5** | 🏆 Ensemble |

*v13/v14/v16 在 199 axial cases 上评估
*v16 full = 200 epochs with ensemble breast mask (ResEncUNetL f0 OR PlainConvUNet f4)
*KFold Ens. = 4-model ensemble on 1353 cases (DUKE+ISPY2+YUNNAN), 每 case 由 3 个未见过它的模型平均

**v16 分析**: ensemble breast mask (200 epochs) 在 Dice (+2.7%) 和 HD95 (-7.5%) 上超越 v14。
**结论: v16 为最佳提交版本。** Dice 0.722, HD95 63.7。

**v20 分析**: 使用 distilled 2D mask (Dataset920) 替代 3D multi-channel mask，消除训练/推理 mask 域差。
- MSE 0.191 (新最佳, -10% vs v14), LPIPS 0.100 (新最佳, -8% vs v17), SSIM_tumor 0.704 (新最佳)
- Dice 0.718 / HD95 67.0: 略低于 v17 (0.731/62.8)，但高于 v14 (0.703/68.9)
- FRD 12.52: 最差之一，说明 radiomics feature 分布偏移稍大
- **像素精度和感知质量全面超越所有版本**，分割指标接近最佳

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

## 10. 当前执行状态 (2026-06-17 11:09)

### Berzelius 上正在跑的 Jobs
- 19 个 GPU job 在排队 (mask_0_80 ~ mask_1440_1520)
- 执行方案 A: `mask_and_preprocess.py` + Dataset932 (4ch 3D)
- 输出到: `data_split_v5/train/`
- 排除 160 个 motion cases
- 每个 job 处理 80 patients, 预计 1h/job

### 本地已完成
- `data_split_v5/train/mha/` 在本地有 1506 cases (方案B Step1, 无mask的干净数据)
- `data_split_v5/train/mha/breast_mask/` 有 1506 个 Dataset910 2D mask (本地生成)
- `data_split_v5_lius/` 是从 Berzelius 同步回来的旧结果 (1049 cases, 无mask)

### 等待中
- GPU jobs 开始执行后，`data_split_v5` 会被重写为方案 A 的结果(masked)
- 全部完成后 → 用 v12 脚本训练

### 可以同时在 CPU 上做的
- 在 Berzelius CPU node 上用 Dataset910 2D 给本地 preprocess 结果生成 breast mask (~60min)
- 这样方案 B 的数据也准备好了，训练时用 --breast_mask_dir 即可

### 下一步
1. 等 GPU jobs 完成 → data_split_v5 方案A数据就绪
2. 或在 CPU 上跑 Dataset910 breast mask → 方案B数据就绪
3. 用完整数据训练 v12 (方案A) 或新版本 (方案B)
4. 评估 → 选最佳 → 提交 GC

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
| Epochs | 50 (quick test: niter=25 + niter_decay=25) |
| 其余超参 | 同 v14 (MSEC=50, tumor=10, etc.) |

### 状态: ⏳ 待训练

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

## 19. K-Fold Cross-Validation 结果 (v14 config, 无 LABREAST)

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

**状态**: ⏳ 待训练

**后续**: 若 12 blocks 有效，可叠加 ngf=96（方案3）进一步提升。

---

## 19. v23: Local Enhancer 全分辨率精修

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

**预期效果**:
- 高频细节改善 → LPIPS ↓, SSIM ↑
- Tumor 边界更锐利 → Dice ↑, HD95 ↓
- 确定性 forward（不像 SDEdit 有随机性），不会降低结构一致性
- 推理额外耗时 < 0.5s

**状态**: ⏳ 待训练

---

## 20. K-Fold Ensemble 结果

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

## 21. v24: Breast Mask 作为条件输入 (Mask-as-Input)

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
