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

data_split_v4 = DUKE + ISPY1 + ISPY2 + NACT + LA-Breast + Yunnan + AMBL (7 域)

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
| v14 | data_split_v4 | 2528 | ✅ loss mask | ✅ | v11 去掉 sagittal (axial only) ⏳ |

### 评估结果: data_split_v2/test (150 cases)

| Metric | v5 | v6 | v7 | v8 | v9 |
|--------|:-:|:-:|:-:|:-:|:-:|
| MSE ↓ | 0.283 | 0.87 | 0.89 | **0.258** | 0.284 |
| LPIPS ↓ | 0.074 | 0.114 | 0.138 | **0.071** | 0.115 |
| SSIM_tumor ↑ | 0.701 | 0.423 | 0.407 | 0.718 | **0.731** |
| FRD ↓ | 10.41 | 12.78 | 9.85 | **9.77** | 9.85 |
| AUROC ↑ | 0.930 | 0.926 | 0.926 | 0.929 | — |
| Dice ↑ | 0.684 | 0.492 | 0.437 | **0.708** | 0.680 |
| HD95 ↓ | 55.6 | 105.8 | 115.8 | **37.7** | 54.1 |

### 评估结果: data_split/test — 全版本对比

| Metric | v5 | v9 | v11 | v12 | v13* | v14* | v16* | Best |
|--------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|------|
| MSE ↓ | 0.618 | 0.270 | 0.246 | 1.017 | 1.193 | **0.212** | 0.469 | v14 |
| LPIPS ↓ | 0.148 | 0.118 | 0.125 | 0.238 | 0.208 | 0.126 | **0.119** | v16 |
| SSIM_tumor ↑ | 0.361 | 0.599 | 0.581 | 0.321 | 0.384 | **0.689** | 0.448 | v14 |
| FRD ↓ | 10.75 | **9.41** | 9.41 | 10.32 | 12.02 | 11.90 | 10.11 | v11 |
| AUROC ↑ | 0.867 | 0.837 | 0.828 | 0.891 | 0.892 | 0.831 | **0.854** | v12/v13 |
| Dice ↑ | 0.388 | 0.561 | 0.565 | 0.363 | 0.319 | **0.703** | 0.507 | v14 |
| HD95 ↓ | 186.7 | 115.4 | 108.1 | 203.9 | 276.9 | **68.9** | 135.5 | v14 |

*v13/v14/v16 在 199 axial cases 上评估（test set 去掉了 ISPY1/NACT sagittal cases）
*v16 只训练了 50 epochs（其余为 200 epochs）

**v14 分析**: 去掉 sagittal 数据(ISPY1+NACT+AMBL)后，模型专注 axial → Dice +24%, HD95 -36%。
**v16 分析**: ensemble breast mask 有潜力(LPIPS/FRD 更好)，但 50 epochs 欠训练导致 Dice/SSIM 落后。
**结论: v14 为最佳提交版本。** Dice 0.703, HD95 68.9 远超其他版本。

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

## 7. 待办 / 下一步

- [ ] v10/v11 训练完成后评估
- [ ] 选择最佳版本提交 GC
- [ ] 考虑 ensemble (v8 精度 + v7/v9 泛化)
- [ ] Attention branch (Supervisor 建议, 让 G 学会定位 lesion)

---

## 8. GC 评估指标重要性 & Breast Masking Pipeline 流程

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

## 13. v8 评估结果 → 已合并到「4. 训练版本对比」

**v8 = 当前最佳模型** (v5 + intensity_aug)
- 本地: MSE=0.26, LPIPS=0.071, SSIM=0.718, Dice=0.708, HD95=37.7
- Yunnan: MSE=0.009, LPIPS=0.045, SSIM=0.759
- 详见 Section 4 对比表格

---

## 14. Breast Segmentation 模型对比

### 可用模型 (weights/breast_seg/)

| | Dataset910 (PlainConv 2D) | Dataset910 (ResEncUNet 2D) | Dataset932 (3D fullres) |
|---|---|---|---|
| 路径 | `nnUNetTrainer__nnUNetPlans__2d` | `nnUNetTrainer__nnUNetResEncUNetLPlans__2d` | `nnUNetTrainer__nnUNetPlans__3d_fullres` |
| 输入 | 1ch: T1 (pre-contrast) | 1ch: T1 (pre-contrast) | 4ch: P0, P1, d_early, d_late |
| 输出 | 10类 (tissue, vessel, muscle, bone, lesion, lymphnode, heart, liver, implant) | 同左 | 4类 (breast, FGT, tumor) |
| 训练数据 | 973 cases | 973 cases | 1280 cases |
| 架构 | 标准卷积 | **残差编码器 (更精确)** | 3D fullres |
| 推理时可用 | ✅ | ✅ | ❌ (需要 post-contrast) |
| Breast mask 提取 | label 1+2+5+6+9 | label 1+2+5+6+9 | label 1+2+3 |
| 适用场景 | 推理时 breast mask | 推理时 breast mask (更好) | 训练时 breast mask (最准) |

### Dataset933 (计划中)
- 目标: 单通道 T1 输入，输出 breast/FGT/tumor (3类)
- 方法: 用 Dataset932 生成 pseudo labels → 训练 nnUNet 2D
- 优势: 推理时也能做精确的 breast+FGT+tumor 分割
- 状态: ⏳ 待训练 (berzelius_train_breast_seg.sh)

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

## 15. 当前执行状态 (2026-06-17 11:09)

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

## 10. Axial-Only 训练发现 (v13/v14)

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

## 11. v16 Breast Mask Ensemble Pipeline

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
