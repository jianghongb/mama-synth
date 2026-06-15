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

### 评估结果: data_split_v2/test (150 cases)

| Metric | v5 | v6 | v7 | v8 |
|--------|:-:|:-:|:-:|:-:|
| MSE ↓ | 0.283 | 0.87 | 0.89 | **0.258** |
| LPIPS ↓ | 0.074 | 0.114 | 0.138 | **0.071** |
| SSIM_tumor ↑ | 0.701 | 0.423 | 0.407 | **0.718** |
| FRD ↓ | 10.41 | 12.78 | **9.85** | **9.77** |
| AUROC ↑ | 0.930 | 0.926 | 0.926 | 0.929 |
| Dice ↑ | 0.684 | 0.492 | 0.437 | **0.708** |
| HD95 ↓ | 55.6 | 105.8 | 115.8 | **37.7** |

### 评估结果: data_split/test (144 cases) — v5 vs v9

| Metric | v5 | v9 |
|--------|:-:|:-:|
| MSE ↓ | **0.641** | 0.841 |
| LPIPS ↓ | 0.163 | **0.133** |
| SSIM_tumor ↑ | 0.351 | **0.447** |
| FRD ↓ | **5.51** | 9.92 |
| Dice ↑ | 0.320 | **0.472** |
| HD95 ↓ | 235.9 | **124.8** |

### Yunnan 外部验证 (100 cases, 独立数据)

| Metric | v5 | v6 | v7 |
|--------|:-:|:-:|:-:|
| MSE ↓ | 0.393 | 0.399 | **0.125** |
| LPIPS ↓ | 0.235 | 0.290 | **0.120** |
| SSIM_tumor ↑ | 0.405 | 0.324 | **0.482** |
| FRD ↓ | 29.55 | 30.45 | **28.56** |
| AUROC ↑ | 0.879 | **0.934** | 0.844 |
| Dice ↑ | **0.095** | 0.076 | 0.090 |
| HD95 ↓ | **574.7** | 584.6 | 651.2 |

### 关键发现

1. **v8 本地测试集最强**: intensity aug 全面提升，HD95 -32%
2. **v7 外部泛化最强**: Yunnan 上 MSE 降 3×, LPIPS 降一半
3. **v6 < v5**: motion data 拉低质量
4. **v9 tumor 指标最强**: Dice +47%, SSIM_tumor +27%（但 MSE/FRD 变差）
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
