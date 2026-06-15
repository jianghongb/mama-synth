# MAMA-SYNTH Project Summary (2026-06-14)

## 输入/输出

### 训练数据
- **路径**: `/Users/ehogjig/git/kth/data_split/train/mha/`
- **结构**: `input/` (pre-contrast), `ground_truth/` (subtraction), `mask/` (tumor)
- **数量**: 1074 cases (DUKE 161, ISPY2 776, ISPY1 94, NACT 43)
- **格式**: MHA, float32, z-score normalized
- **分辨率**: 256×256 (64%), 384×384, 448×448, 512×512, 256×60 (sagittal)

### 测试数据
- **本地**: `/Users/ehogjig/git/kth/data_split/test/mha/` — 144 cases
- **GC validation**: 70 cases (含 P_* 49 + RV_07_* 21 未知数据)

### GC 提交格式
- Input: `/input/images/pre-contrast-dce-mri-slice-breast/<uuid>.mha`
- Output: `/output/images/synthetic-contrast-dce-mri-slice-breast/output.mha`

## 模型

### 架构: Pix2PixHD GlobalGenerator (Residual Mode)
- Encoder: 4× downsample (512→32), ngf=64→1024
- Bottleneck: 9× ResNet blocks
- Decoder: 4× upsample (32→512)
- Output: `pre + Δ` (residual)
- Params: ~174M, weights: 696MB

### 训练配置 (v5, 最佳)
| 参数 | 值 |
|------|-----|
| 数据 | 1074 cases, resize to 512×512 |
| Epochs | 200 (100 stable + 100 decay) |
| Batch size | 8 |
| LR | 0.0002 (Adam, β1=0.5) |
| GAN loss (LSGAN) | λ=1.0 |
| Feature matching | λ=10 |
| VGG perceptual | λ=10 |
| MSEC (multi-scale subtraction) | **λ=50** |
| Tumor L1 | λ=10 |
| SSIM loss | 0 (disabled) |

### 推理配置
- 所有 case resize → 512×512 → model → resize back (关键改动！)
- Pad to 16 倍数 (512 时 no-op)
- 输出保留原始 spatial metadata

## Loss 组件

| Loss | 作用 | 权重 |
|------|------|------|
| G_GAN | 对抗分布匹配 | 1.0 |
| G_GAN_Feat | D 特征匹配 | 10 |
| G_VGG | VGG perceptual | 10 |
| **G_MSSC** | 多尺度 subtraction L1 (avg_pool 3级) | **50** |
| G_Tumor | Tumor mask L1 | 10 |

### MSEC Loss (核心改进)
```
L_msec = Σ_{r=1,1/2,1/4} ||down_r(fake-pre) - down_r(gt-pre)||_1
```
- 直接监督 enhancement map (subtraction)
- 在 3 个尺度 (512, 256, 128) 上强制一致性
- 来源: Yang et al. MICCAI 2026 论文

## 实验结果

### 本地评估对比 (144 test cases)

| Metric | v1 (baseline) | v3 (v1+MSSC=10) | **v5 (MSEC=50, best)** | GC #1 |
|--------|:-:|:-:|:-:|:-:|
| MSE ↓ | 1.55 | 1.32 | **0.25** | 0.81 |
| LPIPS ↓ | 0.253 | 0.242 | **0.079** | 0.098 |
| SSIM_tumor ↑ | 0.270 | 0.277 | **0.722** | 0.501 |
| FRD ↓ | 11.52 | 11.88 | **11.19** | 23.10 |
| AUROC contrast | 0.858 | 0.854 | **0.906** | 0.813 |
| Dice ↑ | 0.251 | 0.271 | **0.646** | 0.487 |
| HD95 ↓ | 173.5 | 186.6 | **103.8** | 107.4 |

### GC Validation 实际排名 (70 cases)

**v3 + resize (排名 27th, 2026-06-14)**
- Mean Position: 24.3
- MSE=1.28, LPIPS=0.26, SSIM_tumor=0.22, FRD=31.71
- AUROC_contrast=0.82, AUROC_tumor=0.54, Dice=0.34, HD95=222
- 推理已包含 resize to 512
- 主要瓶颈: LPIPS=0.26 (pos 36), FRD=31.71 (pos 38)
- 原因: RV_07 未知数据 domain gap + 模型 MSEC 权重不够 (λ=10)

**v3 无 resize (排名 30th, 2026-06-13)**
- Mean Position: 25.4
- MSE=1.22, LPIPS=0.26, SSIM_tumor=0.23, FRD=31.88
- Dice=0.31, HD95=257

### 关键改进历程
1. **v1→v3**: 加 MSSC loss (Laplacian pyramid, λ=10) — 小幅改善
2. **v3→v5**: 修正 MSEC 为 avg_pool + λ=50 — 巨大提升
3. **推理 resize**: 不 resize 时 LPIPS 0.24, resize 后 0.08 — 决定性改动

## 版本追踪

| 版本 | 关键配置 | 数据 | 本地 MSE | 本地 LPIPS | 本地 SSIM_t | 本地 Dice | 状态 |
|------|---------|------|:-:|:-:|:-:|:-:|------|
| v1 | baseline residual | 1074 | 1.55 | 0.253 | 0.270 | 0.251 | ✅ 完成 |
| v2 | 全改 (失败) | 1074 | 1.90 | 0.258 | 0.272 | 0.244 | ❌ 弃用 |
| v3 | v1 + MSSC=10 | 1074 | 1.32 | 0.242 | 0.277 | 0.271 | ✅ 已提交 GC (排30) |
| v4 | v3 + square_only + tumor=40 | 938 | 1.61 | 0.257 | 0.270 | 0.241 | ❌ 弃用 |
| **v5** | MSEC=50 + resize 推理 | 1074 | **0.25** | **0.079** | **0.722** | **0.646** | ✅ 已提交 GC (排27) |
| **v6** | v5 + data_split_v2 (含 motion) | 1356 | 0.87 | 0.114 | 0.423 | 0.492 | ✅ 完成 (比v5差) |
| **v7** | v6 + breast mask (loss only in breast) | 1356 | 0.89 | 0.138 | 0.407 | 0.437 | ✅ 完成 |
| **v8** | v5 + intensity_aug (scale+bias) | 1074 | — | — | — | — | ⏳ 训练中 |
| **v9** | v5 + breast mask (loss only in breast) | 1074 | — | — | — | — | ⏳ 训练中 |

## 进行中

### 加 Motion Data 重训练增强泛化 (v6)

**目标**：提高模型在 GC 未知数据 (RV_07) 上的泛化能力

**做法**：
- 使用 data_split_v2：包含 motion cases 的完整数据集
- Train: 1356 cases (原 1074 + ~282 motion/新增)
- Test: 150 cases (10% 均匀抽样，DUKE 29 + ISPY2 98 + ISPY1 17 + NACT 6)
- 训练配置与 v5 完全一致 (MSEC=50)

**动机**：
- v5 本地评估已全面超越 leaderboard #1
- 但 v3 提交 GC 后排名 27（RV_07 未知数据严重拖分）
- 更多数据 + 更多样化的输入可能改善 domain gap

**风险**：
- Motion cases 的 GT 可能不够准确（pre/post 不对齐）
- 如果噪声太大可能反而降低质量

**验证计划**：
- 训练完后在 data_split_v2/test (150 cases) 上评估
- 与 v5 对比，确认没有退步
- 如果提升则提交 GC

### 加入 AMBL 外部数据 (Advanced-MRI-Breast-Lesions)

**来源**: TCIA, 99 patients with SEG, 1.5T GE, 以色列

**处理状态**:
- DICOM → NIfTI (multi-phase split): ✅ 485 cases
- NIfTI → MHA (preprocess.py): ⏳ 待执行

**预期效果**: 5 个 post-contrast phases 提供真实的增强幅度多样性，替代人工 intensity augmentation，改善 OOD 泛化

### v7: Breast Masking (训练时 loss 只在乳房区域计算)

**改动**：v6 基础上，用 nnUNet (Dataset910_BreastSegNet) 生成 breast mask，训练时 loss 只在 breast region 内计算，忽略胸壁区域。

**结果 — Test Set (data_split_v2/test, 150 cases)**：

| Metric | v5 | v6 | v7 |
|--------|-----|-----|-----|
| MSE ↓ | 0.25 | 0.87 | 0.89 |
| LPIPS ↓ | 0.079 | 0.114 | 0.138 |
| SSIM_tumor ↑ | 0.722 | 0.423 | 0.407 |
| FRD ↓ | 11.19 | 12.78 | **9.85** |
| AUROC ↑ | 0.906 | 0.926 | 0.926 |
| Dice ↑ | 0.646 | 0.492 | 0.437 |
| HD95 ↓ | 103.8 | 105.8 | 115.8 |

**结果 — Yunnan 外部验证 (100 cases)**：

| Metric | v5 | v6 | v7 |
|--------|-----|-----|-----|
| MSE ↓ | 0.393 | 0.399 | **0.125** |
| LPIPS ↓ | 0.235 | 0.290 | **0.120** |
| SSIM_tumor ↑ | 0.405 | 0.324 | **0.482** |
| FRD ↓ | 29.55 | 30.45 | **28.56** |
| AUROC ↑ | 0.879 | 0.934 | 0.844 |
| Dice ↑ | 0.095 | 0.076 | 0.090 |
| HD95 ↓ | 574.7 | 584.6 | 651.2 |

**分析**：
- v7 在外部验证集 (Yunnan) 上**大幅领先**：MSE 降 3 倍，LPIPS 降一半，SSIM_tumor 提升 19%
- 但在 test set 上 MSE/LPIPS/SSIM 不如 v5（注意：v5 metrics 可能是在旧 test set 上跑的，不完全公平）
- FRD 9.85 是所有版本在 test set 上的最佳，说明整体分布更真实
- Breast mask 策略有效提升了外部泛化能力
- AUROC 略降可能因为胸壁保持 pre-contrast 值

### 可能的改进方案 (v8+)

1. **排除 motion 数据重训练 (v5v2)**：用 data_split_v2 但排除 149 个 motion cases (1207 cases)，验证 motion 数据是否是退化原因
2. **v7 + 排除 motion**：breast mask + 无 motion 数据
3. **提交 v7 到 GC**：Yunnan 外部验证结果优异，可能在 GC 未知数据上也表现好

## 代码仓库

| 仓库 | 用途 |
|------|------|
| `github.com/jianghongb/mama-synth` (dev) | 主工程，含 submission-synthesis |
| `github.com/jianghongb/SimulatingDCE` (main) | Pix2PixHD 训练代码 |

### 关键路径
- 训练代码: `mama-synth/src/submission/submission-synthesis/models/`
- 推理入口: `mama-synth/src/submission/submission-synthesis/inference.py`
- Berzelius 脚本: `models/scripts/berzelius_train_v3_mssc.sh`
- Docker: `mama-synth/src/submission/submission-synthesis/Dockerfile`

## 下一步

1. **提交 v5 到 GC validation** (剩余 2 次机会)
2. 如果 GC 结果好 → 提交 final test phase
3. 如果 GC 结果差 (RV_07 拖分) → 考虑:
   - 加 motion data 重训练增强泛化
   - Data augmentation (随机平移/旋转)
   - Delta scale 后处理 (需要 per-dataset 调参)
