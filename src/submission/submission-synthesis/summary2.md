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

| 版本 | 关键配置 | 本地 MSE | 状态 |
|------|---------|:-:|------|
| v1 | baseline residual | 1.55 | ✅ 完成 |
| v2 | 全改 (失败) | 1.90 | ❌ 弃用 |
| v3 | v1 + MSSC=10 | 1.32 | ✅ 已提交 GC (排27) |
| v4 | v3 + square_only + tumor=40 | 1.61 | ❌ 弃用 |
| **v5** | MSEC=50 (论文方法) + resize | **0.25** | ✅ 待提交 GC |
| **v6** | v5 + data_split_v2 (1356 cases, 含 motion) | — | ⏳ 训练中 |

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
