# Breast Masking 方案 — 最终决策 & 实现

## 决策摘要

| 阶段 | 方案 | 模型 | 说明 |
|------|------|------|------|
| **数据准备** | 方案 A: Preprocess 阶段嵌入 | Exp4x Dataset932 (4ch, 3D) | 高质量 mask，去胸壁后保存 |
| **训练** | Loss 限制在 breast 区域 | — | 训练数据已 masked |
| **推理** | 不用 mask | — | 纯 Generator 直通 |

**理由：**
- 训练时用高质量 3D mask 去胸壁 → 模型学会只增强乳腺区域
- 推理时不需要额外分割模型 → Docker 体积小（~1.5 GB），推理快
- v9 结果证明 breast mask 训练对 tumor 指标有显著提升

---

## 实现: mask_and_preprocess.py

```
原始 3D NIfTI (multi-phase DCE + tumour segmentation)
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  mask_and_preprocess.py                              │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Step 1: 加载所有 phase + tumour seg                 │
│         │                                           │
│         ▼                                           │
│  Step 2: 构造 4 通道输入                              │
│    P0 = phase[0] (pre-contrast)                     │
│    P1 = phase[1] (first post)                       │
│    d_early = P1 - P0                                │
│    d_late = phase[-1] - P1                          │
│         │                                           │
│         ▼                                           │
│  Step 3: Exp4x (Dataset932, nnUNet 3D fullres)      │
│    输入: 4ch → 输出: breast(1) + FGT(2) + tumor(3)  │
│    breast_mask = label ∈ {1, 2, 3}                  │
│         │                                           │
│         ▼                                           │
│  Step 4: 去胸壁                                      │
│    all_phases × breast_mask → masked phases         │
│         │                                           │
│         ▼                                           │
│  Step 5: 选 peak phase + 最大 tumour slice           │
│         │                                           │
│         ▼                                           │
│  Step 6: Z-score 归一化 + 旋转 90° CCW              │
│         │                                           │
│         ▼                                           │
│  Step 7: 保存                                        │
│    mha/input/         ← masked pre-contrast (2D)    │
│    mha/ground_truth/  ← masked peak-enhancement     │
│    mha/mask/          ← tumour mask                 │
│    mha/breast_mask/   ← breast mask (备用)           │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 数据清洗: Motion 排除

- **Motion list**: `src/preprocessing/motion_cases.txt` (160 cases)
- 通过 `--exclude_list` 参数在预处理时排除
- 来源: 手动标注的存在 motion artifact 的 cases

---

## 运行方式

### 本地测试 (Mac MPS, ~2 min/case)

```bash
python src/preprocessing/mask_and_preprocess.py \
    --image_dir /path/to/images \
    --seg_dir /path/to/segmentations/automatic \
    --output_dir /path/to/output \
    --global_stats src/preprocessing/training_pre_stats.json \
    --breast_model_dir /path/to/exp4x_for_maia/Dataset932/nnUNetTrainer__nnUNetPlans__3d_fullres \
    --skip_ambiguous_shapes \
    --exclude_list src/preprocessing/motion_cases.txt
```

### Berzelius GPU (SLURM, ~10s/case)

```bash
sbatch src/preprocessing/run_mask_preprocess.sh
```

---

## 推理阶段 (无 mask)

```
Input.mha → resize 512 → Pix2PixHD (residual) → resize back → Output.mha
```

- 不需要 nnUNet breast segmentation
- Docker 只需打包 Generator 权重 (696 MB)
- 推理时间 << 10 min/case on T4

---

## 实验结论

### Breast mask 对训练的影响

| 对比 | 数据 | SSIM_tumor | Dice | HD95 |
|------|------|:-:|:-:|:-:|
| v5 (无 mask) | data_split | 0.351 | 0.320 | 235.9 |
| **v9 (有 mask)** | data_split | **0.447** | **0.472** | **124.8** |
| 提升 | | +27% | +47% | -47% |

### Yunnan 外部验证

| 对比 | SSIM_tumor | Dice | FRD |
|------|:-:|:-:|:-:|
| v5 (无 mask) | 0.405 | 0.095 | 29.55 |
| **v9 (有 mask)** | **0.818** | **0.474** | **24.94** |

**结论**: Breast mask 在训练阶段的作用明确 — 大幅提升 tumor 区域合成质量和下游分割指标。推理时不需要 mask 因为模型已经学会只在乳腺区域产生增强信号。
