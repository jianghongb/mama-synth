# Breast Masking 方案对比 (供 Supervisor 讨论)

## 方案 A: 在 Preprocess 阶段嵌入 Breast Mask

```
原始 3D NIfTI (pre + post phases + segmentation)
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  Preprocess Pipeline (修改版)                         │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Step 1-6: 正常流程 (提取 2D slice, 计算 subtraction) │
│         │                                           │
│         ▼                                           │
│  Step 6.5 [新增]: 生成 Breast Mask                   │
│                                                     │
│    pre + GT ──→ 构造 4 通道 ──→ Dataset932 (nnUNet)  │
│                  P0, P1,          │                  │
│                  d_early, d_late   ▼                 │
│                              breast_mask (label 1+2+3)
│         │                                           │
│         ▼                                           │
│  Step 7: Z-score 归一化                              │
│                                                     │
│  Step 7.5 [新增]: 应用 Breast Mask                   │
│    pre_norm  = pre_norm × breast_mask  (胸壁→0)     │
│    sub_norm  = sub_norm × breast_mask  (胸壁→0)     │
│         │                                           │
│         ▼                                           │
│  Step 8: 保存 MHA                                    │
│    input.mha       ← masked pre (胸壁已置零)         │
│    ground_truth.mha ← masked subtraction            │
│    mask.mha         ← tumor mask (不变)             │
│    breast_mask.mha  ← breast mask [新增输出]         │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**优点:**
- 数据预处理完即可直接训练，不需要额外步骤
- 保证训练数据一致性（所有 case 都经过相同处理）

**缺点:**
- 修改原始数据，不可逆（除非保留未 mask 版本）
- 换 mask 策略需要重跑整个 preprocess（~1h/dataset）
- Dataset932 需要 GPU，增加 preprocess 硬件要求
- 推理时仍然没有 post-contrast，无法用 Dataset932 做 mask

---

## 方案 B: Preprocess 不变，训练时动态加载 Mask (当前实现)

```
Step 1: Preprocess (不修改，只跑一次)
─────────────────────────────────────
原始 3D NIfTI ──→ 2D MHA (input, ground_truth, mask)
                  (干净数据，无 masking)

Step 2: 生成 Breast Mask (独立步骤，可反复执行)
─────────────────────────────────────
generate_breast_masks.py
  --input_dir mha/input
  --gt_dir mha/ground_truth
  --model_type 932
  --model_dir Dataset932/...
       │
       ▼
  breast_mask/ 目录 (每个 case 一个 .mha)

Step 3: 训练 (通过参数控制是否用 mask)
─────────────────────────────────────
python train.py \
  --dataroot /path/to/data \
  --breast_mask_dir /path/to/breast_mask  ← 有这个参数就用 mask
                                          ← 没有就不用

训练时 dataset 自动:
  input  = input × breast_mask   (胸壁→0)
  gt     = gt × breast_mask      (胸壁→0)
  loss 只在 breast 区域计算
```

**优点:**
- 原始数据不被修改
- 灵活：随时开关 mask，换模型重新生成即可
- 可以 A/B 对比有无 mask 的效果
- Preprocess 不需要 GPU

**缺点:**
- 需要额外步骤（generate_breast_masks.py）
- 训练前需要确认 mask 已生成

---

## 推理阶段 (两种方案都一样)

```
推理时只有 pre-contrast，无法用 Dataset932。

选项:
  A) 不用 breast mask (当前 v8，效果最好)
  B) 用 Dataset910 (单通道 T1，可在推理时跑)
  C) 用简单阈值 (Otsu 自适应)
```

---

## 建议

| 场景 | 推荐方案 |
|------|---------|
| 快速实验对比 | **方案 B** (灵活，当前已实现) |
| 最终生产 pipeline | 方案 A 或 B 均可 |
| 推理时 | 不用 mask (v8 证明不需要) |

**关键问题请 Supervisor 确认：**
1. 训练时 breast mask 的价值：v8 (无 mask) 已是最佳，v7/v9 (有 mask) 反而更差。是否还要继续探索？
2. 如果要用 Dataset932 做 mask，只能在训练阶段用（推理时没有 post-contrast）。这个限制是否可接受？
3. 是否需要在 preprocess 阶段就固化 mask，还是保持训练时灵活加载？
