# MAMA-SYNTH 项目综述 (Master Thesis 铺垫)

## 研究问题

乳腺 DCE-MRI 检查需要注射钆基造影剂（Gadolinium-based Contrast Agent, GBCA），存在肾毒性风险和高昂成本。**Virtual Contrast Enhancement (VCE)** 旨在从无造影的 pre-contrast T1w MRI 直接合成 post-contrast 增强图像，实现"无钆造影"。

## 方法：Pix2PixHD + 残差模式

采用条件 GAN 架构（Pix2PixHD GlobalGenerator），以残差模式学习增强信号：`output = pre + Δ`。模型只需学习造影增强的差异（Δ），背景自动保持不变，收敛更快。

**Loss 组合**：GAN (1×) + Feature Matching (10×) + VGG Perceptual (10×) + Multi-Scale Subtraction Consistency (50×) + Tumor L1 (10×)

## 关键创新

### 1. Breast Masking 策略

**问题**：胸壁区域不吸收造影剂（pre ≈ post），但模型可能在胸壁产生错误增强，浪费容量且产生假阳性。

**方案**：训练时用 nnUNet breast segmentation（Dataset910, 10类）生成 breast mask，所有 loss 仅在乳房区域内计算。推理时不需要 mask — 模型已学会只增强乳房。

**效果**：v5→v9, MSE -56%, Dice +45%, HD95 -38%

### 2. 多域数据整合与质量控制

整合 7 个数据源（DUKE, ISPY2, ISPY1, NACT, LA-Breast, Yunnan, AMBL），通过：
- **Motion detection**：Phase correlation 检测 pre/post 之间的位移，结合 breast mask 区分真实乳房 motion 和胸壁信号变化
- **方向过滤**：GC 只评估 axial cases，去除 sagittal 数据（ISPY1, NACT, AMBL）
- **强度校准**：外部数据用统一的 z-score 统计量归一化（mean=104.86, std=215.86）

### 3. Intensity Augmentation

训练时随机缩放和偏移输入强度（scale ∈ [0.7, 1.3], bias ∈ [-0.2, 0.2]），模拟不同扫描仪的对比度差异。在相同数据上 HD95 改善 32%。

## 实验版本演进

| 阶段 | 版本 | 关键改动 | 效果 |
|------|------|---------|------|
| Baseline | v5 | MSEC=50, 1074 cases | 本地远超 GC #1 |
| 数据扩展 | v6 | +motion data | ❌ 退化（motion 噪声） |
| Breast mask | v7, v9 | Loss masking | ✅ 外部泛化大幅提升 |
| Augmentation | v8 | Intensity aug | ✅ 全面提升 |
| 多域+全组合 | v11 | 2811 cases + mask + aug | ✅ 综合优秀 |
| 预处理去胸壁 | v12 | Dataset932 3D mask (全部数据) | ❌ 推理不匹配 |
| Axial+3D mask | v13 | v12 去掉 ISPY1/NACT (1236 axial cases) | ❌ 同上 |
| **Axial+loss mask** | **v14** | **v11 去掉 sagittal (2528 axial, mask+aug)** | **🏆 最佳 (Dice 0.703, HD95 68.9)** |
| Ensemble mask | v16 | v14 + ensemble breast mask (50 epochs) | 欠训练，潜力待验证 |

## 最终推理方案

v14 推理**不使用 breast mask**：训练时 loss mask 已让模型学会不增强胸壁。
- 输入：z-score float32 MHA → resize 512 → Pix2PixHD → resize back → 输出
- 无需 nnUNet、无需后处理、无需 mask 合并
- Docker 容器只需 Generator 权重 (696 MB)

## v13 vs v14 Ablation

| | v13 | v14 |
|---|---|---|
| Breast mask 方式 | 预处理去胸壁 (Dataset932 4ch 3D) | 训练时 loss mask (Dataset910 2D) |
| Intensity aug | ❌ | ✅ |
| 数据量 | 1236 | 2528 |
| 额外域 | 无 LABREAST | 含 LABREAST |
| 结果 | ❌ domain gap | **🏆 Dice 0.703, HD95 68.9** |

如果 v13 < v14，进一步证明 "loss masking + 更多数据 > 预处理去胸壁 + 少数据"。
如果 v13 > v14，说明 Dataset932 的高精度 mask 能弥补数据量劣势。

## 核心发现

1. **训练时 breast mask > 推理时 breast mask**：让模型学会"在哪增强"比后处理更有效
2. **数据质量 > 数据数量**：去掉 motion/sagittal 后性能提升
3. **域多样性帮助泛化**：7域训练在外部 Yunnan 数据上表现优于单域
4. **预处理去胸壁 ≠ loss masking**：前者改变了输入分布导致推理 domain gap，后者保持输入完整性

## Grand Challenge 评估

8 个指标 Mean Position 排名：MSE, LPIPS, SSIM_tumor, FRD, AUROC_contrast, AUROC_tumor, Dice, HD95。其中 Dice/HD95 通过"合成图→分割→vs GT"间接评估合成质量，breast masking 对此帮助最大。

## 论文贡献点

1. 证明 breast masking 作为训练策略优于推理后处理
2. 提出 breast-masked motion detection 方法恢复被错误排除的训练数据
3. 系统性的多域数据整合框架（7 个异质数据源的质量控制流程）
4. Axial-only 训练策略的必要性验证

## 关键数值结果 (199 axial cases, 无 sagittal)

| Metric | v5 (baseline) | v9 (mask) | v11 (multi-domain) | v14 (axial-only) 🏆 |
|--------|:-:|:-:|:-:|:-:|
| MSE ↓ | 0.618 | 0.270 | 0.246 | **0.212** |
| LPIPS ↓ | 0.148 | **0.118** | 0.125 | 0.126 |
| SSIM_tumor ↑ | 0.361 | 0.599 | 0.581 | **0.689** |
| FRD ↓ | 10.75 | **9.54** | 9.41 | 11.90 |
| Dice ↑ | 0.388 | 0.561 | 0.565 | **0.703** |
| HD95 ↓ | 186.7 | 115.4 | 108.1 | **68.9** |

## 时间线

- 6/25: GC Test phase 开启
- 7/10: 最终提交截止
- 8/1: 结果发布
- 9/27: MICCAI 2026 Deep-Breath Workshop 颁奖
