# MAMA-SYNTH 模型设计文档

## 1. 任务定义

输入：Pre-contrast T1w MRI（单通道, 512×512）  
输出：Synthetic post-contrast DCE-MRI（单通道, 512×512）  
目标：合成对比增强图像，使下游 tumor 分割和分类指标尽可能接近真实 DCE-MRI。

---

## 2. Generator 架构 (v20, GlobalGenerator + Residual Mode)

```
Pre-contrast (1ch, 512×512)
         │
    ReflPad(3) + Conv7×7 → 64ch
         │
    ┌────┴──── Encoder (4× downsample) ────┐
    │  Conv3×3 stride2 + IN + ReLU          │
    │  64 → 128 → 256 → 512                │
    │  512×512 → 256 → 128 → 64 → 32×32    │
    └───────────────────────────────────────┘
         │
    ┌────┴──── Bottleneck (32×32, 512ch) ───┐
    │  ResBlock × 9                          │
    │  [ReflPad + Conv3×3 + IN + ReLU        │
    │   + ReflPad + Conv3×3 + IN] + skip     │
    └───────────────────────────────────────┘
         │
    ┌────┴──── Decoder (4× upsample) ───────┐
    │  ConvTranspose3×3 stride2 + IN + ReLU  │
    │  512 → 256 → 128 → 64                 │
    │  32×32 → 64 → 128 → 256 → 512×512     │
    └───────────────────────────────────────┘
         │
    ReflPad(3) + Conv7×7 → 1ch (delta, 无 Tanh)
         │
    output = input + delta  ← Residual skip connection
```

**参数量**: ~45M  
**感受野**: 9 ResBlocks × 3×3 conv ×2 在 32×32 feature map 上 → 覆盖整个 bottleneck  

### Residual Mode

Generator 输出的是 **delta**（增强差异），而不是完整的 post-contrast 图像。

- 优势：大部分像素 delta ≈ 0，模型只需学习"哪里增强、增强多少"
- 保证：解剖结构由 skip connection 完整保留，避免 GAN hallucination
- 无 Tanh：delta 不受 [-1,1] 限制，允许任意强度的增强

---

## 3. Discriminator (Multi-scale PatchGAN)

```
输入: concat(pre-contrast, post-contrast) → 2ch

D1 (原始分辨率 512×512):
  Conv4×4 s2 → LeakyReLU(0.2)
  Conv4×4 s2 + IN → LeakyReLU
  Conv4×4 s2 + IN → LeakyReLU
  Conv4×4 s1 + IN → LeakyReLU
  Conv4×4 s1 → 1ch patch output

D2 (下采样 256×256):
  同结构，输入先 AvgPool
```

- D1: 关注局部纹理（patch 级别真实感）
- D2: 关注全局结构（更大感受野）
- Feature matching: 从 D1/D2 的中间层提取特征做 L1 匹配

---

## 4. Loss 函数

| Loss | 权重 (λ) | 作用 |
|------|----------|------|
| LSGAN (D1+D2) | 1.0 | 对抗损失，驱动合成真实感 |
| Feature matching (D1+D2) | 10 | 匹配判别器中间特征，稳定训练 |
| VGG perceptual | 10 | 多层感知相似度 (VGG19 5层加权) |
| Tumor L1 | 10 | tumor mask 区域的像素级重建 |
| MSSC | 50 | 多尺度减影一致性 (3级 Laplacian 金字塔) |

### MSSC Loss (Multi-Scale Subtraction Consistency)

```
subtraction_fake = fake - input
subtraction_real = real - input

对两者做 3 级 Laplacian 分解:
  Level 0 (512×512): 高频细节一致
  Level 1 (256×256): 中频结构一致  
  Level 2 (128×128): 低频增强模式一致

L_MSSC = Σ L1(lap_fake[i], lap_real[i])
```

确保增强 pattern 在所有尺度上都跟真实 DCE 一致。

---

## 5. 训练辅助机制

### Breast Mask (Dataset920 2D nnUNet)

- 训练时 loss 只在乳房区域内计算
- 背景/胸壁区域不产生梯度 → 模型不学习无关区域
- 推理时**不使用** mask（GC 不提供，且模型已学会只在乳房内增强）

### Intensity Augmentation

```python
# 随机 scale + bias
scale = uniform(0.8, 1.2)
bias = uniform(-0.1, 0.1)
input_aug = input * scale + bias
```

防止过拟合特定数据源的强度分布（训练集含 DUKE/ISPY2/YUNNAN/LABREAST 4个域）。

---

## 6. 模型变体

### v23: Local Enhancer (全分辨率精修)

在 v20 Global 基础上加一层浅层网络：

```
Input 512×512 ─┬─ AvgPool → 256×256 → [Global (v20, 冻结)] → features
               │                                                 │
               └─ [Conv7×7 + Conv3×3 s2] → 256×256 features ─── ⊕
                                                                 │
                                            [ResBlock × 3] → [Upsample] → 512×512
                                                                 │
                                                    output = input + delta
```

- ngf=32 (内部 Global 为 64，兼容 v20 权重)
- 训练策略: 前 20 epoch 冻结 Global，只训 Local；之后端到端 fine-tune
- 额外参数: ~3M
- 推理额外耗时: < 0.5s

### v22: Deeper Bottleneck

- n_blocks_global: 9 → 12
- 更大感受野，可能改善全局增强一致性
- 参数增加 ~2M

---

## 7. 推理流程 (GC 提交)

```
/input/images/.../uuid.mha
       │
  读取 + z-score 归一化 (training_pre_stats.json)
       │
  Resize to 512×512 (bilinear)
       │
  Generator forward (residual mode)
       │
  Resize back to original size
       │
  保存 /output/images/.../output.mha (保留原始 spatial metadata)
```

约束: NVIDIA T4 (16GB), < 10min/case, < 10GB container.  
实际: ~0.3s/case, 容器 ~2.5GB.

---

## 8. 评估指标

| 指标 | 类型 | 评估什么 |
|------|------|---------|
| MSE ↓ | 像素级 | 整体重建精度 |
| LPIPS ↓ | 感知级 | 视觉相似度 |
| SSIM-ROI ↑ | 区域级 | tumor 区域结构保留 |
| FRD ↓ | 统计级 | radiomics 特征分布距离 |
| AUROC ↑ | 分类级 | 增强信号的诊断价值 |
| Dice ↑ | 分割级 | 合成图上 tumor 可分割性 |
| HD95 ↓ | 分割级 | 分割边界精度 |

---

## 9. 当前最佳结果 (v20, data_split/test 199 cases)

| MSE ↓ | LPIPS ↓ | Dice ↑ | HD95 ↓ |
|--------|---------|--------|--------|
| **0.191** | **0.100** | 0.718 | 67.0 |
