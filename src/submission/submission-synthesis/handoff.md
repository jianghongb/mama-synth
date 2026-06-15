# Pix2PixHD for MAMA-SYNTH — 模型学习笔记

## 1. GlobalGenerator 结构 (models/networks.py)

### Encoder-Bottleneck-Decoder 架构

```
输入: [B, 1, 512, 512]  (pre-contrast MRI, z-score normalized)

── ENCODER (逐步降分辨率，升通道数) ──
  ReflectionPad2d(3) + Conv2d(1→64, k=7)     → [B, 64, 512, 512]
  Conv2d(64→128, k=3, s=2)                   → [B, 128, 256, 256]   ↓2
  Conv2d(128→256, k=3, s=2)                  → [B, 256, 128, 128]   ↓4
  Conv2d(256→512, k=3, s=2)                  → [B, 512, 64, 64]     ↓8
  Conv2d(512→1024, k=3, s=2)                 → [B, 1024, 32, 32]    ↓16

── BOTTLENECK (在最低分辨率做变换) ──
  9× ResnetBlock(1024)                        → [B, 1024, 32, 32]
  每个block: x + Conv-IN-ReLU-Conv-IN(x)     (残差连接，梯度直通)

── DECODER (逐步升分辨率，降通道数) ──
  ConvTranspose2d(1024→512, s=2)              → [B, 512, 64, 64]    ↑2
  ConvTranspose2d(512→256, s=2)               → [B, 256, 128, 128]  ↑4
  ConvTranspose2d(256→128, s=2)               → [B, 128, 256, 256]  ↑8
  ConvTranspose2d(128→64, s=2)                → [B, 64, 512, 512]   ↑16
  ReflectionPad2d(3) + Conv2d(64→1, k=7)      → [B, 1, 512, 512]

── OUTPUT ──
  残差模式: output = input + model(input)   (无Tanh，允许任意范围)
```

### 关键参数

- `input_nc=1`: 单通道灰度输入（pre-contrast MRI）
- `ngf=64`: 第一层 64 通道，逐层 ×2 到 bottleneck 1024 通道，控制网络容量
- `n_downsampling=4`: 4 次下采样，输入必须被 16 整除，bottleneck=32×32
- `n_blocks=9`: 9 个 ResNet block，在低分辨率做核心变换，感受野覆盖整图

### 设计理念

- ReflectionPad 而非 ZeroPad：避免边界黑色伪影
- InstanceNorm：对每张图独立归一化，适合医学图像对比度差异大的场景
- 残差模式：网络只学 Δ（造影增强差异），背景自动不变，收敛更快

---

## 2. forward() 完整训练步骤 (models/pix2pixHD_model.py)

```
STEP 1: 生成假图
  fake = Generator(pre)  = pre + Δ

STEP 2: D 判断假图 → loss_D_fake
  D(pre, fake) → 应判为"假"
  loss_D_fake = MSE(D(pre,fake), 0)

STEP 3: D 判断真图 → loss_D_real
  D(pre, real) → 应判为"真"
  loss_D_real = MSE(D(pre,real), 1)

STEP 4: G 的 GAN loss → loss_G_GAN
  D(pre, fake) → G 希望 D 判为"真"
  loss_G_GAN = MSE(D(pre,fake), 1)

STEP 5: Feature Matching → loss_G_GAN_Feat
  L1(D中间层特征(fake), D中间层特征(real))
  让 D 在每个观察步骤的感受都跟看真品一样

STEP 6: VGG Perceptual → loss_G_VGG
  L1(VGG特征(fake), VGG特征(real))
  在感知特征空间比较，容忍小位移，保证语义结构正确

STEP 7: Tumor L1 → loss_G_Tumor [新加的]
  L1(fake×mask, real×mask) × tumor_weight
  强制肿瘤区域精确重建

最终:
  loss_G = G_GAN + 10×G_GAN_Feat + 10×G_VGG + 10×G_Tumor
  loss_D = 0.5 × (D_fake + D_real)
```

### 对抗博弈

- G 学"如何骗过 D"，D 学"如何区分真假"
- 交替更新防止同时更新导致的振荡
- 理想平衡：D 判断正确率 ≈ 50%

---

## 3. VGGLoss + GANLoss (models/networks.py)

### GANLoss (LSGAN)

```python
loss = MSELoss(D的输出, target)
# target = 1.0 (真) 或 0.0 (假)
```

- 使用 MSE 而非 BCE：D 离目标越远梯度越大，训练更稳定
- MultiscaleDiscriminator：两个尺度的 D 都要判断对/被骗过

### VGGLoss

```python
weights = [1/32, 1/16, 1/8, 1/4, 1.0]  # 深层权重大

loss = Σ weights[i] × L1(VGG_layer_i(fake), VGG_layer_i(real))
```

- VGG19 (ImageNet 预训练，冻结) 提取 5 层特征
- 浅层=边缘/纹理，深层=语义/全局结构
- 比 pixel L1 好：容忍小位移，鼓励锐利边缘
- 单通道适配：`x.repeat(1,3,1,1)` 复制到 3 通道给 VGG

### 三个 Loss 协作

```
G_GAN:       让生成图统计分布像真图
G_GAN_Feat:  让纹理细节真实（多层次匹配）
G_VGG:       保证内容结构正确（语义一致）
G_Tumor:     最重要区域精确（任务导向）
```

---

## 4. train.py 训练循环

### 每个 iteration

```
1. 取 batch: pre[B,1,512,512], gt[B,1,512,512], mask[B,1,512,512]
2. Forward: 一次性算出所有 loss + 生成的 fake image
3. 组合 loss:
   loss_G = G_GAN + G_GAN_Feat + G_VGG + G_Tumor
   loss_D = 0.5 × (D_fake + D_real)
4. 更新 G:
   optimizer_G.zero_grad()
   loss_G.backward()
   optimizer_G.step()
5. 更新 D:
   optimizer_D.zero_grad()
   loss_D.backward()
   optimizer_D.step()
6. 日志/保存 checkpoint
```

### G/D 交替的实现

- 更新 G 时：D 参数通过 `.detach()` 切断梯度，不受影响
- 更新 D 时：fake_image 用 `.detach()` 切断到 G 的梯度
- D loss × 0.5：防止 D 学太快压制 G

### 学习率调度

```
epoch 1-100:   lr = 0.0002 (稳定学习)
epoch 101-200: lr 从 0.0002 线性衰减到 0 (精细化)
```

---

## 5. data/mha_dataset.py 数据加载

### 数据格式

```
dataroot/mha/
  input/         *.mha  pre-contrast, float32, z-score normalized
  ground_truth/  *.mha  post-contrast peak, float32, z-score normalized
  mask/          *.mha  binary tumor mask, int16
```

### 两种模式

- **训练模式** (`resize_or_crop='resize'`): 所有图 resize 到 512×512，支持 batch>1
- **推理模式** (`resize_or_crop='none'`): pad 到 16 倍数，保持原始分辨率

### 数据特征

- z-score 归一化: `(pixel - 104.86) / 215.86`
- 背景(空气) = -0.4858 (固定常数)
- 尺寸可变: 128×128 ~ 512×512，含非方形
- 1077 训练 / 274 测试样本
- 训练时随机水平翻转增强

### 返回字典

```python
{
    'label': input_tensor,    # pre-contrast [1,H,W]
    'image': gt_tensor,       # post-contrast [1,H,W]
    'mask': mask_tensor,      # tumor mask [1,H,W]
    'inst': placeholder,      # 兼容性占位
    'feat': placeholder,
    'path': filepath,
    'orig_size': (H, W),      # 推理时用于 crop 回原始尺寸
}
```

---

## 6. 当前评估结果与提升方案

### 当前模型表现 (eval_results_512, 272 test cases)

| Metric | 你的 512 | Leaderboard #1 | Gap |
|--------|----------|----------------|-----|
| MSE ↓ | 1.55 | 0.75 | 2× |
| LPIPS ↓ | 0.25 | 0.08-0.10 | 2.5× |
| SSIM_tumor ↑ | 0.27 | 0.45-0.47 | -0.20 |
| FRD ↓ | 11.52 | 21-23 | ✅ 你好 |
| AUROC contrast | 0.86 | 0.80 | ✅ 你好 |
| Dice ↑ | 0.25 | 0.48 | -0.23 |
| HD95 ↓ | 173 | 121 | +52px |

**特点：** 全局分布好（FRD、AUROC），局部细节差（LPIPS、tumor metrics）

### 提升方案优先级

#### P0: 推理时 Resize（零训练成本，立即提升）

当前问题：64% 训练数据是 256×256，但模型只在 512 上训练。推理时非 512 case 直接 pad，纹理完全不对。

修改 `process.py`：所有 case 先 resize 到 512×512 推理，再 resize 回原分辨率。

预期：LPIPS 0.25 → ~0.18

#### P1: 全数据 Resize 512 重训练（最大收益）

当前只用了 ~5% 数据（约 50 个 512 case）。把全部 1074 个 train case resize 到 512×512 重训练。

```bash
# data/mha_dataset.py 已支持 resize_or_crop='resize' 模式
# 确认训练命令里 --resize_or_crop resize --loadSize 512
```

预期：LPIPS 0.25 → ~0.15, SSIM_tumor 0.27 → ~0.35

#### P2: 加强 Perceptual Loss

提升 VGG perceptual loss 权重 + 加 SSIM loss：

```python
# pix2pixHD_model.py
loss_G_VGG = self.criterionVGG(fake_image, real_image) * 20  # 从10提到20

# 加 SSIM loss
from torchmetrics.functional import structural_similarity_index_measure as ssim
loss_ssim = (1 - ssim(fake, real, data_range=2.0)) * 10
```

预期：LPIPS ~0.15 → ~0.12

#### P3: 降低 GAN Loss 权重

GAN 让分布好但细节糊。降低对抗 loss：

```python
loss_G = loss_G_GAN * 0.1 + loss_G_feat * 10 + loss_G_VGG * 20 + loss_G_tumor * 10
```

预期：LPIPS → ~0.10，但 FRD 可能变差

#### P4: Tumor-Weighted Loss 加强

当前 tumor_weight=10，可以进一步提高到 20-50，尤其配合 SSIM loss 在 mask 区域：

```python
# mask 区域单独算 SSIM
tumor_ssim = 1 - ssim(fake * mask, real * mask, data_range=2.0)
loss_G_tumor = L1_tumor * 20 + tumor_ssim * 10
```

预期：SSIM_tumor 0.27 → 0.35+, Dice 0.25 → 0.35+

#### P5: 加回 Motion Cases（谨慎）

176 个 motion case 被排除。如果 motion ≤ 3px 的 case 加回，+16% 数据量。
需要验证哪些是轻度 motion，逐步加入。

### Docker 提交状态

已准备好：`synthesis/pix2pixHD/docker_submission/`

```
docker_submission/
├── Dockerfile          # pytorch:2.0.1-cuda11.7, non-root user
├── process.py          # GC I/O: /input/.../pre-contrast-dce-mri-slice-breast/<uuid>.mha
│                       #        → /output/.../synthetic-contrast-dce-mri-slice-breast/output.mha
├── do_build.sh
├── do_test_run.sh
├── do_save.sh
├── models/networks.py
├── weights/latest_net_G.pth (696MB)
└── test/input/images/pre-contrast-dce-mri-slice-breast/
```

GC Interface slugs:
- Input: `pre-contrast-dce-mri-slice-breast`
- Output: `synthetic-contrast-dce-mri-slice-breast`
- Output 文件名固定为 `output.mha`（每 job 一个 case）

---

## 7. Validation Phase 对比 (2026-06-12)

| Metric | 你的 v3 (best) | #1 s-vsp00 v2.1 (6/11) | #2 s-vsp00 v2 (6/8) | #3 s-vsp00 v1 (6/6) |
|--------|:-:|:-:|:-:|:-:|
| MSE ↓ | 1.32 | **0.81** | 0.82 | 0.75 |
| LPIPS ↓ | 0.242 | **0.10** | 0.10 | 0.10 |
| SSIM_tumor ↑ | 0.277 | **0.50** | 0.45 | 0.47 |
| FRD ↓ | **11.88** | 23.10 | 23.10 | 23.10 |
| AUROC contrast | **0.854** | 0.81 | 0.80 | 0.82 |
| Dice ↑ | 0.271 | **0.49** | 0.48 | 0.43 |
| HD95 ↓ | 186.6 | **107.4** | 120.8 | 133.5 |

**结论：** GAN 的 FRD/AUROC 领先，但 LPIPS/SSIM_tumor/Dice 差距 2-3×，Pix2PixHD 架构天花板明显。#1 使用 img2img (可能是 diffusion-based)。

### Mean Position 预估 (v3 模型)

Mean Position ≈ **16.1**（预计排名 10-15 名）

| Metric | 值 | 预估排名 | 评价 |
|--------|------|:-:|:-:|
| MSE | 1.32 | 36 | 🔴 最拖分 |
| LPIPS | 0.242 | 27 | 🔴 |
| SSIM_tumor | 0.277 | 20 | 🟡 |
| AUROC tumor | — | ~20 | 🟡 |
| FRD | 11.88 | **8** | ✅ |
| AUROC contrast | 0.854 | **2** | ✅ 最强 |
| Dice | 0.271 | 11 | 🟡 |
| HD95 | 186.6 | **5** | ✅ |

计算方式：Mean Position = 8 个指标各自排名的算术平均。

---

## Lesion-Focused Enhancement（Supervisor 建议，待实现）

**核心问题**：推理时只有 pre-contrast T1w 输入，没有 lesion mask。模型如何学会在 lesion 区域生成正确的增强信号？

### 方案 1：Attention Branch（推荐实现）

在 Generator 中加 attention branch，训练时用 mask 做辅助监督，推理时自动预测 lesion 区域：

```
Pre-contrast → Encoder → 特征图
                            ↓
                    Attention Branch → 预测 lesion 概率图（辅助 loss: BCE vs real mask）
                            ↓
               主分支 × attention weight → Decoder → Post-contrast
```

- 训练时：`loss = loss_recon + 0.1 * BCE(attention_map, real_mask)`
- 推理时：attention branch 自动聚焦 lesion 区域，不需要 mask 输入
- 优势：让模型显式学会"哪里需要增强"

### 方案 2：Mask 作为条件输入

- 训练时：`input = [pre-contrast, real_mask]`（2通道）
- 推理时：先用 nnU-Net 预测 mask，再作为条件输入
- 缺点：依赖分割模型质量，增加推理复杂度

### 方案 3：加大 tumor_weight + MSSC（当前最快）

- `--tumor_weight 50 --lambda_mssc 50`
- MSSC 在 subtraction map 上约束，间接强化 lesion 增强信号
- 不需要改架构

### 实现优先级

1. 方案 3 先跑一版对比（无需改代码）
2. 方案 1 实现 attention branch（改 GlobalGenerator，加一个 side branch）
3. 方案 2 作为 ablation 实验或 future work

### 当前训练版本状态

| 版本 | 配置 | 状态 | 结果 |
|------|------|------|------|
| v3 (mamasynth_residual) | residual + tumor_weight=10 | ✅ 完成 | MSE=1.32, Dice=0.27 |
| mamasynth_standard | 无 residual, 无 tumor weight | ⏳ 未开始/进行中 |  |
| v5 (baseline_norm) | per-image min-max 归一化 | ⏳ 待跑 |  |
| v6 (tumor+MSSC 加强) | tumor_weight=50 + lambda_mssc=50 | ⏳ 待跑 |  |
| v7 (attention branch) | 方案1，待实现 | 📝 设计中 |  |

### 方案 4：胸壁 Masking（训练/推理时统一处理）

**思路**：将胸壁区域替换为黑色（背景），让模型只学乳腺区域。推理时胸壁直接用 pre-contrast 原值填回。

**实现**：
```
训练时：
  input  = pre_contrast * breast_mask      （胸壁变黑）
  target = post_contrast * breast_mask     （胸壁也变黑）

推理时：
  input  = pre_contrast * breast_mask
  output = model(input)
  final  = output * breast_mask + pre_contrast * (1 - breast_mask)
           ↑ 乳腺区域用模型输出    ↑ 胸壁区域直接复制 pre（不增强，假设合理）
```

**breast_mask 生成**：简单阈值（z-score > -0.3 为前景），或更精确的形态学方法。

**优点**：
- 模型 100% 容量给乳腺，不浪费在胸壁
- 推理时胸壁区域无 artifact（直接用原图）
- 假设合理：胸壁不吸收造影剂，pre ≈ post

**缺点/风险**：
- 乳腺-胸壁交界处可能有边界 artifact
- 需要 breast_mask 质量好，否则误切乳腺组织
- 需要训练和推理完全一致

**和 residual mode 的关系**：本质类似——胸壁的 delta≈0，只是这里用 mask 显式排除而非让模型隐式学出。

**优先级**：可作为 v8 实验，在 attention branch (v7) 之后尝试。

---

## 8. Breast Masking Pipeline (v7)

### 概述

在推理时，用 nnUNet 预训练的 breast segmentation 模型将图像分为"乳房区域"和"胸壁区域"：
- **乳房区域**: 使用 Pix2PixHD 合成的 post-contrast 结果
- **胸壁区域**: 直接保留 pre-contrast 原值（胸壁不吸收造影剂，pre ≈ post）

### 模型来源

- **权重**: `nnUNet_pretrained_weights/Dataset910_BreastSegNet/`
- **来源**: MAIA server `~/renda/weights/Dataset910_BreastSegNet`
- **两个变体**:
  - `nnUNetTrainer__nnUNetPlans__2d` — PlainConvUNet (~268MB)
  - `nnUNetTrainer__nnUNetResEncUNetLPlans__2d` — ResEncUNetL (~848MB, 效果更好)
- **训练数据**: 973 个 2D slices ("QihangBreast" dataset)
- **配置**: 2D, fold_0, ZScoreNormalization

### 分割标签

| Label | 类别 | 归属 |
|-------|------|------|
| 0 | background | 排除 |
| 1 | tissue | **乳房** ✓ |
| 2 | vessel | **乳房** ✓ |
| 3 | muscle | 排除（胸壁） |
| 4 | bone | 排除 |
| 5 | lesion | **乳房** ✓ |
| 6 | lymphnode | **乳房** ✓ |
| 7 | heart | 排除 |
| 8 | liver | 排除 |
| 9 | implant | **乳房** ✓ |

Breast mask = label ∈ {1, 2, 5, 6, 9}

### 推理 Pipeline

```
pre-contrast input
  │
  ├─→ nnUNet BreastSeg → breast_mask (binary)
  │
  ├─→ Resize 512 → Pix2PixHD → Resize back → synthetic_post
  │
  └─→ output = where(breast_mask, synthetic_post, pre_contrast)
```

### 推理代码

`inference.py` 中使用 `predict_single_npy_array` 避免 multiprocessing 问题：

```python
predictor.predict_single_npy_array(
    arr,       # shape (1, 1, H, W) for 2D
    props,     # spacing/origin metadata
    None, None, False
)
```

### Docker 容器大小

| 组件 | 大小 |
|------|------|
| PyTorch base | ~4.5 GB |
| nnunetv2 + deps | ~200 MB |
| Pix2PixHD weights | 730 MB |
| BreastSeg weights (ResEncUNetL) | 848 MB |
| **总计** | **~6.3 GB** ✅ (< 10GB) |

### 预期收益

- MSE ↓: 胸壁不再产生错误增强
- LPIPS ↓: 减少胸壁伪影
- Dice ↑: 减少假阳性增强区域

### 文件变更

- `inference.py` — 集成 breast seg + synthesis 合并逻辑
- `Dockerfile` — 添加 nnunetv2 依赖 + breast_seg 权重
- `weights/breast_seg/` — BreastSegNet 模型文件

---

## 9. 训练版本 v7

### v7 vs v6 区别

v7 在**训练时**也使用 breast mask，让模型只学习乳房区域的增强，不浪费容量在胸壁上。

| 对比 | v6 | v7 |
|------|----|----|
| 数据 | data_split_v2 (1356) | data_split_v2 (1356) |
| 训练时 masking | 无 | breast_mask × loss |
| 推理时 masking | 无 | nnUNet breast mask |
| Loss 改动 | 无 | 所有 loss 只在 breast 区域计算 |

### 训练时 breast mask 的用法

```python
# 在训练 forward 中:
fake = Generator(pre_masked)  # 输入: pre × breast_mask
loss_G = loss_fn(fake × breast_mask, real × breast_mask)

# 或者更简单: 只改 loss, 不改输入
fake = Generator(pre)
loss_G = loss_fn(fake × breast_mask, real × breast_mask)  # loss 只看乳房区域
```

### 训练脚本

`berzelius_train_v7.sh` — 基于 v6，添加 `--breast_mask_dir` 参数。

### v7 评估结果 (2026-06-15)

#### Test Set (data_split_v2/test, 150 cases)

| Metric | v5 | v6 | v7 |
|--------|-----|-----|-----|
| MSE ↓ | 0.25 | 0.87 | 0.89 |
| LPIPS ↓ | 0.079 | 0.114 | 0.138 |
| SSIM_tumor ↑ | 0.722 | 0.423 | 0.407 |
| FRD ↓ | 11.19 | 12.78 | **9.85** |
| AUROC ↑ | 0.906 | 0.926 | 0.926 |
| Dice ↑ | 0.646 | 0.492 | 0.437 |
| HD95 ↓ | 103.8 | 105.8 | 115.8 |

#### Yunnan 外部验证 (100 cases, 独立数据，不在训练集中)

| Metric | v5 | v6 | v7 |
|--------|-----|-----|-----|
| MSE ↓ | 0.393 | 0.399 | **0.125** |
| LPIPS ↓ | 0.235 | 0.290 | **0.120** |
| SSIM_tumor ↑ | 0.405 | 0.324 | **0.482** |
| FRD ↓ | 29.55 | 30.45 | **28.56** |
| AUROC ↑ | 0.879 | 0.934 | 0.844 |
| Dice ↑ | 0.095 | 0.076 | 0.090 |
| HD95 ↓ | 574.7 | 584.6 | 651.2 |

#### 分析

- **外部泛化能力大幅提升**：v7 在 Yunnan 上 MSE 降 3 倍，LPIPS 降一半，SSIM_tumor +19%
- **FRD 9.85 是 test set 全版本最佳**：整体图像分布更真实
- **Test set 上 pixel metrics 不如 v5**：可能因为 v5 的 test set metrics 来自旧 split (144 cases)，不公平对比
- **Breast mask 策略有效**：胸壁不再产生错误增强，外部数据表现显著改善
- **AUROC 略降**：胸壁保持 pre-contrast 值导致分类器"看到"更少的造影信号

#### 结论

v7 是**外部泛化能力最强**的版本。推荐提交 GC validation（GC test set 含未知域数据 RV_07，v7 的泛化优势应能体现）。

### 可能的改进方案 (v8+)

1. **v7 + 排除 motion 数据 (v7v2)**：1207 cases + breast mask，可能进一步提升
2. **v5v2 (ablation)**：v5 超参 + data_split_v2 排除 motion (1207 cases)，验证 motion 是否为退化原因
3. **推理时加 breast mask**：v5 推理时也加上 breast mask 后处理，看是否能获得 v7 类似的泛化提升（零成本实验）
4. **更强的 breast seg 模型**：当前一些 case 返回空 mask，影响评估准确性

---

## 10. 训练版本 v9

### v9 = v5 + breast mask (最佳数据 + 最佳训练策略)

| 对比 | v5 | v7 | v9 |
|------|----|----|-----|
| 数据 | data_split (1074, 无 motion) | data_split_v2 (1356, 含 motion) | data_split (1074, 无 motion) |
| Breast mask | ❌ | ✅ | ✅ |
| 其余超参 | MSEC=50 | MSEC=50 | MSEC=50 |

**假设**：
- v5 数据干净 → 图像质量好（test set 上 LPIPS/SSIM 最优）
- v7 breast mask → 泛化能力强（Yunnan 外部验证最优）
- v9 结合两者优势，预期在两个维度都表现好

**训练脚本**: `models/scripts/berzelius_train_v9.sh`
- 先生成 `data_split/train/mha/breast_mask/`（nnUNet, 一次性）
- 再训练 200 epochs with `--breast_mask_dir`

**状态**: ⏳ 训练中

---

## 10. 训练版本 v8 (Intensity Augmentation)

### 目标
提高模型在 GC 未知数据 (RV_07) 上的泛化能力。

### 问题分析
- v5 本地全面超越 #1，但 GC 上排 27
- RV_07 cases: LPIPS 和 #1 一样 (0.14)，但 Dice=0、MSE 高
- 说明图像结构对，但强度/对比度 scale 不匹配
- 根因：模型只见过训练集的特定强度分布

### 方案：训练时随机强度增强
```python
# 每个 sample 随机：
scale = uniform(0.7, 1.3)   # 模拟不同扫描仪信号强度
bias = uniform(-0.2, 0.2)   # 模拟不同归一化基线
input = input * scale + bias
gt = gt * scale + bias       # 同步变换，保持 enhancement pattern
```

### v8 配置
- 基于 v5 (MSEC=50, 1074 cases, resize 512)
- 唯一新增: `--intensity_aug`
- 训练脚本: `berzelius_train_v8.sh`
- 推理不变（纯 resize，无后处理）

### 为什么不用自适应推理缩放
- 已测试 adaptive delta scaling → 本地变差 (MSE 0.28→0.59)
- 原因：in-distribution 数据不需要缩放，强制缩放反而破坏
- 正确做法是从训练端解决，让模型本身对强度鲁棒

---

## 11. AMBL 数据处理 (Advanced-MRI-Breast-Lesions)

### 数据来源
- TCIA: https://www.cancerimagingarchive.net/collection/advanced-mri-breast-lesions/
- 632 patients total, 99 with segmentation (ROI)
- 以色列单中心, 1.5T GE, 2018-2021
- License: CC BY 4.0

### 数据结构
每个患者有：
- `AX Sen Vibrant MASK` — pre-contrast (116 slices, 512×512)
- `AX Sen Vibrant MultiPhase` — 5 post-contrast phases (580=116×5 slices)
- `ROI` — 单 slice 肿瘤标注
- `+C` — 单个 post-contrast phase

### 处理流程
```
DICOM (964 folders, 99 patients)
  → convert_ambl_dicom.py (multi-phase split)
    → NIfTI (485 cases = 92 patients × ~5 phases)
      → preprocess.py (z-score normalization, slice extraction)
        → MHA (input/ground_truth/mask)
```

### 关键脚本
- `models/convert_ambl_dicom.py` — DICOM→NIfTI，按 phase 拆分为独立 case

### 多 Phase 拆分的好处
1. 数据量 5× 放大 (92→485 cases)
2. 模型学会不同增强阶段（弱→强→衰减），提升泛化能力
3. 对未知数据（RV_07）不同增强幅度更鲁棒
4. 用真实物理变化代替人工 intensity augmentation

### 处理结果
- 成功转换: 485 cases (images + segmentations)
- 路径: `/Users/ehogjig/git/kth/ambl_nifti/`
- 待 preprocess.py 处理后可合并到训练集
