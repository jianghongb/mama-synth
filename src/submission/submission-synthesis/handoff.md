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
