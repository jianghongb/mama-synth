# SimulatingDCE 项目技术解析

> 论文：*"Simulating Dynamic Tumor Contrast Enhancement in Breast MRI using Conditional Generative Adversarial Networks"*
>
> 核心目标：从乳腺 MRI 的预对比（pre-contrast）图像，通过条件 GAN 生成动态增强后对比（post-contrast）DCE-MRI 图像。

---

## 一、数据处理流程

### 1.1 原始数据

**Duke 乳腺癌 MRI 数据集**（922 位患者）：
- `_0000`：Pre-contrast（注射对比剂前）
- `_0001`：Post-contrast Phase 1（早期增强）
- `_0002`：Post-contrast Phase 2（中期增强）
- `_0003`：Post-contrast Phase 3（晚期增强）
- 原始格式：DICOM
- 附带肿瘤标注框信息（`Annotation_Boxes.xlsx`）和 254 个肿瘤分割 mask（Caballo et al.）

### 1.2 DICOM → NIfTI 转换

**文件**：`synthesis/utils/convert_to_nifti_whole_dataset.py`

```python
d2n.dicom_series_to_nifti(
    original_dicom_directory,
    output_file=os.path.join(output_folder, filename),
    reorient_nifti=True
)
```

- 读取 CSV 配置文件（记录每位患者各 phase 对应的 DICOM 文件夹名）
- 使用 `dicom2nifti` 库逐 phase 转换
- 输出命名：`Breast_MRI_XXX_000{digit}.nii.gz`

### 1.3 MRI 预处理（Bias Field Correction）

**文件**：`nnUNet/custom_scripts/preprocess_breast_mri.py`

```python
# 1. 读取为 float32
image_itk = sitk.ReadImage(nifti_mri, sitk.sitkFloat32)

# 2. N4 Bias Field Correction（磁场不均匀校正）
mask_breast = sitk.OtsuThreshold(image_itk, 0, 1)
corrector = sitk.N4BiasFieldCorrectionImageFilter()
corrected_image = corrector.Execute(shrinked_image_itk, shrinked_mask_breast)
corrected_image_itk = image_itk / sitk.Exp(log_bias_field)

# 3. 强度归一化到 [0, 255]
sitk.RescaleIntensity(image_itk, 0, 255)
```

关键点：
- N4 Bias Field Correction 修正 RF 线圈造成的亮度不均
- Shrink factor=4 加速（在低分辨率上估计 bias field，再 upsample 回原始分辨率）

### 1.4 NIfTI → 2D PNG（单序列模式）

**文件**：`synthesis/utils/nifti_png_conversion.py`

处理流程：
```
3D NIfTI体积 [X, Y, Z]
    ↓ nibabel 加载
    ↓ 全局归一化: (v - min) / (max - min) * 255
    ↓ 逐轴状切片(axial)提取: img = array[:, :, i]
    ↓ 像素尺寸校正: cv2.resize(img, (new_w, new_h), INTER_AREA)
    ↓ 旋转 90° 逆时针
    ↓ 统一 resize 到 512×512 (INTER_CUBIC)
    ↓ 保存为 PNG
```

**归一化**：每个 3D volume 独立 min-max 归一化到 [0, 255]（保持切片间相对强度）

**像素尺寸校正**：
```python
pix_dim = header['pixdim'][1:4]  # 例如 [0.7, 0.7, 1.5] mm
new_dims = np.multiply(array.shape, pix_dim)
# 校正非各向同性像素间距
```

**插值选择**：
- 缩小：`INTER_AREA`（抗锯齿最佳）
- 放大到 512：`INTER_CUBIC`（与 pix2pixHD 内部一致）

**数据集划分**（硬编码列表）：
- 训练集：~670 patients
- 验证集：~200 patients
- 测试集：30 patients

**输出结构**：
```
train/train_A/  ← Breast_MRI_XXX_0000_sliceYY.png  (pre-contrast)
train/train_B/  ← Breast_MRI_XXX_0001_sliceYY.png  (post-contrast phase 1)
test/test_A/
test/test_B/
```

### 1.5 NIfTI → 2D PNG（多序列模式）

**文件**：`synthesis/utils/nifti_png_conversion_mpost.py`

与单序列版本的关键区别：
1. **提取多个 phase**：`DIGITS_TO_STORE = [0, 1, 2]`
2. **肿瘤 ROI 裁剪**：
   - 从 `Annotation_Boxes.xlsx` 读取肿瘤 bounding box
   - 只提取包含肿瘤的切片范围
   - 裁剪肿瘤空间区域
3. **切片范围计算**：
   ```python
   # Duke 切片索引与 NIfTI 数组索引方向相反
   SLIDE_MIN = total_slices - (end_slice + 1)
   SLIDE_MAX = total_slices - (start_slice + 1)
   ```

### 1.6 多时相通道拼接

**文件**：`synthesis/utils/dce_phase_to_channels_conversion.py`

**拼接**：将 3 个 post-contrast phase 的灰度图编码为 1 张 RGB 图：
```python
image_phase_1 → channel 0
image_phase_2 → channel 1
image_phase_3 → channel 2
```

**反向提取**（推理后）：
```python
concatenated_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
image_phase_1 = concatenated_image[:, :, 0]
image_phase_2 = concatenated_image[:, :, 1]
image_phase_3 = concatenated_image[:, :, 2]
```

### 1.7 PNG → NIfTI 重建

**文件**：`synthesis/utils/png_nifti_conversion.py`

将合成的 2D 切片重新组装为 3D NIfTI：
```python
# 1. 读取原始 NIfTI 的 header（spacing, origin, direction）
# 2. 将 PNG resize 回原始分辨率
img = cv2.resize(img, (int(header["dim[2]"]), int(header["dim[1]"])), INTER_AREA)
# 3. SimpleITK 组装为 3D volume
reader = sitk.ImageSeriesReader()
volume = reader.Execute()
# 4. 复制原始空间信息
volume.CopyInformation(base_nifti_image)
```

### 1.8 减影图像（Subtraction）

**文件**：`synthesis/utils/subtraction.py`

```python
subtracted_image = cv2.subtract(postcontrast_image, precontrast_image)
# 结果：只保留增强部分（肿瘤亮起来）
```

---

## 二、GAN 网络结构

### 2.1 整体架构

条件 GAN（pix2pixHD）：
```
Input (pre-contrast, 3ch)  →  Generator  →  Fake post-contrast (3ch)
                                   ↕
                         Discriminator(s)
                                   ↕
                         Real post-contrast (3ch)
```

### 2.2 生成器：GlobalGenerator

```python
# input_nc=3, output_nc=3, ngf=64, n_downsampling=4, n_blocks=9
```

完整结构：
```
ReflectionPad(3) + Conv(3→64, k=7) + InstanceNorm + ReLU     → 512×512×64

--- 4次下采样 ---
Conv(64→128, k=3, s=2) + IN + ReLU     → 256×256×128
Conv(128→256, k=3, s=2) + IN + ReLU    → 128×128×256
Conv(256→512, k=3, s=2) + IN + ReLU    → 64×64×512
Conv(512→1024, k=3, s=2) + IN + ReLU   → 32×32×1024

--- 9个 ResNet Blocks (32×32×1024) ---
每个 Block:
  ReflectionPad(1) + Conv(1024→1024, k=3) + IN + ReLU
  ReflectionPad(1) + Conv(1024→1024, k=3) + IN
  + 残差连接

--- 4次上采样 ---
ConvTranspose(1024→512, k=3, s=2) + IN + ReLU   → 64×64×512
ConvTranspose(512→256, k=3, s=2) + IN + ReLU    → 128×128×256
ConvTranspose(256→128, k=3, s=2) + IN + ReLU    → 256×256×128
ConvTranspose(128→64, k=3, s=2) + IN + ReLU     → 512×512×64

ReflectionPad(3) + Conv(64→3, k=7) + Tanh        → 512×512×3
```

设计要点：
- **InstanceNorm**：对每个样本独立归一化，适合图像翻译
- **ReflectionPadding**：减少边缘伪影
- **9个 ResNet Block**：在最低分辨率(32×32)上进行强特征转换
- **Tanh 输出**：值域 [-1, 1]，匹配输入归一化

### 2.3 ResNet Block

```python
def forward(self, x):
    out = x + self.conv_block(x)  # 残差连接
    return out

# conv_block:
# ReflectionPad(1) → Conv(dim, dim, k=3) → InstanceNorm → ReLU
# ReflectionPad(1) → Conv(dim, dim, k=3) → InstanceNorm
```

### 2.4 判别器：MultiscaleDiscriminator

**多尺度策略**（`num_D=2`）：
```
原始图像 (512×512)  → Discriminator 1 (高分辨率判别)
        ↓ AvgPool(3, s=2)
缩小图像 (256×256)  → Discriminator 2 (低分辨率判别)
```

每个 NLayerDiscriminator（PatchGAN，`n_layers=3`）：
```
输入: concat(条件图, 目标图) = 6通道

Conv(6→64, k=4, s=2) + LeakyReLU(0.2)           → 256×256×64
Conv(64→128, k=4, s=2) + IN + LReLU             → 128×128×128
Conv(128→256, k=4, s=2) + IN + LReLU            → 64×64×256
Conv(256→512, k=4, s=1) + IN + LReLU            → 64×64×512
Conv(512→1, k=4, s=1)                           → 64×64×1
```

**PatchGAN 核心**：
- 输出 64×64 概率图，每个像素对应一个局部感受野
- 鼓励生成器关注局部纹理细节

**两个判别器的互补**：
- D1（高分辨率）→ 精细纹理
- D2（低分辨率）→ 全局结构

### 2.5 VGG-19 感知网络

```python
# ImageNet 预训练，参数冻结
# 提取 5 层特征用于 perceptual loss：
# relu1_1, relu2_1, relu3_1, relu4_1, relu5_1
weights = [1/32, 1/16, 1/8, 1/4, 1.0]  # 深层权重更大
```

### 2.6 备选模型：U-Net

**文件**：`synthesis/U-Net/`

另一种合成方法（非 GAN），直接回归：
```
Input (pre-contrast, 1ch)
  ↓ DoubleConv(1→64, k=1)          512×512×64
  ↓ Down: Conv(s=2) + DoubleConv   256×256×128
  ↓ Down + Dropout(0.5)            128×128×256
  ↓ Down + Dropout(0.5)            64×64×512
  ↑ Up + skip + Dropout(0.5)       128×128×256
  ↑ Up + skip                      256×256×128
  ↑ Up + skip                      512×512×64
  ↓ Conv(64→n_time_points, k=1)
  ↓ Sigmoid
Output: 多个 post-contrast phases
```

- 损失：SSIM + MAE 组合
- 输入为 HDF5 patch 数据
- 支持减影模式（`subtraction=True`）

---

## 三、训练细节

### 3.1 数据加载

```python
# AlignedDataset
A = Image.open(A_path)                  # pre-contrast
B = Image.open(B_path).convert('RGB')   # post-contrast / CONCAT

transforms = [
    Scale([512, 512], BICUBIC),
    RandomCrop(512),
    RandomHorizontalFlip(p=0.5),
    ToTensor(),                          # [0,255] → [0,1]
    Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))  # [0,1] → [-1,1]
]
```

A 和 B 使用相同随机参数（保证配对一致性）。

### 3.2 损失函数

#### LSGAN Loss（最小二乘 GAN）

$$L_D = \frac{1}{2} E[(D(x,y) - 1)^2] + \frac{1}{2} E[D(x, G(x))^2]$$

$$L_{G,GAN} = E[(D(x, G(x)) - 1)^2]$$

使用 MSELoss 而非 BCE，梯度更稳定。

#### Feature Matching Loss

$$L_{FM} = \sum_{i=1}^{2} \sum_{j=1}^{4} \frac{4}{5} \cdot \frac{1}{2} \cdot \lambda_{feat} \cdot \| D_i^{(j)}(x, G(x)) - D_i^{(j)}(x, y) \|_1$$

让 fake 图在判别器各中间层的特征与 real 图匹配。

#### VGG Perceptual Loss

$$L_{VGG} = \lambda_{feat} \cdot \sum_{i=1}^{5} w_i \| VGG_i(G(x)) - VGG_i(y) \|_1$$

权重 $w = [1/32, 1/16, 1/8, 1/4, 1.0]$

#### 总损失

```
L_G = L_GAN + L_FM + L_VGG
L_D = 0.5 * (L_D_fake + L_D_real)
```

### 3.3 优化策略

| 参数 | 值 |
|------|-----|
| 优化器 | Adam (β1=0.5, β2=0.999) |
| 学习率 | 0.0002 |
| Batch size | 8 |
| 总 epochs | 200 (100 稳定 + 100 线性衰减至 0) |
| ngf | 64 |
| ndf | 64 |
| n_downsample | 4 |
| n_blocks | 9 |
| num_D | 2 |
| n_layers_D | 3 |
| lambda_feat | 10.0 |
| 图像尺寸 | 512×512 |
| 归一化 | Instance Norm |
| 数据增强 | 随机水平翻转 + resize_and_crop |

### 3.4 训练循环核心

```python
for epoch in range(1, 201):
    for data in dataset:
        # Forward
        fake_image = netG(input_label)
        
        # D loss
        pred_fake = netD(concat(input, fake.detach()))
        pred_real = netD(concat(input, real))
        loss_D = 0.5 * (MSE(pred_fake, 0) + MSE(pred_real, 1))
        
        # G loss
        pred_fake = netD(concat(input, fake))  # 不 detach
        loss_G = MSE(pred_fake, 1) + L_FM + L_VGG
        
        # Backward
        optimizer_D.step(); optimizer_G.step()
    
    # 学习率衰减（epoch > 100 后线性衰减）
    if epoch > 100:
        lr -= lr_initial / 100
```

### 3.5 推理

```python
# test.py: batch_size=1, 无 flip, 无 shuffle
generated = model.inference(data['label'], data['inst'], data['image'])
# 输出 [-1,1] → tensor2im → [0,255] PNG
```

---

## 四、评估方法

### 4.1 图像到图像比较指标

**文件**：`synthesis/utils/metrics.py`

```bash
python metrics.py <真实图目录> <合成图目录> --phase 0001
```

| 指标 | 库 | 方向 | 说明 |
|------|-----|------|------|
| SSIM | torchmetrics | ↑ | 结构相似性 |
| MS-SSIM | torchmetrics | ↑ | 多尺度结构相似性 |
| PSNR | torchmetrics | ↑ | 峰值信噪比 (dB) |
| LPIPS | torchmetrics (VGG) | ↓ | 学习感知距离 |
| MSE | torchmetrics | ↓ | 均方误差 |
| MAE | torchmetrics | ↓ | 平均绝对误差 |

**流程**：
1. 严格文件名配对（`check_if_files_correspond`）
2. 所有图像 resize 到 224×224
3. 逐对计算指标
4. 可选：用 segmentation mask 裁剪肿瘤 ROI 后再计算

### 4.2 分布级指标：FID

**文件**：`synthesis/utils/fid.py`

两种特征提取器：
- **ImageNet FID**：TF Hub InceptionV3，提取 pool_3 层 (2048维)
- **RadImageNet FID**：放射医学影像预训练的 InceptionV3（更适合医学图像）

$$FID = \|\mu_r - \mu_g\|^2 + Tr(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$$

**特殊功能**：
- Lower bound：真实数据 50/50 分割计算 FID 下界（衡量度量噪声）
- Per-patient split：按患者分割，避免信息泄露
- Segmentation mask ROI 裁剪后的 FID

### 4.3 对比增强动力学分析

**文件**：`synthesis/utils/contrast_enhancement_patterns.ipynb`

- 计算各 DCE-MRI phase 的平均像素强度
- 绘制时间-信号增强曲线
- 比较真实 vs. 合成数据的增强动力学模式
- 临床意义：增强曲线形态（washout/plateau/persistent）是肿瘤良恶性的重要指标

### 4.4 下游任务：nnU-Net 肿瘤分割

**文件**：`nnUNet/custom_scripts/`

验证合成数据的临床实用性：

1. **数据准备**（`convert_data_to_nnunet_204.py`）：
   - 裁剪为单乳房（基于分割 mask 判断左/右）
   - 去除多灶性病例

2. **训练**：
   ```bash
   nnUNetv2_train 208 3d_fullres {0,1,2,3,4} --npz  # 5折交叉验证
   ```

3. **推理**：
   ```bash
   nnUNetv2_predict -i <input> -o <output> -d 208 -c 3d_fullres
   ```

4. **评估**（`dice_calculation.py`）：
   ```python
   results = compute_metrics(gt_path, pred_path, labels_or_regions=[0, 1])
   dice = results['metrics'][1]['Dice']
   ```

**实验对比**：
- 仅用真实 pre+post 数据训练分割模型
- 用真实 pre+post + 合成 post 数据训练
- 比较 Dice 系数变化 → 验证合成数据是否提升分割性能

---

## 五、整体流程图

```
DICOM (Duke Dataset, 922 patients)
    ↓ dicom2nifti
NIfTI 3D Volumes
    ↓ N4 Bias Field Correction
预处理后的 NIfTI
    ↓ 切片提取 + 归一化 + resize
2D PNG (512×512, [0,255])
    ↓ [多序列: 3 phase → RGB 拼接]
    ↓ 分为 train_A(pre) / train_B(post)
    
┌─────────────────────────────────────────┐
│  pix2pixHD Training                     │
│  G: GlobalGenerator (4↓ + 9 ResBlock + 4↑) │
│  D: 2× MultiscaleDiscriminator (PatchGAN)   │
│  Loss: LSGAN + FeatureMatch + VGG      │
│  200 epochs, lr=0.0002, bs=8           │
└─────────────────────────────────────────┘
    ↓ 推理
合成 Post-contrast PNG
    ↓ [拆分 RGB → 3 个 phase]
    ↓ PNG → NIfTI 重建
    
┌─────────────────────────────────────────┐
│  评估                                    │
│  • 逐对: SSIM/MS-SSIM/PSNR/LPIPS/MSE/MAE │
│  • 分布: FID (ImageNet + RadImageNet)   │
│  • 动力学: 增强曲线对比                  │
│  • 下游: nnU-Net 分割 Dice              │
└─────────────────────────────────────────┘
```

---

## 六、明天早上学习计划（3小时）

### 第一小时：数据流 + 项目骨架（09:00 - 10:00）

**目标**：理解数据从原始 DICOM 到模型输入的完整链路

| 时间 | 内容 | 对应文件 |
|------|------|----------|
| 09:00-09:15 | 阅读本文档第一章 + README | `README.md` |
| 09:15-09:30 | 通读 `convert_to_nifti_whole_dataset.py`，理解 DICOM→NIfTI | `synthesis/utils/` |
| 09:30-09:45 | 通读 `nifti_png_conversion.py`，重点理解归一化、像素校正、数据划分 | `synthesis/utils/` |
| 09:45-10:00 | 通读 `dce_phase_to_channels_conversion.py` + `aligned_dataset.py`，理解多通道拼接和数据加载 | `synthesis/pix2pixHD/data/` |

**检验标准**：能画出 "DICOM → NIfTI → PNG → Tensor[-1,1]" 的完整数据流图，说清楚每步的输入输出尺寸和值域。

### 第二小时：GAN 网络 + 训练（10:00 - 11:00）

**目标**：理解模型结构和训练过程

| 时间 | 内容 | 对应文件 |
|------|------|----------|
| 10:00-10:20 | 精读 `networks.py` 中的 `GlobalGenerator` 和 `ResnetBlock`，画出网络图 | `synthesis/pix2pixHD/models/networks.py` |
| 10:20-10:35 | 精读 `MultiscaleDiscriminator` + `NLayerDiscriminator`，理解 PatchGAN 输出 | 同上 |
| 10:35-10:50 | 精读 `pix2pixHD_model.py` 的 `forward()` 方法，理解 3 个 loss 的计算 | `synthesis/pix2pixHD/models/pix2pixHD_model.py` |
| 10:50-11:00 | 通读 `train.py`，理解训练循环 + 学习率衰减策略 | `synthesis/pix2pixHD/train.py` |

**检验标准**：能手写 Generator/Discriminator 的层级结构（通道数、尺寸变化），能解释为什么用 LSGAN + Feature Matching + VGG 三个 loss。

### 第三小时：评估 + 下游任务（11:00 - 12:00）

**目标**：理解如何衡量合成质量和验证临床价值

| 时间 | 内容 | 对应文件 |
|------|------|----------|
| 11:00-11:15 | 通读 `metrics.py`，理解 6 个指标的计算方式和适用场景 | `synthesis/utils/metrics.py` |
| 11:15-11:30 | 通读 `fid.py`，理解 ImageNet FID vs RadImageNet FID 的区别 | `synthesis/utils/fid.py` |
| 11:30-11:45 | 通读 `full_pipeline.sh` + `convert_data_to_nnunet_204.py` + `dice_calculation.py`，理解分割实验设计 | `nnUNet/custom_scripts/` |
| 11:45-12:00 | 回顾整体流程图，串联所有模块，记录疑问点 | 本文档第五章 |

**检验标准**：能解释 FID 的数学含义和为什么 RadImageNet 更适合医学图像；能说清楚 "合成数据提升分割性能" 的实验设计逻辑。

---

### 学习建议

1. **带着问题读代码**：每个文件先看 main/入口，再看核心函数，跳过 utils
2. **画图辅助**：准备一张纸，画出数据流和网络结构
3. **对照本文档**：遇到不明白的地方回来查对应章节
4. **不要纠结细节**：第一遍重点是理解 "为什么这么设计"，而非每行代码
