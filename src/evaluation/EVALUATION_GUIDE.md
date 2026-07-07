# MAMA-SYNTH Evaluation Code Guide

## 目录结构

```
src/evaluation/
├── evaluate.py                 ← 入口：加载数据、调用 evaluators、输出 metrics.json
├── evaluators/
│   ├── base.py                 ← Case 数据类 + BaseEvaluator ABC
│   ├── image_metrics.py        ← MSE + LPIPS（全图）
│   ├── roi_metrics.py          ← SSIM_tumor + FRD（tumor ROI）
│   ├── classification.py       ← AUROC_contrast + AUROC_tumor_ROI
│   ├── segmentation.py         ← Dice + HD95（nnUNet 分割）
│   └── mirror_utils.py         ← Midline detection + 对侧镜像
├── models/
│   ├── classification/         ← pre-trained radiomics classifiers (.pkl)
│   └── segmentation/           ← nnUNet 权重 (fold_0)
└── ground_truth/               ← GT data for Docker test runs
```

---

## 执行流程 (evaluate.py)

```
1. 读取环境变量 → 确定数据路径
2. 发现 cases: 匹配 prediction/*.mha 与 GT/*.mha 和 mask/*.mha
3. 构建 Case 对象列表 (prediction, ground_truth, mask, precontrast)
4. 依次调用 4 个 Evaluator:
   ├── ImageMetricsEvaluator   → MSE, LPIPS
   ├── ROIMetricsEvaluator     → SSIM_tumor, FRD
   ├── ClassificationEvaluator → AUROC_contrast, AUROC_tumor_ROI
   └── SegmentationEvaluator   → Dice, HD95
5. 合并结果 → 写入 metrics.json
```

---

## Case 数据结构 (base.py)

```python
@dataclass
class Case:
    case_id: str                    # 文件名（不含 .mha）
    prediction: np.ndarray          # 模型输出（z-score float32）
    ground_truth: np.ndarray        # GT post-contrast（z-score float32）
    mask: Optional[np.ndarray]      # tumor binary mask (0/1)
    precontrast: Optional[np.ndarray]  # pre-contrast input（用于 AUROC）
    # + 文件路径用于 FRD
```

---

## 指标详解

### 1. MSE (image_metrics.py)

```python
def evaluate(case):
    mse = np.mean((prediction - ground_truth) ** 2)
```

- **输入**：全图，z-score normalized
- **无 mask**：全部像素参与
- **敏感性**：对极端 pixel 值非常敏感（少数 outlier 可以拉高均值）

### 2. LPIPS (image_metrics.py)

```python
def _compute_lpips(pred, gt):
    # Step 1: clip 到 ±5σ
    pred_clipped = np.clip(pred, -5.0, 5.0)
    gt_clipped = np.clip(gt, -5.0, 5.0)
    
    # Step 2: 线性映射到 [-1, 1]
    pred_norm = pred_clipped / 5.0
    gt_norm = gt_clipped / 5.0
    
    # Step 3: expand 到 3 通道 (LPIPS 要求 RGB)
    # (1, 1, H, W) → (1, 3, H, W) 重复 3 次
    
    # Step 4: torchmetrics AlexNet 计算感知距离
    lpips = AlexNet_perceptual_distance(pred_norm, gt_norm)
```

- **±5σ clip**：因为 post-contrast intensity 可能很高（增强后 z-score 远超 pre-contrast 分布）
- **AlexNet 特征**：提取 conv1-conv5 的 feature maps，计算 L2 距离加权和
- **不用 mask**：全图参与

### 3. SSIM Tumor (roi_metrics.py)

```python
def evaluate(case):
    # Step 1: 计算全图 SSIM map (local window, data_range=10.0)
    _, ssim_map = structural_similarity(
        prediction, ground_truth,
        data_range=10.0,  # 对应 ±5σ 范围
        full=True         # 返回逐像素 SSIM map
    )
    
    # Step 2: 只取 tumor mask 内的平均值
    ssim_tumor = np.mean(ssim_map[tumor_mask > 0])
```

- **data_range=10.0**：z-score 空间中 ±5σ = 范围 10
- **只在 tumor ROI 内评估**：衡量 tumor 区域的结构保真度
- **如果 mask 为空**：跳过该 case

### 4. FRD - Fréchet Radiomics Distance (roi_metrics.py)

```python
def _compute_frd(cases):
    # Step 1: 对每个 case，从 prediction 和 GT 的 tumor ROI 提取 radiomics features
    #         使用 pyradiomics（~464 features: firstorder, GLCM, GLRLM, GLDM, GLSZM, NGTDM）
    #         + LoG 和 Wavelet filter banks
    
    # Step 2: 使用 frd-score 库计算 Fréchet Distance
    #         类似 FID，但用 radiomics 替代 Inception features
    #
    # FRD = ||μ_pred - μ_gt||² + Tr(Σ_pred + Σ_gt - 2√(Σ_pred · Σ_gt))
    
    frd = compute_frd(pred_paths, gt_paths, mask_paths)
```

- **Aggregate-only**：一个数值代表整个 test set（不是 per-case）
- **需要 tumor mask**：features 从 mask 区域提取
- **衡量分布差异**：不是比较单张图，而是比较"合成图像群体"vs"真实图像群体"的 radiomics 分布

### 5. AUROC Contrast (classification.py)

```python
def _compute_auroc_contrast(cases):
    features = []
    labels = []
    
    for case in cases:
        # 从 synthetic post-contrast 的 tumor ROI 提取 radiomics
        syn_features = extract_radiomic_features(case.prediction, case.mask)
        features.append(syn_features)
        labels.append(1)  # label 1 = post-contrast
        
        # 从 real pre-contrast 的 tumor ROI 提取 radiomics
        pre_features = extract_radiomic_features(case.precontrast, case.mask)
        features.append(pre_features)
        labels.append(0)  # label 0 = pre-contrast
    
    # 用 pre-trained classifier 预测
    probs = classifier.predict_proba(features)[:, 1]  # P(post-contrast)
    
    auroc = roc_auc_score(labels, probs)
```

- **Pre-trained classifier**：XGBoost/RF，训练在真实 pre vs post-contrast radiomics 上
- **逻辑**：如果合成图像增强够强 → classifier 轻松判为 post → AUROC 高
- **AUROC ≈ 0.5**：合成图像和 pre-contrast 无区别 = 没增强 = 失败
- **需要 pre-contrast input**：如果不提供则跳过

### 6. AUROC Tumour-ROI (classification.py)

```python
def _compute_auroc_tumor_roi(cases):
    features = []
    labels = []
    
    for case in cases:
        # 从 tumor ROI 提取 features
        tumor_features = extract_radiomic_features(case.prediction, case.mask)
        features.append(tumor_features)
        labels.append(1)  # tumor region
        
        # 从 contralateral mirrored ROI 提取 features
        mirrored_mask = create_mirrored_mask(case.mask)  # 镜像翻转到对侧
        mirror_features = extract_radiomic_features(case.prediction, mirrored_mask)
        features.append(mirror_features)
        labels.append(0)  # normal region
    
    probs = classifier.predict_proba(features)[:, 1]
    auroc = roc_auc_score(labels, probs)
```

- **衡量**：tumor 区域 vs 对侧正常组织的增强差异
- **Mirror**：检测 midline（胸骨中线），将 tumor mask 翻转到对侧
- **AUROC 高**：说明 tumor 增强有区分性（不是全图均匀增强）

### 7. Dice (segmentation.py)

```python
def evaluate(case):
    # Step 1: 用 nnUNet 对 synthetic prediction 做 tumor 分割
    pred_mask = nnunet_segment(case.prediction)  # → binary mask
    
    # Step 2: 和 GT tumor mask 比较
    gt_mask = case.mask
    
    # Step 3: 计算 Dice
    intersection = np.sum(pred_mask & gt_mask)
    dice = 2 * intersection / (np.sum(pred_mask) + np.sum(gt_mask))
    
    # 如果 nnUNet 输出空 mask → dice = 0
```

- **nnUNet 模型**：2D nnUNet，fold_0，在 MAMA-MIA 数据上训练
- **关键含义**：合成图像的增强是否足够让分割模型找到 tumor
- **如果增强太弱/模糊** → nnUNet 找不到 → Dice = 0

### 8. HD95 (segmentation.py)

```python
def compute_hausdorff_95(pred_mask, gt_mask):
    # Step 1: 计算 pred_mask 边界点到 gt_mask 的最近距离
    dist_pred_to_gt = distance_transform_edt(~gt_mask)
    surface_pred = pred_mask & ~ndimage.binary_erosion(pred_mask)
    d_pred = dist_pred_to_gt[surface_pred]
    
    # Step 2: 反过来，gt → pred
    dist_gt_to_pred = distance_transform_edt(~pred_mask)
    surface_gt = gt_mask & ~ndimage.binary_erosion(gt_mask)
    d_gt = dist_gt_to_pred[surface_gt]
    
    # Step 3: 合并取 P95
    all_distances = np.concatenate([d_pred, d_gt])
    hd95 = np.percentile(all_distances, 95)
    
    # 如果 nnUNet 输出空 mask → hd95 = 图像对角线 = sqrt(H² + W²)
```

- **衡量边界精度**：分割轮廓偏差多远
- **P95**：去掉极端 5% 的离群点，更鲁棒
- **单位**：像素距离

---

## 关键依赖关系

```
                    ┌─── MSE (全图)
prediction ────────┤
                    └─── LPIPS (全图, ±5σ clip → AlexNet)

                    ┌─── SSIM_tumor (mask 内 local-window SSIM)
prediction + mask ──┤
                    └─── FRD (mask 内 radiomics → Fréchet distance)

                    ┌─── AUROC_contrast (pred_ROI vs pre_ROI → classifier)
pred + mask + pre ──┤
                    └─── AUROC_tumor_ROI (tumor_ROI vs mirror_ROI → classifier)

                    ┌─── Dice (nnUNet(pred) vs gt_mask)
prediction + mask ──┤
                    └─── HD95 (nnUNet(pred) boundary vs gt_mask boundary)
```

## 影响因素总结

| 如果你的模型... | 影响的指标 | 方向 |
|---|---|---|
| 全图 pixel 不准 | MSE ↑, LPIPS ↑ | 差 |
| tumor 区域增强不够强 | AUROC_contrast ↓, Dice ↓ | 差 |
| tumor 增强模糊/边界不清 | HD95 ↑, SSIM_tumor ↓ | 差 |
| 增强不够有区分性（全图均匀亮） | AUROC_tumor_ROI ↓ | 差 |
| radiomics 分布偏移 | FRD ↑ | 差 |
| 背景有假增强 | MSE ↑, nnUNet 假阳性 → Dice ↓ | 差 |

---

## 🚀 针对各指标的改进策略

### 当前状态 (v31_pinorm, v3 test 299 cases)

| Metric | 当前 | GC #1 | 差距 | 优先级 |
|--------|:---:|:---:|:---:|:---:|
| MSE | 1.04 | 0.57 | 1.8x | 🔴 高 |
| LPIPS | 0.126 | 0.08 | 1.6x | 🔴 高 |
| SSIM_tumor | 0.476 | 0.43 | ✅ 超过 | 🟢 已满足 |
| FRD | 24.0 | 25.1 | ✅ 超过 | 🟢 已满足 |
| AUROC | 0.757 | 0.80 | -5% | 🟡 中 |
| Dice | 0.539 | 0.48 | ✅ 超过 | 🟢 已满足 |
| HD95 | 125.6 | 120.6 | ≈ 持平 | 🟡 中 |

---

### 🔴 降低 MSE (1.04 → <0.7)

**根因分析**：
- Top 10% outlier cases 贡献 42% 总 MSE
- 这些 cases GT enhancement 极强 (max 24-67)
- Per-image de-norm 后误差被放大 (乘以 img_std)
- Median MSE = 0.40（大多数 case 表现好）

**改进方案**:

| # | 方案 | 预期效果 | 代价 | 状态 |
|---|------|---------|------|------|
| 1 | **Ensemble 多模型平均** | MSE -30~50% | 推理时间 Nx | 可立即做 |
| 2 | **ngf=96 + pinorm (v26_pinorm)** | MSE -10~15% | 训练 1x | ⏳ 训练中 |
| 3 | **Huber loss + GT P99 clip** | MSE -5~10% | 训练 1x | ⏳ 重训中 |
| 4 | **更深网络 n_blocks=15** | MSE -3~5% | 训练 +10% | ⏳ 训练中 |
| 5 | **TTA (Test-Time Augmentation)** | MSE -5~10% | 推理 4-8x | 未实现 |
| 6 | **增大训练数据 (AMBL)** | 减少 outlier | 无额外推理代价 | ✅ 已加入训练 |

**Ensemble 具体做法**:
```python
# 最简单：2-3 个模型的输出平均
output = (pred_v31_pinorm + pred_v22_msv3 + pred_v26_pinorm) / 3
```
- 已验证 kfold ensemble MSE 降 76%（0.717 → 0.174）
- 3 模型推理 ~3x 时间 ≈ 3s（T4 10min 限制内）

---

### 🔴 降低 LPIPS (0.126 → <0.10)

**根因**：LPIPS 由 AlexNet 高级特征决定，需要更精细的纹理/边界。

**改进方案**:

| # | 方案 | 预期效果 | 状态 |
|---|------|---------|------|
| 1 | **SDEdit diffusion refiner** | LPIPS -13~15% | 待训练 |
| 2 | **加大 lambda_vgg (10→20)** | LPIPS -3~5% | ⏳ v22_vgg20 训练中 |
| 3 | **Ensemble 平均** | LPIPS -5~10% | 可立即做 |
| 4 | **换 LPIPS 网络 (VGG → SqueezeNet)** | 不确定 | 未实现 |

**SDEdit refiner 最有效**:
- 在 GAN 输出上加 30% 噪声 → 20 步 DDIM 去噪 → 精修 tumor 边界和纹理
- v17 验证：LPIPS 从 0.126 降到 0.109 (-13%)
- 推理额外 +5s/case（仍在限制内）

---

### 🟡 提升 AUROC Contrast (0.757 → 0.80)

**根因**：tumor ROI 内增强信号不够强/不够逼真。

**改进方案**:

| # | 方案 | 做法 | 预期 |
|---|------|------|------|
| 1 | 加大 tumor_weight | `--tumor_weight 20`（当前 10） | 强制 tumor 区域精确重建 |
| 2 | 加强 MSSC loss | `--lambda_mssc 100`（当前 50） | 增强 subtraction consistency |
| 3 | 加入 SSIM loss | `--lambda_ssim 5 --lambda_vgg 5` | 保留 tumor 结构细节 |
| 4 | 不做 GT clip | 去掉 P99 clip | 保留 tumor 强增强信号 |
| 5 | 只用 v3 数据训练 | GT 统一 peak phase | GT 一致 → 增强更强 |

**关键发现**：v31b_v3 (GT P95 clip) AUROC=0.545 崩溃证明 GT clip 太激进会截断 tumor 信号。
修复后 (P99) 预期 AUROC 恢复到 ~0.70-0.75。

---

### 🟡 降低 HD95 (125.6 → <120)

**根因**：nnUNet 分割的 tumor 边界偏差。

**改进方案**:

| # | 方案 | 预期效果 | 状态 |
|---|------|---------|------|
| 1 | **SDEdit refiner** | HD95 -9% (v17 验证) | 待训练 |
| 2 | **更深网络 (n_blocks=15)** | 更大感受野 → 更好 tumor 边界 | ⏳ 训练中 |
| 3 | **Ensemble** | 边界平滑，减少分割噪声 | 可立即做 |
| 4 | **训练时加 edge loss** | 显式惩罚边界模糊 | 未实现 |

---

### 🟢 保持已超过的指标 (SSIM, FRD, Dice)

这三个指标已超过 GC #1。风险：新改进可能 trade-off 伤害它们。

**保护策略**:
- 不要做太激进的 GT clip（伤害 SSIM/Dice）
- 保持 per-image norm（核心贡献）
- Ensemble 通常不会降低这些指标（只会更好）
- 监控 AUROC：如果 AUROC < 0.7 说明增强太弱

---

## 📋 综合改进路线图

### 第一阶段：不需要新训练（立即可做）

| 方案 | 做法 | 影响指标 |
|------|------|---------|
| **Ensemble** | v31_pinorm + v22_msv3 + v26_msv3 平均 | MSE ↓↓, LPIPS ↓, HD95 ↓ |
| **TTA** | 水平翻转推理取平均 | MSE ↓, LPIPS ↓ |

### 第二阶段：正在训练

| 模型 | 预期最强指标 | 完成时间 |
|------|------------|---------|
| v26_pinorm (ngf=96 + pinorm) | MSE 最低 | ~18h |
| v31_deeper (n_blocks=15) | HD95/Dice 改善 | ~20h |
| v31b_v3 (huber + P99 clip) | MSE + 均衡 | ~18h |

### 第三阶段：需要新训练

| 方案 | 目标 | 预期效果 |
|------|------|---------|
| SDEdit on v31_pinorm | LPIPS, HD95 | LPIPS -15%, HD95 -9% |
| v26_pinorm + SDEdit | 全面最强单模型 | 接近 GC #1 |
| Final ensemble | 提交用 | MSE -40%, 冲 GC #1 |

### 最终提交策略

```
Best single model: v26_pinorm (预期)
Best ensemble:     v31_pinorm + v26_pinorm + v31_deeper (3 models)
Docker size:       3 × 700MB weights = 2.1GB + nnUNet 1.6GB + base = ~5GB (< 10GB)
Inference time:    3 × 1s + breast_seg 1s = ~4s (< 10min)
```
