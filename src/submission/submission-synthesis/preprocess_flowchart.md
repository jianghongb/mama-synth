# Preprocess 流程图

## 输入

```
原始数据 (每个患者一个文件夹):
  images/PATIENT_ID/
    PATIENT_ID_0000.nii.gz    ← pre-contrast (phase 0)
    PATIENT_ID_0001.nii.gz    ← post-contrast phase 1
    PATIENT_ID_0002.nii.gz    ← post-contrast phase 2 (可选)
    ...

  segmentations/
    PATIENT_ID.nii.gz         ← 3D tumor segmentation mask
```

## 处理流程

```
对每个患者:

Step 1: 加载分割 mask
   segmentation.nii.gz → 3D numpy array
         │
         ▼
Step 2: 确定 axial 切面方向
   • 优先用 NIfTI orientation (axcodes: S/I = axial)
   • fallback: shape heuristic (方形面 = in-plane)
   • 三维都不同 → skip (--skip_ambiguous_shapes)
         │
         ▼
Step 3: 找到 tumor 最大的 slice
   • 沿 axial 方向，找 tumor mask 面积最大的 slice index
   • 这就是我们提取的那一层
         │
         ▼
Step 4: 确定 peak enhancement phase
   • 对每个 post-contrast phase:
     - 提取同一层 slice
     - 计算 tumor 区域平均强度
   • 选平均强度最高的 phase = peak phase
         │
         ▼
Step 5: 提取 2D slices
   • pre = phase_0 的 axial slice [H, W]
   • post_peak = peak_phase 的 axial slice [H, W]
   • mask = segmentation 的 axial slice [H, W]
         │
         ▼
Step 6: 计算 subtraction image
   • subtraction = post_peak - pre
   • 这是 GT (ground truth): 造影增强差异图
         │
         ▼
Step 7: Z-score 归一化
   • pre_norm = (pre - global_mean) / global_std
   • sub_norm = (subtraction - global_mean) / global_std
   • 使用 training_pre_stats.json 中的 mean/std
         │
         ▼
Step 8: 保存
   • output/mha/input/PATIENT_ID.mha         ← pre (z-score, float32)
   • output/mha/ground_truth/PATIENT_ID.mha  ← subtraction (z-score, float32)
   • output/mha/mask/PATIENT_ID.mha          ← binary tumor mask (int16)
   • output/png/...                           ← 可视化用
   • output/intensity_plots/...               ← 增强曲线图
```

## 输出

```
output/
├── mha/
│   ├── input/           ← pre-contrast, 2D, z-score float32
│   ├── ground_truth/    ← subtraction (post-pre), 2D, z-score float32
│   └── mask/            ← tumor binary mask, 2D, int16
├── png/                 ← 可视化 PNG
├── intensity_plots/     ← 每个患者的增强曲线
└── report.csv           ← 处理报告
```

## 关键参数

| 参数 | 来源 | 用途 |
|------|------|------|
| `global_mean` | training_pre_stats.json | z-score 归一化 |
| `global_std` | training_pre_stats.json | z-score 归一化 |
| `slice_axis` | NIfTI header / shape | 确定 axial 方向 |
| `peak_phase` | tumor 区域强度 | 选最强增强的时间点 |

## 命令

```bash
python preprocess.py \
  --image_dir /path/to/images \
  --seg_dir /path/to/segmentations \
  --output_dir /path/to/output \
  --global_stats training_pre_stats.json \
  --skip_ambiguous_shapes
```
