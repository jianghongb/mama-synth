# Preprocessing

将 3D DCE-MRI volume 转换为 2D slice，用于模型训练和评估。

## 文件说明

| 文件 | 用途 |
|------|------|
| `compute_dataset_stats.py` | 计算全局 z-score 归一化参数 (mean/std) |
| `preprocess.py` | 3D→2D 转换核心逻辑 |
| `preprocess_split.py` | 按 train/test split 执行 3D→2D 转换 |
| `training_pre_stats.json` | 预计算的归一化参数 |
| `test_preprocess.py` | 预处理单元测试 |

## 使用方法

### 1. 按 train/test 划分并预处理数据

```bash
cd /Users/ehogjig/git/kth/mama-synth

.venv/bin/python src/preprocessing/preprocess_split.py \
    --image_dir ./images \
    --seg_dir ./segmentations/automatic \
    --output_dir ./data_split \
    --splits_csv ./train_test_splits.csv \
    --global_stats ./src/preprocessing/training_pre_stats.json \
    --skip_ambiguous_shapes
```

参数说明：
- `--image_dir`：3D DCE-MRI 数据目录（每个患者一个子目录，含多个 phase）
- `--seg_dir`：肿瘤分割目录（`segmentations/automatic` 或 `segmentations/expert`）
- `--output_dir`：输出目录
- `--splits_csv`：train/test 划分 CSV 文件
- `--global_stats`：全局归一化参数 JSON
- `--skip_ambiguous_shapes`：跳过无法确定切片方向的患者（三个维度都不同）

### 2. 输出结构

```
data_split/
├── train/
│   ├── mha/
│   │   ├── input/DUKE_001.mha          # 2D pre-contrast (z-score float32)
│   │   ├── ground_truth/DUKE_001.mha   # 2D peak-enhancement
│   │   └── mask/DUKE_001.mha           # 2D tumour mask
│   ├── png/                            # 可视化用
│   │   ├── input/
│   │   ├── ground_truth/
│   │   └── mask/
│   ├── intensity_plots/                # per-patient 强度曲线图
│   └── report.csv                      # per-patient 预处理报告
└── test/
    ├── mha/
    │   ├── input/
    │   ├── ground_truth/
    │   └── mask/
    ├── png/
    ├── intensity_plots/
    └── report.csv
```

### 3. 处理逻辑

1. 加载患者所有 3D phase volume 和肿瘤分割
2. 确定 peak enhancement phase（肿瘤区域平均强度最高的 phase）
3. 选择肿瘤面积最大的 2D slice
4. 使用全局 mean/std 进行 z-score 归一化
5. 保存为 .mha（float32）

### 4. 仅计算归一化参数（已预计算）

```bash
.venv/bin/python src/preprocessing/compute_dataset_stats.py \
    --image_dir ./images \
    --output_path ./src/preprocessing/training_pre_stats.json
```
