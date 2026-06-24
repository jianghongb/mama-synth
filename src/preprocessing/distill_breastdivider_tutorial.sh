#!/bin/bash
# ============================================================
# 3D → 2D Breast Segmentation 蒸馏教程 (本地版)
#
# 概念：
#   Teacher (3D): BreastDivider — 已训好的 3D nnUNet，输入 3D volume，输出 3D breast mask
#   Student (2D): 你要训的 2D nnUNet，输入 2D slice，输出 2D breast mask
#
# 流程：
#   3D volume → Teacher → 3D pseudo label → 提取 2D slice → 训练 Student
#
# 为什么不直接用 3D 模型？
#   推理时你的输入是 2D MHA slice（比赛只给 2D），没有 3D volume 可用。
#   所以需要一个能处理 2D 输入的 breast seg 模型。
# ============================================================

set -e

# === 路径配置 (MAIA) ===
BASE="/home/maia-user/jh"
MODEL_DIR="$BASE/BreastDividerModel"
BD_DATASET="$BASE/BreastDividerDataset"    # 下载的额外数据集（提升泛化）
INPUT_3D="$BASE/distill/input_3d"
PSEUDO_3D="$BASE/distill/pseudo_labels"
DATASET_DIR="$BASE/distill/Dataset930"

export nnUNet_raw="$BASE/distill/nnUNet_raw"
export nnUNet_preprocessed="$BASE/distill/nnUNet_preprocessed"
export nnUNet_results="$BASE/distill/nnUNet_results"
mkdir -p $nnUNet_raw $nnUNet_preprocessed $nnUNet_results

# ============================================================
# Step 1: 准备 3D 输入
# ============================================================
# nnUNet 要求输入命名为: <case_id>_0000.nii.gz (0000 = 第一个通道)
#
# 你的数据: images/<patient>/<patient>_0000.nii.gz (pre-contrast)
# 需要 symlink 到一个 flat 目录：
#
# input_3d/
#     DUKE_001_0000.nii.gz → images/DUKE_001/DUKE_001_0000.nii.gz
#     DUKE_002_0000.nii.gz → images/DUKE_002/DUKE_002_0000.nii.gz

echo "=== Step 1: 准备输入 ==="
mkdir -p $INPUT_3D

# 使用 BreastDividerDataset（多中心、多模态 3D breast MRI）
if [ -d "$BD_DATASET" ]; then
    for f in $BD_DATASET/imagesTr/*_0000.nii.gz; do
        [ -f "$f" ] && ln -sf "$f" "$INPUT_3D/$(basename $f)"
    done
    for f in $BD_DATASET/imagesTs/*_0000.nii.gz; do
        [ -f "$f" ] && ln -sf "$f" "$INPUT_3D/$(basename $f)"
    done
fi

echo "  输入文件: $(ls $INPUT_3D/*.nii.gz 2>/dev/null | wc -l)"

# ============================================================
# Step 2: 用 3D Teacher 生成 pseudo labels
# ============================================================
# BreastDivider 输出: 每个 voxel 的 label (0=background, 1+=breast regions)
#
# 命令: nnUNetv2_predict_from_modelfolder
#   -i: 输入目录 (3D volumes)
#   -o: 输出目录 (3D segmentation masks)
#   -m: 模型目录
#   --disable_tta: 关闭 test-time augmentation (更快)

echo "=== Step 2: Teacher 推理 (3D → 3D pseudo labels) ==="
mkdir -p $PSEUDO_3D

nnUNetv2_predict_from_modelfolder \
    -i $INPUT_3D \
    -o $PSEUDO_3D \
    -m $MODEL_DIR \
    --disable_tta

echo "  Pseudo labels: $(ls $PSEUDO_3D/*.nii.gz 2>/dev/null | wc -l)"

# ============================================================
# Step 3: 提取 2D slices → nnUNet 训练数据
# ============================================================
# 对每个 3D pseudo label，提取和你 preprocess 相同的 2D slice
# 同时准备对应的 2D input image
#
# nnUNet 训练数据格式:
#   Dataset930/
#       imagesTr/          ← 2D input images (patient_0000.nii.gz)
#       labelsTr/          ← 2D labels (patient.nii.gz)
#       dataset.json       ← 描述文件

echo "=== Step 3: 提取 2D slices ==="
python3 << 'EOF'
import json, numpy as np, nibabel as nib
from pathlib import Path

input_3d_dir = Path("/home/maia-user/jh/distill/input_3d")
pseudo_dir = Path("/home/maia-user/jh/distill/pseudo_labels")
dataset_dir = Path("/home/maia-user/jh/distill/Dataset930")
images_dir = dataset_dir / "imagesTr"
labels_dir = dataset_dir / "labelsTr"
images_dir.mkdir(parents=True, exist_ok=True)
labels_dir.mkdir(parents=True, exist_ok=True)

count = 0
for pseudo_file in sorted(pseudo_dir.glob("*.nii.gz")):
    case_id = pseudo_file.stem.replace(".nii", "")

    # 对应的 3D input image
    input_file = input_3d_dir / f"{case_id}_0000.nii.gz"
    if not input_file.exists():
        continue

    # 加载
    img_3d = nib.load(str(input_file)).get_fdata().astype(np.float32)
    label_3d = nib.load(str(pseudo_file)).get_fdata()

    # 取 axial 中间 slice（最后一个 axis）
    mid = img_3d.shape[2] // 2
    img_2d = img_3d[:, :, mid]
    label_2d = (label_3d[:, :, mid] > 0).astype(np.int16)

    # 跳过没有 breast 的 slice
    if label_2d.sum() < 100:
        continue

    # 保存 (nnUNet 格式: shape = (1, H, W))
    affine = np.eye(4)
    nib.save(nib.Nifti1Image(img_2d[np.newaxis], affine),
             str(images_dir / f"{case_id}_0000.nii.gz"))
    nib.save(nib.Nifti1Image(label_2d[np.newaxis], affine),
             str(labels_dir / f"{case_id}.nii.gz"))
    count += 1

# dataset.json
dataset_json = {
    "channel_names": {"0": "T1"},
    "labels": {"background": 0, "breast": 1},
    "numTraining": count,
    "file_ending": ".nii.gz"
}
with open(dataset_dir / "dataset.json", "w") as f:
    json.dump(dataset_json, f, indent=2)

print(f"Dataset930: {count} training cases ready")
EOF

# ============================================================
# Step 4: 训练 2D Student
# ============================================================
# nnUNet 自动处理 preprocessing + training

echo "=== Step 4: 训练 2D nnUNet ==="

# 把 Dataset930 放到 nnUNet_raw
ln -sf $(realpath $DATASET_DIR) $nnUNet_raw/Dataset930_BreastDivider2D

# Plan + Preprocess
nnUNetv2_plan_and_preprocess -d 930 -c 2d --verify_dataset_integrity

# Train fold 0
nnUNetv2_train 930 2d 0 --npz

echo "=== 完成！==="
echo "模型在: $nnUNet_results/Dataset930_BreastDivider2D/nnUNetTrainer__nnUNetPlans__2d/fold_0/"
echo ""
echo "使用方式 (推理时):"
echo "  nnUNetv2_predict -i input_2d/ -o output/ -d 930 -c 2d -f 0"
