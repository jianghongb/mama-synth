#!/bin/bash
# 简化版蒸馏：直接用 BreastDividerDataset 的 GT labels 训练 2D nnUNet
# 跳过 3D Teacher 推理（因为数据集已有 labels）
set -e

BASE="/home/maia-user/jh"
BD_DATASET="$BASE/BreastDividerDataset"
DATASET_DIR="$BASE/distill/Dataset930"

export nnUNet_raw="$BASE/distill/nnUNet_raw"
export nnUNet_preprocessed="$BASE/distill/nnUNet_preprocessed"
export nnUNet_results="$BASE/distill/nnUNet_results"
mkdir -p $nnUNet_raw $nnUNet_preprocessed $nnUNet_results

# ============================================================
# Step 1: 从 3D 数据提取 2D slices → nnUNet 训练格式
# ============================================================
echo "=== Step 1: 提取 2D slices ==="
python3 << 'EOF'
import json, numpy as np, nibabel as nib
from pathlib import Path
from tqdm import tqdm

bd = Path("/home/maia-user/jh/BreastDividerDataset")
dataset_dir = Path("/home/maia-user/jh/distill/Dataset930")
images_dir = dataset_dir / "imagesTr"
labels_dir = dataset_dir / "labelsTr"
images_dir.mkdir(parents=True, exist_ok=True)
labels_dir.mkdir(parents=True, exist_ok=True)

# Collect all image/label pairs from batch1 + batch2
pairs = []
for batch in ["batch1", "batch2"]:
    img_dir = bd / f"imagesTr_{batch}"
    lbl_dir = bd / f"labelsTr_{batch}"
    if not img_dir.exists():
        continue
    for lbl_file in sorted(lbl_dir.glob("*.nii.gz")):
        case_id = lbl_file.name.replace(".nii.gz", "")
        img_file = img_dir / f"{case_id}_0000.nii.gz"
        if img_file.exists():
            pairs.append((img_file, lbl_file, case_id))

print(f"Found {len(pairs)} image/label pairs")

count = 0
for img_file, lbl_file, case_id in tqdm(pairs):
    out_img = images_dir / f"{case_id}_0000.nii.gz"
    out_lbl = labels_dir / f"{case_id}.nii.gz"
    if out_img.exists() and out_lbl.exists():
        count += 1
        continue

    # Load 3D
    img_3d = nib.load(str(img_file)).get_fdata().astype(np.float32)
    lbl_3d = nib.load(str(lbl_file)).get_fdata()

    # Take slice with largest breast area (axial = last axis)
    breast_area = (lbl_3d > 0).sum(axis=(0, 1))
    mid = int(np.argmax(breast_area))
    img_2d = img_3d[:, :, mid]
    lbl_2d = (lbl_3d[:, :, mid] > 0).astype(np.int16)  # binary: left+right = breast

    # Skip empty slices
    if lbl_2d.sum() < 100:
        continue

    # Save as (1, H, W) for nnUNet 2D
    affine = np.eye(4)
    nib.save(nib.Nifti1Image(img_2d[np.newaxis], affine), str(out_img))
    nib.save(nib.Nifti1Image(lbl_2d[np.newaxis], affine), str(out_lbl))
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

print(f"Done: {count} 2D training cases → {dataset_dir}")
EOF

# ============================================================
# Step 2: 链接到 nnUNet_raw
# ============================================================
echo "=== Step 2: 链接到 nnUNet_raw ==="
ln -sfn $DATASET_DIR $nnUNet_raw/Dataset930_BreastDivider2D

# ============================================================
# Step 3: nnUNet Plan + Preprocess + Train
# ============================================================
echo "=== Step 3: nnUNet plan + preprocess ==="
nnUNetv2_plan_and_preprocess -d 930 -c 2d --verify_dataset_integrity

echo "=== Step 4: Training fold 0 ==="
nnUNetv2_train 930 2d 0 --npz

echo "=== 完成！==="
echo "模型: $nnUNet_results/Dataset930_BreastDivider2D/nnUNetTrainer__nnUNetPlans__2d/fold_0/"
