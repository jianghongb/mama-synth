# BreastDivider 2D Distillation on Kaggle
# 
# Setup:
# 1. Create Kaggle Notebook with GPU (T4/P100)
# 2. Add dataset: huggingface dataset "Bubenpo/BreastDividerDataset"
#    OR upload as Kaggle dataset
# 3. Copy this into a code cell and run

# ============================================================
# Cell 1: Install dependencies
# ============================================================
!pip install nnunetv2 nibabel -q

# ============================================================
# Cell 2: Extract 2D slices from 3D volumes
# ============================================================
import nibabel as nib
import numpy as np
import json, os
from pathlib import Path

# Kaggle dataset mount path (adjust if different)
BD_DIR = Path("/kaggle/input/breastdividerdataset")  
OUT_DIR = Path("/kaggle/working/Dataset930")
IMAGES_DIR = OUT_DIR / "imagesTr"
LABELS_DIR = OUT_DIR / "labelsTr"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
LABELS_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 64
count = 0

for batch in ['imagesTr_batch1', 'imagesTr_batch2']:
    img_batch = BD_DIR / batch
    lbl_batch = BD_DIR / batch.replace('imagesTr', 'labelsTr')
    if not img_batch.exists():
        print(f"Skipping {batch} (not found)")
        continue
    for img_f in sorted(img_batch.glob('*_0000.nii.gz')):
        stem = img_f.name.replace('_0000.nii.gz', '')
        lbl_f = lbl_batch / f'{stem}.nii.gz'
        if not lbl_f.exists():
            continue
        img = nib.load(str(img_f))
        lbl = nib.load(str(lbl_f))
        img_data = img.get_fdata()
        lbl_data = lbl.get_fdata()
        
        if img_data.ndim == 3:
            mid = img_data.shape[2] // 2
            img_2d = img_data[:, :, mid].astype(np.float32)
            lbl_2d = lbl_data[:, :, mid].astype(np.uint8)
        else:
            img_2d = img_data.astype(np.float32)
            lbl_2d = lbl_data.astype(np.uint8)
        
        if min(img_2d.shape[:2]) < THRESHOLD:
            continue
        
        lbl_2d = (lbl_2d > 0).astype(np.uint8)
        affine = np.eye(4)
        nib.save(nib.Nifti1Image(img_2d[:, :, np.newaxis], affine), str(IMAGES_DIR / f'{stem}_0000.nii.gz'))
        nib.save(nib.Nifti1Image(lbl_2d[:, :, np.newaxis], affine), str(LABELS_DIR / f'{stem}.nii.gz'))
        count += 1
        if count % 2000 == 0:
            print(f"  {count} cases...")

ds = {'channel_names': {'0': 'MRI'}, 'labels': {'background': 0, 'breast': 1}, 
      'numTraining': count, 'file_ending': '.nii.gz'}
with open(OUT_DIR / 'dataset.json', 'w') as f:
    json.dump(ds, f, indent=2)
print(f"Done: {count} 2D cases extracted")

# ============================================================
# Cell 3: Setup nnUNet paths and preprocess
# ============================================================
import os
os.environ['nnUNet_raw'] = '/kaggle/working/nnUNet_raw'
os.environ['nnUNet_preprocessed'] = '/kaggle/working/nnUNet_preprocessed'
os.environ['nnUNet_results'] = '/kaggle/working/nnUNet_results'
os.environ['TORCHDYNAMO_DISABLE'] = '1'

os.makedirs('/kaggle/working/nnUNet_raw', exist_ok=True)
os.makedirs('/kaggle/working/nnUNet_preprocessed', exist_ok=True)
os.makedirs('/kaggle/working/nnUNet_results', exist_ok=True)
os.symlink('/kaggle/working/Dataset930', '/kaggle/working/nnUNet_raw/Dataset930_BreastDivider2D')

!nnUNetv2_plan_and_preprocess -d 930 -c 2d --verify_dataset_integrity

# ============================================================
# Cell 4: Train (will run until 12h limit)
# ============================================================
# Use --c to continue from checkpoint if re-running
!nnUNetv2_train 930 2d 0 --npz --c

# ============================================================
# Cell 5: Save results (download from Kaggle output)
# ============================================================
# After training, the model will be in:
# /kaggle/working/nnUNet_results/Dataset930_BreastDivider2D/nnUNetTrainer__nnUNetPlans__2d/fold_0/
# 
# Download checkpoint_final.pth and checkpoint_best.pth from Kaggle output
import shutil
results_dir = Path('/kaggle/working/nnUNet_results/Dataset930_BreastDivider2D/nnUNetTrainer__nnUNetPlans__2d/fold_0')
if results_dir.exists():
    for f in results_dir.glob('checkpoint_*.pth'):
        shutil.copy(f, f'/kaggle/working/{f.name}')
        print(f"Saved: {f.name}")
