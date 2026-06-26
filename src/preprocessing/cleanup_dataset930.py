#!/usr/bin/env python
"""Remove bad cases (min dimension < 32) from Dataset930 and retrain."""
import nibabel as nib
import os, json
from pathlib import Path

img_dir = Path('/home/maia-user/jh/distill/Dataset930/imagesTr')
lbl_dir = Path('/home/maia-user/jh/distill/Dataset930/labelsTr')
bad = []

for f in sorted(img_dir.glob('*.nii.gz')):
    img = nib.load(str(f))
    sz = img.shape
    if min(sz[:2]) < 32:
        bad.append(f.stem.replace('_0000', ''))

print(f'Removing {len(bad)} bad cases')
for stem in bad:
    img_f = img_dir / f'{stem}_0000.nii.gz'
    lbl_f = lbl_dir / f'{stem}.nii.gz'
    if img_f.exists():
        os.remove(img_f)
    if lbl_f.exists():
        os.remove(lbl_f)
    print(f'  removed {stem}')

remaining = len(list(img_dir.glob('*.nii.gz')))
ds_json = Path('/home/maia-user/jh/distill/Dataset930/dataset.json')
with open(ds_json) as f:
    d = json.load(f)
d['numTraining'] = remaining
with open(ds_json, 'w') as f:
    json.dump(d, f, indent=2)
print(f'Updated numTraining to {remaining}')
