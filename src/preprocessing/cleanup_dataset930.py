#!/usr/bin/env python
"""Remove bad cases from Dataset930. 
Run multiple times with increasing threshold if training still crashes."""
import nibabel as nib
import os, json
from pathlib import Path
from collections import Counter

img_dir = Path('/home/maia-user/jh/distill/Dataset930/imagesTr')
lbl_dir = Path('/home/maia-user/jh/distill/Dataset930/labelsTr')

# First: show size distribution
print("=== Size distribution ===")
sizes = []
for f in sorted(img_dir.glob('*.nii.gz')):
    img = nib.load(str(f))
    sz = img.shape[:2]  # (H, W)
    sizes.append(min(sz))

size_bins = Counter()
for s in sizes:
    if s < 32: size_bins['<32'] += 1
    elif s < 64: size_bins['32-63'] += 1
    elif s < 128: size_bins['64-127'] += 1
    elif s < 256: size_bins['128-255'] += 1
    else: size_bins['>=256'] += 1

for k in sorted(size_bins.keys()):
    print(f"  {k}: {size_bins[k]} cases")
print(f"  Total: {len(sizes)} cases")
print()

# Remove cases with min dimension < THRESHOLD
THRESHOLD = 64  # Increase this if training still crashes
bad = []
for f in sorted(img_dir.glob('*.nii.gz')):
    img = nib.load(str(f))
    sz = img.shape[:2]
    if min(sz) < THRESHOLD:
        bad.append(f.stem.replace('_0000', ''))

print(f'Removing {len(bad)} cases with min dim < {THRESHOLD}')
for stem in bad:
    img_f = img_dir / f'{stem}_0000.nii.gz'
    lbl_f = lbl_dir / f'{stem}.nii.gz'
    if img_f.exists():
        os.remove(img_f)
    if lbl_f.exists():
        os.remove(lbl_f)

remaining = len(list(img_dir.glob('*.nii.gz')))
ds_json = Path('/home/maia-user/jh/distill/Dataset930/dataset.json')
with open(ds_json) as f:
    d = json.load(f)
d['numTraining'] = remaining
with open(ds_json, 'w') as f:
    json.dump(d, f, indent=2)
print(f'Done. Remaining: {remaining} cases')
