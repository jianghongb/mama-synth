"""
move_ambl_to_separate.py — Move AMBL data from data_multislice_v3 to data_multislice_ambl.

Moves all AMBL_* files from both train and test to a new separate directory.
This keeps AMBL available for training but removes it from evaluation.

Usage:
    python src/preprocessing/move_ambl_to_separate.py
"""
from pathlib import Path
import shutil

PROJ = Path('/proj/berzbiomedicalimagingkth/users/x_honji')
src_base = PROJ / 'data_multislice_v3'
dst_base = PROJ / 'data_multislice_ambl'

subdirs = ['input', 'ground_truth', 'mask', 'breast_mask']
splits = ['train', 'test']

moved = 0
for split in splits:
    for subdir in subdirs:
        src_dir = src_base / split / 'mha' / subdir
        dst_dir = dst_base / split / 'mha' / subdir
        dst_dir.mkdir(parents=True, exist_ok=True)

        if not src_dir.exists():
            continue

        for f in sorted(src_dir.glob('AMBL_*.mha')):
            shutil.move(str(f), str(dst_dir / f.name))
            moved += 1

print(f'Moved {moved} AMBL files to {dst_base}')

# Verify
for split in splits:
    v3_count = len(list((src_base / split / 'mha' / 'input').glob('*.mha')))
    ambl_count = len(list((dst_base / split / 'mha' / 'input').glob('*.mha')))
    print(f'  {split}: v3 remaining={v3_count}, ambl moved={ambl_count}')
