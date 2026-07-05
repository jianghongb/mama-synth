"""
split_ambl_test_v2.py — Move 5% of AMBL patients from train to test in data_multislice_v2.

Usage:
    python src/preprocessing/split_ambl_test_v2.py
"""
import random
from pathlib import Path
import shutil

random.seed(42)

base = Path('/proj/berzbiomedicalimagingkth/users/x_honji/data_multislice_v2')
train_input = base / 'train' / 'mha' / 'input'
test_input = base / 'test' / 'mha' / 'input'

ambl_files = sorted(train_input.glob('AMBL_*.mha'))
patient_ids = sorted(set(f.name.rsplit('_p', 1)[0] for f in ambl_files))
print(f'AMBL patients in v2: {len(patient_ids)}')

n_test = max(1, int(len(patient_ids) * 0.05))
random.shuffle(patient_ids)
test_patients = set(patient_ids[:n_test])
print(f'Test patients ({n_test}): {sorted(test_patients)[:5]}...')

subdirs = ['input', 'ground_truth', 'mask', 'breast_mask']
moved = 0
for subdir in subdirs:
    src_dir = base / 'train' / 'mha' / subdir
    dst_dir = base / 'test' / 'mha' / subdir
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(src_dir.glob('AMBL_*.mha')):
        pid = f.name.rsplit('_p', 1)[0]
        if pid in test_patients:
            shutil.move(str(f), str(dst_dir / f.name))
            moved += 1

print(f'Moved {moved} files to test')
print(f'Train AMBL remaining: {len(list(train_input.glob("AMBL_*.mha")))}')
print(f'Test AMBL: {len(list(test_input.glob("AMBL_*.mha")))}')
