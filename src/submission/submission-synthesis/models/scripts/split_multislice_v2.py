"""Split data_multislice_v2 into train/test (95%/5%) by case_id per dataset.

Ensures no case leakage: all slices from a case go to either train OR test.
Stratified by dataset prefix (DUKE, ISPY2, etc.) so each source contributes 5% to test.

Usage:
    python split_multislice_v2.py \
        --dataroot /proj/berzbiomedicalimagingkth/users/x_honji/data_multislice_v2 \
        --test_ratio 0.05 \
        --seed 42

Creates:
    dataroot/train/mha/{input,ground_truth,mask,breast_mask}/  (symlinks)
    dataroot/test/mha/{input,ground_truth,mask,breast_mask}/   (symlinks)
    dataroot/split_info.json  (metadata)
"""
import argparse
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path


def extract_case_id(filename):
    """Extract case_id from filename.
    
    Naming convention: {DATASET}_{ID}_p{phase}[_s{offset}].mha
    Examples:
        DUKE_001_p1.mha         → case_id=DUKE_001, dataset=DUKE
        DUKE_001_p1_s+1.mha     → case_id=DUKE_001, dataset=DUKE
        DUKE_001_p2_s-2.mha     → case_id=DUKE_001, dataset=DUKE
        ISPY2_123456_p1.mha     → case_id=ISPY2_123456, dataset=ISPY2
        ISPY2_123456_p3_s+1.mha → case_id=ISPY2_123456, dataset=ISPY2
    
    Returns (dataset_prefix, case_id, full_stem).
    case_id = everything before _p{digit} (patient identifier).
    """
    stem = Path(filename).stem
    
    # Match: {case_id}_p{phase}[_s{offset}]
    # case_id is everything before the first _p followed by a digit
    match = re.match(r'^(.+?)_p\d+', stem)
    if match:
        case_id = match.group(1)  # e.g. DUKE_001
    else:
        # Fallback: use whole stem
        case_id = stem
    
    # Extract dataset prefix (first part before _)
    parts = case_id.split('_')
    dataset_prefix = parts[0]  # DUKE, ISPY2, NACT, etc.
    
    return dataset_prefix, case_id, stem


def main():
    parser = argparse.ArgumentParser(description='Split data_multislice_v2 into train/test')
    parser.add_argument('--dataroot', required=True, help='Path to data_multislice_v2')
    parser.add_argument('--test_ratio', type=float, default=0.05, help='Fraction for test (default 5%%)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dry_run', action='store_true', help='Only print stats, do not create symlinks')
    args = parser.parse_args()

    random.seed(args.seed)
    dataroot = Path(args.dataroot)
    input_dir = dataroot / 'mha' / 'input'

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # ─── Step 1: Discover all files and group by case_id ───────────
    all_files = sorted(f for f in os.listdir(input_dir) if f.endswith('.mha'))
    print(f"Total files in input/: {len(all_files)}")

    # Group files by case_id, track dataset
    case_to_files = defaultdict(list)  # case_id -> [filename, ...]
    case_to_dataset = {}  # case_id -> dataset_prefix

    for f in all_files:
        dataset_prefix, case_id, _ = extract_case_id(f)
        case_to_files[case_id].append(f)
        case_to_dataset[case_id] = dataset_prefix

    # Group cases by dataset
    dataset_to_cases = defaultdict(list)
    for case_id, dataset in case_to_dataset.items():
        dataset_to_cases[dataset].append(case_id)

    print(f"\nDatasets found:")
    for ds in sorted(dataset_to_cases.keys()):
        cases = dataset_to_cases[ds]
        n_files = sum(len(case_to_files[c]) for c in cases)
        print(f"  {ds}: {len(cases)} cases, {n_files} files")

    # ─── Step 2: Stratified split per dataset ──────────────────────
    train_cases = set()
    test_cases = set()

    for ds in sorted(dataset_to_cases.keys()):
        cases = sorted(dataset_to_cases[ds])
        random.shuffle(cases)
        n_test = max(1, int(len(cases) * args.test_ratio))  # at least 1 per dataset
        test_subset = set(cases[:n_test])
        train_subset = set(cases[n_test:])
        test_cases.update(test_subset)
        train_cases.update(train_subset)

    # Verify no overlap
    overlap = train_cases & test_cases
    assert len(overlap) == 0, f"LEAK: {len(overlap)} cases in both train and test!"

    # Count files
    train_files = [f for c in train_cases for f in case_to_files[c]]
    test_files = [f for c in test_cases for f in case_to_files[c]]

    print(f"\nSplit result:")
    print(f"  Train: {len(train_cases)} cases, {len(train_files)} files")
    print(f"  Test:  {len(test_cases)} cases, {len(test_files)} files")
    print(f"  Test ratio: {len(test_cases)/(len(train_cases)+len(test_cases)):.1%}")
    print(f"\nPer-dataset breakdown:")
    for ds in sorted(dataset_to_cases.keys()):
        n_train = len([c for c in dataset_to_cases[ds] if c in train_cases])
        n_test = len([c for c in dataset_to_cases[ds] if c in test_cases])
        print(f"  {ds}: {n_train} train / {n_test} test ({n_test/(n_train+n_test):.1%})")

    if args.dry_run:
        print("\n[DRY RUN] No files created.")
        return

    # ─── Step 3: Filter test files to center slices only ─────────
    # Test set: only keep center slices (no _s{offset} suffix)
    # This means 1 slice per phase per case in test — clean evaluation
    # Train set: keep ALL slices (including neighbors for augmentation)
    test_files_center = [f for f in test_files if '_s+' not in f and '_s-' not in f]
    print(f"\n  Test (center slices only): {len(test_files_center)} files "
          f"(removed {len(test_files) - len(test_files_center)} neighbor slices)")

    # ─── Step 4: Create directory structure with symlinks ──────────
    # Detect which subdirs exist under mha/
    mha_dir = dataroot / 'mha'
    subdirs = [d.name for d in mha_dir.iterdir() if d.is_dir()]
    print(f"\nSubdirectories to split: {subdirs}")

    for split_name, file_list in [('train', train_files), ('test', test_files_center)]:
        for subdir in subdirs:
            src_dir = mha_dir / subdir
            dst_dir = dataroot / split_name / 'mha' / subdir
            dst_dir.mkdir(parents=True, exist_ok=True)

            created = 0
            for fname in file_list:
                src_path = src_dir / fname
                dst_path = dst_dir / fname
                if src_path.exists() and not dst_path.exists():
                    os.symlink(src_path.resolve(), dst_path)
                    created += 1
            print(f"  {split_name}/mha/{subdir}: {created} symlinks")

    # ─── Step 5: Save split metadata ──────────────────────────────
    split_info = {
        'seed': args.seed,
        'test_ratio': args.test_ratio,
        'total_cases': len(train_cases) + len(test_cases),
        'total_files': len(train_files) + len(test_files),
        'train_cases': len(train_cases),
        'train_files': len(train_files),
        'test_cases': len(test_cases),
        'test_files_all': len(test_files),
        'test_files_center_only': len(test_files_center),
        'test_case_ids': sorted(test_cases),
        'datasets': {
            ds: {
                'train_cases': len([c for c in dataset_to_cases[ds] if c in train_cases]),
                'test_cases': len([c for c in dataset_to_cases[ds] if c in test_cases]),
            }
            for ds in sorted(dataset_to_cases.keys())
        },
    }
    info_path = dataroot / 'split_info.json'
    with open(info_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    print(f"\nSplit info saved to: {info_path}")
    print("Done!")


if __name__ == '__main__':
    main()
