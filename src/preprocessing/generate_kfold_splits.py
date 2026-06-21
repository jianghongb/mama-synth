"""
generate_kfold_splits.py — Generate stratified K-fold splits for cross-validation.

Stratified by data source (DUKE, ISPY2, ISPY1, NACT, YUNNAN, LABREAST, AMBL).
Only includes cases that have tumor mask (needed for Dice/HD95 evaluation).

Usage:
    python generate_kfold_splits.py \
        --data_dir /path/to/all_data/mha/input \
        --mask_dir /path/to/all_data/mha/mask \
        --output_csv kfold_splits.csv \
        --k 5
"""
import argparse
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np


def get_source(name: str) -> str:
    if name.startswith('DUKE'): return 'DUKE'
    elif name.startswith('ISPY2'): return 'ISPY2'
    elif name.startswith('ISPY1'): return 'ISPY1'
    elif name.startswith('NACT'): return 'NACT'
    elif name.startswith('YUNNAN'): return 'YUNNAN'
    elif name.startswith('LABREAST'): return 'LABREAST'
    elif name.startswith('AMBL'): return 'AMBL'
    return 'OTHER'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Directory with all input .mha files")
    parser.add_argument("--mask_dir", default="", help="Mask dir — only include cases with mask")
    parser.add_argument("--output_csv", default="kfold_splits.csv")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    mask_dir = Path(args.mask_dir) if args.mask_dir else None

    # Get all case IDs
    all_files = sorted(data_dir.glob("*.mha"))
    case_ids = [f.stem for f in all_files]

    # Filter: only cases with tumor mask (if mask_dir given)
    if mask_dir and mask_dir.exists():
        case_ids = [c for c in case_ids if (mask_dir / f"{c}.mha").exists()]
        print(f"Filtered to {len(case_ids)} cases with tumor mask")

    # Group by source
    source_groups = defaultdict(list)
    for c in case_ids:
        source_groups[get_source(c)].append(c)

    print(f"Total: {len(case_ids)} cases")
    for src, cases in sorted(source_groups.items()):
        print(f"  {src}: {len(cases)}")

    # Stratified K-fold: shuffle within each source, assign folds
    rng = np.random.default_rng(args.seed)
    fold_assignments = {}

    for src, cases in source_groups.items():
        shuffled = cases.copy()
        rng.shuffle(shuffled)
        for i, c in enumerate(shuffled):
            fold_assignments[c] = i % args.k

    # Write CSV
    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['case_id', 'source', 'fold'])
        for c in sorted(case_ids):
            writer.writerow([c, get_source(c), fold_assignments[c]])

    # Print summary
    print(f"\nFold distribution:")
    for fold in range(args.k):
        fold_cases = [c for c, f in fold_assignments.items() if f == fold]
        fold_sources = defaultdict(int)
        for c in fold_cases:
            fold_sources[get_source(c)] += 1
        src_str = ", ".join(f"{s}:{n}" for s, n in sorted(fold_sources.items()))
        print(f"  Fold {fold}: {len(fold_cases)} cases ({src_str})")

    print(f"\nSaved: {args.output_csv}")
    print(f"\nUsage in training:")
    print(f"  For fold i: train = cases where fold != i, test = cases where fold == i")


if __name__ == "__main__":
    main()
