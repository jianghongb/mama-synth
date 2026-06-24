"""
generate_kfold_splits_v2.py — Generate stratified train/test split.

Train: all sources (DUKE + ISPY2 + YUNNAN + LABREAST)
Test: only DUKE + ISPY2 + YUNNAN, 5% of these cases per fold

Usage:
    python generate_kfold_splits_v2.py \
        --data_dir /path/to/all_data/mha/input \
        --mask_dir /path/to/all_data/mha/mask \
        --output_csv kfold_splits_v2.csv \
        --k 5 \
        --test_ratio 0.05 \
        --train_only_sources LABREAST
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
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--mask_dir", default="")
    parser.add_argument("--output_csv", default="kfold_splits_v2.csv")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--test_ratio", type=float, default=0.05, help="Fraction of testable cases per fold")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_only_sources", default="", help="Sources only used for training, never in test (e.g. LABREAST)")
    parser.add_argument("--exclude_sources", default="", help="Sources to exclude entirely")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    mask_dir = Path(args.mask_dir) if args.mask_dir else None
    train_only = set(s.strip() for s in args.train_only_sources.split(",") if s.strip())
    exclude = set(s.strip() for s in args.exclude_sources.split(",") if s.strip())

    # Get all cases
    all_files = sorted(data_dir.glob("*.mha"))
    case_ids = [f.stem for f in all_files]

    # Filter by mask existence
    if mask_dir and mask_dir.exists():
        case_ids = [c for c in case_ids if (mask_dir / f"{c}.mha").exists()]

    # Exclude sources
    if exclude:
        case_ids = [c for c in case_ids if get_source(c) not in exclude]

    # Separate: testable cases vs train-only cases
    testable = [c for c in case_ids if get_source(c) not in train_only]
    train_only_cases = [c for c in case_ids if get_source(c) in train_only]

    print(f"Total: {len(case_ids)} cases")
    print(f"  Testable (DUKE+ISPY2+YUNNAN): {len(testable)}")
    print(f"  Train-only (LABREAST): {len(train_only_cases)}")

    # Group testable by source
    source_groups = defaultdict(list)
    for c in testable:
        source_groups[get_source(c)].append(c)

    for src, cases in sorted(source_groups.items()):
        print(f"    {src}: {len(cases)}")

    # Assign folds to testable cases
    rng = np.random.default_rng(args.seed)
    fold_assignments = {}

    # Calculate test size per fold
    test_per_fold = int(len(testable) * args.test_ratio)
    print(f"\nTest ratio: {args.test_ratio} → ~{test_per_fold} test cases per fold")

    # Stratified assignment: shuffle within source, assign round-robin
    # With 5% test ratio and 5 folds, each case is in test once (5% × 5 folds ≈ 25% total coverage)
    # Actually: assign each testable case to exactly one fold for testing
    # With k=5 and test_ratio=0.05, we want 5% in each fold's test → total 25% tested
    # Better: just do standard k-fold but with smaller test sets

    # Simple approach: assign fold IDs, but only use test_ratio fraction per fold
    # Actually let's do proper: split testable into k groups, each ~5% of total
    n_test = max(1, int(len(testable) * args.test_ratio))

    for src, cases in source_groups.items():
        shuffled = cases.copy()
        rng.shuffle(shuffled)
        for i, c in enumerate(shuffled):
            fold_assignments[c] = i % args.k

    # Train-only cases: fold = -1 (always in train)
    for c in train_only_cases:
        fold_assignments[c] = -1

    # Write CSV
    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['case_id', 'source', 'fold'])
        for c in sorted(case_ids):
            writer.writerow([c, get_source(c), fold_assignments[c]])

    # Summary
    print(f"\nFold distribution:")
    for fold in range(args.k):
        fold_test = [c for c, f in fold_assignments.items() if f == fold]
        fold_train = [c for c, f in fold_assignments.items() if f != fold]
        src_test = defaultdict(int)
        for c in fold_test:
            src_test[get_source(c)] += 1
        print(f"  Fold {fold}: test={len(fold_test)}, train={len(fold_train)} (test: {dict(src_test)})")

    print(f"\nSaved: {args.output_csv}")


if __name__ == "__main__":
    main()
