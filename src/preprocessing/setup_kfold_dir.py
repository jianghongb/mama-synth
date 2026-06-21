"""
setup_kfold_dir.py — Create train/test directories for a specific fold using symlinks.

Usage:
    python setup_kfold_dir.py \
        --splits_csv kfold_splits.csv \
        --data_dir /path/to/all_data/mha \
        --output_dir /path/to/kfold/fold_0 \
        --fold 0
"""
import argparse
import csv
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_csv", required=True)
    parser.add_argument("--data_dir", required=True, help="Directory with input/, ground_truth/, mask/, breast_mask/")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()

    # Load splits
    fold_map = {}
    with open(args.splits_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            fold_map[row['case_id']] = int(row['fold'])

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    subdirs = ['input', 'ground_truth', 'mask', 'breast_mask']

    for split in ['train', 'test']:
        for sub in subdirs:
            src_dir = data_dir / sub
            if not src_dir.exists():
                continue
            dst_dir = output_dir / split / 'mha' / sub
            dst_dir.mkdir(parents=True, exist_ok=True)

            for case_id, fold in fold_map.items():
                src_file = src_dir / f"{case_id}.mha"
                if not src_file.exists():
                    continue

                # test = current fold, train = all other folds
                is_test = (fold == args.fold)
                if (split == 'test' and is_test) or (split == 'train' and not is_test):
                    dst_file = dst_dir / f"{case_id}.mha"
                    if not dst_file.exists():
                        os.symlink(src_file, dst_file)

    # Print summary
    train_count = len(list((output_dir / 'train' / 'mha' / 'input').glob('*.mha')))
    test_count = len(list((output_dir / 'test' / 'mha' / 'input').glob('*.mha')))
    print(f"Fold {args.fold}: train={train_count}, test={test_count}")


if __name__ == "__main__":
    main()
