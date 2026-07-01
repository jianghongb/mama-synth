"""
split_data_multislice_v2.py — Split data_multislice_v2 into train/test by patient_id.

Uses the same patient-level train/test split as data_multislice.
Files are moved (not copied) into train/ and test/ subdirectories.

Usage:
    python split_data_multislice_v2.py \
        --data_dir /path/to/data_multislice_v2 \
        --splits_csv /path/to/train_test_split.csv
"""
import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path


def get_patient_id(filename: str) -> str:
    """Extract patient_id from filename like DUKE_001_p1_s+2.mha → DUKE_001."""
    stem = filename.replace(".mha", "")
    # Remove _s+N/_s-N suffix
    if "_s+" in stem or "_s-" in stem:
        stem = stem.rsplit("_s", 1)[0]
    # Remove _pN suffix
    if "_p" in stem:
        stem = stem.rsplit("_p", 1)[0]
    return stem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--splits_csv", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Load patient-level splits
    patient_splits = {}
    with open(args.splits_csv) as f:
        for row in csv.DictReader(f):
            patient_splits[row["patient_id"]] = row["split"]

    subdirs = ["input", "ground_truth", "mask"]
    
    # Create output dirs
    for split in ("train", "test"):
        for subdir in subdirs:
            (data_dir / split / "mha" / subdir).mkdir(parents=True, exist_ok=True)

    # Move files
    counts = defaultdict(int)
    skipped = 0
    
    src_base = data_dir / "mha"
    for subdir in subdirs:
        src_dir = src_base / subdir
        if not src_dir.exists():
            continue
        for f in sorted(src_dir.iterdir()):
            if not f.suffix == ".mha":
                continue
            patient_id = get_patient_id(f.name)
            split = patient_splits.get(patient_id)
            if split is None:
                skipped += 1
                continue
            dst = data_dir / split / "mha" / subdir / f.name
            shutil.move(str(f), str(dst))
            if subdir == "input":
                counts[split] += 1

    print(f"Train: {counts['train']} files")
    print(f"Test:  {counts['test']} files")
    print(f"Skipped (no split info): {skipped}")

    # Clean up empty source dirs
    shutil.rmtree(str(src_base), ignore_errors=True)
    print("Done.")


if __name__ == "__main__":
    main()
