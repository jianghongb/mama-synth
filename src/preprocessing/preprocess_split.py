#!/usr/bin/env python3
"""
preprocess_split.py — 3D→2D preprocessing with train/test split.

Wraps preprocess.py logic: converts 3D DCE-MRI volumes to 2D slices,
then saves results into train/ or test/ based on train_test_splits.csv.

Output structure:
    output_dir/
        train/
            mha/input/       # pre-contrast slices
            mha/ground_truth/# peak-enhancement slices
            mha/mask/        # tumour masks
        test/
            mha/input/
            mha/ground_truth/
            mha/mask/

Usage:
    python src/preprocessing/preprocess_split.py \
        --image_dir ./images \
        --seg_dir ./segmentations \
        --output_dir ./data_split \
        --splits_csv ./train_test_splits.csv \
        --global_stats ./src/preprocessing/training_pre_stats.json \
        --skip_ambiguous_shapes
"""
import argparse
import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from preprocess import Preprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_splits(csv_path: Path) -> dict[str, str]:
    """Return {patient_id: 'train' or 'test'}."""
    mapping = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = row["train_split"].strip()
            s = row["test_split"].strip()
            if t:
                mapping[t] = "train"
            if s:
                mapping[s] = "test"
    return mapping


def main():
    parser = argparse.ArgumentParser(description="3D→2D preprocessing with train/test split")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--seg_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--splits_csv", required=True)
    parser.add_argument("--global_stats", required=True)
    parser.add_argument("--skip_ambiguous_shapes", action="store_true", default=False)
    args = parser.parse_args()

    splits = load_splits(Path(args.splits_csv))
    logger.info(f"Loaded splits: {sum(1 for v in splits.values() if v=='train')} train, "
                f"{sum(1 for v in splits.values() if v=='test')} test")

    output_dir = Path(args.output_dir)

    # Create separate preprocessors for train and test
    for split in ("train", "test"):
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

    # Use a single preprocessor but redirect output per patient
    # We'll process all patients, then move files to correct split dirs
    # More efficient: process per split

    for split in ("train", "test"):
        split_patients = [pid for pid, s in splits.items() if s == split]
        if not split_patients:
            continue

        split_dir = output_dir / split
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {split} split: {len(split_patients)} patients")
        logger.info(f"{'='*60}")

        preprocessor = Preprocessor(
            image_dir=args.image_dir,
            segmentation_dir=args.seg_dir,
            output_dir=str(split_dir),
            csv_output_path=str(split_dir / "report.csv"),
            global_stats_path=args.global_stats,
            skip_ambiguous_shapes=args.skip_ambiguous_shapes,
        )

        # Filter to only process patients in this split
        all_phases = preprocessor.get_patient_phases()
        filtered = {pid: phases for pid, phases in all_phases.items() if pid in split_patients}

        found = len(filtered)
        missing = len(split_patients) - found
        if missing > 0:
            logger.warning(f"  {missing} patients from {split} split not found in image_dir")

        # Monkey-patch get_patient_phases to return only this split's patients
        preprocessor.get_patient_phases = lambda f=filtered: f

        results_df = preprocessor.process()
        if not results_df.empty:
            preprocessor.save_report(results_df)

        logger.info(f"  {split}: processed {len(results_df)} patients → {split_dir}")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
