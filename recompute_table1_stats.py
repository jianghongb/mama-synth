#!/usr/bin/env python3
"""Recompute the Table 1 intensity statistics under four conventions.

Table 1 of the paper reports, per dataset, "Mean (avg +/- std)" and
"Std (avg +/- std)" with the caption claiming they are computed over breast
foreground pixels in global z-score space. The aggregation dimension is not
stated, and the numbers could not be reproduced from the test split locally,
so this script computes all four plausible conventions and prints them next
to the published values. Whichever column matches is the convention the
caption should state.

Conventions:
    FG / slice    per-slice mean and std over breast-foreground pixels,
                  then avg +/- std across slices
    FG / patient  as above, but per-slice values are first averaged within
                  each patient, then avg +/- std across patients
    ALL / slice   same as FG/slice but over every pixel (no mask)
    ALL / patient same as FG/patient but over every pixel (no mask)

Patient id is the filename with any trailing _s+N / _s-N slice suffix removed.

Usage (on Berzelius):
    python recompute_table1_stats.py \
        --root /proj/berzbiomedicalimagingkth/users/x_honji/data_multislice_v3/train/mha
    # add --limit 200 for a quick sample per dataset
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk

# Values as printed in Table 1 of the paper, for side-by-side comparison.
PUBLISHED = {
    "ISPY2":     ("0.29 +/- 0.56", "0.97 +/- 0.72"),
    "AMBL":      ("-0.30 +/- 0.10", "0.30 +/- 0.16"),
    "LAB":       ("-0.40 +/- 0.01", "0.12 +/- 0.01"),  # LA-Breast
    "DUKE":      ("0.06 +/- 0.48", "0.87 +/- 0.78"),
    "YUNNAN":    ("-0.33 +/- 0.05", "0.23 +/- 0.07"),
}

SLICE_SUFFIX = re.compile(r"_s[+-]\d+$")


def load(path: Path) -> np.ndarray:
    """Read a 2D .mha slice as float32, squeezing any singleton axis."""
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32).squeeze()


def patient_id(stem: str) -> str:
    """Strip the trailing slice-offset suffix to get the patient identifier."""
    return SLICE_SUFFIX.sub("", stem)


def fmt(values: np.ndarray) -> str:
    """Format an array as 'avg +/- std' with two decimals."""
    if values.size == 0:
        return "n/a"
    return f"{values.mean():.2f} +/- {values.std():.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="Path to data_multislice_v3/train/mha")
    ap.add_argument("--limit", type=int, default=0,
                    help="Sample at most N slices per dataset (0 = all)")
    args = ap.parse_args()

    root = Path(args.root)
    in_dir, mask_dir = root / "input", root / "breast_mask"
    for d in (in_dir, mask_dir):
        if not d.is_dir():
            raise SystemExit(f"missing directory: {d}")

    by_dataset = defaultdict(list)
    for f in sorted(in_dir.glob("*.mha")):
        by_dataset[f.stem.split("_")[0]].append(f)

    # stats[dataset][convention] -> list of (mean, std) per slice
    stats = defaultdict(lambda: defaultdict(list))
    # patient grouping for the patient-level conventions
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for ds, files in sorted(by_dataset.items()):
        if args.limit:
            files = files[: args.limit]
        skipped = 0
        for f in files:
            mask_path = mask_dir / f.name
            if not mask_path.exists():
                skipped += 1
                continue
            x = load(f)
            m = load(mask_path) > 0.5
            fg = x[m]
            if fg.size < 100:
                skipped += 1
                continue
            pid = patient_id(f.stem)
            for conv, (mu, sd) in (
                ("FG", (float(fg.mean()), float(fg.std()))),
                ("ALL", (float(x.mean()), float(x.std()))),
            ):
                stats[ds][conv].append((mu, sd))
                grouped[ds][conv][pid].append((mu, sd))
        print(f"{ds}: {len(files) - skipped} slices used"
              f"{f', {skipped} skipped' if skipped else ''}, "
              f"{len(grouped[ds]['FG'])} patients")

    header = (f"\n{'dataset':<11}{'convention':<15}"
              f"{'Mean (avg+/-std)':>20}{'Std (avg+/-std)':>20}"
              f"{'published Mean':>18}{'published Std':>18}")
    print(header)
    print("-" * len(header))

    for ds in sorted(stats):
        pub_mean, pub_std = PUBLISHED.get(ds, ("?", "?"))
        for conv in ("FG", "ALL"):
            arr = np.array(stats[ds][conv])
            if arr.size == 0:
                continue
            # slice-level aggregation
            print(f"{ds:<11}{conv + ' / slice':<15}"
                  f"{fmt(arr[:, 0]):>20}{fmt(arr[:, 1]):>20}"
                  f"{pub_mean:>18}{pub_std:>18}")
            # patient-level aggregation
            per_pat = grouped[ds][conv]
            pm = np.array([np.mean([v[0] for v in vs]) for vs in per_pat.values()])
            ps = np.array([np.mean([v[1] for v in vs]) for vs in per_pat.values()])
            print(f"{'':<11}{conv + ' / patient':<15}"
                  f"{fmt(pm):>20}{fmt(ps):>20}"
                  f"{'':>18}{'':>18}")
        print()

    print("Whichever row reproduces the published pair is the convention the "
          "Table 1 caption should state.")


if __name__ == "__main__":
    main()
