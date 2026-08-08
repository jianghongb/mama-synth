#!/usr/bin/env python3
"""Recompute the Table 1 intensity statistics under four conventions.

Table 1 of the paper reports, per dataset, "Mean (avg +/- std)" and
"Std (avg +/- std)" with the caption claiming they are computed over breast
foreground pixels in global z-score space. The aggregation dimension is not
stated, and on the test split the published values are reproduced by an
all-pixels convention rather than a foreground one, so this script computes
all four plausible conventions and prints them next to the published values.
Whichever row matches is the convention the caption should state.

Conventions:
    FG / slice    per-slice mean and std over breast-foreground pixels,
                  then avg +/- std across slices
    FG / patient  as above, but per-slice values are first averaged within
                  each patient, then avg +/- std across patients
    ALL / slice   same as FG/slice but over every pixel (no mask)
    ALL / patient same as FG/patient but over every pixel (no mask)

Patient id is the filename with any trailing _s+N / _s-N slice suffix removed.

Results are printed per dataset as soon as that dataset finishes, smallest
dataset first, so a partial run is still informative. Output is unbuffered.

Usage:
    python recompute_table1_stats.py --root <path>/data_multislice_v3/train/mha
    python recompute_table1_stats.py --root <path> --limit 400   # quick sample
"""
import argparse
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk

# Values as printed in Table 1 of the paper, for side-by-side comparison.
PUBLISHED = {
    "ISPY2":  ("0.29 +/- 0.56", "0.97 +/- 0.72"),
    "AMBL":   ("-0.30 +/- 0.10", "0.30 +/- 0.16"),
    "LAB":    ("-0.40 +/- 0.01", "0.12 +/- 0.01"),  # LA-Breast
    "DUKE":   ("0.06 +/- 0.48", "0.87 +/- 0.78"),
    "YUNNAN": ("-0.33 +/- 0.05", "0.23 +/- 0.07"),
}

SLICE_SUFFIX = re.compile(r"_s[+-]\d+$")


def say(msg: str = "") -> None:
    """Print and flush immediately, so SLURM logs stay useful if we are killed."""
    print(msg, flush=True)


def one_slice(args):
    """Compute (patient_id, fg_mean, fg_std, all_mean, all_std) for one slice.

    Returns None if the mask is missing or the foreground is too small.
    """
    img_path, mask_path = args
    try:
        x = sitk.GetArrayFromImage(sitk.ReadImage(str(img_path))).astype(np.float32).squeeze()
        if not mask_path.exists():
            return None
        m = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path))).astype(np.float32).squeeze() > 0.5
        fg = x[m]
        if fg.size < 100:
            return None
        return (SLICE_SUFFIX.sub("", img_path.stem),
                float(fg.mean()), float(fg.std()),
                float(x.mean()), float(x.std()))
    except Exception as exc:  # noqa: BLE001 - report and skip unreadable slices
        say(f"    warning: {img_path.name}: {exc}")
        return None


def fmt(values: np.ndarray) -> str:
    """Format an array as 'avg +/- std' with two decimals."""
    return "n/a" if values.size == 0 else f"{values.mean():.2f} +/- {values.std():.2f}"


def report(dataset: str, rows: list) -> None:
    """Print the four conventions for one dataset against the published values."""
    pub_mean, pub_std = PUBLISHED.get(dataset, ("?", "?"))
    fg = np.array([(r[1], r[2]) for r in rows])
    al = np.array([(r[3], r[4]) for r in rows])

    grouped = defaultdict(list)
    for r in rows:
        grouped[r[0]].append(r)

    for label, arr, idx in (("FG", fg, (1, 2)), ("ALL", al, (3, 4))):
        say(f"{dataset:<11}{label + ' / slice':<15}"
            f"{fmt(arr[:, 0]):>20}{fmt(arr[:, 1]):>20}"
            f"{pub_mean:>18}{pub_std:>18}")
        pm = np.array([np.mean([r[idx[0]] for r in v]) for v in grouped.values()])
        ps = np.array([np.mean([r[idx[1]] for r in v]) for v in grouped.values()])
        say(f"{'':<11}{label + ' / patient':<15}"
            f"{fmt(pm):>20}{fmt(ps):>20}")
    say()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to .../mha (with input/ and breast_mask/)")
    ap.add_argument("--limit", type=int, default=0, help="Sample at most N slices per dataset (0 = all)")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    root = Path(args.root)
    in_dir, mask_dir = root / "input", root / "breast_mask"
    for d in (in_dir, mask_dir):
        if not d.is_dir():
            raise SystemExit(f"missing directory: {d}")

    by_dataset = defaultdict(list)
    for f in sorted(in_dir.glob("*.mha")):
        by_dataset[f.stem.split("_")[0]].append(f)

    # Smallest dataset first: a partial run then still yields complete rows.
    order = sorted(by_dataset, key=lambda d: len(by_dataset[d]))
    say(f"datasets: {', '.join(f'{d}({len(by_dataset[d])})' for d in order)}")
    say(f"workers: {args.workers}   limit/dataset: {args.limit or 'all'}")
    say()
    header = (f"{'dataset':<11}{'convention':<15}"
              f"{'Mean (avg+/-std)':>20}{'Std (avg+/-std)':>20}"
              f"{'published Mean':>18}{'published Std':>18}")
    say(header)
    say("-" * len(header))

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for dataset in order:
            files = by_dataset[dataset]
            if args.limit:
                files = files[: args.limit]
            tasks = [(f, mask_dir / f.name) for f in files]
            rows = [r for r in pool.map(one_slice, tasks, chunksize=8) if r is not None]
            if not rows:
                say(f"{dataset:<11}no usable slices")
                continue
            say(f"# {dataset}: {len(rows)}/{len(files)} slices, "
                f"{len({r[0] for r in rows})} patients")
            report(dataset, rows)

    say("The row reproducing the published pair is the convention the Table 1 "
        "caption should state.")


if __name__ == "__main__":
    sys.exit(main())
