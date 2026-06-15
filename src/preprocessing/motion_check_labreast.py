"""Motion check for LA-Breast DCE-MRI (2D TIFF slices).

Uses phase correlation to detect displacement between d0 and d1-d5.
Outputs a summary of cases with motion and optionally removes them from data_split_v4.

Usage:
    python motion_check_labreast.py \
        --src "/Users/ehogjig/Downloads/LA-Breast DCE-MRI Dataset/breast_data" \
        --split train --threshold 2
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


def phase_corr_shift(ref, mov):
    """Compute pixel displacement between ref and mov using phase correlation."""
    cross = np.fft.fft2(ref) * np.conj(np.fft.fft2(mov))
    cross /= np.abs(cross) + 1e-10
    cc = np.fft.ifft2(cross).real
    idx = np.unravel_index(np.argmax(cc), cc.shape)
    dy = idx[0] if idx[0] <= ref.shape[0] // 2 else idx[0] - ref.shape[0]
    dx = idx[1] if idx[1] <= ref.shape[1] // 2 else idx[1] - ref.shape[1]
    return dy, dx


def check_case(src, split, row):
    """Check motion for one case across d0-d5. Returns max displacement."""
    d0_path = src / "d0" / split / row["A2_d0"]
    if not d0_path.exists():
        return 0

    ref = np.array(Image.open(d0_path)).astype(np.float32)
    max_disp = 0

    for i in range(1, 6):
        col = f"A2_d{i}"
        if col not in row or not row[col]:
            continue
        di_path = src / f"d{i}" / split / row[col]
        if not di_path.exists():
            continue
        mov = np.array(Image.open(di_path)).astype(np.float32)
        dy, dx = phase_corr_shift(ref, mov)
        max_disp = max(max_disp, abs(dy), abs(dx))

    return max_disp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--threshold", type=int, default=2)
    args = parser.parse_args()

    src = Path(args.src)
    with open(src / "metadata" / f"{args.split}.csv") as f:
        rows = list(csv.DictReader(f))

    print(f"Checking {len(rows)} slices ({args.split})...")

    flagged = []
    for i, r in enumerate(rows):
        d = check_case(src, args.split, r)
        case_id = f"LABREAST_{r['patient'].replace('Breast_Mri_', '')}_{r['ROI']}_{Path(r['A2_d0']).stem.split('_')[-1]}"
        if d > args.threshold:
            flagged.append((case_id, d))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(rows)}, flagged so far: {len(flagged)}")

    print(f"\nDone. Flagged {len(flagged)}/{len(rows)} cases with displacement > {args.threshold}px")
    if flagged:
        print("\nFlagged cases:")
        for case_id, d in sorted(flagged, key=lambda x: -x[1])[:20]:
            print(f"  {case_id}: {d}px")

    # Save full list
    out_path = Path(f"motion_labreast_{args.split}.txt")
    with open(out_path, "w") as f:
        for case_id, d in sorted(flagged, key=lambda x: -x[1]):
            f.write(f"{case_id}: {d}\n")
    print(f"\nFull list saved to {out_path}")


if __name__ == "__main__":
    main()
