"""Survey the raw DCE volumes to decide whether in-domain pretraining is affordable.

Masked-reconstruction pretraining would reuse the existing GlobalGenerator
backbone, so no architecture changes are needed and the GAN-versus-diffusion
comparison stays intact. What is not yet known is whether the data can be staged
at all, and that is decided by two hard limits rather than by GPU time:

  inode quota -- the project sits at 2,887,795 of a 3,000,000 hard limit, leaving
    112,205 free. Writing one file per slice would exhaust it and lock the whole
    project for every user. Everything must therefore go into a single packed
    array plus one index file, two inodes in total. This script reports the
    inode cost of each option so that constraint stays visible.

  disk -- the raw volumes are 115 GB. A packed copy at full resolution in float16
    could exceed that, so the script prices several resolutions, dtypes and slice
    strides instead of assuming one.

Counts come from file headers for every volume, which is cheap and exact.
Foreground statistics need pixel data, so they come from a patient sample and are
extrapolated; the script prints the sample size so the extrapolation can be
judged.

What this script deliberately does not do. The downstream model normalises each
image by the mean and standard deviation inside the breast region and zeroes the
background (inference.py:152-159). Reproducing that needs the breast_seg_930
nnU-Net, which is a GPU job. Pretraining has to use the same normalisation or
the learned features will not transfer, so this script measures a cheap
intensity-based proxy and reports how far it sits from a real breast mask on the
test slices where ground-truth masks already exist. That number decides whether
the proxy is usable or whether a GPU mask-generation pass has to come first.

Run:
    python scripts/survey_pretrain_data.py
    python scripts/survey_pretrain_data.py --sample 80
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk

IMAGES = Path("/Users/ehogjig/git/kth/images")
TEST_MHA = Path("/Users/ehogjig/git/kth/data_multislice_v3/test/mha")
REPO = Path(__file__).resolve().parent.parent

INODE_FREE = 112_205          # project headroom, 2,887,795 of 3,000,000


def patients():
    return sorted(p for p in os.listdir(IMAGES)
                  if (IMAGES / p).is_dir() and not p.startswith("."))


def header_shape(path):
    """Slice count and in-plane size without reading pixel data."""
    r = sitk.ImageFileReader()
    r.SetFileName(str(path))
    r.ReadImageInformation()
    sx, sy, sz = r.GetSize()
    return sz, (sy, sx)


def count_all():
    """Exact slice totals per cohort, from headers only."""
    per_cohort = defaultdict(lambda: dict(patients=0, volumes=0, slices=0))
    sizes = defaultdict(int)
    for pid in patients():
        vols = sorted((IMAGES / pid).glob("*.nii.gz"))
        if not vols:
            continue
        cohort = pid.split("_")[0]
        per_cohort[cohort]["patients"] += 1
        for v in vols:
            try:
                nz, inplane = header_shape(v)
            except Exception:
                continue
            per_cohort[cohort]["volumes"] += 1
            per_cohort[cohort]["slices"] += nz
            sizes[inplane] += 1
    return per_cohort, sizes


def foreground_profile(sample, stride):
    """Fraction of slices holding enough tissue to be worth training on.

    The filter is intensity-based because no breast mask exists for these
    volumes yet. It keeps a slice when the body occupies a plausible share of
    the frame: Otsu-style splitting is overkill here, so the threshold is the
    volume's own 75th percentile and the slice is kept when more than 8 % of
    pixels exceed it. Pure-air slices at either end of the stack fail this,
    which is the main thing that needs removing.
    """
    kept = total = 0
    areas = []
    for pid in sample:
        for v in sorted((IMAGES / pid).glob("*.nii.gz")):
            try:
                a = sitk.GetArrayFromImage(sitk.ReadImage(str(v))).astype(np.float32)
            except Exception:
                continue
            thr = float(np.percentile(a, 75))
            frac = (a > thr).mean(axis=(1, 2))
            sel = frac[::stride]
            total += sel.size
            kept += int((sel > 0.08).sum())
            areas.extend(frac[frac > 0.08].tolist())
    return kept, total, float(np.mean(areas)) if areas else 0.0


def proxy_against_real_masks(limit=40):
    """How close is the intensity proxy to a real breast mask?

    The test split has breast masks produced by breast_seg_930, so the proxy can
    be scored directly where ground truth happens to exist. A high Dice would
    mean pretraining can be staged on CPU with the proxy; a low one means a GPU
    mask pass has to run first, because the downstream per-image normalisation
    is defined over the breast region and a mismatch there would undo the
    transfer.
    """
    bm_dir = TEST_MHA / "breast_mask"
    in_dir = TEST_MHA / "input"
    if not bm_dir.is_dir():
        return None
    dices, cover = [], []
    for f in sorted(bm_dir.glob("*.mha"))[:limit]:
        src = in_dir / f.name
        if not src.exists():
            continue
        img = sitk.GetArrayFromImage(sitk.ReadImage(str(src))).astype(np.float32)
        ref = sitk.GetArrayFromImage(sitk.ReadImage(str(f))).astype(np.float32)
        img = img[0] if img.ndim == 3 else img
        ref = (ref[0] if ref.ndim == 3 else ref) > 0.5
        # the slices are already z-scored, so use a fixed quantile of the slice
        prox = img > float(np.percentile(img, 75))
        inter = float((prox & ref).sum())
        d = 2 * inter / max(float(prox.sum() + ref.sum()), 1.0)
        dices.append(d)
        cover.append(inter / max(float(ref.sum()), 1.0))
    if not dices:
        return None
    return dict(dice_mean=float(np.mean(dices)), dice_median=float(np.median(dices)),
                recall_mean=float(np.mean(cover)), n=len(dices))


def price(n_slices, res, dtype_bytes):
    return n_slices * res * res * dtype_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=40,
                    help="patients to read pixel data for")
    ap.add_argument("--stride", type=int, default=3,
                    help="slice stride; neighbours are near-duplicates")
    args = ap.parse_args()

    print("=== 1. exact slice counts (headers only) ===\n")
    per_cohort, sizes = count_all()
    print(f"  {'cohort':<12}{'patients':>10}{'volumes':>9}{'slices':>10}")
    tot_p = tot_v = tot_s = 0
    for c in sorted(per_cohort):
        d = per_cohort[c]
        print(f"  {c:<12}{d['patients']:>10}{d['volumes']:>9}{d['slices']:>10,}")
        tot_p += d["patients"]; tot_v += d["volumes"]; tot_s += d["slices"]
    print(f"  {'TOTAL':<12}{tot_p:>10}{tot_v:>9}{tot_s:>10,}")
    print(f"\n  in-plane sizes: " +
          ", ".join(f"{k[0]}x{k[1]} ({v} vols)"
                    for k, v in sorted(sizes.items(), key=lambda x: -x[1])[:4]))

    print(f"\n=== 2. usable fraction (sample of {args.sample} patients, "
          f"stride {args.stride}) ===\n")
    sample = patients()[:args.sample]
    kept, total, area = foreground_profile(sample, args.stride)
    rate = kept / max(total, 1)
    est = int(tot_s / args.stride * rate)
    print(f"  sampled slices after stride : {total:,}")
    print(f"  passed the tissue filter    : {kept:,}  ({rate:.1%})")
    print(f"  mean tissue area when kept  : {area:.1%} of the frame")
    print(f"  extrapolated usable total   : {est:,} slices")
    print(f"\n  Adjacent slices are near-duplicates, so stride {args.stride} costs")
    print(f"  little information while cutting cost {args.stride}x. The effective")
    print(f"  sample size is lower than the count regardless, since slices from")
    print(f"  one patient and one phase are strongly correlated.")

    print("\n=== 3. staging cost, and the inode limit ===\n")
    print(f"  {'layout':<34}{'files':>9}{'inodes ok':>11}{'size':>10}")
    one_per_slice = est
    print(f"  {'one .mha per slice':<34}{one_per_slice:>9,}"
          f"{'NO':>11}{'-':>10}")
    for res in (256, 384, 512):
        for dt, nb in (("uint8", 1), ("float16", 2)):
            gb = price(est, res, nb) / 1e9
            print(f"  {'packed memmap ' + f'{res}^2 {dt}':<34}{2:>9}"
                  f"{'yes':>11}{gb:>9.0f}G")
    print(f"\n  inode headroom is {INODE_FREE:,}. One file per slice needs")
    print(f"  {one_per_slice:,}, so that option is not merely wasteful, it would")
    print(f"  hit the hard limit and lock the project for everyone. A packed")
    print(f"  memmap plus one JSON index costs 2 inodes at any scale.")

    print("\n=== 4. can the breast mask be skipped? ===\n")
    pr = proxy_against_real_masks()
    if pr:
        print(f"  intensity proxy against breast_seg_930, n={pr['n']}")
        print(f"    Dice   mean {pr['dice_mean']:.3f}   median {pr['dice_median']:.3f}")
        print(f"    recall mean {pr['recall_mean']:.3f}")
        if pr["dice_mean"] > 0.85:
            print("  -> close enough to stage pretraining on CPU with the proxy.")
        else:
            print("  -> too far apart to substitute. Downstream normalisation is")
            print("     defined over the breast region (inference.py:152-159), so")
            print("     a proxy this different would normalise pretraining inputs")
            print("     on a different basis than finetuning, which is the kind of")
            print("     mismatch that quietly removes any benefit from transfer.")
            print("     A GPU pass with breast_seg_930 has to come first.")
    else:
        print("  no breast masks available to compare against")

    out = REPO / "scripts" / "pretrain_data_survey.json"
    out.write_text(json.dumps(dict(
        cohorts={k: dict(v) for k, v in per_cohort.items()},
        total_slices=tot_s, stride=args.stride, sample=args.sample,
        keep_rate=rate, usable_estimate=est, proxy=pr), indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
