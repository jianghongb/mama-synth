"""Build the volume manifest for in-domain pretraining.

Pretraining will reuse the existing GlobalGenerator backbone with a masked
reconstruction objective, so this stage only has to decide which volumes may be
read. Two decisions are made here and both are recorded in the output so a later
run can be audited rather than trusted.

THE EXCLUSION, WHICH IS THE POINT OF THIS SCRIPT

The 60 patients the thesis evaluates on must not be seen during pretraining,
otherwise every number in the results chapter becomes meaningless. Getting that
exclusion right is not straightforward, because the repository holds two
different splits:

  train_test_splits.csv        1200 train / 306 test, the challenge's own split
  data_multislice_v3/{train,test}   1179 train / 60 test, the split actually used

These do not agree. 52 of the 60 evaluated patients appear in the CSV's
*train_split* column, and the five YUNNAN ones appear in neither column. Taking
the CSV's test_split as the exclusion list, which is the obvious reading, would
therefore pull 52 evaluated patients into pretraining.

So the exclusion is derived from the filenames in data_multislice_v3/test/mha/
input, the split that governs the trained models, and the CSV is used only to
cross-check. The script refuses to write a manifest if the exclusion list does
not come out at the expected size, so the check cannot be skipped by accident.

THE RESOLUTION DECISION

In-plane sizes are mixed: 4583 volumes at 256x256, 837 at 384, 802 at 512, 420
at 448. Keeping only volumes at 448 or above would avoid any resampling, but it
costs almost the whole ISPY2 cohort, since 979 of its 980 patients are 256x256.
ISPY2 supplies 44 of the 60 evaluated patients, so that filter would leave
pretraining with almost no data from the cohort the model is mostly judged on,
and a centre mismatch of that size cannot be repaired by finetuning.

Pretraining therefore runs at 256x256, the native size of most of the data, and
larger volumes are downsampled to match. Downsampling discards high-frequency
detail but does not invent any, unlike upsampling 256 to 512, which would teach
the backbone interpolated texture that real 512 data does not have.

The cost is that pretraining and finetuning then operate at different scales, so
a convolution covers a different physical extent in each. For a fully
convolutional backbone that is the milder problem: convolution is translation
equivariant by construction and scale is something the network learns, so the
200 finetuning epochs have room to re-adapt. Missing centres do not get a second
chance.

Only the volume list is produced here. Per-slice tissue filtering is left to the
packing stage, which has to read pixel data anyway; doing it here would mean
reading 115 GB twice.

Run:
    python scripts/build_pretrain_manifest.py
    python scripts/build_pretrain_manifest.py --min-inplane 448
"""

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import SimpleITK as sitk

KTH = Path("/Users/ehogjig/git/kth")
IMAGES = KTH / "images"
TEST_INPUT = KTH / "data_multislice_v3/test/mha/input"
TRAIN_INPUT = KTH / "data_multislice_v3/train/mha/input"
SPLIT_CSV = KTH / "mama-synth/train_test_splits.csv"
REPO = Path(__file__).resolve().parent.parent

EXPECTED_HELD_OUT = 60


def patients_from_slices(d):
    """Patient ids behind a flat directory of 2-D slice files.

    Names look like DUKE_032.mha or DUKE_032_s+1.mha, so the neighbour suffix has
    to come off before the id is usable.
    """
    out = set()
    for f in Path(d).glob("*.mha"):
        out.add(re.sub(r"(_s[+-]\d+)?$", "", f.stem))
    return out


def resolve_exclusions():
    """The patients pretraining must not touch, with the CSV as a cross-check."""
    held_out = patients_from_slices(TEST_INPUT)
    if len(held_out) != EXPECTED_HELD_OUT:
        raise SystemExit(
            f"expected {EXPECTED_HELD_OUT} held-out patients, found {len(held_out)}. "
            f"The evaluation split may have changed; verify {TEST_INPUT} before "
            f"continuing, because this list is the only thing keeping pretraining "
            f"away from the test set.")

    csv_train = csv_test = set()
    if SPLIT_CSV.exists():
        rows = list(csv.DictReader(SPLIT_CSV.open()))
        csv_train = {r["train_split"].strip() for r in rows if r["train_split"].strip()}
        csv_test = {r["test_split"].strip() for r in rows if r["test_split"].strip()}

    trap = held_out & csv_train
    return held_out, csv_test, trap


def scan(min_inplane, exclude):
    kept = defaultdict(list)
    stats = dict(volumes_seen=0, volumes_kept=0, slices_kept=0,
                 dropped_resolution=0, dropped_excluded=0, unreadable=0)
    by_cohort = defaultdict(lambda: dict(patients=set(), volumes=0, slices=0))

    for pid in sorted(os.listdir(IMAGES)):
        pdir = IMAGES / pid
        if not pdir.is_dir() or pid.startswith("."):
            continue
        vols = sorted(pdir.glob("*.nii.gz"))
        if not vols:
            continue
        if pid in exclude:
            stats["dropped_excluded"] += len(vols)
            continue
        for v in vols:
            stats["volumes_seen"] += 1
            try:
                r = sitk.ImageFileReader()
                r.SetFileName(str(v))
                r.ReadImageInformation()
                sx, sy, sz = r.GetSize()
            except Exception:
                stats["unreadable"] += 1
                continue
            if min(sx, sy) < min_inplane:
                stats["dropped_resolution"] += 1
                continue
            kept[pid].append(dict(path=str(v.relative_to(KTH)),
                                  n_slices=int(sz), inplane=[int(sy), int(sx)]))
            stats["volumes_kept"] += 1
            stats["slices_kept"] += int(sz)
            c = pid.split("_")[0]
            by_cohort[c]["patients"].add(pid)
            by_cohort[c]["volumes"] += 1
            by_cohort[c]["slices"] += int(sz)
    return kept, stats, by_cohort


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-inplane", type=int, default=256)
    ap.add_argument("--pretrain-size", type=int, default=256,
                    help="resolution pretraining runs at; larger volumes are "
                         "downsampled to it")
    ap.add_argument("--stride", type=int, default=3,
                    help="recorded for the packing stage, not applied here")
    ap.add_argument("--out", default=str(REPO / "scripts" / "pretrain_manifest.json"))
    args = ap.parse_args()

    held_out, csv_test, trap = resolve_exclusions()

    print("=== exclusions ===\n")
    print(f"  held out, from {TEST_INPUT.relative_to(KTH)} : {len(held_out)}")
    print(f"  CSV test_split, for cross-check only          : {len(csv_test)}")
    print(f"  held-out patients sitting in CSV train_split  : {len(trap)}")
    if trap:
        print(f"\n  Using the CSV's test_split as the exclusion list would have let")
        print(f"  {len(trap)} evaluated patients into pretraining, for example")
        print(f"  {', '.join(sorted(trap)[:3])}. The manifest excludes the")
        print(f"  data_multislice_v3 test patients instead.")

    print(f"\n=== scan, in-plane >= {args.min_inplane} ===\n")
    kept, stats, by_cohort = scan(args.min_inplane, held_out)

    print(f"  {'cohort':<10}{'patients':>10}{'volumes':>9}{'slices':>10}")
    tp = tv = ts = 0
    for c in sorted(by_cohort):
        d = by_cohort[c]
        print(f"  {c:<10}{len(d['patients']):>10}{d['volumes']:>9}{d['slices']:>10,}")
        tp += len(d["patients"]); tv += d["volumes"]; ts += d["slices"]
    print(f"  {'TOTAL':<10}{tp:>10}{tv:>9}{ts:>10,}")

    print(f"\n  volumes seen            : {stats['volumes_seen']:,}")
    print(f"  dropped, below {args.min_inplane:<4}     : {stats['dropped_resolution']:,}")
    print(f"  dropped, held-out patient: {stats['dropped_excluded']:,}")
    print(f"  unreadable              : {stats['unreadable']:,}")

    res = defaultdict(int)
    for vs in kept.values():
        for v in vs:
            res[tuple(v["inplane"])] += v["n_slices"]
    print(f"\n=== resampling to {args.pretrain_size}^2 ===\n")
    for k in sorted(res, key=lambda x: -res[x]):
        how = ("native" if k[0] == args.pretrain_size
               else ("downsample" if k[0] > args.pretrain_size else "UPSAMPLE"))
        print(f"  {k[0]}x{k[1]:<6}{res[k]:>10,} slices   {how}")
    up = sum(v for k, v in res.items() if k[0] < args.pretrain_size)
    if up:
        print(f"\n  {up:,} slices would be upsampled, which fabricates detail.")
    else:
        print(f"\n  Nothing is upsampled, so no detail is fabricated.")

    strided = ts // args.stride
    sz = args.pretrain_size
    print(f"\n=== what the packing stage will see ===\n")
    print(f"  slices at stride {args.stride}      : {strided:,}")
    print(f"  after a tissue filter    : roughly {int(strided * 0.95):,} at the")
    print(f"                             95 % pass rate the survey measured")
    gb = strided * 0.95 * sz * sz / 1e9
    print(f"  packed {sz}^2 uint8       : {gb:.0f} GB in 2 inodes")
    print(f"\n  For comparison, supervised training uses 9,504 slices, so this is")
    print(f"  about {strided * 0.95 / 9504:.0f}x more. Slices from one patient and phase stay")
    print(f"  highly correlated, so the effective sample size is well below the")
    print(f"  count and the stride costs little.")

    manifest = dict(
        min_inplane=args.min_inplane, pretrain_size=args.pretrain_size,
        stride=args.stride,
        excluded_patients=sorted(held_out),
        exclusion_source=str(TEST_INPUT.relative_to(KTH)),
        held_out_in_csv_train=sorted(trap),
        totals=dict(patients=tp, volumes=tv, slices=ts, strided=strided),
        volumes={k: v for k, v in sorted(kept.items())})
    Path(args.out).write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {args.out}")
    print(f"  ({len(kept)} patients, 1 inode)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
