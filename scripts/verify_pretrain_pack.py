"""Verify a packed pretraining set before any GPU time is spent on it.

The pack is built in one pass with masks discarded as they are used, so there is
no intermediate artefact to inspect afterwards. These checks are the only thing
standing between a corrupt pack and a pretraining run that appears to work and
produces a backbone trained on noise.

The first check is the one that matters. If any of the 60 evaluated patients
reached the pack, every number in the results chapter becomes meaningless, and
the failure would be invisible in the loss curves. It is checked against the
evaluation directory directly rather than against the manifest, so that an error
in the manifest cannot hide an error in the pack.

Run:
    python scripts/verify_pretrain_pack.py
    python scripts/verify_pretrain_pack.py --pack /tmp/pack_test
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

KTH = Path("/Users/ehogjig/git/kth")
TEST_INPUT = KTH / "data_multislice_v3/test/mha/input"


def held_out():
    return {re.sub(r"(_s[+-]\d+)?$", "", f.stem) for f in TEST_INPUT.glob("*.mha")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=str(KTH / "pretrain_pack"))
    ap.add_argument("--sample", type=int, default=400,
                    help="slices to read for the content checks")
    args = ap.parse_args()

    p = Path(args.pack)
    idx = json.load(open(p / "pretrain_index.json"))
    n, size = idx["n"], idx["size"]
    entries = idx["slices"]
    imgs = np.load(p / f"pretrain_{size}.npy", mmap_mode="r")
    msks = np.load(p / f"pretrain_{size}_mask.npy", mmap_mode="r")

    fails = []

    print(f"=== {p.name}: {n:,} slices at {size}^2 ===\n")

    ho = held_out()
    present = {e["patient"] for e in entries}
    leak = present & ho
    print(f"1. held-out patients")
    print(f"   evaluated patients        : {len(ho)}")
    print(f"   patients in the pack      : {len(present)}")
    print(f"   intersection              : {len(leak)}")
    if leak:
        fails.append(f"{len(leak)} held-out patients in the pack: {sorted(leak)[:5]}")
        print(f"   LEAK: {sorted(leak)[:5]}")
    else:
        print(f"   clean")

    print(f"\n2. index consistency")
    ok_len = len(entries) == n
    ids = [e["i"] for e in entries]
    ok_seq = ids == list(range(n))
    print(f"   index entries match n     : {ok_len}  ({len(entries):,} vs {n:,})")
    print(f"   positions are 0..n-1      : {ok_seq}")
    if not ok_len:
        fails.append("index length disagrees with the recorded slice count")
    if not ok_seq:
        fails.append("index positions are not a dense 0..n-1 range")

    print(f"\n3. content, on {min(args.sample, n)} sampled slices")
    rng = np.random.default_rng(0)
    sel = rng.choice(n, size=min(args.sample, n), replace=False)
    mx, nz, ma, bad = [], [], [], 0
    for i in sel:
        a = np.asarray(imgs[i], dtype=np.float32)
        if not np.isfinite(a).all():
            bad += 1
            continue
        mx.append(float(a.max()))
        nz.append(float((a != 0).mean()))
        ma.append(int(np.unpackbits(msks[i]).sum()))
    mx, nz, ma = np.array(mx), np.array(nz), np.array(ma)
    print(f"   non-finite slices         : {bad}")
    print(f"   all-zero slices           : {int((mx == 0).sum())}")
    print(f"   per-slice max, median     : {np.median(mx):+.2f} sigma")
    print(f"   per-slice max, range      : [{mx.min():+.2f}, {mx.max():+.2f}]")
    print(f"   breast area, median       : {np.median(ma)/size**2:.1%} of frame")
    print(f"   breast area, min          : {ma.min()/size**2:.1%}")
    print(f"   non-zero pixels, median   : {np.median(nz):.1%}")

    if bad:
        fails.append(f"{bad} non-finite slices")
    if int((mx == 0).sum()):
        fails.append(f"{int((mx == 0).sum())} all-zero slices")
    # the evaluation slices reach +10.2 sigma in the breast on median, and these
    # volumes include pre-contrast phases with no enhancement, so a lower median
    # is expected; a median below 1 would mean the normalisation did not take
    if np.median(mx) < 1.0:
        fails.append(f"per-slice max median {np.median(mx):+.2f} is too low; "
                     f"the per-image normalisation may not have been applied")
    if np.median(ma) / size**2 < 0.02:
        fails.append("median breast area under 2 % of frame; masks look wrong")

    print(f"\n4. image and mask agree")
    dis = []
    for i in sel[:60]:
        a = np.asarray(imgs[i], dtype=np.float32)
        m = np.unpackbits(msks[i]).reshape(size, size).astype(bool)
        outside_nonzero = float((a[~m] != 0).mean()) if (~m).any() else 0.0
        dis.append(outside_nonzero)
    print(f"   non-zero pixels outside the mask, mean : {np.mean(dis):.2%}")
    print(f"   (the background is zeroed, so this should be near 0)")
    if np.mean(dis) > 0.01:
        fails.append(f"{np.mean(dis):.1%} of out-of-mask pixels are non-zero; "
                     f"the background zeroing did not hold")

    print(f"\n5. cohort balance against the evaluation set")
    from collections import Counter
    cp = Counter(e["patient"].split("_")[0] for e in entries)
    ce = Counter(x.split("_")[0] for x in ho)
    tot = sum(cp.values())
    print(f"   {'cohort':<10}{'pack':>12}{'pack %':>9}{'eval %':>9}")
    for c in sorted(set(cp) | set(ce)):
        pe = cp.get(c, 0) / max(tot, 1)
        ee = ce.get(c, 0) / max(sum(ce.values()), 1)
        print(f"   {c:<10}{cp.get(c,0):>12,}{pe:>8.1%}{ee:>9.1%}")
    print(f"   A large gap here is not an error, but it bounds how far")
    print(f"   pretraining can help on the cohorts that dominate evaluation.")

    print(f"\n=== {'PASS' if not fails else 'FAIL'} ===")
    for f in fails:
        print(f"  - {f}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
