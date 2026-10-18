"""Pack the manifest volumes into one array for masked-reconstruction pretraining.

Turns 7,403 3-D volumes into a single 2-D slice array plus a breast-mask array
and an index. Three constraints shaped the design, and each is measured rather
than assumed.

ONE PASS, BECAUSE OF INODES

The obvious structure is two stages: segment every slice and save the masks,
then pack. That cannot run here. The project holds 2,887,795 files against a
3,000,000 hard limit, leaving 112,205, and 294,166 mask files would hit the
limit and lock the project for every user on it. So masks are produced, used and
discarded inside this one pass, and only the normalised result is kept. The cost
is that the pass is not resumable for free, which is why progress is
checkpointed explicitly below.

FLOAT16, BECAUSE UINT8 LOSES THE LESION

The intention was uint8 at 18 GB. Measuring the real range on 60 test slices
ruled it out: inside the breast the median maximum is +10.2 sigma and reaches
+50.9, with 75 % of slices exceeding +3 sigma, so clipping at +/-3 would remove
most lesion peaks. A linear map over the full range quantises at 0.21 sigma,
which is coarser than the parenchymal differences the backbone should learn.
float16 costs 37 GB instead and carries a relative error near 1e-3.

Breast masks go in a separate bit-packed array at 8 KB per slice, 2.3 GB in
total. They could be recovered from the zeroed background, but a z-scored pixel
inside the breast can legitimately be 0.0, so storing them is both exact and
cheap. Pretraining needs them to keep the reconstruction loss inside the breast,
since the background is constant and would otherwise dominate.

WHOLE STACKS, NOT SINGLE SLICES

Measured on this machine with breast_seg_930: 223 ms per slice called one at a
time, 35 ms per slice when 24 are passed together as a pseudo-3D array, a 6.4x
difference. Over 294,166 slices that is 18 hours against 2.9. nnU-Net's 2-D
configuration iterates the stack internally with batching, so the only change
needed is the shape of the input.

ORDER OF OPERATIONS, MATCHING INFERENCE

inference.py segments the native-resolution slice, computes per-image statistics
from the breast region, zeroes the background, and only then resizes
(inference.py:140-164). This script follows the same order. Normalising after a
resize would compute statistics over interpolated pixels, giving the backbone
inputs on a different basis than finetuning will supply, and that mismatch is
the kind that quietly cancels the benefit of transfer.

Run a small test first, then the full pass:
    python scripts/pack_pretrain_data.py --limit-patients 2
    python scripts/pack_pretrain_data.py
    python scripts/pack_pretrain_data.py --resume
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

KTH = Path("/Users/ehogjig/git/kth")
REPO = Path(__file__).resolve().parent.parent
SEG_PATH = REPO / "src/submission/submission-synthesis/weights/breast_seg_930"

MIN_MASK_FRAC = 0.01      # breast must cover this much of the frame
MIN_MASK_PX = 500         # and this many pixels, for stable z-score statistics


def load_predictor(device):
    import sys
    sys.path.insert(0, str(REPO / "src/submission/submission-synthesis"))
    os.environ.setdefault("nnUNet_raw", "/tmp/nnunet_raw")
    os.environ.setdefault("nnUNet_preprocessed", "/tmp/nnunet_preprocessed")
    os.environ.setdefault("nnUNet_results", "/tmp/nnunet_results")
    from inference import build_nnunet_predictor
    p = build_nnunet_predictor(str(SEG_PATH), device, "checkpoint_final.pth")
    if p is None:
        raise SystemExit(f"breast segmenter not found at {SEG_PATH}")
    return p


NN_PROPS = {"sitk_stuff": {"spacing": (1.0, 1.0, 1.0), "origin": (0.0, 0.0, 0.0),
                           "direction": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)},
            "spacing": [1.0, 1.0, 1.0]}


def segment_stack(predictor, stack, batch):
    """Breast masks for a stack of native-resolution slices.

    Passed as (1, N, Y, X) so nnU-Net's 2-D configuration walks the stack
    internally with batching, which measured 6.4x faster than per-slice calls.
    """
    out = []
    for i in range(0, stack.shape[0], batch):
        chunk = stack[i:i + batch][None].astype(np.float32)
        pred = predictor.predict_single_npy_array(chunk, NN_PROPS, None, None, False)
        pred = np.asarray(pred)
        while pred.ndim > 3:
            pred = pred[0]
        if pred.ndim == 2:
            pred = pred[None]
        out.append(pred)
    return np.concatenate(out, axis=0) > 0


def normalise_and_resize(sl, mask, size):
    """Per-image z-score inside the breast, background zeroed, then resize.

    This is inference.py:152-164 in the same order. Returns None when the breast
    region is too small for the statistics to mean anything, which is the tissue
    filter: slices of mostly air at either end of the stack fail here.
    """
    px = int(mask.sum())
    if px < MIN_MASK_PX or px < MIN_MASK_FRAC * mask.size:
        return None, None
    fg = sl[mask]
    mean = float(fg.mean())
    std = max(float(fg.std()), 1e-8)
    norm = (sl - mean) / std * mask

    t = torch.from_numpy(norm.astype(np.float32))[None, None]
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    m = torch.from_numpy(mask.astype(np.float32))[None, None]
    m = F.interpolate(m, size=(size, size), mode="nearest")
    return t[0, 0].numpy().astype(np.float16), (m[0, 0].numpy() > 0.5)


def upper_bound(manifest):
    return sum(math.ceil(v["n_slices"] / manifest["stride"])
               for vs in manifest["volumes"].values() for v in vs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(REPO / "scripts" / "pretrain_manifest.json"))
    ap.add_argument("--out-dir", default=str(KTH / "pretrain_pack"))
    ap.add_argument("--batch", type=int, default=24,
                    help="slices per predictor call; 24 measured near the plateau")
    ap.add_argument("--limit-patients", type=int, default=0,
                    help="process only the first N patients, for a smoke test")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    man = json.load(open(args.manifest))
    size = man["pretrain_size"]
    stride = man["stride"]
    excluded = set(man["excluded_patients"])

    pats = sorted(man["volumes"])
    if args.limit_patients:
        pats = pats[:args.limit_patients]

    leak = set(pats) & excluded
    if leak:
        raise SystemExit(f"manifest contains held-out patients: {sorted(leak)[:5]}. "
                         f"Refusing to pack; rebuild the manifest.")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    img_path = out / f"pretrain_{size}.npy"
    msk_path = out / f"pretrain_{size}_mask.npy"
    idx_path = out / "pretrain_index.json"
    prog_path = out / "pack_progress.json"

    cap = upper_bound({**man, "volumes": {p: man["volumes"][p] for p in pats}})
    bytes_img = cap * size * size * 2
    packed_row = size * size // 8
    print(f"=== packing {len(pats)} patients at {size}^2, stride {stride} ===\n")
    print(f"  upper bound on slices : {cap:,}")
    print(f"  image array           : {bytes_img/1e9:.1f} GB float16")
    print(f"  mask array            : {cap*packed_row/1e9:.1f} GB bit-packed")
    print(f"  files written         : 4  (2 arrays, index, progress)\n")

    done, n = set(), 0
    if args.resume and prog_path.exists():
        pr = json.load(open(prog_path))
        done, n = set(pr["volumes_done"]), pr["n_written"]
        print(f"  resuming: {len(done)} volumes done, {n:,} slices written\n")

    mode = "r+" if (args.resume and img_path.exists()) else "w+"
    imgs = np.lib.format.open_memmap(img_path, mode=mode, dtype=np.float16,
                                     shape=(cap, size, size))
    msks = np.lib.format.open_memmap(msk_path, mode=mode, dtype=np.uint8,
                                     shape=(cap, packed_row))
    index = json.load(open(idx_path))["slices"] if (args.resume and idx_path.exists()) else []

    predictor = load_predictor(torch.device("cuda" if torch.cuda.is_available()
                                            else "mps" if torch.backends.mps.is_available()
                                            else "cpu"))
    t0 = time.time()
    rejected = 0
    for pi, pid in enumerate(pats):
        start_of_patient = n
        for vol in man["volumes"][pid]:
            if vol["path"] in done:
                continue
            try:
                arr = sitk.GetArrayFromImage(sitk.ReadImage(str(KTH / vol["path"])))
            except Exception as e:
                print(f"  skip {vol['path']}: {type(e).__name__}")
                done.add(vol["path"])
                continue
            arr = arr.astype(np.float32)[::stride]
            masks = segment_stack(predictor, arr, args.batch)
            for k in range(arr.shape[0]):
                img, m = normalise_and_resize(arr[k], masks[k], size)
                if img is None:
                    rejected += 1
                    continue
                if n >= cap:
                    print("  reached the allocated bound; stopping")
                    break
                imgs[n] = img
                msks[n] = np.packbits(m.reshape(-1))
                index.append(dict(i=n, patient=pid, volume=vol["path"],
                                  slice=int(k * stride)))
                n += 1
            done.add(vol["path"])

        if (pi + 1) % 10 == 0 or pi == len(pats) - 1:
            el = time.time() - t0
            rate = n / max(el, 1e-9)
            eta = (cap - n) / max(rate, 1e-9) / 3600
            print(f"  {pi+1:>5}/{len(pats)} patients  {n:>8,} slices  "
                  f"{rate:>6.1f}/s  eta {eta:>5.1f} h  rejected {rejected:,}")
            imgs.flush(); msks.flush()
            prog_path.write_text(json.dumps(dict(volumes_done=sorted(done),
                                                 n_written=n), indent=2))
            idx_path.write_text(json.dumps(dict(
                size=size, stride=stride, n=n, capacity=cap,
                patient_boundary_hint=start_of_patient, slices=index), indent=2))

    imgs.flush(); msks.flush()
    idx_path.write_text(json.dumps(dict(size=size, stride=stride, n=n, capacity=cap,
                                        slices=index), indent=2))
    prog_path.write_text(json.dumps(dict(volumes_done=sorted(done),
                                         n_written=n), indent=2))

    el = time.time() - t0
    print(f"\n=== done in {el/3600:.2f} h ===\n")
    print(f"  slices written  : {n:,} of {cap:,} allocated")
    print(f"  rejected        : {rejected:,} ({rejected/max(n+rejected,1):.1%}) "
          f"for too little breast")
    print(f"  patients        : {len({e['patient'] for e in index})}")
    print(f"\n  The arrays keep their allocated length, so read them as [:n]; n is")
    print(f"  recorded in the index. Verify no held-out patient is present with:")
    print(f"    python scripts/verify_pretrain_pack.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
