"""Build the chest-region loss mask for the second (chest-specialist) generator.

The GAN confines its losses to the breast (pix2pixHD_model.py:224-228 multiplies
both fake and real by the mask), and inference then copies everything outside the
breast from the pre-contrast slice (inference.py:180). Measured on the 299 test
slices, that leaves 74.5 % of the total squared error in a region the model never
sees and never predicts: if the outside were perfect, MSE would fall from 0.821
to 0.210. This mask is what lets a second generator be trained on that region.

REGION DEFINITION

    chest      = foreground AND NOT breast
    transition = a band of `rim` pixels just inside the breast edge
    loss_mask  = chest OR transition

The transition band exists so both generators are supervised across the seam
merge_two_generators.py blends over, which otherwise shows a visible step.

This differs from the flow-matching implementation on purpose. train_fm.py:205-209
computes the band as `max_pool2d(mask) * mask`, and since dilation produces a
superset of the mask, that product is the whole breast again, making its
`loss_mask` equal to `chest + breast`, i.e. the entire foreground. That makes its
chest mode differ from its full-target mode only outside the foreground. Whether
that was intended there is unclear; here the band is the dilated breast minus the
*eroded* breast, so it really is a band and the chest generator is not supervised
on breast interior that generator A already handles.

Run (on Berzelius, where all four mask directories hold 9,504 files):
    python scripts/make_chest_mask.py --split train
    python scripts/make_chest_mask.py --split test --rim 16
"""

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_dilation, binary_erosion, iterate_structure
from scipy.ndimage import generate_binary_structure


def read(p):
    img = sitk.ReadImage(str(p))
    a = sitk.GetArrayFromImage(img).astype(np.float32)
    return (a[0] if a.ndim == 3 and a.shape[0] == 1 else a), img, a.ndim


def band(mask, rim):
    """Dilated minus eroded breast: a genuine band straddling the edge."""
    st = generate_binary_structure(2, 1)
    big = iterate_structure(st, rim)
    outer = binary_dilation(mask, structure=big)
    inner = binary_erosion(mask, structure=big)
    return outer & ~inner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../data_multislice_v3",
                    help="dataset root holding <split>/mha/{foreground,breast_mask}")
    ap.add_argument("--split", default="train")
    ap.add_argument("--rim", type=int, default=16,
                    help="band half-width in pixels; 16 matches the FM setting")
    ap.add_argument("--out", default="chest_mask")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    base = Path(args.root) / args.split / "mha"
    fg_dir, br_dir = base / "foreground", base / "breast_mask"
    out_dir = base / args.out
    for d in (fg_dir, br_dir):
        if not d.is_dir():
            raise SystemExit(f"missing {d}")
    out_dir.mkdir(exist_ok=True)

    names = sorted(p.name for p in fg_dir.glob("*.mha"))
    paired = [n for n in names if (br_dir / n).exists()]
    if len(paired) != len(names):
        print(f"  {len(names) - len(paired)} foreground files have no breast mask "
              f"and are skipped")
    if args.limit:
        paired = paired[:args.limit]

    print(f"=== chest mask, {args.split}, rim {args.rim} px ===")
    print(f"  pairs: {len(paired)}\n")

    fr_fg, fr_br, fr_chest, fr_band, fr_loss = [], [], [], [], []
    empty = 0
    for k, n in enumerate(paired):
        fg, img, nd = read(fg_dir / n)
        br, _, _ = read(br_dir / n)
        fg, br = fg > 0.5, br > 0.5

        chest = fg & ~br
        tb = band(br, args.rim) & fg          # keep the band inside the body
        loss = chest | tb

        if loss.sum() == 0:
            empty += 1
        fr_fg.append(fg.mean()); fr_br.append(br.mean())
        fr_chest.append(chest.mean()); fr_band.append(tb.mean())
        fr_loss.append(loss.mean())

        arr = loss.astype(np.float32)
        o = sitk.GetImageFromArray(arr[np.newaxis, ...] if nd == 3 else arr)
        o.CopyInformation(img)
        sitk.WriteImage(o, str(out_dir / n))

        if (k + 1) % 2000 == 0:
            print(f"  {k+1}/{len(paired)}")

    f = lambda v: f"{np.mean(v):.1%}"
    print(f"\n=== area as a fraction of the frame ===\n")
    print(f"  foreground          {f(fr_fg)}")
    print(f"  breast              {f(fr_br)}")
    print(f"  chest = fg - breast {f(fr_chest)}")
    print(f"  transition band     {f(fr_band)}")
    print(f"  loss mask           {f(fr_loss)}")
    print(f"\n  empty loss masks    {empty}")
    print(f"\n  Generator A is supervised on {f(fr_br)} of the frame, B on "
          f"{f(fr_loss)}.")
    print(f"  Overlap is the band only, so the two do not compete for the breast")
    print(f"  interior. If 'loss mask' came out close to 'foreground', the band is")
    print(f"  swallowing the breast and --rim is too large.")
    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
