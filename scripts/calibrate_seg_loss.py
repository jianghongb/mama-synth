"""Calibrate lambda_seg against the converged model before spending a GPU run.

The segmentation-consistency term is new, so its scale relative to the existing
losses is unknown. Setting lambda_seg blind risks either a term so small it does
nothing or one large enough to dominate, and either way the answer only arrives
after a 48 h training run.

This measures the term directly on predictions_v32_no_msec, the outputs of the
configuration the new term would be added to (row 5 of the ablation,
lambda_MSEC = 0). Nothing is trained: the predictions already exist, so the
numbers come back in minutes.

Three things it answers:

* the magnitude of the raw Tversky and of the baselined relu form on a converged
  model, which sets the scale lambda_seg has to match;
* how many cases sit on the non-zero side of the relu, i.e. how many the term
  would actually act on. If almost none do, the term is inert at this operating
  point and the weight is irrelevant;
* which cases those are, since the term concentrates on exactly the slices the
  frozen segmenter finds harder in the synthetic image than in the real one, and
  those are worth inspecting before trusting the signal.

Run:
    python scripts/calibrate_seg_loss.py
    python scripts/calibrate_seg_loss.py --pred predictions_v32_final
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src/submission/submission-synthesis/models"))

import networks  # noqa: E402

LOSS_SEG = REPO / "src/submission/submission-synthesis/weights/tumor_seg_940"
DATA = Path("/Users/ehogjig/git/kth/data_multislice_v3/test/mha")


def read(p):
    a = sitk.GetArrayFromImage(sitk.ReadImage(str(p))).astype(np.float32)
    return a[0] if a.ndim == 3 and a.shape[0] == 1 else a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="predictions_v32_no_msec",
                    help="prediction directory, relative to the repo root")
    ap.add_argument("--limit", type=int, default=0, help="stop after N cases")
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--beta", type=float, default=0.7)
    args = ap.parse_args()

    pred_dir = REPO / args.pred
    if not pred_dir.exists():
        print(f"not found: {pred_dir}")
        return 1

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    crit = networks.SegConsistencyLoss(LOSS_SEG, alpha=args.alpha,
                                       beta=args.beta, use_gt_baseline=True)
    crit.net = crit.net.to(dev)
    print(f"segmenter : {crit.dataset_name}  alpha={args.alpha} beta={args.beta}")
    print(f"prediction: {args.pred}")
    print(f"device    : {dev}\n")

    cases = sorted(pred_dir.glob("*.mha"))
    if args.limit:
        cases = cases[:args.limit]

    rows = []
    for f in cases:
        gt_p = DATA / "ground_truth" / f.name
        m_p = DATA / "mask" / f.name
        if not gt_p.exists() or not m_p.exists():
            continue
        m = (read(m_p) > 0.5).astype(np.float32)
        if m.sum() < 20:
            continue
        fake = torch.from_numpy(read(f))[None, None].to(dev)
        real = torch.from_numpy(read(gt_p))[None, None].to(dev)
        mask = torch.from_numpy(m)[None, None].to(dev)
        with torch.no_grad():
            raw = float(crit._tversky(fake, mask).mean())
            floor = float(crit._tversky(real, mask).mean())
            baselined = float(crit(fake, real, mask))
        rows.append((f.name, raw, floor, baselined))

    if not rows:
        print("no usable cases")
        return 1

    raws = np.array([r[1] for r in rows])
    floors = np.array([r[2] for r in rows])
    base = np.array([r[3] for r in rows])
    active = base > 1e-6

    print(f"=== scale on {len(rows)} cases ===")
    print(f"  raw Tversky on prediction : mean {raws.mean():.4f}  "
          f"median {np.median(raws):.4f}  max {raws.max():.4f}")
    print(f"  raw Tversky on real image : mean {floors.mean():.4f}  "
          f"median {np.median(floors):.4f}  max {floors.max():.4f}")
    print(f"  baselined relu form       : mean {base.mean():.4f}  "
          f"median {np.median(base):.4f}  max {base.max():.4f}")

    print(f"\n=== how often the term is awake ===")
    print(f"  cases above the floor : {int(active.sum())}/{len(rows)} "
          f"({100*active.mean():.1f} %)")
    if active.any():
        print(f"  their mean loss       : {base[active].mean():.4f}")
    print(f"  cases at zero         : {int((~active).sum())} "
          f"(the prediction is already as segmentable as the real image)")

    # The existing terms are what lambda_seg has to be commensurate with. Their
    # weights come from the paper's reported configuration.
    print(f"\n=== what the other terms contribute, for scale ===")
    print(f"  lambda_vgg   = 10    lambda_feat = 10")
    print(f"  tumor_weight = 10    lambda_gan  = 1")
    print(f"  A loss term is only felt if lambda * value lands in the same")
    print(f"  decade as those products, not if it is orders below.")
    mean_active = base[active].mean() if active.any() else base.mean()
    for lam in (1, 5, 10, 25, 50):
        print(f"    lambda_seg {lam:>3} -> contribution {lam*mean_active:.4f} "
              f"on the cases it acts on")

    print(f"\n=== the cases it would act on ===")
    worst = sorted(rows, key=lambda r: -r[3])[:12]
    print(f"  {'case':<24}{'pred':>8}{'real':>8}{'loss':>8}")
    for nm, r, fl, b in worst:
        if b <= 1e-6:
            break
        print(f"  {nm[:22]:<24}{r:>8.4f}{fl:>8.4f}{b:>8.4f}")

    # A high floor means the segmenter cannot see the real lesion either. Those
    # cases are excluded by the baseline, which is the point, but a large share of
    # them means the term is supervising a small subset of the data.
    hard = floors > 0.5
    print(f"\n=== cases the segmenter cannot see even in the real image ===")
    print(f"  floor > 0.5 : {int(hard.sum())}/{len(rows)} "
          f"({100*hard.mean():.1f} %) -- excluded by the baseline")

    out = REPO / "scripts" / "seg_loss_calibration.json"
    out.write_text(json.dumps({
        "prediction_dir": args.pred,
        "segmenter": crit.dataset_name,
        "alpha": args.alpha, "beta": args.beta,
        "n_cases": len(rows),
        "raw_mean": float(raws.mean()), "floor_mean": float(floors.mean()),
        "baselined_mean": float(base.mean()),
        "baselined_mean_active": float(mean_active),
        "active_fraction": float(active.mean()),
        "hard_floor_fraction": float(hard.mean()),
        "per_case": [{"case": n, "pred": r, "real": f, "loss": b}
                     for n, r, f, b in rows],
    }, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
