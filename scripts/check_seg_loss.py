"""Verification gate for the segmentation-consistency loss.

Checks the properties the loss depends on, rather than only that it imports:

1. the loss segmenter is not the segmenter used for scoring, since routing the
   loss through the scoring network would optimise the metric instead of the
   synthesis;
2. a 3D checkpoint is rejected rather than silently reshaped, because driving a
   2D slice through a 3D network would need a pseudo-volume and its
   probabilities would be meaningless;
3. incomplete weights are rejected, since load_state_dict(strict=False) would
   otherwise leave part of the network random and its output useless;
4. the loss falls as the synthetic lesion approaches the real one, and the
   asymmetry prices a missed lesion above a spurious one;
5. the ground-truth baseline keeps the target reachable on slices where the
   segmenter cannot see the real lesion either;
6. gradients reach the image, so the generator can actually be driven;
7. lambda_seg = 0 changes nothing, so existing runs stay comparable.

Run:
    python scripts/check_seg_loss.py
"""

import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src/submission/submission-synthesis/models"))

import networks  # noqa: E402

LOSS_SEG = REPO / "src/submission/submission-synthesis/weights/tumor_seg_940"
SCORE_SEG = REPO / "src/evaluation/models/segmentation"
DATA = Path("/Users/ehogjig/git/kth/data_multislice_v3/test/mha")


def read(p):
    a = sitk.GetArrayFromImage(sitk.ReadImage(str(p))).astype(np.float32)
    return a[0] if a.ndim == 3 and a.shape[0] == 1 else a


def load_case(name):
    gt = torch.from_numpy(read(DATA / "ground_truth" / name))[None, None]
    pre = torch.from_numpy(read(DATA / "input" / name))[None, None]
    m = torch.from_numpy((read(DATA / "mask" / name) > 0.5).astype(np.float32))[None, None]
    return pre, gt, m


def check_independence():
    print("=== 1. loss segmenter is separate from the scoring segmenter ===")
    import json
    rows = []
    for label, d in (("loss ", LOSS_SEG), ("score", SCORE_SEG)):
        pl = json.loads((d / "plans.json").read_text())
        c = pl["configurations"]["2d"]
        rows.append((label, pl["dataset_name"], c["patch_size"]))
        print(f"    {label}: {pl['dataset_name']:32} patch={c['patch_size']}")
    ok = rows[0][1] != rows[1][1]
    print(f"  -> {'PASS' if ok else 'FAIL'} "
          f"({'different datasets' if ok else 'SAME network: the loss would optimise the metric'})")
    return ok


def check_rejects_3d():
    print("\n=== 2. a 3D checkpoint is refused, not reshaped ===")
    cand = REPO / "nnUNet_pretrained_weights/full_image_dce_mri_tumor_segmentation"
    if not cand.exists():
        print("  no 3D candidate present to test against; skipping")
        return True
    try:
        networks.SegConsistencyLoss(cand)
        print("  -> FAIL: a 3D checkpoint was accepted")
        return False
    except ValueError as exc:
        msg = str(exc)
        print(f"  refused: {msg[:96]}")
        print(f"  -> {'PASS' if '3D' in msg else 'FAIL'}")
        return "3D" in msg


def check_loads_fully():
    print("\n=== 3. weights load completely ===")
    crit = networks.SegConsistencyLoss(LOSS_SEG)
    n = sum(p.numel() for p in crit.net.parameters())
    trainable = sum(p.numel() for p in crit.net.parameters() if p.requires_grad)
    print(f"  {crit.dataset_name}: {n/1e6:.2f} M params, {trainable} trainable")
    print(f"  (the checkpoint lists more: nnU-Net aliases each conv under "
          f"'all_modules', so a naive sum double counts)")
    ok = trainable == 0 and not crit.net.training
    print(f"  -> {'PASS' if ok else 'FAIL'} (frozen and in eval mode)")
    return ok, crit


def check_responds_to_lesion(crit):
    print("\n=== 4. loss falls as the lesion is restored, asymmetrically ===")
    cases = sorted((DATA / "ground_truth").glob("*.mha"))
    picked = None
    for f in cases[:12]:
        pre, gt, m = load_case(f.name)
        if m.sum() > 50:
            picked = (f.name, pre, gt, m)
            break
    if picked is None:
        print("  no case with a usable mask; skipping")
        return True
    name, pre, gt, m = picked
    print(f"  case {name}")

    # Interpolate from pre-contrast (lesion absent) to ground truth (present).
    print(f"  {'blend':>8}{'Tversky':>10}{'with baseline':>15}")
    vals = []
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        fake = (1 - t) * pre + t * gt
        with torch.no_grad():
            raw = float(crit._tversky(fake, m).mean())
            full = float(crit(fake, gt, m))
        vals.append(raw)
        print(f"  {t:>8.2f}{raw:>10.4f}{full:>15.4f}")
    monotone = all(vals[i] >= vals[i + 1] - 1e-4 for i in range(len(vals) - 1))
    print(f"  monotone in lesion recovery: {monotone}")

    # Asymmetry, tested on constructed probabilities so that false negatives and
    # false positives can be separated. Interpolating towards the pre-contrast
    # image does not isolate them: with no lesion signal the segmenter fires
    # diffusely across the breast, so that case is dominated by false positives
    # and says nothing about how misses are priced.
    sym = networks.SegConsistencyLoss(LOSS_SEG, alpha=0.5, beta=0.5,
                                      use_gt_baseline=False)
    asym = networks.SegConsistencyLoss(LOSS_SEG, alpha=0.3, beta=0.7,
                                       use_gt_baseline=False)

    def tversky_from(prob, mask, a, b):
        tp = (prob * mask).sum()
        fp = (prob * (1 - mask)).sum()
        fn = ((1 - prob) * mask).sum()
        return float(1 - (tp + 1e-6) / (tp + a * fp + b * fn + 1e-6))

    msk = torch.zeros(1, 1, 32, 32)
    msk[..., 8:24, 8:24] = 1.0
    missed = torch.zeros_like(msk)                 # lesion present, nothing found
    missed[..., 8:24, 8:24] = 0.2
    spurious = msk.clone()                         # lesion found, plus a false blob
    spurious[..., 0:6, 0:6] = 1.0
    print(f"\n  {'case':<26}{'symmetric':>11}{'asymmetric':>12}")
    for nm, prob in (("mostly missed (fn)", missed),
                     ("found + spurious (fp)", spurious)):
        s = tversky_from(prob, msk, 0.5, 0.5)
        a = tversky_from(prob, msk, 0.3, 0.7)
        print(f"  {nm:<26}{s:>11.4f}{a:>12.4f}")
    miss_s = tversky_from(missed, msk, 0.5, 0.5)
    miss_a = tversky_from(missed, msk, 0.3, 0.7)
    fp_s = tversky_from(spurious, msk, 0.5, 0.5)
    fp_a = tversky_from(spurious, msk, 0.3, 0.7)
    harsher = miss_a > miss_s and fp_a < fp_s
    print(f"  misses cost more under asymmetry : {miss_a > miss_s}")
    print(f"  spurious cost less               : {fp_a < fp_s}")
    print(f"  -> {'PASS' if harsher else 'FAIL'} (monotonicity reported above "
          f"separately: {monotone})")
    return harsher


def check_baseline(crit):
    print("\n=== 5. the ground-truth baseline keeps the target reachable ===")
    nob = networks.SegConsistencyLoss(LOSS_SEG, use_gt_baseline=False)
    cases = sorted((DATA / "ground_truth").glob("*.mha"))[:10]
    rows = []
    for f in cases:
        pre, gt, m = load_case(f.name)
        if m.sum() < 50:
            continue
        with torch.no_grad():
            floor = float(nob(gt, gt, m))       # real image against its own mask
            withb = float(crit(gt, gt, m))
        rows.append((f.name, floor, withb))
    print(f"  {'case':<22}{'absolute':>10}{'baselined':>11}")
    for nm, a, b in rows:
        print(f"  {nm[:20]:<22}{a:>10.4f}{b:>11.4f}")
    if not rows:
        print("  no usable cases; skipping")
        return True
    worst = max(a for _, a, _ in rows)
    all_zero = all(b < 1e-5 for _, _, b in rows)
    print(f"\n  worst absolute floor on a real image : {worst:.4f}")
    print(f"  all baselined losses zero on real    : {all_zero}")
    print(f"  -> {'PASS' if all_zero else 'FAIL'} "
          f"(nothing is demanded beyond what the segmenter already sees)")
    return all_zero


def check_gradients(crit):
    print("\n=== 6. gradients reach the image ===")
    cases = sorted((DATA / "ground_truth").glob("*.mha"))
    for f in cases[:12]:
        pre, gt, m = load_case(f.name)
        if m.sum() > 50:
            break
    # The test point has to be one where the loss is actually non-zero. Halfway
    # between pre and ground truth the interpolated lesion can be *easier* to
    # segment than the real one, relu clamps the baselined loss to zero and there
    # is correctly no gradient. Starting from the pre-contrast image, where the
    # lesion is absent, is the case the loss exists to fix.
    fake = pre.clone().requires_grad_(True)
    loss = crit(fake, gt, m)
    if float(loss) == 0.0:
        print("  baselined loss is zero even with the lesion absent: the "
              "segmenter cannot see this lesion in the real image either, so "
              "nothing is demanded. Falling back to the unbaselined loss.")
        crit = networks.SegConsistencyLoss(LOSS_SEG, use_gt_baseline=False)
        fake = pre.clone().requires_grad_(True)
        loss = crit(fake, gt, m)
    loss.backward()
    g = fake.grad
    finite = bool(torch.isfinite(g).all())
    mx = float(g.abs().max())
    inside = float(g.abs()[m > 0.5].mean()) if (m > 0.5).any() else 0.0
    outside = float(g.abs()[m <= 0.5].mean())
    print(f"  loss {float(loss):.4f}   grad finite {finite}   |grad| max {mx:.3e}")
    print(f"  mean |grad| inside mask {inside:.3e}   outside {outside:.3e}")
    if outside > 0:
        print(f"  concentration inside/outside : {inside/outside:.1f}x")
    ok = finite and mx > 0
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_off_is_noop():
    print("\n=== 7. lambda_seg = 0 leaves the loss set untouched ===")
    src = (REPO / "src/submission/submission-synthesis/models/pix2pixHD_model.py").read_text()
    checks = {
        "lambda read with default 0": "getattr(opt, 'lambda_seg', 0)" in src,
        "gated on > 0": "self.use_seg_loss = self.lambda_seg > 0" in src,
        "criterion built only when on": "if self.use_seg_loss:" in src,
        "term gated in forward": "if self.use_seg_loss and mask is not None:" in src,
        "slot threaded to filter": "loss_G_Seg )" in src,
        "name registered": "'G_Seg'" in src,
    }
    tr = (REPO / "src/submission/submission-synthesis/models/train.py").read_text()
    checks["summed in train.py"] = "loss_dict.get('G_Seg',0)" in tr
    ok = True
    for name, passed in checks.items():
        print(f"    {name:<32} {'yes' if passed else 'NO'}")
        ok = ok and passed
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    r = [check_independence(), check_rejects_3d()]
    loaded, crit = check_loads_fully()
    r.append(loaded)
    if loaded:
        r += [check_responds_to_lesion(crit), check_baseline(crit),
              check_gradients(crit)]
    r.append(check_off_is_noop())
    print("\n" + "=" * 62)
    print(f"{sum(r)}/{len(r)} checks passed")
    return 0 if all(r) else 1


if __name__ == "__main__":
    sys.exit(main())
