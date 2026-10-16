"""Probe whether DINOv2 features are a better perceptual signal than VGG here.

The supervisor suggested DINO as a backbone. Before committing GPU time, this
answers the question that decides whether any DINO route is worth taking: do its
features respond to contrast enhancement at all, and do they respond better than
the VGG features already in use?

The reasoning behind the test. A perceptual loss is only useful if moving in its
feature space corresponds to moving in the quantity we care about. For virtual
contrast enhancement that quantity is enhancement magnitude inside the lesion.
So the probe measures, over real pre/post pairs:

1. separation -- how far apart pre-contrast and post-contrast sit in feature
   space, normalised by how far apart two nearby slices of the *same* phase sit.
   A ratio near 1 means the feature cannot tell enhancement from ordinary
   anatomical variation, and a loss built on it would supervise noise;
2. correlation -- whether the feature distance tracks the actual enhancement
   magnitude in the tumour. A perceptual loss pushes to reduce feature distance;
   if that distance is uncorrelated with enhancement, reducing it says nothing
   about getting the enhancement right;
3. domain fit -- whether the activations on greyscale MRI look like the
   activations the network was trained to produce. DINOv2 saw 142M natural RGB
   images; dead or saturated features would mean the greyscale input is outside
   its operating range.

Both networks are compared on identical pairs so the numbers are directly
comparable. VGG-19 is the loss currently used by the GAN at lambda 10.

Note on availability: the timm DINOv2 and DINOv3 weights are gated on
HuggingFace and return 401 without a license acceptance. The official torch.hub
entrypoint works without authentication, so that is what this uses. DINOv3's
ConvNeXt variants would align better with the generator's 32x32 bottleneck than
a patch-14 ViT does, and are worth revisiting if the license is accepted.

Run:
    python scripts/probe_dino_features.py
    python scripts/probe_dino_features.py --n 60
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
DATA = Path("/Users/ehogjig/git/kth/data_multislice_v3/test/mha")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def read(p):
    a = sitk.GetArrayFromImage(sitk.ReadImage(str(p))).astype(np.float32)
    return a[0] if a.ndim == 3 and a.shape[0] == 1 else a


def to_network_input(x, size):
    """Greyscale z-scored slice -> 3-channel ImageNet-normalised tensor.

    The slices are per-image z-scored, so they are centred near zero with a few
    large positive outliers in the lesion. Both networks expect roughly [0,1]
    before ImageNet normalisation, so the mapping has to be fixed and shared:
    clipping at +/-3 sigma and rescaling keeps the lesion inside range while
    treating both networks identically, which is what makes their numbers
    comparable.
    """
    t = torch.from_numpy(x)[None, None]
    t = t.clamp(-3, 3) / 6.0 + 0.5
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    t = t.repeat(1, 3, 1, 1)
    return (t - IMAGENET_MEAN) / IMAGENET_STD


class DinoFeat:
    """DINOv2 ViT-S/14 patch tokens, L2-normalised per token."""

    name = "DINOv2 ViT-S/14"
    size = 518          # 14 * 37; 512 is not a multiple of the patch size

    def __init__(self, dev):
        self.net = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                                  verbose=False).to(dev).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.dev = dev

    @torch.no_grad()
    def __call__(self, x):
        out = self.net.forward_features(to_network_input(x, self.size).to(self.dev))
        f = out["x_norm_patchtokens"]           # (1, 1369, 384)
        return F.normalize(f, dim=-1).flatten()


class Dinov3ConvNextFeat:
    """DINOv3 ConvNeXt-S multi-scale features.

    Geometrically this is the candidate that actually fits. At 512 input it
    returns 128/64/32/16 maps, so stage 2 lands exactly on the generator's 32x32
    bottleneck and the set mirrors the multi-scale structure VGGLoss already
    uses, which means it could replace VGGLoss without changing its interface.
    The patch-14 ViT cannot: 512 is not a multiple of 14, so it needs a resample
    to 518 and its 1369 tokens have to be interpolated back onto a spatial grid.

    Stages are concatenated after per-stage L2 normalisation so that no single
    stage dominates by virtue of having more elements.
    """

    name = "DINOv3 ConvNeXt-S"
    size = 512

    def __init__(self, dev):
        import os
        os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
        import timm
        self.net = timm.create_model("convnext_small.dinov3_lvd1689m",
                                     pretrained=True, features_only=True).to(dev).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.dev = dev

    @torch.no_grad()
    def __call__(self, x):
        fs = self.net(to_network_input(x, self.size).to(self.dev))
        return torch.cat([F.normalize(f.flatten(1), dim=-1).flatten() for f in fs])


class VggFeat:
    """VGG-19 relu activations, the loss the GAN uses today."""

    name = "VGG-19 (current)"
    size = 512

    def __init__(self, dev):
        from torchvision.models import VGG19_Weights, vgg19
        self.net = vgg19(weights=VGG19_Weights.DEFAULT).features[:27].to(dev).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.dev = dev

    @torch.no_grad()
    def __call__(self, x):
        f = self.net(to_network_input(x, self.size).to(self.dev))
        return F.normalize(f.flatten(1), dim=-1).flatten()


def collect_cases(n):
    """Pairs that have a pre, a post, a mask, and a same-phase neighbour.

    The neighbour is what makes the separation number interpretable: without a
    baseline for ordinary anatomical variation, a raw pre-to-post distance has no
    scale.
    """
    out = []
    for f in sorted((DATA / "ground_truth").glob("*.mha")):
        stem = f.stem
        if "_s" in stem:                 # use only the centre slice as the anchor
            continue
        nb = DATA / "input" / f"{stem}_s+1.mha"
        if not (DATA / "input" / f.name).exists() or not nb.exists():
            continue
        m = read(DATA / "mask" / f.name) > 0.5
        if m.sum() < 50:
            continue
        out.append((f.name, f"{stem}_s+1.mha"))
        if n and len(out) >= n:
            break
    return out


def test_error_alignment(ex, cases, preds_dir):
    """Does feature distance track how wrong the synthesised enhancement is?

    This is the test that matters, because it reproduces what the loss actually
    does. The first test uses real pre/post pairs, but the perceptual loss acts
    between a synthesised image and the real one, and pix2pixHD_model.py:268
    feeds it fake_masked and real_masked, i.e. restricted to the breast. Both
    are mirrored here: real predictions from the row (5) baseline, multiplied by
    the breast mask.

    For every case it measures the feature distance d(prediction, ground truth)
    and the error in mean tumour enhancement, then correlates them. A perceptual
    loss descends the first quantity, so it only improves enhancement accuracy
    to the extent that the first tracks the second. A rho near zero means
    descending this feature distance carries no information about getting the
    enhancement right, whatever else it may be enforcing.
    """
    from scipy.stats import spearmanr

    dists, errors = [], []
    for name, _ in cases:
        p = preds_dir / name
        if not p.exists():
            continue
        pred, gt = read(p), read(DATA / "ground_truth" / name)
        pre = read(DATA / "input" / name)
        breast = read(DATA / "breast_mask" / name) > 0.5
        tumour = read(DATA / "mask" / name) > 0.5
        if tumour.sum() < 50:
            continue
        dists.append(float((ex(pred * breast) - ex(gt * breast)).norm()))
        errors.append(abs(float((pred - pre)[tumour].mean())
                          - float((gt - pre)[tumour].mean())))
    if len(dists) < 8:
        return None
    rho, p = spearmanr(dists, errors)
    return dict(rho=float(rho), p=float(p), n=len(dists))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="number of cases")
    ap.add_argument("--preds", default=str(REPO / "predictions_v32_no_msec"),
                    help="row (5) baseline predictions, for the error-alignment test")
    args = ap.parse_args()

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cases = collect_cases(args.n)
    if not cases:
        print("no usable cases found")
        return 1
    preds_dir = Path(args.preds)
    print(f"device: {dev}   cases: {len(cases)}   preds: {preds_dir.name}\n")

    results = {}
    for Extractor in (VggFeat, DinoFeat, Dinov3ConvNextFeat):
        ex = Extractor(dev)
        sep, enh, act = [], [], []
        for name, nb_name in cases:
            pre = read(DATA / "input" / name)
            post = read(DATA / "ground_truth" / name)
            nb = read(DATA / "input" / nb_name)
            mask = read(DATA / "mask" / name) > 0.5

            f_pre, f_post, f_nb = ex(pre), ex(post), ex(nb)
            d_phase = float((f_post - f_pre).norm())      # enhancement
            d_anat = float((f_nb - f_pre).norm())         # ordinary variation
            sep.append(d_phase / max(d_anat, 1e-8))
            enh.append(float((post - pre)[mask].mean()))
            act.append(float(f_post.abs().mean()))

        sep = np.array(sep); enh = np.array(enh)
        from scipy.stats import spearmanr
        rho, p = spearmanr(sep, enh)
        align = test_error_alignment(ex, cases, preds_dir) if preds_dir.is_dir() else None
        results[ex.name] = dict(
            sep_mean=float(sep.mean()), sep_median=float(np.median(sep)),
            sep_min=float(sep.min()), sep_max=float(sep.max()),
            rho=float(rho), p=float(p), act_mean=float(np.mean(act)),
            dead=float(np.mean([a < 1e-6 for a in act])), align=align)
        del ex

    print("=== 1. separation: enhancement against ordinary anatomical variation ===")
    print("    ratio = ||f(post) - f(pre)|| / ||f(neighbour slice) - f(pre)||")
    print("    at 1.0 the feature cannot distinguish enhancement from moving one slice\n")
    print(f"  {'feature':<22}{'mean':>8}{'median':>9}{'min':>8}{'max':>8}")
    for k, v in results.items():
        print(f"  {k:<22}{v['sep_mean']:>8.2f}{v['sep_median']:>9.2f}"
              f"{v['sep_min']:>8.2f}{v['sep_max']:>8.2f}")

    print("\n=== 2. does feature distance track real enhancement magnitude? ===")
    print("    Spearman between the ratio above and mean tumour delta\n")
    print(f"  {'feature':<22}{'rho':>8}{'p':>10}")
    for k, v in results.items():
        print(f"  {k:<22}{v['rho']:>+8.3f}{v['p']:>10.4f}")

    print("\n=== 3. domain fit on greyscale MRI ===")
    print(f"  {'feature':<22}{'mean |act|':>12}{'dead frac':>11}")
    for k, v in results.items():
        print(f"  {k:<22}{v['act_mean']:>12.4f}{v['dead']:>11.2f}")

    print("\n=== 4. the test that matches how the loss is used ===")
    print("    d(prediction, ground truth) on breast-masked images, as in")
    print("    pix2pixHD_model.py:268, against the error in mean tumour")
    print("    enhancement. The loss descends this distance, so a rho near zero")
    print("    means descending it says nothing about enhancement accuracy.\n")
    print(f"  {'feature':<22}{'rho':>8}{'p':>10}{'n':>5}")
    for k, v in results.items():
        a = v["align"]
        if a:
            print(f"  {k:<22}{a['rho']:>+8.3f}{a['p']:>10.4f}{a['n']:>5}")
        else:
            print(f"  {k:<22}{'n/a':>8}")

    print("\n=== reading ===")
    vgg = results["VGG-19 (current)"]
    for k, v in results.items():
        if k == "VGG-19 (current)":
            continue
        a, av = v["align"], vgg["align"]
        print(f"  {k} against VGG-19:")
        print(f"    separation       {v['sep_mean']:.2f} vs {vgg['sep_mean']:.2f}"
              f"   {'better' if v['sep_mean'] > vgg['sep_mean'] else 'worse'}")
        print(f"    magnitude rho    {v['rho']:+.3f} vs {vgg['rho']:+.3f}"
              f"   {'better' if abs(v['rho']) > abs(vgg['rho']) else 'worse'}")
        if a and av:
            print(f"    error alignment  {a['rho']:+.3f} vs {av['rho']:+.3f}"
                  f"   {'better' if a['rho'] > av['rho'] else 'worse'}"
                  f"  (p {a['p']:.4f} vs {av['p']:.4f})")
    print("\n  Test 4 carries the most weight: it is the only one measured on")
    print("  synthetic output under the masking the loss actually applies. If no")
    print("  feature space shows a significant positive rho there, then no")
    print("  perceptual loss built on these features supervises enhancement")
    print("  accuracy, and swapping one for another cannot help. That would make")
    print("  the v35 ablation (--no_vgg_loss) the informative experiment and")
    print("  leave in-domain pretraining as the only route that could change it.")

    out = REPO / "scripts" / "dino_probe.json"
    out.write_text(json.dumps({"n_cases": len(cases), "preds": str(preds_dir),
                               "results": results}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
