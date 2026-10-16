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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="number of cases")
    args = ap.parse_args()

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cases = collect_cases(args.n)
    if not cases:
        print("no usable cases found")
        return 1
    print(f"device: {dev}   cases: {len(cases)}\n")

    results = {}
    for Extractor in (VggFeat, DinoFeat):
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
        results[ex.name] = dict(
            sep_mean=float(sep.mean()), sep_median=float(np.median(sep)),
            sep_min=float(sep.min()), sep_max=float(sep.max()),
            rho=float(rho), p=float(p), act_mean=float(np.mean(act)),
            dead=float(np.mean([a < 1e-6 for a in act])))
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

    vgg, dino = results["VGG-19 (current)"], results["DINOv2 ViT-S/14"]
    print("\n=== reading ===")
    better_sep = dino["sep_mean"] > vgg["sep_mean"]
    better_rho = abs(dino["rho"]) > abs(vgg["rho"])
    print(f"  DINO separates enhancement better : {better_sep} "
          f"({dino['sep_mean']:.2f} vs {vgg['sep_mean']:.2f})")
    print(f"  DINO tracks magnitude better      : {better_rho} "
          f"(rho {dino['rho']:+.3f} vs {vgg['rho']:+.3f})")
    if better_sep and better_rho:
        print("  -> a DINO perceptual loss is worth one ablation run")
    elif not better_sep and not better_rho:
        print("  -> DINO features are not more informative about enhancement here.")
        print("     Swapping the perceptual loss would not be motivated, and an")
        print("     in-domain pretraining effort is the only route left that could")
        print("     change this.")
    else:
        print("  -> mixed. The two criteria disagree, so the case is not clear and")
        print("     the cheaper experiment is still the perceptual-loss swap.")

    out = REPO / "scripts" / "dino_probe.json"
    out.write_text(json.dumps({"n_cases": len(cases), "results": results}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
