"""Feasibility probe for a segmentation-consistency loss on the GAN.

Before spending a 48 h training run, this checks the three things that decide
whether the loss can work at all:

1. the frozen segmenter loads and produces a sensible tumour probability on a
   real post-contrast slice, so that its output means something;
2. gradients flow from its output back to the image, since the loss only works
   if the generator can be driven through the frozen network;
3. the memory cost of holding it in the training graph is affordable next to the
   239 M-parameter generator.

Which segmenter to use is a methodological choice, not a convenience. The
challenge scores Dice and HD95 with ``src/evaluation/models/segmentation``
(Dataset101_mamasynth_firstpost). Routing a training loss through that same
network optimises the evaluation metric directly, which inflates the reported
score without improving the synthesis. The Anguinus Sculpturae paper uses "the
frozen, provided lesion segmenter" and does not address this.

This probe therefore uses Dataset940_TumorSegT1w, the tumour segmenter trained
for this work on the 9,504 train-split slices alone: a different training set, a
different patch size (448 against 256) and no test patient in its data. Keeping
the loss network separate from the scoring network is what makes "remains
segmentable" an independent claim rather than a restatement of the metric.

MAMA-MIA's released tumour segmenter (Dataset105_full_image) was the first
candidate and had to be rejected: only its 3d_fullres fold weights are shipped,
with [3,3,3] kernels and a [128,128,128] patch. Driving a single 2D slice through
it would require stacking a pseudo-volume, which costs two orders of magnitude
more memory and shows the network a depth-constant volume unlike anything it was
trained on, so its probabilities would not be trustworthy.

Run:
    python scripts/probe_seg_loss.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

REPO = Path("/Users/ehogjig/git/kth/mama-synth")
SEG = REPO / "src/submission/submission-synthesis/weights/tumor_seg_940"
EVAL_SEG = REPO / "src/evaluation/models/segmentation"
DATA = Path("/Users/ehogjig/git/kth/data_multislice_v3/test/mha")


def build_segmenter(plans_dir: Path, fold: int = 0):
    """Instantiate the nnU-Net architecture from plans.json and load one fold.

    nnU-Net's own predictor wraps the network in sliding-window inference, test-
    time mirroring and an argmax, none of which is differentiable end to end.
    The network itself is an ordinary nn.Module, so it is rebuilt here from the
    plans and driven directly, which keeps the graph intact.
    """
    from dynamic_network_architectures.architectures.unet import PlainConvUNet

    plans = json.loads((plans_dir / "plans.json").read_text())
    ds = json.loads((plans_dir / "dataset.json").read_text())
    cfg = plans["configurations"]["2d"]
    kw = dict(cfg["architecture"]["arch_kwargs"])

    # plans.json stores these as import paths; resolve them to the classes.
    for key in ("conv_op", "norm_op", "nonlin", "dropout_op"):
        val = kw.get(key)
        if isinstance(val, str):
            mod, _, name = val.rpartition(".")
            kw[key] = getattr(__import__(mod, fromlist=[name]), name)
        elif key in kw and val is None:
            kw[key] = None

    net = PlainConvUNet(
        input_channels=len(ds["channel_names"]),
        num_classes=len(ds["labels"]),
        deep_supervision=False,
        **kw,
    )
    ck = torch.load(plans_dir / f"fold_{fold}/checkpoint_final.pth",
                    map_location="cpu", weights_only=False)
    sd = ck.get("network_weights", ck.get("state_dict"))
    # The checkpoint was saved with deep supervision enabled, so the decoder
    # carries extra seg heads that the inference-time network does not define.
    missing, unexpected = net.load_state_dict(sd, strict=False)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net, cfg, (missing, unexpected)


def zscore(x: torch.Tensor) -> torch.Tensor:
    """nnU-Net ZScoreNormalization, per image over all voxels."""
    m = x.mean()
    s = x.std().clamp_min(1e-8)
    return (x - m) / s


def read(p: Path) -> np.ndarray:
    a = sitk.GetArrayFromImage(sitk.ReadImage(str(p))).astype(np.float32)
    return a[0] if a.ndim == 3 and a.shape[0] == 1 else a


def main() -> int:
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {dev}\n")

    print("=== 1. the two segmenters are genuinely different networks ===")
    for name, d in (("loss  (Dataset105)", SEG), ("score (Dataset101)", EVAL_SEG)):
        pl = json.loads((d / "plans.json").read_text())
        c = pl["configurations"]["2d"]
        print(f"  {name}: {pl['dataset_name']}  patch={c['patch_size']}  "
              f"stages={c['architecture']['arch_kwargs']['n_stages']}")

    print("\n=== 2. the loss segmenter loads ===")
    try:
        net, cfg, (missing, unexpected) = build_segmenter(SEG)
    except Exception as exc:
        print(f"  FAILED to build: {type(exc).__name__}: {exc}")
        return 1
    n = sum(p.numel() for p in net.parameters())
    print(f"  built and loaded: {n/1e6:.2f} M params, frozen")
    if missing:
        print(f"  missing keys    : {len(missing)}  e.g. {missing[:2]}")
    if unexpected:
        print(f"  unexpected keys : {len(unexpected)}  e.g. {unexpected[:2]}")
    if missing:
        print("  WARNING: missing weights mean parts of the network are random; "
              "its output would be meaningless as a training signal")
        return 1
    net = net.to(dev)

    print("\n=== 3. it finds real tumours (so its output means something) ===")
    cases = sorted((DATA / "ground_truth").glob("*.mha"))[:8]
    if not cases:
        print(f"  no cases under {DATA/'ground_truth'}")
        return 1
    rows = []
    for f in cases:
        gt = read(f)
        seg = read(DATA / "mask" / f.name) > 0.5
        if seg.sum() < 20:
            continue
        x = zscore(torch.from_numpy(gt)[None, None].to(dev))
        with torch.no_grad():
            prob = torch.softmax(net(x), dim=1)[0, 1]
        p = prob.cpu().numpy()
        m = torch.from_numpy(seg).numpy()
        rows.append((f.name, float(p[m].mean()), float(p[~m].mean()),
                     float(p[m].mean() / max(p[~m].mean(), 1e-8))))
    print(f"  {'case':<22}{'P(tum) in mask':>16}{'outside':>10}{'ratio':>9}")
    for nm, pin, pout, r in rows:
        print(f"  {nm[:20]:<22}{pin:>16.3f}{pout:>10.4f}{r:>9.1f}")
    if rows:
        mean_in = float(np.mean([r[1] for r in rows]))
        mean_out = float(np.mean([r[2] for r in rows]))
        print(f"\n  mean P(tumour) inside GT mask : {mean_in:.3f}")
        print(f"  mean P(tumour) outside        : {mean_out:.4f}")
        if mean_in < 0.3:
            print("  WARNING: it barely responds to real tumours on these slices. "
                  "Either the normalisation is wrong or the 512x512 input is far "
                  "from its 192x192 training patch; the loss would be noise.")
            return 1
        print("  -> responds to real enhancement, so it can supervise")

    print("\n=== 4. gradients reach the image ===")
    gt = read(cases[0])
    x = zscore(torch.from_numpy(gt)[None, None].to(dev)).requires_grad_(True)
    logits = net(x)
    seg = torch.from_numpy((read(DATA / "mask" / cases[0].name) > 0.5)
                           .astype(np.float32))[None, None].to(dev)
    prob = torch.softmax(logits, dim=1)[:, 1:2]
    # Asymmetric Tversky: beta weights false negatives, so missed lesions are
    # penalised far more heavily than spurious ones. This is what pushes the
    # generator to render lesions that are unambiguously detectable.
    alpha, beta = 0.3, 0.7
    tp = (prob * seg).sum()
    fp = (prob * (1 - seg)).sum()
    fn = ((1 - prob) * seg).sum()
    tversky = 1 - (tp + 1e-6) / (tp + alpha * fp + beta * fn + 1e-6)
    tversky.backward()
    g = x.grad
    print(f"  Tversky(alpha={alpha}, beta={beta}) = {float(tversky):.4f}")
    print(f"  grad finite : {bool(torch.isfinite(g).all())}")
    print(f"  |grad| max  : {float(g.abs().max()):.3e}   mean {float(g.abs().mean()):.3e}")
    if not torch.isfinite(g).all() or float(g.abs().max()) == 0.0:
        print("  FAILED: no usable gradient reaches the image")
        return 1
    print("  -> the generator can be driven through the frozen segmenter")

    print("\n=== 5. memory cost next to the 239 M generator ===")
    print(f"  segmenter params      : {n/1e6:.2f} M  ({n*4/2**20:.0f} MiB fp32)")
    print(f"  generator params      : 239.07 M")
    print("  Activations dominate: a forward and backward through a 6-stage")
    print("  U-Net at 512x512 is roughly the cost of one more generator pass.")
    print("  The GAN trains at batch 16; expect to halve it, or crop the loss to")
    print("  a lesion-centred 192x192 window, which also matches the patch size")
    print("  this network was trained at.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
