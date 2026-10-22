"""Put the trained models next to the baselines that bound each metric.

Over 130 evaluations are recorded in this project and none of them was a
baseline, so every metric has been read against other trained models and none
has a known floor. Two readings have already been retracted for exactly that
reason: AUROC_contrast, where the two highest-scoring runs had the two worst
SSIM and MSE values, and FRD, where a run that collapsed to the identity map
scored 10.47 against 25.8-28.7 for every model that synthesises anything.

This script reports each metric against its floor rather than in isolation, and
answers the specific question each baseline was built for.

Run:
    python scripts/compare_baselines.py
"""

import json
from pathlib import Path

MAMA = Path("/Users/ehogjig/git/kth/mama-synth")
DSB = Path("/Users/ehogjig/git/kth/DSB_for_chanllenge")

BASELINES = [
    ("identity", "pre-contrast copied", MAMA),
    ("const_offset", "breast +0.775 sigma, no spatial model", MAMA),
    ("oracle_const", "true mask +3.75 sigma, no texture", MAMA),
    ("gt_ceiling", "ground truth as prediction", MAMA),
]

MODELS = [
    ("v32_no_msec", "GAN row (5)", MAMA),
    ("v32_final", "GAN submission", MAMA),
    ("v34_seg", "GAN + seg-consistency", MAMA),
    ("v35_novgg", "GAN without VGG", MAMA),
    ("i2sb_v3b", "I2SB bridge, 50 steps", DSB),
    ("inr_base", "INR SIREN (collapsed)", DSB),
]

KEYS = ["mse", "lpips", "ssim_tumor", "frd", "dice", "hausdorff_95",
        "auroc_contrast"]
LOWER_BETTER = {"mse", "lpips", "frd", "hausdorff_95"}


def load(name, root):
    for cand in (root / f"eval_results_{name}" / "metrics.json",
                 root / f"eval_{name}" / "metrics.json"):
        if cand.exists():
            return json.load(open(cand))
    return None


def agg(m, key):
    if m is None:
        return None
    a = m.get("aggregates", {}).get(key)
    if a is None:
        return None
    return a.get("mean") if isinstance(a, dict) else a


def fmt(v, key):
    if v is None:
        return "—"
    return f"{v:.3f}" if key != "hausdorff_95" else f"{v:.1f}"


def row(label, m, width=30):
    cells = "".join(f"{fmt(agg(m, k), k):>12}" for k in KEYS)
    return f"  {label:<{width}}{cells}"


def main():
    hdr = "".join(f"{k.replace('hausdorff_95','HD95').replace('auroc_contrast','AUROC'):>12}"
                  for k in KEYS)
    print(f"\n  {'':<30}{hdr}")
    print(f"  {'-'*30}{'-'*(12*len(KEYS))}")

    loaded = {}
    print("  BASELINES")
    for name, desc, root in BASELINES:
        m = load(name, root)
        loaded[name] = m
        print(row(f"  {desc}", m))
    print("\n  MODELS")
    for name, desc, root in MODELS:
        m = load(name, root)
        loaded[name] = m
        print(row(f"  {desc}", m))

    ident = loaded.get("identity")
    oracle = loaded.get("oracle_const")
    ceil = loaded.get("gt_ceiling")
    const = loaded.get("const_offset")

    print("\n\n=== what each baseline settles ===")

    if ident:
        print("\n1. Which metrics does the identity map already win?")
        print("   A trained model that cannot beat a copy of its own input is not")
        print("   measuring synthesis on that metric.\n")
        print(f"   {'metric':<16}{'identity':>12}{'GAN row(5)':>14}{'I2SB v3b':>12}  verdict")
        gan, i2sb = loaded.get("v32_no_msec"), loaded.get("i2sb_v3b")
        for k in KEYS:
            vi, vg, vb = agg(ident, k), agg(gan, k), agg(i2sb, k)
            if vi is None:
                continue
            better = []
            for nm, v in (("GAN", vg), ("I2SB", vb)):
                if v is None:
                    continue
                wins = (v < vi) if k in LOWER_BETTER else (v > vi)
                if not wins:
                    better.append(nm)
            note = ("identity beats " + "+".join(better)) if better else "both beat identity"
            print(f"   {k:<16}{fmt(vi,k):>12}{fmt(vg,k):>14}{fmt(vb,k):>12}  {note}")

    if oracle:
        print("\n2. Do Dice and HD95 measure synthesis, or only localisation?")
        print("   oracle_const is a flat block of the mean enhancement placed in")
        print("   the true mask: perfect localisation, no texture, no model.\n")
        for k in ("dice", "hausdorff_95", "ssim_tumor", "mse"):
            vo = agg(oracle, k)
            vg = agg(loaded.get("v32_no_msec"), k)
            if vo is None or vg is None:
                continue
            print(f"   {k:<16}oracle_const {fmt(vo,k):>8}   GAN row(5) {fmt(vg,k):>8}")
        vo, vg = agg(oracle, "dice"), agg(loaded.get("v32_no_msec"), "dice")
        if vo and vg:
            print(f"\n   oracle_const reaches {vo/vg:.0%} of the GAN's Dice while")
            print("   synthesising nothing. The closer that is to 100 %, the more")
            print("   Dice and HD95 describe mask placement rather than image")
            print("   quality, and the more carefully the GAN's Dice advantage over")
            print("   I2SB has to be worded.")

    if const:
        print("\n3. How much of the gain needs spatial modelling?")
        print("   const_offset adds one number everywhere inside the breast.\n")
        for k in ("mse", "lpips", "ssim_tumor", "dice"):
            vc, vi = agg(const, k), agg(ident, k) if ident else None
            vg = agg(loaded.get("v32_no_msec"), k)
            if vc is None:
                continue
            print(f"   {k:<16}identity {fmt(vi,k):>8}   const {fmt(vc,k):>8}   "
                  f"GAN {fmt(vg,k):>8}")

    if ceil:
        print("\n4. Where is AUROC_contrast's ceiling?")
        va = agg(ceil, "auroc_contrast")
        vb = agg(loaded.get("i2sb_v3b"), "auroc_contrast")
        vi = agg(ident, "auroc_contrast") if ident else None
        print(f"\n   ground truth {fmt(va,'auroc_contrast')}   "
              f"I2SB v3b {fmt(vb,'auroc_contrast')}   identity {fmt(vi,'auroc_contrast')}")
        if va and vb and vi:
            span = va - vi
            got = (vb - vi) / span if span else float("nan")
            print(f"   I2SB covers {got:.0%} of the usable range ({vi:.3f} to {va:.3f}).")
            print("   Without the ceiling there was no way to tell whether 0.760")
            print("   was close to the maximum or far from it.")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
