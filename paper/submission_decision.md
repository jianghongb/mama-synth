# v32 Submission Decision — Supervisor Discussion

## Two Candidate Approaches

### Option A: v32 Final (1ch + heart offset)
- **Model**: v32_vflip, 1-channel GAN (input = pre-contrast only)
- **Post-processing**: Heart offset (Dataset910 label=7) + hard breast composite + hflip TTA
- **Strength**: MSE/LPIPS optimal — pixel-level and perceptual accuracy

### Option B: v32 Conditional (3ch, blur + TTA)
- **Model**: v32_conditional, 3-channel GAN (input = pre + breast_mask + tumor_mask)
- **Post-processing**: Soft blur composite (σ=5) + hflip TTA, no heart offset
- **Strength**: Dice/HD95 optimal — downstream segmentation quality

---

## Metric Comparison (internal test set, 299 cases)

| Metric | Option A (1ch+heart) | Option B (3ch+blur) | Δ | Winner |
|--------|:---:|:---:|---|---|
| MSE ↓ | **0.970** | 1.127 | -14% | A |
| LPIPS ↓ | **0.116** | 0.123 | -6% | A |
| SSIM_tumor ↑ | 0.493 | **0.509** | +3% | B |
| FRD ↓ | **26.88** | 27.06 | -1% | A (marginal) |
| Dice ↑ | 0.567 | **0.624** | +10% | B |
| HD95 ↓ | 108.5 | **69.4** | -36% | B |

Score: **A wins 3 metrics, B wins 3 metrics**.

---

## GC Ranking Analysis (Mean Position across 8 metrics)

GC ranks each metric independently, then takes the arithmetic mean of ranks.

### Estimated ranks based on validation leaderboard (#1 has MSE=0.57, Dice=0.48, HD95=120.6):

| Metric | Option A est. rank | Option B est. rank | Notes |
|--------|:---:|:---:|---|
| MSE ↓ | ~20-25 | ~30-35 | A benefits from heart offset |
| LPIPS ↓ | ~25-30 | ~30-35 | Both behind top (0.08) |
| SSIM_tumor ↑ | ~15-20 | ~10-15 | Both beat GC #1 (0.43) |
| FRD ↓ | ~30-35 | ~30-35 | Similar, both ~27 vs top 25 |
| AUROC_contrast | ~20-25 | ~20-25 | Unknown, estimated |
| AUROC_tumor | ~15-20 | ~15-20 | Unknown, estimated |
| Dice ↑ | ~**3-5** | ~**1-2** | GC #1 only 0.48, we have 0.57/0.62 |
| HD95 ↓ | ~**3-5** | ~**1** | GC #1 has 120.6, we have 108/69 |

### Estimated Mean Position:

| | Option A | Option B |
|---|:---:|:---:|
| Sum of est. ranks | ~135-165 | ~140-170 |
| Mean Position (÷8) | **~17-20** | **~18-21** |

**Both are competitive for top 3.** The difference is small and depends heavily on actual competitor distribution.

---

## Strategic Analysis

### Why Option A might win:
1. **No extreme weakness** — MSE rank ~20 instead of ~35; avoids being punished on pixel metrics
2. **Heart offset is novel** — corrects systematic error in cardiac region that all other methods likely have
3. **Simpler pipeline** — 1ch GAN, less chance of inference bugs on unseen test data
4. **FRD slightly better** — radiomics realism matters for clinical interpretation

### Why Option B might win:
1. **Dice/HD95 dominance** — could be rank 1 on both (vs current #1's rank 3)
2. **These two metrics won the championship** for MamoAnd (their best ranks: Dice=3, HD95=3)
3. **Tumor-aware synthesis** — clinically more meaningful (enhancement WHERE it matters)
4. **HD95 69 vs 108** is a massive gap — this alone could swing 5+ positions

### Key insight from leaderboard:
> **MamoAnd won (#1, Mean Position 16.8) with Dice rank 3 + HD95 rank 3.**
> Their MSE rank was only 10 and SSIM rank was 36.
> → Dice and HD95 have less competition at the top (fewer teams optimize for them)
> → A dominant Dice/HD95 rank (1-2) provides more ranking value than moderate MSE improvement

---

## Risk Assessment

| Risk | Option A | Option B |
|---|---|---|
| Domain shift (unseen test data) | Medium — heart offset tuned on internal data | Medium — tumor seg may fail on new data |
| Pipeline complexity | Low (1ch GAN + 2 seg models) | Higher (1ch GAN + 3 seg models, 3ch concat) |
| Inference time | ~5s | ~8s |
| Failure mode | Heart offset wrong → MSE spike on some cases | Tumor mask wrong → Dice drops slightly |
| Docker size | ~2.5 GB | ~3.5 GB |

---

## Recommendation

**If forced to choose one: Option B (3ch conditional + blur + TTA)**

Reasoning:
1. MamoAnd's winning strategy proves Dice/HD95 dominance wins championships
2. Option B could be rank 1 on Dice AND HD95 (unprecedented advantage)
3. MSE gap (0.97 vs 1.13) only costs ~10 rank positions, but HD95 gap (108 vs 69) saves ~5 positions
4. Net benefit of Option B ≈ +2-3 Mean Position improvement on segmentation metrics vs -2-3 on pixel metrics → roughly neutral with upside

**However**: if the test set has many cases without tumors or with failed tumor segmentation, Option A is safer.

---

## Can we submit both?

**No — GC only allows one submission.**

---

## Final Recommendation

### **→ Submit Option B (3ch conditional + blur + TTA)**

| Factor | Reasoning |
|--------|-----------|
| **Ranking math** | rank 1 × 2 (Dice+HD95) saves ~4 Mean Position; MSE drop costs ~1.25 → net +2.7 |
| **Leaderboard proof** | MamoAnd won with Dice rank 3 + HD95 rank 3. We have 0.62 and 69.4 — likely rank 1 on both |
| **Competition density** | MSE/LPIPS: many teams cluster around 0.5-1.0 (dense, hard to rank high). Dice/HD95: sparse at top (easy rank 1) |
| **Clinical relevance** | Tumor segmentation quality (Dice/HD95) is more clinically meaningful than pixel MSE |
| **Paper value** | "Conditional GAN with predicted tumor mask" is a more compelling contribution than "heart offset postprocessing" |
| **Risk** | If tumor seg fails on some test cases, Dice may drop to ~0.55 (still beats GC #1's 0.48) |

### What we sacrifice:
- MSE: 0.970 → 1.127 (+16%, ~10 rank positions)
- LPIPS: 0.116 → 0.123 (+6%, ~5 rank positions)

### What we gain:
- Dice: 0.567 → 0.624 (+10%, ~2-3 rank positions saved at the top)
- HD95: 108.5 → 69.4 (-36%, ~3-5 rank positions saved at the top)
- SSIM: 0.493 → 0.509 (+3%, ~5 rank positions)

**Net expected improvement: +2-3 Mean Position → could be the difference between top 3 and top 5.**

---

## Summary for Discussion

| Question | Answer |
|---|---|
| Which has better MSE? | Option A (-14%) |
| Which has better Dice? | Option B (+10%) |
| Which would win GC? | **Option B** — Dice/HD95 dominance at rank 1 outweighs MSE weakness |
| Which is safer? | Option A (simpler, less dependency on tumor seg) |
| Which is more novel? | Option B (conditional on predicted tumor mask) |
| Which is better for the paper? | Option B (more interesting method to describe) |
| **Final choice** | **Option B (3ch conditional + blur + hflip TTA)** |
