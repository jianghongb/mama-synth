# MAMA-SYNTH — Pix2PixHD + MSSC Synthesis Baseline

Pix2PixHD GlobalGenerator with Multi-Scale Subtraction Consistency (MSSC) loss.

## Model

- Architecture: GlobalGenerator (4↓ + 9 ResNet Blocks + 4↑), ngf=64
- Mode: Residual (output = pre + Δ)
- Training: 938 axial cases, 200 epochs, MSSC + VGG + GAN + Tumor L1

## Build & Test

```bash
./do_build.sh           # Build Docker image
./do_test_run.sh        # Test with sample input
./do_save.sh v1.0.0     # Export .tar.gz for GC upload
```

## Weights

Place `latest_net_G.pth` in `weights/` before building. Not committed to git (696MB).
