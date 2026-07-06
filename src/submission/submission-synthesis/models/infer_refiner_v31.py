"""
infer_refiner_v31.py — Inference: v31 GAN + SDEdit refiner.

Pipeline:
  1. pre → per-image normalize → v31 GAN → de-normalize → gan_out
  2. gan_out + noise (strength=0.3) → DDIM 20 steps → refined output

Usage:
    python infer_refiner_v31.py \
        --gan_weights /path/to/mamasynth_v31_pinorm/latest_net_G.pth \
        --refiner_weights /path/to/refiner_v31_pinorm/refiner_latest.pth \
        --input_dir /path/to/test/mha/input \
        --breast_mask_dir /path/to/test/mha/breast_mask \
        --output_dir /path/to/predictions
"""
import argparse
import os
import sys
from pathlib import Path
from functools import partial

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refiner_network import RefinerUNet
from networks import GlobalGenerator


def cosine_beta_schedule(T, s=0.008):
    steps = torch.arange(T + 1, dtype=torch.float64)
    alpha_bar = torch.cos(((steps / T) + s) / (1 + s) * (np.pi / 2)) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clamp(betas, 0.0001, 0.999).float()


class DDIMSampler:
    """DDIM sampler for fast inference (20 steps instead of 1000)."""

    def __init__(self, T=1000, num_steps=20, device='cpu'):
        self.T = T
        self.num_steps = num_steps
        betas = cosine_beta_schedule(T)
        alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(alphas, dim=0).to(device)

        # Subset of timesteps for DDIM
        self.timesteps = torch.linspace(T - 1, 0, num_steps, dtype=torch.long, device=device)

    def add_noise(self, x0, strength=0.3):
        """Add noise at strength fraction of total schedule.

        strength=0.3 means start at t=0.3*T (mild noise).
        """
        t_start = int(strength * self.T)
        t_tensor = torch.tensor([t_start], device=x0.device)
        ab = self.alpha_bar[t_start]
        noise = torch.randn_like(x0)
        noisy = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise
        return noisy, t_start

    def denoise_step(self, model, xt, t_idx, pre, gan_out):
        """Single DDIM denoising step."""
        t = self.timesteps[t_idx]
        t_prev = self.timesteps[t_idx + 1] if t_idx + 1 < self.num_steps else torch.tensor(0, device=xt.device)

        ab_t = self.alpha_bar[t]
        ab_prev = self.alpha_bar[t_prev] if t_prev > 0 else torch.tensor(1.0, device=xt.device)

        # Predict noise
        t_batch = t.unsqueeze(0).expand(xt.shape[0])
        model_input = torch.cat([xt, pre, gan_out], dim=1)
        eps_pred = model(model_input, t_batch)

        # DDIM update (eta=0, deterministic)
        x0_pred = (xt - torch.sqrt(1 - ab_t) * eps_pred) / torch.sqrt(ab_t)
        xt_prev = torch.sqrt(ab_prev) * x0_pred + torch.sqrt(1 - ab_prev) * eps_pred

        return xt_prev

    def sample(self, model, gan_out, pre, strength=0.3):
        """Full DDIM sampling from noised gan_out."""
        # Add noise to gan_out
        xt, t_start = self.add_noise(gan_out, strength)

        # Find starting index in timesteps
        start_idx = 0
        for i, t in enumerate(self.timesteps):
            if t <= t_start:
                start_idx = i
                break

        # Denoise
        with torch.no_grad():
            for i in range(start_idx, self.num_steps - 1):
                xt = self.denoise_step(model, xt, i, pre, gan_out)

        return xt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gan_weights", required=True)
    parser.add_argument("--refiner_weights", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--breast_mask_dir", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--strength", type=float, default=0.3,
                        help="Noise strength for SDEdit (0=no change, 1=full denoise)")
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # Load models
    norm_layer = partial(nn.InstanceNorm2d, affine=False)
    gan = GlobalGenerator(1, 1, 64, 4, 12, norm_layer, residual_mode=True)
    gan.load_state_dict(torch.load(args.gan_weights, map_location=device))
    gan.to(device).eval()

    refiner = RefinerUNet(in_ch=3, out_ch=1, base_ch=64, ch_mult=(1, 2, 4, 4))
    refiner.load_state_dict(torch.load(args.refiner_weights, map_location=device))
    refiner.to(device).eval()

    sampler = DDIMSampler(T=1000, num_steps=args.ddim_steps, device=device)

    print(f"GAN: {args.gan_weights}")
    print(f"Refiner: {args.refiner_weights}")
    print(f"Strength: {args.strength}, DDIM steps: {args.ddim_steps}")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    files = sorted(input_dir.glob("*.mha"))
    print(f"Processing {len(files)} test cases...")

    for f in tqdm(files):
        out_path = output_dir / f.name
        if out_path.exists():
            continue

        # Read input
        img = sitk.ReadImage(str(f))
        arr = sitk.GetArrayFromImage(img).astype(np.float32).squeeze()
        orig_h, orig_w = arr.shape

        # Get breast mask
        if args.breast_mask_dir:
            bm_path = Path(args.breast_mask_dir) / f.name
            if bm_path.exists():
                bm = sitk.GetArrayFromImage(sitk.ReadImage(str(bm_path))).astype(np.float32).squeeze()
                breast_mask = (bm > 0).astype(np.float32)
            else:
                breast_mask = (arr > -0.3).astype(np.float32)
        else:
            breast_mask = (arr > -0.3).astype(np.float32)

        # Per-image normalize (for v31 GAN)
        fg = arr[breast_mask > 0.5]
        if fg.size > 100:
            mu, sigma = float(fg.mean()), max(float(fg.std()), 1e-8)
        else:
            mu, sigma = 0.0, 1.0

        arr_norm = (arr - mu) / sigma * breast_mask

        # To tensor, resize to 512
        pre_t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float().to(device)
        pre_norm_t = torch.from_numpy(arr_norm).unsqueeze(0).unsqueeze(0).float().to(device)
        pre_512 = F.interpolate(pre_t, size=(512, 512), mode='bilinear', align_corners=False)
        pre_norm_512 = F.interpolate(pre_norm_t, size=(512, 512), mode='bilinear', align_corners=False)

        # Step 1: v31 GAN (per-image norm space)
        with torch.no_grad():
            gan_out_norm = gan(pre_norm_512)

        # De-normalize GAN output
        gan_out_512 = gan_out_norm * sigma + mu

        # Step 2: SDEdit refiner
        refined_512 = sampler.sample(refiner, gan_out_512, pre_512, strength=args.strength)

        # Resize back
        result = F.interpolate(refined_512, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        result = result[0, 0].cpu().numpy()

        if sitk.GetArrayFromImage(img).ndim == 3:
            result = result[np.newaxis, ...]

        out_img = sitk.GetImageFromArray(result.astype(np.float32))
        out_img.CopyInformation(img)
        sitk.WriteImage(out_img, str(out_path))

    print(f"Done. {len(list(output_dir.glob('*.mha')))} predictions saved.")


if __name__ == "__main__":
    main()
