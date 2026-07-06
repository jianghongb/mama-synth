"""
train_refiner_v31.py — SDEdit refiner for v31 (per-image normalized GAN).

Key difference from train_refiner.py:
  The frozen GAN (v31) was trained with per-image z-score normalization.
  So when generating gan_out, we must:
    1. Per-image normalize the input (breast foreground mean/std)
    2. Run GAN
    3. De-normalize the output back to original space

The refiner then operates in the ORIGINAL intensity space (same as GT),
learning to clean up GAN artifacts and sharpen tumor boundaries.

Usage:
    python train_refiner_v31.py \
        --dataroot /path/to/data_multislice_v3/train \
        --gan_weights /path/to/mamasynth_v31_pinorm/latest_net_G.pth \
        --breast_mask_dir /path/to/breast_mask \
        --checkpoints_dir /path/to/checkpoints \
        --name refiner_v31_pinorm
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refiner_network import RefinerUNet
from networks import GlobalGenerator
from data.mha_dataset import MhaDataset


def build_gan_v31(weights_path, device):
    """Load frozen v31 GAN (per-image norm, ngf=64, blocks=12)."""
    norm_layer = partial(nn.InstanceNorm2d, affine=False)
    netG = GlobalGenerator(
        input_nc=1, output_nc=1, ngf=64,
        n_downsampling=4, n_blocks=12,
        norm_layer=norm_layer, residual_mode=True,
    )
    netG.load_state_dict(torch.load(weights_path, map_location=device))
    netG.to(device).eval()
    for p in netG.parameters():
        p.requires_grad = False
    return netG


def gan_forward_with_pinorm(gan, pre, breast_mask, device):
    """Run v31 GAN with per-image normalization and de-normalization.

    Args:
        gan: frozen GlobalGenerator
        pre: [B, 1, H, W] input in global z-score space
        breast_mask: [B, 1, H, W] binary breast mask

    Returns:
        gan_out: [B, 1, H, W] in global z-score space (de-normalized)
    """
    B = pre.shape[0]
    outputs = []

    for i in range(B):
        p = pre[i:i+1]      # [1, 1, H, W]
        bm = breast_mask[i:i+1]  # [1, 1, H, W]

        # Per-image normalize (foreground only)
        fg_pixels = p[bm > 0.5]
        if fg_pixels.numel() > 100:
            mu = fg_pixels.mean()
            sigma = fg_pixels.std().clamp(min=1e-8)
        else:
            mu = torch.tensor(0.0, device=device)
            sigma = torch.tensor(1.0, device=device)

        p_norm = (p - mu) / sigma * bm

        # GAN forward
        with torch.no_grad():
            out_norm = gan(p_norm)

        # De-normalize back to global z-score space
        out = out_norm * sigma + mu
        outputs.append(out)

    return torch.cat(outputs, dim=0)


def cosine_beta_schedule(T, s=0.008):
    """Cosine noise schedule."""
    steps = torch.arange(T + 1, dtype=torch.float64)
    alpha_bar = torch.cos(((steps / T) + s) / (1 + s) * (np.pi / 2)) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clamp(betas, 0.0001, 0.999).float()


class DiffusionSchedule:
    def __init__(self, T=1000, device='cpu'):
        self.T = T
        betas = cosine_beta_schedule(T)
        alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(alphas, dim=0).to(device)

    def q_sample(self, x0, t, noise=None):
        """Forward diffusion: add noise at timestep t."""
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self.alpha_bar[t][:, None, None, None]
        return torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise, noise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--gan_weights", required=True)
    parser.add_argument("--breast_mask_dir", default="")
    parser.add_argument("--checkpoints_dir", default="./checkpoints")
    parser.add_argument("--name", default="refiner_v31_pinorm")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument("--save_freq", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=512)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_ids}" if torch.cuda.is_available() else "cpu")

    ckpt_dir = Path(args.checkpoints_dir) / args.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Load frozen v31 GAN
    gan = build_gan_v31(args.gan_weights, device)
    print(f"Loaded frozen v31 GAN from {args.gan_weights}")

    # Refiner: 3ch input [noisy_gt, pre, gan_out] → 1ch noise prediction
    refiner = RefinerUNet(in_ch=3, out_ch=1, base_ch=64, ch_mult=(1, 2, 4, 4)).to(device)
    print(f"Refiner params: {sum(p.numel() for p in refiner.parameters()) / 1e6:.1f}M")

    schedule = DiffusionSchedule(T=args.T, device=device)
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=args.lr, weight_decay=1e-4)

    # Dataset
    class DataOpt:
        dataroot = args.dataroot
        loadSize = args.image_size
        fineSize = args.image_size
        resize_or_crop = 'resize'
        isTrain = True
        no_flip = True
        label_nc = 0
        input_nc = 1
        output_nc = 1
        no_instance = True
        batchSize = args.batch_size
        max_dataset_size = float('inf')
        breast_mask_dir = args.breast_mask_dir
        breast_mask = False
        breast_mask_thresh = -0.3
        intensity_aug = False
        noise_aug = False
        n_downsample_global = 4
        square_only = False
        mask_as_input = False

    opt = DataOpt()
    dataset = MhaDataset()
    dataset.initialize(opt)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=4, pin_memory=True, drop_last=True)

    print(f"Dataset: {len(dataset)} images, {len(dataloader)} batches/epoch")
    print(f"Training for {args.epochs} epochs...")

    # Training loop
    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        refiner.train()
        losses = []

        for batch in tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}"):
            pre = batch['label'].to(device)           # [B, 1, H, W]
            gt = batch['image'].to(device)            # [B, 1, H, W]
            breast_mask = batch['breast_mask'].to(device)  # [B, 1, H, W]

            # Generate GAN output with per-image norm/denorm
            gan_out = gan_forward_with_pinorm(gan, pre, breast_mask, device)

            # Sample random timesteps
            t = torch.randint(0, args.T, (pre.shape[0],), device=device)

            # Forward diffusion on GT
            noisy_gt, noise = schedule.q_sample(gt, t)

            # Refiner input: [noisy_gt, pre, gan_out]
            refiner_input = torch.cat([noisy_gt, pre, gan_out], dim=1)

            # Predict noise
            noise_pred = refiner(refiner_input, t)

            # Loss
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        avg_loss = np.mean(losses)
        print(f"Epoch {epoch}: loss={avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(refiner.state_dict(), ckpt_dir / "refiner_best.pth")

        if epoch % args.save_freq == 0:
            torch.save(refiner.state_dict(), ckpt_dir / f"refiner_epoch{epoch}.pth")
            torch.save(refiner.state_dict(), ckpt_dir / "refiner_latest.pth")

    torch.save(refiner.state_dict(), ckpt_dir / "refiner_latest.pth")
    print(f"Training complete. Best loss: {best_loss:.6f}")


if __name__ == "__main__":
    main()
