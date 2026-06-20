"""
train_residual_refiner.py — Stage 2: Train residual refiner on top of frozen GAN.

The refiner directly predicts the residual Δ, and the final output = pre + Δ.
Input: [pre, gan_output] (2ch)
Output: Δ (1ch), trained with L1 loss against (gt - pre)

Usage:
    python train_residual_refiner.py \
        --dataroot /path/to/data/train \
        --gan_weights /path/to/latest_net_G.pth \
        --checkpoints_dir /path/to/checkpoints \
        --name resrefiner_v1 \
        --epochs 100
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
from networks import GlobalGenerator
from data.mha_dataset import MhaDataset


class ResidualRefiner(nn.Module):
    """Simple UNet that predicts residual Δ. Output = pre + Δ.
    
    Input: 2ch [pre, gan_output]
    Output: 1ch Δ (residual enhancement)
    """
    def __init__(self, base_ch=64):
        super().__init__()
        # Encoder
        self.enc1 = self._block(2, base_ch)
        self.enc2 = self._block(base_ch, base_ch * 2)
        self.enc3 = self._block(base_ch * 2, base_ch * 4)
        self.enc4 = self._block(base_ch * 4, base_ch * 8)

        # Bottleneck
        self.bottleneck = self._block(base_ch * 8, base_ch * 8)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_ch * 8, base_ch * 8, 2, 2)
        self.dec4 = self._block(base_ch * 16, base_ch * 4)
        self.up3 = nn.ConvTranspose2d(base_ch * 4, base_ch * 4, 2, 2)
        self.dec3 = self._block(base_ch * 8, base_ch * 2)
        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 2, 2)
        self.dec2 = self._block(base_ch * 4, base_ch)
        self.up1 = nn.ConvTranspose2d(base_ch, base_ch, 2, 2)
        self.dec1 = self._block(base_ch * 2, base_ch)

        # Output: predict residual
        self.out_conv = nn.Conv2d(base_ch, 1, 1)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, pre, gan_output):
        """
        Args:
            pre: (B, 1, H, W) pre-contrast
            gan_output: (B, 1, H, W) GAN synthesis
        Returns:
            (B, 1, H, W) refined output = pre + predicted_residual
        """
        x = torch.cat([pre, gan_output], dim=1)  # (B, 2, H, W)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        e4 = self.enc4(F.avg_pool2d(e3, 2))

        # Bottleneck
        b = self.bottleneck(F.avg_pool2d(e4, 2))

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        # Predict residual and add to pre
        delta = self.out_conv(d1)
        return pre + delta


def build_gan(weights_path, device):
    norm_layer = partial(nn.InstanceNorm2d, affine=False)
    netG = GlobalGenerator(1, 1, 64, 4, 9, norm_layer, residual_mode=True)
    netG.load_state_dict(torch.load(weights_path, map_location=device))
    netG.to(device).eval()
    for p in netG.parameters():
        p.requires_grad = False
    return netG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--gan_weights", required=True)
    parser.add_argument("--checkpoints_dir", default="./checkpoints")
    parser.add_argument("--name", default="resrefiner_v1")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument("--save_freq", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=512)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_ids}" if torch.cuda.is_available() else "cpu")

    ckpt_dir = Path(args.checkpoints_dir) / args.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Models
    gan = build_gan(args.gan_weights, device)
    refiner = ResidualRefiner(base_ch=64).to(device)
    print(f"Refiner params: {sum(p.numel() for p in refiner.parameters()) / 1e6:.1f}M")

    optimizer = torch.optim.Adam(refiner.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Dataset
    class SimpleOpt:
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
        breast_mask_dir = ''
        intensity_aug = False
        n_downsample_global = 4
        square_only = False

    opt = SimpleOpt()
    dataset = MhaDataset()
    dataset.initialize(opt)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=4, pin_memory=True, drop_last=True)
    print(f"Dataset: {len(dataset)} images")

    # Training
    for epoch in range(1, args.epochs + 1):
        refiner.train()
        losses = []

        for batch in tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}"):
            pre = batch['label'].to(device)   # (B, 1, H, W)
            gt = batch['image'].to(device)    # (B, 1, H, W)

            # Frozen GAN output
            with torch.no_grad():
                gan_out = gan(pre)

            # Refiner: predict output = pre + Δ
            output = refiner(pre, gan_out)

            # L1 loss against GT
            loss = F.l1_loss(output, gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        avg_loss = np.mean(losses)
        print(f"Epoch {epoch}: L1={avg_loss:.6f}")

        if epoch % args.save_freq == 0:
            torch.save(refiner.state_dict(), ckpt_dir / f"refiner_epoch{epoch}.pth")
            torch.save(refiner.state_dict(), ckpt_dir / "refiner_latest.pth")

    torch.save(refiner.state_dict(), ckpt_dir / "refiner_latest.pth")
    print(f"Done. Weights: {ckpt_dir / 'refiner_latest.pth'}")


if __name__ == "__main__":
    main()
