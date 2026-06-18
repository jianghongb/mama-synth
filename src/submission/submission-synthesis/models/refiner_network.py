"""
refiner_network.py — Lightweight UNet denoiser for SDEdit refinement.

Input: [noisy_image, pre_contrast, gan_output] (3ch) + timestep embedding
Output: predicted noise ε (1ch)
~18M params at base_ch=64.
"""
import math
import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(ch, ch, 3, 1, 1))

    def forward(self, x):
        return self.conv(x)


class RefinerUNet(nn.Module):
    """Compact UNet denoiser (~18M params).

    Args:
        in_ch: input channels (3 = noisy + pre + gan_output)
        out_ch: output channels (1 = predicted noise)
        base_ch: base channel width
        ch_mult: channel multipliers per level
    """

    def __init__(self, in_ch=3, out_ch=1, base_ch=64, ch_mult=(1, 2, 4, 4)):
        super().__init__()
        time_dim = base_ch * 4

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(base_ch),
            nn.Linear(base_ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Input projection
        self.input_conv = nn.Conv2d(in_ch, base_ch, 3, 1, 1)

        # Encoder
        self.encoder = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        channels = [base_ch]
        ch = base_ch
        for mult in ch_mult:
            out = base_ch * mult
            self.encoder.append(ResBlock(ch, out, time_dim))
            self.encoder.append(ResBlock(out, out, time_dim))
            channels.append(out)
            ch = out
            self.downsamples.append(Downsample(ch))

        # Bottleneck
        self.mid1 = ResBlock(ch, ch, time_dim)
        self.mid2 = ResBlock(ch, ch, time_dim)

        # Decoder
        self.decoder = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for mult in reversed(ch_mult):
            out = base_ch * mult
            self.upsamples.append(Upsample(ch))
            self.decoder.append(ResBlock(ch + channels.pop(), out, time_dim))  # skip cat
            self.decoder.append(ResBlock(out, out, time_dim))
            ch = out

        # Output
        self.out_norm = nn.GroupNorm(8, ch)
        self.out_conv = nn.Conv2d(ch, out_ch, 3, 1, 1)

    def forward(self, x, t):
        """
        Args:
            x: (B, 3, H, W) — [noisy_image, pre_contrast, gan_output]
            t: (B,) — timestep indices
        Returns:
            (B, 1, H, W) — predicted noise
        """
        t_emb = self.time_embed(t)
        h = self.input_conv(x)

        # Encoder
        skips = [h]
        for i in range(0, len(self.encoder), 2):
            h = self.encoder[i](h, t_emb)
            h = self.encoder[i + 1](h, t_emb)
            skips.append(h)
            h = self.downsamples[i // 2](h)

        # Bottleneck
        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        # Decoder
        for i in range(0, len(self.decoder), 2):
            h = self.upsamples[i // 2](h)
            h = torch.cat([h, skips.pop()], dim=1)
            h = self.decoder[i](h, t_emb)
            h = self.decoder[i + 1](h, t_emb)

        return self.out_conv(nn.functional.silu(self.out_norm(h)))
