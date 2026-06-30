"""Semi-Disentangled Spatiotemporal Generator for DCE-MRI Synthesis.

Core idea from Implicit Neural Representations (INR):
    I(x,y,t) = anatomy(x,y) + enhancement(x,y) * dynamics(t)

For MAMA-SYNTH (fixed t=peak), this simplifies to:
    post(x,y) = pre(x,y) + E(x,y)

where E(x,y) is the spatially-varying enhancement map.

The key insight is SEMI-DISENTANGLEMENT:
- Static pathway: locks the anatomy (pre-contrast) as an identity pass-through
- Dynamic pathway: a dedicated sub-network learns ONLY the enhancement delta,
  gated by a spatial attention mask that focuses compute on enhancing regions

This eliminates hallucinated lesions in non-enhancing tissue and concentrates
all generative capacity on the contrast-uptake dynamics.

Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                     SemiDisentangledGenerator                        │
    │                                                                     │
    │  pre ──┬──────────────────────────────────────────── (identity) ──┐ │
    │        │                                                          + │
    │        └── SharedEncoder ──┬── EnhancementDecoder → E(x,y)        │ │
    │                            │                          ×            │ │
    │                            └── GateDecoder → σ(G(x,y)) ──────────┘ │
    │                                                                     │
    │  output = pre + gate * enhancement                                  │
    │                                                                     │
    │  gate ∈ [0,1]: spatial attention, learns where enhancement happens  │
    │  enhancement ∈ ℝ: unbounded residual intensity change               │
    └─────────────────────────────────────────────────────────────────────┘

Loss design for semi-disentanglement:
    L_total = L_enhance + λ_anatomy * L_anatomy + λ_gate * L_gate_reg

    L_enhance:  standard GAN + VGG + L1 on (pre + gate*E) vs gt
    L_anatomy:  L1(output * (1-breast_mask), pre * (1-breast_mask))  [force bg=0 change]
    L_gate_reg: sparsity prior on gate (most tissue should NOT enhance)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import numpy as np


class ResnetBlock(nn.Module):
    """Standard ResNet block with reflection padding."""

    def __init__(self, dim, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0),
            norm_layer(dim),
            activation,
        ]
        if use_dropout:
            layers += [nn.Dropout(0.5)]
        layers += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0),
            norm_layer(dim),
        ]
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv_block(x)


class SharedEncoder(nn.Module):
    """Shared encoder that extracts multi-scale features from pre-contrast input.

    Outputs bottleneck features + skip connections for the two decoder heads.
    """

    def __init__(self, input_nc=1, ngf=64, n_downsampling=4, n_blocks=6,
                 norm_layer=nn.InstanceNorm2d):
        super().__init__()
        activation = nn.ReLU(True)

        # Initial conv
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            norm_layer(ngf),
            activation,
        )

        # Downsampling layers (store for skip connections)
        self.down_layers = nn.ModuleList()
        for i in range(n_downsampling):
            mult = 2 ** i
            self.down_layers.append(nn.Sequential(
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                norm_layer(ngf * mult * 2),
                activation,
            ))

        # Bottleneck ResNet blocks
        mult = 2 ** n_downsampling
        bottleneck_layers = []
        for _ in range(n_blocks):
            bottleneck_layers.append(
                ResnetBlock(ngf * mult, norm_layer=norm_layer, activation=activation)
            )
        self.bottleneck = nn.Sequential(*bottleneck_layers)

        self.bottleneck_dim = ngf * mult
        self.n_downsampling = n_downsampling
        self.ngf = ngf

    def forward(self, x):
        """Returns (bottleneck_feat, skip_features_list)."""
        skips = []
        feat = self.initial(x)
        skips.append(feat)

        for down in self.down_layers:
            feat = down(feat)
            skips.append(feat)

        feat = self.bottleneck(feat)
        return feat, skips


class EnhancementDecoder(nn.Module):
    """Decoder that predicts the raw enhancement map E(x,y).

    Output is unbounded — represents the intensity change at peak enhancement.
    Uses skip connections from encoder for high-frequency detail.
    """

    def __init__(self, output_nc=1, ngf=64, n_upsampling=4, n_blocks=3,
                 norm_layer=nn.InstanceNorm2d):
        super().__init__()
        activation = nn.ReLU(True)
        mult = 2 ** n_upsampling
        bottleneck_dim = ngf * mult

        # Additional bottleneck blocks specific to enhancement
        self.refine = nn.Sequential(*[
            ResnetBlock(bottleneck_dim, norm_layer=norm_layer, activation=activation)
            for _ in range(n_blocks)
        ])

        # Upsampling with skip connections
        self.up_layers = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        for i in range(n_upsampling):
            in_ch = ngf * mult
            out_ch = in_ch // 2
            self.up_layers.append(nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2,
                                   padding=1, output_padding=1),
                norm_layer(out_ch),
                activation,
            ))
            # Skip connection: concat then reduce
            self.skip_convs.append(nn.Sequential(
                nn.Conv2d(out_ch * 2, out_ch, kernel_size=1),
                norm_layer(out_ch),
                activation,
            ))
            mult //= 2

        # Final output: unbounded residual
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            # No activation — unbounded enhancement
        )

    def forward(self, bottleneck_feat, skips):
        """
        Args:
            bottleneck_feat: [B, C_bottleneck, H_low, W_low]
            skips: list of encoder features [initial, down1, down2, ..., down_n]
                   skips[-1] is at bottleneck resolution, skips[0] at full resolution
        """
        feat = self.refine(bottleneck_feat)

        # Upsample and fuse with skip connections (reverse order)
        # skips = [initial_feat, after_down1, after_down2, ..., after_down_n]
        # We want to fuse with after_down_{n-1}, after_down_{n-2}, ..., initial
        for i, (up, skip_conv) in enumerate(zip(self.up_layers, self.skip_convs)):
            feat = up(feat)
            # Skip from encoder: skips[-(i+2)] gives matching resolution
            skip_idx = len(skips) - 2 - i
            if skip_idx >= 0:
                skip_feat = skips[skip_idx]
                # Handle potential size mismatch
                if feat.shape[2:] != skip_feat.shape[2:]:
                    feat = F.interpolate(feat, size=skip_feat.shape[2:],
                                         mode='bilinear', align_corners=False)
                feat = skip_conv(torch.cat([feat, skip_feat], dim=1))

        return self.final(feat)


class GateDecoder(nn.Module):
    """Decoder that predicts spatial attention gate G(x,y) ∈ [0, 1].

    The gate determines WHERE enhancement happens. Enforces:
    - Gate ≈ 0 outside breast / in non-enhancing parenchyma
    - Gate > 0 in regions with contrast uptake (tumors, vessels, BPE)

    Lightweight compared to EnhancementDecoder — spatial gating is a simpler task.
    """

    def __init__(self, output_nc=1, ngf=64, n_upsampling=4, n_blocks=2,
                 norm_layer=nn.InstanceNorm2d):
        super().__init__()
        activation = nn.ReLU(True)
        mult = 2 ** n_upsampling
        bottleneck_dim = ngf * mult

        # Fewer refinement blocks (gating is easier than enhancement)
        self.refine = nn.Sequential(*[
            ResnetBlock(bottleneck_dim, norm_layer=norm_layer, activation=activation)
            for _ in range(n_blocks)
        ])

        # Simple upsampling (no skip connections — gate should be smooth)
        up_layers = []
        for i in range(n_upsampling):
            in_ch = ngf * mult
            out_ch = in_ch // 2
            up_layers += [
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2,
                                   padding=1, output_padding=1),
                norm_layer(out_ch),
                activation,
            ]
            mult //= 2

        # Final: sigmoid for [0, 1] gating
        up_layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Sigmoid(),
        ]
        self.decoder = nn.Sequential(*up_layers)

    def forward(self, bottleneck_feat):
        feat = self.refine(bottleneck_feat)
        return self.decoder(feat)


class SemiDisentangledGenerator(nn.Module):
    """Semi-Disentangled Generator for DCE-MRI Virtual Contrast Enhancement.

    Decomposes synthesis into:
        output = pre + gate(x,y) * enhancement(x,y)

    where:
        - pre is passed through unchanged (anatomy lock)
        - gate ∈ [0,1] learns spatial attention (where contrast enhances)
        - enhancement ∈ ℝ learns the intensity delta (how much it enhances)

    The "semi" in semi-disentangled: encoder is shared (joint anatomical
    understanding), but the two decoder heads produce structurally different
    outputs — one spatial attention, one intensity residual.
    """

    def __init__(self, input_nc=1, output_nc=1, ngf=64, n_downsampling=4,
                 n_encoder_blocks=6, n_enhance_blocks=3, n_gate_blocks=2,
                 norm='instance', breast_mask_input=False):
        super().__init__()

        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False) \
            if norm == 'instance' else functools.partial(nn.BatchNorm2d, affine=True)

        # Optional: feed breast mask as additional input channel
        enc_input_nc = input_nc + (1 if breast_mask_input else 0)
        self.breast_mask_input = breast_mask_input

        # Shared encoder
        self.encoder = SharedEncoder(
            input_nc=enc_input_nc, ngf=ngf, n_downsampling=n_downsampling,
            n_blocks=n_encoder_blocks, norm_layer=norm_layer,
        )

        # Enhancement head (heavy — this does the actual synthesis work)
        self.enhancement_decoder = EnhancementDecoder(
            output_nc=output_nc, ngf=ngf, n_upsampling=n_downsampling,
            n_blocks=n_enhance_blocks, norm_layer=norm_layer,
        )

        # Gate head (lightweight — spatial attention)
        self.gate_decoder = GateDecoder(
            output_nc=output_nc, ngf=ngf, n_upsampling=n_downsampling,
            n_blocks=n_gate_blocks, norm_layer=norm_layer,
        )

        self.n_downsampling = n_downsampling

    def forward(self, pre, breast_mask=None):
        """
        Args:
            pre: [B, 1, H, W] pre-contrast input (z-score normalized)
            breast_mask: [B, 1, H, W] optional binary breast mask (0=background, 1=breast)

        Returns:
            output: [B, 1, H, W] synthesized post-contrast image
            gate: [B, 1, H, W] spatial attention map (for visualization/loss)
            enhancement: [B, 1, H, W] raw enhancement map (for visualization/loss)
        """
        # Build encoder input
        if self.breast_mask_input and breast_mask is not None:
            enc_input = torch.cat([pre, breast_mask], dim=1)
        else:
            enc_input = pre

        # Shared encoding
        bottleneck_feat, skips = self.encoder(enc_input)

        # Enhancement prediction (with skip connections for detail)
        enhancement = self.enhancement_decoder(bottleneck_feat, skips)

        # Gate prediction (smooth spatial attention)
        gate = self.gate_decoder(bottleneck_feat)

        # Hard mask constraint: force gate=0 outside breast
        if breast_mask is not None:
            gate = gate * breast_mask

        # Semi-disentangled composition:
        # Anatomy is LOCKED (pre passes through unchanged)
        # Only the gated enhancement is added
        output = pre + gate * enhancement

        return output, gate, enhancement


class DisentangledLoss(nn.Module):
    """Loss functions for training the Semi-Disentangled Generator.

    Enforces the disentanglement priors:
    1. Anatomy consistency: output should equal pre outside the enhancement gate
    2. Gate sparsity: most pixels should NOT enhance (L1 on gate)
    3. Gate-anatomy alignment: gate should be zero outside breast mask
    4. Enhancement quality: standard reconstruction + perceptual + GAN losses
    """

    def __init__(self, lambda_anatomy=5.0, lambda_gate_sparsity=0.5,
                 lambda_gate_tv=0.1, lambda_enhance_l1=10.0,
                 lambda_tumor_boost=5.0):
        super().__init__()
        self.lambda_anatomy = lambda_anatomy
        self.lambda_gate_sparsity = lambda_gate_sparsity
        self.lambda_gate_tv = lambda_gate_tv
        self.lambda_enhance_l1 = lambda_enhance_l1
        self.lambda_tumor_boost = lambda_tumor_boost

    def forward(self, output, gt, pre, gate, enhancement,
                breast_mask=None, tumor_mask=None):
        """
        Args:
            output: [B, 1, H, W] predicted post-contrast
            gt: [B, 1, H, W] ground-truth post-contrast
            pre: [B, 1, H, W] pre-contrast input
            gate: [B, 1, H, W] spatial gate map
            enhancement: [B, 1, H, W] enhancement map
            breast_mask: [B, 1, H, W] binary breast mask (optional)
            tumor_mask: [B, 1, H, W] binary tumor mask (optional)

        Returns:
            total_loss, loss_dict
        """
        losses = {}

        # === 1. Anatomy Lock Loss ===
        # Outside the gate region, output must equal input exactly
        # Weighted by (1 - gate) so regions with gate≈0 are heavily penalized for changes
        anatomy_diff = (output - pre).abs() * (1.0 - gate.detach())
        if breast_mask is not None:
            # Extra strong: completely outside breast must be ZERO change
            outside_breast = (1.0 - breast_mask)
            outside_pixels = outside_breast.bool().expand_as(output)
            if outside_pixels.any():
                losses['anatomy_outside'] = (output - pre).abs()[outside_pixels].mean()
            else:
                losses['anatomy_outside'] = torch.tensor(0.0, device=output.device)

            inside_pixels = breast_mask.bool().expand_as(anatomy_diff)
            if inside_pixels.any():
                losses['anatomy_inside'] = anatomy_diff[inside_pixels].mean()
            else:
                losses['anatomy_inside'] = torch.tensor(0.0, device=output.device)

            loss_anatomy = losses['anatomy_outside'] * 2.0 + losses['anatomy_inside']
        else:
            loss_anatomy = anatomy_diff.mean()
        losses['anatomy'] = loss_anatomy.item()

        # === 2. Gate Sparsity ===
        # Most of the breast does NOT enhance strongly — encourage sparse gate
        loss_gate_sparse = gate.mean()
        losses['gate_sparsity'] = loss_gate_sparse.item()

        # === 3. Gate Total Variation ===
        # Gate should be spatially smooth (no salt-and-pepper noise)
        gate_tv_h = (gate[:, :, 1:, :] - gate[:, :, :-1, :]).abs().mean()
        gate_tv_w = (gate[:, :, :, 1:] - gate[:, :, :, :-1]).abs().mean()
        loss_gate_tv = gate_tv_h + gate_tv_w
        losses['gate_tv'] = loss_gate_tv.item()

        # === 4. Enhancement Quality (L1) ===
        # Global L1
        loss_enhance = F.l1_loss(output, gt)
        losses['enhance_l1'] = loss_enhance.item()

        # === 5. Tumor-boosted loss ===
        # Extra weight on tumor region reconstruction
        loss_tumor = torch.tensor(0.0, device=output.device)
        if tumor_mask is not None and tumor_mask.sum() > 0:
            tumor_region_pred = output * tumor_mask
            tumor_region_gt = gt * tumor_mask
            loss_tumor = F.l1_loss(tumor_region_pred, tumor_region_gt)
            losses['tumor_l1'] = loss_tumor.item()

        # === Total ===
        total = (
            self.lambda_enhance_l1 * loss_enhance
            + self.lambda_anatomy * loss_anatomy
            + self.lambda_gate_sparsity * loss_gate_sparse
            + self.lambda_gate_tv * loss_gate_tv
            + self.lambda_tumor_boost * loss_tumor
        )
        losses['total'] = total.item()

        return total, losses


def build_semi_disentangled_generator(
    ngf=64,
    n_downsampling=4,
    n_encoder_blocks=6,
    n_enhance_blocks=3,
    n_gate_blocks=2,
    breast_mask_input=False,
):
    """Factory function matching the interface expected by training scripts."""
    return SemiDisentangledGenerator(
        input_nc=1,
        output_nc=1,
        ngf=ngf,
        n_downsampling=n_downsampling,
        n_encoder_blocks=n_encoder_blocks,
        n_enhance_blocks=n_enhance_blocks,
        n_gate_blocks=n_gate_blocks,
        breast_mask_input=breast_mask_input,
    )
