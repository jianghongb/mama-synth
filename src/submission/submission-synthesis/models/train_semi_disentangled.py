"""Training script for Semi-Disentangled Generator (v30).

Combines:
- Semi-disentangled architecture (anatomy lock + gated enhancement)
- Multi-scale discriminator from Pix2PixHD
- VGG perceptual loss
- Feature matching loss
- Multi-Scale Subtraction Consistency (MSSC)
- Tumor-weighted reconstruction
- Gate regularisation (sparsity + TV)

Usage:
    python train_semi_disentangled.py \
        --name mamasynth_v30_sdinr \
        --dataroot /path/to/data \
        --checkpoints_dir /path/to/checkpoints \
        ...
"""
import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from semi_disentangled_generator import (
    SemiDisentangledGenerator,
    DisentangledLoss,
)
from networks import (
    MultiscaleDiscriminator,
    GANLoss,
    VGGLoss,
    get_norm_layer,
    weights_init,
)
from data.mha_dataset import MhaDataset


class MSSCLoss(nn.Module):
    """Multi-Scale Subtraction Consistency Loss."""

    def __init__(self, levels=3):
        super().__init__()
        self.levels = levels

    def forward(self, pred, gt, pre):
        sub_pred = pred - pre
        sub_real = gt - pre
        loss = F.l1_loss(sub_pred, sub_real)
        for s in range(1, self.levels):
            scale = 2 ** s
            size = (pred.shape[2] // scale, pred.shape[3] // scale)
            if size[0] < 4 or size[1] < 4:
                break
            sp = F.interpolate(sub_pred, size=size, mode='bilinear', align_corners=False)
            sr = F.interpolate(sub_real, size=size, mode='bilinear', align_corners=False)
            loss += F.l1_loss(sp, sr)
        return loss / self.levels


def create_parser():
    parser = argparse.ArgumentParser(description='Train Semi-Disentangled Generator')

    # Experiment
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--checkpoints_dir', type=str, required=True)
    parser.add_argument('--gpu_ids', type=str, default='0')

    # Data
    parser.add_argument('--dataroot', type=str, required=True)
    parser.add_argument('--breast_mask_dir', type=str, default='')
    parser.add_argument('--loadSize', type=int, default=512)
    parser.add_argument('--batchSize', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--no_flip', action='store_true')

    # Generator architecture
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--n_downsampling', type=int, default=4)
    parser.add_argument('--n_encoder_blocks', type=int, default=6)
    parser.add_argument('--n_enhance_blocks', type=int, default=3)
    parser.add_argument('--n_gate_blocks', type=int, default=2)
    parser.add_argument('--breast_mask_input', action='store_true',
                        help='Feed breast mask as additional input channel to encoder')

    # Discriminator
    parser.add_argument('--num_D', type=int, default=2)
    parser.add_argument('--n_layers_D', type=int, default=3)
    parser.add_argument('--ndf', type=int, default=64)

    # Training schedule
    parser.add_argument('--niter', type=int, default=100, help='Epochs at initial LR')
    parser.add_argument('--niter_decay', type=int, default=100, help='Epochs for LR decay')
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)

    # Loss weights
    parser.add_argument('--lambda_enhance_l1', type=float, default=10.0)
    parser.add_argument('--lambda_anatomy', type=float, default=5.0)
    parser.add_argument('--lambda_gate_sparsity', type=float, default=0.5)
    parser.add_argument('--lambda_gate_tv', type=float, default=0.1)
    parser.add_argument('--lambda_tumor', type=float, default=5.0)
    parser.add_argument('--lambda_vgg', type=float, default=10.0)
    parser.add_argument('--lambda_gan', type=float, default=1.0)
    parser.add_argument('--lambda_feat', type=float, default=10.0)
    parser.add_argument('--lambda_mssc', type=float, default=20.0)
    parser.add_argument('--mssc_levels', type=int, default=3)

    # Logging & saving
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--save_epoch_freq', type=int, default=5)
    parser.add_argument('--continue_train', action='store_true')

    return parser


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def update_learning_rate(optimizer, opt, epoch):
    """Linear decay after niter epochs."""
    if epoch > opt.niter:
        decay_frac = (epoch - opt.niter) / float(opt.niter_decay)
        new_lr = opt.lr * (1.0 - decay_frac)
        new_lr = max(new_lr, 1e-6)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


def main():
    parser = create_parser()
    opt = parser.parse_args()

    # Device
    gpu_ids = [int(x) for x in opt.gpu_ids.split(',') if x]
    device = torch.device(f'cuda:{gpu_ids[0]}' if gpu_ids and torch.cuda.is_available() else 'cpu')

    # Checkpoint dir
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    os.makedirs(save_dir, exist_ok=True)

    # ─── Build Dataset ───────────────────────────────────────────
    # Create a minimal opt-like object for MhaDataset
    class DataOpt:
        pass
    data_opt = DataOpt()
    data_opt.dataroot = opt.dataroot
    data_opt.isTrain = True
    data_opt.no_flip = opt.no_flip
    data_opt.resize_or_crop = 'resize'
    data_opt.loadSize = opt.loadSize
    data_opt.n_downsample_global = opt.n_downsampling
    data_opt.breast_mask_dir = opt.breast_mask_dir
    data_opt.breast_mask = True if not opt.breast_mask_dir else False
    data_opt.breast_mask_thresh = -0.3
    data_opt.intensity_aug = True
    data_opt.square_only = False
    data_opt.mask_as_input = False

    dataset = MhaDataset()
    dataset.initialize(data_opt)
    dataloader = DataLoader(
        dataset, batch_size=opt.batchSize, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, drop_last=True,
    )
    print(f'[Dataset] {len(dataset)} training samples, batch_size={opt.batchSize}')

    # ─── Build Generator ─────────────────────────────────────────
    netG = SemiDisentangledGenerator(
        input_nc=1,
        output_nc=1,
        ngf=opt.ngf,
        n_downsampling=opt.n_downsampling,
        n_encoder_blocks=opt.n_encoder_blocks,
        n_enhance_blocks=opt.n_enhance_blocks,
        n_gate_blocks=opt.n_gate_blocks,
        breast_mask_input=opt.breast_mask_input,
    ).to(device)
    netG.apply(weights_init)

    n_params = sum(p.numel() for p in netG.parameters()) / 1e6
    print(f'[Generator] SemiDisentangled: {n_params:.1f}M params')

    # ─── Build Discriminator ─────────────────────────────────────
    # D sees (pre_concat_output, real/fake) pairs → input_nc = 2
    norm_layer = get_norm_layer(norm_type='instance')
    netD = MultiscaleDiscriminator(
        input_nc=2,  # pre + output
        ndf=opt.ndf,
        n_layers=opt.n_layers_D,
        norm_layer=norm_layer,
        use_sigmoid=False,
        num_D=opt.num_D,
        getIntermFeat=True,  # for feature matching
    ).to(device)
    netD.apply(weights_init)

    # ─── Losses ──────────────────────────────────────────────────
    criterionGAN = GANLoss(use_lsgan=True, tensor=torch.cuda.FloatTensor if device.type == 'cuda' else torch.FloatTensor)
    criterionVGG = VGGLoss(gpu_ids) if opt.lambda_vgg > 0 else None
    criterionMSSC = MSSCLoss(levels=opt.mssc_levels) if opt.lambda_mssc > 0 else None
    criterionDisentangle = DisentangledLoss(
        lambda_anatomy=opt.lambda_anatomy,
        lambda_gate_sparsity=opt.lambda_gate_sparsity,
        lambda_gate_tv=opt.lambda_gate_tv,
        lambda_enhance_l1=opt.lambda_enhance_l1,
        lambda_tumor_boost=opt.lambda_tumor,
    ).to(device)
    criterionFeat = nn.L1Loss()

    # ─── Optimizers ──────────────────────────────────────────────
    optimizer_G = torch.optim.Adam(
        netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
    )
    optimizer_D = torch.optim.Adam(
        netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
    )

    # ─── Resume ──────────────────────────────────────────────────
    start_epoch = 1
    if opt.continue_train:
        g_path = os.path.join(save_dir, 'latest_net_G.pth')
        d_path = os.path.join(save_dir, 'latest_net_D.pth')
        if os.path.exists(g_path):
            netG.load_state_dict(torch.load(g_path, map_location=device))
            print(f'[Resume] Loaded G from {g_path}')
        if os.path.exists(d_path):
            netD.load_state_dict(torch.load(d_path, map_location=device))
            print(f'[Resume] Loaded D from {d_path}')

    # ─── Training Loop ───────────────────────────────────────────
    total_epochs = opt.niter + opt.niter_decay
    print(f'\n=== Training for {total_epochs} epochs ===')
    print(f'    Losses: enhance_l1={opt.lambda_enhance_l1}, anatomy={opt.lambda_anatomy}, '
          f'gate_sparse={opt.lambda_gate_sparsity}, gate_tv={opt.lambda_gate_tv}')
    print(f'    GAN={opt.lambda_gan}, VGG={opt.lambda_vgg}, feat={opt.lambda_feat}, '
          f'MSSC={opt.lambda_mssc}, tumor={opt.lambda_tumor}')

    for epoch in range(start_epoch, total_epochs + 1):
        epoch_start = time.time()
        netG.train()
        netD.train()

        loss_accum = {}
        n_batches = 0

        for i, data in enumerate(dataloader):
            pre = data['label'].to(device)        # [B, 1, H, W]
            gt = data['image'].to(device)         # [B, 1, H, W]
            tumor_mask = data['mask'].to(device)  # [B, 1, H, W]
            breast_mask = data['breast_mask'].to(device)  # [B, 1, H, W]

            # ─── Forward G ───────────────────────────────────
            output, gate, enhancement = netG(pre, breast_mask)

            # ─── Update D ────────────────────────────────────
            optimizer_D.zero_grad()

            # Real pair
            real_pair = torch.cat([pre, gt], dim=1)
            pred_real = netD(real_pair)
            loss_D_real = criterionGAN(pred_real, True)

            # Fake pair
            fake_pair = torch.cat([pre, output.detach()], dim=1)
            pred_fake = netD(fake_pair)
            loss_D_fake = criterionGAN(pred_fake, False)

            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()

            # ─── Update G ────────────────────────────────────
            optimizer_G.zero_grad()

            # GAN loss (G wants D to say "real")
            fake_pair_g = torch.cat([pre, output], dim=1)
            pred_fake_g = netD(fake_pair_g)
            loss_G_GAN = criterionGAN(pred_fake_g, True) * opt.lambda_gan

            # Feature matching loss
            loss_G_feat = torch.tensor(0.0, device=device)
            if opt.lambda_feat > 0:
                real_pair_for_feat = torch.cat([pre, gt], dim=1)
                pred_real_feat = netD(real_pair_for_feat)
                feat_weights = 4.0 / (opt.n_layers_D + 1)
                D_weights = 1.0 / opt.num_D
                for di in range(opt.num_D):
                    for fi in range(len(pred_fake_g[di]) - 1):
                        loss_G_feat += D_weights * feat_weights * \
                            criterionFeat(pred_fake_g[di][fi], pred_real_feat[di][fi].detach())
                loss_G_feat *= opt.lambda_feat

            # VGG perceptual loss
            loss_G_VGG = torch.tensor(0.0, device=device)
            if criterionVGG is not None:
                loss_G_VGG = criterionVGG(output, gt) * opt.lambda_vgg

            # MSSC loss (multi-scale subtraction consistency)
            loss_G_MSSC = torch.tensor(0.0, device=device)
            if criterionMSSC is not None:
                loss_G_MSSC = criterionMSSC(output, gt, pre) * opt.lambda_mssc

            # Disentanglement losses (anatomy lock + gate regularization + tumor boost)
            loss_disentangle, disentangle_breakdown = criterionDisentangle(
                output, gt, pre, gate, enhancement,
                breast_mask=breast_mask,
                tumor_mask=tumor_mask,
            )

            # Total G loss
            loss_G = loss_G_GAN + loss_G_feat + loss_G_VGG + loss_G_MSSC + loss_disentangle
            loss_G.backward()
            optimizer_G.step()

            # ─── Accumulate losses ───────────────────────────
            batch_losses = {
                'G_total': loss_G.item(),
                'D': loss_D.item(),
                'GAN': loss_G_GAN.item(),
                'feat': loss_G_feat.item(),
                'VGG': loss_G_VGG.item(),
                'MSSC': loss_G_MSSC.item(),
                'anatomy': disentangle_breakdown.get('anatomy', 0),
                'gate_sp': disentangle_breakdown.get('gate_sparsity', 0),
                'enhance': disentangle_breakdown.get('enhance_l1', 0),
                'tumor': disentangle_breakdown.get('tumor_l1', 0),
                'gate_mean': gate.mean().item(),
            }
            for k, v in batch_losses.items():
                loss_accum[k] = loss_accum.get(k, 0) + v
            n_batches += 1

            # Print
            if (i + 1) % opt.print_freq == 0:
                parts = ' '.join(f'{k}={v/n_batches:.4f}' for k, v in loss_accum.items())
                print(f'  [{epoch}/{total_epochs}][{i+1}/{len(dataloader)}] {parts}')

        # ─── Epoch summary ───────────────────────────────────
        elapsed = time.time() - epoch_start
        avg = {k: v / n_batches for k, v in loss_accum.items()}
        parts = ' '.join(f'{k}={v:.4f}' for k, v in avg.items())
        lr = get_lr(optimizer_G)
        print(f'[{epoch}/{total_epochs}] {parts} lr={lr:.2e} time={elapsed:.0f}s')

        # ─── LR decay ────────────────────────────────────────
        update_learning_rate(optimizer_G, opt, epoch)
        update_learning_rate(optimizer_D, opt, epoch)

        # ─── Save ────────────────────────────────────────────
        if epoch % opt.save_epoch_freq == 0 or epoch == total_epochs:
            torch.save(netG.state_dict(), os.path.join(save_dir, f'{epoch}_net_G.pth'))
            torch.save(netD.state_dict(), os.path.join(save_dir, f'{epoch}_net_D.pth'))
            print(f'  Saved epoch {epoch} checkpoint')

        # Always save latest
        torch.save(netG.state_dict(), os.path.join(save_dir, 'latest_net_G.pth'))
        torch.save(netD.state_dict(), os.path.join(save_dir, 'latest_net_D.pth'))

    print(f'\n=== Training complete. Weights in {save_dir} ===')


if __name__ == '__main__':
    main()
