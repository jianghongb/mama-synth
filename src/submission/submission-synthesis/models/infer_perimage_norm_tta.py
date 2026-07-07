"""Inference with Test-Time Augmentation (TTA) + Per-Image Z-Score Normalization.

Applies spatial transforms (H-flip, V-flip, HV-flip), runs model on each,
then averages results for more stable predictions.

Usage:
    python infer_perimage_norm_tta.py \
        --weights /path/to/latest_net_G.pth \
        --input_dir /path/to/test/mha/input \
        --output_dir /path/to/predictions \
        --breast_mask_dir /path/to/test/mha/breast_mask \
        --tta_mode hflip        # Options: none, hflip, 4flip, 8rot
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from networks import GlobalGenerator, get_norm_layer

import functools


def apply_transform(img: np.ndarray, transform: str) -> np.ndarray:
    """Apply spatial transform to 2D image."""
    if transform == 'none':
        return img
    elif transform == 'hflip':
        return np.flip(img, axis=1).copy()
    elif transform == 'vflip':
        return np.flip(img, axis=0).copy()
    elif transform == 'hvflip':
        return np.flip(np.flip(img, axis=0), axis=1).copy()
    elif transform == 'rot90':
        return np.rot90(img, k=1).copy()
    elif transform == 'rot180':
        return np.rot90(img, k=2).copy()
    elif transform == 'rot270':
        return np.rot90(img, k=3).copy()
    elif transform == 'rot90_hflip':
        return np.flip(np.rot90(img, k=1), axis=1).copy()
    return img


def inverse_transform(img: np.ndarray, transform: str) -> np.ndarray:
    """Inverse spatial transform."""
    if transform == 'none':
        return img
    elif transform == 'hflip':
        return np.flip(img, axis=1).copy()
    elif transform == 'vflip':
        return np.flip(img, axis=0).copy()
    elif transform == 'hvflip':
        return np.flip(np.flip(img, axis=0), axis=1).copy()
    elif transform == 'rot90':
        return np.rot90(img, k=-1).copy()
    elif transform == 'rot180':
        return np.rot90(img, k=-2).copy()
    elif transform == 'rot270':
        return np.rot90(img, k=-3).copy()
    elif transform == 'rot90_hflip':
        return np.rot90(np.flip(img, axis=1), k=-1).copy()
    return img


TTA_MODES = {
    'none': ['none'],
    'hflip': ['none', 'hflip'],
    '4flip': ['none', 'hflip', 'vflip', 'hvflip'],
    '8rot': ['none', 'hflip', 'vflip', 'hvflip', 'rot90', 'rot180', 'rot270', 'rot90_hflip'],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--breast_mask_dir', default='')
    parser.add_argument('--breast_mask_thresh', type=float, default=-0.3)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--n_blocks', type=int, default=12)
    parser.add_argument('--n_downsampling', type=int, default=4)
    parser.add_argument('--tta_mode', default='hflip', choices=list(TTA_MODES.keys()))
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    norm_layer = functools.partial(torch.nn.InstanceNorm2d, affine=False)
    netG = GlobalGenerator(1, 1, args.ngf, args.n_downsampling, args.n_blocks,
                           norm_layer, residual_mode=True)
    netG.load_state_dict(torch.load(args.weights, map_location=device))
    netG.to(device).eval()

    transforms = TTA_MODES[args.tta_mode]
    print(f'Model loaded: ngf={args.ngf}, blocks={args.n_blocks}, device={device}')
    print(f'TTA mode: {args.tta_mode} ({len(transforms)} transforms: {transforms})')

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    files = sorted(input_dir.glob('*.mha'))
    print(f'Processing {len(files)} test cases...')

    for f in tqdm(files):
        out_path = output_dir / f.name
        if out_path.exists():
            continue

        # Read input
        img = sitk.ReadImage(str(f))
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        sl = arr.squeeze()
        orig_h, orig_w = sl.shape

        # Get breast mask
        if args.breast_mask_dir:
            bm_path = Path(args.breast_mask_dir) / f.name
            if bm_path.exists():
                bm = sitk.GetArrayFromImage(sitk.ReadImage(str(bm_path))).astype(np.float32).squeeze()
                breast_mask = (bm > 0).astype(np.float32)
            else:
                breast_mask = (sl > args.breast_mask_thresh).astype(np.float32)
        else:
            breast_mask = (sl > args.breast_mask_thresh).astype(np.float32)

        # Per-image z-score stats
        fg_pixels = sl[breast_mask > 0.5]
        if fg_pixels.size > 100:
            img_mean = float(fg_pixels.mean())
            img_std = float(fg_pixels.std())
            img_std = max(img_std, 1e-8)
        else:
            img_mean, img_std = 0.0, 1.0

        # TTA: run model for each transform, collect results
        results = []
        for transform in transforms:
            # Apply transform to input and mask
            sl_t = apply_transform(sl, transform)
            bm_t = apply_transform(breast_mask, transform)

            # Normalize
            sl_norm = (sl_t - img_mean) / img_std * bm_t

            # To tensor
            t = torch.from_numpy(sl_norm).unsqueeze(0).unsqueeze(0).float().to(device)
            h, w = sl_t.shape
            t = F.interpolate(t, size=(512, 512), mode='bilinear', align_corners=False)

            # Model
            with torch.no_grad():
                out = netG(t)

            # Resize back
            result_norm = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
            result_norm = result_norm[0, 0].cpu().numpy()

            # De-normalize
            result = result_norm * img_std + img_mean
            result = bm_t * result + (1 - bm_t) * sl_t

            # Inverse transform
            result = inverse_transform(result, transform)
            results.append(result)

        # Average all TTA results
        final = np.mean(results, axis=0).astype(np.float32)

        # Write output
        if arr.ndim == 3:
            final = final[np.newaxis, ...]
        out_img = sitk.GetImageFromArray(final)
        out_img.CopyInformation(img)
        sitk.WriteImage(out_img, str(out_path))

    print(f'Done. {len(list(output_dir.glob("*.mha")))} predictions saved to {output_dir}')


if __name__ == '__main__':
    main()
