"""Inference with Per-Image Z-Score Normalization.

For models trained with per-image normalization (v31+):
1. Load GC input (already global z-score normalized)
2. Compute breast-region mean/std from the input
3. Re-normalize to per-image z-score
4. Run model
5. De-normalize output back to global z-score space (matches GT)

Usage:
    python infer_perimage_norm.py \
        --weights /path/to/latest_net_G.pth \
        --input_dir /path/to/test/mha/input \
        --output_dir /path/to/predictions \
        --breast_mask_dir /path/to/test/mha/breast_mask
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--breast_mask_dir', default='',
                        help='Pre-computed breast masks (same filenames as input)')
    parser.add_argument('--breast_mask_thresh', type=float, default=-0.3,
                        help='Threshold for breast mask if no pre-computed masks')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--n_blocks', type=int, default=12)
    parser.add_argument('--n_downsampling', type=int, default=4)
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
    print(f'Model loaded: ngf={args.ngf}, blocks={args.n_blocks}, device={device}')

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    files = sorted(input_dir.glob('*.mha'))
    print(f'Processing {len(files)} test cases...')

    for f in tqdm(files):
        out_path = output_dir / f.name
        if out_path.exists():
            continue

        # Read input (global z-score normalized from GC preprocessing)
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

        # === Per-image z-score (foreground only) ===
        fg_pixels = sl[breast_mask > 0.5]
        if fg_pixels.size > 100:
            img_mean = float(fg_pixels.mean())
            img_std = float(fg_pixels.std())
            img_std = max(img_std, 1e-8)
        else:
            img_mean, img_std = 0.0, 1.0

        # Normalize
        sl_norm = (sl - img_mean) / img_std
        sl_norm = sl_norm * breast_mask  # zero background

        # To tensor and resize to 512x512
        t = torch.from_numpy(sl_norm).unsqueeze(0).unsqueeze(0).float().to(device)
        t = F.interpolate(t, size=(512, 512), mode='bilinear', align_corners=False)

        # Model inference
        with torch.no_grad():
            out = netG(t)

        # Resize back
        result_norm = F.interpolate(out, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        result_norm = result_norm[0, 0].cpu().numpy()

        # === De-normalize: convert back to global z-score space ===
        # Only de-norm the breast region; background keeps original input values
        result = result_norm * img_std + img_mean
        # Composite: breast region = de-normed model output, background = original input
        # Background offset: contrast agent causes subtle global intensity increase
        # (empirical mean offset from training data non-breast regions)
        BACKGROUND_OFFSET = 0.17
        result = breast_mask * result + (1 - breast_mask) * (sl + BACKGROUND_OFFSET)

        # Ensure correct shape
        if arr.ndim == 3:
            result = result[np.newaxis, ...]

        # Write output preserving metadata
        out_img = sitk.GetImageFromArray(result.astype(np.float32))
        out_img.CopyInformation(img)
        sitk.WriteImage(out_img, str(out_path))

    print(f'Done. {len(list(output_dir.glob("*.mha")))} predictions saved to {output_dir}')


if __name__ == '__main__':
    main()
