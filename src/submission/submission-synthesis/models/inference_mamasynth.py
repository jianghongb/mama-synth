#!/usr/bin/env python3
"""MAMA-SYNTH Docker inference script.

Reads pre-contrast MHA from /input/images/pre-contrast/,
runs pix2pixHD generator (residual mode),
writes synthetic post-contrast subtraction to /output/images/synthetic-post-contrast/.

Preserves spatial metadata (origin, spacing, direction) from input.
"""
import os
import sys
import torch
import numpy as np
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(__file__))
from models.networks import define_G


def pad_to_multiple(arr, base=16):
    """Pad 2D array to nearest multiple of base, return padded array and original shape."""
    h, w = arr.shape
    new_h = int(np.ceil(h / base) * base)
    new_w = int(np.ceil(w / base) * base)
    if new_h == h and new_w == w:
        return arr, (h, w)
    padded = np.zeros((new_h, new_w), dtype=arr.dtype)
    padded[:h, :w] = arr
    return padded, (h, w)


def main():
    input_dir = os.environ.get('MAMA_INPUT_DIR', '/input/images/pre-contrast')
    output_dir = os.environ.get('MAMA_OUTPUT_DIR', '/output/images/synthetic-post-contrast')
    model_dir = os.environ.get('MAMA_MODEL_DIR', '/opt/app/weights')

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load generator
    netG = define_G(
        input_nc=1, output_nc=1, ngf=64, netG='global',
        n_downsample_global=4, n_blocks_global=9,
        norm='instance', gpu_ids=[], residual_mode=True
    )
    weights_path = os.path.join(model_dir, 'latest_net_G.pth')
    netG.load_state_dict(torch.load(weights_path, map_location=device))
    netG.to(device)
    netG.eval()

    # Process each case
    mha_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.mha')])
    for fname in mha_files:
        input_path = os.path.join(input_dir, fname)
        img = sitk.ReadImage(input_path)
        arr = sitk.GetArrayFromImage(img).astype(np.float32)

        # Pad
        padded, (orig_h, orig_w) = pad_to_multiple(arr, base=16)

        # Inference
        tensor = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            output = netG(tensor)

        # Crop back and convert
        result = output[0, 0, :orig_h, :orig_w].cpu().numpy().astype(np.float32)

        # Write with original metadata
        out_img = sitk.GetImageFromArray(result)
        out_img.CopyInformation(img)
        sitk.WriteImage(out_img, os.path.join(output_dir, fname))

        print(f'Processed {fname}: {arr.shape}')


if __name__ == '__main__':
    main()
