"""Inference script for Semi-Disentangled Generator.

Loads the trained model and runs inference on test MHA files.
Optionally generates breast masks on-the-fly using the 2D nnUNet model.

Usage:
    python infer_semi_disentangled.py \
        --weights /path/to/latest_net_G.pth \
        --input_dir /path/to/test/mha/input \
        --output_dir /path/to/predictions \
        --breast_seg_model /path/to/nnUNet_results/Dataset920_BreastSeg2D/...
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
from semi_disentangled_generator import SemiDisentangledGenerator


def load_generator(weights_path, device, **kwargs):
    """Load trained Semi-Disentangled Generator."""
    netG = SemiDisentangledGenerator(
        input_nc=kwargs.get('input_nc', 1),
        output_nc=kwargs.get('output_nc', 1),
        ngf=kwargs.get('ngf', 64),
        n_downsampling=kwargs.get('n_downsampling', 4),
        n_encoder_blocks=kwargs.get('n_encoder_blocks', 6),
        n_enhance_blocks=kwargs.get('n_enhance_blocks', 3),
        n_gate_blocks=kwargs.get('n_gate_blocks', 2),
        breast_mask_input=kwargs.get('breast_mask_input', False),
    )
    state_dict = torch.load(weights_path, map_location=device)
    netG.load_state_dict(state_dict)
    netG.to(device).eval()
    n_params = sum(p.numel() for p in netG.parameters()) / 1e6
    print(f'[Generator] Loaded {n_params:.1f}M params from {weights_path}')
    return netG


def load_breast_seg_predictor(model_folder, device):
    """Load 2D nnUNet breast segmentation model."""
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
        perform_everything_on_device=True, device=device, verbose=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        model_folder, use_folds=(0,), checkpoint_name='checkpoint_final.pth'
    )
    return predictor


def predict_breast_mask(predictor, slice_2d):
    """Predict binary breast mask from 2D slice using nnUNet."""
    from scipy import ndimage as ndi

    props = {
        'sitk_stuff': {
            'spacing': (1., 1., 1.),
            'origin': (0., 0., 0.),
            'direction': (1., 0., 0., 0., 1., 0., 0., 0., 1.),
        },
        'spacing': [1., 1., 1.],
    }
    input_arr = slice_2d[np.newaxis, np.newaxis, :, :]
    pred_mask = predictor.predict_single_npy_array(input_arr, props, None, None, False)

    while pred_mask.ndim > 2:
        pred_mask = pred_mask[0]

    bm = (pred_mask > 0).astype(np.uint8)

    # Remove small components
    labeled, n = ndi.label(bm)
    if n > 1:
        sizes = ndi.sum(bm, labeled, range(1, n + 1))
        thresh = sizes.max() * 0.1
        bm = np.zeros_like(bm, dtype=np.uint8)
        for i, s in enumerate(sizes):
            if s >= thresh:
                bm[labeled == (i + 1)] = 1

    return bm.astype(np.float32)


def pad_to_divisible(arr, divisor=16):
    """Pad 2D array so both dims are divisible by divisor."""
    h, w = arr.shape
    new_h = int(np.ceil(h / divisor) * divisor)
    new_w = int(np.ceil(w / divisor) * divisor)
    if new_h == h and new_w == w:
        return arr, (0, 0)
    padded = np.zeros((new_h, new_w), dtype=arr.dtype)
    padded[:h, :w] = arr
    return padded, (h, w)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True, help='Path to latest_net_G.pth')
    parser.add_argument('--input_dir', required=True, help='Directory with input .mha files')
    parser.add_argument('--output_dir', required=True, help='Output directory for predictions')
    parser.add_argument('--breast_seg_model', default='', help='Path to nnUNet breast seg model folder')
    parser.add_argument('--breast_mask_dir', default='', help='Pre-computed breast masks directory')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--n_downsampling', type=int, default=4)
    parser.add_argument('--n_encoder_blocks', type=int, default=6)
    parser.add_argument('--n_enhance_blocks', type=int, default=3)
    parser.add_argument('--n_gate_blocks', type=int, default=2)
    parser.add_argument('--breast_mask_input', action='store_true')
    parser.add_argument('--save_gate', action='store_true', help='Also save gate maps for visualization')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # Load generator
    netG = load_generator(
        args.weights, device,
        ngf=args.ngf,
        n_downsampling=args.n_downsampling,
        n_encoder_blocks=args.n_encoder_blocks,
        n_enhance_blocks=args.n_enhance_blocks,
        n_gate_blocks=args.n_gate_blocks,
        breast_mask_input=args.breast_mask_input,
    )

    # Load breast segmentation model (if needed)
    breast_predictor = None
    if args.breast_seg_model and not args.breast_mask_dir:
        breast_predictor = load_breast_seg_predictor(args.breast_seg_model, device)
        print(f'[Breast Seg] Loaded from {args.breast_seg_model}')

    # Gather input files
    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob('*.mha'))
    print(f'Processing {len(files)} test cases...')

    gate_dir = None
    if args.save_gate:
        gate_dir = Path(args.output_dir) / 'gate_maps'
        gate_dir.mkdir(exist_ok=True)

    divisor = 2 ** args.n_downsampling

    for f in tqdm(files):
        out_path = Path(args.output_dir) / f.name
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
                bm = sitk.GetArrayFromImage(sitk.ReadImage(str(bm_path))).squeeze()
                breast_mask = (bm > 0).astype(np.float32)
            else:
                breast_mask = np.ones_like(sl)
        elif breast_predictor is not None:
            breast_mask = predict_breast_mask(breast_predictor, sl)
        else:
            # Fallback: threshold-based
            breast_mask = (sl > -0.3).astype(np.float32)

        # Pad to divisible
        sl_padded, orig_size = pad_to_divisible(sl, divisor)
        bm_padded, _ = pad_to_divisible(breast_mask, divisor)

        # To tensor
        pre_t = torch.from_numpy(sl_padded).unsqueeze(0).unsqueeze(0).float().to(device)
        bm_t = torch.from_numpy(bm_padded).unsqueeze(0).unsqueeze(0).float().to(device)

        # Inference
        with torch.no_grad():
            output, gate, enhancement = netG(pre_t, bm_t)

        # Extract result
        result = output[0, 0].cpu().numpy()

        # Crop back to original size
        if orig_size != (0, 0):
            result = result[:orig_size[0], :orig_size[1]]

        # Ensure correct shape
        if arr.ndim == 3:
            result = result[np.newaxis, ...]

        # Write output preserving metadata
        out_img = sitk.GetImageFromArray(result.astype(np.float32))
        out_img.CopyInformation(img)
        sitk.WriteImage(out_img, str(out_path))

        # Optionally save gate map
        if gate_dir is not None:
            gate_arr = gate[0, 0].cpu().numpy()
            if orig_size != (0, 0):
                gate_arr = gate_arr[:orig_size[0], :orig_size[1]]
            if arr.ndim == 3:
                gate_arr = gate_arr[np.newaxis, ...]
            gate_img = sitk.GetImageFromArray(gate_arr.astype(np.float32))
            gate_img.CopyInformation(img)
            sitk.WriteImage(gate_img, str(gate_dir / f.name))

    print(f'Done. {len(list(Path(args.output_dir).glob("*.mha")))} predictions saved.')


if __name__ == '__main__':
    main()
