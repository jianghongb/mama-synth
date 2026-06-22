"""
split_bilateral.py — Split bilateral (both breasts) 2D axial MHA slices into
two unilateral crops with chest wall removed.

Purpose:
    Doubles the training set by splitting each bilateral slice at the midline.
    The chest wall (background between breast and image center) is excluded.
    This is useful because GC validation/test cases are bilateral axial slices.

Strategy:
    1. Load 2D pre-contrast, ground_truth, tumour mask, and breast mask.
    2. Find the midline from the breast mask (column separating L/R breast blobs).
    3. For each side: crop the breast bounding box + small padding, zero-out
       anything outside the breast mask (removes chest wall).
    4. Optionally resize to a target size.
    5. Save as <patient_id>_L.mha and <patient_id>_R.mha.

Inference integration:
    At inference time, split the bilateral input at midline, run the model on
    each half independently, then stitch back together.

Usage:
    python split_bilateral.py \
        --input_dir /path/to/mha/input \
        --gt_dir /path/to/mha/ground_truth \
        --mask_dir /path/to/mha/mask \
        --breast_mask_dir /path/to/mha/breast_mask \
        --output_dir /path/to/output_unilateral \
        --target_size 256 \
        --pad_ratio 0.05
"""
import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def read_mha(path: Path) -> np.ndarray:
    """Read MHA file as float32 numpy array."""
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)


def save_mha(arr: np.ndarray, path: Path, is_label: bool = False) -> None:
    """Save numpy array as MHA."""
    if is_label:
        arr = np.rint(arr).astype(np.int16)
    else:
        arr = arr.astype(np.float32)
    sitk.WriteImage(sitk.GetImageFromArray(arr), str(path))


def find_midline(breast_mask: np.ndarray) -> Optional[int]:
    """Find the column index that best separates left and right breasts.

    Returns None if only one breast is detected (no valid midline).
    Uses the column profile of the breast mask: the midline is the column
    with minimum breast coverage in the central 40% of the image width.
    """
    h, w = breast_mask.shape
    col_sum = np.sum(breast_mask > 0, axis=0)

    # Check if breast tissue exists on both sides of center
    half = w // 2
    left_mass = col_sum[:half].sum()
    right_mass = col_sum[half:].sum()
    total_mass = left_mass + right_mass

    # If >90% of breast is on one side, it's a single-breast case
    if total_mass == 0:
        return None
    if left_mass / total_mass > 0.9 or right_mass / total_mass > 0.9:
        return None

    # Search in central 40% of image
    margin = int(w * 0.3)
    search_start = margin
    search_end = w - margin

    if search_start >= search_end:
        return w // 2

    central_sums = col_sum[search_start:search_end]
    # Midline should have near-zero breast coverage
    min_val = central_sums.min()
    if min_val > h * 0.3:
        # No clear gap between breasts — likely single breast spanning center
        return None

    midline = search_start + int(np.argmin(central_sums))
    return midline


def crop_breast_side(
    image: np.ndarray,
    breast_mask: np.ndarray,
    col_start: int,
    col_end: int,
    pad_pixels: int = 5,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """Crop one side of the image to the breast bounding box.

    Returns:
        cropped_image: the cropped and chest-masked image
        cropped_mask: the cropped breast mask
        bbox: (row_start, row_end, col_start, col_end) used for cropping
    """
    h, w = image.shape
    side_mask = breast_mask[:, col_start:col_end].copy()
    side_img = image[:, col_start:col_end].copy()

    # Find bounding box of breast in this side
    rows = np.any(side_mask > 0, axis=1)
    cols = np.any(side_mask > 0, axis=0)

    if not rows.any() or not cols.any():
        return None, None, None

    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]

    # Add padding
    r_min = max(0, r_min - pad_pixels)
    r_max = min(side_mask.shape[0] - 1, r_max + pad_pixels)
    c_min = max(0, c_min - pad_pixels)
    c_max = min(side_mask.shape[1] - 1, c_max + pad_pixels)

    cropped_img = side_img[r_min:r_max + 1, c_min:c_max + 1]
    cropped_mask = side_mask[r_min:r_max + 1, c_min:c_max + 1]

    # Zero-out anything outside breast mask (chest wall removal)
    cropped_img = cropped_img * (cropped_mask > 0).astype(np.float32)

    bbox = (r_min, r_max + 1, col_start + c_min, col_start + c_max + 1)
    return cropped_img, cropped_mask, bbox


def resize_array(arr: np.ndarray, target_size: int, is_label: bool = False) -> np.ndarray:
    """Resize 2D array to target_size x target_size."""
    from PIL import Image
    mode = Image.NEAREST if is_label else Image.BICUBIC
    pil = Image.fromarray(arr)
    resized = pil.resize((target_size, target_size), mode)
    return np.array(resized, dtype=arr.dtype)


def process_patient(
    patient_id: str,
    input_path: Path,
    gt_path: Optional[Path],
    mask_path: Optional[Path],
    breast_mask_path: Path,
    output_dirs: dict,
    target_size: Optional[int],
    pad_ratio: float,
    mask_output: bool = True,
) -> list:
    """Split one bilateral patient into L/R unilateral crops.

    Returns list of generated case IDs.
    """
    pre = read_mha(input_path)
    breast_mask = read_mha(breast_mask_path)
    breast_mask = (breast_mask > 0).astype(np.float32)

    gt = read_mha(gt_path) if gt_path and gt_path.exists() else None
    tumour_mask = read_mha(mask_path) if mask_path and mask_path.exists() else None

    h, w = pre.shape
    pad_pixels = max(3, int(max(h, w) * pad_ratio))

    # Find midline — None means single-breast case
    midline = find_midline(breast_mask)
    generated = []

    if midline is None:
        # Single breast: crop entire breast region as one sample (suffix _S)
        sides = [("S", 0, w)]
    else:
        sides = [("L", 0, midline), ("R", midline, w)]

    for side, col_start, col_end in sides:
        # Crop pre-contrast
        cropped_pre, cropped_bm, bbox = crop_breast_side(
            pre, breast_mask, col_start, col_end, pad_pixels
        )
        if cropped_pre is None:
            logger.warning(f"  {patient_id}_{side}: no breast tissue found, skipping")
            continue

        # Crop GT with same bbox (relative to side)
        r0, r1 = bbox[0], bbox[1]
        c0_local = bbox[2] - col_start
        c1_local = bbox[3] - col_start

        if gt is not None:
            side_gt = gt[:, col_start:col_end]
            cropped_gt = side_gt[r0:r1, c0_local:c1_local]
            cropped_gt = cropped_gt * (cropped_bm > 0).astype(np.float32)
        else:
            cropped_gt = None

        if tumour_mask is not None:
            side_tm = tumour_mask[:, col_start:col_end]
            cropped_tm = side_tm[r0:r1, c0_local:c1_local]
        else:
            cropped_tm = None

        # Resize if requested
        if target_size:
            cropped_pre = resize_array(cropped_pre, target_size)
            cropped_bm = resize_array(cropped_bm, target_size, is_label=True)
            if cropped_gt is not None:
                cropped_gt = resize_array(cropped_gt, target_size)
            if cropped_tm is not None:
                cropped_tm = resize_array(cropped_tm, target_size, is_label=True)

        # Save
        case_id = f"{patient_id}_{side}"
        save_mha(cropped_pre, output_dirs["input"] / f"{case_id}.mha")
        if cropped_gt is not None:
            save_mha(cropped_gt, output_dirs["gt"] / f"{case_id}.mha")
        if cropped_tm is not None:
            save_mha(cropped_tm, output_dirs["mask"] / f"{case_id}.mha", is_label=True)
        if mask_output:
            save_mha(cropped_bm, output_dirs["breast_mask"] / f"{case_id}.mha", is_label=True)

        generated.append(case_id)

    return generated


def main():
    parser = argparse.ArgumentParser(
        description="Split bilateral breast MHA slices into unilateral crops (chest removed)."
    )
    parser.add_argument("--input_dir", required=True, help="Dir with bilateral pre-contrast .mha")
    parser.add_argument("--gt_dir", default=None, help="Dir with bilateral ground_truth .mha")
    parser.add_argument("--mask_dir", default=None, help="Dir with tumour mask .mha")
    parser.add_argument("--breast_mask_dir", required=True, help="Dir with breast mask .mha")
    parser.add_argument("--output_dir", required=True, help="Output root directory")
    parser.add_argument("--target_size", type=int, default=None,
                        help="Resize each unilateral crop to NxN (e.g. 256)")
    parser.add_argument("--pad_ratio", type=float, default=0.05,
                        help="Padding around breast bbox as ratio of image size (default: 0.05)")
    parser.add_argument("--no_breast_mask_output", action="store_true",
                        help="Don't save cropped breast masks")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    gt_dir = Path(args.gt_dir) if args.gt_dir else None
    mask_dir = Path(args.mask_dir) if args.mask_dir else None
    breast_mask_dir = Path(args.breast_mask_dir)
    output_dir = Path(args.output_dir)

    # Create output directories
    out_input = output_dir / "mha" / "input"
    out_gt = output_dir / "mha" / "ground_truth"
    out_mask = output_dir / "mha" / "mask"
    out_bm = output_dir / "mha" / "breast_mask"
    for d in (out_input, out_gt, out_mask, out_bm):
        d.mkdir(parents=True, exist_ok=True)
    output_dirs = {"input": out_input, "gt": out_gt, "mask": out_mask, "breast_mask": out_bm}

    # Discover cases
    mha_files = sorted(input_dir.glob("*.mha"))
    logger.info(f"Found {len(mha_files)} bilateral cases in {input_dir}")

    total_generated = 0
    for mha_path in mha_files:
        patient_id = mha_path.stem
        bm_path = breast_mask_dir / f"{patient_id}.mha"
        if not bm_path.exists():
            logger.warning(f"SKIP {patient_id}: no breast mask at {bm_path}")
            continue

        gt_path = (gt_dir / f"{patient_id}.mha") if gt_dir else None
        mask_path = (mask_dir / f"{patient_id}.mha") if mask_dir else None

        generated = process_patient(
            patient_id=patient_id,
            input_path=mha_path,
            gt_path=gt_path,
            mask_path=mask_path,
            breast_mask_path=bm_path,
            output_dirs=output_dirs,
            target_size=args.target_size,
            pad_ratio=args.pad_ratio,
            mask_output=not args.no_breast_mask_output,
        )
        total_generated += len(generated)
        if generated:
            logger.info(f"  ✓ {patient_id} → {generated}")

    logger.info(f"\nDone. Generated {total_generated} unilateral crops from {len(mha_files)} bilateral cases → {output_dir}")


if __name__ == "__main__":
    main()
