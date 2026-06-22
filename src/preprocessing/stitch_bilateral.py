"""
stitch_bilateral.py — Inference utility for unilateral-trained models.

At inference time, the GC input is a bilateral (both breasts) axial slice.
If the model was trained on unilateral crops, this script:
1. Splits the bilateral input at the midline
2. Crops each side to the breast bounding box (chest removed)
3. Resizes to the model's expected input size
4. After model inference, resizes output back and places into full bilateral canvas

Usage (as a library):
    from split_bilateral import find_midline, crop_breast_side
    from stitch_bilateral import BilateralSplitter

    splitter = BilateralSplitter(target_size=256)
    left, right, meta = splitter.split(pre_contrast, breast_mask)
    # ... run model on left and right ...
    output = splitter.stitch(pred_left, pred_right, meta)
"""
import numpy as np
from typing import Optional, Tuple, Dict, Any


class BilateralSplitter:
    """Split bilateral breast slices into unilateral crops and stitch back."""

    def __init__(self, target_size: Optional[int] = None, pad_ratio: float = 0.05):
        self.target_size = target_size
        self.pad_ratio = pad_ratio

    def _find_midline(self, breast_mask: np.ndarray) -> Optional[int]:
        """Find column separating L/R breast. Returns None if single-breast."""
        h, w = breast_mask.shape
        col_sum = np.sum(breast_mask > 0, axis=0)

        half = w // 2
        left_mass = col_sum[:half].sum()
        right_mass = col_sum[half:].sum()
        total_mass = left_mass + right_mass

        if total_mass == 0:
            return None
        if left_mass / total_mass > 0.9 or right_mass / total_mass > 0.9:
            return None

        margin = int(w * 0.3)
        search_start, search_end = margin, w - margin
        if search_start >= search_end:
            return w // 2
        central_sums = col_sum[search_start:search_end]
        if central_sums.min() > h * 0.3:
            return None
        return search_start + int(np.argmin(central_sums))

    def _crop_side(self, image: np.ndarray, mask: np.ndarray,
                   col_start: int, col_end: int, pad_pixels: int
                   ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """Crop one side to breast bbox. Returns (cropped, bbox) or (None, None)."""
        side_mask = mask[:, col_start:col_end]
        side_img = image[:, col_start:col_end]

        rows = np.any(side_mask > 0, axis=1)
        cols = np.any(side_mask > 0, axis=0)
        if not rows.any() or not cols.any():
            return None, None

        r_min, r_max = np.where(rows)[0][[0, -1]]
        c_min, c_max = np.where(cols)[0][[0, -1]]

        r_min = max(0, r_min - pad_pixels)
        r_max = min(side_mask.shape[0] - 1, r_max + pad_pixels)
        c_min = max(0, c_min - pad_pixels)
        c_max = min(side_mask.shape[1] - 1, c_max + pad_pixels)

        cropped = side_img[r_min:r_max + 1, c_min:c_max + 1].copy()
        cropped *= (side_mask[r_min:r_max + 1, c_min:c_max + 1] > 0).astype(np.float32)

        bbox = (r_min, r_max + 1, col_start + c_min, col_start + c_max + 1)
        return cropped, bbox

    def _resize(self, arr: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Resize using PIL."""
        from PIL import Image
        return np.array(
            Image.fromarray(arr.astype(np.float32)).resize(
                (size[1], size[0]), Image.BICUBIC
            ), dtype=np.float32
        )

    def split(self, image: np.ndarray, breast_mask: np.ndarray
              ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
        """Split bilateral image into L/R crops.

        Returns:
            left_crop, right_crop, metadata (for stitching back)
        """
        h, w = image.shape
        breast_mask_bin = (breast_mask > 0).astype(np.float32)
        pad_pixels = max(3, int(max(h, w) * self.pad_ratio))
        midline = self._find_midline(breast_mask_bin)

        meta = {"orig_shape": (h, w), "midline": midline, "breast_mask": breast_mask_bin}
        crops = {}

        if midline is None:
            # Single breast: crop entire image
            sides = [("S", 0, w)]
        else:
            sides = [("L", 0, midline), ("R", midline, w)]

        for side, cs, ce in sides:
            cropped, bbox = self._crop_side(image, breast_mask_bin, cs, ce, pad_pixels)
            if cropped is not None:
                meta[f"{side}_bbox"] = bbox
                meta[f"{side}_crop_shape"] = cropped.shape
                if self.target_size:
                    cropped = self._resize(cropped, (self.target_size, self.target_size))
                crops[side] = cropped
            else:
                crops[side] = None
                meta[f"{side}_bbox"] = None

        meta["sides"] = [s[0] for s in sides]
        return crops.get("L", crops.get("S")), crops.get("R"), meta

    def stitch(self, pred_left: Optional[np.ndarray],
               pred_right: Optional[np.ndarray],
               meta: Dict[str, Any]) -> np.ndarray:
        """Stitch L/R (or single S) predictions back into full bilateral canvas."""
        h, w = meta["orig_shape"]
        canvas = np.zeros((h, w), dtype=np.float32)
        breast_mask = meta["breast_mask"]

        sides = meta.get("sides", ["L", "R"])
        preds = {"L": pred_left, "R": pred_right, "S": pred_left}

        for side in sides:
            pred = preds.get(side)
            bbox = meta.get(f"{side}_bbox")
            if pred is None or bbox is None:
                continue

            r0, r1, c0, c1 = bbox
            crop_h, crop_w = r1 - r0, c1 - c0

            # Resize prediction back to original crop size
            if pred.shape != (crop_h, crop_w):
                pred_resized = self._resize(pred, (crop_h, crop_w))
            else:
                pred_resized = pred

            # Place into canvas, masked by breast
            region_mask = breast_mask[r0:r1, c0:c1]
            canvas[r0:r1, c0:c1] += pred_resized * (region_mask > 0).astype(np.float32)

        return canvas
