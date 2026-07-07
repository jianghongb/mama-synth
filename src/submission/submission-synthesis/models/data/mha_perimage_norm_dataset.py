"""MHA Dataset with Per-Image Z-Score Normalization.

Instead of relying on global z-score (pre-computed mean/std for entire dataset),
this dataset re-normalizes each image independently using the breast-region
mean and std. This eliminates cross-scanner intensity variation.

The model learns mappings in a canonical normalized space where every input
has mean≈0, std≈1 within the breast region.

At inference time:
    1. Compute per-image stats from input breast region
    2. Normalize input
    3. Run model → get normalized output
    4. De-normalize output back to original intensity space
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk
from data.base_dataset import BaseDataset


class MhaPerImageNormDataset(BaseDataset):
    """MHA dataset with per-image foreground z-score normalization."""

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        self.dir_input = os.path.join(opt.dataroot, 'mha', 'input')
        self.dir_gt = os.path.join(opt.dataroot, 'mha', 'ground_truth')
        self.dir_mask = os.path.join(opt.dataroot, 'mha', 'mask')
        self.dir_breast_mask = getattr(opt, 'breast_mask_dir', '') or ''

        all_files = sorted([
            f for f in os.listdir(self.dir_input) if f.endswith('.mha')
        ])
        self.filenames = all_files
        self.dataset_size = len(self.filenames)

        # Padding base: must be divisible by 2^n_downsample_global
        self.pad_base = 2 ** opt.n_downsample_global
        self.fixed_size = opt.resize_or_crop != 'none'
        self.target_size = opt.loadSize if self.fixed_size else None

    def __getitem__(self, index):
        fname = self.filenames[index]

        # Read raw arrays (these are already global z-score normalized from preprocessing)
        input_arr = self._read_mha(os.path.join(self.dir_input, fname)).squeeze()
        gt_path = os.path.join(self.dir_gt, fname)
        gt_arr = self._read_mha(gt_path).squeeze() if os.path.exists(gt_path) else np.zeros_like(input_arr)

        mask_path = os.path.join(self.dir_mask, fname)
        mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(mask_path)).astype(np.float32).squeeze() if os.path.exists(mask_path) else np.zeros_like(input_arr)

        # Load breast mask for foreground definition
        breast_mask_arr = None
        if self.dir_breast_mask and os.path.exists(os.path.join(self.dir_breast_mask, fname)):
            bm = sitk.GetArrayFromImage(
                sitk.ReadImage(os.path.join(self.dir_breast_mask, fname))
            ).astype(np.float32).squeeze()
            breast_mask_arr = (bm > 0).astype(np.float32)
        else:
            # Fallback: threshold-based breast region
            breast_mask_arr = (input_arr > -0.3).astype(np.float32)

        # === Per-image z-score normalization (foreground only) ===
        fg_pixels = input_arr[breast_mask_arr > 0.5]
        if fg_pixels.size > 100:
            img_mean = float(fg_pixels.mean())
            img_std = float(fg_pixels.std())
            img_std = max(img_std, 1e-8)  # prevent div by zero
        else:
            img_mean, img_std = 0.0, 1.0

        # Re-normalize both input and GT with per-image stats
        input_norm = (input_arr - img_mean) / img_std
        gt_norm = (gt_arr - img_mean) / img_std

        # Clip GT to prevent extreme values (P99 — less aggressive than P95)
        gt_p99 = float(np.percentile(gt_norm[breast_mask_arr > 0.5], 99)) if (breast_mask_arr > 0.5).sum() > 100 else 5.0
        gt_norm = np.clip(gt_norm, -2.0, max(gt_p99, 5.0))

        # Zero out background
        input_norm = input_norm * breast_mask_arr
        gt_norm = gt_norm * breast_mask_arr

        orig_h, orig_w = input_arr.shape

        # Convert to tensors [1, H, W]
        input_t = torch.from_numpy(input_norm.astype(np.float32)).unsqueeze(0)
        gt_t = torch.from_numpy(gt_norm.astype(np.float32)).unsqueeze(0)
        mask_t = torch.from_numpy(mask_arr).unsqueeze(0)
        breast_mask_t = torch.from_numpy(breast_mask_arr).unsqueeze(0)

        if self.fixed_size:
            s = self.target_size
            input_t = F.interpolate(input_t.unsqueeze(0), size=(s, s), mode='bilinear', align_corners=False).squeeze(0)
            gt_t = F.interpolate(gt_t.unsqueeze(0), size=(s, s), mode='bilinear', align_corners=False).squeeze(0)
            mask_t = F.interpolate(mask_t.unsqueeze(0), size=(s, s), mode='nearest').squeeze(0)
            breast_mask_t = F.interpolate(breast_mask_t.unsqueeze(0), size=(s, s), mode='nearest').squeeze(0)
        else:
            input_t = self._pad_tensor(input_t)
            gt_t = self._pad_tensor(gt_t)
            mask_t = self._pad_tensor(mask_t)
            breast_mask_t = self._pad_tensor(breast_mask_t)

        # Random horizontal flip
        if self.opt.isTrain and not self.opt.no_flip and torch.rand(1).item() > 0.5:
            input_t = input_t.flip(-1)
            gt_t = gt_t.flip(-1)
            mask_t = mask_t.flip(-1)
            breast_mask_t = breast_mask_t.flip(-1)

        # Random vertical flip
        if self.opt.isTrain and getattr(self.opt, 'vflip', False) and torch.rand(1).item() > 0.5:
            input_t = input_t.flip(-2)
            gt_t = gt_t.flip(-2)
            mask_t = mask_t.flip(-2)
            breast_mask_t = breast_mask_t.flip(-2)

        # Intensity augmentation (scale only, no bias — preserves normalization semantics)
        if self.opt.isTrain and getattr(self.opt, 'intensity_aug', False):
            scale = 0.8 + torch.rand(1).item() * 0.4  # [0.8, 1.2] — narrower range
            input_t = input_t * scale
            gt_t = gt_t * scale

        return {
            'label': input_t,
            'inst': torch.zeros(1),
            'image': gt_t,
            'feat': torch.zeros(1),
            'mask': mask_t,
            'breast_mask': breast_mask_t,
            'path': os.path.join(self.dir_input, fname),
            'orig_size': (orig_h, orig_w),
            'img_mean': img_mean,  # needed for de-normalization at inference
            'img_std': img_std,
        }

    def __len__(self):
        return self.dataset_size

    def name(self):
        return 'MhaPerImageNormDataset'

    def _read_mha(self, path):
        img = sitk.ReadImage(path)
        return sitk.GetArrayFromImage(img).astype(np.float32)

    def _pad_tensor(self, t):
        _, h, w = t.shape
        new_h = int(np.ceil(h / self.pad_base) * self.pad_base)
        new_w = int(np.ceil(w / self.pad_base) * self.pad_base)
        if new_h == h and new_w == w:
            return t
        return F.pad(t, (0, new_w - w, 0, new_h - h), mode='constant', value=0)
