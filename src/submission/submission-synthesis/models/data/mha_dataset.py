"""MHA Dataset for MAMA-SYNTH Grand Challenge.

Reads pre-contrast (input), post-contrast subtraction (ground_truth),
and tumour mask (mask) from z-score normalized float32 MHA files.

Data layout expected:
    dataroot/
        mha/
            input/          *.mha  (pre-contrast, z-score float32)
            ground_truth/   *.mha  (subtraction, z-score float32)
            mask/           *.mha  (binary int16)

Training mode (resize_or_crop != 'none'):
    All images resized to loadSize x loadSize for batched training.

Inference mode (resize_or_crop == 'none'):
    Images padded to nearest multiple of 2^n_downsample_global.
    Original size stored for cropping output back.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk
from data.base_dataset import BaseDataset


class MhaDataset(BaseDataset):
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

        # Filter out non-square (sagittal) cases during training
        if getattr(opt, 'square_only', False) and opt.isTrain:
            self.filenames = []
            for f in all_files:
                img = sitk.ReadImage(os.path.join(self.dir_input, f))
                s = img.GetSize()
                if s[0] == s[1]:
                    self.filenames.append(f)
            print(f'[MhaDataset] square_only: kept {len(self.filenames)}/{len(all_files)} cases')
        else:
            self.filenames = all_files

        self.dataset_size = len(self.filenames)

        # Padding base: must be divisible by 2^n_downsample_global
        self.pad_base = 2 ** opt.n_downsample_global
        # Whether to resize to fixed size (for batched training)
        self.fixed_size = opt.resize_or_crop != 'none'
        self.target_size = opt.loadSize if self.fixed_size else None

    def __getitem__(self, index):
        fname = self.filenames[index]

        # Read input (pre-contrast)
        input_arr = self._read_mha(os.path.join(self.dir_input, fname))

        # Read ground truth (subtraction) if available
        gt_path = os.path.join(self.dir_gt, fname)
        if os.path.exists(gt_path):
            gt_arr = self._read_mha(gt_path)
        else:
            gt_arr = np.zeros_like(input_arr)

        # Read mask if available
        mask_path = os.path.join(self.dir_mask, fname)
        if os.path.exists(mask_path):
            mask_arr = sitk.GetArrayFromImage(
                sitk.ReadImage(mask_path)
            ).astype(np.float32)
        else:
            mask_arr = np.zeros_like(input_arr)

        # Store original shape
        input_arr = input_arr.squeeze()
        gt_arr = gt_arr.squeeze()
        mask_arr = mask_arr.squeeze()
        orig_h, orig_w = input_arr.shape

        # Convert to tensors [1, H, W]
        input_t = torch.from_numpy(input_arr).unsqueeze(0)
        gt_t = torch.from_numpy(gt_arr).unsqueeze(0)
        mask_t = torch.from_numpy(mask_arr).unsqueeze(0)

        if self.fixed_size:
            # Resize to fixed size for batched training
            s = self.target_size
            input_t = F.interpolate(input_t.unsqueeze(0), size=(s, s), mode='bilinear', align_corners=False).squeeze(0)
            gt_t = F.interpolate(gt_t.unsqueeze(0), size=(s, s), mode='bilinear', align_corners=False).squeeze(0)
            mask_t = F.interpolate(mask_t.unsqueeze(0), size=(s, s), mode='nearest').squeeze(0)
        else:
            # Pad to multiple of pad_base (for inference / batchSize=1)
            input_t = self._pad_tensor(input_t)
            gt_t = self._pad_tensor(gt_t)
            mask_t = self._pad_tensor(mask_t)

        # Random horizontal flip during training
        if self.opt.isTrain and not self.opt.no_flip and torch.rand(1).item() > 0.5:
            input_t = input_t.flip(-1)
            gt_t = gt_t.flip(-1)
            mask_t = mask_t.flip(-1)

        # Random intensity augmentation (improves cross-scanner generalization)
        if self.opt.isTrain and getattr(self.opt, 'intensity_aug', False):
            scale = 0.7 + torch.rand(1).item() * 0.6  # [0.7, 1.3]
            bias = (torch.rand(1).item() - 0.5) * 0.4  # [-0.2, 0.2]
            input_t = input_t * scale + bias
            gt_t = gt_t * scale + bias

        # Load breast mask (precomputed or threshold)
        breast_mask_t = None
        if self.dir_breast_mask and os.path.exists(os.path.join(self.dir_breast_mask, fname)):
            bm_arr = sitk.GetArrayFromImage(
                sitk.ReadImage(os.path.join(self.dir_breast_mask, fname))
            ).astype(np.float32)
            bm_arr = (bm_arr.squeeze() > 0).astype(np.float32)
            breast_mask_t = torch.from_numpy(bm_arr).unsqueeze(0)
            if self.fixed_size:
                s = self.target_size
                breast_mask_t = F.interpolate(breast_mask_t.unsqueeze(0), size=(s, s), mode='nearest').squeeze(0)
            else:
                breast_mask_t = self._pad_tensor(breast_mask_t)
        elif getattr(self.opt, 'breast_mask', False):
            thresh = getattr(self.opt, 'breast_mask_thresh', -0.3)
            breast_mask_t = (input_t > thresh).float()

        # Random horizontal flip during training
        if self.opt.isTrain and not self.opt.no_flip and torch.rand(1).item() > 0.5:
            input_t = input_t.flip(-1)
            gt_t = gt_t.flip(-1)
            mask_t = mask_t.flip(-1)
            if breast_mask_t is not None:
                breast_mask_t = breast_mask_t.flip(-1)

        return {
            'label': input_t,
            'inst': torch.zeros(1),
            'image': gt_t,
            'feat': torch.zeros(1),
            'mask': mask_t,
            'breast_mask': breast_mask_t if breast_mask_t is not None else torch.ones_like(input_t),
            'path': os.path.join(self.dir_input, fname),
            'orig_size': (orig_h, orig_w),
        }

    def __len__(self):
        return self.dataset_size

    def name(self):
        return 'MhaDataset'

    def _read_mha(self, path):
        """Read MHA and return float32 numpy array."""
        img = sitk.ReadImage(path)
        return sitk.GetArrayFromImage(img).astype(np.float32)

    def _pad_tensor(self, t):
        """Pad [1, H, W] tensor so H and W are multiples of pad_base."""
        _, h, w = t.shape
        new_h = int(np.ceil(h / self.pad_base) * self.pad_base)
        new_w = int(np.ceil(w / self.pad_base) * self.pad_base)
        if new_h == h and new_w == w:
            return t
        # F.pad expects (left, right, top, bottom)
        return F.pad(t, (0, new_w - w, 0, new_h - h), mode='constant', value=0)
