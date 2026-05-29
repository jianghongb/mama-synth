#!/usr/bin/env python3
"""MAMA-SYNTH — Pix2PixHD GAN Baseline (medigan model 00023).

Grand Challenge I/O contract
-----------------------------
Input  (one file per job):
    /input/images/pre-contrast-dce-mri-slice-breast/<uuid>.mha
    2-D float32 .mha, z-score-normalised with MAMA-SYNTH pre-contrast stats.

Output (one file per job):
    /output/images/synthetic-contrast-dce-mri-slice-breast/output.mha
    2-D float32 .mha, z-score-normalised with the same reference stats.

Normalisation strategy
----------------------
The Pix2PixHD model was trained on 8-bit grayscale PNG images drawn from
raw (un-normalised) MRI intensities.  The challenge pipeline delivers
z-score-normalised float32 inputs.  We bridge the gap as follows:

  Forward pass (MHA → PNG for model input):
    1. Invert z-score → approximate raw MRI intensity:
           raw = z * STD + MEAN     (training_pre_stats.json)
    2. Clip to [0, ∞)               (MRI intensities are non-negative)
    3. Record (raw_min, raw_max) per image
    4. Scale linearly to [0, 255] uint8  (per-image min-max, same as
       training-time normalisation in the reference synthesis pipeline)

  Inverse pass (model PNG output → MHA):
    1. Rescale [0, 255] → approximate raw using the input's intensity range:
           raw_out = (png / 255) * (raw_max - raw_min) + raw_min
       (Pre- and post-contrast breast DCE-MRI share similar intensity
       support from the same acquisition; using the input range preserves
       the natural intensity contrast without introducing per-output bias.)
    2. Re-normalise to z-score:
           z_out = (raw_out - MEAN) / STD

This keeps the output on the same scale as all other images in the
evaluation pipeline while giving the GAN inputs that match its training
distribution as closely as possible without access to the original raw data.

Environment overrides
---------------------
    MAMA_INPUT_DIR        default: /input
    MAMA_OUTPUT_DIR       default: /output
    MAMA_INPUT_SLUG       default: pre-contrast-dce-mri-slice-breast
    MAMA_PREDICTION_SLUG  default: synthetic-contrast-dce-mri-slice-breast
    MAMA_MODELS_DIR       default: /opt/app/models
    MAMA_STATS_FILE       default: /opt/app/training_pre_stats.json
    MAMA_GPU_ID           default: 0   (set to -1 to force CPU)
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
from glob import glob
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image

# ---------------------------------------------------------------------------
# Paths (all overridable for local testing)
# ---------------------------------------------------------------------------
INPUT_PATH = Path(os.environ.get("MAMA_INPUT_DIR", "/input"))
OUTPUT_PATH = Path(os.environ.get("MAMA_OUTPUT_DIR", "/output"))
INPUT_SLUG = os.environ.get(
    "MAMA_INPUT_SLUG", "pre-contrast-dce-mri-slice-breast"
)
OUTPUT_SLUG = os.environ.get(
    "MAMA_PREDICTION_SLUG", "synthetic-contrast-dce-mri-slice-breast"
)
MODELS_DIR = Path(os.environ.get("MAMA_MODELS_DIR", "/opt/app/models"))
STATS_FILE = Path(
    os.environ.get("MAMA_STATS_FILE", "/opt/app/training_pre_stats.json")
)

MODEL_ID = "00023"
MODEL_WEIGHTS = MODELS_DIR / MODEL_ID / "30_net_G.pth"

# Pix2PixHD expects 512×512 input
MODEL_SIZE = 512


# ---------------------------------------------------------------------------
# Pre-contrast normalisation stats
# ---------------------------------------------------------------------------

def load_stats() -> tuple[float, float]:
    """Load mean and std from training_pre_stats.json."""
    with open(STATS_FILE) as fh:
        stats = json.load(fh)
    mean = float(stats["mean"])
    std = float(stats["std"])
    print(f"Stats loaded: mean={mean:.2f}  std={std:.2f}  (from {STATS_FILE})")
    return mean, std


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def zscore_to_png(
    arr: np.ndarray, mean: float, std: float
) -> tuple[np.ndarray, float, float]:
    """Convert a z-score float32 2-D image to uint8 PNG [0, 255].

    Returns
    -------
    png_uint8 : np.ndarray  uint8 [0, 255]
    raw_min   : float       per-image minimum in approximate raw intensity space
    raw_max   : float       per-image maximum in approximate raw intensity space

    The raw_min / raw_max are returned so the inverse transform can recover
    a consistent intensity scale from the model's output.
    """
    # Step 1: invert z-score → approximate raw MRI intensity
    raw = arr.astype(np.float64) * std + mean
    # Step 2: clip negatives (MRI intensities are non-negative)
    raw = np.clip(raw, 0.0, None)
    # Step 3: per-image min-max scale to [0, 255]
    raw_min = float(raw.min())
    raw_max = float(raw.max())
    if raw_max > raw_min:
        png = (raw - raw_min) / (raw_max - raw_min) * 255.0
    else:
        # Flat image (e.g. pure background) — map to mid-range
        png = np.full_like(raw, 128.0)
    return png.astype(np.uint8), raw_min, raw_max


def png_to_zscore(
    png_arr: np.ndarray,
    raw_min: float,
    raw_max: float,
    mean: float,
    std: float,
) -> np.ndarray:
    """Convert a uint8 [0, 255] model output back to z-score float32.

    Uses the input image's (raw_min, raw_max) to define the intensity scale,
    then applies the MAMA-SYNTH pre-contrast z-score normalisation.
    """
    # Step 1: rescale to approximate raw intensity using input's range
    raw_out = png_arr.astype(np.float64) / 255.0 * (raw_max - raw_min) + raw_min
    # Step 2: z-score normalise with pre-contrast reference stats
    z_out = (raw_out - mean) / std
    return z_out.astype(np.float32)


# ---------------------------------------------------------------------------
# Model import (importlib needed because "00023" starts with a digit)
# ---------------------------------------------------------------------------

def _load_model_module():
    """Import the medigan Pix2PixHD model package via importlib.

    Directory layout expected in the container::

        /opt/app/
            models/
                __init__.py    (empty — makes models/ a Python package)
                00023/
                    __init__.py   (contains generate() function)
                    30_net_G.pth
                    src/
                        prepost_data/
                        prepost_model/
                        prepost_options/
                        prepost_util/

    importlib.import_module is used instead of a normal import because
    Python identifiers cannot start with a digit; importlib bypasses this
    restriction when the directory is on sys.path as a package.
    """
    # Only the *parent* of models/ needs to be on sys.path so that
    # importlib.import_module("models.00023") resolves to
    # /opt/app/models/00023/__init__.py.
    # Adding models/ itself would make Python search for models/00023
    # inside /opt/app/models, not /opt/app.
    app_dir = str(MODELS_DIR.parent)  # /opt/app
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)

    # Ensure models/__init__.py exists so Python treats it as a package
    init_file = MODELS_DIR / "__init__.py"
    if not init_file.exists():
        init_file.touch()

    module = importlib.import_module(f"models.{MODEL_ID}")
    return module


# ---------------------------------------------------------------------------
# Input discovery
# ---------------------------------------------------------------------------

def find_input_image() -> Path:
    """Return the single input .mha file for this GC job."""
    search_dir = INPUT_PATH / "images" / INPUT_SLUG
    candidates: list[str] = []
    for ext in ("*.mha", "*.nii.gz", "*.nii"):
        candidates.extend(glob(str(search_dir / ext)))

    if not candidates:
        # Emit a detailed diagnostic to help debug slug mismatches
        lines = [
            f"No input image found in: {search_dir}",
            f"INPUT_SLUG: '{INPUT_SLUG}'",
            "",
            "Contents of /input/images (if present):",
        ]
        images_dir = INPUT_PATH / "images"
        if images_dir.exists():
            for entry in sorted(images_dir.rglob("*")):
                lines.append(f"  {entry}")
        else:
            lines.append(f"  {images_dir} does not exist")
        raise FileNotFoundError("\n".join(lines))

    if len(candidates) > 1:
        print(
            f"WARNING: {len(candidates)} files found; using first: {candidates[0]}",
            file=sys.stderr,
        )
    return Path(candidates[0])


# ---------------------------------------------------------------------------
# Main inference
# ---------------------------------------------------------------------------

def run() -> int:
    print("=" * 60)
    print("  MAMA-SYNTH Pix2PixHD GAN Baseline")
    print("=" * 60)

    # ---- 1. Load pre-contrast normalisation stats --------------------
    mean, std = load_stats()

    # ---- 2. Locate input image ---------------------------------------
    input_file = find_input_image()
    print(f"Input : {input_file}")

    # ---- 3. Load as float32 2-D array --------------------------------
    sitk_img = sitk.ReadImage(str(input_file))
    arr = sitk.GetArrayFromImage(sitk_img)  # [H, W] or [1, H, W]

    # Squeeze a leading size-1 dimension (SimpleITK may add one for .mha)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        print(
            f"ERROR: expected 2-D input, got shape {arr.shape}",
            file=sys.stderr,
        )
        return 1

    orig_h, orig_w = arr.shape
    print(f"  Shape  : {arr.shape}   dtype: {arr.dtype}")
    print(f"  z-score range: [{float(arr.min()):.3f}, {float(arr.max()):.3f}]")

    # ---- 4. Z-score → PNG [0, 255] -----------------------------------
    # Detect if input is raw intensity (uint16 or values clearly not z-score)
    is_raw_input = (arr.dtype == np.uint16) or (arr.min() >= 0 and arr.max() > 50)
    if is_raw_input:
        print("  Detected raw intensity input (not z-score). Using direct min-max scaling.")
        raw = arr.astype(np.float64)
        raw_min, raw_max = float(raw.min()), float(raw.max())
        if raw_max > raw_min:
            png_arr = ((raw - raw_min) / (raw_max - raw_min) * 255).astype(np.uint8)
        else:
            png_arr = np.full_like(raw, 128, dtype=np.uint8)
    else:
        png_arr, raw_min, raw_max = zscore_to_png(arr, mean, std)
    print(f"  Raw intensity range: [{raw_min:.1f}, {raw_max:.1f}]")

    # Convert to PIL; resize to MODEL_SIZE×MODEL_SIZE for the GAN
    pil_input = Image.fromarray(png_arr, mode="L")
    if pil_input.size != (MODEL_SIZE, MODEL_SIZE):
        print(
            f"  Resizing {pil_input.size[0]}×{pil_input.size[1]}"
            f" → {MODEL_SIZE}×{MODEL_SIZE} for GAN input",
        )
        pil_input = pil_input.resize((MODEL_SIZE, MODEL_SIZE), Image.BICUBIC)

    # ---- 5. Run Pix2PixHD inside a temp directory --------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        in_dir = tmp / "input"
        out_dir = tmp / "output"
        in_dir.mkdir()
        out_dir.mkdir()

        # Save as PNG with a short, predictable name
        pil_input.save(str(in_dir / "input_0.png"))

        # Resolve GPU
        gpu_id_env = os.environ.get("MAMA_GPU_ID", "0")
        try:
            gpu_id = int(gpu_id_env)
        except ValueError:
            gpu_id = 0

        import torch  # already in base image; checked here for clarity
        if not torch.cuda.is_available():
            print("  CUDA not available — switching to CPU (gpu_id=-1)")
            gpu_id = -1
        else:
            device_name = torch.cuda.get_device_name(0) if gpu_id >= 0 else "cpu"
            print(f"  Device : {device_name}")

        print(f"  Running Pix2PixHD inference (gpu_id={gpu_id}) ...")

        # Import and call the model
        model_module = _load_model_module()
        model_module.generate(
            model_file=str(MODEL_WEIGHTS),
            image_size=MODEL_SIZE,
            input_path=str(in_dir),
            output_path=str(out_dir),
            num_samples=1,
            save_images=True,
            gpu_id=gpu_id,
        )

        # The model saves: {stem}_syn_{i}.jpg  → "input_0_syn_0.jpg"
        outputs = sorted(out_dir.glob("*_syn_*.jpg"))
        if not outputs:
            # Fallback: any image file the model may have produced
            outputs = sorted(
                f for f in out_dir.rglob("*")
                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            )
        if not outputs:
            contents = list(out_dir.rglob("*"))
            print(
                f"ERROR: Pix2PixHD produced no output images.\n"
                f"  out_dir contents: {contents}",
                file=sys.stderr,
            )
            return 1

        out_jpg = outputs[0]
        print(f"  Model output  : {out_jpg.name}")

        # ---- 6. Load model output ------------------------------------
        pil_out = Image.open(str(out_jpg)).convert("L")  # [0, 255] grayscale

        # Resize back to original spatial dimensions if needed
        if pil_out.size != (orig_w, orig_h):
            print(
                f"  Resizing output {pil_out.size[0]}×{pil_out.size[1]}"
                f" → {orig_w}×{orig_h}",
            )
            pil_out = pil_out.resize((orig_w, orig_h), Image.BICUBIC)

        out_png_arr = np.array(pil_out, dtype=np.float32)  # [H, W], [0, 255]

        # ---- 7. PNG → z-score float32 --------------------------------
        if is_raw_input:
            # For raw input: scale output back to raw range, then z-score
            raw_out = out_png_arr / 255.0 * (raw_max - raw_min) + raw_min
            z_out = ((raw_out - mean) / std).astype(np.float32)
        else:
            z_out = png_to_zscore(out_png_arr, raw_min, raw_max, mean, std)
        print(
            f"  Output z-score range: [{float(z_out.min()):.3f}, {float(z_out.max()):.3f}]"
        )

    # ---- 8. Write output .mha preserving input metadata --------------
    out_sitk = sitk.GetImageFromArray(z_out)
    # Only copy metadata if dimensions match; otherwise set spacing manually
    if out_sitk.GetDimension() == sitk_img.GetDimension():
        out_sitk.CopyInformation(sitk_img)
    else:
        # Input was read as 3D but output is 2D — copy spacing for available dims
        spacing = sitk_img.GetSpacing()[:out_sitk.GetDimension()]
        out_sitk.SetSpacing(spacing)

    out_dir_gc = OUTPUT_PATH / "images" / OUTPUT_SLUG
    out_dir_gc.mkdir(parents=True, exist_ok=True)
    out_path = out_dir_gc / "output.mha"
    sitk.WriteImage(out_sitk, str(out_path))

    print(f"Output: {out_path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(run())
