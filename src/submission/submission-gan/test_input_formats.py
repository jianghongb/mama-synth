#!/usr/bin/env python3
"""Test inference.py I/O pipeline against various input formats.

Mocks the GAN model to test the full input→preprocess→postprocess→output
pipeline without needing model weights or GPU.

Usage:
    cd /Users/ehogjig/git/kth/mama-synth
    .venv/bin/python src/submission/submission-gan/test_input_formats.py
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk
from PIL import Image

os.environ.setdefault("MAMA_GPU_ID", "-1")

SCRIPT_DIR = Path(__file__).parent
STATS_FILE = SCRIPT_DIR / "training_pre_stats.json"


def fake_generate(model_file, image_size, input_path, output_path, num_samples, save_images, gpu_id):
    """Mock GAN: reads input PNG → saves a grayscale 'synthetic' output."""
    in_dir = Path(input_path)
    out_dir = Path(output_path)
    for png in sorted(in_dir.glob("*.png")):
        img = Image.open(png).convert("L")
        arr = np.array(img, dtype=np.float32)
        # Simulate contrast enhancement: brighten by 20%
        arr = np.clip(arr * 1.2, 0, 255).astype(np.uint8)
        out_img = Image.fromarray(arr, mode="L")
        stem = png.stem
        out_img.save(str(out_dir / f"{stem}_syn_0.jpg"))


def run_inference(input_dir: Path, output_dir: Path) -> int:
    """Run inference.py with mocked model."""
    os.environ["MAMA_INPUT_DIR"] = str(input_dir)
    os.environ["MAMA_OUTPUT_DIR"] = str(output_dir)
    os.environ["MAMA_STATS_FILE"] = str(STATS_FILE)
    os.environ["MAMA_MODELS_DIR"] = str(SCRIPT_DIR / "models")

    mod_name = "inference_under_test"
    if mod_name in sys.modules:
        del sys.modules[mod_name]

    spec = importlib.util.spec_from_file_location(mod_name, SCRIPT_DIR / "inference.py")
    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
        # Mock the model module's generate function
        with patch.object(module, "_load_model_module") as mock_load:
            class FakeModel:
                generate = staticmethod(fake_generate)
            mock_load.return_value = FakeModel()
            return module.run()
    except Exception as e:
        print(f"  EXCEPTION: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return -1


def validate_output(output_dir: Path, input_shape: tuple) -> list[str]:
    """Validate output .mha. Returns list of issues."""
    issues = []
    slug = os.environ.get("MAMA_PREDICTION_SLUG", "synthetic-contrast-dce-mri-slice-breast")
    out_path = output_dir / "images" / slug / "output.mha"

    if not out_path.exists():
        return ["Output file not created"]

    img = sitk.ReadImage(str(out_path))
    arr = sitk.GetArrayFromImage(img)

    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim != 2:
        issues.append(f"Output is not 2D: shape={arr.shape}")

    if arr.dtype != np.float32:
        issues.append(f"Output dtype is {arr.dtype}, expected float32")

    expected_h, expected_w = input_shape[-2], input_shape[-1]
    if arr.ndim == 2 and (arr.shape[0] != expected_h or arr.shape[1] != expected_w):
        issues.append(f"Shape mismatch: input ({expected_h},{expected_w}) vs output {arr.shape}")

    if np.any(np.isnan(arr)):
        issues.append("Output contains NaN")
    if np.any(np.isinf(arr)):
        issues.append("Output contains Inf")

    if np.max(np.abs(arr)) > 100:
        issues.append(f"Z-score range too large: max |z|={np.max(np.abs(arr)):.1f}")

    return issues


def generate_test_cases():
    """Generate various input formats."""
    cases = []

    # 1. Standard z-score float32, 512x512
    arr = np.random.randn(512, 512).astype(np.float32)
    cases.append(("zscore_512x512_float32", arr, sitk.sitkFloat32))

    # 2. Non-square (like the GC failure: 1070x605)
    arr = np.random.randn(1070, 605).astype(np.float32)
    cases.append(("zscore_1070x605_float32", arr, sitk.sitkFloat32))

    # 3. Very small
    arr = np.random.randn(64, 64).astype(np.float32)
    cases.append(("zscore_64x64_float32", arr, sitk.sitkFloat32))

    # 4. Very large
    arr = np.random.randn(2048, 1024).astype(np.float32)
    cases.append(("zscore_2048x1024_float32", arr, sitk.sitkFloat32))

    # 5. Raw uint16 non-square (actual GC format)
    arr = (np.random.rand(1070, 605) * 4095).astype(np.uint16)
    cases.append(("raw_uint16_1070x605", arr, sitk.sitkUInt16))

    # 6. Raw uint16 square
    arr = (np.random.rand(512, 512) * 4095).astype(np.uint16)
    cases.append(("raw_uint16_512x512", arr, sitk.sitkUInt16))

    # 7. 3D (1, H, W) — SimpleITK may read 2D as 3D
    arr = np.random.randn(1, 256, 256).astype(np.float32)
    cases.append(("zscore_3d_1x256x256", arr, sitk.sitkFloat32))

    # 8. Float64
    arr = np.random.randn(512, 512).astype(np.float64)
    cases.append(("zscore_512x512_float64", arr, sitk.sitkFloat64))

    # 9. All zeros (flat image)
    arr = np.zeros((512, 512), dtype=np.float32)
    cases.append(("all_zeros_512x512", arr, sitk.sitkFloat32))

    # 10. Extreme z-scores
    arr = np.random.randn(512, 512).astype(np.float32) * 10
    cases.append(("extreme_zscore_512x512", arr, sitk.sitkFloat32))

    # 11. Odd dimensions
    arr = np.random.randn(333, 777).astype(np.float32)
    cases.append(("zscore_333x777_float32", arr, sitk.sitkFloat32))

    # 12. With custom spacing/origin (metadata preservation)
    cases.append(("zscore_with_metadata", None, None))

    return cases


def create_mha_with_metadata(path: Path):
    """Create .mha with non-default spacing and origin."""
    arr = np.random.randn(512, 512).astype(np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((0.5, 0.7))
    img.SetOrigin((10.0, 20.0))
    sitk.WriteImage(img, str(path))


def main():
    cases = generate_test_cases()
    slug = "pre-contrast-dce-mri-slice-breast"
    results = []

    print("=" * 70)
    print("  INPUT FORMAT ROBUSTNESS TESTS (mocked model)")
    print("=" * 70)
    print()

    for name, arr, pixel_type in cases:
        print(f"{'─' * 70}")
        print(f"TEST: {name}")
        print(f"{'─' * 70}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            in_dir = tmp / "input" / "images" / slug
            out_dir = tmp / "output"
            in_dir.mkdir(parents=True)
            out_dir.mkdir()

            mha_path = in_dir / "test_case.mha"

            if name == "zscore_with_metadata":
                create_mha_with_metadata(mha_path)
                input_shape = (512, 512)
            else:
                img = sitk.GetImageFromArray(arr)
                if pixel_type:
                    img = sitk.Cast(img, pixel_type)
                sitk.WriteImage(img, str(mha_path))
                input_shape = arr.shape

            print(f"  Input shape: {input_shape}")

            ret = run_inference(tmp / "input", out_dir)

            if ret == 0:
                issues = validate_output(out_dir, input_shape)
                if issues:
                    status = "⚠️  WARN"
                    detail = "; ".join(issues)
                else:
                    status = "✅ PASS"
                    detail = ""
            elif ret == 1:
                status = "❌ FAIL (code 1)"
                detail = "inference.py returned error"
            else:
                status = "❌ FAIL (exception)"
                detail = "see above"

            results.append((name, status, detail))
            print(f"  Result: {status} {detail}")
            print()

    # Summary
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for name, status, detail in results:
        extra = f" — {detail}" if detail else ""
        print(f"  {status}  {name}{extra}")

    n_pass = sum(1 for _, s, _ in results if "PASS" in s)
    n_warn = sum(1 for _, s, _ in results if "WARN" in s)
    n_total = len(results)
    print(f"\n  {n_pass}/{n_total} passed, {n_warn} warnings")
    return 0 if (n_pass + n_warn) == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
