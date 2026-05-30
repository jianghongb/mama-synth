#!/usr/bin/env python3
"""检查 MAMA-SYNTH 数据集的基本特征。

用法：
    python src/inspect_dataset.py --data_dir /path/to/mha/files

    # 如果有 mask 目录：
    python src/inspect_dataset.py \
        --data_dir /path/to/input \
        --mask_dir /path/to/masks \
        --gt_dir /path/to/ground_truth

输出：
    - 每个文件的 shape、dtype、值域、均值、标准差
    - 整体数据集统计汇总
    - 可选：保存直方图和示例图像到 output 目录
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False


def load_mha(path: Path) -> tuple[np.ndarray, dict]:
    """加载 .mha 文件，返回 numpy 数组和 metadata。"""
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    # squeeze 掉可能的 [1, H, W] 维度
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    meta = {
        "spacing": img.GetSpacing(),
        "origin": img.GetOrigin(),
        "direction": img.GetDirection(),
        "size": img.GetSize(),
        "pixel_type": img.GetPixelIDTypeAsString(),
    }
    return arr, meta


def compute_stats(arr: np.ndarray) -> dict:
    """计算单张图像的统计量。"""
    return {
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "pct_zero": float(np.sum(arr == 0) / arr.size * 100),
        "has_nan": bool(np.any(np.isnan(arr))),
        "has_inf": bool(np.any(np.isinf(arr))),
    }


def inspect_directory(data_dir: Path, label: str = "data") -> list[dict]:
    """检查目录中所有 .mha 文件。"""
    files = sorted(data_dir.glob("*.mha"))
    if not files:
        print(f"  ⚠️  {data_dir} 中没有找到 .mha 文件")
        return []

    print(f"\n{'='*60}")
    print(f"  {label}: {data_dir}")
    print(f"  文件数量: {len(files)}")
    print(f"{'='*60}")

    all_stats = []
    all_means = []
    all_stds = []
    all_mins = []
    all_maxs = []
    shapes = set()

    for i, f in enumerate(files):
        arr, meta = load_mha(f)
        stats = compute_stats(arr)
        stats["filename"] = f.name
        stats["meta"] = meta
        all_stats.append(stats)

        all_means.append(stats["mean"])
        all_stds.append(stats["std"])
        all_mins.append(stats["min"])
        all_maxs.append(stats["max"])
        shapes.add(stats["shape"])

        # 打印前 5 个和最后 1 个的详细信息
        if i < 5 or i == len(files) - 1:
            print(f"\n  [{i+1}/{len(files)}] {f.name}")
            print(f"    Shape: {stats['shape']}  dtype: {stats['dtype']}")
            print(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"    Mean:  {stats['mean']:.4f}  Std: {stats['std']:.4f}")
            print(f"    Zero%: {stats['pct_zero']:.1f}%  NaN: {stats['has_nan']}  Inf: {stats['has_inf']}")
            print(f"    Spacing: {meta['spacing']}  Origin: {meta['origin']}")
        elif i == 5:
            print(f"\n  ... (省略中间 {len(files) - 6} 个文件) ...")

    # 汇总统计
    print(f"\n{'─'*60}")
    print(f"  📊 汇总统计 ({label})")
    print(f"{'─'*60}")
    print(f"  文件数:     {len(files)}")
    print(f"  Shape 种类: {shapes}")
    print(f"  值域范围:   [{min(all_mins):.4f}, {max(all_maxs):.4f}]")
    print(f"  均值范围:   [{min(all_means):.4f}, {max(all_means):.4f}]")
    print(f"  均值的均值: {np.mean(all_means):.4f} ± {np.std(all_means):.4f}")
    print(f"  标准差范围: [{min(all_stds):.4f}, {max(all_stds):.4f}]")

    # 检查是否是 z-score 归一化的
    overall_mean = np.mean(all_means)
    overall_std_of_means = np.std(all_means)
    if abs(overall_mean) < 1.0 and min(all_mins) < -2 and max(all_maxs) > 2:
        print(f"\n  ✅ 数据看起来是 z-score 归一化的（均值≈0，有正负值）")
    elif min(all_mins) >= 0 and max(all_maxs) <= 255:
        print(f"\n  ⚠️  数据看起来是 [0, 255] 范围（可能是 uint8 或未归一化）")
    elif min(all_mins) >= 0 and max(all_maxs) <= 1:
        print(f"\n  ⚠️  数据看起来是 [0, 1] 范围")
    else:
        print(f"\n  ℹ️  数据范围不符合常见归一化模式，请检查")

    return all_stats


def inspect_masks(mask_dir: Path) -> list[dict]:
    """检查 mask 目录的特殊统计。"""
    files = sorted(mask_dir.glob("*.mha"))
    if not files:
        print(f"  ⚠️  {mask_dir} 中没有找到 .mha 文件")
        return []

    print(f"\n{'='*60}")
    print(f"  Masks: {mask_dir}")
    print(f"  文件数量: {len(files)}")
    print(f"{'='*60}")

    tumor_sizes = []
    tumor_fractions = []

    for f in files:
        arr, _ = load_mha(f)
        mask = arr > 0
        n_tumor = int(np.sum(mask))
        fraction = n_tumor / arr.size * 100
        tumor_sizes.append(n_tumor)
        tumor_fractions.append(fraction)

    print(f"\n  📊 肿瘤 Mask 统计:")
    print(f"  肿瘤像素数范围: [{min(tumor_sizes)}, {max(tumor_sizes)}]")
    print(f"  肿瘤面积占比:   [{min(tumor_fractions):.2f}%, {max(tumor_fractions):.2f}%]")
    print(f"  平均肿瘤面积:   {np.mean(tumor_fractions):.2f}% ± {np.std(tumor_fractions):.2f}%")
    empty = sum(1 for s in tumor_sizes if s == 0)
    if empty > 0:
        print(f"  ⚠️  空 mask 数量: {empty}/{len(files)}")

    return [{"filename": f.name, "tumor_pixels": s, "tumor_fraction": frac}
            for f, s, frac in zip(files, tumor_sizes, tumor_fractions)]


def save_visualizations(data_dir: Path, output_dir: Path, n_samples: int = 4):
    """保存示例图像和直方图。"""
    if not HAS_PLT:
        print("  ⚠️  matplotlib 未安装，跳过可视化")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(data_dir.glob("*.mha"))[:n_samples]

    if not files:
        return

    # 示例图像
    fig, axes = plt.subplots(1, len(files), figsize=(4 * len(files), 4))
    if len(files) == 1:
        axes = [axes]
    for ax, f in zip(axes, files):
        arr, _ = load_mha(f)
        ax.imshow(arr, cmap="gray")
        ax.set_title(f.stem[:20], fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Sample images from {data_dir.name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(str(output_dir / f"samples_{data_dir.name}.png"), dpi=150)
    plt.close(fig)

    # 值分布直方图
    fig, ax = plt.subplots(figsize=(8, 4))
    for f in files[:4]:
        arr, _ = load_mha(f)
        ax.hist(arr.ravel(), bins=100, alpha=0.5, label=f.stem[:15], density=True)
    ax.set_xlabel("Pixel value")
    ax.set_ylabel("Density")
    ax.set_title(f"Value distribution ({data_dir.name})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(str(output_dir / f"histogram_{data_dir.name}.png"), dpi=150)
    plt.close(fig)

    print(f"  📁 可视化已保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="检查 MAMA-SYNTH 数据集特征")
    parser.add_argument("--data_dir", required=True, help="输入图像目录（pre-contrast .mha）")
    parser.add_argument("--gt_dir", default=None, help="Ground truth 目录（post-contrast .mha）")
    parser.add_argument("--mask_dir", default=None, help="Mask 目录（肿瘤分割 .mha）")
    parser.add_argument("--output_dir", default=None, help="保存可视化的目录（可选）")
    parser.add_argument("--no_viz", action="store_true", help="跳过可视化")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"❌ 目录不存在: {data_dir}")
        sys.exit(1)

    # 检查输入数据
    inspect_directory(data_dir, label="Pre-contrast (input)")

    # 检查 ground truth
    if args.gt_dir:
        gt_dir = Path(args.gt_dir)
        if gt_dir.exists():
            inspect_directory(gt_dir, label="Post-contrast (ground truth)")

    # 检查 masks
    if args.mask_dir:
        mask_dir = Path(args.mask_dir)
        if mask_dir.exists():
            inspect_masks(mask_dir)

    # 可视化
    if not args.no_viz and args.output_dir:
        output_dir = Path(args.output_dir)
        save_visualizations(data_dir, output_dir)
        if args.gt_dir and Path(args.gt_dir).exists():
            save_visualizations(Path(args.gt_dir), output_dir)

    print(f"\n{'='*60}")
    print("  ✅ 检查完成")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
