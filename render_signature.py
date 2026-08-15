#!/usr/bin/env python3
"""Render a handwritten-signature SVG to a transparent PNG for Word.

Word's SVG support is inconsistent across versions, so a high-resolution PNG
with an alpha channel is the safer thing to paste into the licence form. The
signature is a set of stroked cubic Bezier paths with no fills, so parsing the
path data directly is enough; no full SVG renderer is needed.

Usage:
    python render_signature.py <in.svg> <out.png> [--dpi 600] [--pad 0.02]
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402

NUM = re.compile(r"-?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_path(d: str):
    """Parse an SVG path limited to absolute M/C/L commands into vertices+codes.

    Returns (vertices, codes) for matplotlib, or (None, None) if unsupported
    commands appear, so the caller can skip that path rather than mis-draw it.
    """
    tokens = re.findall(r"[MCLZmclz]|" + NUM.pattern, d)
    verts, codes = [], []
    i = 0
    while i < len(tokens):
        cmd = tokens[i]
        if cmd in "Mm":
            x, y = float(tokens[i + 1]), float(tokens[i + 2])
            verts.append((x, y))
            codes.append(MPath.MOVETO)
            i += 3
        elif cmd in "Cc":
            pts = [float(t) for t in tokens[i + 1:i + 7]]
            verts.extend([(pts[0], pts[1]), (pts[2], pts[3]), (pts[4], pts[5])])
            codes.extend([MPath.CURVE4] * 3)
            i += 7
        elif cmd in "Ll":
            verts.append((float(tokens[i + 1]), float(tokens[i + 2])))
            codes.append(MPath.LINETO)
            i += 3
        elif cmd in "Zz":
            i += 1
        else:
            # Bare coordinate pairs continue the previous cubic command.
            try:
                pts = [float(t) for t in tokens[i:i + 6]]
            except ValueError:
                return None, None
            if len(pts) < 6:
                return None, None
            verts.extend([(pts[0], pts[1]), (pts[2], pts[3]), (pts[4], pts[5])])
            codes.extend([MPath.CURVE4] * 3)
            i += 6
    return (verts, codes) if verts else (None, None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("svg")
    ap.add_argument("png")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--pad", type=float, default=0.02, help="margin as fraction of size")
    ap.add_argument("--lw", type=float, default=2.0, help="stroke width in points")
    args = ap.parse_args()

    svg = Path(args.svg).read_text()
    vb = re.search(r'viewBox="([\d.\-\s]+)"', svg)
    if not vb:
        raise SystemExit("no viewBox found")
    vx, vy, vw, vh = (float(v) for v in vb.group(1).split())

    paths = re.findall(r'<path[^>]*\sd="([^"]+)"', svg)
    if not paths:
        raise SystemExit("no <path> elements found")

    fig_w = 8.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * vh / vw))
    drawn = 0
    for d in paths:
        verts, codes = parse_path(d)
        if verts is None:
            print(f"  skipped an unsupported path ({d[:30]}...)")
            continue
        ax.add_patch(PathPatch(MPath(verts, codes), fill=False,
                               edgecolor="black", lw=args.lw,
                               capstyle="round", joinstyle="round"))
        drawn += 1

    mx, my = vw * args.pad, vh * args.pad
    ax.set_xlim(vx - mx, vx + vw + mx)
    ax.set_ylim(vy + vh + my, vy - my)  # SVG y grows downward
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(args.png, dpi=args.dpi, transparent=True,
                bbox_inches="tight", pad_inches=0.05)
    print(f"rendered {drawn}/{len(paths)} paths -> {args.png}")


if __name__ == "__main__":
    main()
