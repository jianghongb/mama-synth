#!/usr/bin/env python3
"""Stamp a signature image into the signature box of the Springer LTP form PDF.

Word silently drops a programmatically inserted DrawingML picture from this
template, so the signature is applied to the exported PDF instead. The box is
located from the page geometry rather than hard-coded coordinates: the form's
signature cell is the table cell whose top-left corner sits directly below the
"Signed for and on behalf of the" label.

The signatory should review the output: this affixes a signature image on their
behalf, and the form is a legal declaration of responsibility.

Usage:
    python stamp_signature_pdf.py <in.pdf> <signature.png> <out.pdf> [--width-pt 150]
"""
import argparse
from pathlib import Path

import fitz

LABEL = "Signed for and on behalf"
# Table borders in this form are drawn as thin filled rectangles, not strokes.
RULE_MAX_THICKNESS = 1.0


def find_signature_box(page: fitz.Page) -> fitz.Rect:
    """Return the inner rectangle of the signature cell on the given page.

    The cell is bounded by four thin filled rectangles. The two vertical ones
    are found first, since they are unambiguous: they are the only tall thin
    fills that start just below the label.

    Raises:
        SystemExit: if the label or the surrounding rules cannot be found.
    """
    hits = page.search_for(LABEL)
    if not hits:
        raise SystemExit(f"{LABEL!r} not found on page {page.number + 1}")
    label = hits[0]

    verticals = [
        d["rect"]
        for d in page.get_drawings()
        if d["rect"].width <= RULE_MAX_THICKNESS
        and d["rect"].height > 20
        and label.y1 < d["rect"].y0 < label.y1 + 30
    ]
    if len(verticals) < 2:
        raise SystemExit("could not find the signature cell's vertical rules")

    # Keep the pair nearest the label: the cell the label belongs to.
    verticals.sort(key=lambda r: abs(r.x0 - label.x0))
    left, right = sorted(verticals[:2], key=lambda r: r.x0)
    if right.x0 - left.x1 < 50:
        raise SystemExit("signature cell rules are implausibly close together")

    return fitz.Rect(left.x1, left.y0, right.x0, min(left.y1, right.y1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("png")
    ap.add_argument("out")
    ap.add_argument("--page", type=int, default=0,
                    help="1-based page number; 0 searches every page")
    ap.add_argument("--width-pt", type=float, default=150.0,
                    help="signature width in points")
    ap.add_argument("--inset-pt", type=float, default=10.0,
                    help="left inset from the box edge")
    args = ap.parse_args()

    png = Path(args.png)
    doc = fitz.open(args.pdf)
    try:
        pages = [doc[args.page - 1]] if args.page else list(doc)
        page = next((p for p in pages if p.search_for(LABEL)), None)
        if page is None:
            raise SystemExit(f"{LABEL!r} not found in {args.pdf}")

        box = find_signature_box(page)
        with fitz.open(png) as im:
            aspect = im[0].rect.width / im[0].rect.height

        width = min(args.width_pt, box.width - 2 * args.inset_pt)
        height = width / aspect
        if height > box.height - 4:
            height = box.height - 4
            width = height * aspect

        x0 = box.x0 + args.inset_pt
        y0 = box.y0 + (box.height - height) / 2
        target = fitz.Rect(x0, y0, x0 + width, y0 + height)
        if not fitz.Rect(box).contains(target):
            raise SystemExit(f"signature {target} would fall outside the box {box}")

        page.insert_image(target, filename=str(png), keep_proportion=True)
        print(f"page {page.number + 1}: box {box}")
        print(f"stamped signature at {target} ({width:.1f} x {height:.1f} pt)")

        doc.save(args.out, garbage=3, deflate=True)
    finally:
        doc.close()

    print(f"wrote {args.out}")
    print("Have the signatory review this PDF before submission.")


if __name__ == "__main__":
    main()
