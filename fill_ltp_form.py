#!/usr/bin/env python3
"""Fill the seven author-supplied fields in the Springer licence-to-publish form.

The form has two kinds of blanks. The title, author list and corresponding
author sit in content controls (w:sdt) carrying aliases Title, Author(s) and
Corresponding Author, and show "Click here to enter text." until filled. Print
Name, Date, Address and Email are legacy FORMTEXT fields whose displayed value
is the w:t run following the field separator.

Everything the organisers prefilled -- licensee, volume title, volume editors,
series, edition id, PS -- is left untouched, and the script asserts the
prefilled strings are still present afterwards.

Usage:
    python fill_ltp_form.py <in.docx> <out.docx>
"""
import argparse
import re
import shutil
import zipfile
from pathlib import Path

# Author-supplied values.
SDT_VALUES = {
    "Title": ("Tumor-Aware Conditional Pix2PixHD with Per-Image Normalization "
              "for Virtual Contrast Enhancement in Breast MRI"),
    "Author(s)": "Hong Jiang, Zhikai Yang, Rodrigo Moreno",
    "Corresponding Author": "Yang, Zhikai",
}
# FORMTEXT fields, in the order they appear after the signature block.
FORMTEXT_VALUES = [
    "Zhikai Yang",                              # Print Name
    "8 August 2026",                            # Date
    "Hälsovägen 11C, 141 57 Huddinge, Sweden",  # Address
    "zhikai@kth.se",                            # Email
]
# Prefilled strings that must survive untouched. Kept short because Word splits
# long runs, e.g. the volume title is broken after "Computer Assisted".
MUST_SURVIVE = [
    "Springer Nature Switzerland AG",
    "Medical Image Computing and Computer Assisted",
    "Lecture Notes in Computer Science",
    "Gewerbestrasse 11, 6330 Cham, Switzerland",
]

PLACEHOLDER = "Click here to enter text."


def esc(s: str) -> str:
    """Escape XML special characters for insertion into a w:t element."""
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def fill_sdt(xml: str) -> str:
    """Replace the placeholder text inside each aliased content control."""
    for alias, value in SDT_VALUES.items():
        # Locate the control by alias, then the first placeholder after it.
        m = re.search(r'<w:alias w:val="' + re.escape(alias) + r'"', xml)
        if not m:
            raise SystemExit(f"content control not found: {alias}")
        idx = xml.find(PLACEHOLDER, m.end())
        if idx == -1:
            raise SystemExit(f"placeholder not found for: {alias}")
        xml = xml[:idx] + esc(value) + xml[idx + len(PLACEHOLDER):]
        # Drop the grey "showing placeholder" flag so the text renders normally.
        seg_start = xml.rfind("<w:sdtPr>", 0, m.start())
        seg_end = xml.find("</w:sdtPr>", m.start())
        if seg_start != -1 and seg_end != -1:
            seg = xml[seg_start:seg_end]
            xml = xml[:seg_start] + seg.replace("<w:showingPlcHdr/>", "") + xml[seg_end:]
        print(f"  {alias}: {value[:60]}{'...' if len(value) > 60 else ''}")
    return xml


def fill_formtext(xml: str) -> str:
    """Set the displayed value of each FORMTEXT field after the signature block."""
    anchor = xml.find("Signed for and on behalf")
    if anchor == -1:
        raise SystemExit("signature block not found")

    # Each field looks like: fldChar separate ... <w:t>value</w:t> ... fldChar end
    positions = [m.start() for m in re.finditer(r'FORMTEXT', xml) if m.start() > anchor]
    if len(positions) < len(FORMTEXT_VALUES):
        raise SystemExit(f"expected {len(FORMTEXT_VALUES)} FORMTEXT fields, found {len(positions)}")

    # Work backwards so earlier offsets stay valid.
    for pos, value in reversed(list(zip(positions[:len(FORMTEXT_VALUES)], FORMTEXT_VALUES))):
        sep = xml.find('w:fldCharType="separate"', pos)
        end = xml.find('w:fldCharType="end"', pos)
        if sep == -1 or end == -1 or sep > end:
            raise SystemExit(f"malformed FORMTEXT field near offset {pos}")
        seg = xml[sep:end]
        m = re.search(r'(<w:t[^>]*>)([^<]*)(</w:t>)', seg)
        if not m:
            raise SystemExit(f"no text run inside FORMTEXT near offset {pos}")
        new_seg = seg[:m.start()] + m.group(1) + esc(value) + m.group(3) + seg[m.end():]
        xml = xml[:sep] + new_seg + xml[end:]
        print(f"  FORMTEXT: {value}")
    return xml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    shutil.copy(src, dst)

    with zipfile.ZipFile(src) as z:
        names = z.namelist()
        parts = {n: z.read(n) for n in names}

    xml = parts["word/document.xml"].decode("utf-8")
    original = xml

    print("Filling content controls:")
    xml = fill_sdt(xml)
    print("Filling legacy fields:")
    xml = fill_formtext(xml)

    for s in MUST_SURVIVE:
        if s not in xml:
            raise SystemExit(f"prefilled content lost: {s}")
    if PLACEHOLDER in xml:
        print(f"  warning: {xml.count(PLACEHOLDER)} placeholder(s) still present")

    parts["word/document.xml"] = xml.encode("utf-8")
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.writestr(n, parts[n])

    print(f"\nwrote {dst}  ({len(xml) - len(original):+d} bytes of text)")


if __name__ == "__main__":
    main()
