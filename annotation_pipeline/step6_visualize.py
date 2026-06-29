"""
step6_visualize.py — Draw bounding boxes on manuscript pages for annotation QA.

Opens each page, overlays the annotated bounding boxes in colour, and saves
annotated images to output/visualizations/.  Useful for catching mis-aligned
boxes, wrong transcription order, or missed lines before training.

Usage:
  python step6_visualize.py              # visualize all pages
  python step6_visualize.py --pages 1 3  # only pages 1 and 3
  python step6_visualize.py --max 10     # first 10 pages only
"""

import argparse
import csv
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from config import OUTPUT_DIR, VISUALIZE_DIR
from step1_parse import parse_label_studio_json, LS_JSON_PATH


# colour palette — cycles if more than 8 lines per page
COLOURS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
]


def visualize_page(page_idx: int, page, out_dir: Path):
    if not page.image_path.is_file():
        print(f"  ⚠  Missing: {page.image_path}")
        return

    img = Image.open(page.image_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size

    try:
        # use a default PIL font; won't render Devanagari but shows line numbers
        font = ImageFont.load_default()
    except Exception:
        font = None

    for line_idx, region in enumerate(page.regions):
        colour = COLOURS[line_idx % len(COLOURS)]
        x0, y0, x1, y1 = region.pixel_box(w, h, padding=0)

        # semi-transparent fill
        draw.rectangle([x0, y0, x1, y1], fill=colour + "33", outline=colour, width=2)

        # line number label
        label = f"L{line_idx+1}"
        draw.text((x0 + 4, y0 + 2), label, fill=colour, font=font)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"page{page_idx+1:03d}_annotated.jpg"
    img.save(out_path, format="JPEG", quality=85)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", nargs="+", type=int, default=None,
                        help="1-based page numbers to visualize")
    parser.add_argument("--max",   type=int, default=None,
                        help="Maximum number of pages to process")
    args = parser.parse_args()

    print("\n── Step 6: Visualize Annotations ───────────────────────────────")
    pages = parse_label_studio_json(LS_JSON_PATH)

    if args.pages:
        indices = [p - 1 for p in args.pages]
    else:
        indices = list(range(len(pages)))

    if args.max:
        indices = indices[:args.max]

    print(f"  Rendering {len(indices)} pages …")
    for idx in indices:
        if idx >= len(pages):
            continue
        out = visualize_page(idx, pages[idx], VISUALIZE_DIR)
        if out:
            n_regions = len(pages[idx].regions)
            print(f"    page {idx+1:3d}  {n_regions:2d} regions → {out.name}")

    print(f"\n  Visualizations → {VISUALIZE_DIR}/")


if __name__ == "__main__":
    main()
