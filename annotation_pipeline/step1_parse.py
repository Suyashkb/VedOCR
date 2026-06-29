"""
step1_parse.py — Parse Label Studio JSON into clean Python objects.

Outputs:
  output/master.tsv          — one row per crop: path TAB ground_truth
  output/skipped.tsv         — rows that were dropped (with reason)
  output/crops/              — cropped line images named page{N:03d}_line{M:03d}.png

Usage:
  python step1_parse.py
"""

import json
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
from typing import Optional

from PIL import Image

# ── import config ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    LS_JSON_PATH, IMAGE_ROOT, OUTPUT_DIR, CROPS_DIR,
    CROP_PADDING, CROP_FORMAT, SKIP_EMPTY_TRANSCRIPTIONS,
    ACCEPTED_STATES, KEEP_LABELS,
)


# ────────────────────────────────────────────────────────────────────────────
# Data model
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class TextRegion:
    """One annotated text line / bounding box."""
    region_id:      str
    x_pct:         float   # left edge, % of image width
    y_pct:         float   # top edge,  % of image height
    w_pct:         float   # width,     % of image width
    h_pct:         float   # height,    % of image height
    rotation:      float
    original_w:    int
    original_h:    int
    label:         str     # e.g. "Sanskrit Text"
    transcription: str     # ground truth unicode string

    def pixel_box(self, img_w: int, img_h: int, padding: int = 0):
        """Return (x0, y0, x1, y1) in absolute pixels, with optional padding."""
        x0 = max(0, int(self.x_pct / 100 * img_w) - padding)
        y0 = max(0, int(self.y_pct / 100 * img_h) - padding)
        x1 = min(img_w, int((self.x_pct + self.w_pct) / 100 * img_w) + padding)
        y1 = min(img_h, int((self.y_pct + self.h_pct) / 100 * img_h) + padding)
        return x0, y0, x1, y1


@dataclass
class PageRecord:
    """One manuscript page / image task."""
    task_id:    int
    image_path: Path
    regions:    list[TextRegion] = field(default_factory=list)


# ────────────────────────────────────────────────────────────────────────────
# Parser
# ────────────────────────────────────────────────────────────────────────────

def _strip_hash_prefix(fname: str) -> str:
    """
    Label Studio sometimes prepends a random hex string to uploaded filenames:
      'a1b2c3d4-00000004.png'  →  '00000004.png'
    Detect the pattern (8 hex chars + dash) and strip it.
    """
    import re
    return re.sub(r'^[0-9a-f]{8}-', '', fname)


def _resolve_image_path(ocr_field: str) -> Path:
    """
    Label Studio stores paths like '/data/upload/1/image.jpg' or just
    a filename.  Try several strategies to find the actual file.
    """
    # Strategy 1: absolute path as-is
    p = Path(ocr_field)
    if p.is_file():
        return p

    # Strategy 2: just the filename under IMAGE_ROOT
    fname = Path(ocr_field).name
    candidate = IMAGE_ROOT / fname
    if candidate.is_file():
        return candidate

    # Strategy 3: strip Label Studio hash prefix  (e.g. 'a1b2c3d4-00000004.png' → '00000004.png')
    clean_fname = _strip_hash_prefix(fname)
    candidate = IMAGE_ROOT / clean_fname
    if candidate.is_file():
        return candidate

    # Strategy 4: strip leading /data/upload/<n>/ prefix then try with/without hash
    parts = p.parts
    for i, part in enumerate(parts):
        if part not in ("data", "upload") and i > 0:
            candidate = IMAGE_ROOT / Path(*parts[i:])
            if candidate.is_file():
                return candidate
            # also try hash-stripped version
            candidate = IMAGE_ROOT / _strip_hash_prefix(str(Path(*parts[i:])))
            if candidate.is_file():
                return candidate

    # Strategy 5: return best-guess with hash stripped so the error message is useful
    return IMAGE_ROOT / clean_fname


def _group_results_by_id(results: list[dict]) -> dict[str, dict]:
    """
    Each region in LS has three result items sharing the same 'id':
      rectangle  → bounding box geometry
      labels     → category label
      textarea   → transcription text
    Group them into {region_id: {type: item}}.
    """
    groups: dict[str, dict] = defaultdict(dict)
    for item in results:
        groups[item["id"]][item["type"]] = item
    return groups


def parse_label_studio_json(json_path: Path) -> list[PageRecord]:
    """
    Read the LS JSON export and return a list of PageRecord objects.
    Applies filters from config (ACCEPTED_STATES, KEEP_LABELS).
    """
    with open(json_path, encoding="utf-8") as f:
        tasks = json.load(f)

    pages: list[PageRecord] = []
    skipped_tasks = 0

    for task in tasks:
        # ── find the best annotation ────────────────────────────────────────
        annotations = task.get("annotations", [])
        if not annotations:
            skipped_tasks += 1
            continue

        # prefer ACCEPTED, fall back to first annotation
        annotation = None
        for ann in annotations:
            state = ann.get("state", "")
            if ACCEPTED_STATES is None or state in ACCEPTED_STATES:
                annotation = ann
                break
        if annotation is None:
            annotation = annotations[0]   # fallback

        # ── resolve image path ───────────────────────────────────────────────
        ocr_field  = task["data"]["ocr"]
        image_path = _resolve_image_path(ocr_field)

        page = PageRecord(task_id=task["id"], image_path=image_path)

        # ── group result items by region id ─────────────────────────────────
        groups = _group_results_by_id(annotation.get("result", []))

        for region_id, items in groups.items():
            rect  = items.get("rectangle")
            label_item = items.get("labels")
            text_item  = items.get("textarea")

            # need all three to form a valid region
            if not (rect and label_item and text_item):
                continue

            label = (label_item["value"].get("labels") or [""])[0]
            if KEEP_LABELS and label not in KEEP_LABELS:
                continue

            transcription = (text_item["value"].get("text") or [""])[0].strip()

            region = TextRegion(
                region_id      = region_id,
                x_pct          = rect["value"]["x"],
                y_pct          = rect["value"]["y"],
                w_pct          = rect["value"]["width"],
                h_pct          = rect["value"]["height"],
                rotation       = rect["value"].get("rotation", 0.0),
                original_w     = rect.get("original_width",  rect["value"].get("original_width",  0)),
                original_h     = rect.get("original_height", rect["value"].get("original_height", 0)),
                label          = label,
                transcription  = transcription,
            )
            page.regions.append(region)

        # sort regions top-to-bottom, then left-to-right
        page.regions.sort(key=lambda r: (r.y_pct, r.x_pct))
        pages.append(page)

    print(f"  Parsed {len(pages)} page tasks  ({skipped_tasks} tasks had no annotations)")
    return pages


# ────────────────────────────────────────────────────────────────────────────
# Cropper
# ────────────────────────────────────────────────────────────────────────────

def crop_and_export(pages: list[PageRecord]) -> tuple[list[dict], list[dict]]:
    """
    For each region, crop the source image and save to CROPS_DIR.
    Returns (records, skipped_records) where each record is a dict:
      {crop_path, page_id, line_idx, label, transcription, image_path, box_pixels}
    """
    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    records  = []
    skipped  = []
    missing_images = set()

    for page_idx, page in enumerate(pages):
        page_num = page_idx + 1   # 1-based for human-readable names

        if not page.image_path.is_file():
            if page.image_path not in missing_images:
                print(f"  ⚠  Image not found: {page.image_path}")
                missing_images.add(page.image_path)
            for region in page.regions:
                skipped.append({
                    "crop_path":     "",
                    "transcription": region.transcription,
                    "reason":        "image_not_found",
                    "image_path":    str(page.image_path),
                })
            continue

        img = Image.open(page.image_path).convert("RGB")
        img_w, img_h = img.size

        for line_idx, region in enumerate(page.regions):
            line_num = line_idx + 1   # 1-based

            # ── skip empty transcriptions ────────────────────────────────────
            if SKIP_EMPTY_TRANSCRIPTIONS and not region.transcription:
                skipped.append({
                    "crop_path":     "",
                    "transcription": "",
                    "reason":        "empty_transcription",
                    "image_path":    str(page.image_path),
                })
                continue

            # ── compute crop box ─────────────────────────────────────────────
            x0, y0, x1, y1 = region.pixel_box(img_w, img_h, padding=CROP_PADDING)

            if x1 <= x0 or y1 <= y0:
                skipped.append({
                    "crop_path":     "",
                    "transcription": region.transcription,
                    "reason":        "degenerate_bbox",
                    "image_path":    str(page.image_path),
                })
                continue

            crop = img.crop((x0, y0, x1, y1))

            # ── handle rotation (most vedic manuscripts have 0°) ─────────────
            if region.rotation:
                crop = crop.rotate(-region.rotation, expand=True)

            # ── save crop ────────────────────────────────────────────────────
            crop_name = f"page{page_num:03d}_line{line_num:03d}.png"
            crop_path = CROPS_DIR / crop_name
            crop.save(crop_path, format=CROP_FORMAT)

            records.append({
                "crop_path":     str(crop_path),
                "page_id":       page.task_id,
                "page_num":      page_num,
                "line_num":      line_num,
                "label":         region.label,
                "transcription": region.transcription,
                "image_path":    str(page.image_path),
                "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            })

    return records, skipped


# ────────────────────────────────────────────────────────────────────────────
# Writers
# ────────────────────────────────────────────────────────────────────────────

def write_master_tsv(records: list[dict], path: Path):
    """Write the canonical tab-separated ground truth file."""
    if not records:
        print("  ✗  master.tsv  → 0 records written (all images were missing)")
        print("     Check that IMAGE_ROOT in config.py points to your binarized image folder.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(records)
    print(f"  ✓  master.tsv  → {len(records)} lines")


def write_skipped_tsv(skipped: list[dict], path: Path):
    if not skipped:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["crop_path","transcription","reason","image_path"], delimiter="\t")
        writer.writeheader()
        writer.writerows(skipped)
    print(f"  ⚠  skipped.tsv → {len(skipped)} dropped regions")


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    print("\n── Step 1: Parse & Crop ────────────────────────────────────────")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  Loading: {LS_JSON_PATH}")
    pages = parse_label_studio_json(LS_JSON_PATH)

    total_regions = sum(len(p.regions) for p in pages)
    print(f"  Total annotated regions: {total_regions}")

    print("  Cropping images …")
    records, skipped = crop_and_export(pages)

    write_master_tsv(records,  OUTPUT_DIR / "master.tsv")
    write_skipped_tsv(skipped, OUTPUT_DIR / "skipped.tsv")

    print(f"\n  Done.  {len(records)} crops saved to {CROPS_DIR}/")


if __name__ == "__main__":
    main()