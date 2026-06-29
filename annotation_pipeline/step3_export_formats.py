"""
step3_export_formats.py — Export the splits to every framework format you'll need.

Formats produced
────────────────
1. Plain TSV (path \\t text)           → for any custom evaluation script
2. PaddleOCR recognition format        → train_list.txt / val_list.txt / test_list.txt
3. HuggingFace datasets (metadata.csv) → load with datasets.load_dataset("imagefolder")
4. Tesseract LSTM (.box files)         → for fine-tuning Tesseract
5. JSON lines (.jsonl)                 → generic; readable by most deep-learning code

Reads:   output/splits/{train,val,test}.tsv
Writes:  output/paddle_format/
         output/hf_dataset/
         output/tesseract_box/
         output/jsonl/

Usage:
  python step3_export_formats.py
"""

import csv
import json
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import OUTPUT_DIR, SPLITS_DIR, PADDLE_DIR, HF_DIR


# ── helpers ──────────────────────────────────────────────────────────────────

def read_split(name: str) -> list[dict]:
    path = SPLITS_DIR / f"{name}.tsv"
    if not path.is_file():
        print(f"  ⚠  {path} not found — run step2_split.py first")
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


SPLITS = ["train", "val", "test"]


# ────────────────────────────────────────────────────────────────────────────
# 1.  Plain TSV  (path TAB text) — simplest possible format
# ────────────────────────────────────────────────────────────────────────────

def export_plain_tsv(data: dict[str, list[dict]]):
    """
    Minimal two-column TSV: image_path \\t ground_truth
    Compatible with: jiwer, editdistance, any custom eval script.
    """
    out_dir = OUTPUT_DIR / "plain_tsv"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, records in data.items():
        with open(out_dir / f"{split}.tsv", "w", encoding="utf-8") as f:
            for r in records:
                f.write(f"{r['crop_path']}\t{r['transcription']}\n")
    print(f"  ✓  Plain TSV          → {out_dir}/")


# ────────────────────────────────────────────────────────────────────────────
# 2.  PaddleOCR recognition format
# ────────────────────────────────────────────────────────────────────────────

def export_paddle(data: dict[str, list[dict]]):
    """
    PaddleOCR expects:
      image/relative/path.png\\tground truth text\\n
    and a separate dict.txt listing every unique character.
    See: https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_en/recognition_en.md
    """
    PADDLE_DIR.mkdir(parents=True, exist_ok=True)

    all_chars: set[str] = set()

    for split, records in data.items():
        list_file = PADDLE_DIR / f"{split}_list.txt"
        with open(list_file, "w", encoding="utf-8") as f:
            for r in records:
                text = r["transcription"]
                all_chars.update(text)
                # PaddleOCR wants relative paths from its data root
                f.write(f"{r['crop_path']}\t{text}\n")

    # character dictionary — PaddleOCR requires one char per line
    dict_path = PADDLE_DIR / "sanskrit_dict.txt"
    with open(dict_path, "w", encoding="utf-8") as f:
        for ch in sorted(all_chars):
            f.write(ch + "\n")

    print(f"  ✓  PaddleOCR format   → {PADDLE_DIR}/  ({len(all_chars)} unique chars in dict)")


# ────────────────────────────────────────────────────────────────────────────
# 3.  HuggingFace datasets  (ImageFolder with metadata.csv)
# ────────────────────────────────────────────────────────────────────────────

def export_huggingface(data: dict[str, list[dict]]):
    """
    HuggingFace `datasets` ImageFolder format:
      dataset/
        train/
          metadata.csv      ← file_name, text columns
          page001_line001.png
          …
        val/
          metadata.csv
          …

    Load with:
      from datasets import load_dataset
      ds = load_dataset("imagefolder", data_dir="output/hf_dataset", drop_labels=True)
      # ds["train"][0] → {"image": <PIL>, "text": "…"}
    """
    for split, records in data.items():
        split_dir = HF_DIR / split
        split_dir.mkdir(parents=True, exist_ok=True)

        meta_path = split_dir / "metadata.csv"
        with open(meta_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["file_name", "text"])
            for r in records:
                src = Path(r["crop_path"])
                if src.is_file():
                    dst = split_dir / src.name
                    if not dst.exists():
                        shutil.copy2(src, dst)
                    writer.writerow([src.name, r["transcription"]])

    print(f"  ✓  HuggingFace format  → {HF_DIR}/")
    print( "     Load: datasets.load_dataset('imagefolder', data_dir='output/hf_dataset', drop_labels=True)")


# ────────────────────────────────────────────────────────────────────────────
# 4.  Tesseract LSTM  (.box files)
# ────────────────────────────────────────────────────────────────────────────

def export_tesseract_box(data: dict[str, list[dict]]):
    """
    Tesseract fine-tuning requires a .box file alongside each image:
      <char> <x0> <y0> <x1> <y1> <page>
    For a line image, each character spans the full height of the crop.
    Since we only have line-level transcription (not character boxes),
    we generate a single 'wordstr' box per line, which Tesseract's
    lstmtraining accepts.

    Format (wordstr box):
      WordStr <x0> <y0> <x1> <y1> <page> #<transcription text>
      \\t <x1> <y0> <x1+1> <y1> <page>

    See: https://tesseract-ocr.github.io/tessapi/5.x/a02486.html
    """
    out_dir = OUTPUT_DIR / "tesseract_box"
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, records in data.items():
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        for r in records:
            src = Path(r["crop_path"])
            if not src.is_file():
                continue

            # copy the image
            shutil.copy2(src, split_dir / src.name)

            # get actual crop dimensions
            from PIL import Image
            with Image.open(src) as im:
                w, h = im.size

            text = r["transcription"]
            box_path = split_dir / (src.stem + ".box")
            with open(box_path, "w", encoding="utf-8") as f:
                # wordstr box spanning full line
                f.write(f"WordStr 0 0 {w} {h} 0 #{text}\n")
                f.write(f"\t {w} 0 {w+1} {h} 0\n")

    print(f"  ✓  Tesseract .box     → {out_dir}/")


# ────────────────────────────────────────────────────────────────────────────
# 5.  JSON lines  (.jsonl)
# ────────────────────────────────────────────────────────────────────────────

def export_jsonl(data: dict[str, list[dict]]):
    """
    One JSON object per line.  Compatible with most deep-learning frameworks
    (TrOCR custom training loops, LLM fine-tuning, etc.).

    Each line:
      {"image_path": "…", "text": "…", "page": N, "line": N}
    """
    out_dir = OUTPUT_DIR / "jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, records in data.items():
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in records:
                obj = {
                    "image_path": r["crop_path"],
                    "text":       r["transcription"],
                    "page":       int(r["page_num"]),
                    "line":       int(r["line_num"]),
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"  ✓  JSONL              → {out_dir}/")


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    print("\n── Step 3: Export to Framework Formats ─────────────────────────")

    data = {split: read_split(split) for split in SPLITS}
    total = sum(len(v) for v in data.values())
    if total == 0:
        print("  No records found. Run step1_parse.py and step2_split.py first.")
        return

    export_plain_tsv(data)
    export_paddle(data)
    export_huggingface(data)
    export_tesseract_box(data)
    export_jsonl(data)

    print("\n  All formats exported.")


if __name__ == "__main__":
    main()
