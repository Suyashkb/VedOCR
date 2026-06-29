"""
step4_vocab_and_stats.py — Build character vocabulary and print dataset statistics.

Outputs
───────
  output/vocab/
    char_vocab.txt       one Unicode codepoint per line (sorted by frequency desc)
    char_freq.tsv        codepoint \\t unicode_name \\t count
    unicode_blocks.tsv   block_name \\t char_count  (useful for Devanagari audit)
  Prints a rich summary to stdout.

Usage:
  python step4_vocab_and_stats.py
"""

import csv
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from config import OUTPUT_DIR, SPLITS_DIR


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def read_all_splits() -> list[dict]:
    records = []
    for split in ["train", "val", "test"]:
        path = SPLITS_DIR / f"{split}.tsv"
        if path.is_file():
            with open(path, encoding="utf-8") as f:
                records.extend(csv.DictReader(f, delimiter="\t"))
    return records


def unicode_block(ch: str) -> str:
    """Rough Unicode block name for a character."""
    cp = ord(ch)
    # common blocks relevant to Sanskrit / Devanagari
    blocks = [
        (0x0900, 0x097F, "Devanagari"),
        (0x1CD0, 0x1CFF, "Vedic Extensions"),
        (0x0000, 0x007F, "ASCII"),
        (0x0080, 0x00FF, "Latin-1 Supplement"),
        (0x2000, 0x206F, "General Punctuation"),
        (0x0020, 0x0020, "Space"),
    ]
    for lo, hi, name in blocks:
        if lo <= cp <= hi:
            return name
    return f"U+{cp:04X}–other"


# ────────────────────────────────────────────────────────────────────────────
# Vocabulary
# ────────────────────────────────────────────────────────────────────────────

def build_vocab(records: list[dict]) -> Counter:
    char_freq: Counter = Counter()
    for r in records:
        char_freq.update(r["transcription"])
    return char_freq


def write_vocab(char_freq: Counter, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # char_vocab.txt — ordered by frequency (most common first)
    vocab_path = out_dir / "char_vocab.txt"
    with open(vocab_path, "w", encoding="utf-8") as f:
        for ch, _ in char_freq.most_common():
            f.write(ch + "\n")

    # char_freq.tsv — full detail
    freq_path = out_dir / "char_freq.tsv"
    with open(freq_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["char", "codepoint", "unicode_name", "block", "count"])
        for ch, cnt in char_freq.most_common():
            try:
                name = unicodedata.name(ch)
            except ValueError:
                name = "<no name>"
            writer.writerow([ch, f"U+{ord(ch):04X}", name, unicode_block(ch), cnt])

    # unicode_blocks.tsv — aggregate by block
    block_counts: Counter = Counter()
    for ch, cnt in char_freq.items():
        block_counts[unicode_block(ch)] += cnt

    block_path = out_dir / "unicode_blocks.tsv"
    with open(block_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["block", "char_count"])
        for block, cnt in block_counts.most_common():
            writer.writerow([block, cnt])

    print(f"  ✓  Vocabulary ({len(char_freq)} unique chars) → {out_dir}/")


# ────────────────────────────────────────────────────────────────────────────
# Statistics
# ────────────────────────────────────────────────────────────────────────────

def print_stats(records: list[dict], char_freq: Counter):
    transcriptions = [r["transcription"] for r in records]
    line_lengths   = [len(t) for t in transcriptions]
    pages          = sorted({int(r["page_num"]) for r in records})

    print("\n" + "─" * 56)
    print("  DATASET STATISTICS")
    print("─" * 56)
    print(f"  Pages annotated      : {len(pages)}")
    print(f"  Total lines          : {len(records)}")
    print(f"  Total characters     : {sum(line_lengths)}")
    print(f"  Unique characters    : {len(char_freq)}")
    print()
    print(f"  Chars per line")
    print(f"    min   : {min(line_lengths)}")
    print(f"    max   : {max(line_lengths)}")
    print(f"    mean  : {sum(line_lengths)/len(line_lengths):.1f}")
    median_len = sorted(line_lengths)[len(line_lengths)//2]
    print(f"    median: {median_len}")
    print()

    # per-split stats
    for split in ["train", "val", "test"]:
        path = SPLITS_DIR / f"{split}.tsv"
        if not path.is_file():
            continue
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        chars = sum(len(r["transcription"]) for r in rows)
        pages_in_split = len({r["page_num"] for r in rows})
        print(f"  {split:6s}  {pages_in_split:3d} pages  {len(rows):4d} lines  {chars:6d} chars")

    # Unicode block breakdown
    print()
    print("  Unicode block breakdown:")
    block_counts: Counter = Counter()
    for ch, cnt in char_freq.items():
        block_counts[unicode_block(ch)] += cnt
    total_chars = sum(block_counts.values())
    for block, cnt in block_counts.most_common():
        pct = 100 * cnt / total_chars
        print(f"    {block:<30s} {cnt:6d}  ({pct:.1f}%)")

    # Top-20 most frequent characters
    print()
    print("  Top 20 most frequent characters:")
    print("    Rank  Char  Codepoint   Name                          Count")
    for i, (ch, cnt) in enumerate(char_freq.most_common(20), 1):
        try:
            name = unicodedata.name(ch)[:28]
        except ValueError:
            name = "<no name>"
        print(f"    {i:3d}   {ch!r:<6s} U+{ord(ch):04X}   {name:<30s} {cnt}")

    # Rare characters (singletons) — common source of labelling errors
    singletons = [(ch, cnt) for ch, cnt in char_freq.items() if cnt == 1]
    if singletons:
        print(f"\n  ⚠  {len(singletons)} character(s) appear only once — review for typos:")
        for ch, cnt in singletons[:20]:
            try:
                name = unicodedata.name(ch)
            except ValueError:
                name = "<no name>"
            print(f"       {ch!r}  U+{ord(ch):04X}  {name}")

    # Crop image size distribution
    print()
    print("  Crop image dimensions (first 200 checked):")
    widths, heights = [], []
    for r in records[:200]:
        p = Path(r["crop_path"])
        if p.is_file():
            with Image.open(p) as im:
                widths.append(im.width)
                heights.append(im.height)
    if widths:
        print(f"    Width  — min {min(widths)}  max {max(widths)}  mean {sum(widths)//len(widths)}")
        print(f"    Height — min {min(heights)}  max {max(heights)}  mean {sum(heights)//len(heights)}")

    print("─" * 56)


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    print("\n── Step 4: Vocabulary & Statistics ─────────────────────────────")
    records = read_all_splits()
    if not records:
        print("  No split files found.  Run steps 1 and 2 first.")
        return

    char_freq = build_vocab(records)
    write_vocab(char_freq, OUTPUT_DIR / "vocab")
    print_stats(records, char_freq)


if __name__ == "__main__":
    main()
