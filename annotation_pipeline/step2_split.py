"""
step2_split.py — Create reproducible train / val / test splits.

Splits at the PAGE level (not line level) so that no page leaks
across splits, which would inflate evaluation scores.

Reads:   output/master.tsv
Writes:  output/splits/train.tsv
         output/splits/val.tsv
         output/splits/test.tsv

Usage:
  python step2_split.py
"""

import csv
import sys
import random
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from config import OUTPUT_DIR, SPLITS_DIR, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED


def read_master(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_split(records: list[dict], path: Path, name: str):
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(records)
    chars = sum(len(r["transcription"]) for r in records)
    print(f"  {name:6s}  {len(records):4d} lines   {chars:6d} chars   → {path}")


def main():
    print("\n── Step 2: Train / Val / Test Split ────────────────────────────")

    assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    master_path = OUTPUT_DIR / "master.tsv"
    records = read_master(master_path)
    print(f"  Total lines: {len(records)}")

    # ── group by page_num so splits are page-level ──────────────────────────
    pages: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        pages[r["page_num"]].append(r)

    page_ids = sorted(pages.keys(), key=lambda x: int(x))
    print(f"  Total pages: {len(page_ids)}")

    random.seed(RANDOM_SEED)
    random.shuffle(page_ids)

    n = len(page_ids)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    # test gets the remainder to avoid floating-point gaps
    n_test  = n - n_train - n_val

    train_pages = page_ids[:n_train]
    val_pages   = page_ids[n_train : n_train + n_val]
    test_pages  = page_ids[n_train + n_val :]

    def flatten(page_list):
        out = []
        for pid in sorted(page_list, key=lambda x: int(x)):
            out.extend(pages[pid])
        return out

    train = flatten(train_pages)
    val   = flatten(val_pages)
    test  = flatten(test_pages)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    write_split(train, SPLITS_DIR / "train.tsv", "train")
    write_split(val,   SPLITS_DIR / "val.tsv",   "val")
    write_split(test,  SPLITS_DIR / "test.tsv",  "test")

    # ── summary ─────────────────────────────────────────────────────────────
    print(f"\n  Page split:  {len(train_pages)} train  |  {len(val_pages)} val  |  {len(test_pages)} test")
    print(f"  Line split:  {len(train)} train  |  {len(val)} val  |  {len(test)} test")


if __name__ == "__main__":
    main()
