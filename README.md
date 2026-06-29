# VedOCR

**An Expert-Annotated Dataset and Transformer OCR Baseline for Vedic Sanskrit Manuscripts**

Published at DALL 2026 @ ICDAR (Workshop on Documents Analysis of Low-Resource Languages), Vienna, September 2026.

---

## Overview

VedOCR provides:
- **57 expert-annotated manuscript pages** — 681 line-level transcriptions, 41,286 character instances, 87 unique grapheme classes (including archaic conjunct consonants and Vedic tone marks)
- **MorphBG** — a colour-aware binarisation pipeline for ochre-parchment manuscripts (DRD 2.38, F-measure 0.981)
- **OCR baselines** — zero-shot Tesseract/EasyOCR evaluations and a fine-tuned TrOCR model (CER reduced from 75.74% → 48.89%, −26.85 pp)
- A modular **annotation pipeline** from Label Studio JSON to model-ready tensors

---

## Repository Structure

```
vedocr_release/
├── pipeline/                   # Core binarisation pipeline (MorphBG)
│   ├── 01_morphbg.py           # MorphBG algorithm
│   ├── 02_batch_process.py     # Batch processing runner
│   ├── 03_benchmark.py         # Evaluation against ground truth
│   ├── 04_parse_annotations.py # Label Studio annotation parser
│   └── run_pipeline.sh         # End-to-end runner
│
├── annotation_pipeline/        # Label Studio JSON → model-ready tensors
│   ├── step1_parse.py          # Resolve UUIDs, extract bounding boxes
│   ├── step2_split.py          # Page-level train/val/test split
│   ├── step3_export_formats.py # Export TSV, .box, HuggingFace CSV
│   ├── step4_vocab_and_stats.py# Vocabulary analysis, singleton detection
│   ├── step5_evaluate.py       # CER/WER via Levenshtein distance
│   ├── step6_visualize.py      # Visual QA overlay
│   ├── run_pipeline.py         # Pipeline orchestrator
│   └── config.py               # Configuration
│
├── ocr/                        # OCR experiment scripts
│   ├── 05_zero_shot_ocr.py     # Tesseract / EasyOCR / PaddleOCR evaluation
│   └── zero_shot_compare.py    # Extended comparison (adds Surya, TrOCR)
│
├── binarization_experiments/   # Extended binarisation method comparison
│   ├── 08_comprehensive_experiments.py  # 9-method comparison on full dataset
│   ├── 12_improved_hybrid_method.py     # Region-adaptive hybrid binarisation
│   ├── 13_final_hybrid_v2.py            # MorphBG + Sauvola hybrid v2
│   └── 14_ensemble_binarization.py      # Multi-method fusion
│
├── notebooks/
│   ├── trocr_finetuning_vedocr.ipynb    # TrOCR fine-tuning (with full outputs)
│   └── crnn_training.ipynb              # CRNN training (Colab-compatible)
│
├── dataset/
│   ├── master.tsv              # Master annotation file (all 681 lines)
│   ├── line_crops/             # 681 PNG line-level crops (the dataset)
│   ├── splits/                 # train.tsv / val.tsv / test.tsv
│   ├── jsonl/                  # train.jsonl / val.jsonl / test.jsonl
│   ├── hf_format/              # HuggingFace-compatible metadata.csv + images
│   ├── vocab/                  # char_vocab.txt, char_freq.tsv
│   └── sample_pages/           # Sample original manuscript scans
│
├── results/
│   ├── binarization/
│   │   ├── summary_statistics.csv       # Per-method aggregate metrics
│   │   ├── experimental_results.json    # Full per-image results (all 9 methods)
│   │   └── visualizations/             # Method comparison plots
│   └── trocr/
│       ├── training_log.csv            # Loss/WER per epoch
│       ├── test_results.csv            # Per-line CER/WER on test set
│       ├── training_curves.png
│       └── error_analysis.png
│
├── figures/                    # Publication figures (PDF + PNG)
└── requirements.txt
```

---

## Dataset

The VedOCR dataset covers a *Pada-pāṭha* Vedic Sanskrit manuscript — an analytical form that decomposes compound words into morphemes separated by spaces. This tradition creates segmentation challenges absent from modern Devanagari OCR benchmarks.

| Property | Value |
|---|---|
| Annotated pages | 57 |
| Line-level segments | 681 |
| Character instances | 41,286 |
| Unique grapheme classes | 87 |
| Singleton grapheme classes (≤1 example) | 9 |
| Train / Val / Test split (pages) | 45 / 5 / 7 |

Splits are enforced at **page level** — all lines from a page appear in exactly one split.

The dataset will be released under **CC BY-NC-SA 4.0** upon paper acceptance.

---

## MorphBG: Binarisation for Ochre Parchment

Standard binarisation fails on ochre-treated parchment because iron-gall ink and substrate differ in hue, not just brightness. MorphBG uses:

1. CLAHE-V normalisation (HSV space)
2. Per-channel morphological background estimation
3. Weber-contrast ink signal per channel
4. Scalar confidence map → Otsu threshold
5. Connected-component cleanup

| Method | IoU | F-measure | DRD ↓ |
|---|---|---|---|
| Otsu | 0.9622 | 0.9807 | 3.290 |
| Sauvola | 0.9382 | 0.9681 | 8.551 |
| Niblack | 0.6143 | 0.7583 | 75.68 |
| **MorphBG (ours)** | **0.9636** | **0.9815** | **2.379** |

---

## OCR Results

| Model | CER (%) ↓ | WER (%) ↓ |
|---|---|---|
| Tesseract-san (zero-shot) | 77.71 | 94.20 |
| Tesseract-hin (zero-shot) | 75.74 | 92.15 |
| EasyOCR-hi (zero-shot) | 71.41 | 99.93 |
| **TrOCR fine-tuned (ours)** | **48.89** | **68.42** |

Fine-tuning `microsoft/trocr-base-handwritten` on 45 training pages achieves a **26.85 pp CER reduction**. The fine-tuned checkpoint will be released on HuggingFace upon acceptance.

---

## Setup

```bash
pip install -r requirements.txt
```

For Tesseract, also install the `san` and `hin` language packs:
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-san tesseract-ocr-hin
```

---

## Citation

```bibtex
@inproceedings{vedocr2026,
  title     = {VedOCR: An Expert-Annotated Dataset and Transformer OCR Baseline
               for Vedic Sanskrit Manuscripts},
  booktitle = {Proceedings of DALL 2026 @ ICDAR},
  year      = {2026}
}
```

*(Full citation with authors and DOI added after camera-ready.)*
