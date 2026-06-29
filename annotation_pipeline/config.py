"""
config.py — Central configuration for the Sanskrit OCR pipeline.
Edit the paths here; all other scripts import from this file.
"""

from pathlib import Path

# ── Input ──────────────────────────────────────────────────────────────────
# Path to your Label Studio JSON export file
LS_JSON_PATH = Path("/Users/suyash/Desktop/project/export_252119_project-252119-at-2026-04-26-21-10-0295a732.json")

# Root folder that contains the original manuscript images.
# The 'data.ocr' field in the JSON will be resolved relative to this root.
# If your JSON already has absolute paths, set this to Path("/")
IMAGE_ROOT = Path("/Users/suyash/Desktop/project/experimental_results/binarized/MORPHBG_default")

# ── Output ─────────────────────────────────────────────────────────────────
OUTPUT_DIR       = Path("output")
CROPS_DIR        = OUTPUT_DIR / "crops"          # cropped line images
SPLITS_DIR       = OUTPUT_DIR / "splits"         # train / val / test TSVs
PADDLE_DIR       = OUTPUT_DIR / "paddle_format"  # PaddleOCR-ready
HF_DIR           = OUTPUT_DIR / "hf_dataset"     # HuggingFace datasets-ready
VISUALIZE_DIR    = OUTPUT_DIR / "visualizations" # bbox-overlay debug images

# ── Dataset split ratios ────────────────────────────────────────────────────
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10   # must sum to 1.0
RANDOM_SEED = 42

# ── Image cropping ──────────────────────────────────────────────────────────
# Padding (in pixels) added around each bounding box crop
CROP_PADDING = 4

# If True, skip regions whose transcription text is empty or whitespace-only
SKIP_EMPTY_TRANSCRIPTIONS = True

# Image format for saved crops
CROP_FORMAT = "PNG"   # PNG preserves quality; use JPEG to save disk space

# ── Label Studio specifics ──────────────────────────────────────────────────
# Only keep annotations with this review state.  Set to None to keep all.
ACCEPTED_STATES = {"ACCEPTED"}   # e.g. {"ACCEPTED", "SUBMITTED"}

# Which label type to keep (from the "labels" result items).
# Set to None to keep all label types.
KEEP_LABELS = None   # e.g. {"Sanskrit Text"}