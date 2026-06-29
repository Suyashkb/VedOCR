"""
zero_shot_compare.py
────────────────────────────────────────────────────────────────────
Zero-shot OCR comparison for the Vedic Sanskrit manuscript dataset.

Engines tested (all CPU-runnable, all free / open-source):
  1. EasyOCR          — hi (Devanagari)
  2. PaddleOCR        — hi (Devanagari)
  3. Surya            — multilingual transformer OCR
  4. TrOCR base       — microsoft/trocr-base-handwritten (zero-shot)
  5. TrOCR large      — microsoft/trocr-large-handwritten (zero-shot)
  [6. Tesseract       — hin / san reproduced for reference]

Usage:
  pip install easyocr paddlepaddle paddleocr surya-ocr transformers \
              torch torchvision pillow editdistance jiwer tqdm rich

  python zero_shot_compare.py \
      --test_tsv  output/splits/test.tsv \
      --crops_dir output/crops \
      --out_dir   output/zero_shot_results \
      [--engines  easyocr paddle surya trocr_base trocr_large tesseract] \
      [--limit    20]          # optional: only run N lines (debug)

Outputs:
  output/zero_shot_results/
    ├── predictions/
    │   ├── easyocr.tsv          # crop_path | reference | prediction | cer | wer
    │   ├── paddle.tsv
    │   └── ...
    ├── summary.tsv              # engine | CER | WER | lines_empty | runtime_s
    └── summary.txt              # human-readable table

Notes:
  • Each engine is imported lazily — a missing install skips that engine
    gracefully rather than crashing the whole script.
  • CER = character error rate (Levenshtein / ref_length).
  • WER = word error rate (token-level Levenshtein / ref_word_count).
  • Empty predictions count toward CER (distance = len(reference)).
"""

import argparse
import csv
import importlib
import os
import sys
import time
import traceback
from pathlib import Path

# ── Third-party (always required) ─────────────────────────────────
try:
    import editdistance
except ImportError:
    sys.exit("Install editdistance:  pip install editdistance")

try:
    from PIL import Image
except ImportError:
    sys.exit("Install Pillow:  pip install pillow")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):          # minimal fallback
        return x

# ══════════════════════════════════════════════════════════════════
#  Metrics
# ══════════════════════════════════════════════════════════════════

def cer(ref: str, hyp: str) -> float:
    """Character Error Rate (0–∞, lower is better)."""
    ref = ref.strip()
    hyp = hyp.strip()
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return editdistance.eval(ref, hyp) / len(ref)


def wer(ref: str, hyp: str) -> float:
    """Word Error Rate — space-tokenised."""
    ref_tokens = ref.strip().split()
    hyp_tokens = hyp.strip().split()
    if len(ref_tokens) == 0:
        return 0.0 if len(hyp_tokens) == 0 else 1.0
    return editdistance.eval(ref_tokens, hyp_tokens) / len(ref_tokens)


# ══════════════════════════════════════════════════════════════════
#  Engine wrappers
#  Each returns (predicted_text: str) for a given PIL image.
#  Return "" on failure.
# ══════════════════════════════════════════════════════════════════

# ── 1. EasyOCR ────────────────────────────────────────────────────

def build_easyocr(gpu: bool = False):
    try:
        import easyocr
    except ImportError:
        print("[EasyOCR] not installed — skipping.  pip install easyocr")
        return None
    print("[EasyOCR] loading model …")
    reader = easyocr.Reader(["hi"], gpu=gpu, verbose=False)

    def predict(img: Image.Image) -> str:
        import numpy as np
        arr = np.array(img.convert("RGB"))
        results = reader.readtext(arr, detail=0, paragraph=True)
        return " ".join(results).strip()

    return predict


# ── 2. PaddleOCR ──────────────────────────────────────────────────

def build_paddle(use_gpu: bool = False):
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        print("[PaddleOCR] not installed — skipping.  pip install paddlepaddle paddleocr")
        return None
    print("[PaddleOCR] loading model …")

    # PaddleOCR v3+ dropped show_log and use_angle_cls.
    # Build kwargs defensively so the same script works on v2 and v3.
    import inspect
    sig = inspect.signature(PaddleOCR.__init__).parameters

    kwargs = {"lang": "hi"}

    # use_gpu / device
    if "use_gpu" in sig:
        kwargs["use_gpu"] = use_gpu
    elif "device" in sig:
        kwargs["device"] = "gpu" if use_gpu else "cpu"

    # orientation classifier (angle cls)
    if "use_angle_cls" in sig:
        kwargs["use_angle_cls"] = False
    elif "use_textline_orientation" in sig:
        kwargs["use_textline_orientation"] = False

    # logging suppression
    if "show_log" in sig:
        kwargs["show_log"] = False

    ocr = PaddleOCR(**kwargs)

    def predict(img: Image.Image) -> str:
        import numpy as np
        arr = np.array(img.convert("RGB"))

        # v2: ocr.ocr(arr, cls=False)  → list of list of [bbox, (text, score)]
        # v3: ocr.predict(arr)          → list of OCRResult objects
        try:
            # Try v3 predict() first
            results = ocr.predict(arr)
            if not results:
                return ""
            # v3 OCRResult has .rec_texts (list[str]) or iterate text_lines
            first = results[0]
            if hasattr(first, "rec_texts"):
                return " ".join(t for t in first.rec_texts if t).strip()
            # fallback: stringify whatever came back
            return str(first).strip()
        except AttributeError:
            pass

        # v2 fallback
        try:
            result = ocr.ocr(arr, cls=False)
            if result and result[0]:
                lines = [item[1][0] for item in result[0] if item and item[1]]
                return " ".join(lines).strip()
        except Exception as e:
            print(f"  [PaddleOCR] inference error: {e}")
        return ""

    return predict


# ── 3. Surya ──────────────────────────────────────────────────────

def build_surya():
    """
    Surya is a transformer-based multilingual OCR (supports Devanagari).
    pip install surya-ocr
    """
    try:
        from surya.recognition import batch_recognition
        from surya.model.recognition.model import load_model as load_rec_model
        from surya.model.recognition.processor import load_processor as load_rec_processor
    except ImportError:
        print("[Surya] not installed — skipping.  pip install surya-ocr")
        return None
    print("[Surya] loading model …")
    model     = load_rec_model()
    processor = load_rec_processor()

    def predict(img: Image.Image) -> str:
        try:
            # Surya expects a list of images and a list of language lists
            predictions = batch_recognition(
                [img.convert("RGB")],
                [["hi"]],          # Devanagari / Hindi
                model,
                processor,
            )
            if predictions and predictions[0].text_lines:
                return " ".join(t.text for t in predictions[0].text_lines).strip()
            return ""
        except Exception as e:
            print(f"  [Surya] inference error: {e}")
            return ""

    return predict


# ── 4 & 5. TrOCR (zero-shot, no fine-tuning) ─────────────────────

def build_trocr(model_name: str):
    """
    microsoft/trocr-base-handwritten or microsoft/trocr-large-handwritten.
    These are English-trained, so CER will be very high — useful as a lower
    bound to show how much fine-tuning on Sanskrit helps.
    """
    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        import torch
    except ImportError:
        print(f"[TrOCR {model_name}] transformers/torch not installed — skipping.")
        return None
    print(f"[TrOCR] loading {model_name} …")
    processor = TrOCRProcessor.from_pretrained(model_name)
    model     = VisionEncoderDecoderModel.from_pretrained(model_name)
    model.eval()
    device = "cpu"
    model.to(device)

    def predict(img: Image.Image) -> str:
        try:
            import torch
            pixel_values = processor(img.convert("RGB"), return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device)
            with torch.no_grad():
                ids = model.generate(pixel_values)
            return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        except Exception as e:
            print(f"  [TrOCR {model_name}] inference error: {e}")
            return ""

    return predict


# ── 6. Tesseract (reference baseline) ────────────────────────────

def build_tesseract(lang: str = "hin"):
    try:
        import pytesseract
    except ImportError:
        print(f"[Tesseract-{lang}] pytesseract not installed — skipping.")
        return None
    # Quick smoke-test
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        print(f"[Tesseract-{lang}] tesseract binary not found — skipping.")
        return None

    def predict(img: Image.Image) -> str:
        try:
            import pytesseract
            cfg = f"--oem 1 --psm 7 -l {lang}"
            return pytesseract.image_to_string(img.convert("RGB"), config=cfg).strip()
        except Exception as e:
            print(f"  [Tesseract-{lang}] error: {e}")
            return ""

    return predict


# ══════════════════════════════════════════════════════════════════
#  Engine registry
# ══════════════════════════════════════════════════════════════════

ENGINE_BUILDERS = {
    "easyocr":     lambda: build_easyocr(gpu=False),
    "paddle":      lambda: build_paddle(use_gpu=False),
    "surya":       lambda: build_surya(),
    "trocr_base":  lambda: build_trocr("microsoft/trocr-base-handwritten"),
    "trocr_large": lambda: build_trocr("microsoft/trocr-large-handwritten"),
    "tesseract_hin": lambda: build_tesseract("hin"),
    "tesseract_san": lambda: build_tesseract("san"),
}

ALL_ENGINES = list(ENGINE_BUILDERS.keys())


# ══════════════════════════════════════════════════════════════════
#  I/O helpers
# ══════════════════════════════════════════════════════════════════

def load_test_tsv(path: Path):
    """
    Returns list of dicts with keys: crop_path, transcription.
    Accepts the master.tsv / split TSV schema from your pipeline.
    """
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            crop  = row.get("crop_path", "").strip()
            text  = row.get("transcription", "").strip()
            if crop and text:
                rows.append({"crop_path": crop, "transcription": text})
    return rows


def resolve_crop(crop_path: str, crops_dir: Path) -> Path:
    """
    crop_path in the TSV may be an absolute path or just a filename.
    Try absolute first, then relative to crops_dir.
    """
    p = Path(crop_path)
    if p.exists():
        return p
    candidate = crops_dir / p.name
    if candidate.exists():
        return candidate
    # last resort: search by name anywhere under crops_dir
    matches = list(crops_dir.rglob(p.name))
    if matches:
        return matches[0]
    return p   # will raise FileNotFoundError downstream


# ══════════════════════════════════════════════════════════════════
#  Per-engine evaluation
# ══════════════════════════════════════════════════════════════════

def evaluate_engine(name: str, predict_fn, rows, crops_dir: Path,
                    out_dir: Path) -> dict:
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    out_tsv  = pred_dir / f"{name}.tsv"

    total_cer    = 0.0
    total_wer    = 0.0
    total_chars  = 0
    empty_count  = 0
    n            = 0

    fieldnames = ["crop_path", "reference", "prediction", "cer", "wer"]

    with open(out_tsv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for row in tqdm(rows, desc=f"  {name}", leave=False):
            ref       = row["transcription"]
            crop_path = row["crop_path"]

            try:
                img_path = resolve_crop(crop_path, crops_dir)
                img      = Image.open(img_path)
            except Exception as e:
                print(f"  [{name}] Cannot open {crop_path}: {e}")
                pred = ""
            else:
                try:
                    pred = predict_fn(img)
                except Exception as e:
                    print(f"  [{name}] Inference failed on {crop_path}: {e}")
                    pred = ""

            if not pred:
                empty_count += 1

            c = cer(ref, pred)
            w = wer(ref, pred)

            # Weighted accumulation (by character count like standard CER)
            ref_len      = max(len(ref), 1)
            total_cer   += editdistance.eval(ref.strip(), pred.strip())
            total_chars += len(ref.strip())
            total_wer   += w
            n            += 1

            writer.writerow({
                "crop_path":  crop_path,
                "reference":  ref,
                "prediction": pred,
                "cer":        f"{c:.4f}",
                "wer":        f"{w:.4f}",
            })

    corpus_cer = total_cer / max(total_chars, 1)
    mean_wer   = total_wer / max(n, 1)

    return {
        "engine":       name,
        "lines":        n,
        "lines_empty":  empty_count,
        "corpus_CER":   round(corpus_cer, 4),
        "mean_WER":     round(mean_wer, 4),
    }


# ══════════════════════════════════════════════════════════════════
#  Pretty summary table (no rich dependency)
# ══════════════════════════════════════════════════════════════════

def print_table(results: list[dict], runtime: dict[str, float]):
    col_w = [22, 8, 8, 12, 10]
    header = ["Engine", "CER", "WER", "Empty lines", "Time (s)"]

    sep   = "─" * (sum(col_w) + len(col_w) * 3 + 1)
    fmt   = " | ".join(f"{{:<{w}}}" for w in col_w)

    print()
    print(sep)
    print(fmt.format(*header))
    print(sep)
    for r in sorted(results, key=lambda x: x["corpus_CER"]):
        t = runtime.get(r["engine"], 0)
        print(fmt.format(
            r["engine"],
            f"{r['corpus_CER']*100:.2f}%",
            f"{r['mean_WER']*100:.2f}%",
            f"{r['lines_empty']}/{r['lines']}",
            f"{t:.1f}",
        ))
    print(sep)
    print()


def write_summary(results, runtime, out_dir: Path):
    # TSV
    tsv_path = out_dir / "summary.tsv"
    with open(tsv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["engine", "corpus_CER", "mean_WER",
                        "lines_empty", "lines", "runtime_s"],
            delimiter="\t",
        )
        writer.writeheader()
        for r in sorted(results, key=lambda x: x["corpus_CER"]):
            writer.writerow({
                **r,
                "runtime_s": round(runtime.get(r["engine"], 0), 1),
            })

    # Human-readable text
    txt_path = out_dir / "summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Zero-shot OCR comparison — Vedic Sanskrit test split\n")
        f.write("=" * 60 + "\n")
        for r in sorted(results, key=lambda x: x["corpus_CER"]):
            t = runtime.get(r["engine"], 0)
            f.write(
                f"  {r['engine']:<22}  CER={r['corpus_CER']*100:.2f}%  "
                f"WER={r['mean_WER']*100:.2f}%  "
                f"empty={r['lines_empty']}/{r['lines']}  "
                f"time={t:.1f}s\n"
            )

    print(f"Results written to:  {out_dir}/")
    print(f"  Per-engine predictions: {out_dir}/predictions/<engine>.tsv")
    print(f"  Summary TSV:            {tsv_path}")
    print(f"  Summary text:           {txt_path}")


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot OCR comparison for Vedic Sanskrit crops."
    )
    parser.add_argument(
        "--test_tsv",
        default="files/output/splits/test.tsv",
        help="Path to test split TSV (default: output/splits/test.tsv)",
    )
    parser.add_argument(
        "--crops_dir",
        default="files/output/crops",
        help="Directory containing PNG crops (default: output/crops)",
    )
    parser.add_argument(
        "--out_dir",
        default="files/output/zero_shot_results",
        help="Output directory (default: output/zero_shot_results)",
    )
    parser.add_argument(
        "--engines",
        nargs="+",
        default=ALL_ENGINES,
        choices=ALL_ENGINES,
        metavar="ENGINE",
        help=(
            "Engines to run. Choices: "
            + ", ".join(ALL_ENGINES)
            + "  (default: all)"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process first N lines (for quick debugging).",
    )
    args = parser.parse_args()

    test_tsv  = Path(args.test_tsv)
    crops_dir = Path(args.crops_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not test_tsv.exists():
        sys.exit(f"Test TSV not found: {test_tsv}")
    if not crops_dir.exists():
        sys.exit(f"Crops directory not found: {crops_dir}")

    rows = load_test_tsv(test_tsv)
    if args.limit:
        rows = rows[: args.limit]
    print(f"Loaded {len(rows)} lines from {test_tsv}")

    results = []
    runtime = {}

    for engine_name in args.engines:
        print(f"\n{'─'*50}")
        print(f"[{engine_name.upper()}] building …")
        try:
            predict_fn = ENGINE_BUILDERS[engine_name]()
        except Exception as e:
            print(f"  Build failed: {e}")
            traceback.print_exc()
            continue

        if predict_fn is None:
            continue   # skipped (missing install)

        t0 = time.time()
        try:
            stats = evaluate_engine(
                engine_name, predict_fn, rows, crops_dir, out_dir
            )
        except Exception as e:
            print(f"  Evaluation failed: {e}")
            traceback.print_exc()
            continue
        elapsed = time.time() - t0

        runtime[engine_name] = elapsed
        results.append(stats)
        print(
            f"  ✓ CER={stats['corpus_CER']*100:.2f}%  "
            f"WER={stats['mean_WER']*100:.2f}%  "
            f"empty={stats['lines_empty']}/{stats['lines']}  "
            f"({elapsed:.1f}s)"
        )

    if not results:
        print("\nNo engines produced results. Check your installs.")
        return

    print_table(results, runtime)
    write_summary(results, runtime, out_dir)


if __name__ == "__main__":
    main()