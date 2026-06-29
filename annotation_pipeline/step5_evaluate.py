"""
step5_evaluate.py — Zero-shot CER/WER evaluation on the test split.

Supports:
  --engine tesseract   Run Tesseract (must have 'san' or 'hin' traineddata installed)
  --engine google      Google Cloud Vision API  (needs GOOGLE_APPLICATION_CREDENTIALS)
  --engine file        Load predictions from a TSV file  (--pred-file path/to/pred.tsv)

Writes:
  output/eval/results_{engine}.tsv   per-line: crop_path, ground_truth, predicted, cer
  output/eval/summary_{engine}.txt   aggregate CER / WER

Usage examples:
  python step5_evaluate.py --engine tesseract --lang san
  python step5_evaluate.py --engine file --pred-file my_predictions.tsv
"""

import argparse
import csv
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import OUTPUT_DIR, SPLITS_DIR


# ────────────────────────────────────────────────────────────────────────────
# CER / WER  (pure Python, no extra dependencies)
# ────────────────────────────────────────────────────────────────────────────

def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two strings."""
    if a == b:
        return 0
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            curr[j] = (
                prev[j - 1] if a[i - 1] == b[j - 1]
                else 1 + min(curr[j - 1], prev[j], prev[j - 1])
            )
        prev = curr
    return prev[n]


def cer(ref: str, hyp: str) -> float:
    """Character Error Rate = edit_distance(ref, hyp) / len(ref)."""
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def wer(ref: str, hyp: str) -> float:
    """Word Error Rate on whitespace-split tokens."""
    ref_words = ref.split()
    hyp_words = hyp.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return edit_distance(ref_words, hyp_words) / len(ref_words)


def normalize(text: str) -> str:
    """NFC normalize and strip leading/trailing whitespace."""
    return unicodedata.normalize("NFC", text.strip())


# ────────────────────────────────────────────────────────────────────────────
# Engines
# ────────────────────────────────────────────────────────────────────────────

def predict_tesseract(image_path: str, lang: str = "san") -> str:
    """Run Tesseract on a single line image and return predicted text."""
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(image_path)
        # PSM 7 = single text line
        config = f"--psm 7 --oem 1 -l {lang}"
        return pytesseract.image_to_string(img, config=config).strip()
    except ImportError:
        raise SystemExit("pytesseract not installed.  pip install pytesseract")
    except Exception as e:
        return f"<error: {e}>"


def predict_google_vision(image_path: str) -> str:
    """Run Google Cloud Vision DOCUMENT_TEXT_DETECTION on a single image."""
    try:
        from google.cloud import vision
        client = vision.ImageAnnotatorClient()
        with open(image_path, "rb") as f:
            content = f.read()
        image  = vision.Image(content=content)
        response = client.document_text_detection(image=image)
        return response.full_text_annotation.text.strip().replace("\n", " ")
    except ImportError:
        raise SystemExit("google-cloud-vision not installed.  pip install google-cloud-vision")
    except Exception as e:
        return f"<error: {e}>"


def load_predictions_from_file(pred_file: str) -> dict[str, str]:
    """
    Load pre-computed predictions from a two-column TSV:
      image_path \\t predicted_text
    Returns a dict {image_path: predicted_text}.
    """
    preds = {}
    with open(pred_file, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                preds[parts[0]] = parts[1]
    return preds


# ────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ────────────────────────────────────────────────────────────────────────────

def evaluate(engine: str, lang: str = "san", pred_file: str | None = None,
             split: str = "test"):
    print(f"\n── Step 5: Evaluate [{engine}] on [{split}] split ─────────────")

    test_path = SPLITS_DIR / f"{split}.tsv"
    if not test_path.is_file():
        print(f"  {test_path} not found.  Run step2_split.py first.")
        return

    with open(test_path, encoding="utf-8") as f:
        records = list(csv.DictReader(f, delimiter="\t"))
    print(f"  {len(records)} lines to evaluate …")

    # pre-load predictions if using file engine
    file_preds = {}
    if engine == "file":
        if not pred_file:
            raise SystemExit("--pred-file required when using --engine file")
        file_preds = load_predictions_from_file(pred_file)

    out_dir = OUTPUT_DIR / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"results_{engine}.tsv"

    total_chars = 0
    total_edits = 0
    results     = []

    with open(results_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["crop_path", "ground_truth", "predicted", "cer", "wer"])

        for i, r in enumerate(records):
            ref = normalize(r["transcription"])
            img = r["crop_path"]

            # ── get prediction ─────────────────────────────────────────────
            if engine == "tesseract":
                hyp = normalize(predict_tesseract(img, lang=lang))
            elif engine == "google":
                hyp = normalize(predict_google_vision(img))
            elif engine == "file":
                hyp = normalize(file_preds.get(img, ""))
            else:
                raise ValueError(f"Unknown engine: {engine}")

            line_cer = cer(ref, hyp)
            line_wer = wer(ref, hyp)

            total_chars += len(ref)
            total_edits += edit_distance(ref, hyp)

            writer.writerow([img, ref, hyp, f"{line_cer:.4f}", f"{line_wer:.4f}"])
            results.append((img, ref, hyp, line_cer))

            if (i + 1) % 50 == 0:
                running_cer = total_edits / max(total_chars, 1)
                print(f"    {i+1}/{len(records)}  running CER = {running_cer:.3f}")

    # ── aggregate ────────────────────────────────────────────────────────────
    overall_cer = total_edits / max(total_chars, 1)
    mean_line_cer = sum(r[3] for r in results) / len(results)

    summary_path = out_dir / f"summary_{engine}.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        lines = [
            f"Engine          : {engine}",
            f"Split           : {split}",
            f"Lines evaluated : {len(results)}",
            f"Total ref chars : {total_chars}",
            f"Total edits     : {total_edits}",
            f"Overall CER     : {overall_cer:.4f}  ({overall_cer*100:.2f}%)",
            f"Mean line CER   : {mean_line_cer:.4f}  ({mean_line_cer*100:.2f}%)",
        ]
        f.write("\n".join(lines) + "\n")
        print("\n" + "\n".join(lines))

    # ── worst lines ──────────────────────────────────────────────────────────
    worst = sorted(results, key=lambda x: x[3], reverse=True)[:10]
    print("\n  Worst 10 lines:")
    for img, ref, hyp, c in worst:
        print(f"    CER={c:.3f}  REF: {ref[:40]}")
        print(f"           HYP: {hyp[:40]}")

    print(f"\n  Results → {results_path}")
    print(f"  Summary → {summary_path}")


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Zero-shot OCR evaluation")
    parser.add_argument("--engine",    default="tesseract",
                        choices=["tesseract", "google", "file"],
                        help="OCR engine to use")
    parser.add_argument("--lang",      default="san",
                        help="Tesseract language code (default: san)")
    parser.add_argument("--pred-file", default=None,
                        help="TSV of predictions when --engine file")
    parser.add_argument("--split",     default="test",
                        choices=["train", "val", "test"],
                        help="Which split to evaluate (default: test)")
    args = parser.parse_args()

    evaluate(
        engine    = args.engine,
        lang      = args.lang,
        pred_file = args.pred_file,
        split     = args.split,
    )


if __name__ == "__main__":
    main()
