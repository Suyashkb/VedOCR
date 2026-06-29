"""
Zero-shot OCR evaluation on Sanskrit manuscripts.
Tests Tesseract, EasyOCR, and PaddleOCR without fine-tuning.

Author: Suyash Kumar Bhagat
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import json
from datetime import datetime
from tqdm import tqdm
import argparse
import Levenshtein

# OCR engines
import pytesseract
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("Warning: EasyOCR not available")

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    print("Warning: PaddleOCR not available")


def compute_cer(pred: str, gt: str) -> float:
    """Compute Character Error Rate."""
    if len(gt) == 0:
        return 1.0 if len(pred) > 0 else 0.0

    distance = Levenshtein.distance(pred, gt)
    return distance / len(gt)


def compute_wer(pred: str, gt: str) -> float:
    """Compute Word Error Rate."""
    pred_words = pred.split()
    gt_words = gt.split()

    if len(gt_words) == 0:
        return 1.0 if len(pred_words) > 0 else 0.0

    distance = Levenshtein.distance(' '.join(pred_words), ' '.join(gt_words))
    return distance / sum(len(w) + 1 for w in gt_words)  # +1 for spaces


class OCREngine:
    """Base class for OCR engines."""

    def __init__(self, name: str):
        self.name = name

    def recognize(self, image_path: str) -> str:
        """Run OCR on an image. Returns recognized text."""
        raise NotImplementedError


class TesseractEngine(OCREngine):
    """Tesseract OCR engine."""

    def __init__(self, lang: str = 'san+hin'):
        """
        Initialize Tesseract.

        Args:
            lang: Language pack (default: san+hin for Sanskrit+Hindi)
        """
        super().__init__(f"Tesseract-{lang}")
        self.lang = lang

    def recognize(self, image_path: str) -> str:
        """Run Tesseract OCR."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return ""

            text = pytesseract.image_to_string(
                image,
                lang=self.lang,
                config='--psm 6'  # Assume uniform block of text
            )

            return text.strip()

        except Exception as e:
            print(f"Tesseract error on {image_path}: {e}")
            return ""


class EasyOCREngine(OCREngine):
    """EasyOCR engine."""

    def __init__(self):
        """Initialize EasyOCR with Devanagari."""
        super().__init__("EasyOCR-Devanagari")

        if not EASYOCR_AVAILABLE:
            raise ImportError("EasyOCR not installed")

        self.reader = easyocr.Reader(['hi'], gpu=False)  # Hindi/Devanagari

    def recognize(self, image_path: str) -> str:
        """Run EasyOCR."""
        try:
            results = self.reader.readtext(image_path, detail=0)
            return ' '.join(results).strip()

        except Exception as e:
            print(f"EasyOCR error on {image_path}: {e}")
            return ""


class PaddleOCREngine(OCREngine):
    """PaddleOCR engine."""

    def __init__(self):
        """Initialize PaddleOCR with Hindi."""
        super().__init__("PaddleOCR-Hindi")

        if not PADDLEOCR_AVAILABLE:
            raise ImportError("PaddleOCR not installed")

        self.ocr = PaddleOCR(
            lang='hi',
            use_angle_cls=True,
            use_gpu=False,
            show_log=False
        )

    def recognize(self, image_path: str) -> str:
        """Run PaddleOCR."""
        try:
            results = self.ocr.ocr(image_path, cls=True)

            if not results or not results[0]:
                return ""

            # Extract text from results
            texts = [line[1][0] for line in results[0]]
            return ' '.join(texts).strip()

        except Exception as e:
            print(f"PaddleOCR error on {image_path}: {e}")
            return ""


class ZeroShotEvaluator:
    """Evaluate multiple OCR engines on annotated data."""

    def __init__(
        self,
        parsed_annotations_file: str,
        results_dir: str = "results/zero_shot_ocr"
    ):
        """
        Initialize evaluator.

        Args:
            parsed_annotations_file: Path to parsed annotations JSON
            results_dir: Directory to save results
        """
        self.annotations_file = Path(parsed_annotations_file)
        self.results_dir = Path(results_dir)

        self.results_dir.mkdir(exist_ok=True, parents=True)

        # Load annotations
        with open(self.annotations_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.lines = self.data['lines']

        print(f"Loaded {len(self.lines)} annotated lines")

    def evaluate_engine(self, engine: OCREngine, max_samples: int = None) -> Dict:
        """
        Evaluate a single OCR engine.

        Args:
            engine: OCR engine to evaluate
            max_samples: Maximum number of samples (for testing)

        Returns:
            Results dictionary
        """
        print(f"\nEvaluating {engine.name}...")

        lines_to_eval = self.lines[:max_samples] if max_samples else self.lines

        results = []
        cer_scores = []
        wer_scores = []

        for line in tqdm(lines_to_eval, desc=f"{engine.name}"):
            # Run OCR
            image_path = line['line_image_path']
            gt_text = line['text']

            pred_text = engine.recognize(image_path)

            # Compute metrics
            cer = compute_cer(pred_text, gt_text)
            wer = compute_wer(pred_text, gt_text)

            cer_scores.append(cer)
            wer_scores.append(wer)

            results.append({
                'line_filename': line['line_filename'],
                'image_path': image_path,
                'ground_truth': gt_text,
                'prediction': pred_text,
                'cer': cer,
                'wer': wer
            })

        # Aggregate statistics
        output = {
            'engine': engine.name,
            'timestamp': datetime.now().isoformat(),
            'num_lines': len(results),
            'metrics': {
                'mean_cer': float(np.mean(cer_scores)),
                'std_cer': float(np.std(cer_scores)),
                'mean_wer': float(np.mean(wer_scores)),
                'std_wer': float(np.std(wer_scores)),
                'accuracy': float(1.0 - np.mean(cer_scores))  # Character accuracy
            },
            'per_line_results': results
        }

        # Save results
        engine_name_safe = engine.name.replace('+', '_').replace(' ', '_')
        results_file = self.results_dir / f"{engine_name_safe}_results.json"

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        # Print summary
        print(f"\n{engine.name} Results:")
        print(f"  CER: {output['metrics']['mean_cer']:.4f} ± {output['metrics']['std_cer']:.4f}")
        print(f"  WER: {output['metrics']['mean_wer']:.4f} ± {output['metrics']['std_wer']:.4f}")
        print(f"  Accuracy: {output['metrics']['accuracy']*100:.2f}%")
        print(f"  Results saved to: {results_file}")

        return output

    def evaluate_all(self, max_samples: int = None) -> Dict:
        """
        Evaluate all available OCR engines.

        Args:
            max_samples: Maximum samples per engine (for testing)

        Returns:
            Comparison dictionary
        """
        engines = []

        # Tesseract variants
        for lang in ['san', 'hin', 'san+hin']:
            try:
                engines.append(TesseractEngine(lang))
            except Exception as e:
                print(f"Failed to initialize Tesseract-{lang}: {e}")

        # EasyOCR
        if EASYOCR_AVAILABLE:
            try:
                engines.append(EasyOCREngine())
            except Exception as e:
                print(f"Failed to initialize EasyOCR: {e}")

        # PaddleOCR
        if PADDLEOCR_AVAILABLE:
            try:
                engines.append(PaddleOCREngine())
            except Exception as e:
                print(f"Failed to initialize PaddleOCR: {e}")

        if not engines:
            print("No OCR engines available!")
            return {}

        # Evaluate each engine
        all_results = []

        for engine in engines:
            result = self.evaluate_engine(engine, max_samples)
            all_results.append(result)

        # Create comparison summary
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'num_lines_evaluated': len(self.lines[:max_samples] if max_samples else self.lines),
            'engines': [
                {
                    'name': r['engine'],
                    'mean_cer': r['metrics']['mean_cer'],
                    'mean_wer': r['metrics']['mean_wer'],
                    'accuracy': r['metrics']['accuracy']
                }
                for r in all_results
            ]
        }

        # Save comparison
        comparison_file = self.results_dir / "comparison.json"
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2)

        # Print comparison table
        print(f"\n{'='*80}")
        print("Zero-Shot OCR Comparison")
        print(f"{'='*80}")
        print(f"{'Engine':<25} {'CER':<15} {'WER':<15} {'Accuracy':<10}")
        print(f"{'-'*80}")

        for engine_result in comparison['engines']:
            print(f"{engine_result['name']:<25} "
                  f"{engine_result['mean_cer']:<15.4f} "
                  f"{engine_result['mean_wer']:<15.4f} "
                  f"{engine_result['accuracy']*100:<10.2f}%")

        print(f"{'='*80}")
        print(f"Comparison saved to: {comparison_file}\n")

        return comparison


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Zero-shot OCR evaluation on Sanskrit manuscripts"
    )
    parser.add_argument(
        '--annotations',
        type=str,
        default='parsed_annotations/parsed_annotations.json',
        help='Path to parsed annotations JSON'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results/zero_shot_ocr',
        help='Directory to save results'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum samples to evaluate (for testing)'
    )
    parser.add_argument(
        '--engine',
        type=str,
        choices=['tesseract-san', 'tesseract-hin', 'tesseract-san+hin', 'easyocr', 'paddleocr', 'all'],
        default='all',
        help='Which engine to evaluate'
    )

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = ZeroShotEvaluator(
        parsed_annotations_file=args.annotations,
        results_dir=args.results_dir
    )

    # Evaluate
    if args.engine == 'all':
        evaluator.evaluate_all(max_samples=args.max_samples)
    else:
        # Single engine
        if args.engine.startswith('tesseract'):
            lang = args.engine.replace('tesseract-', '')
            engine = TesseractEngine(lang)
        elif args.engine == 'easyocr':
            engine = EasyOCREngine()
        elif args.engine == 'paddleocr':
            engine = PaddleOCREngine()

        evaluator.evaluate_engine(engine, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
