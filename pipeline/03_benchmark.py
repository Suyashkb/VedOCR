"""
Benchmarking script for binarisation methods.
Computes IoU, Dice, F-measure, Precision, Recall, and DRD metrics.

Author: Suyash Kumar Bhagat
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import json
from datetime import datetime
from tqdm import tqdm
import argparse

# Import MorphBG from the pipeline module
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import importlib.util
spec = importlib.util.spec_from_file_location("morphbg_pipeline", Path(__file__).parent / "01_morphbg.py")
morphbg_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(morphbg_module)
MorphBG = morphbg_module.MorphBG


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection over Union."""
    intersection = np.logical_and(pred > 0, gt > 0).sum()
    union = np.logical_or(pred > 0, gt > 0).sum()
    return intersection / union if union > 0 else 0.0


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice coefficient (F1 score)."""
    intersection = np.logical_and(pred > 0, gt > 0).sum()
    return 2 * intersection / (pred.sum() + gt.sum()) if (pred.sum() + gt.sum()) > 0 else 0.0


def compute_precision_recall_f1(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float, float]:
    """Compute precision, recall, and F1."""
    tp = np.logical_and(pred > 0, gt > 0).sum()
    fp = np.logical_and(pred > 0, gt == 0).sum()
    fn = np.logical_and(pred == 0, gt > 0).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def compute_drd(pred: np.ndarray, gt: np.ndarray, block_size: int = 8) -> float:
    """
    Distance Reciprocal Distortion (DRD) metric.

    Lower is better. Measures spatial distortion of strokes.

    Reference: H. Lu, A. Kot, Y. Shi, "Distance reciprocal distortion measure
    for binary document images," IEEE Signal Process. Lett., 2004.
    """
    # Ensure binary
    pred_bin = (pred > 0).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)

    # Find mismatched pixels
    mismatch = np.logical_xor(pred_bin, gt_bin)

    if mismatch.sum() == 0:
        return 0.0

    # Distance transform on ground truth
    # For each mismatch pixel, compute distance to nearest GT stroke
    dist_transform = cv2.distanceTransform(
        (1 - gt_bin).astype(np.uint8),
        cv2.DIST_L2,
        5
    )

    # DRD weights: closer to GT strokes = higher penalty
    # Use reciprocal: 1 / (1 + distance)
    weights = 1.0 / (1.0 + dist_transform)

    # Sum weighted distortion over mismatched pixels
    drd_sum = (mismatch * weights).sum()

    # Normalize by number of GT foreground pixels
    num_gt_pixels = gt_bin.sum()

    if num_gt_pixels == 0:
        return 0.0

    drd = drd_sum / num_gt_pixels

    return drd


def benchmark_image_pair(
    pred_path: Path,
    gt_path: Path
) -> Dict:
    """
    Compute all metrics for a single prediction-GT pair.

    Args:
        pred_path: Path to predicted binary image
        gt_path: Path to ground truth binary image

    Returns:
        Dictionary of metrics
    """
    # Read images
    pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)

    if pred is None or gt is None:
        return {'error': 'Failed to load images'}

    # Ensure same size
    if pred.shape != gt.shape:
        # Resize prediction to match GT
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Compute metrics
    iou = compute_iou(pred, gt)
    dice = compute_dice(pred, gt)
    precision, recall, f1 = compute_precision_recall_f1(pred, gt)
    drd = compute_drd(pred, gt)

    return {
        'iou': iou,
        'dice': dice,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'drd': drd,
        'pred_path': str(pred_path),
        'gt_path': str(gt_path)
    }


class BenchmarkRunner:
    """Run benchmarks on a set of images."""

    def __init__(
        self,
        processed_dir: str = "processed_images",
        gt_dir: str = "benchmark_folder",
        results_dir: str = "benchmark_results"
    ):
        """
        Initialize benchmark runner.

        Args:
            processed_dir: Directory with binarised predictions
            gt_dir: Directory with ground truth binary images
            results_dir: Directory to save results
        """
        self.processed_dir = Path(processed_dir)
        self.gt_dir = Path(gt_dir)
        self.results_dir = Path(results_dir)

        self.results_dir.mkdir(exist_ok=True, parents=True)

    def find_matching_pairs(self) -> List[Tuple[Path, Path]]:
        """
        Find matching prediction-GT pairs by filename.

        Returns:
            List of (pred_path, gt_path) tuples
        """
        pairs = []

        # Get all GT files
        gt_files = {f.stem: f for f in self.gt_dir.glob("*.png")}
        gt_files.update({f.stem: f for f in self.gt_dir.glob("*.jpg")})
        gt_files.update({f.stem: f for f in self.gt_dir.glob("*.tif")})

        # Match with predictions
        for pred_file in self.processed_dir.glob("*_binary.png"):
            # Remove _binary suffix to match original name
            original_stem = pred_file.stem.replace('_binary', '')

            if original_stem in gt_files:
                pairs.append((pred_file, gt_files[original_stem]))

        return pairs

    def run(self) -> Dict:
        """
        Run benchmark on all matching pairs.

        Returns:
            Results dictionary
        """
        pairs = self.find_matching_pairs()

        if not pairs:
            print(f"No matching prediction-GT pairs found")
            print(f"  Processed dir: {self.processed_dir}")
            print(f"  GT dir: {self.gt_dir}")
            return {}

        print(f"Found {len(pairs)} matching image pairs")

        # Compute metrics for each pair
        results = []
        for pred_path, gt_path in tqdm(pairs, desc="Computing metrics"):
            metrics = benchmark_image_pair(pred_path, gt_path)
            metrics['image_name'] = pred_path.stem.replace('_binary', '')
            results.append(metrics)

        # Aggregate statistics
        valid_results = [r for r in results if 'error' not in r]

        if not valid_results:
            print("No valid results computed")
            return {}

        aggregate = {
            'num_images': len(valid_results),
            'mean': {
                'iou': np.mean([r['iou'] for r in valid_results]),
                'dice': np.mean([r['dice'] for r in valid_results]),
                'precision': np.mean([r['precision'] for r in valid_results]),
                'recall': np.mean([r['recall'] for r in valid_results]),
                'f1': np.mean([r['f1'] for r in valid_results]),
                'drd': np.mean([r['drd'] for r in valid_results]),
            },
            'std': {
                'iou': np.std([r['iou'] for r in valid_results]),
                'dice': np.std([r['dice'] for r in valid_results]),
                'precision': np.std([r['precision'] for r in valid_results]),
                'recall': np.std([r['recall'] for r in valid_results]),
                'f1': np.std([r['f1'] for r in valid_results]),
                'drd': np.std([r['drd'] for r in valid_results]),
            }
        }

        # Save results
        output = {
            'timestamp': datetime.now().isoformat(),
            'processed_dir': str(self.processed_dir),
            'gt_dir': str(self.gt_dir),
            'aggregate': aggregate,
            'per_image': valid_results
        }

        results_path = self.results_dir / 'benchmark_results.json'
        with open(results_path, 'w') as f:
            json.dump(output, f, indent=2)

        # Print summary
        print(f"\n{'='*60}")
        print("Benchmark Results (Mean ± Std)")
        print(f"{'='*60}")
        print(f"  IoU:       {aggregate['mean']['iou']:.4f} ± {aggregate['std']['iou']:.4f}")
        print(f"  Dice:      {aggregate['mean']['dice']:.4f} ± {aggregate['std']['dice']:.4f}")
        print(f"  Precision: {aggregate['mean']['precision']:.4f} ± {aggregate['std']['precision']:.4f}")
        print(f"  Recall:    {aggregate['mean']['recall']:.4f} ± {aggregate['std']['recall']:.4f}")
        print(f"  F1:        {aggregate['mean']['f1']:.4f} ± {aggregate['std']['f1']:.4f}")
        print(f"  DRD:       {aggregate['mean']['drd']:.4f} ± {aggregate['std']['drd']:.4f}")
        print(f"{'='*60}")
        print(f"Results saved to: {results_path}\n")

        return output


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Benchmark binarisation results against ground truth"
    )
    parser.add_argument(
        '--processed-dir',
        type=str,
        default='processed_images',
        help='Directory with binarised predictions'
    )
    parser.add_argument(
        '--gt-dir',
        type=str,
        default='benchmark_folder',
        help='Directory with ground truth images'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='benchmark_results',
        help='Directory to save results'
    )

    args = parser.parse_args()

    # Run benchmark
    runner = BenchmarkRunner(
        processed_dir=args.processed_dir,
        gt_dir=args.gt_dir,
        results_dir=args.results_dir
    )

    runner.run()


if __name__ == "__main__":
    main()
