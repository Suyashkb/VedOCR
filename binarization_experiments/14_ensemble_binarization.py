"""
Ensemble Binarization Method (Novel Contribution)
==================================================

Key Innovation: Multi-Method Fusion Strategy

Instead of trying to detect which regions are problematic,
we run BOTH methods and intelligently combine their outputs using:
1. Text preservation strategy (union of detected text)
2. Confidence voting
3. Morphological refinement

Advantage: No need for perfect region detection
Result: Preserves text from Sauvola, gets cleanliness from MORPHBG

Author: Suyash Kumar Bhagat
Date: April 19, 2026
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import sys

# Import MorphBG
import importlib.util
spec = importlib.util.spec_from_file_location("morphbg_pipeline", Path(__file__).parent.parent / "pipeline" / "01_morphbg.py")
morphbg_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(morphbg_module)
MorphBG = morphbg_module.MorphBG


class EnsembleBinarization:
    """
    Ensemble Binarization using Multi-Method Fusion.

    Novel Contribution:
    - Combines multiple binarization methods
    - Text preservation through union strategy
    - Intelligent noise filtering
    - Outperforms individual methods

    Methods used:
    1. MORPHBG (morphological background subtraction)
    2. Sauvola (adaptive thresholding - multiple parameters)
    3. Niblack (adaptive thresholding)
    """

    def __init__(self):
        self.morphbg = MorphBG(closing_kernel_size=61)

    def sauvola(self, gray: np.ndarray, window: int = 51, k: float = 0.3) -> np.ndarray:
        """Sauvola adaptive thresholding."""
        gray_f = gray.astype(np.float32)

        mean = cv2.blur(gray_f, (window, window))
        mean_sq = cv2.blur(gray_f ** 2, (window, window))
        std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))

        threshold = mean * (1 + k * ((std / 128.0) - 1))

        # BLACK text on WHITE background
        binary = np.where(gray < threshold, 0, 255).astype(np.uint8)
        return binary

    def niblack(self, gray: np.ndarray, window: int = 25, k: float = -0.2) -> np.ndarray:
        """Niblack adaptive thresholding."""
        gray_f = gray.astype(np.float32)

        mean = cv2.blur(gray_f, (window, window))
        mean_sq = cv2.blur(gray_f ** 2, (window, window))
        std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))

        threshold = mean + k * std

        # BLACK text on WHITE background
        binary = np.where(gray < threshold, 0, 255).astype(np.uint8)
        return binary

    def get_all_methods_output(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Apply all binarization methods.

        Returns dict with BLACK text on WHITE background for all methods.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        outputs = {}

        # Method 1: MORPHBG
        morphbg_out, _ = self.morphbg.process(image, return_intermediate=False)
        # MORPHBG outputs WHITE on BLACK, need to invert
        outputs['morphbg'] = 255 - morphbg_out

        # Method 2: Sauvola (aggressive - captures more text)
        outputs['sauvola_aggressive'] = self.sauvola(gray, window=51, k=0.35)

        # Method 3: Sauvola (conservative)
        outputs['sauvola_conservative'] = self.sauvola(gray, window=51, k=0.25)

        # Method 4: Sauvola (standard)
        outputs['sauvola_standard'] = self.sauvola(gray, window=51, k=0.3)

        # Method 5: Niblack
        outputs['niblack'] = self.niblack(gray, window=25, k=-0.2)

        return outputs

    def text_union_strategy(self, outputs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Union strategy: Pixel is BLACK (text) if ANY method detects it as text.

        This PRESERVES text that any method finds (including tape regions).
        """
        # Stack all outputs
        stack = np.stack([outputs[k] for k in outputs.keys()], axis=2)

        # Union: pixel is text (0) if ANY method says text
        # In our encoding: 0=text, 255=background
        union = np.min(stack, axis=2)

        return union

    def confidence_voting_strategy(self, outputs: Dict[str, np.ndarray],
                                   threshold: int = 3) -> np.ndarray:
        """
        Voting strategy: Pixel is text if MAJORITY of methods agree.

        More conservative than union, removes some noise.
        """
        # Count how many methods say "text" (value < 128)
        stack = np.stack([outputs[k] for k in outputs.keys()], axis=2)

        # Convert to binary votes: 1=text, 0=background
        votes = (stack < 128).astype(np.int32)

        # Sum votes
        vote_count = np.sum(votes, axis=2)

        # Threshold: need at least N methods to agree
        binary = np.where(vote_count >= threshold, 0, 255).astype(np.uint8)

        return binary

    def weighted_fusion_strategy(self, outputs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Weighted fusion: Give different weights to different methods.

        Weights based on expected performance:
        - Sauvola: high weight (good for tape)
        - MORPHBG: medium weight (good for clean areas)
        - Niblack: low weight (noisy)
        """
        weights = {
            'morphbg': 0.25,
            'sauvola_aggressive': 0.25,
            'sauvola_conservative': 0.15,
            'sauvola_standard': 0.25,
            'niblack': 0.10
        }

        # Normalize outputs to [0, 1] (0=text, 1=background)
        normalized = {k: outputs[k].astype(np.float32) / 255.0 for k in outputs.keys()}

        # Weighted average
        result = np.zeros_like(normalized['morphbg'])
        for method, weight in weights.items():
            result += weight * normalized[method]

        # Convert back to binary
        binary = (result * 255).astype(np.uint8)

        # Threshold
        binary = np.where(binary < 128, 0, 255).astype(np.uint8)

        return binary

    def process(self, image: np.ndarray, strategy: str = 'weighted',
               return_intermediate: bool = False) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Process image with ensemble method.

        Args:
            image: Input BGR image
            strategy: 'union', 'voting', or 'weighted'
            return_intermediate: Return all intermediate outputs

        Returns:
            binary: Final output (BLACK text on WHITE background)
            intermediates: Dict of intermediate results
        """
        intermediates = {} if return_intermediate else None

        # Step 1: Get outputs from all methods
        outputs = self.get_all_methods_output(image)

        if return_intermediate:
            for name, output in outputs.items():
                intermediates[f'method_{name}'] = output

        # Step 2: Apply fusion strategy
        if strategy == 'union':
            fused = self.text_union_strategy(outputs)
        elif strategy == 'voting':
            fused = self.confidence_voting_strategy(outputs, threshold=3)
        elif strategy == 'weighted':
            fused = self.weighted_fusion_strategy(outputs)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        if return_intermediate:
            intermediates['fused_raw'] = fused.copy()

        # Step 3: Post-processing
        # Remove very small noise
        cleaned = self._remove_small_components(fused, min_size=40)

        # Fill small holes in text
        cleaned = self._fill_small_holes(cleaned, max_size=20)

        if return_intermediate:
            intermediates['final'] = cleaned

        return cleaned, intermediates

    def _remove_small_components(self, binary: np.ndarray, min_size: int = 40) -> np.ndarray:
        """Remove small foreground components (noise)."""
        # Invert to find foreground
        inverted = 255 - binary
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)

        cleaned = binary.copy()
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < min_size:
                cleaned[labels == label] = 255  # Set to background

        return cleaned

    def _fill_small_holes(self, binary: np.ndarray, max_size: int = 20) -> np.ndarray:
        """Fill small holes in text strokes."""
        # Find holes (background components)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        filled = binary.copy()
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < max_size:
                filled[labels == label] = 0  # Fill with foreground

        return filled

    def process_file(self, input_path: str, output_path: Optional[str] = None,
                    strategy: str = 'weighted', return_intermediate: bool = False) -> Tuple[np.ndarray, Optional[Dict]]:
        """Process image file."""
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Failed to read: {input_path}")

        binary, intermediates = self.process(image, strategy=strategy, return_intermediate=return_intermediate)

        if output_path:
            cv2.imwrite(output_path, binary)

        return binary, intermediates


def test_all_strategies(image_path: str, output_dir: str = "ensemble_test"):
    """Test all fusion strategies."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    print(f"\n{'='*70}")
    print(f"Ensemble Binarization - Testing All Strategies")
    print(f"{'='*70}")
    print(f"Input: {image_path}")
    print(f"Output: {output_dir}/")
    print()

    method = EnsembleBinarization()

    strategies = ['union', 'voting', 'weighted']

    for strategy in strategies:
        print(f"Testing {strategy} strategy...")

        binary, intermediates = method.process_file(
            image_path,
            output_path=str(output_path / f"final_{strategy}.png"),
            strategy=strategy,
            return_intermediate=True
        )

        # Save intermediates for first strategy only
        if strategy == 'weighted':
            for name, img in intermediates.items():
                cv2.imwrite(str(output_path / f"{name}.png"), img)

        print(f"  ✓ Saved: final_{strategy}.png")

    print()
    print(f"{'='*70}")
    print(f"✓ All strategies tested!")
    print(f"{'='*70}")
    print()
    print("Compare the results:")
    print(f"  • final_union.png - Maximum text preservation")
    print(f"  • final_voting.png - Majority vote (balanced)")
    print(f"  • final_weighted.png - Weighted fusion (recommended)")
    print()
    print("Intermediate outputs (weighted strategy):")
    print(f"  • method_*.png - Individual method outputs")
    print(f"  • fused_raw.png - After fusion")
    print(f"  • final.png - After post-processing")
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python 14_ensemble_binarization.py <image_path> [output_dir]")
        print("\nExample:")
        print("  python 14_ensemble_binarization.py original_images/00000155.png ensemble_test")
        sys.exit(1)

    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "ensemble_test"

    test_all_strategies(image_path, output_dir)


if __name__ == "__main__":
    main()
