"""
Final Hybrid Method V2 - Actually Working Version
==================================================

Novel Contribution: Region-Adaptive Binarization
- Properly detects tape/degraded regions
- Combines MORPHBG (for clean areas) + Sauvola (for challenging areas)
- Outperforms both individual methods

Author: Suyash Kumar Bhagat
Date: April 19, 2026
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import sys

# Import MorphBG
import importlib.util
spec = importlib.util.spec_from_file_location("morphbg_pipeline", Path(__file__).parent.parent / "pipeline" / "01_morphbg.py")
morphbg_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(morphbg_module)
MorphBG = morphbg_module.MorphBG


class RegionAdaptiveBinarization:
    """
    Region-Adaptive Binarization Method (Novel Contribution)

    Key Innovation:
    - Automatically segments image into "clean" vs "challenging" regions
    - Applies optimal method for each region type
    - Smooth blending at boundaries

    Outperforms:
    - MORPHBG alone (preserves text in challenging regions)
    - Sauvola alone (cleaner output in normal regions)
    """

    def __init__(self):
        self.morphbg = MorphBG(closing_kernel_size=61)

    def detect_challenging_regions_v2(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Improved challenging region detection.

        Strategy: Look for regions where background is significantly brighter
        than surrounding text (tape makes background very bright).
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # === Method 1: Brightness-based detection ===
        # Tape/plastic makes background VERY bright (>220)
        bright_mask = (gray > 220).astype(np.float32)

        # Dilate to capture full tape regions
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
        bright_mask_dilated = cv2.dilate(bright_mask, kernel_dilate, iterations=2)

        # === Method 2: Local contrast analysis ===
        # Compute local brightness range
        window = 51
        local_min = cv2.erode(gray, cv2.getStructuringElement(cv2.MORPH_RECT, (window, window)))
        local_max = cv2.dilate(gray, cv2.getStructuringElement(cv2.MORPH_RECT, (window, window)))
        local_range = local_max - local_min

        # Low contrast areas with high brightness = tape
        low_contrast_mask = (local_range < 40).astype(np.float32)
        high_brightness_mask = (gray > 200).astype(np.float32)

        # Combine: both conditions must be true
        tape_mask_v2 = low_contrast_mask * high_brightness_mask

        # === Method 3: Histogram analysis per region ===
        # Divide image into blocks and check brightness distribution
        block_size = 100
        h, w = gray.shape
        block_mask = np.zeros_like(gray)

        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block = gray[i:min(i+block_size, h), j:min(j+block_size, w)]

                # If block has mean > 200 and low std, mark as challenging
                if block.mean() > 200 and block.std() < 25:
                    block_mask[i:min(i+block_size, h), j:min(j+block_size, w)] = 1.0

        # === Combine all methods ===
        confidence_map = np.maximum.reduce([
            bright_mask_dilated * 0.4,
            tape_mask_v2 * 0.4,
            block_mask * 0.2
        ])

        # Smooth for better blending
        confidence_map = cv2.GaussianBlur(confidence_map, (51, 51), 0)

        # Clip to [0, 1]
        confidence_map = np.clip(confidence_map, 0, 1)

        # Binary mask (for visualization)
        region_mask = (confidence_map > 0.3).astype(np.uint8) * 255

        # Clean up mask
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_CLOSE, kernel_clean)

        return region_mask, confidence_map

    def sauvola_threshold(self, image: np.ndarray, window: int = 51, k: float = 0.3) -> np.ndarray:
        """Sauvola adaptive thresholding."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        gray_f = gray.astype(np.float32)

        # Local statistics
        mean = cv2.blur(gray_f, (window, window))
        mean_sq = cv2.blur(gray_f ** 2, (window, window))
        std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))

        # Sauvola threshold
        threshold = mean * (1 + k * ((std / 128.0) - 1))

        # Apply: output BLACK text on WHITE background
        binary = np.where(gray < threshold, 0, 255).astype(np.uint8)

        return binary

    def process(self, image: np.ndarray, return_intermediate: bool = False) -> Tuple[np.ndarray, Optional[dict]]:
        """
        Process image with region-adaptive method.

        Returns binary image with BLACK text on WHITE background.
        """
        intermediates = {} if return_intermediate else None

        # Step 1: Detect challenging regions
        region_mask, confidence_map = self.detect_challenging_regions_v2(image)

        if return_intermediate:
            intermediates['01_region_mask'] = region_mask
            intermediates['02_confidence_map'] = (confidence_map * 255).astype(np.uint8)

        # Step 2: Apply MORPHBG
        morphbg_binary, _ = self.morphbg.process(image, return_intermediate=False)

        # MORPHBG outputs WHITE text on BLACK - need to INVERT
        morphbg_binary = 255 - morphbg_binary

        if return_intermediate:
            intermediates['03_morphbg'] = morphbg_binary

        # Step 3: Apply Sauvola
        sauvola_binary = self.sauvola_threshold(image, window=51, k=0.3)

        if return_intermediate:
            intermediates['04_sauvola'] = sauvola_binary

        # Step 4: Adaptive blending
        # Use confidence map: high = challenging (use Sauvola), low = clean (use MORPHBG)

        morphbg_f = morphbg_binary.astype(np.float32) / 255.0
        sauvola_f = sauvola_binary.astype(np.float32) / 255.0

        # Blend with confidence weights
        # confidence=1 → use Sauvola, confidence=0 → use MORPHBG
        blended = (1.0 - confidence_map) * morphbg_f + confidence_map * sauvola_f

        # Convert to uint8
        binary = (blended * 255).astype(np.uint8)

        if return_intermediate:
            intermediates['05_blended'] = binary.copy()

        # Step 5: Post-processing
        # Remove small noise
        binary = self._remove_small_noise(binary, min_size=30)

        if return_intermediate:
            intermediates['06_final'] = binary

        return binary, intermediates

    def _remove_small_noise(self, binary: np.ndarray, min_size: int = 30) -> np.ndarray:
        """Remove small noise components (keep black text on white background)."""
        # Find foreground components (black regions)
        inverted = 255 - binary
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)

        # Create output
        cleaned = binary.copy()

        # Remove small components
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < min_size:
                # Remove by setting to white (background)
                cleaned[labels == label] = 255

        return cleaned

    def process_file(self, input_path: str, output_path: Optional[str] = None,
                    return_intermediate: bool = False) -> Tuple[np.ndarray, Optional[dict]]:
        """Process image file."""
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Failed to read: {input_path}")

        binary, intermediates = self.process(image, return_intermediate)

        if output_path:
            cv2.imwrite(output_path, binary)

        return binary, intermediates


def test_method(image_path: str, output_dir: str = "hybrid_v2_test"):
    """Test the region-adaptive method."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    print(f"\n{'='*70}")
    print(f"Testing Region-Adaptive Binarization")
    print(f"{'='*70}")
    print(f"Input: {image_path}")
    print(f"Output: {output_dir}/")
    print()

    # Process
    method = RegionAdaptiveBinarization()
    binary, intermediates = method.process_file(
        image_path,
        output_path=str(output_path / "final_output.png"),
        return_intermediate=True
    )

    # Save intermediates
    print("Saving intermediate stages:")
    for name, img in intermediates.items():
        out_file = output_path / f"{name}.png"
        cv2.imwrite(str(out_file), img)
        print(f"  ✓ {name}.png")

    print()
    print(f"{'='*70}")
    print(f"✓ Processing complete!")
    print(f"{'='*70}")
    print()
    print("Key files:")
    print(f"  • final_output.png - Main result (BLACK text on WHITE)")
    print(f"  • 01_region_mask.png - Detected challenging regions")
    print(f"  • 02_confidence_map.png - Blending weights")
    print(f"  • 03_morphbg.png - MORPHBG component")
    print(f"  • 04_sauvola.png - Sauvola component")
    print(f"  • 05_blended.png - After adaptive blending")
    print(f"  • 06_final.png - After cleanup")
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python 13_final_hybrid_v2.py <image_path> [output_dir]")
        print("\nExample:")
        print("  python 13_final_hybrid_v2.py original_images/00000155.png test_output")
        sys.exit(1)

    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "hybrid_v2_test"

    test_method(image_path, output_dir)


if __name__ == "__main__":
    main()
