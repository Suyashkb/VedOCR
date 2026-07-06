"""
Improved Hybrid Binarization Method
====================================

Novel contribution: Adaptive hybrid approach that:
1. Detects challenging regions (tape, degradation, uneven lighting)
2. Applies optimal method for each region
3. Combines results with intelligent blending

Goal: Outperform both MORPHBG and Sauvola individually

Author: Suyash Kumar Bhagat
Date: April 19, 2026
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import sys

# Import MorphBG and Sauvola from experimental pipeline
import importlib.util
spec = importlib.util.spec_from_file_location("morphbg_pipeline", Path(__file__).parent.parent / "pipeline" / "01_morphbg.py")
morphbg_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(morphbg_module)
MorphBG = morphbg_module.MorphBG


class ImprovedHybridMethod:
    """
    Improved hybrid binarization with proper artifact detection.

    Key improvements:
    1. Multi-modal artifact detection (not just tape)
    2. Confidence-weighted blending
    3. Edge-aware transitions
    4. Post-processing refinement
    """

    def __init__(self):
        self.morphbg = MorphBG(closing_kernel_size=61)

    def detect_challenging_regions(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect challenging regions using multiple indicators.

        Returns:
            region_mask: Binary mask (255=challenging, 0=normal)
            confidence_map: Float [0-1] confidence that region is challenging
        """
        h, w = image.shape[:2]

        # Convert to LAB for better color analysis
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0].astype(np.float32)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # ==== Indicator 1: High brightness with low variance (tape/adhesive) ====
        window_size = 51
        l_mean = cv2.blur(l_channel, (window_size, window_size))
        l_sq_mean = cv2.blur(l_channel ** 2, (window_size, window_size))
        l_var = l_sq_mean - l_mean ** 2

        # Tape regions: bright AND smooth
        tape_indicator = np.logical_and(l_mean > 180, l_var < 150).astype(np.float32)

        # ==== Indicator 2: Color saturation analysis ====
        # Tape/plastic often reduces saturation
        a_channel = lab[:, :, 1].astype(np.float32)
        b_channel = lab[:, :, 2].astype(np.float32)
        saturation = np.sqrt(a_channel**2 + b_channel**2)

        sat_mean = cv2.blur(saturation, (window_size, window_size))
        low_sat_indicator = (sat_mean < 30).astype(np.float32)

        # ==== Indicator 3: Texture analysis ====
        # Compute local standard deviation in grayscale
        gray_mean = cv2.blur(gray, (window_size, window_size))
        gray_sq_mean = cv2.blur(gray ** 2, (window_size, window_size))
        gray_std = np.sqrt(np.maximum(gray_sq_mean - gray_mean ** 2, 0))

        # Low texture (smooth regions) = challenging
        low_texture_indicator = (gray_std < 15).astype(np.float32)

        # ==== Indicator 4: Brightness gradient ====
        # Sharp brightness changes indicate tape edges
        gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)
        grad_x = cv2.Sobel(gray_blur, cv2.CV_32F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(gray_blur, cv2.CV_32F, 0, 1, ksize=5)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)

        # Normalize and threshold
        gradient_norm = cv2.normalize(gradient_mag, None, 0, 1, cv2.NORM_MINMAX)
        edge_indicator = (gradient_norm > 0.2).astype(np.float32)

        # ==== Combine indicators with weights ====
        confidence_map = (
            0.35 * tape_indicator +           # Primary: tape detection
            0.25 * low_sat_indicator +        # Secondary: saturation
            0.20 * low_texture_indicator +    # Tertiary: texture
            0.20 * edge_indicator             # Boundaries
        )

        # Clip to [0, 1]
        confidence_map = np.clip(confidence_map, 0, 1)

        # Smooth confidence map for better blending
        confidence_map = cv2.GaussianBlur(confidence_map, (31, 31), 0)

        # Create binary mask (threshold at 0.5)
        region_mask = (confidence_map > 0.5).astype(np.uint8) * 255

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_CLOSE, kernel)
        region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_OPEN, kernel)

        return region_mask, confidence_map

    def sauvola_threshold(self, image: np.ndarray, window_size: int = 51, k: float = 0.3) -> np.ndarray:
        """Apply Sauvola thresholding."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        gray_f = gray.astype(np.float32)

        # Local statistics
        mean = cv2.blur(gray_f, (window_size, window_size))
        mean_sq = cv2.blur(gray_f ** 2, (window_size, window_size))
        std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))

        # Sauvola threshold
        threshold = mean * (1 + k * ((std / 128.0) - 1))

        # Apply (black text on white background)
        binary = np.where(gray < threshold, 0, 255).astype(np.uint8)

        return binary

    def process(self, image: np.ndarray, return_intermediate: bool = False) -> Tuple[np.ndarray, Optional[dict]]:
        """
        Process image with improved hybrid method.

        Args:
            image: Input BGR image
            return_intermediate: Return intermediate results for debugging

        Returns:
            binary: Final binary output
            intermediates: Dict of intermediate images (if requested)
        """
        intermediates = {} if return_intermediate else None

        # Step 1: Detect challenging regions
        region_mask, confidence_map = self.detect_challenging_regions(image)

        if return_intermediate:
            intermediates['region_mask'] = region_mask
            intermediates['confidence_map'] = (confidence_map * 255).astype(np.uint8)

        # Step 2: Apply MORPHBG
        morphbg_result, _ = self.morphbg.process(image, return_intermediate=False)

        if return_intermediate:
            intermediates['morphbg_result'] = morphbg_result

        # Step 3: Apply Sauvola
        sauvola_result = self.sauvola_threshold(image, window_size=51, k=0.3)

        if return_intermediate:
            intermediates['sauvola_result'] = sauvola_result

        # Step 4: Confidence-weighted blending
        # Use confidence map for smooth transitions

        # Convert to float for blending
        morphbg_f = morphbg_result.astype(np.float32) / 255.0
        sauvola_f = sauvola_result.astype(np.float32) / 255.0

        # Blend: high confidence (challenging) → use Sauvola
        #        low confidence (normal) → use MORPHBG
        blended = (1 - confidence_map) * morphbg_f + confidence_map * sauvola_f

        # Convert back to uint8
        binary = (blended * 255).astype(np.uint8)

        if return_intermediate:
            intermediates['blended_raw'] = binary.copy()

        # Step 5: Post-processing refinement
        # Remove very small noise components
        binary = self._cleanup_noise(binary)

        if return_intermediate:
            intermediates['final'] = binary.copy()

        return binary, intermediates

    def _cleanup_noise(self, binary: np.ndarray, min_size: int = 50) -> np.ndarray:
        """Remove small noise components."""
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            255 - binary,  # Invert: find foreground components
            connectivity=8
        )

        # Create clean output
        cleaned = binary.copy()

        # Remove small components
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < min_size:
                # Remove this component (set to white/background)
                cleaned[labels == label] = 255

        return cleaned

    def process_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        return_intermediate: bool = False
    ) -> Tuple[np.ndarray, Optional[dict]]:
        """Process image file."""
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Failed to read: {input_path}")

        binary, intermediates = self.process(image, return_intermediate)

        if output_path:
            cv2.imwrite(output_path, binary)

        return binary, intermediates


def test_on_problematic_image(image_path: str, output_dir: str = "hybrid_test"):
    """Test improved hybrid on a problematic image."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    print(f"\nTesting on: {image_path}")
    print("="*60)

    # Process with improved hybrid
    method = ImprovedHybridMethod()
    binary, intermediates = method.process_file(
        image_path,
        output_path=str(output_path / "final_hybrid.png"),
        return_intermediate=True
    )

    # Save all intermediate stages
    for name, img in intermediates.items():
        cv2.imwrite(str(output_path / f"{name}.png"), img)

    print(f"✓ Results saved to: {output_dir}/")
    print(f"  - final_hybrid.png (main output)")
    print(f"  - region_mask.png (detected challenging regions)")
    print(f"  - confidence_map.png (blending weights)")
    print(f"  - morphbg_result.png (MORPHBG component)")
    print(f"  - sauvola_result.png (Sauvola component)")
    print(f"  - blended_raw.png (before cleanup)")
    print(f"  - final.png (after cleanup)")
    print()


def main():
    """Test the improved hybrid method."""
    if len(sys.argv) < 2:
        print("Usage: python 12_improved_hybrid_method.py <image_path> [output_dir]")
        print("\nExample:")
        print("  python 12_improved_hybrid_method.py original_images/00000155.png hybrid_test")
        sys.exit(1)

    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "hybrid_test"

    test_on_problematic_image(image_path, output_dir)


if __name__ == "__main__":
    main()
