"""
MorphBG: Morphological Background-Subtraction Binarisation
A colour-aware pipeline for ancient Sanskrit manuscript binarisation

Author: Suyash Kumar Bhagat
Institution: Delhi Technological University
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import time


class MorphBG:
    """
    MorphBG binarisation pipeline for ochre-parchment manuscripts.

    Pipeline stages:
    1. CLAHE-V: Illumination normalisation in HSV space
    2. Per-channel morphological background estimation
    3. Weber-contrast ink signal computation
    4. Ink confidence map generation
    5. Otsu thresholding
    6. CCA cleanup
    """

    def __init__(
        self,
        closing_kernel_size: int = 61,
        clahe_clip_limit: float = 2.0,
        clahe_tile_size: int = 8,
        min_component_size: int = 150,
        max_hole_size: int = 15,
        use_clahe: bool = True,
        use_cca: bool = True
    ):
        """
        Initialize MorphBG pipeline.

        Args:
            closing_kernel_size: Size of morphological closing kernel (must be odd)
            clahe_clip_limit: Contrast limiting threshold for CLAHE
            clahe_tile_size: Tile grid size for CLAHE
            min_component_size: Minimum area (px²) for foreground components
            max_hole_size: Maximum area (px²) for holes to fill
            use_clahe: Whether to apply CLAHE preprocessing
            use_cca: Whether to apply CCA cleanup
        """
        self.closing_kernel_size = closing_kernel_size
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        self.min_component_size = min_component_size
        self.max_hole_size = max_hole_size
        self.use_clahe = use_clahe
        self.use_cca = use_cca

        # Create morphological structuring element
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (closing_kernel_size, closing_kernel_size)
        )

        # Create CLAHE object
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=(clahe_tile_size, clahe_tile_size)
        )

    def process(
        self,
        image: np.ndarray,
        return_intermediate: bool = False
    ) -> Tuple[np.ndarray, Optional[dict]]:
        """
        Apply MorphBG pipeline to an image.

        Args:
            image: Input BGR image (uint8)
            return_intermediate: If True, return intermediate results

        Returns:
            binary_image: Binarised output (0/255)
            intermediates: Dict of intermediate images (if return_intermediate=True)
        """
        start_time = time.time()
        intermediates = {} if return_intermediate else None

        # Stage 1: CLAHE-V illumination normalisation
        if self.use_clahe:
            image = self._apply_clahe(image)
            if return_intermediate:
                intermediates['01_clahe'] = image.copy()

        # Stage 2: Per-channel morphological background estimation
        bg_map = self._estimate_background(image)
        if return_intermediate:
            intermediates['02_background'] = bg_map.copy()

        # Stage 3: Weber-contrast ink signal
        ink_signals = self._compute_ink_signals(image, bg_map)
        if return_intermediate:
            intermediates['03_ink_b'] = ink_signals[:, :, 0]
            intermediates['03_ink_g'] = ink_signals[:, :, 1]
            intermediates['03_ink_r'] = ink_signals[:, :, 2]

        # Stage 4: Ink confidence map
        confidence_map = self._compute_confidence_map(ink_signals)
        if return_intermediate:
            intermediates['04_confidence'] = confidence_map.copy()

        # Stage 5: Otsu thresholding
        binary = self._otsu_threshold(confidence_map)
        if return_intermediate:
            intermediates['05_otsu'] = binary.copy()

        # Stage 6: CCA cleanup
        if self.use_cca:
            binary = self._cca_cleanup(binary)
            if return_intermediate:
                intermediates['06_final'] = binary.copy()

        elapsed = time.time() - start_time
        if return_intermediate:
            intermediates['_elapsed_seconds'] = elapsed

        return binary, intermediates

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE to V channel in HSV space."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = self.clahe.apply(hsv[:, :, 2])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _estimate_background(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate background color using per-channel morphological closing.

        Returns:
            bg_map: Float32 image, same shape as input
        """
        bg_map = np.zeros_like(image, dtype=np.float32)

        for c in range(3):  # B, G, R channels
            channel = image[:, :, c].astype(np.float32)
            bg_map[:, :, c] = cv2.morphologyEx(
                channel,
                cv2.MORPH_CLOSE,
                self.kernel
            )

        return bg_map

    def _compute_ink_signals(
        self,
        image: np.ndarray,
        bg_map: np.ndarray
    ) -> np.ndarray:
        """
        Compute Weber-contrast ink signal for each channel.

        ink[c] = max(bg[c] - px[c], 0) / (bg[c] + 1)

        Returns:
            ink_signals: Float32 (H, W, 3) in range [0, 1]
        """
        image_f = image.astype(np.float32)

        # Compute contrast: how much darker is pixel than background
        contrast = np.maximum(bg_map - image_f, 0)

        # Weber normalisation: divide by local background brightness
        ink_signals = contrast / (bg_map + 1.0)

        return ink_signals

    def _compute_confidence_map(self, ink_signals: np.ndarray) -> np.ndarray:
        """
        Average the three channel ink signals into a scalar confidence map.

        Returns:
            confidence_map: Uint8 image in range [0, 255]
        """
        # Average across channels
        confidence = np.mean(ink_signals, axis=2)

        # Scale to [0, 255]
        confidence = np.clip(confidence * 255, 0, 255).astype(np.uint8)

        return confidence

    def _otsu_threshold(self, confidence_map: np.ndarray) -> np.ndarray:
        """
        Apply Otsu's method to the confidence map.

        Returns:
            binary: Uint8 image with values {0, 255}
        """
        _, binary = cv2.threshold(
            confidence_map,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        return binary

    def _cca_cleanup(self, binary: np.ndarray) -> np.ndarray:
        """
        Two-pass CCA cleanup:
        1. Remove small foreground noise components
        2. Fill small holes in foreground strokes

        Returns:
            cleaned: Uint8 binary image
        """
        # Pass 1: Remove small foreground components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )

        cleaned = np.zeros_like(binary)
        for label in range(1, num_labels):  # Skip background (0)
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= self.min_component_size:
                cleaned[labels == label] = 255

        # Pass 2: Fill small holes
        inverted = 255 - cleaned
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            inverted,
            connectivity=8
        )

        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area <= self.max_hole_size:
                cleaned[labels == label] = 255

        return cleaned

    def process_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        return_intermediate: bool = False
    ) -> Tuple[np.ndarray, Optional[dict]]:
        """
        Process an image file.

        Args:
            input_path: Path to input color image
            output_path: Path to save binary output (optional)
            return_intermediate: Whether to return intermediate results

        Returns:
            binary: Binarised image
            intermediates: Dict of intermediate results (if requested)
        """
        # Read image
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Failed to read image: {input_path}")

        # Process
        binary, intermediates = self.process(image, return_intermediate)

        # Save if output path provided
        if output_path is not None:
            cv2.imwrite(output_path, binary)

        return binary, intermediates


def main():
    """Example usage."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python 01_morphbg.py <input_image> [output_image]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output_binary.png"

    # Initialize pipeline
    pipeline = MorphBG()

    # Process
    print(f"Processing: {input_path}")
    binary, intermediates = pipeline.process_file(
        input_path,
        output_path,
        return_intermediate=True
    )

    print(f"Saved binary image to: {output_path}")
    print(f"Processing time: {intermediates['_elapsed_seconds']:.2f} seconds")


if __name__ == "__main__":
    main()
