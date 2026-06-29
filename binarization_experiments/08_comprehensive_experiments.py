"""
Comprehensive Experimental Pipeline for Sanskrit Manuscript Binarization
=========================================================================

This script runs a complete experimental comparison of multiple binarization methods:
1. Baseline MORPHBG (default settings)
2. MORPHBG variants (different kernel sizes)
3. Sauvola adaptive thresholding
4. Niblack adaptive thresholding
5. Wolf-Jolion adaptive thresholding
6. Otsu global thresholding
7. Hybrid methods for tape-affected regions

Special focus on problematic images 150-160 with tape impressions.

Author: Suyash Kumar Bhagat
Date: April 18, 2026
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import time
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Import MorphBG
import importlib.util
spec = importlib.util.spec_from_file_location("morphbg_pipeline", Path(__file__).parent / "01_morphbg_pipeline.py")
morphbg_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(morphbg_module)
MorphBG = morphbg_module.MorphBG


class BinarizationMethod:
    """Base class for binarization methods."""

    def __init__(self, name: str):
        self.name = name

    def process(self, image: np.ndarray) -> np.ndarray:
        """Process image and return binary result."""
        raise NotImplementedError


class MORPHBGMethod(BinarizationMethod):
    """MORPHBG method wrapper."""

    def __init__(self, kernel_size: int = 61, name: str = None):
        super().__init__(name or f"MORPHBG_k{kernel_size}")
        self.pipeline = MorphBG(closing_kernel_size=kernel_size)
        self.kernel_size = kernel_size

    def process(self, image: np.ndarray) -> np.ndarray:
        binary, _ = self.pipeline.process(image, return_intermediate=False)
        return binary


class SauvolaMethod(BinarizationMethod):
    """Sauvola adaptive thresholding."""

    def __init__(self, window_size: int = 25, k: float = 0.2):
        super().__init__(f"Sauvola_w{window_size}_k{k}")
        self.window_size = window_size
        self.k = k

    def process(self, image: np.ndarray) -> np.ndarray:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Sauvola thresholding
        # T(x,y) = m(x,y) * (1 + k * ((s(x,y) / R) - 1))
        # where m is local mean, s is local std dev, R is max std dev (128 for uint8)

        # Compute local mean and std
        mean = cv2.blur(gray.astype(np.float32), (self.window_size, self.window_size))
        mean_sq = cv2.blur((gray.astype(np.float32) ** 2), (self.window_size, self.window_size))
        std = np.sqrt(mean_sq - mean ** 2)

        # Sauvola threshold
        threshold = mean * (1 + self.k * ((std / 128.0) - 1))

        # Apply threshold
        binary = np.where(gray < threshold, 255, 0).astype(np.uint8)

        return binary


class NiblackMethod(BinarizationMethod):
    """Niblack adaptive thresholding."""

    def __init__(self, window_size: int = 25, k: float = -0.2):
        super().__init__(f"Niblack_w{window_size}_k{k}")
        self.window_size = window_size
        self.k = k

    def process(self, image: np.ndarray) -> np.ndarray:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Niblack thresholding: T(x,y) = m(x,y) + k * s(x,y)
        mean = cv2.blur(gray.astype(np.float32), (self.window_size, self.window_size))
        mean_sq = cv2.blur((gray.astype(np.float32) ** 2), (self.window_size, self.window_size))
        std = np.sqrt(mean_sq - mean ** 2)

        threshold = mean + self.k * std
        binary = np.where(gray < threshold, 255, 0).astype(np.uint8)

        return binary


class WolfMethod(BinarizationMethod):
    """Wolf-Jolion adaptive thresholding."""

    def __init__(self, window_size: int = 25, k: float = 0.5):
        super().__init__(f"Wolf_w{window_size}_k{k}")
        self.window_size = window_size
        self.k = k

    def process(self, image: np.ndarray) -> np.ndarray:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Wolf-Jolion: T(x,y) = (1-k)*m(x,y) + k*min + k*(s(x,y)/max_s)*(m(x,y)-min)
        mean = cv2.blur(gray.astype(np.float32), (self.window_size, self.window_size))
        mean_sq = cv2.blur((gray.astype(np.float32) ** 2), (self.window_size, self.window_size))
        std = np.sqrt(mean_sq - mean ** 2)

        min_gray = np.min(gray)
        max_std = np.max(std)

        if max_std > 0:
            threshold = (1 - self.k) * mean + self.k * min_gray + \
                       self.k * (std / max_std) * (mean - min_gray)
        else:
            threshold = mean

        binary = np.where(gray < threshold, 255, 0).astype(np.uint8)

        return binary


class OtsuMethod(BinarizationMethod):
    """Otsu global thresholding."""

    def __init__(self):
        super().__init__("Otsu")

    def process(self, image: np.ndarray) -> np.ndarray:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply Otsu thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        return binary


class HybridMethod(BinarizationMethod):
    """
    Hybrid method that detects tape-affected regions and uses
    different strategies for those regions.
    """

    def __init__(self):
        super().__init__("Hybrid_TapeAware")
        self.morphbg = MorphBG(closing_kernel_size=61)

    def detect_tape_regions(self, image: np.ndarray) -> np.ndarray:
        """
        Detect regions with tape impressions using color analysis.
        Tape regions typically have different color characteristics.
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Tape regions often have different brightness and color
        # Use combination of low variance and specific color range

        # Compute local variance in L channel
        blur_size = 31
        l_float = l.astype(np.float32)
        l_mean = cv2.blur(l_float, (blur_size, blur_size))
        l_sq_mean = cv2.blur(l_float ** 2, (blur_size, blur_size))
        l_var = l_sq_mean - l_mean ** 2

        # High brightness with low variance suggests tape
        tape_mask = ((l > 200) & (l_var < 100)).astype(np.uint8) * 255

        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        tape_mask = cv2.morphologyEx(tape_mask, cv2.MORPH_CLOSE, kernel)
        tape_mask = cv2.morphologyEx(tape_mask, cv2.MORPH_OPEN, kernel)

        return tape_mask

    def process(self, image: np.ndarray) -> np.ndarray:
        # Detect tape regions
        tape_mask = self.detect_tape_regions(image)

        # Apply MORPHBG to entire image
        morphbg_result, _ = self.morphbg.process(image, return_intermediate=False)

        # Apply Sauvola to entire image (better for tape regions)
        sauvola = SauvolaMethod(window_size=51, k=0.3)
        sauvola_result = sauvola.process(image)

        # Combine: use Sauvola in tape regions, MORPHBG elsewhere
        binary = np.where(tape_mask > 0, sauvola_result, morphbg_result)

        return binary


class ExperimentalPipeline:
    """Main experimental pipeline."""

    def __init__(
        self,
        input_dir: str = "original_images",
        output_dir: str = "experimental_results",
        problematic_range: Tuple[int, int] = (150, 160)
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.problematic_range = problematic_range

        # Create output directory structure
        self.output_dir.mkdir(exist_ok=True, parents=True)
        (self.output_dir / "binarized").mkdir(exist_ok=True)
        (self.output_dir / "comparisons").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "problematic_analysis").mkdir(exist_ok=True)

        # Initialize all methods to test
        self.methods = [
            MORPHBGMethod(kernel_size=61, name="MORPHBG_default"),
            MORPHBGMethod(kernel_size=41, name="MORPHBG_k41"),
            MORPHBGMethod(kernel_size=81, name="MORPHBG_k81"),
            SauvolaMethod(window_size=25, k=0.2),
            SauvolaMethod(window_size=51, k=0.3),
            NiblackMethod(window_size=25, k=-0.2),
            WolfMethod(window_size=25, k=0.5),
            OtsuMethod(),
            HybridMethod()
        ]

        # Results storage
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'methods': [m.name for m in self.methods],
            'images_processed': [],
            'processing_times': defaultdict(list),
            'problematic_images': []
        }

    def get_image_files(self) -> List[Path]:
        """Get all image files."""
        extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
        files = []
        for ext in extensions:
            files.extend(self.input_dir.glob(f'*{ext}'))
            files.extend(self.input_dir.glob(f'*{ext.upper()}'))
        return sorted(files)

    def is_problematic_image(self, image_path: Path) -> bool:
        """Check if image is in problematic range."""
        try:
            # Extract image number from filename (e.g., 00000150.png -> 150)
            num = int(''.join(filter(str.isdigit, image_path.stem)))
            return self.problematic_range[0] <= num <= self.problematic_range[1]
        except:
            return False

    def is_already_processed(self, image_path: Path) -> bool:
        """Check if image has already been processed (by checking first method output)."""
        first_method = self.methods[0]
        output_path = self.output_dir / "binarized" / first_method.name / image_path.name
        return output_path.exists()

    def process_single_image(
        self,
        image_path: Path,
        save_all: bool = False,
        skip_existing: bool = False
    ) -> Dict:
        """Process a single image with all methods."""
        # Check if already processed (resume support)
        if skip_existing and self.is_already_processed(image_path):
            return {
                'image_name': image_path.name,
                'image_path': str(image_path),
                'skipped': True,
                'reason': 'Already processed (resume mode)'
            }

        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            return {'error': f'Failed to read {image_path}'}

        is_problematic = self.is_problematic_image(image_path)

        results = {
            'image_name': image_path.name,
            'image_path': str(image_path),
            'is_problematic': is_problematic,
            'methods': {}
        }

        # Process with each method
        for method in self.methods:
            start_time = time.time()
            try:
                binary = method.process(image)
                elapsed = time.time() - start_time

                # Save result
                if save_all or is_problematic:
                    method_dir = self.output_dir / "binarized" / method.name
                    method_dir.mkdir(exist_ok=True, parents=True)
                    output_path = method_dir / image_path.name
                    cv2.imwrite(str(output_path), binary)

                results['methods'][method.name] = {
                    'success': True,
                    'processing_time': elapsed,
                    'output_path': str(output_path) if (save_all or is_problematic) else None
                }

                self.results['processing_times'][method.name].append(elapsed)

            except Exception as e:
                results['methods'][method.name] = {
                    'success': False,
                    'error': str(e)
                }

        return results

    def create_comparison_image(
        self,
        image_path: Path,
        methods_subset: List[str] = None
    ) -> np.ndarray:
        """Create side-by-side comparison of methods for one image."""
        image = cv2.imread(str(image_path))
        if image is None:
            return None

        if methods_subset is None:
            methods_subset = [m.name for m in self.methods]

        # Create comparison
        comparisons = [image]
        titles = ['Original']

        for method in self.methods:
            if method.name in methods_subset:
                binary = method.process(image)
                # Convert to BGR for consistent display
                binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                comparisons.append(binary_bgr)
                titles.append(method.name)

        # Stack images
        # Calculate grid size
        n_images = len(comparisons)
        cols = min(3, n_images)
        rows = (n_images + cols - 1) // cols

        # Resize all to same size
        target_width = 800
        target_height = int(target_width * image.shape[0] / image.shape[1])

        resized = []
        for img in comparisons:
            resized.append(cv2.resize(img, (target_width, target_height)))

        # Create grid
        grid_rows = []
        for i in range(rows):
            row_images = resized[i*cols:(i+1)*cols]
            # Pad if needed
            while len(row_images) < cols:
                row_images.append(np.ones_like(resized[0]) * 255)
            grid_rows.append(np.hstack(row_images))

        comparison_image = np.vstack(grid_rows)

        return comparison_image

    def analyze_problematic_images(self):
        """Detailed analysis of problematic images."""
        print("\n" + "="*80)
        print("Analyzing Problematic Images (Tape Impression Analysis)")
        print("="*80 + "\n")

        image_files = self.get_image_files()
        problematic_images = [f for f in image_files if self.is_problematic_image(f)]

        if not problematic_images:
            print("No problematic images found in range")
            return

        print(f"Found {len(problematic_images)} problematic images")

        # Create detailed comparisons for problematic images
        for img_path in tqdm(problematic_images, desc="Analyzing problematic images"):
            # Create full comparison
            comparison = self.create_comparison_image(img_path)
            if comparison is not None:
                output_path = self.output_dir / "problematic_analysis" / f"comparison_{img_path.stem}.png"
                cv2.imwrite(str(output_path), comparison)

            # Store info
            self.results['problematic_images'].append({
                'image_name': img_path.name,
                'image_path': str(img_path),
                'comparison_path': str(output_path)
            })

    def run_full_experiment(self, save_all_outputs: bool = False, resume: bool = False):
        """Run complete experimental pipeline."""
        print("\n" + "="*80)
        print("COMPREHENSIVE BINARIZATION EXPERIMENTAL PIPELINE")
        print("="*80 + "\n")

        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Problematic range: images {self.problematic_range[0]}-{self.problematic_range[1]}")
        print(f"Resume mode: {'ENABLED (skipping already processed)' if resume else 'DISABLED (processing all)'}")
        print(f"Testing {len(self.methods)} methods:")
        for method in self.methods:
            print(f"  - {method.name}")
        print()

        # Get images
        image_files = self.get_image_files()
        print(f"Found {len(image_files)} images to process")

        # Check how many already processed if resuming
        if resume:
            already_processed = sum(1 for img in image_files if self.is_already_processed(img))
            print(f"Already processed: {already_processed} images")
            print(f"Remaining: {len(image_files) - already_processed} images")
        print()

        # Process all images
        skipped_count = 0
        for img_path in tqdm(image_files, desc="Processing all images"):
            result = self.process_single_image(img_path, save_all=save_all_outputs, skip_existing=resume)
            if result.get('skipped', False):
                skipped_count += 1
            self.results['images_processed'].append(result)

        if resume and skipped_count > 0:
            print(f"\n✓ Skipped {skipped_count} already-processed images")

        # Analyze problematic images
        self.analyze_problematic_images()

        # Generate visualizations
        self.generate_visualizations()

        # Save results
        self.save_results()

        # Print summary
        self.print_summary()

    def generate_visualizations(self):
        """Generate all visualization plots."""
        print("\nGenerating visualizations...")

        viz_dir = self.output_dir / "visualizations"

        # 1. Processing time comparison
        self.plot_processing_times(viz_dir / "processing_times.png")

        # 2. Method comparison statistics
        self.plot_method_statistics(viz_dir / "method_comparison.png")

        # 3. Problematic vs normal images analysis
        self.plot_problematic_analysis(viz_dir / "problematic_analysis.png")

    def plot_processing_times(self, output_path: Path):
        """Plot processing time comparison."""
        plt.figure(figsize=(12, 6))

        method_names = []
        times = []

        for method in self.methods:
            if method.name in self.results['processing_times']:
                method_times = self.results['processing_times'][method.name]
                if method_times:
                    method_names.append(method.name)
                    times.append(method_times)

        if not times:
            plt.close()
            return

        plt.boxplot(times, labels=method_names)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Processing Time (seconds)')
        plt.title('Processing Time Comparison Across Methods')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_method_statistics(self, output_path: Path):
        """Plot method success rates and statistics."""
        plt.figure(figsize=(12, 6))

        method_names = []
        success_counts = []

        for method in self.methods:
            successes = sum(
                1 for img in self.results['images_processed']
                if img.get('methods', {}).get(method.name, {}).get('success', False)
            )
            method_names.append(method.name)
            success_counts.append(successes)

        plt.bar(method_names, success_counts)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Number of Successful Processings')
        plt.title('Method Success Rate')
        plt.axhline(y=len(self.results['images_processed']), color='r', linestyle='--', label='Total Images')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_problematic_analysis(self, output_path: Path):
        """Plot analysis specific to problematic images."""
        plt.figure(figsize=(14, 8))

        # Compute average processing times for problematic vs normal
        prob_times = defaultdict(list)
        normal_times = defaultdict(list)

        for img_result in self.results['images_processed']:
            is_prob = img_result.get('is_problematic', False)
            for method_name, method_result in img_result.get('methods', {}).items():
                if method_result.get('success', False):
                    time_val = method_result.get('processing_time', 0)
                    if is_prob:
                        prob_times[method_name].append(time_val)
                    else:
                        normal_times[method_name].append(time_val)

        # Plot comparison
        method_names = list(prob_times.keys())
        x = np.arange(len(method_names))
        width = 0.35

        prob_means = [np.mean(prob_times[m]) if prob_times[m] else 0 for m in method_names]
        normal_means = [np.mean(normal_times[m]) if normal_times[m] else 0 for m in method_names]

        plt.bar(x - width/2, normal_means, width, label='Normal Images', alpha=0.8)
        plt.bar(x + width/2, prob_means, width, label='Problematic Images (Tape)', alpha=0.8)

        plt.xlabel('Method')
        plt.ylabel('Average Processing Time (seconds)')
        plt.title('Processing Time: Normal vs Problematic Images')
        plt.xticks(x, method_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def save_results(self):
        """Save all results to JSON."""
        # Compute aggregate statistics
        self.results['aggregate'] = {
            'total_images': len(self.results['images_processed']),
            'problematic_images': len(self.results['problematic_images']),
            'methods_tested': len(self.methods),
            'average_processing_times': {
                method.name: np.mean(self.results['processing_times'][method.name])
                if self.results['processing_times'][method.name] else 0
                for method in self.methods
            }
        }

        # Save to JSON
        results_path = self.output_dir / 'experimental_results.json'
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\nResults saved to: {results_path}")

    def print_summary(self):
        """Print experimental summary."""
        print("\n" + "="*80)
        print("EXPERIMENTAL SUMMARY")
        print("="*80 + "\n")

        print(f"Total images processed: {len(self.results['images_processed'])}")
        print(f"Problematic images: {len(self.results['problematic_images'])}")
        print(f"Methods tested: {len(self.methods)}\n")

        print("Average Processing Times:")
        for method in self.methods:
            if self.results['processing_times'][method.name]:
                avg_time = np.mean(self.results['processing_times'][method.name])
                print(f"  {method.name:30s}: {avg_time:.4f} seconds")

        print("\n" + "="*80)
        print("Output Structure:")
        print("="*80)
        print(f"  {self.output_dir}/")
        print(f"    ├── binarized/               (Binary outputs for each method)")
        print(f"    ├── comparisons/             (Side-by-side comparisons)")
        print(f"    ├── visualizations/          (Graphs and plots)")
        print(f"    ├── problematic_analysis/    (Detailed tape impression analysis)")
        print(f"    └── experimental_results.json (Complete results)")
        print("="*80 + "\n")


def main():
    """Run experimental pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Comprehensive binarization experimental pipeline"
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='original_images',
        help='Input directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='experimental_results',
        help='Output directory'
    )
    parser.add_argument(
        '--save-all',
        action='store_true',
        help='Save all binary outputs (not just problematic images)'
    )
    parser.add_argument(
        '--prob-start',
        type=int,
        default=150,
        help='Start of problematic image range'
    )
    parser.add_argument(
        '--prob-end',
        type=int,
        default=160,
        help='End of problematic image range'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from previous run (skip already processed images)'
    )

    args = parser.parse_args()

    # Create and run pipeline
    pipeline = ExperimentalPipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        problematic_range=(args.prob_start, args.prob_end)
    )

    pipeline.run_full_experiment(save_all_outputs=args.save_all, resume=args.resume)


if __name__ == "__main__":
    main()
