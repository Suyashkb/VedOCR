"""
Batch processing script for MorphBG pipeline.
Processes all images in original_images/ folder.

Author: Suyash Kumar Bhagat
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
from typing import List, Dict
import argparse

# Import MorphBG from the pipeline module
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Import after adding current directory to path
import importlib.util
spec = importlib.util.spec_from_file_location("morphbg_pipeline", Path(__file__).parent / "01_morphbg.py")
morphbg_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(morphbg_module)
MorphBG = morphbg_module.MorphBG


class BatchProcessor:
    """Batch process images with MorphBG pipeline."""

    def __init__(
        self,
        input_dir: str = "original_images",
        output_dir: str = "processed_images",
        intermediate_dir: str = "intermediate_outputs",
        save_intermediates: bool = False,
        **morphbg_kwargs
    ):
        """
        Initialize batch processor.

        Args:
            input_dir: Directory containing input images
            output_dir: Directory for binary outputs
            intermediate_dir: Directory for intermediate visualizations
            save_intermediates: Whether to save intermediate pipeline stages
            **morphbg_kwargs: Arguments passed to MorphBG constructor
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.intermediate_dir = Path(intermediate_dir)
        self.save_intermediates = save_intermediates

        # Create output directories
        self.output_dir.mkdir(exist_ok=True, parents=True)
        if save_intermediates:
            self.intermediate_dir.mkdir(exist_ok=True, parents=True)

        # Initialize MorphBG pipeline
        self.pipeline = MorphBG(**morphbg_kwargs)

        # Processing log
        self.log = {
            'timestamp': datetime.now().isoformat(),
            'input_dir': str(self.input_dir),
            'output_dir': str(self.output_dir),
            'morphbg_config': {
                'closing_kernel_size': self.pipeline.closing_kernel_size,
                'clahe_clip_limit': self.pipeline.clahe_clip_limit,
                'clahe_tile_size': self.pipeline.clahe_tile_size,
                'min_component_size': self.pipeline.min_component_size,
                'max_hole_size': self.pipeline.max_hole_size,
                'use_clahe': self.pipeline.use_clahe,
                'use_cca': self.pipeline.use_cca,
            },
            'files_processed': []
        }

    def get_image_files(self) -> List[Path]:
        """Get all image files from input directory."""
        extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
        files = []

        for ext in extensions:
            files.extend(self.input_dir.glob(f'*{ext}'))
            files.extend(self.input_dir.glob(f'*{ext.upper()}'))

        return sorted(files)

    def process_single(self, input_path: Path) -> Dict:
        """
        Process a single image.

        Returns:
            Dictionary with processing info
        """
        try:
            # Read image
            image = cv2.imread(str(input_path))
            if image is None:
                raise ValueError(f"Failed to read: {input_path}")

            # Get output paths
            stem = input_path.stem
            output_path = self.output_dir / f"{stem}_binary.png"

            # Process
            binary, intermediates = self.pipeline.process(
                image,
                return_intermediate=self.save_intermediates
            )

            # Save binary output
            cv2.imwrite(str(output_path), binary)

            # Save intermediates if requested
            if self.save_intermediates and intermediates is not None:
                inter_dir = self.intermediate_dir / stem
                inter_dir.mkdir(exist_ok=True, parents=True)

                for stage_name, stage_img in intermediates.items():
                    if stage_name.startswith('_'):  # Skip metadata
                        continue

                    # Handle both grayscale and color intermediate images
                    if len(stage_img.shape) == 2:
                        stage_img_save = stage_img
                    else:
                        stage_img_save = stage_img

                    stage_path = inter_dir / f"{stage_name}.png"
                    cv2.imwrite(str(stage_path), stage_img_save)

            # Return processing info
            result = {
                'input_file': str(input_path),
                'output_file': str(output_path),
                'input_shape': image.shape,
                'success': True,
                'elapsed_seconds': intermediates.get('_elapsed_seconds', 0) if intermediates else 0
            }

        except Exception as e:
            result = {
                'input_file': str(input_path),
                'success': False,
                'error': str(e)
            }

        return result

    def process_all(self) -> Dict:
        """
        Process all images in input directory.

        Returns:
            Processing log dictionary
        """
        image_files = self.get_image_files()

        if not image_files:
            print(f"No images found in {self.input_dir}")
            return self.log

        print(f"Found {len(image_files)} images to process")
        print(f"Output directory: {self.output_dir}")

        # Process each image
        for img_path in tqdm(image_files, desc="Processing images"):
            result = self.process_single(img_path)
            self.log['files_processed'].append(result)

        # Summary statistics
        successful = sum(1 for r in self.log['files_processed'] if r['success'])
        failed = len(self.log['files_processed']) - successful
        total_time = sum(
            r.get('elapsed_seconds', 0)
            for r in self.log['files_processed']
            if r['success']
        )

        self.log['summary'] = {
            'total_files': len(image_files),
            'successful': successful,
            'failed': failed,
            'total_processing_time_seconds': total_time,
            'average_time_per_image_seconds': total_time / successful if successful > 0 else 0
        }

        # Save log
        log_path = self.output_dir / 'processing_log.json'
        with open(log_path, 'w') as f:
            json.dump(self.log, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Processing complete!")
        print(f"  Successful: {successful}/{len(image_files)}")
        print(f"  Failed: {failed}/{len(image_files)}")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Average time: {total_time/successful:.2f}s per image" if successful > 0 else "")
        print(f"  Log saved to: {log_path}")
        print(f"{'='*60}\n")

        return self.log


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Batch process manuscript images with MorphBG pipeline"
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='original_images',
        help='Input directory containing manuscript images'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='processed_images',
        help='Output directory for binary images'
    )
    parser.add_argument(
        '--save-intermediates',
        action='store_true',
        help='Save intermediate pipeline stages for visualization'
    )
    parser.add_argument(
        '--kernel-size',
        type=int,
        default=61,
        help='Morphological closing kernel size (default: 61)'
    )
    parser.add_argument(
        '--no-clahe',
        action='store_true',
        help='Disable CLAHE preprocessing'
    )
    parser.add_argument(
        '--no-cca',
        action='store_true',
        help='Disable CCA cleanup'
    )

    args = parser.parse_args()

    # Initialize processor
    processor = BatchProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        save_intermediates=args.save_intermediates,
        closing_kernel_size=args.kernel_size,
        use_clahe=not args.no_clahe,
        use_cca=not args.no_cca
    )

    # Process all images
    processor.process_all()


if __name__ == "__main__":
    main()
