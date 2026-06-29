"""
Parse Label Studio JSON annotations and prepare data for OCR training/evaluation.

Author: Suyash Kumar Bhagat
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
import argparse


class LabelStudioParser:
    """Parse Label Studio annotations for OCR."""

    def __init__(
        self,
        annotation_file: str,
        images_dir: str,
        output_dir: str = "parsed_annotations"
    ):
        """
        Initialize parser.

        Args:
            annotation_file: Path to Label Studio JSON export
            images_dir: Directory containing the annotated images
            output_dir: Directory to save parsed data
        """
        self.annotation_file = Path(annotation_file)
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Load annotations
        with open(self.annotation_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)

        print(f"Loaded {len(self.annotations)} annotated images")

    def parse_all(self) -> Dict:
        """
        Parse all annotations and extract line images + transcriptions.

        Returns:
            Dictionary with parsed data
        """
        all_lines = []
        character_counts = Counter()

        # Create line images directory
        line_images_dir = self.output_dir / "line_images"
        line_images_dir.mkdir(exist_ok=True)

        for idx, item in enumerate(self.annotations):
            # Parse this image's annotations
            image_data = self.parse_single_image(item, line_images_dir, idx)

            all_lines.extend(image_data['lines'])

            # Count characters
            for line in image_data['lines']:
                character_counts.update(line['text'])

        # Extract character inventory
        character_inventory = sorted(character_counts.keys())

        # Save parsed data
        output_data = {
            'num_images': len(self.annotations),
            'num_lines': len(all_lines),
            'num_characters': sum(character_counts.values()),
            'unique_characters': len(character_inventory),
            'character_inventory': character_inventory,
            'character_counts': dict(character_counts),
            'lines': all_lines
        }

        # Save as JSON
        output_json = self.output_dir / "parsed_annotations.json"
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        # Save character inventory
        char_inv_path = self.output_dir / "character_inventory.txt"
        with open(char_inv_path, 'w', encoding='utf-8') as f:
            for char in character_inventory:
                f.write(char + '\n')

        # Save ground truth file (for Tesseract training format)
        gt_file = self.output_dir / "ground_truth.txt"
        with open(gt_file, 'w', encoding='utf-8') as f:
            for line in all_lines:
                f.write(f"{line['line_image_path']}\t{line['text']}\n")

        print(f"\n{'='*60}")
        print("Parsing Complete")
        print(f"{'='*60}")
        print(f"  Total images:     {output_data['num_images']}")
        print(f"  Total lines:      {output_data['num_lines']}")
        print(f"  Total characters: {output_data['num_characters']}")
        print(f"  Unique glyphs:    {output_data['unique_characters']}")
        print(f"  Output dir:       {self.output_dir}")
        print(f"{'='*60}\n")

        return output_data

    def parse_single_image(
        self,
        annotation_item: Dict,
        line_images_dir: Path,
        image_idx: int
    ) -> Dict:
        """
        Parse annotations for a single image.

        Args:
            annotation_item: Label Studio annotation object
            line_images_dir: Directory to save cropped line images
            image_idx: Index of this image

        Returns:
            Dictionary with parsed data for this image
        """
        # Get image file name
        if 'data' in annotation_item and 'image' in annotation_item['data']:
            image_filename = Path(annotation_item['data']['image']).name
        elif 'file_upload' in annotation_item:
            image_filename = Path(annotation_item['file_upload']).name
        else:
            print(f"Warning: Cannot find image filename in annotation item {image_idx}")
            return {'lines': []}

        image_path = self.images_dir / image_filename

        # Load image
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            return {'lines': []}

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Failed to read image: {image_path}")
            return {'lines': []}

        h, w = image.shape[:2]

        # Parse annotations
        lines_data = []

        if 'annotations' not in annotation_item or not annotation_item['annotations']:
            return {'lines': []}

        # Get first annotation (assuming single annotator)
        annotation = annotation_item['annotations'][0]

        if 'result' not in annotation:
            return {'lines': []}

        for result_idx, result in enumerate(annotation['result']):
            # Extract bounding box (as percentages)
            if result['type'] != 'rectanglelabels':
                continue

            value = result['value']

            # Convert percentage coordinates to pixels
            x_pct = value['x']
            y_pct = value['y']
            w_pct = value['width']
            h_pct = value['height']

            x1 = int(x_pct * w / 100)
            y1 = int(y_pct * h / 100)
            x2 = int((x_pct + w_pct) * w / 100)
            y2 = int((y_pct + h_pct) * h / 100)

            # Ensure within bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            # Extract transcription
            transcription = value.get('text', [''])[0] if 'text' in value else ''

            if not transcription:
                continue

            # Crop line image
            line_image = image[y1:y2, x1:x2]

            # Save line image
            line_filename = f"img{image_idx:04d}_line{result_idx:04d}.png"
            line_path = line_images_dir / line_filename
            cv2.imwrite(str(line_path), line_image)

            # Add to dataset
            lines_data.append({
                'image_filename': image_filename,
                'line_image_path': str(line_path),
                'line_filename': line_filename,
                'bbox': [x1, y1, x2, y2],
                'text': transcription
            })

        return {
            'image_filename': image_filename,
            'lines': lines_data
        }


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Parse Label Studio annotations for OCR"
    )
    parser.add_argument(
        '--annotation-file',
        type=str,
        required=True,
        help='Path to Label Studio JSON export'
    )
    parser.add_argument(
        '--images-dir',
        type=str,
        default='original_images',
        help='Directory containing annotated images'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='parsed_annotations',
        help='Directory to save parsed data'
    )

    args = parser.parse_args()

    # Parse annotations
    parser_obj = LabelStudioParser(
        annotation_file=args.annotation_file,
        images_dir=args.images_dir,
        output_dir=args.output_dir
    )

    parser_obj.parse_all()


if __name__ == "__main__":
    main()
