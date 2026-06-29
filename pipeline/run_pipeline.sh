#!/bin/bash

# VedOCR Complete Pipeline Runner
# Author: Suyash Kumar Bhagat

set -e  # Exit on error

echo "=================================================="
echo "VedOCR Pipeline - Complete Processing Workflow"
echo "=================================================="
echo ""

# Check if original_images directory exists and has images
if [ ! -d "original_images" ] || [ -z "$(ls -A original_images 2>/dev/null)" ]; then
    echo "ERROR: original_images/ directory is empty or doesn't exist"
    echo "Please add your manuscript images to original_images/ before running"
    exit 1
fi

# Count images
NUM_IMAGES=$(find original_images -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.tif" \) | wc -l)
echo "Found $NUM_IMAGES images in original_images/"
echo ""

# Step 1: Batch Processing with MorphBG
echo "Step 1: Running MorphBG binarisation..."
echo "----------------------------------------"
python 02_batch_process.py \
    --input-dir original_images \
    --output-dir processed_images \
    --save-intermediates

if [ $? -eq 0 ]; then
    echo "✓ Binarisation complete"
else
    echo "✗ Binarisation failed"
    exit 1
fi
echo ""

# Step 2: Benchmarking (optional, only if ground truth exists)
if [ -d "benchmark_folder" ] && [ ! -z "$(ls -A benchmark_folder 2>/dev/null)" ]; then
    echo "Step 2: Running benchmark evaluation..."
    echo "----------------------------------------"
    python 03_benchmark.py \
        --processed-dir processed_images \
        --gt-dir benchmark_folder \
        --results-dir benchmark_results

    if [ $? -eq 0 ]; then
        echo "✓ Benchmarking complete"
    else
        echo "✗ Benchmarking failed (continuing anyway)"
    fi
else
    echo "Step 2: Skipping benchmark (no ground truth in benchmark_folder/)"
fi
echo ""

# Step 3: Parse annotations (only if Label Studio JSON exists)
if [ -f "annotations/annotations.json" ]; then
    echo "Step 3: Parsing Label Studio annotations..."
    echo "----------------------------------------"
    python 04_parse_annotations.py \
        --annotation-file annotations/annotations.json \
        --images-dir original_images \
        --output-dir parsed_annotations

    if [ $? -eq 0 ]; then
        echo "✓ Annotation parsing complete"
    else
        echo "✗ Annotation parsing failed (continuing anyway)"
    fi
else
    echo "Step 3: Skipping annotation parsing (no annotations/annotations.json found)"
    echo "    Export your Label Studio annotations to annotations/annotations.json to enable OCR evaluation"
fi
echo ""

# Step 4: Zero-shot OCR evaluation (only if annotations parsed)
if [ -f "parsed_annotations/parsed_annotations.json" ]; then
    echo "Step 4: Running zero-shot OCR evaluation..."
    echo "----------------------------------------"
    python 05_zero_shot_ocr.py \
        --annotations parsed_annotations/parsed_annotations.json \
        --results-dir results/zero_shot_ocr

    if [ $? -eq 0 ]; then
        echo "✓ Zero-shot OCR evaluation complete"
    else
        echo "✗ Zero-shot OCR evaluation failed"
    fi
else
    echo "Step 4: Skipping OCR evaluation (no parsed annotations)"
fi
echo ""

# Summary
echo "=================================================="
echo "Pipeline Execution Summary"
echo "=================================================="
echo ""
echo "Completed steps:"
echo "  ✓ MorphBG binarisation"
if [ -d "benchmark_results" ]; then
    echo "  ✓ Benchmark evaluation"
fi
if [ -d "parsed_annotations" ]; then
    echo "  ✓ Annotation parsing"
fi
if [ -d "results/zero_shot_ocr" ]; then
    echo "  ✓ Zero-shot OCR evaluation"
fi
echo ""
echo "Output directories:"
echo "  - processed_images/      : Binarised images"
if [ -d "benchmark_results" ]; then
    echo "  - benchmark_results/     : Benchmark metrics"
fi
if [ -d "parsed_annotations" ]; then
    echo "  - parsed_annotations/    : Parsed annotation data"
fi
if [ -d "results/zero_shot_ocr" ]; then
    echo "  - results/zero_shot_ocr/ : OCR evaluation results"
fi
echo ""
echo "Next steps:"
echo "  1. Review results in the output directories"
echo "  2. Open 06_training_notebook.ipynb in Jupyter to train custom OCR model"
echo "  3. See README.md for detailed usage instructions"
echo ""
echo "=================================================="
