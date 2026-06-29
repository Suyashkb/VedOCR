"""
run_pipeline.py — Run the full Sanskrit OCR data pipeline in one command.

  python run_pipeline.py          # all steps
  python run_pipeline.py --from 3 # resume from step 3
  python run_pipeline.py --only 4 # run only step 4
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

STEPS = {
    1: ("Parse Label Studio JSON + crop images",      "step1_parse",         "main"),
    2: ("Create train / val / test splits",            "step2_split",         "main"),
    3: ("Export to PaddleOCR / HuggingFace / JSONL",  "step3_export_formats","main"),
    4: ("Build vocabulary + print statistics",         "step4_vocab_and_stats","main"),
    6: ("Visualize bounding boxes (QA check)",         "step6_visualize",     "main"),
}


def run_step(step_num: int):
    label, module_name, fn_name = STEPS[step_num]
    print(f"\n{'='*58}")
    print(f"  STEP {step_num}: {label}")
    print(f"{'='*58}")
    import importlib
    mod = importlib.import_module(module_name)
    getattr(mod, fn_name)()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="from_step", type=int, default=1)
    parser.add_argument("--only", dest="only_step", type=int, default=None)
    args = parser.parse_args()

    if args.only_step:
        run_step(args.only_step)
    else:
        for step_num in sorted(STEPS):
            if step_num >= args.from_step:
                run_step(step_num)

    print("\n\n  ✓  Pipeline complete.")
    print("  Evaluate zero-shot with:")
    print("    python step5_evaluate.py --engine tesseract --lang san")
    print("    python step5_evaluate.py --engine file --pred-file my_preds.tsv")


if __name__ == "__main__":
    main()
