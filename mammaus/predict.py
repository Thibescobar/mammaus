"""Frame-by-frame image classification and per-acquisition report generation."""

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from transformers import pipeline

from mammaus.constants import (
    DEFAULT_MALIGNANT_THRESHOLD,
    DEFAULT_MIN_RUN,
    LABELS,
    MODEL_ID,
    setup_logging,
)
from mammaus.reporting import make_acquisition_figure, print_acquisition_report

logging.getLogger("transformers.pipelines.base").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


def predict_cli() -> None:
    """CLI entry point: classify ultrasound frames and generate reports."""
    parser = argparse.ArgumentParser(description="AI classification on ultrasound video frames")
    parser.add_argument("input_path", help="Folder containing preprocessed PNGs")
    parser.add_argument("--output", default="results", help="Output folder (default: results)")
    parser.add_argument("--model", default=MODEL_ID, help=f"HuggingFace model ID (default: {MODEL_ID})")
    parser.add_argument("--min-run", type=int, default=DEFAULT_MIN_RUN, help=f"Min consecutive malignant frames for alert (default: {DEFAULT_MIN_RUN})")
    parser.add_argument("--threshold", type=float, default=DEFAULT_MALIGNANT_THRESHOLD, help=f"Malignant confidence threshold in %% (default: {DEFAULT_MALIGNANT_THRESHOLD})")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed logging")
    args = parser.parse_args()

    setup_logging(args.verbose)

    p = Path(args.input_path)
    if p.is_file():
        acquisitions = {"single": [p]}
    else:
        all_pngs = sorted(p.rglob("*.png"))
        if not all_pngs:
            print(f"No PNG images found in: {args.input_path}")
            sys.exit(1)
        acquisitions = defaultdict(list)
        for img in all_pngs:
            acq_name = img.parent.name
            acquisitions[acq_name].append(img)
    print(f"Loading model {args.model}...")
    import torch
    if torch.cuda.is_available():
        logger.debug("CUDA available: using GPU (%s)", torch.cuda.get_device_name(0))
    else:
        print("  ⚠ Running on CPU (no CUDA detected). Inference will be slow.")
        print("    For GPU support: pip install torch --index-url https://download.pytorch.org/whl/cu124")
    classifier = pipeline("image-classification", model=args.model)  # type: ignore[arg-type]
    results_dir = Path(args.output)
    results_dir.mkdir(parents=True, exist_ok=True)
    total_images = sum(len(imgs) for imgs in acquisitions.values())
    print(f"\n{'='*60}")
    print(f" ANALYSIS — {len(acquisitions)} acquisition(s), {total_images} image(s)")
    print(f"{'='*60}")
    for acq_name, images in sorted(acquisitions.items()):
        n = len(images)
        print(f"\n  >>> {acq_name} ({n} frames)...")
        scores = {"benign": [], "malignant": [], "normal": []}
        for idx, img_path in enumerate(images, 1):
            pct = idx * 100 // n
            bar_len = 30
            filled = pct * bar_len // 100
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"\r  [{bar}] {pct:3d}%  ({idx}/{n})", end="", flush=True)
            results = classifier(str(img_path))
            score_map = {}
            for r in results:
                label = LABELS.get(r["label"], r["label"])
                score_map[label] = r["score"] * 100
            for label in ("benign", "malignant", "normal"):
                scores[label].append(score_map.get(label, 0.0))
        print()  # end of progress bar
        logger.debug("Acquisition %s: %d frames classified", acq_name, n)
        scores_dir = results_dir / "scores"
        scores_dir.mkdir(exist_ok=True)
        npz_path = scores_dir / f"{acq_name}_scores.npz"
        np.savez(
            npz_path,
            benign=np.array(scores["benign"]),
            malignant=np.array(scores["malignant"]),
            normal=np.array(scores["normal"]),
        )
        print_acquisition_report(acq_name, scores, results_dir, threshold=args.threshold, min_run=args.min_run)
        fig_path = make_acquisition_figure(acq_name, scores, results_dir, threshold=args.threshold)
        print(f"  Figure saved: {fig_path}")
    print(f"\n{'='*60}")
    print(f"  Figures in: {results_dir.resolve()}")
    print("  WARNING: Assistance tool — does not replace medical advice")
    print(f"{'='*60}")
