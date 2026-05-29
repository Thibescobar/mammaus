"""Frame-by-frame image classification and per-acquisition report generation."""

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from transformers import pipeline

from mammaus.constants import LABELS, MODEL_ID
from mammaus.reporting import make_acquisition_figure, print_acquisition_report

logging.getLogger("transformers.pipelines.base").setLevel(logging.ERROR)


def predict_cli() -> None:
    """CLI entry point: classify ultrasound frames and generate reports."""
    parser = argparse.ArgumentParser(description="AI classification on ultrasound video frames")
    parser.add_argument("input_path", help="Folder containing preprocessed PNGs")
    parser.add_argument("--output", default="results", help="Output folder (default: results)")
    args = parser.parse_args()
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
    print(f"Loading model {MODEL_ID}...")
    classifier = pipeline("image-classification", model=MODEL_ID)  # type: ignore[arg-type]
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
        scores_dir = results_dir / "scores"
        scores_dir.mkdir(exist_ok=True)
        npz_path = scores_dir / f"{acq_name}_scores.npz"
        np.savez(
            npz_path,
            benign=np.array(scores["benign"]),
            malignant=np.array(scores["malignant"]),
            normal=np.array(scores["normal"]),
        )
        print_acquisition_report(acq_name, scores, results_dir)
        fig_path = make_acquisition_figure(acq_name, scores, results_dir)
        print(f"  Figure saved: {fig_path}")
    print(f"\n{'='*60}")
    print(f"  Figures in: {results_dir.resolve()}")
    print("  WARNING: Assistance tool — does not replace medical advice")
    print(f"{'='*60}")
