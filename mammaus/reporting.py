"""Clinical reporting: figures, statistics, and text reports."""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from mammaus.constants import LABEL_COLORS, LABEL_EN, SERIES_NAMES_EN


def series_display_name(acq_name: str) -> str:
    """Return an explicit English name for a series."""
    parts = acq_name.split("_", 1)
    code = parts[1] if len(parts) > 1 else parts[0]
    en_name = SERIES_NAMES_EN.get(code, code)
    return f"{acq_name}  —  {en_name}"

def find_malignant_frames(scores: dict, threshold: float = 30.0) -> tuple[list, list]:
    """Identify frames with a significant malignant score.

    Returns:
        top1_frames: frames where malignant is the dominant class.
        suspect_frames: frames where malignant score >= threshold but not dominant.
    """
    malignant = np.array(scores["malignant"])
    top1_frames = []
    suspect_frames = []
    for i in range(len(malignant)):
        frame_scores = {lbl: scores[lbl][i] for lbl in ("benign", "malignant", "normal")}
        top_label = max(frame_scores, key=frame_scores.get)
        if top_label == "malignant":
            top1_frames.append((i, malignant[i]))
        elif malignant[i] >= threshold:
            suspect_frames.append((i, malignant[i]))
    return top1_frames, suspect_frames

def find_consecutive_malignant_runs(scores: dict, min_run: int = 3) -> list[tuple[int, int, int]]:
    """Find runs of consecutive frames classified as malignant (top-1).

    Returns:
        List of (start_frame, end_frame, run_length) tuples.
    """
    n = len(scores["benign"])
    is_malig = []
    for i in range(n):
        frame_scores = {lbl: scores[lbl][i] for lbl in ("benign", "malignant", "normal")}
        is_malig.append(max(frame_scores, key=frame_scores.get) == "malignant")
    runs = []
    start = None
    for i in range(n):
        if is_malig[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                length = i - start
                if length >= min_run:
                    runs.append((start, i - 1, length))
                start = None
    if start is not None:
        length = n - start
        if length >= min_run:
            runs.append((start, n - 1, length))
    return runs


def generate_global_text_report(all_stats: dict, results_dir: Path, n_acq: int, n_frames: int) -> None:
    """Generate and save a global text report summarizing all acquisitions."""
    n_to_check = sum(1 for s in all_stats.values() if s["to_check"])
    reassuring = n_to_check == 0
    malignant_overall = sum(s['n_malig_top1'] for s in all_stats.values())
    malignant_pct = 100 * malignant_overall / n_frames if n_frames else 0
    summary = []
    summary.append("CLINICAL SUMMARY")
    summary.append("-----------------")
    summary.append(f"This automated analysis covers {n_acq} ultrasound acquisitions, totaling {n_frames} frames.")
    if reassuring:
        summary.append(
            "Across all acquisitions, the vast majority of frames were"
            " classified as benign or normal. No acquisition showed a"
            " significant sequence of consecutive frames classified as"
            " malignant. The overall risk of suspicious findings is low,"
            " and the results are globally reassuring."
        )
    else:
        summary.append(
            f"Of the {n_acq} acquisitions, {n_to_check} showed at least"
            f" one sequence of ≥3 consecutive frames classified as"
            f" malignant, which may warrant further review. The overall"
            f" proportion of frames classified as malignant is"
            f" {malignant_pct:.1f}%. Most acquisitions remain predominantly"
            f" benign or normal, but targeted attention is recommended for"
            f" those flagged as 'TO CHECK' in the table below."
        )
    summary.append("")
    summary.append(
        "Interpretation: This report provides a global overview of the"
        " distribution of benign, malignant, and normal classifications"
        " across all breast ultrasound acquisitions. For each acquisition,"
        " the percentage of frames classified as malignant, as well as the"
        " mean and maximum malignant confidence scores, are provided."
        " Acquisitions flagged as 'TO CHECK' contain sequences of"
        " consecutive malignant frames and may correspond to areas of"
        " clinical concern. These findings should be interpreted in the"
        " context of the full clinical and imaging picture, and do not"
        " replace expert radiological or oncological assessment."
    )
    summary.append("")
    summary.append("For detailed per-acquisition results, refer to the individual reports.")
    summary.append("")
    # Table and rest of report
    lines = []
    lines.append("="*60)
    lines.extend(summary)
    lines.append("="*60)
    lines.append("  GLOBAL REPORT — MULTI-ACQUISITION SUMMARY")
    lines.append(f"  {n_acq} acquisitions, {n_frames} frames analyzed")
    lines.append("="*60)
    lines.append("")
    lines.append("SUMMARY BY ACQUISITION")
    lines.append("----------------------")
    header = (
        f"{'Acquisition':<12} {'Frames':>6} {'Benign':>8}"
        f" {'Normal':>8} {'Malignant':>10} {'MaligMean':>10}"
        f" {'MaligMax':>10} {'Review':>10}"
    )
    lines.append(header)
    lines.append("-"*len(header))
    for acq in sorted(all_stats.keys()):
        s = all_stats[acq]
        review = "TO CHECK" if s["to_check"] else "Reassuring"
        row = (
            f"{acq:<12} {s['n_frames']:>6}"
            f" {s['pct']['benign']:>7.0f}%"
            f" {s['pct']['normal']:>7.0f}%"
            f" {s['pct']['malignant']:>9.1f}%"
            f" {s['mean_malig']:>9.1f}%"
            f" {s['max_malig']:>9.1f}%"
            f" {review:>10}"
        )
        lines.append(row)
    lines.append("")
    if reassuring:
        lines.append("✓ All acquisitions are overall reassuring.")
    else:
        lines.append(f"⚠ {n_to_check} acquisition(s) require review (consecutive malignant frames detected).")
    lines.append("")
    lines.append("This result is provided for information only.")
    lines.append("It does not replace a radiologist's opinion in any way.")
    lines.append("="*60)
    report_path = results_dir / "global_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Global text report saved: {report_path}")


def compute_stats(scores: dict, min_run: int = 3) -> dict:
    """Compute statistics for a given acquisition's scores."""
    benign, malignant, normal = scores["benign"], scores["malignant"], scores["normal"]
    n = len(benign)
    top_labels = []
    for i in range(n):
        fs = {"benign": benign[i], "malignant": malignant[i], "normal": normal[i]}
        top_labels.append(max(fs, key=fs.get))
    counts = {lbl: top_labels.count(lbl) for lbl in ("benign", "malignant", "normal")}
    scores_dict = {"benign": benign, "malignant": malignant, "normal": normal}
    runs = find_consecutive_malignant_runs(scores_dict, min_run=min_run)
    return {
        "n_frames": n,
        "counts": counts,
        "pct": {lbl: counts[lbl] / n * 100 for lbl in counts},
        "mean_malig": float(malignant.mean()),
        "max_malig": float(malignant.max()),
        "n_malig_top1": counts["malignant"],
        "consecutive_runs": runs,
        "to_check": len(runs) > 0,
    }

def make_acquisition_figure(acq_name: str, scores: dict, out_dir: Path):
    """Generate and save a per-acquisition confidence plot with color bar."""
    n_frames = len(scores["benign"])
    frames = np.arange(n_frames)
    malignant_top1, malignant_suspect = find_malignant_frames(scores)
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={"height_ratios": [3, 1]})
    display_name = series_display_name(acq_name)
    fig.suptitle(display_name, fontsize=13, fontweight="bold")
    ax = axes[0]
    for label in ("benign", "malignant", "normal"):
        ax.plot(frames, scores[label], label=LABEL_EN[label], color=LABEL_COLORS[label], alpha=0.85, linewidth=1)
    if malignant_top1:
        idx = [f[0] for f in malignant_top1]
        vals = [f[1] for f in malignant_top1]
        ax.scatter(
            idx, vals, color="#e74c3c", s=60, zorder=5,
            marker="v", label=f"Malignant top-1 ({len(idx)} frames)",
        )
    if malignant_suspect:
        idx = [f[0] for f in malignant_suspect]
        vals = [f[1] for f in malignant_suspect]
        ax.scatter(
            idx, vals, color="#e74c3c", s=30, zorder=4, marker="o",
            facecolors="none", linewidths=1.5,
            label=f"Malignant ≥30% ({len(idx)} frames)",
        )
    ax.set_ylabel("Model confidence (%)")
    ax.set_xlabel("Frame number")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax2 = axes[1]
    top_labels = []
    for i in range(n_frames):
        frame_scores = {lbl: scores[lbl][i] for lbl in ("benign", "malignant", "normal")}
        top_labels.append(max(frame_scores, key=frame_scores.get))
    colors = [LABEL_COLORS[lbl] for lbl in top_labels]
    ax2.bar(frames, [1] * n_frames, color=colors, width=1.0, edgecolor="none")
    ax2.set_yticks([])
    ax2.set_xlabel("Frame number")
    ax2.set_title("Result per frame (color = predicted class)", fontsize=10)
    legend_elements = [
        Patch(facecolor=LABEL_COLORS[lbl], label=LABEL_EN[lbl])
        for lbl in ("benign", "malignant", "normal")
    ]
    ax2.legend(handles=legend_elements, loc="upper right", ncol=3, fontsize=8)
    plt.tight_layout()
    fig_path = out_dir / f"{acq_name}.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path

def print_acquisition_report(acq_name: str, scores: dict, out_dir: Path) -> None:
    """Generate and save a detailed text report for a single acquisition."""
    n_frames = len(scores["benign"])
    benign = np.array(scores["benign"])
    malignant = np.array(scores["malignant"])
    normal = np.array(scores["normal"])
    display_name = series_display_name(acq_name)
    top_per_frame = []
    for i in range(n_frames):
        frame_scores = {"benign": benign[i], "malignant": malignant[i], "normal": normal[i]}
        top_per_frame.append(max(frame_scores, key=frame_scores.get))
    counts = {lbl: top_per_frame.count(lbl) for lbl in ("benign", "malignant", "normal")}
    malignant_top1, malignant_suspect = find_malignant_frames(scores)
    lines = []
    lines.append(f"{'═'*60}")
    lines.append(f"  {display_name}")
    lines.append(f"  {n_frames} frames analyzed")
    lines.append(f"{'═'*60}")
    lines.append("")
    lines.append("SUMMARY")
    lines.append("------")
    lines.append("The model analyzed each frame of this acquisition and")
    lines.append("assigned a confidence score for 3 categories:")
    lines.append("  • Benign    = non-cancerous appearance")
    lines.append("  • Malignant = suspicious/cancerous appearance")
    lines.append("  • Normal    = no visible abnormality")
    lines.append("")
    lines.append("RESULT BY CATEGORY")
    lines.append("------------------")
    for label in ("benign", "malignant", "normal"):
        pct = counts[label] / n_frames * 100
        bar = "█" * int(pct / 2)
        lines.append(f"  {LABEL_EN[label]:>20s} : {counts[label]:4d} / {n_frames} frames ({pct:5.1f}%)  {bar}")
    lines.append("")
    lines.append("MODEL MEAN CONFIDENCE")
    lines.append("---------------------")
    lines.append("  (For each frame, the model gives a confidence %")
    lines.append("   for each category. Below: the mean over all frames, as well as min/max values.)")
    lines.append(
        f"  {LABEL_EN['benign']:>20s} : {benign.mean():5.1f}%"
        f"  [min {benign.min():.1f}% — max {benign.max():.1f}%]"
    )
    lines.append(
        f"  {LABEL_EN['malignant']:>20s} : {malignant.mean():5.1f}%"
        f"  [min {malignant.min():.1f}% — max {malignant.max():.1f}%]"
    )
    lines.append(
        f"  {LABEL_EN['normal']:>20s} : {normal.mean():5.1f}%"
        f"  [min {normal.min():.1f}% — max {normal.max():.1f}%]"
    )
    lines.append("")
    if malignant_top1 or malignant_suspect:
        lines.append("⚠  SUSPICIOUS FRAMES (malignancy)")
        lines.append("----------------------------------")
        if malignant_top1:
            lines.append("  Frames classified as MALIGNANT (dominant category):")
            for frame_idx, score in malignant_top1:
                lines.append(f"    → frame_{frame_idx:03d}.png  (malignant confidence: {score:.1f}%)")
        if malignant_suspect:
            lines.append("  Frames with malignant score ≥ 30% (not dominant but notable):")
            for frame_idx, score in malignant_suspect:
                lines.append(f"    → frame_{frame_idx:03d}.png  (malignant confidence: {score:.1f}%)")
    else:
        lines.append("✓  No suspicious malignant frame detected.")
    consecutive_runs = find_consecutive_malignant_runs(scores, min_run=3)
    lines.append("")
    lines.append("GENERAL ASSESSMENT (automatic heuristic)")
    lines.append("----------------------------------------")
    lines.append("  Method: the model may misclassify an isolated frame.")
    lines.append("  In ultrasound, a true lesion appears on several consecutive")
    lines.append("  frames as the probe sweeps a continuous area. Thus, a sequence")
    lines.append("  of ≥ 3 consecutive malignant frames is considered significant, while")
    lines.append("  isolated malignant frames are likely model misclassifications.")
    lines.append("")
    if consecutive_runs:
        lines.append("  ⚠  RESULT: TO CHECK")
        lines.append(f"  {len(consecutive_runs)} sequence(s) of consecutive malignant frames detected:")
        for start, end, length in consecutive_runs:
            lines.append(f"    → frames {start:03d} to {end:03d}  ({length} consecutive frames)")
    else:
        lines.append("  ✓  RESULT: OVERALL REASSURING")
        if malignant_top1:
            lines.append(f"  {len(malignant_top1)} frame(s) classified as malignant but isolated,")
            lines.append("  without a consecutive sequence ≥ 3 — likely model outliers.")
        else:
            lines.append("  No frame classified as malignant.")
    lines.append("")
    lines.append(f"{'═'*60}")
    lines.append("  This result is provided for information only.")
    lines.append("  It does not replace a radiologist's opinion in any way.")
    lines.append(f"{'═'*60}")
    report = "\n".join(lines)
    report_path = out_dir / f"{acq_name}_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Report saved: {report_path}")


def make_global_figure(all_data: dict, all_stats: dict, out_dir: Path):
    """Generate and save a global summary figure (bar charts + table)."""
    acq_names = list(all_data.keys())
    n_acq = len(acq_names)
    labels_short = [a.split('_', 1)[1] if '_' in a else a for a in acq_names]
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.45, wspace=0.30)
    ax1 = fig.add_subplot(gs[0, 0])
    y_pos = np.arange(n_acq)
    bottoms = np.zeros(n_acq)
    for label in ("benign", "normal", "malignant"):
        vals = [all_stats[a]["pct"][label] for a in acq_names]
        ax1.barh(
            y_pos, vals, left=bottoms, color=LABEL_COLORS[label],
            label=LABEL_EN[label], edgecolor="white", linewidth=0.5,
        )
        bottoms += vals
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels_short, fontsize=9)
    ax1.set_xlabel("% of frames")
    ax1.set_title("Distribution by category", fontsize=11, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.set_xlim(0, 100)
    ax1.invert_yaxis()
    ax2 = fig.add_subplot(gs[0, 1])
    means = [all_stats[a]["mean_malig"] for a in acq_names]
    maxes = [all_stats[a]["max_malig"] for a in acq_names]
    x = np.arange(n_acq)
    w = 0.35
    ax2.bar(x - w / 2, means, w, color="#e74c3c", alpha=0.6, label="Malignant mean")
    ax2.bar(x + w / 2, maxes, w, color="#e74c3c", alpha=1.0, label="Malignant max")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_short, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Malignant score (%)")
    ax2.set_title("Malignancy score per acquisition", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.axhline(y=30, color="orange", linestyle="--", alpha=0.7, label="30% threshold")
    ax2.set_ylim(0, max(maxes) * 1.15 if maxes else 100)
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis("off")
    col_labels = [
        "Acquisition", "Frames", "Benign", "Normal",
        "Malignant", "Malignant\nmean", "Malignant\nmax", "Review",
    ]
    rows = []
    cell_colors = []
    for a in acq_names:
        s = all_stats[a]
        review = "⚠ To check" if s["to_check"] else "✓ Reassuring"
        row = [
            labels_short[acq_names.index(a)],
            str(s["n_frames"]),
            f"{s['pct']['benign']:.0f}%",
            f"{s['pct']['normal']:.0f}%",
            f"{s['pct']['malignant']:.1f}%",
            f"{s['mean_malig']:.1f}%",
            f"{s['max_malig']:.1f}%",
            review,
        ]
        rows.append(row)
        if s["to_check"]:
            cell_colors.append(["#ffeaea"] * len(col_labels))
        else:
            cell_colors.append(["#eafff0"] * len(col_labels))
    table = ax3.table(
        cellText=rows, colLabels=col_labels,
        cellColours=cell_colors, loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)
    for j in range(len(col_labels)):
        table[0, j].set_text_props(fontweight="bold")
        table[0, j].set_facecolor("#d5d5d5")
    ax3.text(
        0.5, -0.18, "Summary by acquisition", fontsize=11,
        fontweight="bold", ha="center", va="top",
        transform=ax3.transAxes,
    )
    n_to_check = sum(1 for s in all_stats.values() if s["to_check"])
    total_frames = sum(s["n_frames"] for s in all_stats.values())
    title = f"Global report — {n_acq} acquisitions, {total_frames} frames"
    if n_to_check == 0:
        title += "\n✓ All acquisitions are overall reassuring"
    else:
        title += f"\n⚠ {n_to_check} acquisition(s) to check"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    fig_path = out_dir / "global_report.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def report_global_cli() -> None:
    """CLI entry point: generate global multi-acquisition report from saved scores."""
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Generate global multi-acquisition report")
    parser.add_argument("--results", default="results", help="Results folder (default: results)")
    args = parser.parse_args()

    results_dir = Path(args.results)
    scores_dir = results_dir / "scores"
    if not scores_dir.exists():
        print(f"No scores folder found at: {scores_dir}")
        sys.exit(1)

    npz_files = sorted(scores_dir.glob("*_scores.npz"))
    if not npz_files:
        print(f"No score files found in: {scores_dir}")
        sys.exit(1)

    all_data = {}
    all_stats = {}
    for npz_path in npz_files:
        acq_name = npz_path.stem.replace("_scores", "")
        data = np.load(npz_path)
        scores = {k: data[k] for k in ("benign", "malignant", "normal")}
        all_data[acq_name] = scores
        all_stats[acq_name] = compute_stats(scores)

    n_acq = len(all_data)
    n_frames = sum(s["n_frames"] for s in all_stats.values())

    generate_global_text_report(all_stats, results_dir, n_acq, n_frames)
    fig_path = make_global_figure(all_data, all_stats, results_dir)
    print(f"Global figure saved: {fig_path}")
