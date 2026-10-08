"""Render directed confusion diagnostics as a standalone static PNG.

Usage:
    python plot_online_confusion.py --input merged_result.json --output figure.png
    python plot_online_confusion.py --input merged_result.json --output warmup.png --checkpoint warmup
    python plot_online_confusion.py --input merged_result.json --output trusted.png --view trusted

Expected input: {"checkpoints": {"warmup": {"metrics": ...,
"broad_PAR_reference_metrics": ...}, "final": {"metrics": ...,
"broad_PAR_reference_metrics": ...}}, "perclass_coverage": [...]}. The default
broad view uses broad_PAR_reference_metrics; trusted uses metrics. A direct
evaluate_matrices() result is accepted for the trusted view only. Optional
class_ids and class_names at checkpoint or top level specify channel order.
No GT is fed back into training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


VOC_CLASS_NAMES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor",
]


def _checkpoint(payload, requested, view):
    checkpoints = payload.get("checkpoints")
    if checkpoints is None:
        checkpoint = payload
    else:
        if not isinstance(checkpoints, dict) or requested not in checkpoints:
            available = ", ".join(checkpoints) if isinstance(checkpoints, dict) else "none"
            raise ValueError(f"Checkpoint {requested!r} is unavailable; available: {available}")
        checkpoint = checkpoints[requested]
    if not isinstance(checkpoint, dict):
        raise ValueError("Each checkpoint must be a mapping")
    if view == "broad":
        metrics = checkpoint.get("broad_PAR_reference_metrics")
        if metrics is None:
            raise ValueError("Broad view requires checkpoint['broad_PAR_reference_metrics']; use --view trusted for gated metrics")
    else:
        metrics = checkpoint.get("metrics", checkpoint)
    if not isinstance(metrics, dict) or "matrices" not in metrics:
        raise ValueError("Checkpoint metrics must be an evaluate_matrices() result")
    return requested, checkpoint, metrics


def _matrix(rows, size, name):
    if not isinstance(rows, list) or len(rows) != size:
        raise ValueError(f"{name} must contain {size} rows")
    result = np.full((size, size), np.nan, dtype=np.float64)
    for i, row in enumerate(rows):
        if row is None:
            continue
        values = np.asarray(row, dtype=np.float64)
        if values.shape != (size,) or not np.isfinite(values).all():
            raise ValueError(f"{name} row {i} must contain {size} finite probabilities")
        if (values < 0).any() or (values > 1 + 1e-8).any():
            raise ValueError(f"{name} row {i} contains a probability outside [0, 1]")
        if not np.isclose(values.sum(), 1, atol=1e-6, rtol=1e-6):
            raise ValueError(f"{name} row {i} must be normalized over all classes")
        result[i] = values
    return result


def _labels(payload, checkpoint, size):
    ids = checkpoint.get("class_ids", payload.get("class_ids", list(range(size))))
    if not isinstance(ids, list) or len(ids) != size or len(set(ids)) != size:
        raise ValueError("class_ids must contain one unique VOC ID per matrix channel")
    if any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(VOC_CLASS_NAMES) for i in ids):
        raise ValueError("class_ids must contain VOC IDs in [0, 20]")
    names = checkpoint.get("class_names", payload.get("class_names"))
    if names is None:
        names = [VOC_CLASS_NAMES[i] for i in ids]
    if not isinstance(names, list) or len(names) != size or not all(isinstance(name, str) for name in names):
        raise ValueError("class_names must contain one string per matrix channel")
    indices = [i for i, class_id in enumerate(ids) if class_id != 0]
    if not indices:
        raise ValueError("At least one foreground class is required")
    return indices, [names[i] for i in indices]


def render(payload, output, checkpoint_name="final", view="broad"):
    """Write only the requested figure; return its resolved path."""
    # Select a noninteractive backend before importing pyplot. This is safe on
    # headless workers and does not open a UI or contact any external service.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle

    if view not in ("broad", "trusted"):
        raise ValueError("View must be broad or trusted")
    selected_name, checkpoint, metrics = _checkpoint(payload, checkpoint_name, view)
    anchor_label = "Broad PAR anchors" if view == "broad" else "CAM-gated anchors"
    size = int(metrics.get("num_classes", len(metrics["matrices"]["estimated_row_probabilities"])))
    indices, labels = _labels(payload, checkpoint, size)
    matrices = metrics["matrices"]
    estimate = _matrix(matrices["estimated_row_probabilities"], size, "estimated")
    full_gt = _matrix(matrices["gt_all_row_probabilities"], size, "GT full")
    same_gt = _matrix(matrices["gt_same_support_row_probabilities"], size, "GT same support")
    # Slice only after normalization: the displayed foreground block deliberately
    # retains probability mass lost to background in the full output distribution.
    foreground = np.ix_(indices, indices)
    estimate, full_gt, same_gt = estimate[foreground], full_gt[foreground], same_gt[foreground]
    difference = estimate - same_gt
    panels = [
        (estimate, "Pseudo-anchor estimate", False),
        (full_gt, "GT reference: full diagnostic population", False),
        (same_gt, "GT reference: same evidence support", False),
        (difference, "Estimate − GT reference (same support)", True),
    ]
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(2, 2, figsize=(18, 16), facecolor="white")
    fig.subplots_adjust(left=0.105, right=0.96, top=0.92, bottom=0.155, wspace=0.29, hspace=0.50)
    n = len(labels)
    probability_map = plt.get_cmap("YlGnBu").copy()
    difference_map = plt.get_cmap("RdBu_r").copy()
    probability_map.set_bad("#d6d6d6")
    difference_map.set_bad("#d6d6d6")
    for ax, (values, title, is_difference) in zip(axes.flat, panels):
        norm = Normalize(-1, 1) if is_difference else Normalize(0, 1)
        im = ax.imshow(
            np.ma.masked_invalid(values), cmap=difference_map if is_difference else probability_map,
            norm=norm, interpolation="nearest", aspect="equal", origin="upper",
        )
        ax.set_title(title, pad=13)
        ax.set_xticks(np.arange(n), labels=labels, rotation=65, ha="right", rotation_mode="anchor")
        ax.set_yticks(np.arange(n), labels=labels)
        ax.set_xlabel("Student prediction (target class)", labelpad=9)
        ax.set_ylabel("Source / anchor class", labelpad=9)
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.35)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(which="major", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        # Keep the diagonal visible. No symmetrization or diagonal suppression.
        for i in range(n):
            ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#565656", linewidth=0.45))
        # Missing observations are distinct from a measured zero confusion rate.
        for i in np.flatnonzero(np.isnan(values).all(axis=1)):
            ax.add_patch(Rectangle(
                (-0.5, i - 0.5), n, 1, facecolor="none", edgecolor="#888888",
                linewidth=0, hatch="///", zorder=4,
            ))
        ticks = [-1, -0.5, 0, 0.5, 1] if is_difference else [0, 0.25, 0.5, 0.75, 1]
        colorbar = fig.colorbar(im, ax=ax, fraction=0.037, pad=0.025, ticks=ticks)
        colorbar.set_label("Conditional-rate difference" if is_difference else "Conditional probability", labelpad=8)
        colorbar.outline.set_linewidth(0.5)
    fig.suptitle(f"Directed class-confusion diagnostics · {selected_name} · {anchor_label}", fontsize=18, y=0.975)
    fig.text(
        0.5, 0.069,
        "Rows are source classes; columns are student predictions. Foreground block shown; rates are normalized over all outputs, including background.",
        ha="center", va="center", fontsize=10,
    )
    fig.text(
        0.5, 0.049,
        "Estimate conditions on pseudo anchors; GT references condition on true classes. The difference is a diagnostic discrepancy between these conditional distributions.",
        ha="center", va="center", fontsize=10,
    )
    fig.text(
        0.5, 0.029,
        "Same-support reference uses exactly the selected evidence pixels and weights. Hatched gray rows have no observed support; they do not mean zero confusion.",
        ha="center", va="center", fontsize=10,
    )
    output = Path(output).resolve()
    if output.suffix.lower() != ".png":
        plt.close(fig)
        raise ValueError("Output must be a .png file")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white", metadata={
        "Title": f"Directed class-confusion diagnostics: {selected_name}; {anchor_label}",
        "Description": "Foreground directed conditional confusion matrices. GT is diagnostic only; matrices are not symmetrized.",
    })
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="Merged result JSON")
    parser.add_argument("--output", required=True, type=Path, help="Standalone PNG path")
    parser.add_argument("--checkpoint", default="final", help="Checkpoint key (default: final)")
    parser.add_argument("--view", choices=("broad", "trusted"), default="broad", help="Anchor view (default: broad PAR reference; trusted uses CAM-gated metrics)")
    args = parser.parse_args()
    with args.input.open("r", encoding="utf-8-sig") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        parser.error("Input JSON must be an object")
    try:
        output = render(payload, args.output, args.checkpoint, args.view)
    except (ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(output)


if __name__ == "__main__":
    main()
