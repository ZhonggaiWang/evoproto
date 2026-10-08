"""Plot completed fixed-baseline comparisons within each incremental stage.

Usage:
    python tools/plot_baseline_study.py --study runs/fixed_baseline_v1

Requires all nine final-stage metric records at their expected final iterations.
The PNG and PDF are auxiliary figures; results.json remains the metric source.
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("base", "kd", "sep", "kd_sep")


def checked_project_path(path):
    path = Path(os.path.abspath(path))
    if not path.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"Path must remain inside this project: {path}")
    current = PROJECT_ROOT
    for component in path.relative_to(PROJECT_ROOT).parts:
        current = current / component
        if current.is_symlink():
            raise ValueError(f"Symlink paths are not allowed for plot output: {current}")
        if current.exists() and current.is_mount():
            raise ValueError(f"Nested mount paths are not allowed for plot output: {current}")
    resolved = path.resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"Resolved path escapes this project: {resolved}")
    if resolved.exists() and resolved.is_file() and resolved.stat().st_nlink != 1:
        raise ValueError(f"Plot output must not modify a hardlinked file: {resolved}")
    return resolved


def read_complete_results(study):
    manifest_path = study / "study.json"
    results_path = study / "results.json"
    if not manifest_path.is_file() or not results_path.is_file():
        raise ValueError("Study incomplete: study.json and results.json must both be available.")
    manifest = json.loads(manifest_path.read_text())
    results = json.loads(results_path.read_text())
    if manifest.get("task") != "10-5":
        raise ValueError("This figure is scoped to the VOC 10-5 baseline study.")
    expected_groups = {"shared": (0,), **{variant: (1, 2) for variant in VARIANTS}}
    if not isinstance(results, dict) or set(results) != set(expected_groups):
        raise ValueError("Study incomplete: expected the shared initial stage and all four two-stage variants.")
    for group, steps in expected_groups.items():
        records = results[group]
        if not isinstance(records, dict) or set(records) != {str(step) for step in steps}:
            raise ValueError(f"Study incomplete: {group} must contain final records for steps {steps}.")
        for step in steps:
            record = records[str(step)]
            if not isinstance(record, dict):
                raise ValueError(f"Study incomplete: {group} step {step} has no final metric record.")
            final_iteration = manifest["step0_iters" if step == 0 else "incremental_iters"]
            if record.get("step") != step or record.get("iteration") != final_iteration:
                raise ValueError(f"Study incomplete: {group} step {step} must be at final iteration {final_iteration}.")
            metric_names = ("old_miou", "all_miou") if step == 0 else ("old_miou", "new_miou", "all_miou")
            for name in metric_names:
                value = record.get(name)
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"Study incomplete: {group} step {step} has no finite {name}.")
    return manifest, results


def stage_deltas(results, step):
    baseline = results["base"][str(step)]
    return {
        variant: {name: results[variant][str(step)][name] - baseline[name]
                  for name in ("old_miou", "new_miou", "all_miou")}
        for variant in ("kd", "sep", "kd_sep")
    }


def padded_limits(values):
    lower, upper = min([0.0] + values), max([0.0] + values)
    padding = 0.28 * max(upper - lower, 0.05)
    return lower - padding, upper + padding


def create_figure(manifest, results):
    # Configure the workspace cache before importing matplotlib; install nothing.
    cache = checked_project_path(PROJECT_ROOT / ".runtime" / "cache" / "matplotlib")
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    figure, axes = plt.subplots(1, 2, figsize=(13, 6.4))
    smoke = bool(manifest.get("smoke"))
    ald = "on" if manifest.get("ald") else "off"
    figure.suptitle(f"VOC 10-5 | seed {manifest['seed']} | ALD {ald}", fontsize=16, y=0.97)
    weights = manifest["variants"]
    condition = (f"KD weight {weights['kd'][0]:g}; separation weight {weights['sep'][1]:g}; "
                 f"margin {manifest['prototype_margin']:g}; shared initial checkpoint")
    figure.text(0.5, 0.925, condition, ha="center", fontsize=10, color="#374151")
    if smoke:
        figure.text(0.5, 0.87, "SMOKE — NOT PERFORMANCE", ha="center", color="#b91c1c",
                    fontsize=17, fontweight="bold")

    styles = {
        "kd": ("KD", "#2563eb", "o", 38),
        "sep": ("SEP", "#059669", "s", -2),
        "kd_sep": ("KD+SEP", "#c026d3", "D", -42),
    }
    for step, axis in enumerate(axes, start=1):
        deltas = stage_deltas(results, step)
        axis.axhline(0, color="#6b7280", linewidth=0.9, linestyle="--", zorder=1)
        axis.axvline(0, color="#6b7280", linewidth=0.9, linestyle="--", zorder=1)
        axis.grid(alpha=0.16, linewidth=0.7)
        axis.scatter([0], [0], marker="+", color="#111827", s=155, linewidths=2.0, zorder=5)
        axis.annotate("BASE", (0, 0), xytext=(8, -17), textcoords="offset points",
                      color="#111827", fontsize=9, fontweight="bold", zorder=6)
        x_values = [record["old_miou"] for record in deltas.values()]
        y_values = [record["new_miou"] for record in deltas.values()]
        x_limits = padded_limits(x_values)
        y_limits = padded_limits(y_values)
        axis.set_xlim(x_limits)
        axis.set_ylim(y_limits)
        for variant, (label, color, marker, vertical_offset) in styles.items():
            record = deltas[variant]
            x, y = record["old_miou"], record["new_miou"]
            axis.scatter([x], [y], marker=marker, color=color, s=80,
                          edgecolor="white", linewidth=0.7, zorder=4)
            on_right = x >= (x_limits[0] + x_limits[1]) / 2
            y_fraction = (y - y_limits[0]) / (y_limits[1] - y_limits[0])
            if y_fraction < 0.3 and vertical_offset < 0:
                vertical_offset = max(vertical_offset, -20)
            elif y_fraction > 0.7 and vertical_offset > 0:
                vertical_offset = min(vertical_offset, 20)
            axis.annotate(f"{label}\nΔall+bg = {record['all_miou']:+.2f} pp", (x, y),
                          xytext=(-10 if on_right else 10, vertical_offset),
                          textcoords="offset points", ha="right" if on_right else "left",
                          va="center", fontsize=9, color=color, annotation_clip=False,
                          arrowprops={"arrowstyle": "-", "color": color, "alpha": 0.6,
                                      "linewidth": 0.7}, zorder=6)
        axis.set_title(f"Incremental step {step}: initial 10 / added {5 * step}", fontsize=12, pad=12)
        axis.set_xlabel("Δ initial 10 foreground mIoU (pp)", labelpad=10)
        axis.set_ylabel("Δ new foreground mIoU since step 0 (pp)", labelpad=8)
        if smoke:
            axis.text(0.5, 0.55, "SMOKE", transform=axis.transAxes, ha="center", va="center",
                      fontsize=45, color="#b91c1c", alpha=0.08, fontweight="bold", zorder=0)
    figure.text(0.5, 0.072, "Every point is relative to BASE in the same stage. Δall+bg includes background.",
                ha="center", fontsize=10, color="#374151")
    figure.text(0.5, 0.037, "Single seed; no error bars. No forgetting inference from differences between stages.",
                ha="center", fontsize=10, color="#374151")
    figure.subplots_adjust(left=0.075, right=0.98, bottom=0.22,
                           top=0.75 if smoke else 0.83, wspace=0.32)
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    study_arg = args.study if args.study.is_absolute() else PROJECT_ROOT / args.study
    try:
        study = checked_project_path(study_arg)
        if not study.is_dir():
            raise ValueError("Study incomplete: the study directory does not exist.")
        outputs = {format_: checked_project_path(study / f"comparison.{format_}")
                   for format_ in ("png", "pdf")}
        for output in outputs.values():
            if output.exists() and not output.is_file():
                raise ValueError(f"Plot output must be a regular file: {output}")
        manifest, results = read_complete_results(study)
        figure = create_figure(manifest, results)
        temporary_outputs = {}
        for format_ in outputs:
            with tempfile.NamedTemporaryFile(prefix=".comparison-", suffix=f".{format_}",
                                             dir=study, delete=False) as temporary:
                temporary_outputs[format_] = Path(temporary.name)
            figure.savefig(temporary_outputs[format_], format=format_, dpi=180, bbox_inches="tight",
                           facecolor="white")
        for output in outputs.values():
            checked_project_path(output)
        for format_, output in outputs.items():
            os.replace(temporary_outputs[format_], output)
        import matplotlib.pyplot as plt
        plt.close(figure)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print("Comparison figure generated from all 9 final-stage metric records.")
    if manifest.get("smoke"):
        print("SMOKE: plot-generation check only; these are not performance results.")
    for output in outputs.values():
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
