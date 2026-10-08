"""Summarize a finished or ongoing VOC 10-5 fixed-weight baseline study.

Example: python -B tools/summarize_baseline_study.py runs/fixed_baseline_v1
Writes summary.json alongside study.json and prints a concise progress/table.
All IoUs use percentage units; differences are percentage points. Smoke-study
numbers are explicitly marked as pipeline checks, not performance estimates.
"""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("base", "kd", "sep", "kd_sep")
CLASS_ORDER = (
    "_background_", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)
TIMESTAMP = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3})")
ITERATION = re.compile(r"\bIter:\s*(\d+)")
ELAPSED = re.compile(r"Elasped:\s*((?:\d+ days?, )?\d+:\d\d:\d\d)")


def checked_path(path, output=False):
    path = path.absolute()
    if not path.resolve().is_relative_to(ROOT):
        raise ValueError(f"Path escapes project: {path}")
    for component in (path, *path.parents):
        if component == ROOT.parent:
            break
        if component.is_symlink():
            raise ValueError(f"Symbolic-link path is not accepted: {component}")
        if component != ROOT and component.exists() and component.is_mount():
            raise ValueError(f"Nested mount is not accepted: {component}")
    if output and path.exists() and (not path.is_file() or path.stat().st_nlink != 1):
        raise ValueError(f"Unsafe existing output: {path}")
    return path


def read_json(path, default=None):
    path = checked_path(path)
    return json.loads(path.read_text()) if path.is_file() else default


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def grouped_scores(metric, step):
    class_iou = metric.get("class_iou", {})
    last = 10 + 5 * step
    groups = {
        "background_iou": CLASS_ORDER[:1],
        "old_initial10_miou": CLASS_ORDER[1:11],
        "old_initial_with_background_miou": CLASS_ORDER[:11],
        "new_since_initial_miou": CLASS_ORDER[11:last + 1],
        "all_with_background_miou": CLASS_ORDER[:last + 1],
        "foreground_all_miou": CLASS_ORDER[1:last + 1],
        "previous_foreground_miou": CLASS_ORDER[1:11 + 5 * (step - 1)] if step else (),
        "previous_with_background_miou": CLASS_ORDER[:11 + 5 * (step - 1)] if step else (),
        "current_foreground_miou": CLASS_ORDER[11 + 5 * (step - 1):last + 1] if step else CLASS_ORDER[1:11],
    }
    scores, counts = {}, {}
    for name, classes in groups.items():
        values = [class_iou[c] for c in classes if finite(class_iou.get(c))]
        scores[name] = sum(values) / len(values) if values else None
        counts[name] = {"finite_classes": len(values), "expected_classes": len(classes)}
    return scores, counts


def duration_seconds(text):
    days, _, clock = text.partition(" day")
    if clock:
        clock = clock.lstrip("s, ")
        day_seconds = int(days) * 86400
    else:
        clock, day_seconds = text, 0
    hours, minutes, seconds = map(int, clock.split(":"))
    return day_seconds + hours * 3600 + minutes * 60 + seconds


def log_progress(path):
    path = checked_path(path)
    if not path.is_file():
        return {"last_training_iteration": 0, "logged_training_elapsed_seconds": None,
                "observed_stage_seconds": None}
    first, last, elapsed, iteration = None, None, None, 0
    for line in path.read_text(errors="replace").splitlines():
        stamp = TIMESTAMP.match(line)
        if stamp:
            current = datetime.strptime(stamp.group(1), "%Y-%m-%d %H:%M:%S,%f")
            first = first or current
            last = current
        match = ITERATION.search(line)
        if match:
            iteration = int(match.group(1))
        match = ELAPSED.search(line)
        if match:
            elapsed = duration_seconds(match.group(1))
    return {"last_training_iteration": iteration, "logged_training_elapsed_seconds": elapsed,
            "observed_stage_seconds": (last - first).total_seconds() if first and last else None}


def read_metrics(path, warnings):
    path = checked_path(path)
    if not path.is_file():
        return []
    records = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            metric = json.loads(line)
            if not isinstance(metric, dict):
                raise ValueError("record is not an object")
            records.append(metric)
        except (json.JSONDecodeError, ValueError) as error:
            warnings.append(f"Skipped unreadable metric line {path.relative_to(ROOT)}:{number}: {error}")
    return records


def summarize(study):
    manifest = read_json(study / "study.json")
    if not isinstance(manifest, dict) or manifest.get("task") != "10-5":
        raise ValueError("Expected a VOC 10-5 study.json")
    results = read_json(study / "results.json", {})
    warnings, stages = [], {}
    for variant in ("shared", *VARIANTS):
        stages[variant] = {}
        for step in ((0,) if variant == "shared" else (1, 2)):
            directory = checked_path(study / variant / "10-5" / f"step{step}")
            config = read_json(directory / "config.json", {})
            expected = manifest["step0_iters" if step == 0 else "incremental_iters"]
            progress = log_progress(directory / "train.log")
            records = read_metrics(directory / "metrics.jsonl", warnings)
            matching = [r for r in records if r.get("step") == step]
            final_records = [r for r in matching if r.get("iteration") == expected]
            metric = final_records[-1] if final_records else (matching[-1] if matching else None)
            published = results.get(variant, {}).get(str(step))
            if published is not None and metric is not None and published != metric:
                warnings.append(f"results.json and metrics.jsonl disagree for {variant} step {step}")
            if metric is None and published is not None:
                metric = published
                warnings.append(f"{variant} step {step} has results but no readable metrics.jsonl")
            checkpoint = checked_path(directory / "checkpoints/model_final.pth").is_file()
            complete = bool(final_records and checkpoint and config.get("max_iters") == expected)
            if config and config.get("max_iters") != expected:
                warnings.append(f"Iteration configuration mismatch for {variant} step {step}")
            expected_kd, expected_sep = (0, 0) if step == 0 else manifest["variants"][variant]
            for name, expected_value in {"w_proto_kd": expected_kd, "w_proto_sep": expected_sep,
                                         "ald": False, "confusion_reweight": False}.items():
                if config and config.get(name) != expected_value:
                    warnings.append(f"{variant} step {step}: unexpected {name}={config.get(name)}")
            metric_iteration = metric.get("iteration", 0) if metric else 0
            current = max(progress["last_training_iteration"], metric_iteration)
            status = "complete" if complete else ("in_progress" if config else "pending")
            if final_records and not checkpoint:
                status = "evaluated_waiting_checkpoint"
            elif current >= expected and not final_records:
                status = "awaiting_final_evaluation"
            scores, counts = grouped_scores(metric, step) if metric else ({}, {})
            stages[variant][str(step)] = {
                "status": status, "training_iteration": current, "expected_iterations": expected,
                "latest_evaluation_iteration": metric_iteration or None,
                "final_checkpoint_exists": checkpoint, "scores": scores,
                "score_class_counts": counts, "timing": progress,
                "weights": {"kd": config.get("w_proto_kd"), "sep": config.get("w_proto_sep")},
            }
    for variant in VARIANTS:
        for step in ("1", "2"):
            stage, base = stages[variant][step], stages["base"][step]
            stage["delta_vs_base_pp"] = {
                name: value - base["scores"][name]
                if stage["status"] == base["status"] == "complete"
                and finite(value) and finite(base["scores"].get(name)) else None
                for name, value in stage["scores"].items()
            }
    completed = sum(s["status"] == "complete" for group in stages.values() for s in group.values())
    status = "complete" if completed == 9 and results else ("awaiting_results_manifest" if completed == 9 else "in_progress")
    return {
        "study": str(study), "status": status, "smoke": bool(manifest.get("smoke")),
        "interpretation": "pipeline check only; not formal performance" if manifest.get("smoke") else "formal fixed-weight baseline study",
        "completed_stages": completed, "total_stages": 9,
        "results_manifest_exists": bool(results),
        "metric_units": "mIoU percent; delta_vs_base_pp in percentage points",
        "metric_definitions": {
            "background_iou": "background class 0 only",
            "old_initial10_miou": "foreground classes 1-10, excluding background",
            "old_initial_with_background_miou": "background plus initial foreground classes: class IDs 0-10, 11 classes",
            "new_since_initial_miou": "all foreground classes introduced since step 0: 11-15 at step 1; 11-20 at step 2",
            "all_with_background_miou": "all currently learned classes including background",
            "foreground_all_miou": "all currently learned foreground classes, excluding background",
            "previous_foreground_miou": "step 1: first 10; step 2: first 15 foreground classes",
            "previous_with_background_miou": "step 1: class IDs 0-10, 11 classes; step 2: class IDs 0-15, 16 classes; step 0: undefined",
            "current_foreground_miou": "step 1: classes 11-15; step 2: classes 16-20",
            "missing_classes": "None/nonfinite class IoUs are excluded from each mean, matching Trainer; finite class counts are reported",
            "timing": "logged training elapsed is measured at last training log; observed_stage_seconds is first-to-last log timestamp, including initialization/evaluation",
        },
        "stages": stages, "warnings": warnings,
        "failures": read_json(study / "failures.json", []) if status != "complete" else [],
    }


def display(summary):
    mode = "SMOKE — pipeline verification only, not formal performance" if summary["smoke"] else "Formal baseline"
    print(f"{mode}: {summary['status']} ({summary['completed_stages']}/9 stages complete)")
    initial = summary["stages"]["shared"]["0"]
    def fmt(value):
        return f"{value:.3f}" if finite(value) else "—"
    if initial["status"] == "complete":
        scores = initial["scores"]
        print(f"Shared step 0: initial10={fmt(scores['old_initial10_miou'])}; all+bg={fmt(scores['all_with_background_miou'])}")
    else:
        print(f"Shared step 0: {initial['status']} {initial['training_iteration']}/{initial['expected_iterations']} iterations")
    print("variant step initial10 subsequent all+bg Δold Δnew Δall (percentage points)")
    for variant in VARIANTS:
        for step in ("1", "2"):
            stage = summary["stages"][variant][step]
            if stage["status"] != "complete":
                print(f"{variant:7} {step}: {stage['status']} {stage['training_iteration']}/{stage['expected_iterations']} iterations; latest eval {stage['latest_evaluation_iteration']}")
                continue
            keys = ("old_initial10_miou", "new_since_initial_miou", "all_with_background_miou")
            values = [fmt(stage["scores"].get(k)) for k in keys]
            deltas = [fmt(stage["delta_vs_base_pp"].get(k)) for k in keys]
            print(f"{variant:7} {step}    " + "  ".join(values + deltas))
            if step == "2":
                scores = stage["scores"]
                seconds = stage["timing"]["observed_stage_seconds"]
                print(f"  previous15={fmt(scores['previous_foreground_miou'])}; current5={fmt(scores['current_foreground_miou'])}; foreground20={fmt(scores['foreground_all_miou'])}; stage log span={fmt(seconds)}s")
                print(f"  background={fmt(scores['background_iou'])}; initial10+bg={fmt(scores['old_initial_with_background_miou'])}; previous15+bg={fmt(scores['previous_with_background_miou'])}")
    if summary["smoke"]:
        print("Smoke averages exclude unavailable classes; see score_class_counts in summary.json.")
    for warning in summary["warnings"]:
        print("WARNING:", warning)
    if summary["failures"]:
        print("Recorded failures:", json.dumps(summary["failures"], ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", nargs="?", default="runs/fixed_baseline_v1",
                        help="study directory or its results.json, within this project")
    args = parser.parse_args()
    path = Path(args.study)
    study = checked_path(path if path.is_absolute() else ROOT / path)
    if study.name == "results.json":
        study = study.parent
    if not study.is_dir():
        parser.error("study directory does not exist")
    summary = summarize(study)
    output = checked_path(study / "summary.json", output=True)
    content = (json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    # Check link count again on the opened inode, before truncating a prior report.
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o644)
    with os.fdopen(fd, "wb") as handle:
        if os.fstat(handle.fileno()).st_nlink != 1:
            raise ValueError(f"Refusing to modify a hard-linked report: {output}")
        handle.truncate(0)
        handle.write(content)
    display(summary)
    print("Saved:", output)


if __name__ == "__main__":
    main()
