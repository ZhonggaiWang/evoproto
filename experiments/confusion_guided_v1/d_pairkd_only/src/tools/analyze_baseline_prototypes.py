"""Inspect final baseline prototype checkpoints on CPU.

Usage:
    python tools/analyze_baseline_prototypes.py --study runs/fixed_baseline_v1

The resulting JSON describes parameter geometry, not segmentation performance.
Missing final checkpoints remain explicitly incomplete; intermediate training
checkpoints are never substituted for them.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import re
import sys
import tempfile

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("base", "kd", "sep", "kd_sep")
PROTOTYPE_KEY = re.compile(r"decoder\.class_prototypes\.(\d+)\.prototype$")


def checked_output_path(study):
    """Reject output outside the project, symlink ancestors and hardlinked files."""
    path = Path(os.path.abspath(study))
    if not path.is_relative_to(PROJECT_ROOT):
        raise ValueError("The study output directory must be inside this project.")
    current = PROJECT_ROOT
    for component in path.relative_to(PROJECT_ROOT).parts:
        current = current / component
        if current.is_symlink():
            raise ValueError(f"The study output path must not use symlinks: {current}")
        if current.exists() and current.is_mount():
            raise ValueError(f"The study output path must not use nested mounts: {current}")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(PROJECT_ROOT) or not resolved.is_dir():
        raise ValueError("The study must be an existing directory inside this project.")
    output = resolved / "prototype_diagnostics.json"
    if output.is_symlink():
        raise ValueError(f"The output must not be a symlink: {output}")
    if output.exists() and (not output.is_file() or output.stat().st_nlink != 1):
        raise ValueError(f"The output must be an independent regular file: {output}")
    return resolved, output


def load_prototypes(path, step):
    """Load only the prototype parameters into the diagnostic result, on CPU."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state", checkpoint)
    groups = {}
    for key, value in state.items():
        match = PROTOTYPE_KEY.fullmatch(key.removeprefix("module."))
        if match:
            groups[int(match.group(1))] = value.detach().to(device="cpu", dtype=torch.float64)
    if sorted(groups) != list(range(step + 1)):
        raise ValueError(f"Expected prototype groups 0 through {step}, got {sorted(groups)}")
    expected_sizes = [11] + [5] * step
    for index, count in enumerate(expected_sizes):
        group = groups[index]
        if group.ndim != 2 or group.shape[0] != count:
            raise ValueError(f"Group {index} must contain {count} class prototypes, got {tuple(group.shape)}")
        if not torch.isfinite(group).all():
            raise ValueError(f"Group {index} contains nonfinite prototype values")
    dimensions = {group.shape[1] for group in groups.values()}
    if len(dimensions) != 1:
        raise ValueError("All prototype groups must use the same feature dimension")
    raw = torch.cat([groups[index] for index in range(step + 1)], dim=0)
    zero_norm_rows = (raw.norm(dim=1) == 0).nonzero(as_tuple=True)[0].tolist()
    return F.normalize(raw, p=2, dim=1), zero_norm_rows


def pair_statistics(similarities):
    values = similarities.reshape(-1).clamp(-1.0, 1.0)
    if not values.numel():
        return {"pair_count": 0, "squared_relu_cos_mean": None,
                "positive_pair_fraction": None, "mean_cosine": None, "max_cosine": None}
    return {
        "pair_count": values.numel(),
        "squared_relu_cos_mean": values.clamp_min(0).square().mean().item(),
        "positive_pair_fraction": (values > 0).double().mean().item(),
        "mean_cosine": values.mean().item(),
        "max_cosine": values.max().item(),
    }


def unordered_pairs(prototypes):
    indices = torch.triu_indices(prototypes.shape[0], prototypes.shape[0], offset=1)
    similarity = prototypes @ prototypes.t()
    return similarity[indices[0], indices[1]]


def mean_matching_drift(student, teacher):
    if student.shape != teacher.shape:
        raise ValueError("Matching foreground prototypes must have identical shapes")
    if not student.shape[0]:
        return None
    return (1.0 - (student * teacher).sum(dim=1).clamp(-1.0, 1.0)).mean().item()


def stage_geometry(prototypes, step, zero_norm_rows):
    previous_foreground = 0 if step == 0 else 10 + 5 * (step - 1)
    foreground = prototypes[1:]
    old = foreground[:previous_foreground]
    new = foreground[previous_foreground:]
    return {
        "feature_dimension": prototypes.shape[1],
        "foreground_count": foreground.shape[0],
        "previous_foreground_count": old.shape[0],
        "current_foreground_count": new.shape[0],
        "zero_norm_class_rows": zero_norm_rows,
        "foreground": pair_statistics(unordered_pairs(foreground)),
        "old_old": pair_statistics(unordered_pairs(old)),
        "old_new": pair_statistics(old @ new.t()),
        "new_new": pair_statistics(unordered_pairs(new)),
        "teacher_old_foreground_mean_one_minus_cosine": None,
        "shared_initial_10_mean_one_minus_cosine": None,
    }


def analyze_study(study):
    manifest = json.loads((study / "study.json").read_text())
    if manifest.get("task") != "10-5":
        raise ValueError("This diagnostic is scoped to the VOC 10-5 baseline study.")
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "study": str(study), "task": "10-5", "smoke": bool(manifest.get("smoke")),
        "scope": "Final-checkpoint prototype geometry only; this is not a performance result.",
        "device": "cpu", "geometry_dtype": "float64", "margin": 0.0,
        "background_excluded": True,
        "old_new_definition": "Previous learned foreground classes versus classes introduced in this stage; step 0 has 0 old and 10 initial classes.",
        "shared_initial_drift_definition": "Stage 2 foreground rows 1 through 10 versus the common step 0 prototypes.",
        "expected_final_checkpoints": 9, "analyzed_final_checkpoints": 0,
        "incomplete_checkpoints": [], "shared": {}, "variants": {},
    }
    prototypes_by_stage = {}
    stages = [("shared", 0)] + [(variant, step) for variant in VARIANTS for step in (1, 2)]
    for variant, step in stages:
        path = study / variant / "10-5" / f"step{step}" / "checkpoints" / "model_final.pth"
        entry = {"checkpoint": str(path), "step": step}
        if not path.is_file():
            entry.update(status="incomplete", reason="Final checkpoint is not available; training may not be finished.")
        else:
            try:
                prototypes, zero_rows = load_prototypes(path, step)
                entry.update(status="complete", **stage_geometry(prototypes, step, zero_rows))
                prototypes_by_stage[(variant, step)] = prototypes
                report["analyzed_final_checkpoints"] += 1
            except (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError, EOFError, pickle.UnpicklingError) as error:
                entry.update(status="unreadable_final_checkpoint", reason=str(error))
        if entry["status"] != "complete":
            report["incomplete_checkpoints"].append({"variant": variant, "step": step,
                                                     "checkpoint": str(path), "reason": entry["reason"]})
        if variant == "shared":
            report["shared"][str(step)] = entry
        else:
            report["variants"].setdefault(variant, {})[str(step)] = entry

    for variant in VARIANTS:
        for step in (1, 2):
            student = prototypes_by_stage.get((variant, step))
            if student is None:
                continue
            entry = report["variants"][variant][str(step)]
            teacher_key = ("shared", 0) if step == 1 else (variant, step - 1)
            teacher = prototypes_by_stage.get(teacher_key)
            if teacher is None:
                entry["teacher_alignment_status"] = "unavailable_previous_final_checkpoint"
            else:
                entry["teacher_alignment_status"] = "complete"
                entry["teacher_old_foreground_mean_one_minus_cosine"] = mean_matching_drift(
                    student[1:teacher.shape[0]], teacher[1:]
                )
            if step == 2:
                shared = prototypes_by_stage.get(("shared", 0))
                entry["shared_initial_alignment_status"] = "complete" if shared is not None else "unavailable_shared_final_checkpoint"
                if shared is not None:
                    entry["shared_initial_10_mean_one_minus_cosine"] = mean_matching_drift(
                        student[1:11], shared[1:11]
                    )
    report["status"] = "complete" if not report["incomplete_checkpoints"] else "incomplete"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    study_arg = args.study if args.study.is_absolute() else PROJECT_ROOT / args.study
    try:
        study, output = checked_output_path(study_arg)
        report = analyze_study(study)
        # Write an independent file and atomically replace this diagnostic only.
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=".prototype_diagnostics-",
                                         suffix=".json", dir=study, delete=False) as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
            temporary_output = Path(handle.name)
        checked_output_path(study)
        os.replace(temporary_output, output)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(f"CPU prototype diagnostics: {report['status']}; "
          f"{report['analyzed_final_checkpoints']}/{report['expected_final_checkpoints']} final checkpoints.")
    print("Geometry diagnostics only; these values do not establish segmentation effectiveness.")
    if report["incomplete_checkpoints"]:
        print("Some final checkpoints are unavailable or unreadable; only existing complete finals were analyzed.")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
