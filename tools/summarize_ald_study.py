"""Audit and summarize the matched four-arm ALD study, using CPU only.

Usage: python -B tools/summarize_ald_study.py --study runs/ald_fusion_v1
Incomplete studies require --partial; only audited final stages enter comparisons.
Writes summary.json and scientific comparison.png/.pdf inside the study directory.
"""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tarfile
import uuid

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

ROOT = Path(__file__).resolve().parents[1]
AUTHORIZED_ROOT = Path("/ML-vePFS/infra_rd/kun/others/wzg").resolve()
VARIANTS = ("off", "legacy", "preserve_rejected", "preserve_background")
DISPLAY = {"off": "OFF", "legacy": "Legacy ALD",
           "preserve_rejected": "Preserve all rejected",
           "preserve_background": "Preserve teacher-BG rejected"}


def checked_path(path):
    """Guard all outputs, including cache paths, against indirect outside writes."""
    path = Path(os.path.abspath(path))
    if not ROOT.is_relative_to(AUTHORIZED_ROOT) or not path.is_relative_to(ROOT):
        raise ValueError(f"Path must be inside the authorized project: {path}")
    current = ROOT
    for part in path.relative_to(ROOT).parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"Symlink output component: {current}")
        if current.exists() and current.is_mount():
            raise ValueError(f"Nested mount output component: {current}")
    # is_mount does not detect every bind mount on the same device.
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        fields = line.split()
        mounted = Path(re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), fields[4]))
        if (mounted.is_relative_to(AUTHORIZED_ROOT)
                and (mounted.is_relative_to(ROOT) or ROOT.is_relative_to(mounted))
                and path.is_relative_to(mounted)):
            raise ValueError(f"Output enters a project or ancestor mount: {mounted}")
    resolved = path.resolve()
    if resolved != path or not resolved.is_relative_to(ROOT):
        raise ValueError(f"Noncanonical output path: {path}")
    if path.exists() and path.is_file() and path.stat().st_nlink != 1:
        raise ValueError(f"Hardlinked output file: {path}")
    return path


def read_json(path):
    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON constant {value}: {path}")
    return json.loads(Path(path).read_text(), parse_constant=reject_constant)


def read_jsonl(path):
    return [json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(
        ValueError(f"Nonfinite JSON constant {x}: {path}")))
        for line in Path(path).read_text().splitlines() if line.strip()]


def sha256(path):
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError(f"Input changed during audit: {path}")
    return digest.hexdigest()


def literal_assignment(path, name, source_text=None):
    """Read frozen constants without importing torch or the training modules."""
    tree = ast.parse(Path(path).read_text() if source_text is None else source_text)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"Missing {name} in {path}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def audit_sources(study, manifest, allow_drift):
    # Runner stores code paths relative to ROOT, but list paths as absolute names.
    recorded = {}
    for name, digest in {**manifest["code_sha256"], **manifest["dataset_lists_sha256"]}.items():
        path = ROOT / name
        require(path.is_relative_to(ROOT), f"Recorded source/list path is outside the project: {path}")
        relative = str(path.relative_to(ROOT))
        require(relative not in recorded or recorded[relative] == digest,
                f"Conflicting recorded source/list hashes: {relative}")
        recorded[relative] = digest
    drift = {}
    for relative, expected in recorded.items():
        path = ROOT / relative
        current = sha256(path) if path.is_file() else None
        if current != expected:
            drift[relative] = {"recorded_sha256": expected, "current_sha256": current}
    archive, receipt = study / "source_snapshot.tar.gz", study / "source_snapshot.json"
    texts, archive_verified = {}, False
    archive_sha = None
    if archive.exists() or receipt.exists():
        require(archive.is_file() and receipt.is_file(), "Incomplete source snapshot archive/receipt")
        snapshot = read_json(receipt)
        archive_sha = sha256(archive)
        require(snapshot.get("archive_sha256") == archive_sha and
                snapshot.get("archive_bytes") == archive.stat().st_size,
                "Source snapshot archive digest/size mismatch")
        with tarfile.open(archive, "r:gz") as handle:
            members = handle.getmembers()
            require(len({member.name for member in members}) == len(members) and
                    all(member.isfile() and not Path(member.name).is_absolute() and
                        ".." not in Path(member.name).parts for member in members),
                    "Source archive must contain unique ordinary relative files")
            archive_hashes = {}
            for member in members:
                content = handle.extractfile(member).read()  # Read bytes only; never extract to disk.
                archive_hashes[member.name] = hashlib.sha256(content).hexdigest()
                if member.name in ("datasets/voc.py", "utils/ald_stats.py"):
                    texts[member.name] = content.decode("utf-8")
            require(archive_hashes == snapshot.get("files"), "Source archive files differ from receipt")
            require(all(archive_hashes.get(name) == digest for name, digest in recorded.items()),
                    "Source archive does not reproduce the study's recorded code/data hashes")
        archive_verified = True
    require(not drift or archive_verified or allow_drift,
            "Current source differs from recorded training source and no verified archive is available: " +
            ", ".join(drift) + "; use --allow-current-source-drift only to explicitly disclose this provenance gap")
    verified = archive_verified or not drift
    return {"recorded_source_verified": verified, "archive_verified": archive_verified,
            "archive": str(archive) if archive_verified else None, "archive_sha256": archive_sha,
            "current_source_matches_recorded": not drift, "current_source_diff": drift,
            "allow_current_source_drift_requested": allow_drift,
            "warning": ("Recorded source lacks a verified archive for changed files; current files are not the historical source"
                        if not verified else None)}, texts


def group_mean(values, names, ids):
    selected = [values[names[index]] for index in ids if finite(values[names[index]])]
    return {"miou": sum(selected) / len(selected) if selected else None,
            "class_ids": list(ids), "finite_classes": len(selected), "total_classes": len(ids)}


def metric_groups(record, step, names):
    total, previous = 10 + 5 * step, 5 + 5 * step
    values = record.get("class_iou")
    require(isinstance(values, dict) and list(values) == names[:total + 1],
            f"Step {step} class_iou must follow voc.class_list with {total + 1} entries")
    require(all(value is None or (finite(value) and 0 <= value <= 100)
                for value in values.values()), "class_iou values must be null or finite percentages")
    definitions = {"previous_fg": list(range(1, previous + 1)),
                   "current5_fg": list(range(previous + 1, total + 1)),
                   "all_with_bg": list(range(total + 1)),
                   "all_fg": list(range(1, total + 1)),
                   "initial10_fg": list(range(1, 11)),
                   "cumulative_since_initial_fg": list(range(11, total + 1)),
                   "initial10_with_bg": list(range(11)),
                   "previous_with_bg": list(range(previous + 1)), "background": [0]}
    groups = {name: group_mean(values, names, ids) for name, ids in definitions.items()}
    for raw_key, group in {"old_miou": "initial10_fg", "new_miou": "cumulative_since_initial_fg",
                           "all_miou": "all_with_bg", "foreground_miou": "all_fg"}.items():
        expected, actual = groups[group]["miou"], record.get(raw_key)
        require(actual is None if expected is None else finite(actual) and
                math.isclose(actual, expected, abs_tol=1e-8),
                f"Raw {raw_key} differs from class_iou-derived {group}")
    return groups


def count_identities(counts, mode):
    c = counts
    require(c["valid_box_pixels"] == c["final_valid_pixels"] + c["final_ignore_pixels"],
            "valid pixels != final supervised + final ignore")
    require(c["final_valid_pixels"] == sum(c[key] for key in
            ("final_bg_pixels", "final_old_pixels", "final_new_pixels")),
            "final supervised != BG + old + new")
    r = c["rejected_new_cam_pixels"]
    require(r == c["rejected_teacher_old_pixels"] + c["rejected_teacher_bg_pixels"],
            "Rejected pixels have an invalid teacher class")
    require(r == sum(c[key] for key in ("rejected_final_old_pixels", "rejected_final_bg_pixels",
                                      "rejected_final_ignore_pixels")),
            "Rejected pixels have an invalid final class")
    require(c["retained_ignore_pixels"] == c["rejected_final_ignore_pixels"],
            "Retained ignore != rejected final ignore")
    require(c["padding_labeled_pixels"] == 0, "Padding received pixel supervision")
    require(c["before_new_cam_pixels"] == c["after_new_cam_pixels"] + r,
            "New CAM pixel count is inconsistent with rejection")
    require(c["final_new_pixels"] == c["after_new_cam_pixels"], "Final new supervision differs from accepted CAM")
    require(c["final_ignore_pixels"] == c["retained_ignore_pixels"],
            "Final ignore contains pixels outside the selected rejection rule")
    require(c["valid_box_pixels"] <= c["total_pixels"], "Valid box exceeds image pixels")
    require(c["before_new_cam_pixels"] <= c["valid_box_pixels"], "New CAM exceeds valid image pixels")
    require(c["fallback_images"] == c["fallback_old_images"] + c["fallback_new_images"] <= c["batch_images"],
            "Fallback image count/partition mismatch")
    require(c["gate_rejected_old_classes"] <= c["positive_old_classes"] and
            c["gate_rejected_new_classes"] <= c["positive_new_classes"], "Gate rejection exceeds positive classes")
    if mode == "off":
        require(r == 0 and c["retained_ignore_pixels"] == 0, "OFF rejected or ignored pixels")
        require(sum(c[key] for key in ("gate_rejected_old_classes", "gate_rejected_new_classes",
                                      "fallback_images")) == 0, "OFF has ALD gate diagnostics")
    elif mode == "legacy":
        require(c["retained_ignore_pixels"] == 0, "Legacy fusion retained ignore")
    elif mode == "preserve_rejected":
        require(c["retained_ignore_pixels"] == r, "Preserve-all failed to retain all rejected pixels")
    elif mode == "preserve_background":
        require(c["retained_ignore_pixels"] == c["rejected_teacher_bg_pixels"],
                "Preserve-BG failed to retain exactly teacher-BG rejected pixels")
    else:
        raise ValueError(f"Unexpected mode: {mode}")
    require(c["rejected_final_old_pixels"] == (0 if mode == "preserve_rejected" else c["rejected_teacher_old_pixels"]),
            "Rejected teacher-old pixels have the wrong final policy")
    require(c["rejected_final_bg_pixels"] == (c["rejected_teacher_bg_pixels"] if mode == "legacy" else 0),
            "Rejected teacher-BG pixels have the wrong final policy")


def ratios(counts):
    definitions = {"rejected_fraction_of_new_cam": ("rejected_new_cam_pixels", "before_new_cam_pixels"),
                   "rejected_fraction_of_valid_image": ("rejected_new_cam_pixels", "valid_box_pixels"),
                   "supervised_fraction_of_valid_image": ("final_valid_pixels", "valid_box_pixels"),
                   "retained_ignore_fraction_of_rejected": ("retained_ignore_pixels", "rejected_new_cam_pixels")}
    return {name: counts[num] / counts[den] if counts[den] else None
            for name, (num, den) in definitions.items()}


def aggregate_counts(path, config, step, mode, count_keys):
    records = read_jsonl(path)
    final, interval, warmup = config["max_iters"], config["log_iters"], config["loss_warmup_iters"]
    expected_ends = list(range(interval, final + 1, interval))
    if not expected_ends or expected_ends[-1] != final:
        expected_ends.append(final)
    require([r.get("iteration") for r in records] == expected_ends,
            f"Missing/duplicate diagnostic intervals: {path}")
    groups = {name: {"counts": dict.fromkeys(count_keys, 0), "iterations": 0,
                     "interval_records": 0} for name in ("all", "warmup", "active")}
    start = 0
    for record in records:
        end = record["iteration"]
        require(not (start < warmup < end), "Diagnostic interval crosses the loss warmup boundary")
        active = end > warmup
        require(record.get("step") == step and record.get("ald_mode") == mode and
                type(record.get("segmentation_loss_active")) is bool and
                record["segmentation_loss_active"] == active and
                record.get("interval_batches") == end - start, "Diagnostic interval metadata mismatch")
        counts = {key: record.get(key) for key in count_keys}
        require(all(type(value) is int and value >= 0 for value in counts.values()),
                "Diagnostics require nonnegative integer counts")
        require(counts["batch_images"] == (end - start) * config["spg"],
                "Diagnostic image count differs from single-GPU batch size")
        require(counts["total_pixels"] == counts["batch_images"] * config["crop_size"] ** 2,
                "Diagnostic pixel count differs from the configured crop size")
        count_identities(counts, mode)
        for key, value in ratios(counts).items():
            require(record.get(key) is None if value is None else finite(record.get(key)) and
                    math.isclose(record[key], value, abs_tol=1e-12), "Logged diagnostic ratio mismatch")
        for name in ("all", "active" if active else "warmup"):
            group = groups[name]
            group["iterations"] += end - start
            group["interval_records"] += 1
            for key in count_keys:
                group["counts"][key] += counts[key]
        start = end
    for group in groups.values():
        count_identities(group["counts"], mode)
        group["ratios_from_summed_counts"] = ratios(group["counts"])
    return {"aggregation": "sum integer COUNT_KEYS, then compute ratios; never average interval ratios",
            "loss_warmup_iterations": warmup, "sha256": sha256(path), **groups}


def audit_stage(study, manifest, mode, step, previous, count_keys, names):
    directory = study / mode / "10-5" / f"step{step}"
    artifacts = {name: directory / name for name in
                 ("config.json", "inputs.json", "completion.json", "metrics.jsonl", "ald_metrics.jsonl",
                  "checkpoints/model_final.pth")}
    missing = [name for name, path in artifacts.items() if not path.is_file()]
    if missing:
        require(not artifacts["completion.json"].exists(),
                f"Published completion has missing artifacts at {directory}: {missing}")
        return {"complete": False, "directory": str(directory), "missing_artifacts": missing}
    require(previous is not None, f"Completed step {step} has no audited predecessor: {directory}")
    config, inputs, completion = (read_json(artifacts[name]) for name in
                                 ("config.json", "inputs.json", "completion.json"))
    expected = dict(manifest["common_training_config"], step=step, ald_mode=mode, ald=mode != "off",
                    prev_checkpoint=str(previous.resolve()), work_dir=str(directory),
                    ckpt_dir=str(directory / "checkpoints"), pred_dir=str(directory / "predictions"))
    require(all(config.get(key) == value for key, value in expected.items()),
            f"Configuration mismatch: {directory}")
    require(inputs.get("expected_config") == expected and inputs.get("previous_checkpoint") == str(previous) and
            inputs.get("previous_sha256") == sha256(previous) and
            inputs.get("pretrained_sha256") == manifest["shared_initialization"]["pretrained_sha256"] and
            inputs.get("shared_initialization") == manifest["shared_initialization"] and
            inputs.get("code_sha256") == manifest["code_sha256"] and
            inputs.get("dataset_lists_sha256") == manifest["dataset_lists_sha256"] and
            inputs.get("gpu") == manifest["gpu_mapping"][mode], f"Input provenance mismatch: {directory}")
    records = read_jsonl(artifacts["metrics.jsonl"])
    expected_evaluations = list(range(config["eval_iters"], config["max_iters"] + 1, config["eval_iters"]))
    if not expected_evaluations or expected_evaluations[-1] != config["max_iters"]:
        expected_evaluations.append(config["max_iters"])
    require([record.get("iteration") for record in records] == expected_evaluations,
            f"Evaluation schedule mismatch: {directory}")
    metric_keys = ("step", "ald_mode", "ald", "seed", "confusion_reweight", "w_proto_kd", "w_proto_sep",
                   "w_proto_seg", "proto_margin")
    for record in records:
        require(all(record.get(key) == expected[key] for key in metric_keys) and
                record.get("pixel_padding_ignored") is True, f"Metric configuration mismatch: {directory}")
        metric_groups(record, step, names)
    final = records[-1]
    require(completion.get("returncode") == 0 and completion.get("iteration") == config["max_iters"] and
            completion.get("command") == inputs.get("command") and completion.get("config") == config and
            completion.get("final_metrics") == final, f"Completion mismatch: {directory}")
    digests = {"inputs_sha256": sha256(artifacts["inputs.json"]),
               "config_sha256": sha256(artifacts["config.json"]),
               "checkpoint_sha256": sha256(artifacts["checkpoints/model_final.pth"]),
               "metrics_sha256": sha256(artifacts["metrics.jsonl"])}
    require(all(completion.get(key) == value for key, value in digests.items()),
            f"Completion file/checkpoint digest mismatch: {directory}")
    diagnostics = aggregate_counts(artifacts["ald_metrics.jsonl"], config, step, mode, count_keys)
    if "ald_metrics_sha256" in completion:
        require(completion["ald_metrics_sha256"] == diagnostics["sha256"] and
                completion.get("final_ald_metrics") == read_jsonl(artifacts["ald_metrics.jsonl"])[-1],
                f"Completion ALD coverage digest/final record mismatch: {directory}")
        diagnostics["completion_bound"] = True
    else:
        require(manifest["smoke"], f"Formal completion lacks ALD coverage digest: {directory}")
        verification = study / "supervision-verification.json"
        require(verification.is_file(), "Legacy smoke coverage needs its independent verification receipt")
        rows = read_json(verification).get("stages", [])
        row = [r for r in rows if r.get("stage") == f"{mode}/10-5/step{step}"]
        require(len(row) == 1 and row[0].get("completion_sha256") == sha256(artifacts["completion.json"]) and
                row[0].get("ald_metrics_sha256") == diagnostics["sha256"],
                f"Independent smoke coverage receipt mismatch: {directory}")
        diagnostics["completion_bound"] = False
        diagnostics["independent_receipt"] = str(verification)
    started, finished = (datetime.fromisoformat(completion[key]) for key in ("started_utc", "finished_utc"))
    duration = (finished - started).total_seconds()
    require(started.tzinfo is not None and finished.tzinfo is not None and duration >= 0,
            f"Invalid stage UTC times: {directory}")
    return {"complete": True, "directory": str(directory), "iteration": final["iteration"],
            "gpu": inputs["gpu"], "started_utc": completion["started_utc"],
            "finished_utc": completion["finished_utc"], "elapsed_seconds": duration,
            "checkpoint": str(artifacts["checkpoints/model_final.pth"]), "provenance_sha256": digests,
            "raw_final_metrics": final, "metric_groups": metric_groups(final, step, names),
            "diagnostics": diagnostics}


def audit_study(study, partial=False, allow_drift=False):
    manifest = read_json(study / "study.json")
    require(manifest.get("schema_version") == 1 and manifest.get("study") == "matched ALD fusion" and
            manifest.get("task") == "10-5" and manifest.get("variants") == list(VARIANTS),
            "Expected the matched four-arm VOC 10-5 ALD study")
    smoke = manifest.get("smoke")
    require(type(smoke) is bool and manifest.get("incremental_iters") == (4 if smoke else 8000),
            "Unexpected smoke/formal training budget")
    common = manifest["common_training_config"]
    require(common.get("max_iters") == manifest["incremental_iters"] and common.get("w_proto_kd") == 0 and
            common.get("w_proto_sep") == 0 and common.get("w_proto_seg") == 0.1 and
            common.get("seed") == manifest["seed"] and common.get("confusion_reweight") is False and
            common.get("loss_warmup_iters") == (1 if smoke else 2000) and
            common.get("log_iters") == (1 if smoke else 50) and common.get("spg") == 8 and
            manifest.get("global_batch_size") == 8, "Unexpected matched training protocol")
    require(set(manifest["gpu_mapping"]) == set(VARIANTS) and
            set(manifest["gpu_mapping"].values()) == {"0", "1", "2", "3"}, "Unexpected GPU assignment")
    source = manifest["shared_initialization"]
    require(source.get("initialization_seed") == 0 and source.get("step0_iterations") == 20000,
            "Unexpected shared initial-stage provenance")
    require(sha256(source["checkpoint"]) == source["checkpoint_sha256"] and
            Path(source["checkpoint"]).stat().st_size == source["checkpoint_bytes"] and
            sha256(source["pretrained_checkpoint"]) == source["pretrained_sha256"],
            "Shared checkpoint/pretraining digest mismatch")
    provenance, archived_text = audit_sources(study, manifest, allow_drift)
    for path, digest in source["source_manifests_sha256"].items():
        require(sha256(path) == digest, f"Historical initialization manifest changed: {path}")
    names = literal_assignment(ROOT / "datasets/voc.py", "class_list", archived_text.get("datasets/voc.py"))
    count_keys = literal_assignment(ROOT / "utils/ald_stats.py", "COUNT_KEYS", archived_text.get("utils/ald_stats.py"))
    require(len(names) == 21 and len(set(names)) == 21 and names[0] == "_background_",
            "Unexpected VOC class list")
    status_path, results_path = study / "status.json", study / "results.json"
    status = read_json(status_path) if status_path.exists() else None
    results = read_json(results_path) if results_path.exists() else None
    if results is not None:
        require(isinstance(results, dict) and set(results) == set(VARIANTS) and all(
                isinstance(results[mode], dict) and set(results[mode]) == {"1", "2"} for mode in VARIANTS),
                "Published results must contain exactly eight final-stage records")
    stages, completed = {}, 0
    for mode in VARIANTS:
        stages[mode], previous = {}, Path(source["checkpoint"])
        for step in (1, 2):
            stage = audit_stage(study, manifest, mode, step, previous, count_keys, names)
            stages[mode][str(step)] = stage
            if stage["complete"]:
                completed += 1
                if results is not None:
                    require(results[mode][str(step)] == stage["raw_final_metrics"],
                            f"Published results differ from audited final metrics: {mode}/{step}")
                previous = Path(stage["checkpoint"])
            else:
                previous = None
                require(results is None, "Published results contain an unauditable stage")
    published_complete = bool(status and status.get("status") == "complete")
    if published_complete:
        require(completed == 8 and results is not None and status.get("expected_stages") == 8 and
                status.get("completed_stages") == 8 and status.get("all_stage_returncodes") == 0 and
                status.get("results_sha256") == sha256(results_path), "Runner completion/status mismatch")
    complete = completed == 8 and published_complete and results is not None
    require(partial or complete,
            f"Study incomplete: {completed}/8 audited final stages; use --partial to report only completed stages")
    deltas = {}
    for step in ("1", "2"):
        baseline = stages["off"][step]
        deltas[step] = {}
        if baseline["complete"]:
            for mode in VARIANTS[1:]:
                stage = stages[mode][step]
                if stage["complete"]:
                    deltas[step][mode] = {}
                    for name, group in stage["metric_groups"].items():
                        a, b = group["miou"], baseline["metric_groups"][name]["miou"]
                        deltas[step][mode][name] = a - b if a is not None and b is not None else None
    times = [stage for records in stages.values() for stage in records.values() if stage["complete"]]
    wall_seconds = ((max(datetime.fromisoformat(s["finished_utc"]) for s in times) -
                     min(datetime.fromisoformat(s["started_utc"]) for s in times)).total_seconds() if times else None)
    return {"schema_version": 1, "complete": complete, "partial": not complete,
            "smoke": smoke, "performance_evidence": not smoke and complete and provenance["recorded_source_verified"],
            "generated_utc": datetime.now(timezone.utc).isoformat(), "study": str(study),
            "completed_stages": completed, "expected_stages": 8, "runner_status": status,
            "seed": manifest["seed"], "seed_scope": manifest["seed_scope"],
            "gpu_mapping": manifest["gpu_mapping"], "shared_initialization": source, "provenance": provenance,
            "class_list": names, "count_keys": list(count_keys),
            "comparison_control": "new matched off; historical fixed BASE is reference only",
            "metric_scope": {"step1": "previous foreground 1..10; current 11..15; all+BG 0..15",
                             "step2": "previous foreground 1..15; current 16..20; all+BG 0..20",
                             "raw_old_miou": "initial foreground 1..10 at both stages",
                             "raw_new_miou": "cumulative since initial: 11..15 at step1, 11..20 at step2",
                             "null_iou": "excluded from group means; finite and total class counts reported"},
            "limitations": ["Single incremental seed; no error bars or significance claim",
                            "Stage validation image sets and valid GT label ranges differ; cross-stage drops are not a forgetting measurement",
                            "Warmup diagnostics count constructed labels before segmentation loss is active",
                            "ALD changes CAM filtering/fusion; old image-label hard targets, teacher supervision and common prototype-pixel loss remain",
                            "Smoke is a pipeline check only, with truncated training and eight validation images"],
            "timing": {"completed_stage_seconds_sum": sum(s["elapsed_seconds"] for s in times),
                       "audited_stage_window_seconds": wall_seconds,
                       "scope": "incremental processes including validation; parallel GPU times are not wall-clock sums"},
            "stages": stages, "deltas_vs_matched_off_pp": deltas}


def write_summary(path, summary):
    path = checked_path(path)
    temporary = checked_path(path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp"))
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, checked_path(path))


def plot_comparison(study, summary):
    cache = checked_path(ROOT / ".runtime/cache/matplotlib")
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 7.2))
    figure.suptitle(f"VOC 10-5 | matched ALD fusion | incremental seed {summary['seed']}", y=0.98, fontsize=15)
    figure.text(.5, .935, "KD = SEP = 0; common prototype-pixel weight 0.1; shared step 0; single seed, no error bars",
                ha="center", fontsize=10)
    banner = "SMOKE — NOT PERFORMANCE" if summary["smoke"] else "INCOMPLETE — FINAL STAGES ONLY" if summary["partial"] else ""
    if summary["partial"] and summary["smoke"]:
        banner += " | INCOMPLETE"
    if banner:
        figure.text(.5, .88, banner, ha="center", color="#b91c1c", weight="bold", fontsize=16)
    deltas = summary["deltas_vs_matched_off_pp"]
    points = [(d["previous_fg"], d["current5_fg"]) for stage in deltas.values() for d in stage.values()
              if finite(d["previous_fg"]) and finite(d["current5_fg"])]
    def limits(values):
        low, high = min([0.] + values), max([0.] + values)
        pad = .3 * max(high - low, .15)
        return low - pad, high + pad
    xlim, ylim = limits([p[0] for p in points]), limits([p[1] for p in points])
    styles = {"legacy": ("#2563eb", "o", 38), "preserve_rejected": ("#059669", "s", -2),
              "preserve_background": ("#c026d3", "D", -38)}
    for step, axis in enumerate(axes, 1):
        axis.axhline(0, color="#6b7280", lw=.9, ls="--")
        axis.axvline(0, color="#6b7280", lw=.9, ls="--")
        axis.grid(alpha=.16)
        axis.set_xlim(xlim)
        axis.set_ylim(ylim)
        axis.set_title(f"Step {step}: previous {10 if step == 1 else 15} / current 5", pad=12)
        axis.set_xlabel("Δ previous foreground mIoU vs matched OFF (pp)")
        axis.set_ylabel("Δ current 5 foreground mIoU vs matched OFF (pp)")
        baseline = summary["stages"]["off"][str(step)]
        if baseline["complete"]:
            axis.scatter([0], [0], marker="+", s=150, color="#111827", linewidths=2, zorder=5)
            axis.annotate("OFF", (0, 0), xytext=(8, -16), textcoords="offset points", weight="bold")
        else:
            axis.text(.5, .5, "Matched OFF final stage unavailable", ha="center", transform=axis.transAxes)
        missing = []
        for mode, (color, marker, offset) in styles.items():
            record = deltas[str(step)].get(mode)
            if record is None or not finite(record["previous_fg"]) or not finite(record["current5_fg"]):
                missing.append(DISPLAY[mode])
                continue
            x, y = record["previous_fg"], record["current5_fg"]
            axis.scatter([x], [y], color=color, marker=marker, s=85, edgecolor="white", linewidth=.7, zorder=4)
            side = x >= sum(xlim) / 2
            all_delta = record["all_with_bg"]
            annotation = f"{DISPLAY[mode]}\nΔall+BG = {all_delta:+.2f} pp" if finite(all_delta) else DISPLAY[mode]
            axis.annotate(annotation, (x, y), xytext=(-10 if side else 10, offset),
                          textcoords="offset points", ha="right" if side else "left", va="center",
                          fontsize=8.5, color=color, annotation_clip=False,
                          arrowprops={"arrowstyle": "-", "color": color, "alpha": .55, "lw": .7})
        if missing:
            axis.text(.02, .02, "No final comparison: " + ", ".join(missing), transform=axis.transAxes,
                      fontsize=8, color="#6b7280", wrap=True)
    figure.text(.5, .055, "Common class-group definitions and axis limits; null IoUs excluded. Within-stage comparisons only.",
                ha="center", fontsize=9)
    figure.text(.5, .028, "Shared seed-0 initialization; this OFF includes the common padding correction. Historical BASE is reference only.",
                ha="center", fontsize=9)
    figure.subplots_adjust(left=.08, right=.97, bottom=.17, top=.77, wspace=.29)
    for extension in ("png", "pdf"):
        path = checked_path(study / f"comparison.{extension}")
        temporary = checked_path(path.with_name(f".{path.stem}.{uuid.uuid4().hex}.{extension}"))
        figure.savefig(temporary, dpi=180, format=extension, facecolor="white")
        os.replace(temporary, checked_path(path))
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default="runs/ald_fusion_v1")
    parser.add_argument("--partial", action="store_true", help="Explicitly report incomplete studies; compare audited final stages only")
    parser.add_argument("--allow-current-source-drift", action="store_true",
                        help="Explicitly disclose changed/missing current sources when no verified historical archive is available")
    args = parser.parse_args()
    try:
        study = checked_path(ROOT / args.study)
        require(study.is_dir() and study.is_relative_to(ROOT / "runs") and
                not study.is_relative_to(ROOT / "runs/fixed_baseline_v1"), "Expected a separate project-local ALD study")
        # Preflight every final destination before auditing or importing plotting code.
        for name in ("summary.json", "comparison.png", "comparison.pdf"):
            checked_path(study / name)
        summary = audit_study(study, partial=args.partial, allow_drift=args.allow_current_source_drift)
        plot_comparison(study, summary)
        write_summary(study / "summary.json", summary)
        print(f"{'COMPLETE' if summary['complete'] else 'INCOMPLETE'}: {summary['completed_stages']}/8 audited final stages")
        print("SMOKE — NOT PERFORMANCE" if summary["smoke"] else "Single-seed results; see summary metric scope and limitations.")
        if summary["provenance"]["current_source_diff"]:
            print("Current source differs: " + ", ".join(summary["provenance"]["current_source_diff"]))
        if summary["provenance"]["warning"]:
            print("PROVENANCE GAP: " + summary["provenance"]["warning"])
        print(study / "summary.json")
    except (ValueError, KeyError, FileNotFoundError) as error:
        parser.exit(2, f"ALD summary error: {error}\n")


if __name__ == "__main__":
    main()
