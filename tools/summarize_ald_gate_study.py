"""CPU audit and scientific summary of the isolated two-arm ALD gate study.

Usage: python -B tools/summarize_ald_gate_study.py --study runs/ald_gate_new_v1
Use --partial only to explicitly report unfinished final stages. Writes project-
local summary.json and comparison.png/.pdf; never launches training or uses GPUs.
"""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tarfile
import uuid

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
ARMS = ("legacy", "new_fallback")
STATS_SOURCE = "experiments/ald_gate_new_v1/ald_stats.py"
EXTRA_KEYS = ("new_rescue_images", "new_rescued_class_occurrences", "direct_threshold_new_classes")

# Reuse only CPU/read-only audit primitives, not the primary study's protocol.
_spec = importlib.util.spec_from_file_location("_ald_gate_summary_primary",
                                             ROOT / "tools/summarize_ald_study.py")
primary = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(primary)
require, finite = primary.require, primary.finite
read_json, read_jsonl, sha256 = primary.read_json, primary.read_jsonl, primary.sha256


def checked_path(path):
    # The shared guard rejects links, hardlinks and project/authorized-ancestor
    # mounts, including same-device bind mounts recorded in mountinfo.
    path = primary.checked_path(path)
    if path.exists() and not (path.is_file() or path.is_dir()):
        raise ValueError(f"Output is not an ordinary file/directory: {path}")
    return path


def source_constants(study, manifest):
    provenance, texts = primary.audit_sources(study, manifest, allow_drift=False)
    archive = study / "source_snapshot.tar.gz"
    if provenance["archive_verified"]:
        receipt = read_json(study / "source_snapshot.json")
        if "training_fingerprint" in receipt:
            require(receipt["training_fingerprint"] == manifest["code_sha256"],
                    "Archive receipt training fingerprint differs from the gate manifest")
        with tarfile.open(archive, "r:gz") as handle:
            texts[STATS_SOURCE] = handle.extractfile(STATS_SOURCE).read().decode("utf-8")
        require(sha256(archive) == provenance["archive_sha256"], "Source archive changed during audit")
    else:
        require(sha256(ROOT / STATS_SOURCE) == manifest["code_sha256"].get(STATS_SOURCE),
                "Unarchived isolated counter source differs from recorded training source")
    names = primary.literal_assignment(ROOT / "datasets/voc.py", "class_list", texts.get("datasets/voc.py"))
    old_keys = primary.literal_assignment(ROOT / "utils/ald_stats.py", "COUNT_KEYS", texts.get("utils/ald_stats.py"))
    count_keys = primary.literal_assignment(ROOT / STATS_SOURCE, "COUNT_KEYS", texts.get(STATS_SOURCE))
    require(len(names) == len(set(names)) == 21 and names[0] == "_background_", "Unexpected VOC class names")
    require(len(old_keys) == 25 and len(count_keys) == 28 and count_keys[:25] == old_keys and
            count_keys[25:] == EXTRA_KEYS, "Expected the isolated 25+3 counter schema")
    provenance["counter_source"] = STATS_SOURCE
    provenance["counter_schema"] = "28 integers; original fallback ownership precedes NEW rescue"
    return provenance, names, count_keys


def extra_identities(counts, arm):
    rescued = counts["new_rescue_images"]
    require(rescued == counts["new_rescued_class_occurrences"] <= counts["batch_images"],
            "NEW rescue must add exactly one class per rescued image")
    kept = counts["positive_new_classes"] - counts["gate_rejected_new_classes"]
    require(kept == counts["direct_threshold_new_classes"] + counts["fallback_new_images"] + rescued,
            "Kept NEW != direct threshold + original fallback NEW + NEW rescue")
    require(counts["direct_threshold_new_classes"] <= counts["positive_new_classes"],
            "Direct NEW passes exceed positive NEW occurrences")
    require(arm != "legacy" or rescued == 0, "Legacy control performed NEW rescue")


def all_ratios(counts):
    result = primary.ratios(counts)
    pairs = {"new_rescue_fraction_of_images": ("new_rescue_images", "batch_images"),
             "new_rescued_fraction_of_positive_new_classes": ("new_rescued_class_occurrences", "positive_new_classes"),
             "direct_threshold_fraction_of_positive_new_classes": ("direct_threshold_new_classes", "positive_new_classes")}
    result.update({name: counts[num] / counts[den] if counts[den] else None
                   for name, (num, den) in pairs.items()})
    return result


def aggregate_diagnostics(path, config, step, arm, count_keys):
    # Base aggregation verifies interval coverage, warmup, integers, original
    # fallback OLD/NEW partition and pixel identities under legacy fusion.
    result = primary.aggregate_counts(path, config, step, "legacy", count_keys)
    for record in read_jsonl(path):
        require(record.get("ald_gate_policy") == arm, f"Diagnostic gate policy mismatch: {path}")
        counts = {key: record[key] for key in count_keys}
        extra_identities(counts, arm)
        for key, value in all_ratios(counts).items():
            require(record.get(key) is None if value is None else finite(record.get(key)) and
                    math.isclose(record[key], value, abs_tol=1e-12), f"Logged gate ratio mismatch: {path}/{key}")
    for scope in ("all", "warmup", "active"):
        group, counts = result[scope], result[scope]["counts"]
        extra_identities(counts, arm)
        group["ratios_from_summed_counts"] = all_ratios(counts)
        kept = counts["positive_new_classes"] - counts["gate_rejected_new_classes"]
        group["derived_counts"] = {"kept_new_class_occurrences": kept,
                                   "original_fallback_new_classes": counts["fallback_new_images"]}
        group["derived_ratios"] = {"kept_new_fraction_of_positive_new_classes":
                                   kept / counts["positive_new_classes"] if counts["positive_new_classes"] else None}
    result["fallback_scope"] = "OLD/NEW ownership from legacy gate before additional NEW rescue"
    return result


def audit_admissions(resource, manifest, primary_manifest):
    """Verify historical resource/exit receipts without querying live PIDs/GPUs."""
    admissions = resource.get("attempts") if isinstance(resource, dict) else None
    require(isinstance(admissions, list) and admissions, "Missing resource admission attempts")
    mapping, smoke = manifest["gpu_mapping"], manifest["smoke"]
    expected_phases = {"before_any_launch": set(mapping.values())}
    expected_phases.update({f"{arm}/step{step}": {mapping[arm]} for arm in ARMS for step in (1, 2)})
    primary_root = Path(manifest["primary_study"])
    require(primary_root.is_relative_to(ROOT), "Primary resource-evidence root must be project-local")
    main_mapping = primary_manifest.get("gpu_mapping")
    require(isinstance(main_mapping, dict), "Primary manifest lacks GPU assignment")
    verified = []
    for row in admissions:
        require(isinstance(row, dict), "Resource admission must be an object")
        phase = row.get("phase")
        require(isinstance(phase, str) and phase in expected_phases, "Unknown resource admission phase")
        expected_gpus = expected_phases[phase]
        cards = row.get("gpus")
        require(isinstance(cards, list) and len(cards) == len(expected_gpus) and
                all(isinstance(card, dict) for card in cards), "Admission must contain the phase's exact nonempty GPU set")
        card_ids = [card.get("gpu") for card in cards]
        require(all(type(gpu) is str and gpu in {"0", "1", "2", "3"} for gpu in card_ids) and
                len(set(card_ids)) == len(card_ids) and set(card_ids) == expected_gpus,
                "Admission GPU indices are invalid, duplicated or differ from the phase mapping")
        minimum = row.get("minimum_free_mib")
        require(type(minimum) is int and minimum >= 20000 and minimum == manifest.get("minimum_free_mib") and
                row.get("smoke_shared_allowed") is smoke, "Resource admission policy differs from the study")
        observed = datetime.fromisoformat(row.get("observed_utc", ""))
        require(observed.tzinfo is not None, "Resource admission needs a timezone-aware observation")
        for card in cards:
            total, used, free = (card.get(key) for key in ("memory_total_mib", "memory_used_mib", "memory_free_mib"))
            require(all(type(value) is int for value in (total, used, free)) and total > 0 and
                    0 <= used <= total and minimum <= free <= total and used + free <= total,
                    "Admission lacks valid physical memory counters or the actual free-memory bound")
            require(type(card.get("shared_physical_gpu")) is bool and
                    card["shared_physical_gpu"] == (used > 256), "Admission shared-device flag differs from physical used memory")
        evidence = row.get("primary_exit0_evidence")
        require(isinstance(evidence, dict), "Resource receipt lacks primary exit-evidence object")
        if not smoke or evidence:
            require(set(evidence) == expected_gpus, "Primary exit evidence must cover exactly the admitted GPUs")
            require(primary_manifest.get("smoke") is False and primary_manifest.get("incremental_iters") == 8000,
                    "Formal admission must refer to the original formal primary study")
            for gpu in card_ids:
                arms = [arm for arm, assigned in main_mapping.items() if assigned == gpu]
                require(len(arms) == 1, "Cannot derive a unique primary arm for the admitted GPU")
                primary_arm = arms[0]
                item = evidence[gpu]
                require(isinstance(item, dict) and item.get("primary_arm") == primary_arm,
                        "Primary exit evidence identifies the wrong arm")
                stages = item.get("verified_stages")
                expected_stages = [primary_root / primary_arm / "10-5" / f"step{step}" for step in (1, 2)]
                require(isinstance(stages, list) and len(stages) == 2 and all(isinstance(stage, dict) for stage in stages) and
                        [stage.get("stage") for stage in stages] == [str(path.relative_to(ROOT)) for path in expected_stages],
                        "Primary exit evidence must contain the correct two-stage chain")
                for step, stage, directory in zip((1, 2), stages, expected_stages):
                    completion_path, process_path = directory / "completion.json", directory / "process.json"
                    require(stage.get("completion_sha256") == sha256(completion_path) and
                            stage.get("process_sha256") == sha256(process_path), "Primary completion/process receipt SHA mismatch")
                    completion, process = read_json(completion_path), read_json(process_path)
                    final = completion.get("final_metrics")
                    require(completion.get("returncode") == process.get("returncode") == 0 and
                            completion.get("iteration") == 8000 and isinstance(final, dict) and
                            final.get("step") == step and final.get("iteration") == 8000 and
                            final.get("ald_mode") == primary_arm and completion.get("command") == process.get("command"),
                            "Primary receipt does not establish the mapped arm's actual final exit-0 stages")
        verified.append({"phase": phase, "gpus": card_ids,
                         "primary_exit0_verified": bool(evidence), "memory_bound_verified": True})
    require(any(row["phase"] == "before_any_launch" for row in verified), "Missing initial resource admission")
    return {"admissions": verified, "scope": "historical static receipts; PID reuse cannot determine past process liveness"}


def audit_stage(study, manifest, arm, step, previous, names, count_keys):
    directory = study / arm / "10-5" / f"step{step}"
    files = {name: directory / name for name in ("config.json", "inputs.json", "completion.json",
             "process.json", "metrics.jsonl", "ald_metrics.jsonl", "checkpoints/model_final.pth")}
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        require(not files["completion.json"].exists(), f"Completion published with missing artifacts: {directory}/{missing}")
        return {"complete": False, "directory": str(directory), "missing_artifacts": missing}
    require(previous is not None, f"Completed step lacks an audited predecessor: {directory}")
    config, inputs, completion, process = (read_json(files[name]) for name in
                                          ("config.json", "inputs.json", "completion.json", "process.json"))
    expected = dict(manifest["common_training_config"], step=step, ald=True, ald_mode="legacy",
                    ald_gate_policy=arm, prev_checkpoint=str(previous.resolve()), work_dir=str(directory),
                    ckpt_dir=str(directory / "checkpoints"), pred_dir=str(directory / "predictions"))
    require(config == expected and inputs.get("expected_config") == expected, f"Stage config mismatch: {directory}")
    checks = {"previous_checkpoint": str(previous.resolve()), "previous_sha256": sha256(previous),
              "pretrained_sha256": manifest["shared_initialization"]["pretrained_sha256"],
              "shared_initialization": manifest["shared_initialization"], "code_sha256": manifest["code_sha256"],
              "dataset_lists_sha256": manifest["dataset_lists_sha256"], "gpu": manifest["gpu_mapping"][arm]}
    require(all(inputs.get(key) == value for key, value in checks.items()), f"Input provenance mismatch: {directory}")
    records = read_jsonl(files["metrics.jsonl"])
    final_iteration, eval_interval = config["max_iters"], config["eval_iters"]
    expected_evals = list(range(eval_interval, final_iteration + 1, eval_interval))
    if not expected_evals or expected_evals[-1] != final_iteration:
        expected_evals.append(final_iteration)
    require([record.get("iteration") for record in records] == expected_evals, f"Evaluation schedule mismatch: {directory}")
    metric_keys = ("step", "ald", "ald_mode", "ald_gate_policy", "seed", "w_proto_kd", "w_proto_sep",
                   "w_proto_seg", "proto_margin", "confusion_reweight")
    for record in records:
        require(all(record.get(key) == config[key] for key in metric_keys) and
                record.get("pixel_padding_ignored") is True, f"Metric configuration mismatch: {directory}")
        primary.metric_groups(record, step, names)
    final = records[-1]
    diagnostics = aggregate_diagnostics(files["ald_metrics.jsonl"], config, step, arm, count_keys)
    coverage_final = read_jsonl(files["ald_metrics.jsonl"])[-1]
    require(completion.get("returncode") == process.get("returncode") == 0 and
            completion.get("iteration") == final_iteration and completion.get("config") == config and
            completion.get("final_metrics") == final and completion.get("final_ald_metrics") == coverage_final and
            completion.get("command") == process.get("command") == inputs.get("command"),
            f"Completion/process/final record mismatch: {directory}")
    hashes = {"inputs_sha256": sha256(files["inputs.json"]), "config_sha256": sha256(files["config.json"]),
              "checkpoint_sha256": sha256(files["checkpoints/model_final.pth"]),
              "metrics_sha256": sha256(files["metrics.jsonl"]), "ald_metrics_sha256": diagnostics["sha256"]}
    require(all(completion.get(key) == value for key, value in hashes.items()), f"Completion digest mismatch: {directory}")
    require(type(process.get("pid")) is int and process["pid"] > 0, f"Invalid process receipt: {directory}")
    for key in ("started_utc", "finished_utc"):
        require(process.get(key) == completion.get(key), f"Process/completion UTC time mismatch: {directory}")
    start, finish = (datetime.fromisoformat(completion[key]) for key in ("started_utc", "finished_utc"))
    duration = (finish - start).total_seconds()
    require(start.tzinfo is not None and finish.tzinfo is not None and duration >= 0, "Invalid process UTC timing")
    hashes["completion_sha256"], hashes["process_sha256"] = sha256(files["completion.json"]), sha256(files["process.json"])
    return {"complete": True, "directory": str(directory), "iteration": final_iteration,
            "ald_mode": "legacy", "ald_gate_policy": arm, "gpu": inputs["gpu"],
            "started_utc": completion["started_utc"], "finished_utc": completion["finished_utc"],
            "elapsed_seconds": duration, "checkpoint": str(files["checkpoints/model_final.pth"]),
            "provenance_sha256": hashes, "raw_final_metrics": final,
            "metric_groups": primary.metric_groups(final, step, names), "diagnostics": diagnostics}


def audit_study(study, partial=False):
    manifest = read_json(study / "study.json")
    require(manifest.get("schema_version") == 1 and manifest.get("study") == "isolated current-class gate fallback" and
            manifest.get("task") == "10-5" and manifest.get("variants") == list(ARMS) and
            manifest.get("expected_stages") == 4 and manifest.get("fusion_mode") == "legacy", "Unexpected gate study schema")
    smoke = manifest.get("smoke")
    require(type(smoke) is bool and manifest.get("incremental_iters") == (4 if smoke else 8000), "Unexpected study budget")
    common = manifest["common_training_config"]
    protocol = {"max_iters": manifest["incremental_iters"], "seed": manifest["seed"], "spg": 8,
                "w_proto_kd": 0., "w_proto_sep": 0., "w_proto_seg": .1, "w_seg": .1,
                "confusion_reweight": False, "loss_warmup_iters": 1 if smoke else 2000,
                "warmup_iters": 1 if smoke else 2000, "log_iters": 1 if smoke else 50,
                "eval_iters": 4 if smoke else 2000, "train_limit": 32 if smoke else 0, "val_limit": 8 if smoke else 0}
    require(all(common.get(key) == value for key, value in protocol.items()) and manifest.get("global_batch_size") == 8,
            "Unexpected matched training configuration")
    mapping = manifest["gpu_mapping"]
    require(set(mapping) == set(ARMS) and len(set(mapping.values())) == 2 and
            set(mapping.values()).issubset({"0", "1", "2", "3"}), "Invalid physical GPU mapping")
    source = manifest["shared_initialization"]
    require(source.get("initialization_seed") == 0 and source.get("step0_iterations") == 20000 and
            sha256(source["checkpoint"]) == source["checkpoint_sha256"] and
            Path(source["checkpoint"]).stat().st_size == source["checkpoint_bytes"] and
            sha256(source["pretrained_checkpoint"]) == source["pretrained_sha256"], "Shared initialization digest mismatch")
    for path, digest in source["source_manifests_sha256"].items():
        require(sha256(path) == digest, f"Shared initialization manifest changed: {path}")
    main_manifest_path = Path(manifest["primary_study"]) / "study.json"
    require(sha256(main_manifest_path) == manifest["primary_study_sha256"],
            "Primary study immutable manifest changed")
    provenance, names, count_keys = source_constants(study, manifest)
    resource_path = study / "resource_admission.json"
    require(resource_path.is_file(), "Missing static resource admission receipt")
    resource = read_json(resource_path)
    resource_verification = audit_admissions(resource, manifest, read_json(main_manifest_path))
    admissions = resource["attempts"]
    status = read_json(study / "status.json") if (study / "status.json").is_file() else None
    published = read_json(study / "results.json") if (study / "results.json").is_file() else None
    if published is not None:
        require(isinstance(published, dict) and set(published) == set(ARMS) and all(
                isinstance(published[arm], dict) and set(published[arm]) == {"1", "2"} for arm in ARMS),
                "Published results must contain exactly four final records")
    stages, count = {}, 0
    for arm in ARMS:
        stages[arm], previous = {}, Path(source["checkpoint"])
        for step in (1, 2):
            stage = audit_stage(study, manifest, arm, step, previous, names, count_keys)
            stages[arm][str(step)] = stage
            if stage["complete"]:
                count += 1
                require(any(row.get("phase") == f"{arm}/step{step}" for row in admissions), "Missing stage resource admission")
                if published is not None:
                    require(published[arm][str(step)] == stage["raw_final_metrics"], "Published/actual final metric mismatch")
                previous = Path(stage["checkpoint"])
            else:
                require(published is None, "Published results contain an unauditable stage")
                previous = None
    status_complete = bool(status and status.get("status") == "complete")
    if status_complete:
        require(count == 4 and published is not None and status.get("completed_stages") == 4 and
                status.get("expected_stages") == 4 and status.get("all_stage_returncodes") == 0 and
                status.get("results_sha256") == sha256(study / "results.json"), "Final runner status mismatch")
    complete = count == 4 and status_complete and published is not None and provenance["archive_verified"]
    require(partial or complete, f"Study incomplete: {count}/4 audited stages, archive_verified={provenance['archive_verified']}; use --partial")
    deltas = {}
    for step in ("1", "2"):
        control, candidate = stages["legacy"][step], stages["new_fallback"][step]
        deltas[step] = None
        if control["complete"] and candidate["complete"]:
            deltas[step] = {key: (group["miou"] - control["metric_groups"][key]["miou"]
                                 if group["miou"] is not None and control["metric_groups"][key]["miou"] is not None else None)
                            for key, group in candidate["metric_groups"].items()}
    times = [s for rows in stages.values() for s in rows.values() if s["complete"]]
    window = ((max(datetime.fromisoformat(s["finished_utc"]) for s in times) -
               min(datetime.fromisoformat(s["started_utc"]) for s in times)).total_seconds() if times else None)
    return {"schema_version": 1, "study": str(study), "complete": complete, "partial": not complete,
            "smoke": smoke, "performance_evidence": complete and not smoke,
            "generated_utc": datetime.now(timezone.utc).isoformat(), "completed_stages": count, "expected_stages": 4,
            "runner_status": status, "seed": manifest["seed"], "seed_scope": manifest["seed_scope"],
            "gpu_mapping": mapping, "fusion_mode": "legacy", "raw_gate_threshold": 12.5,
            "shared_initialization": source, "provenance": provenance, "count_keys": list(count_keys), "class_list": names,
            "postprocessing_sources_sha256": {"tools/summarize_ald_gate_study.py": sha256(Path(__file__).resolve()),
                                              "tools/summarize_ald_study.py": sha256(ROOT / "tools/summarize_ald_study.py")},
            "comparison_control": "same isolated fork legacy gate; only gate policy differs",
            "historical_references": {"study": manifest["primary_study"], "arms": ["legacy", "off"],
                                      "role": "context only; not this experiment's paired control; no intermediate scores imported"},
            "metric_scope": {"step1": "previous foreground 1..10, current 11..15, all+BG 0..15",
                             "step2": "previous foreground 1..15, current 16..20, all+BG 0..20",
                             "raw_old_miou": "initial foreground 1..10 at both stages",
                             "raw_new_miou": "cumulative NEW 11..15 at step1, 11..20 at step2",
                             "null_iou": "excluded from group means; finite and total class counts are explicit"},
            "resource_admission": resource,
            "resource_verification": resource_verification,
            "gpu_usage_scope": "static physical-device observations include pre-existing jobs; not per-job peak usage",
            "timing": {"completed_stage_seconds_sum": sum(s["elapsed_seconds"] for s in times),
                       "audited_stage_window_seconds": window, "scope": "incremental processes and validation; parallel stage sum is not wall time"},
            "limitations": ["Single incremental seed, original seed-0 shared initialization, no error bars",
                            "Smoke validates pipeline/counts only and cannot establish improvement",
                            "Warmup constructs labels while pixel loss coefficient is zero; active counts are separate",
                            "Original image NEW labels may refer to objects removed by random crop; rescue can create wrong pixel labels",
                            "Different stage image sets and valid GT label ranges prevent interpreting cross-stage drops as forgetting"],
            "stages": stages, "deltas_vs_same_fork_legacy_pp": deltas}


def plot_comparison(study, summary):
    cache = checked_path(ROOT / ".runtime/cache/matplotlib")
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 7.1))
    figure.suptitle(f"VOC 10-5 | isolated NEW gate rescue | incremental seed {summary['seed']}", y=.98, fontsize=15)
    figure.text(.5, .933, "Both fusion = legacy; raw gate = 12.5; KD = SEP = 0; common prototype-pixel weight 0.1",
                ha="center", fontsize=10)
    banner = "SMOKE — NOT PERFORMANCE" if summary["smoke"] else "INCOMPLETE — FINAL STAGES ONLY" if summary["partial"] else ""
    if summary["smoke"] and summary["partial"]:
        banner += " | INCOMPLETE"
    if banner:
        figure.text(.5, .878, banner, ha="center", color="#b91c1c", fontsize=16, weight="bold")
    deltas = summary["deltas_vs_same_fork_legacy_pp"]
    points = [(d["previous_fg"], d["current5_fg"]) for d in deltas.values()
              if d and finite(d["previous_fg"]) and finite(d["current5_fg"])]
    def limits(values):
        low, high = min([0.] + values), max([0.] + values)
        pad = .3 * max(high - low, .15)
        return low - pad, high + pad
    xlim, ylim = limits([p[0] for p in points]), limits([p[1] for p in points])
    for step, axis in enumerate(axes, 1):
        axis.axhline(0, color="#6b7280", lw=.9, ls="--")
        axis.axvline(0, color="#6b7280", lw=.9, ls="--")
        axis.grid(alpha=.16)
        axis.set_xlim(xlim)
        axis.set_ylim(ylim)
        axis.set_title(f"Step {step}: previous {10 if step == 1 else 15} / current 5", pad=12)
        axis.set_xlabel("Δ previous foreground mIoU vs this fork's LEGACY (pp)")
        axis.set_ylabel("Δ current 5 foreground mIoU vs this fork's LEGACY (pp)")
        if summary["stages"]["legacy"][str(step)]["complete"]:
            axis.scatter([0], [0], marker="+", s=150, color="#111827", linewidths=2, zorder=5)
            axis.annotate("LEGACY gate", (0, 0), xytext=(8, -18), textcoords="offset points", weight="bold")
        delta = deltas[str(step)]
        if delta and finite(delta["previous_fg"]) and finite(delta["current5_fg"]):
            x, y = delta["previous_fg"], delta["current5_fg"]
            color = "#059669"
            axis.scatter([x], [y], s=90, color=color, marker="s", edgecolor="white", linewidth=.7, zorder=4)
            side = x >= sum(xlim) / 2
            all_delta = delta["all_with_bg"]
            label = "NEW fallback" + (f"\nΔall+BG = {all_delta:+.3f} pp" if finite(all_delta) else "")
            axis.annotate(label, (x, y), xytext=(-12 if side else 12, 38), textcoords="offset points",
                          ha="right" if side else "left", va="center", color=color,
                          arrowprops={"arrowstyle": "-", "color": color, "alpha": .6, "lw": .7})
        else:
            axis.text(.5, .5, "No audited final candidate/control comparison", ha="center", transform=axis.transAxes,
                      fontsize=9, color="#6b7280")
    figure.text(.5, .058, "Common class groups and axis limits; null IoUs excluded; single seed, no error bars; within-stage comparisons only.",
                ha="center", fontsize=9)
    figure.text(.5, .031, "Paired control is this isolated fork's LEGACY. Primary legacy/off are reference only; physical GPU usage includes prior jobs.",
                ha="center", fontsize=9)
    figure.subplots_adjust(left=.08, right=.97, bottom=.18, top=.77, wspace=.29)
    for extension in ("png", "pdf"):
        path = checked_path(study / f"comparison.{extension}")
        temporary = checked_path(path.with_name(f".{path.stem}.{uuid.uuid4().hex}.{extension}"))
        figure.savefig(temporary, dpi=180, format=extension, facecolor="white")
        os.replace(temporary, checked_path(path))
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default="runs/ald_gate_new_v1")
    parser.add_argument("--partial", action="store_true", help="Explicitly report incomplete stages; compare final audited pairs only")
    args = parser.parse_args()
    try:
        study = checked_path(ROOT / args.study)
        require(study.is_dir() and study.is_relative_to(ROOT / "runs"), "Expected an existing project-local study directory")
        require(not any(study.is_relative_to(ROOT / "runs" / protected) for protected in
                        ("fixed_baseline_v1", "ald_fusion_v1")), "Output must be separate from the preserved primary studies")
        for name in ("summary.json", "comparison.png", "comparison.pdf"):
            checked_path(study / name)
        summary = audit_study(study, partial=args.partial)
        plot_comparison(study, summary)
        # primary's atomic JSON writer is safe after this stronger path preflight.
        primary.write_summary(checked_path(study / "summary.json"), summary)
        print(f"{'COMPLETE' if summary['complete'] else 'INCOMPLETE'}: {summary['completed_stages']}/4 audited final stages")
        print("SMOKE — NOT PERFORMANCE" if summary["smoke"] else "Single-seed gate comparison; see metric scope and limitations.")
        print(study / "summary.json")
    except (ValueError, KeyError, FileNotFoundError) as error:
        parser.exit(2, f"ALD gate summary error: {error}\n")


if __name__ == "__main__":
    main()
