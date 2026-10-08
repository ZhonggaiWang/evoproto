"""Plot audited step-1 label-coverage trajectories; CPU only.

python -B tools/plot_ald_diagnostics.py --partial
An incomplete four-arm trajectory requires --partial. No model is evaluated.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import sys

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import summarize_ald_study as primary

ROOT = primary.ROOT
METRICS = {
    "new_positive_class_retention": {
        "numerator": "positive_new_classes - gate_rejected_new_classes",
        "denominator": "positive_new_classes",
        "title": "Positive NEW class retention",
        "formula": "(positive NEW - rejected NEW) / positive NEW",
    },
    "rejected_fraction_of_new_cam": {
        "numerator": "rejected_new_cam_pixels", "denominator": "before_new_cam_pixels",
        "title": "Rejected NEW CAM pixels",
        "formula": "Rejected NEW pixels / pre-gate NEW CAM pixels",
    },
    "supervised_fraction_of_valid_image": {
        "numerator": "final_valid_pixels", "denominator": "valid_box_pixels",
        "title": "Final valid pixel targets",
        "formula": "Final supervised pixels / valid img_box pixels (V)",
    },
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def diagnostic_ratios(counts):
    result = primary.ratios(counts)
    denominator = counts["positive_new_classes"]
    result["new_positive_class_retention"] = (
        (denominator - counts["gate_rejected_new_classes"]) / denominator
        if denominator else None)
    return {name: result[name] for name in METRICS}


def snapshot(path):
    """Keep exact bytes read, and every newline-terminated JSON row verbatim."""
    path = primary.checked_path(path)
    started = utc()
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        raw = handle.read()
        after = os.fstat(handle.fileno())
    primary.require(stat.S_ISREG(after.st_mode) and after.st_nlink == 1 and
                    (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino) and
                    after.st_size >= before.st_size and len(raw) <= after.st_size,
                    f"Input is not an ordinary append-only log snapshot: {path}")
    boundary = raw.rfind(b"\n") + 1
    complete = raw[:boundary]
    lines = complete.splitlines(keepends=True)
    # A malformed complete row is an error; only an unterminated tail is omitted.
    records = [json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError(f"Nonfinite JSON constant {value}: {path}"))) for line in lines]
    primary.require(records and all(isinstance(row, dict) for row in records),
                    f"No complete diagnostic JSON records: {path}")
    return records, {
        "path": str(path), "read_started_utc": started, "read_finished_utc": utc(),
        "captured_bytes_sha256": hashlib.sha256(raw).hexdigest(), "captured_byte_size": len(raw),
        "file_byte_size_before_read": before.st_size, "file_byte_size_after_read": after.st_size,
        "complete_rows_sha256": hashlib.sha256(complete).hexdigest(),
        "complete_rows_byte_size": len(complete), "unterminated_tail_byte_size": len(raw) - boundary,
        "complete_rows_verbatim": [line.decode("utf-8") for line in lines],
        "digest_scope": "Exact bytes captured at observation time; not a future terminal log SHA",
    }


def audit_arm(study, manifest, mode, count_keys):
    directory = study / mode / "10-5/step1"
    config_path = primary.checked_path(directory / "config.json")
    config = primary.read_json(config_path)
    common = manifest["common_training_config"]
    primary.require(all(config.get(key) == value for key, value in common.items()) and
                    config.get("step") == 1 and config.get("ald_mode") == mode and
                    config.get("ald") == (mode != "off"), f"Unmatched configuration: {mode}")
    rows, evidence = snapshot(directory / "ald_metrics.jsonl")
    last = rows[-1]["iteration"]
    interval, budget, warmup = config["log_iters"], config["max_iters"], config["loss_warmup_iters"]
    primary.require(type(last) is int and 0 < last <= budget and last % interval == 0 and
                    [row.get("iteration") for row in rows] == list(range(interval, last + 1, interval)),
                    f"Missing, duplicate or out-of-budget intervals: {mode}")
    groups = {name: {"counts": dict.fromkeys(count_keys, 0), "optimizer_updates": 0,
                     "interval_records": 0} for name in ("all", "warmup", "active")}
    points = []
    for row in rows:
        end = row["iteration"]
        counts = {key: row.get(key) for key in count_keys}
        active = end > warmup
        primary.require(row.get("step") == 1 and row.get("ald_mode") == mode and
                        row.get("interval_batches") == interval and
                        type(row.get("segmentation_loss_active")) is bool and
                        row["segmentation_loss_active"] == active and
                        not (end - interval < warmup < end), f"Interval metadata mismatch: {mode}/{end}")
        primary.require(all(type(value) is int and value >= 0 for value in counts.values()) and
                        counts["batch_images"] == interval * config["spg"] and
                        counts["total_pixels"] == counts["batch_images"] * config["crop_size"] ** 2 and
                        counts["positive_new_classes"] <= counts["batch_images"] * 5 and
                        counts["positive_old_classes"] <= counts["batch_images"] * 10,
                        f"Invalid integer counts: {mode}/{end}")
        primary.count_identities(counts, mode)
        for key, value in primary.ratios(counts).items():
            primary.require(row.get(key) is None if value is None else primary.finite(row.get(key)) and
                            math.isclose(row[key], value, abs_tol=1e-12), f"Logged ratio mismatch: {mode}/{end}")
        for name in ("all", "active" if active else "warmup"):
            group = groups[name]
            group["optimizer_updates"] += interval
            group["interval_records"] += 1
            for key in count_keys:
                group["counts"][key] += counts[key]
        points.append({"interval_start_iteration": end - interval + 1, "iteration": end,
                       "segmentation_loss_active": active, "counts": counts,
                       "ratios": diagnostic_ratios(counts)})
    for group in groups.values():
        primary.count_identities(group["counts"], mode)
        group["ratios_from_summed_integer_counts"] = diagnostic_ratios(group["counts"])
    return {"last_complete_iteration": last, "training_budget": budget,
            "all_budget_rows_observed": last == budget, "snapshot": evidence,
            "config_sha256": primary.sha256(config_path), "interval_points": points,
            "count_aggregates": groups}


def plot(summary):
    for variable, relative in (("MPLCONFIGDIR", ".runtime/cache/matplotlib"),
                               ("XDG_CACHE_HOME", ".runtime/cache/xdg"), ("TMPDIR", ".runtime/tmp")):
        cache = primary.checked_path(ROOT / relative)
        cache.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(cache)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 3, figsize=(15, 6.4), sharex=True)
    figure.suptitle("VOC 10-5 | Step 1: ALD supervision-target diagnostics", y=.98, fontsize=15)
    figure.text(.5, .928, "Raw CAM gate threshold 12.5; KD = SEP = 0; shared step 0; seed 0 (single seed)",
                ha="center", fontsize=10)
    if summary["partial"]:
        figure.text(.5, .875, "INCOMPLETE TRAJECTORIES", ha="center", color="#b91c1c",
                    fontsize=16, weight="bold")
    styles = {"off": ("#111827", "--"), "legacy": ("#2563eb", "-"),
              "preserve_rejected": ("#059669", "-"), "preserve_background": ("#c026d3", "-")}
    for axis, (metric, definition) in zip(axes, METRICS.items()):
        axis.axvspan(0, 2000, color="#f59e0b", alpha=.14, zorder=0)
        axis.axvline(2000, color="#b45309", lw=.8, ls=":")
        axis.text(1000, 104, "WARMUP\npixel loss = 0", ha="center", va="top",
                  color="#92400e", fontsize=9, weight="bold")
        for mode in primary.VARIANTS:
            data = summary["arms"][mode]
            points = data["interval_points"]
            values = [point["ratios"][metric] for point in points]
            y = [100 * value if value is not None else float("nan") for value in values]
            color, linestyle = styles[mode]
            axis.plot([point["iteration"] for point in points], y, color=color, ls=linestyle,
                      lw=1.3, alpha=.9, label=f"{primary.DISPLAY[mode]} (to {data['last_complete_iteration']})")
            if values[-1] is not None:
                axis.scatter([points[-1]["iteration"]], [y[-1]], s=22, color=color, zorder=4)
        axis.set_xlim(0, 8000)
        axis.set_ylim(-3, 108)
        axis.set_xticks([0, 2000, 4000, 6000, 8000])
        axis.set_xlabel("Optimizer iteration (50-update interval end)")
        axis.set_ylabel("Interval ratio (%)")
        axis.set_title(definition["title"] + "\n" + definition["formula"], fontsize=10, pad=12)
        axis.grid(alpha=.16)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .135), ncol=2,
                  frameon=False, fontsize=9)
    figure.text(.5, .087, "Each point uses that interval's integer numerator / denominator; no smoothing or extrapolation.",
                ha="center", fontsize=9)
    figure.text(.5, .055, "Label coverage is not CAM correctness, pixel accuracy or mIoU. V excludes padding; zero denominators are gaps.",
                ha="center", fontsize=9)
    figure.text(.5, .023, "Different arms have different observed lengths and later targets; coverage alone does not measure loss or gradient magnitude.",
                ha="center", fontsize=9)
    figure.subplots_adjust(left=.055, right=.985, top=.735, bottom=.315, wspace=.29)
    artifacts = {}
    for extension in ("png", "pdf"):
        buffer = io.BytesIO()
        figure.savefig(buffer, format=extension, dpi=180, facecolor="white")
        artifacts[extension] = buffer.getvalue()
    plt.close(figure)
    return artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default="runs/ald_fusion_v1")
    parser.add_argument("--partial", action="store_true", help="Explicitly mark incomplete four-arm trajectories")
    args = parser.parse_args()
    try:
        study = primary.checked_path(ROOT / args.study)
        primary.require(study.is_dir() and study.is_relative_to(ROOT / "runs"), "Expected project-local study")
        outputs = {extension: primary.checked_path(study / f"diagnostics.{extension}")
                   for extension in ("json", "png", "pdf")}
        manifest_path = primary.checked_path(study / "study.json")
        manifest = primary.read_json(manifest_path)
        common = manifest["common_training_config"]
        primary.require(manifest.get("variants") == list(primary.VARIANTS) and manifest.get("task") == "10-5" and
                        manifest.get("smoke") is False and manifest.get("seed") == 0 and
                        common.get("max_iters") == 8000 and common.get("log_iters") == 50 and
                        common.get("loss_warmup_iters") == 2000 and common.get("spg") == 8 and
                        common.get("w_proto_kd") == common.get("w_proto_sep") == 0,
                        "Expected formal matched seed-0, KD/SEP-off step-1 protocol")
        for relative in manifest["code_sha256"]:
            primary.checked_path(ROOT / relative)
        for path in manifest["dataset_lists_sha256"]:
            primary.checked_path(ROOT / path)
        for filename in ("source_snapshot.json", "source_snapshot.tar.gz"):
            primary.checked_path(study / filename)
        provenance, texts = primary.audit_sources(study, manifest, allow_drift=False)
        primary.require(provenance["archive_verified"] and provenance["current_source_matches_recorded"],
                        "Frozen training source and archive must both verify")
        count_keys = primary.literal_assignment(ROOT / "utils/ald_stats.py", "COUNT_KEYS", texts["utils/ald_stats.py"])
        primary.require(len(count_keys) == len(set(count_keys)) == 25, "Expected original 25 integer count keys")
        arms = {mode: audit_arm(study, manifest, mode, count_keys) for mode in primary.VARIANTS}
        complete = all(data["all_budget_rows_observed"] for data in arms.values())
        primary.require(complete or args.partial, "Incomplete trajectories require explicit --partial")
        summary = {
            "schema": "evoproto.ald.step1.diagnostics.v1", "generated_utc": utc(), "step": 1,
            "study": str(study), "study_manifest_sha256": primary.sha256(manifest_path),
            "plotter_sha256": primary.sha256(primary.checked_path(Path(__file__).absolute())),
            "partial_requested": args.partial, "partial": not complete, "all_step1_budget_rows_observed": complete,
            "completion_scope": "Log-trajectory coverage only; no final checkpoint, runner exit or mIoU audit",
            "optimizer_budget_per_arm": 8000, "log_interval_updates": 50, "global_batch_size": 8,
            "pixel_loss_zero_iterations": [1, 2000], "raw_cam_gate_threshold": 12.5,
            "metric_definitions": METRICS, "zero_denominator_policy": "null in JSON; gap in curve",
            "source_provenance": provenance,
            "source_archive_file_count": len(primary.read_json(study / "source_snapshot.json")["files"]),
            "frozen_core_file_count": len(manifest["code_sha256"]),
            "validation_scope": "Every captured complete step-1 row: interval schedule, 25 nonnegative integers, mode/padding/fallback partitions and logged ratios",
            "aggregation": "Plot individual 50-update ratios; totals sum integer counts before division, never average interval ratios",
            "limitations": ["One seed; diagnostic label coverage, not CAM correctness, pixel accuracy or mIoU",
                            "Observation-time snapshot digests are not future terminal hashes",
                            "Unequal trajectory lengths; no extrapolation or performance comparison across iterations",
                            "Later arm batches and CAMs are not asserted identical",
                            "Warmup targets do not contribute pixel loss; valid-target coverage does not measure loss or gradient magnitude"],
            "arms": arms,
        }
        artifacts = plot(summary)
        summary["figure_sha256"] = {ext: hashlib.sha256(blob).hexdigest() for ext, blob in artifacts.items()}
        artifacts["json"] = (json.dumps(summary, indent=2, allow_nan=False) + "\n").encode("utf-8")
        for extension, blob in artifacts.items():
            path = primary.checked_path(outputs[extension])
            with path.open("wb" if path.exists() else "xb") as handle:
                handle.write(blob)
                handle.flush()
                os.fsync(handle.fileno())
        print("INCOMPLETE TRAJECTORIES" if not complete else "ALL STEP-1 LOG TRAJECTORIES OBSERVED")
        print(json.dumps({mode: {"last_iteration": data["last_complete_iteration"],
                               "first_interval_ratios": data["interval_points"][0]["ratios"],
                               "last_interval_ratios": data["interval_points"][-1]["ratios"]}
                          for mode, data in arms.items()}, indent=2))
        print(outputs["json"])
    except (ValueError, KeyError, FileNotFoundError, json.JSONDecodeError) as error:
        parser.exit(2, f"ALD diagnostics error: {error}\n")


if __name__ == "__main__":
    main()
