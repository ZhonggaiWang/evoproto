"""Ground-truth diagnostics for an online, directed pseudo-label confusion model.

This module never changes a training target or a learning weight. Ground truth is
used only as a diagnostic reference. Counts may contain nonnegative evidence
weights: support is therefore reported as mass, not assumed to be a sample size.
"""

from __future__ import annotations

import math

import numpy as np


def _counts(value, name, size=None):
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric square matrix") from exc
    if result.ndim != 2 or result.shape[0] != result.shape[1] or not result.shape[0]:
        raise ValueError(f"{name} must be a nonempty square matrix")
    if size is not None and result.shape != (size, size):
        raise ValueError(f"{name} must have shape {(size, size)}")
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError(f"{name} must contain finite nonnegative values")
    with np.errstate(over="ignore"):
        finite_totals = np.isfinite(result.sum(axis=1)).all() and np.isfinite(result.sum())
    if not finite_totals:
        raise ValueError(f"{name} has overflowing support totals")
    return result.copy()


def _probabilities(counts):
    support = counts.sum(axis=1)
    normalized = np.divide(
        counts, support[:, None], out=np.zeros_like(counts), where=support[:, None] > 0
    )
    return normalized, support


def _json_rows(normalized, support):
    return [row.tolist() if mass > 0 else None for row, mass in zip(normalized, support)]


def _average_ranks(values):
    """Ascending ranks with the arithmetic mean assigned to exact ties."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def _spearman(left, right):
    if len(left) < 2:
        return None
    lr, rr = _average_ranks(left), _average_ranks(right)
    lr -= lr.mean()
    rr -= rr.mean()
    denominator = float(np.linalg.norm(lr) * np.linalg.norm(rr))
    if denominator == 0:
        return None
    return float(np.clip(np.dot(lr, rr) / denominator, -1.0, 1.0))


def _top_membership(scores, k):
    """Top-k positive scores; ties at the cutoff share the remaining slots."""
    scores = np.asarray(scores, dtype=np.float64)
    membership = np.zeros_like(scores)
    positive = np.flatnonzero(scores > 0)
    if not len(positive):
        return membership
    order = positive[np.argsort(-scores[positive], kind="stable")]
    remaining = min(int(k), len(order))
    start = 0
    while remaining > 0 and start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        fraction = min(1.0, remaining / (end - start))
        membership[order[start:end]] = fraction
        remaining -= min(remaining, end - start)
        start = end
    return membership


def _strongest_targets(scores, target_ids):
    maximum = float(np.max(scores)) if len(scores) else 0.0
    return [int(i) for i, score in zip(target_ids, scores) if score == maximum] if maximum > 0 else []


def _row_comparison(estimated, reference, row):
    target_ids = np.delete(np.arange(len(estimated)), row)
    ep, rp = estimated[target_ids], reference[target_ids]
    strongest_estimate = _strongest_targets(ep, target_ids)
    strongest_reference = _strongest_targets(rp, target_ids)
    # An estimate with zero error mass must not disappear from the accuracy
    # denominator when the GT reference contains a real confusion relationship.
    strongest_defined = bool(strongest_reference)
    em, rm = _top_membership(ep, 3), _top_membership(rp, 3)
    top3_slots = float(max(em.sum(), rm.sum()))
    return {
        "l1_distance": float(np.abs(estimated - reference).sum()),
        "total_variation_distance": float(np.abs(estimated - reference).sum() / 2),
        "estimated_offdiagonal_mass": float(ep.sum()),
        "reference_offdiagonal_mass": float(rp.sum()),
        "estimated_strongest_targets": strongest_estimate,
        "reference_strongest_targets": strongest_reference,
        "strongest_target_correct": bool(set(strongest_estimate) & set(strongest_reference)) if strongest_defined else None,
        "strongest_target_exact_set_match": strongest_estimate == strongest_reference if strongest_defined else None,
        "estimated_top3_targets": [
            {"class_id": int(i), "membership": float(m)} for i, m in zip(target_ids, em) if m > 0
        ],
        "reference_top3_targets": [
            {"class_id": int(i), "membership": float(m)} for i, m in zip(target_ids, rm) if m > 0
        ],
        "top3_overlap_fraction": float(np.minimum(em, rm).sum() / top3_slots) if top3_slots > 0 else None,
        "offdiagonal_spearman": _spearman(ep, rp),
    }


def _mean(values):
    present = [float(v) for v in values if v is not None]
    return float(sum(present) / len(present)) if present else None


def _global_pairs(estimate, reference, eligible_rows, foreground_start, k):
    pairs = [(i, j) for i in eligible_rows for j in range(foreground_start, len(estimate)) if i != j]
    es = np.asarray([estimate[i, j] for i, j in pairs], dtype=np.float64)
    rs = np.asarray([reference[i, j] for i, j in pairs], dtype=np.float64)
    em, rm = _top_membership(es, k), _top_membership(rs, k)
    selected = float(em.sum())
    positive_precision = float(em[rs > 0].sum() / selected) if selected > 0 else None
    return {
        "requested_k": int(k),
        "eligible_pair_count": len(pairs),
        "estimated_effective_k": selected,
        "reference_effective_k": float(rm.sum()),
        "precision_at_k": float(np.minimum(em, rm).sum() / selected) if selected > 0 else None,
        "positive_pair_precision": positive_precision,
        "false_positive_pair_fraction": 1.0 - positive_precision if positive_precision is not None else None,
        "selected_estimated_pairs": [
            {
                "source_class": int(i),
                "target_class": int(j),
                "estimated_conditional_rate": float(e),
                "reference_conditional_rate": float(r),
                "topk_membership": float(m),
            }
            for (i, j), e, r, m in zip(pairs, es, rs, em) if m > 0
        ],
    }


def evaluate_matrices(
    estimated_counts,
    gt_all_counts,
    gt_same_support_counts,
    pseudo_gt_counts=None,
    min_gt_row_pixels=100,
    foreground_start=1,
):
    """Return JSON-safe matrix, support, direction, ranking, and purity diagnostics.

    Matrix rows are source/anchor classes and columns are prediction classes.
    ``estimated_counts`` conditions on pseudo anchors at selected evidence pixels.
    ``gt_all_counts`` conditions on GT across the complete diagnostic population.
    ``gt_same_support_counts`` conditions on GT on exactly those selected evidence
    pixels, using exactly the estimate's weights. Optional ``pseudo_gt_counts``
    uses pseudo-anchor rows and GT columns on that same evidence support.

    Comparisons require positive estimate support and reference row mass >= the
    supplied threshold. Missing or under-supported rows stay unavailable, never
    replaced by a diagonal or a zero-error interpretation. Full-row TV/L1 includes
    the diagonal and background; ranking uses all off-diagonal target classes.
    Aggregate scores use foreground source rows. Global top-k pairs also exclude
    background target columns and the diagonal. Ties at a top-k cutoff share slots;
    precision_at_k measures shared membership mass against reference top-k pairs.
    """
    estimate = _counts(estimated_counts, "estimated_counts")
    size = len(estimate)
    all_gt = _counts(gt_all_counts, "gt_all_counts", size)
    selected_gt = _counts(gt_same_support_counts, "gt_same_support_counts", size)
    anchors = None if pseudo_gt_counts is None else _counts(pseudo_gt_counts, "pseudo_gt_counts", size)
    if isinstance(min_gt_row_pixels, (bool, np.bool_)):
        raise ValueError("min_gt_row_pixels must be a finite nonnegative number")
    try:
        minimum = float(min_gt_row_pixels)
    except (ValueError, TypeError) as exc:
        raise ValueError("min_gt_row_pixels must be a finite nonnegative number") from exc
    if not math.isfinite(minimum) or minimum < 0:
        raise ValueError("min_gt_row_pixels must be a finite nonnegative number")
    if isinstance(foreground_start, (bool, np.bool_)) or not isinstance(foreground_start, (int, np.integer)) or not 0 <= foreground_start < size:
        raise ValueError("foreground_start must be an integer class index within the matrix")
    foreground_start = int(foreground_start)
    ep, es = _probabilities(estimate)
    ap, ass = _probabilities(all_gt)
    sp, ss = _probabilities(selected_gt)
    pp, ps = (None, None) if anchors is None else _probabilities(anchors)
    references = {"full_population": (ap, ass), "same_evidence_support": (sp, ss)}
    rows = []
    for i in range(size):
        comparisons = {}
        for name, (rp, support) in references.items():
            enough_reference = bool(support[i] > 0 and support[i] >= minimum)
            comparison = _row_comparison(ep[i], rp[i], i) if es[i] > 0 and enough_reference else None
            comparisons[name] = comparison
        rows.append({
            "class_id": i,
            "estimated_support_mass": float(es[i]),
            "gt_full_support_mass": float(ass[i]),
            "gt_same_support_mass": float(ss[i]),
            "pseudo_anchor_support_mass": None if ps is None else float(ps[i]),
            "pseudo_anchor_precision": None if ps is None or ps[i] == 0 else float(anchors[i, i] / ps[i]),
            "comparisons": comparisons,
        })
    summaries = {}
    for name, (rp, support) in references.items():
        eligible = [i for i in range(foreground_start, size) if rows[i]["comparisons"][name] is not None]
        comparison_rows = [rows[i]["comparisons"][name] for i in eligible]
        total_mass = float(support[eligible].sum())
        weighted_tv = sum(rows[i]["comparisons"][name]["total_variation_distance"] * float(support[i]) for i in eligible)
        summaries[name] = {
            "eligible_foreground_rows": eligible,
            "eligible_foreground_row_count": len(eligible),
            "mean_l1_distance": _mean([r["l1_distance"] for r in comparison_rows]),
            "mean_total_variation_distance": _mean([r["total_variation_distance"] for r in comparison_rows]),
            "gt_mass_weighted_total_variation_distance": float(weighted_tv / total_mass) if total_mass > 0 else None,
            "strongest_target_accuracy": _mean([r["strongest_target_correct"] for r in comparison_rows]),
            "strongest_target_evaluable_rows": sum(r["strongest_target_correct"] is not None for r in comparison_rows),
            "mean_top3_overlap_fraction": _mean([r["top3_overlap_fraction"] for r in comparison_rows]),
            "mean_offdiagonal_spearman": _mean([r["offdiagonal_spearman"] for r in comparison_rows]),
            "spearman_evaluable_rows": sum(r["offdiagonal_spearman"] is not None for r in comparison_rows),
            "topk_foreground_pairs": {str(k): _global_pairs(ep, rp, eligible, foreground_start, k) for k in (5, 10, 20)},
        }
    anchor_summary = None
    if anchors is not None:
        fg_mass = float(ps[foreground_start:].sum())
        mass = float(ps.sum())
        anchor_summary = {
            "meaning": "Weighted pseudo-anchor correctness on selected evidence; this is label precision, not probability calibration.",
            "weighted_precision_all_classes": float(np.trace(anchors) / mass) if mass > 0 else None,
            "weighted_precision_foreground": float(np.diag(anchors)[foreground_start:].sum() / fg_mass) if fg_mass > 0 else None,
            "total_evidence_mass": mass,
            "foreground_evidence_mass": fg_mass,
            "unsupported_rows": [int(i) for i in np.flatnonzero(ps == 0)],
        }
    scopes = {
        "estimated": "P(student argmax=j | pseudo anchor=i, selected evidence pixels), weighted by evidence mass.",
        "full_population": "P(the same student argmax=j | GT=i, complete diagnostic population); includes evidence-selection and anchor errors.",
        "same_evidence_support": "P(the same student argmax=j | GT=i, exactly the selected evidence pixels and estimate weights); still conditions on GT rather than pseudo anchors.",
        "pseudo_gt": "P(GT=j | pseudo anchor=i, the same selected evidence and weights); independent anchor correctness diagnostic.",
        "population_warning": "Same-support agreement does not establish full-population accuracy. Unobserved or under-supported rows are not evidence of zero confusion.",
        "support_warning": "Support values are weighted mass; they are not effective independent sample sizes or confidence intervals.",
        "topk_ties": "Positive-score ties share remaining top-k slots. Precision compares shared fractional membership mass; a false-positive pair has zero reference off-diagonal count.",
        "strongest_ties": "Strongest-target correctness accepts any intersection between tied maxima; exact set agreement is reported separately. A GT row without any off-diagonal error has no strongest target; a missing estimated target against a nonempty GT target counts as incorrect.",
    }
    summary_lines = []
    for name, metrics in summaries.items():
        tv = metrics["mean_total_variation_distance"]
        strongest = metrics["strongest_target_accuracy"]
        summary_lines.append(
            f"{name}: {metrics['eligible_foreground_row_count']} eligible foreground rows; "
            f"mean TV={tv:.6f}" if tv is not None else f"{name}: no eligible foreground rows; mean TV unavailable"
        )
        summary_lines.append(f"{name}: strongest-target accuracy={strongest:.6f}" if strongest is not None else f"{name}: strongest-target accuracy unavailable")
    if anchor_summary is not None:
        purity = anchor_summary["weighted_precision_foreground"]
        summary_lines.append(f"Foreground pseudo-anchor precision={purity:.6f}; not calibration." if purity is not None else "Foreground pseudo-anchor precision unavailable; not calibration.")
    mass_checks = {
        "estimated_total_mass": float(es.sum()),
        "gt_same_support_total_mass": float(ss.sum()),
        "same_support_total_mass_matches_estimate": bool(np.isclose(es.sum(), ss.sum(), rtol=1e-8, atol=1e-8)),
        "pseudo_gt_total_mass": None if ps is None else float(ps.sum()),
        "pseudo_gt_total_mass_matches_estimate": None if ps is None else bool(np.isclose(es.sum(), ps.sum(), rtol=1e-8, atol=1e-8)),
        "pseudo_anchor_row_mass_matches_estimate": None if ps is None else bool(np.allclose(es, ps, rtol=1e-8, atol=1e-8)),
        "note": "Matching mass is necessary but cannot prove that callers used the same pixels, predictions, and weights.",
    }
    return {
        "schema_version": 1,
        "direction": "row=source/anchor class, column=student prediction; pseudo_gt alone uses GT columns",
        "num_classes": size,
        "foreground_start": foreground_start,
        "minimum_reference_row_mass": minimum,
        "conditioning_scope": scopes,
        "matrices": {
            "estimated_counts": estimate.tolist(),
            "estimated_row_probabilities": _json_rows(ep, es),
            "gt_all_counts": all_gt.tolist(),
            "gt_all_row_probabilities": _json_rows(ap, ass),
            "gt_same_support_counts": selected_gt.tolist(),
            "gt_same_support_row_probabilities": _json_rows(sp, ss),
            "pseudo_gt_counts": None if anchors is None else anchors.tolist(),
            "pseudo_gt_row_probabilities": None if anchors is None else _json_rows(pp, ps),
        },
        "unsupported_rows": {
            "estimated": [int(i) for i in np.flatnonzero(es == 0)],
            "gt_full": [int(i) for i in np.flatnonzero(ass == 0)],
            "gt_same_support": [int(i) for i in np.flatnonzero(ss == 0)],
            "gt_full_below_minimum": [int(i) for i in np.flatnonzero(ass < minimum)],
            "gt_same_support_below_minimum": [int(i) for i in np.flatnonzero(ss < minimum)],
        },
        "per_row": rows,
        "summary": summaries,
        "pseudo_anchor_precision": anchor_summary,
        "support_consistency": mass_checks,
        "summary_lines": summary_lines,
    }
