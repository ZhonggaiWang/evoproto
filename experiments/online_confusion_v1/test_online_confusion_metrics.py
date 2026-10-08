"""Meaningful reference tests for GT-only confusion diagnostics."""

import json
import unittest

import numpy as np

from online_confusion_metrics import _average_ranks, _spearman, evaluate_matrices


class ConfusionMetricsTests(unittest.TestCase):
    def evaluate(self, estimate, all_gt=None, same_gt=None, **kwargs):
        return evaluate_matrices(
            estimate, estimate if all_gt is None else all_gt,
            estimate if same_gt is None else same_gt,
            min_gt_row_pixels=kwargs.pop("min_gt_row_pixels", 1), **kwargs,
        )

    def test_direction_is_asymmetric_and_never_symmetrized(self):
        counts = np.array([[80, 0, 0], [0, 80, 20], [0, 5, 95]], dtype=float)
        result = self.evaluate(counts)
        probabilities = result["matrices"]["estimated_row_probabilities"]
        self.assertEqual(probabilities[1][2], 0.2)
        self.assertEqual(probabilities[2][1], 0.05)
        row = result["per_row"][1]["comparisons"]["full_population"]
        self.assertEqual(row["estimated_strongest_targets"], [2])
        self.assertTrue(row["strongest_target_correct"])

    def test_missing_rows_are_null_and_threshold_is_respected(self):
        estimate = np.array([[20, 0, 0], [0, 0, 0], [0, 2, 8]], dtype=float)
        reference = np.array([[20, 0, 0], [0, 20, 0], [0, 1, 3]], dtype=float)
        result = self.evaluate(estimate, reference, reference, min_gt_row_pixels=5)
        self.assertIsNone(result["matrices"]["estimated_row_probabilities"][1])
        self.assertEqual(result["unsupported_rows"]["estimated"], [1])
        self.assertIsNone(result["per_row"][1]["comparisons"]["full_population"])
        self.assertIsNone(result["per_row"][2]["comparisons"]["full_population"])
        self.assertIsNone(result["summary"]["full_population"]["mean_total_variation_distance"])
        json.dumps(result, allow_nan=False)

    def test_perfect_matrices_and_tied_topk_are_perfect(self):
        counts = np.array([
            [80, 2, 3, 1], [2, 60, 20, 20],
            [1, 20, 60, 20], [0, 20, 20, 60],
        ], dtype=float)
        result = self.evaluate(counts, pseudo_gt_counts=np.diag(counts.sum(axis=1)))
        for summary in result["summary"].values():
            self.assertEqual(summary["mean_total_variation_distance"], 0)
            self.assertEqual(summary["strongest_target_accuracy"], 1)
            self.assertEqual(summary["mean_top3_overlap_fraction"], 1)
            self.assertAlmostEqual(summary["mean_offdiagonal_spearman"], 1)
            for pair in summary["topk_foreground_pairs"].values():
                self.assertAlmostEqual(pair["precision_at_k"], 1)
                self.assertEqual(pair["false_positive_pair_fraction"], 0)
        self.assertEqual(result["pseudo_anchor_precision"]["weighted_precision_foreground"], 1)
        self.assertIn("not probability calibration", result["pseudo_anchor_precision"]["meaning"])

    def test_selection_bias_is_distinct_from_same_support_agreement(self):
        selected = np.array([[50, 0, 0], [0, 95, 5], [0, 5, 95]], dtype=float)
        population = np.array([[50, 0, 0], [0, 500, 500], [0, 400, 600]], dtype=float)
        result = self.evaluate(selected, population, selected)
        self.assertEqual(result["summary"]["same_evidence_support"]["mean_total_variation_distance"], 0)
        self.assertAlmostEqual(result["summary"]["full_population"]["mean_total_variation_distance"], 0.4)
        self.assertIn("does not establish", result["conditioning_scope"]["population_warning"])

    def test_anchor_errors_can_exist_even_with_selected_mass_consistency(self):
        estimate = np.array([[10, 0, 0], [0, 80, 20], [0, 20, 80]], dtype=float)
        anchors = np.array([[10, 0, 0], [0, 60, 40], [0, 10, 90]], dtype=float)
        result = self.evaluate(estimate, pseudo_gt_counts=anchors)
        self.assertEqual(result["per_row"][1]["pseudo_anchor_precision"], 0.6)
        self.assertEqual(result["pseudo_anchor_precision"]["weighted_precision_foreground"], 0.75)
        self.assertTrue(result["support_consistency"]["pseudo_anchor_row_mass_matches_estimate"])

    def test_average_rank_and_spearman_handle_ties(self):
        np.testing.assert_array_equal(_average_ranks([4, 1, 1, 3]), [4, 1.5, 1.5, 3])
        self.assertAlmostEqual(_spearman([1, 1, 2, 3], [5, 5, 7, 9]), 1)
        self.assertAlmostEqual(_spearman([1, 1, 2, 3], [9, 9, 7, 5]), -1)
        self.assertIsNone(_spearman([0, 0, 0], [1, 2, 3]))
        self.assertIsNone(_spearman([1], [2]))

    def test_false_positive_pair_precision_excludes_background(self):
        estimate = np.array([[10, 0, 0], [999, 0, 1], [999, 1, 0]], dtype=float)
        reference = np.array([[10, 0, 0], [0, 100, 0], [0, 0, 100]], dtype=float)
        result = self.evaluate(estimate, reference, reference)
        pairs = result["summary"]["full_population"]["topk_foreground_pairs"]["5"]
        self.assertEqual(pairs["estimated_effective_k"], 2)
        self.assertEqual(pairs["precision_at_k"], 0)
        self.assertEqual(pairs["positive_pair_precision"], 0)
        self.assertEqual(pairs["false_positive_pair_fraction"], 1)
        self.assertTrue(all(p["source_class"] > 0 and p["target_class"] > 0 for p in pairs["selected_estimated_pairs"]))

    def test_no_confusion_has_no_rank_or_strongest_target_score(self):
        result = self.evaluate(np.eye(4) * 100)
        row = result["per_row"][1]["comparisons"]["full_population"]
        self.assertIsNone(row["strongest_target_correct"])
        self.assertIsNone(row["offdiagonal_spearman"])
        self.assertIsNone(row["top3_overlap_fraction"])
        self.assertIsNone(result["summary"]["full_population"]["topk_foreground_pairs"]["5"]["precision_at_k"])
        json.dumps(result, allow_nan=False)

    def test_missing_estimated_confusion_is_incorrect_when_reference_has_errors(self):
        estimate = np.eye(3) * 100
        reference = np.array([[100, 0, 0], [0, 80, 20], [0, 5, 95]], dtype=float)
        result = self.evaluate(estimate, reference, reference)
        summary = result["summary"]["full_population"]
        self.assertEqual(summary["strongest_target_evaluable_rows"], 2)
        self.assertEqual(summary["strongest_target_accuracy"], 0)
        self.assertEqual(summary["mean_top3_overlap_fraction"], 0)

    def test_support_mismatch_is_reported_without_claiming_matched_pixels(self):
        counts = np.eye(3) * 100
        result = self.evaluate(counts, same_gt=counts * 0.5, pseudo_gt_counts=counts * 0.5)
        self.assertFalse(result["support_consistency"]["same_support_total_mass_matches_estimate"])
        self.assertFalse(result["support_consistency"]["pseudo_anchor_row_mass_matches_estimate"])

    def test_inputs_are_not_mutated(self):
        counts = np.array([[90, 5, 5], [1, 80, 19], [2, 18, 80]], dtype=float)
        before = counts.copy()
        self.evaluate(counts)
        np.testing.assert_array_equal(counts, before)

    def test_invalid_dimensions_nonfinite_negative_and_options(self):
        good = np.eye(3)
        invalid = [[], np.ones(3), np.ones((2, 3)), np.array([[1, -1], [0, 1]]), np.full((2, 2), np.nan), np.full((2, 2), np.inf)]
        for bad in invalid:
            with self.subTest(bad=repr(bad)), self.assertRaises(ValueError):
                self.evaluate(bad)
        for name in ("all_gt", "same_gt", "pseudo_gt_counts"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.evaluate(good, **{name: np.eye(2)})
        for bad in (-1, float("nan"), float("inf"), True, "x"):
            with self.subTest(minimum=bad), self.assertRaises(ValueError):
                self.evaluate(good, min_gt_row_pixels=bad)
        for bad in (-1, 3, 0.5, True):
            with self.subTest(foreground_start=bad), self.assertRaises(ValueError):
                self.evaluate(good, foreground_start=bad)
        with self.assertRaises(ValueError):
            self.evaluate(np.ones((3, 3)) * 1e308)


if __name__ == "__main__":
    unittest.main(verbosity=2)
