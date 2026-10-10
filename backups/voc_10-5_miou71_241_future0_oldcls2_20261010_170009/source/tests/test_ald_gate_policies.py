"""CPU behavioral tests for optional NEW rescue, isolated from primary training."""

import ast
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from tools.ald_gate_policies import ALDGateResult, gate_ald_classes


def load_actual_legacy_gate():
    source = Path(__file__).resolve().parents[1] / "utils/camutils.py"
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "cam_high_pass_filter")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[function.name]


class ALDGatePolicyTests(unittest.TestCase):
    def test_legacy_matches_actual_frozen_source(self):
        reference = load_actual_legacy_gate()
        peaks = torch.tensor([[20., 5., 2., 2.], [8., 3., 4., 1.],
                              [4., 4., 4., 4.], [12.5, 13., 0., 1.], [100., 2., 3., 4.]])
        labels = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1],
                               [0, 1, 1, 1], [1, 0, 1, 0], [0, 0, 0, 0]])
        # Only the actual legacy reference needs its hardcoded .cuda neutralized.
        with patch.object(torch.Tensor, "cuda", lambda self, *args, **kwargs: self):
            for dtype in (torch.long, torch.int8, torch.uint8, torch.float32, torch.float64):
                for threshold in (0., 4., 12.5, 200.):
                    with self.subTest(dtype=dtype, threshold=threshold):
                        original_labels = labels.to(dtype)
                        expected = reference(peaks, original_labels, threshold)
                        result = gate_ald_classes(peaks, original_labels, old_classes=2,
                                                  threshold=threshold)
                        torch.testing.assert_close(result.gate_labels, expected)
                        torch.testing.assert_close(result.legacy_gate_labels, expected)
                        self.assertFalse(result.new_rescue_mask.any())

    def test_old_fallback_plus_two_weak_new_restores_strongest_positive_new(self):
        result = gate_ald_classes(torch.tensor([[8., 0., 2., 3., 100.]]),
                                  torch.tensor([[1, 0, 1, 1, 0]]), old_classes=2,
                                  policy="new_fallback")
        torch.testing.assert_close(result.legacy_gate_labels, torch.tensor([[1, 0, 0, 0, 0]]))
        torch.testing.assert_close(result.gate_labels, torch.tensor([[1, 0, 0, 1, 0]]))
        torch.testing.assert_close(result.new_rescue_mask, torch.tensor([True]))

    def test_old_directly_passes_and_new_is_still_rescued(self):
        result = gate_ald_classes(torch.tensor([[30., 20., 4., 3.]]),
                                  torch.ones(1, 4), old_classes=2, policy="new_fallback")
        torch.testing.assert_close(result.legacy_gate_labels, torch.tensor([[1., 1., 0., 0.]]))
        torch.testing.assert_close(result.gate_labels, torch.tensor([[1., 1., 1., 0.]]))
        self.assertTrue(result.new_rescue_mask.item())

    def test_existing_new_survivor_prevents_extra_rescue(self):
        result = gate_ald_classes(torch.tensor([[30., 1., 12.5, 10.]]),
                                  torch.ones(1, 4, dtype=torch.long), old_classes=2,
                                  policy="new_fallback")
        torch.testing.assert_close(result.gate_labels, torch.tensor([[1, 0, 1, 0]]))
        torch.testing.assert_close(result.gate_labels, result.legacy_gate_labels)
        self.assertFalse(result.new_rescue_mask.item())

    def test_legacy_fallback_already_selecting_new_is_unchanged(self):
        result = gate_ald_classes(torch.tensor([[1., 0., 4., 3.]]),
                                  torch.tensor([[1, 0, 1, 1]]), old_classes=2,
                                  policy="new_fallback")
        torch.testing.assert_close(result.gate_labels, torch.tensor([[0, 0, 1, 0]]))
        torch.testing.assert_close(result.gate_labels, result.legacy_gate_labels)
        self.assertFalse(result.new_rescue_mask.item())

    def test_absent_new_positive_and_empty_positive_do_not_create_labels(self):
        result = gate_ald_classes(torch.tensor([[2., 1., 100., 90.], [100., 80., 60., 40.]]),
                                  torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0]]),
                                  old_classes=2, policy="new_fallback")
        torch.testing.assert_close(result.gate_labels, torch.tensor([[1, 0, 0, 0], [0, 0, 0, 0]]))
        self.assertFalse(result.new_rescue_mask.any())

    def test_step1_and_step2_use_correct_current_new_boundary(self):
        for old, total in ((10, 15), (15, 20)):
            with self.subTest(old=old, total=total):
                peaks = torch.zeros(1, total)
                labels = torch.zeros(1, total, dtype=torch.long)
                peaks[0, old - 1], peaks[0, old], peaks[0, total - 1] = 10., 3., 4.
                labels[0, old - 1] = labels[0, old] = labels[0, total - 1] = 1
                result = gate_ald_classes(peaks, labels, old_classes=old, policy="new_fallback")
                self.assertEqual(result.legacy_gate_labels.nonzero().tolist(), [[0, old - 1]])
                self.assertEqual(result.gate_labels.nonzero().tolist(), [[0, old - 1], [0, total - 1]])
                self.assertTrue(result.new_rescue_mask.item())

    def test_ties_choose_lowest_positive_index_in_both_pools(self):
        result = gate_ald_classes(torch.full((1, 5), 2.), torch.tensor([[0, 1, 1, 1, 1]]),
                                  old_classes=3, policy="new_fallback")
        torch.testing.assert_close(result.legacy_gate_labels, torch.tensor([[0, 1, 0, 0, 0]]))
        torch.testing.assert_close(result.gate_labels, torch.tensor([[0, 1, 0, 1, 0]]))

    def test_zero_old_classes_needs_no_extra_rescue(self):
        result = gate_ald_classes(torch.tensor([[1., 3., 2.]]), torch.ones(1, 3),
                                  old_classes=0, policy="new_fallback")
        torch.testing.assert_close(result.gate_labels, torch.tensor([[0., 1., 0.]]))
        self.assertFalse(result.new_rescue_mask.item())

    def test_threshold_zero_retains_zero_peak_positive_classes(self):
        result = gate_ald_classes(torch.zeros(1, 3), torch.tensor([[1, 0, 1]]),
                                  old_classes=1, threshold=0., policy="new_fallback")
        torch.testing.assert_close(result.gate_labels, torch.tensor([[1, 0, 1]]))
        self.assertFalse(result.new_rescue_mask.item())

    def test_noncontiguous_inputs_and_per_image_mask(self):
        peaks = torch.tensor([[20., 20., 2.], [0., 0., 1.], [3., 13., 1.], [2., 4., 1.]]).t()
        labels = torch.tensor([[1, 1, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1]]).t()
        self.assertFalse(peaks.is_contiguous())
        self.assertFalse(labels.is_contiguous())
        result = gate_ald_classes(peaks, labels, old_classes=2, policy="new_fallback")
        torch.testing.assert_close(result.gate_labels,
                                  torch.tensor([[1, 0, 1, 0], [1, 0, 1, 0], [0, 0, 1, 0]]))
        torch.testing.assert_close(result.new_rescue_mask, torch.tensor([True, False, False]))

    def test_outputs_do_not_alias_and_inputs_are_unmodified(self):
        peaks = torch.tensor([[8., 2., 3.]], requires_grad=True)
        labels = torch.ones(1, 3, requires_grad=True)
        original_peaks, original_labels = peaks.detach().clone(), labels.detach().clone()
        result = gate_ald_classes(peaks, labels, old_classes=1, policy="new_fallback")
        self.assertIsInstance(result, ALDGateResult)
        self.assertEqual(result.new_rescue_mask.dtype, torch.bool)
        self.assertEqual(result.new_rescue_mask.shape, (1,))
        self.assertFalse(result.gate_labels.requires_grad)
        self.assertFalse(result.legacy_gate_labels.requires_grad)
        gate_copy, legacy_copy = result.gate_labels.clone(), result.legacy_gate_labels.clone()
        result.gate_labels.fill_(9)
        torch.testing.assert_close(result.legacy_gate_labels, legacy_copy)
        result.legacy_gate_labels.fill_(-1)
        torch.testing.assert_close(result.gate_labels, torch.full_like(gate_copy, 9))
        result.new_rescue_mask.fill_(False)
        torch.testing.assert_close(peaks, original_peaks)
        torch.testing.assert_close(labels, original_labels)

    def test_empty_batch_is_well_defined(self):
        result = gate_ald_classes(torch.empty(0, 15), torch.empty(0, 15, dtype=torch.long),
                                  old_classes=10, policy="new_fallback")
        self.assertEqual(result.gate_labels.shape, (0, 15))
        self.assertEqual(result.legacy_gate_labels.shape, (0, 15))
        self.assertEqual(result.new_rescue_mask.shape, (0,))

    def test_supported_float_peak_dtypes_and_label_dtype_preserved(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                result = gate_ald_classes(torch.tensor([[20., 3., 2.]], dtype=dtype),
                                          torch.ones(1, 3, dtype=dtype), old_classes=1,
                                          policy="new_fallback")
                torch.testing.assert_close(result.gate_labels, torch.tensor([[1., 1., 0.]], dtype=dtype))
                self.assertEqual(result.gate_labels.dtype, dtype)

    def test_invalid_old_boundary_rejected(self):
        for value in (-1, 3, 4):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "old_classes"):
                gate_ald_classes(torch.ones(1, 3), torch.ones(1, 3), old_classes=value)
        for value in (True, 1., "1", torch.tensor(1)):
            with self.subTest(value=value), self.assertRaisesRegex(TypeError, "old_classes"):
                gate_ald_classes(torch.ones(1, 3), torch.ones(1, 3), old_classes=value)

    def test_invalid_threshold_rejected(self):
        for value in (-1., float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "threshold"):
                gate_ald_classes(torch.ones(1, 3), torch.ones(1, 3), old_classes=1, threshold=value)
        for value in (True, "12.5", 2j, torch.tensor(12.5)):
            with self.subTest(value=value), self.assertRaisesRegex(TypeError, "threshold"):
                gate_ald_classes(torch.ones(1, 3), torch.ones(1, 3), old_classes=1, threshold=value)

    def test_invalid_policy_rejected(self):
        with self.assertRaisesRegex(ValueError, "policy"):
            gate_ald_classes(torch.ones(1, 3), torch.ones(1, 3), old_classes=1, policy="auto")

    def test_invalid_peaks_rejected(self):
        for value in (-.1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "cam_peak"):
                gate_ald_classes(torch.tensor([[1., value, 2.]]), torch.ones(1, 3), old_classes=1)
        for dtype in (torch.long, torch.bool, torch.complex64):
            with self.subTest(dtype=dtype), self.assertRaisesRegex(TypeError, "cam_peak"):
                gate_ald_classes(torch.ones(1, 3, dtype=dtype), torch.ones(1, 3), old_classes=1)

    def test_nonbinary_and_invalid_label_dtype_rejected(self):
        for value in (-1., 2., .5, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "binary"):
                gate_ald_classes(torch.ones(1, 3), torch.tensor([[1., value, 0.]]), old_classes=1)
        for dtype in (torch.bool, torch.complex64):
            with self.subTest(dtype=dtype), self.assertRaisesRegex(TypeError, "cls_labels"):
                gate_ald_classes(torch.ones(1, 3), torch.ones(1, 3, dtype=dtype), old_classes=1)

    def test_invalid_shape_device_and_nontensor_rejected(self):
        for peaks, labels in ((torch.ones(3), torch.ones(3)),
                              (torch.ones(1, 3), torch.ones(1, 4)),
                              (torch.empty(1, 0), torch.empty(1, 0))):
            with self.subTest(shape=peaks.shape), self.assertRaises(ValueError):
                gate_ald_classes(peaks, labels, old_classes=0)
        with self.assertRaisesRegex(ValueError, "share a device"):
            gate_ald_classes(torch.ones(1, 3), torch.ones(1, 3, device="meta"), old_classes=1)
        with self.assertRaisesRegex(ValueError, "materialized"):
            gate_ald_classes(torch.ones(1, 3, device="meta"), torch.ones(1, 3, device="meta"), old_classes=1)
        with self.assertRaisesRegex(TypeError, "tensors"):
            gate_ald_classes([[1., 2., 3.]], torch.ones(1, 3), old_classes=1)


if __name__ == "__main__":
    unittest.main()
