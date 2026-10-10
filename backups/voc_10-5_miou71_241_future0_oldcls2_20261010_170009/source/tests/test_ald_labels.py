"""CPU behavioral checks for the isolated ALD teacher-label fusion rules."""

import ast
from pathlib import Path
import unittest

import torch

from model.losses import get_seg_loss
from utils.ald import ALDLabelFusionResult, fuse_ald_labels


def load_actual_legacy_fusion():
    """Read the frozen helper without importing unrelated training utilities."""
    source = Path(__file__).resolve().parents[1] / "utils" / "camutils.py"
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "get_mixed_label")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[function.name]


class ALDLabelFusionTests(unittest.TestCase):
    def setUp(self):
        # Rejected new class over old/BG, existing ignore, filtered old class,
        # background CAM, retained new class, and the first/last new class.
        self.before = torch.tensor([[[12, 12, 255, 7, 0, 13, 11, 15]]])
        self.after = torch.tensor([[[255, 255, 255, 255, 0, 13, 255, 15]]])
        self.teacher = torch.tensor([[[3, 0, 4, 5, 6, 7, 10, 0]]])

    def fuse(self, mode="legacy", **overrides):
        arguments = dict(cam_before=self.before, cam_after=self.after,
                         teacher_labels=self.teacher, total_classes=15,
                         new_classes=5, mode=mode)
        arguments.update(overrides)
        return fuse_ald_labels(**arguments)

    def test_legacy_matches_actual_frozen_helper(self):
        original_teacher = self.teacher.clone()
        actual_helper = load_actual_legacy_fusion()
        expected = actual_helper(self.after.clone(), self.teacher.clone(), 15, 5)
        result = self.fuse()
        self.assertIsInstance(result, ALDLabelFusionResult)
        torch.testing.assert_close(result.labels, expected)
        torch.testing.assert_close(result.labels, torch.tensor([[[3, 0, 4, 5, 6, 13, 10, 15]]]))
        torch.testing.assert_close(self.teacher, original_teacher)
        self.assertFalse(result.retained_ignore_mask.any())
        self.assertEqual(result.rejected_mask.sum().item(), 3)

    def test_preserve_rejected_only_ignores_newly_rejected_current_classes(self):
        result = self.fuse("preserve_rejected")
        torch.testing.assert_close(result.labels, torch.tensor([[[255, 255, 4, 5, 6, 13, 255, 15]]]))
        expected = torch.tensor([[[True, True, False, False, False, False, True, False]]])
        torch.testing.assert_close(result.rejected_mask, expected)
        torch.testing.assert_close(result.retained_ignore_mask, expected)

    def test_preserve_background_keeps_teacher_foreground_supervision(self):
        result = self.fuse("preserve_background")
        torch.testing.assert_close(result.labels, torch.tensor([[[3, 255, 4, 5, 6, 13, 10, 15]]]))
        self.assertEqual(result.rejected_mask.sum().item(), 3)
        torch.testing.assert_close(
            result.retained_ignore_mask,
            torch.tensor([[[False, True, False, False, False, False, False, False]]]),
        )

    def test_second_stage_respects_old_new_class_boundaries(self):
        result = fuse_ald_labels(
            torch.tensor([[[15, 16, 20, 255, 0]]]),
            torch.full((1, 1, 5), 255, dtype=torch.long),
            torch.tensor([[[4, 5, 0, 6, 2]]]),
            total_classes=20, new_classes=5, mode="preserve_rejected",
        )
        torch.testing.assert_close(result.labels, torch.tensor([[[4, 255, 255, 6, 2]]]))
        self.assertEqual(result.rejected_mask.sum().item(), 2)

    def test_identical_cams_disable_rejection_in_every_mode(self):
        expected = torch.tensor([[[12, 12, 4, 5, 6, 13, 11, 15]]])
        for mode in ("legacy", "preserve_rejected", "preserve_background"):
            with self.subTest(mode=mode):
                result = self.fuse(mode, cam_after=self.before)
                torch.testing.assert_close(result.labels, expected)
                self.assertFalse(result.rejected_mask.any())
                self.assertFalse(result.retained_ignore_mask.any())

    def test_inputs_are_unchanged_and_returned_labels_never_alias_teacher(self):
        originals = tuple(value.clone() for value in (self.before, self.after, self.teacher))
        for mode in ("legacy", "preserve_rejected", "preserve_background"):
            with self.subTest(mode=mode):
                result = self.fuse(mode)
                self.assertEqual(result.labels.dtype, torch.long)
                self.assertEqual(result.rejected_mask.dtype, torch.bool)
                self.assertEqual(result.retained_ignore_mask.dtype, torch.bool)
                self.assertEqual(result.labels.device, self.teacher.device)
                result.labels.fill_(99)
                for actual, expected in zip((self.before, self.after, self.teacher), originals):
                    torch.testing.assert_close(actual, expected)

    def test_returned_masks_do_not_alias_each_other(self):
        result = self.fuse("preserve_rejected")
        result.rejected_mask.zero_()
        self.assertEqual(result.retained_ignore_mask.sum().item(), 3)
        self.assertEqual((result.labels == 255).sum().item(), 3)

    def test_mixed_integer_dtypes_and_noncontiguous_inputs(self):
        before = torch.tensor([[[12, 255], [0, 7]]], dtype=torch.uint8).transpose(1, 2)
        after = torch.tensor([[[255, 255], [0, 7]]], dtype=torch.int16).transpose(1, 2)
        teacher = torch.tensor([[[3, 4], [5, 6]]], dtype=torch.int32).transpose(1, 2)
        self.assertFalse(before.is_contiguous())
        result = self.fuse("preserve_rejected", cam_before=before, cam_after=after,
                           teacher_labels=teacher)
        torch.testing.assert_close(result.labels, torch.tensor([[[255, 5], [4, 6]]]))

    def test_all_rejected_pixels_have_finite_zero_ce_and_zero_gradient(self):
        result = self.fuse(
            "preserve_rejected", cam_before=torch.full((1, 2, 2), 12),
            cam_after=torch.full((1, 2, 2), 255),
            teacher_labels=torch.full((1, 2, 2), 3),
        )
        logits = torch.randn(1, 16, 2, 2, requires_grad=True)
        loss = get_seg_loss(logits, result.labels, class_weight=torch.ones(16))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

    def test_background_only_ignore_removes_only_its_pixel_ce_gradient(self):
        result = self.fuse("preserve_background")
        logits = torch.zeros(1, 16, 1, 8, requires_grad=True)
        loss = get_seg_loss(logits, result.labels, class_weight=torch.ones(16))
        loss.backward()
        torch.testing.assert_close(logits.grad[..., 1], torch.zeros_like(logits.grad[..., 1]))
        self.assertGreater(logits.grad[..., 0].abs().sum().item(), 0.0)
        self.assertGreater(logits.grad[..., 5].abs().sum().item(), 0.0)

    def test_empty_batch_has_empty_independent_outputs(self):
        empty = torch.empty((0, 2, 3), dtype=torch.long)
        result = self.fuse("preserve_rejected", cam_before=empty, cam_after=empty,
                           teacher_labels=empty)
        self.assertEqual(result.labels.shape, empty.shape)
        self.assertEqual(result.rejected_mask.numel(), 0)
        self.assertEqual(result.retained_ignore_mask.numel(), 0)

    def test_scalar_configuration_and_mode_errors_are_explicit(self):
        for overrides, error in (
            ({"new_classes": 0}, ValueError), ({"new_classes": 16}, ValueError),
            ({"total_classes": 0}, ValueError), ({"ignore_index": 15}, ValueError),
            ({"ignore_index": 2**63}, ValueError),
            ({"new_classes": True}, TypeError), ({"total_classes": 15.0}, TypeError),
            ({"ignore_index": 255.0}, TypeError), ({"mode": "off"}, ValueError),
            ({"mode": None}, TypeError),
        ):
            with self.subTest(overrides=overrides), self.assertRaises(error):
                self.fuse(**overrides)

    def test_shape_errors_are_explicit(self):
        for name in ("cam_before", "cam_after", "teacher_labels"):
            for value in (torch.zeros((1, 8), dtype=torch.long),
                          torch.zeros((1, 1, 7), dtype=torch.long)):
                with self.subTest(name=name, shape=value.shape), self.assertRaises(ValueError):
                    self.fuse(**{name: value})

    def test_non_tensor_float_and_bool_labels_are_rejected(self):
        for name in ("cam_before", "cam_after", "teacher_labels"):
            for value in ([0], torch.zeros_like(self.before, dtype=torch.float32),
                          torch.zeros_like(self.before, dtype=torch.bool)):
                with self.subTest(name=name, value_type=type(value)), self.assertRaises(TypeError):
                    self.fuse(**{name: value})

    def test_device_mismatch_fails_without_using_gpu(self):
        meta = torch.empty(self.before.shape, dtype=torch.long, device="meta")
        with self.assertRaisesRegex(ValueError, "same device"):
            self.fuse(cam_after=meta)
        with self.assertRaisesRegex(ValueError, "concrete device"):
            self.fuse(cam_before=meta, cam_after=meta, teacher_labels=meta)

    def test_cam_ids_outside_current_label_space_are_rejected(self):
        for name in ("cam_before", "cam_after"):
            for invalid in (-1, 16, 254, 256):
                value = self.before.clone()
                value[..., 0] = invalid
                with self.subTest(name=name, invalid=invalid), self.assertRaisesRegex(ValueError, name):
                    self.fuse(**{name: value})

    def test_teacher_must_contain_previous_stage_argmax_ids(self):
        for invalid in (-1, 11, 15, 255):
            value = self.teacher.clone()
            value[..., 0] = invalid
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "teacher_labels"):
                self.fuse(teacher_labels=value)

    def test_unsigned_labels_do_not_wrap_large_configuration_values(self):
        for name in ("cam_before", "cam_after"):
            inputs = dict(cam_before=torch.zeros((1, 1, 1), dtype=torch.uint8),
                          cam_after=torch.zeros((1, 1, 1), dtype=torch.uint8),
                          teacher_labels=torch.zeros((1, 1, 1), dtype=torch.uint8),
                          ignore_index=511)
            inputs[name].fill_(255)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                self.fuse(**inputs)
        result = self.fuse(cam_before=torch.full((1, 1, 1), 45, dtype=torch.uint8),
                           cam_after=torch.full((1, 1, 1), 45, dtype=torch.uint8),
                           teacher_labels=torch.zeros((1, 1, 1), dtype=torch.uint8),
                           total_classes=300, new_classes=5, ignore_index=511)
        self.assertEqual(result.labels.item(), 0)


if __name__ == "__main__":
    unittest.main()
