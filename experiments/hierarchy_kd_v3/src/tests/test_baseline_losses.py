"""Behavioral checks for the fixed-weight prototype baseline losses.

Run from the repository root with:
    PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
"""

import unittest
import math

import torch

from model.losses import get_seg_loss, prototype_distillation_loss, prototype_separation_loss


class PrototypeDistillationTests(unittest.TestCase):
    def test_matching_classes_ignore_scale_background_and_new_classes(self):
        teacher = torch.tensor([[9.0, 3.0], [2.0, 0.0], [0.0, 4.0]])
        student = torch.tensor(
            [[-5.0, 1.0], [7.0, 0.0], [0.0, 0.5], [-4.0, -3.0]],
            requires_grad=True,
        )
        loss = prototype_distillation_loss(student, teacher)
        self.assertAlmostEqual(loss.item(), 0.0, places=7)
        loss.backward()
        self.assertTrue(torch.equal(student.grad[0], torch.zeros(2)))
        self.assertTrue(torch.equal(student.grad[-1], torch.zeros(2)))

    def test_same_class_order_matters(self):
        teacher = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        swapped_student = torch.tensor([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
        self.assertAlmostEqual(
            prototype_distillation_loss(swapped_student, teacher).item(), 1.0, places=7
        )

    def test_teacher_is_detached_and_student_gradient_improves_alignment(self):
        teacher = torch.tensor([[0.0, 1.0], [1.0, 0.0]], requires_grad=True)
        student = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        loss = prototype_distillation_loss(student, teacher)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertGreater(student.grad[1].abs().sum().item(), 0.0)
        updated_student = student.detach() - 0.1 * student.grad
        self.assertLess(
            prototype_distillation_loss(updated_student, teacher).item(), loss.item()
        )

    def test_no_old_foreground_returns_differentiable_zero(self):
        for old_count in (0, 1):
            with self.subTest(old_count=old_count):
                teacher = torch.empty((old_count, 2), requires_grad=True)
                student = torch.tensor([[1.0, 2.0], [2.0, 1.0]], requires_grad=True)
                loss = prototype_distillation_loss(student, teacher)
                self.assertEqual(loss.item(), 0.0)
                loss.backward()
                self.assertTrue(torch.equal(student.grad, torch.zeros_like(student)))
                self.assertIsNone(teacher.grad)

    def test_zero_norm_prototypes_are_finite(self):
        student = torch.zeros((2, 3), requires_grad=True)
        teacher = torch.zeros((2, 3), requires_grad=True)
        loss = prototype_distillation_loss(student, teacher)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertIsNone(teacher.grad)

    def test_incompatible_class_counts_fail_explicitly(self):
        with self.assertRaises(ValueError):
            prototype_distillation_loss(torch.ones((2, 3)), torch.ones((3, 3)))


class PrototypeSeparationTests(unittest.TestCase):
    def test_orthogonal_and_negative_pairs_are_inactive(self):
        for foreground in ([[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [-1.0, 0.0]]):
            with self.subTest(foreground=foreground):
                prototypes = torch.tensor([[1.0, 0.0]] + foreground, requires_grad=True)
                loss = prototype_separation_loss(prototypes)
                self.assertEqual(loss.item(), 0.0)
                loss.backward()
                self.assertTrue(torch.equal(prototypes.grad, torch.zeros_like(prototypes)))

    def test_positive_pair_gradient_reduces_similarity(self):
        prototypes = torch.tensor([[0.0, 1.0], [2.0, 0.0], [1.0, 1.0]], requires_grad=True)
        loss = prototype_separation_loss(prototypes)
        self.assertAlmostEqual(loss.item(), 0.5, places=6)
        loss.backward()
        self.assertTrue(torch.equal(prototypes.grad[0], torch.zeros(2)))
        self.assertGreater(prototypes.grad[1:].abs().sum().item(), 0.0)
        updated = prototypes.detach() - 0.1 * prototypes.grad
        self.assertLess(prototype_separation_loss(updated).item(), loss.item())

    def test_background_self_pairs_and_pair_count(self):
        # Three foreground pairs: one identical pair and two orthogonal pairs.
        prototypes = torch.tensor([[100.0, 0.0], [1.0, 0.0], [3.0, 0.0], [0.0, 2.0]])
        self.assertAlmostEqual(prototype_separation_loss(prototypes).item(), 1.0 / 3.0, places=6)
        prototypes[0] = torch.tensor([-100.0, -100.0])
        self.assertAlmostEqual(prototype_separation_loss(prototypes).item(), 1.0 / 3.0, places=6)

    def test_margin_stops_penalty_at_required_separation(self):
        # Foreground cosine similarity is 0.5.
        prototypes = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.5, 3.0 ** 0.5 / 2.0]])
        self.assertAlmostEqual(prototype_separation_loss(prototypes, margin=0.6).item(), 0.0, places=7)
        self.assertAlmostEqual(prototype_separation_loss(prototypes, margin=0.4).item(), 0.01, places=6)

    def test_insufficient_foreground_returns_differentiable_zero(self):
        for class_count in (0, 1, 2):
            with self.subTest(class_count=class_count):
                prototypes = torch.ones((class_count, 3), requires_grad=True)
                loss = prototype_separation_loss(prototypes)
                self.assertEqual(loss.item(), 0.0)
                loss.backward()
                self.assertTrue(torch.equal(prototypes.grad, torch.zeros_like(prototypes)))

    def test_zero_norm_prototypes_are_finite(self):
        prototypes = torch.zeros((3, 2), requires_grad=True)
        loss = prototype_separation_loss(prototypes)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(prototypes.grad).all())


class SegmentationLossTests(unittest.TestCase):
    def test_all_ignored_crop_returns_zero_and_no_gradients(self):
        prediction = torch.randn((2, 4, 3, 3), requires_grad=True)
        labels = torch.full((2, 3, 3), 255, dtype=torch.long)
        loss = get_seg_loss(prediction, labels, class_weight=torch.ones(4))
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertTrue(torch.equal(prediction.grad, torch.zeros_like(prediction)))

    def test_ignored_pixels_do_not_affect_valid_pixel_average(self):
        prediction = torch.zeros((1, 3, 1, 2), requires_grad=True)
        labels = torch.tensor([[[1, 255]]])
        loss = get_seg_loss(prediction, labels, class_weight=torch.ones(3))
        # Existing normalization sums the three class BCE terms per valid pixel.
        self.assertAlmostEqual(loss.item(), 3.0 * math.log(2.0), places=6)
        loss.backward()
        self.assertTrue(torch.equal(prediction.grad[:, :, :, 1], torch.zeros((1, 3, 1))))
        self.assertGreater(prediction.grad[:, :, :, 0].abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
