"""Behavioral contract for confusion-selected prototype separation.

Run with the candidate source on PYTHONPATH. Synthetic noncollinear vectors
make real angular gradients observable; no training image or pixel GT enters
this test. A fixed mean includes satisfied eligible pairs in its denominator.
"""
import inspect
import json
import unittest

import torch
import torch.nn.functional as F

try:
    from model.confusion_prototype_sep import confusion_prototype_sep_loss
except ModuleNotFoundError:
    from confusion_prototype_sep import confusion_prototype_sep_loss


def prototypes():
    # BG0, old foreground1/2, current-new foreground3/4.
    return torch.tensor([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.],
                         [.8, .6, 0.], [.6, .8, 0.]])


def measure(targets, values=None, old_classes=2, margin=0.):
    p = (prototypes() if values is None else values).clone().requires_grad_()
    loss, stats = confusion_prototype_sep_loss(p, targets, old_classes, margin=margin)
    loss.backward()
    return loss.detach(), p.grad.detach(), stats


class ConfusionPrototypeSeparationTests(unittest.TestCase):
    def test_old_new_only_selected_new_prototype_receives_gradient(self):
        loss, gradient, stats = measure(torch.tensor([-1, 3, -1, -1, -1]))
        torch.testing.assert_close(loss, torch.tensor(.64), atol=1e-7, rtol=1e-7)
        self.assertEqual(float(gradient[[0, 1, 2, 4]].abs().sum()), 0.)
        self.assertGreater(float(gradient[3].abs().sum()), 0.)
        self.assertEqual(stats['pair_count'], 1)
        self.assertEqual(stats['old_new_pair_count'], 1)
        self.assertEqual(stats['new_new_pair_count'], 0)

    def test_reversing_old_new_direction_still_detaches_the_old_endpoint(self):
        a = measure(torch.tensor([-1, 3, -1, -1, -1]))
        b = measure(torch.tensor([-1, -1, -1, 1, -1]))
        for actual, expected in zip(b[:2], a[:2]):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(float(b[1][1].abs().sum()), 0.)
        self.assertGreater(float(b[1][3].abs().sum()), 0.)

    def test_one_gradient_step_reduces_selected_old_new_cosine(self):
        p = prototypes().requires_grad_()
        targets = torch.tensor([-1, 3, -1, -1, -1])
        before = F.cosine_similarity(p[1], p[3], dim=0).detach()
        loss, _ = confusion_prototype_sep_loss(p, targets, 2)
        loss.backward()
        after_p = p.detach() - .1 * p.grad
        after = F.cosine_similarity(after_p[1], after_p[3], dim=0)
        self.assertLess(float(after), float(before))
        torch.testing.assert_close(after_p[:3], p.detach()[:3], atol=0, rtol=0)
        self.assertLess(float(confusion_prototype_sep_loss(after_p, targets, 2)[0]), float(loss))

    def test_new_new_both_selected_endpoints_receive_gradient(self):
        loss, gradient, stats = measure(torch.tensor([-1, -1, -1, 4, -1]))
        torch.testing.assert_close(loss, torch.tensor(.96 ** 2), atol=2e-7, rtol=2e-7)
        self.assertEqual(float(gradient[:3].abs().sum()), 0.)
        self.assertGreater(float(gradient[3].abs().sum()), 0.)
        self.assertGreater(float(gradient[4].abs().sum()), 0.)
        self.assertEqual(stats['old_new_pair_count'], 0)
        self.assertEqual(stats['new_new_pair_count'], 1)

    def test_one_gradient_step_reduces_new_new_cosine(self):
        p = prototypes().requires_grad_()
        targets = torch.tensor([-1, -1, -1, 4, -1])
        before = F.cosine_similarity(p[3], p[4], dim=0).detach()
        loss, _ = confusion_prototype_sep_loss(p, targets, 2)
        loss.backward()
        after_p = p.detach() - .1 * p.grad
        after = F.cosine_similarity(after_p[3], after_p[4], dim=0)
        self.assertLess(float(after), float(before))

    def test_reverse_duplicate_old_new_does_not_double_loss_or_gradients(self):
        single = measure(torch.tensor([-1, 3, -1, -1, -1]))
        duplicate = measure(torch.tensor([-1, 3, -1, 1, -1]))
        for actual, expected in zip(duplicate[:2], single[:2]):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(duplicate[2]['pair_count'], 1)

    def test_reverse_duplicate_new_new_does_not_double_loss_or_gradients(self):
        single = measure(torch.tensor([-1, -1, -1, 4, -1]))
        duplicate = measure(torch.tensor([-1, -1, -1, 4, 3]))
        for actual, expected in zip(duplicate[:2], single[:2]):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(duplicate[2]['pair_count'], 1)

    def test_satisfied_pair_stays_in_fixed_mean_denominator(self):
        values = prototypes()
        values[4] = torch.tensor([1., -1., 0.])
        single = measure(torch.tensor([-1, 3, -1, -1, -1]), values=values)
        two = measure(torch.tensor([-1, 3, 4, -1, -1]), values=values)
        torch.testing.assert_close(two[0], single[0] / 2., atol=0, rtol=0)
        torch.testing.assert_close(two[1][3], single[1][3] / 2., atol=0, rtol=0)
        self.assertEqual(float(two[1][4].abs().sum()), 0.)
        self.assertEqual(two[2]['pair_count'], 2)
        self.assertEqual(two[2]['active_pair_count'], 1)

    def test_margin_is_applied_before_relu_and_square(self):
        value, gradient, stats = measure(torch.tensor([-1, 3, -1, -1, -1]), margin=.5)
        torch.testing.assert_close(value, torch.tensor(.09), atol=1e-7, rtol=1e-7)
        self.assertGreater(float(gradient[3].abs().sum()), 0.)
        self.assertEqual(stats['margin'], .5)

    def test_satisfied_and_exact_margin_pairs_have_zero_gradient(self):
        orthogonal = prototypes(); orthogonal[3] = torch.tensor([0., 1., 0.])
        for values, margin in [(prototypes(), 1.), (orthogonal, 0.)]:
            with self.subTest(margin=margin):
                value, gradient, stats = measure(torch.tensor([-1, 3, -1, -1, -1]), values=values, margin=margin)
                self.assertEqual(float(value), 0.)
                self.assertEqual(float(gradient.abs().sum()), 0.)
                self.assertEqual(stats['pair_count'], 1)
                self.assertEqual(stats['active_pair_count'], 0)

    def test_old_old_directions_are_skipped_and_graph_connected(self):
        value, gradient, stats = measure(torch.tensor([-1, 2, 1, -1, -1]))
        self.assertEqual(float(value), 0.)
        self.assertEqual(float(gradient.abs().sum()), 0.)
        self.assertEqual(stats['pair_count'], 0)
        self.assertEqual(stats['skipped_old_old_directions'], 2)

    def test_background_and_self_directions_cannot_create_repulsion(self):
        value, gradient, stats = measure(torch.tensor([3, 0, 2, 3, -1]))
        self.assertEqual(float(value), 0.)
        self.assertEqual(float(gradient.abs().sum()), 0.)
        self.assertEqual(stats['pair_count'], 0)
        self.assertEqual(stats['skipped_background_directions'], 2)
        self.assertEqual(stats['skipped_self_directions'], 2)

    def test_empty_selector_returns_scalar_graph_connected_zero(self):
        p = prototypes().requires_grad_()
        value, stats = confusion_prototype_sep_loss(p, torch.full((5,), -1), 2)
        self.assertEqual(value.ndim, 0)
        self.assertTrue(value.requires_grad)
        value.backward()
        self.assertIsNotNone(p.grad)
        self.assertEqual(float(value), 0.)
        self.assertEqual(float(p.grad.abs().sum()), 0.)
        self.assertEqual(stats['pair_count'], 0)

    def test_empty_selector_large_finite_matrix_does_not_overflow_graph_zero(self):
        p = torch.full((5, 3), 3e38, dtype=torch.float32, requires_grad=True)
        value, _ = confusion_prototype_sep_loss(p, torch.full((5,), -1), 2)
        value.backward()
        self.assertTrue(bool(torch.isfinite(value)))
        self.assertEqual(float(value), 0.)
        self.assertEqual(float(p.grad.abs().sum()), 0.)

    def test_positive_per_class_rescaling_preserves_loss_and_angular_gradient(self):
        values = prototypes()
        scale = torch.tensor([2., 5., .5, 3., 7.])[:, None]
        targets = torch.tensor([-1, 3, -1, 4, -1])
        original = measure(targets, values=values)
        scaled = measure(targets, values=values * scale)
        torch.testing.assert_close(scaled[0], original[0], atol=2e-7, rtol=2e-7)
        torch.testing.assert_close(scaled[1] * scale, original[1], atol=2e-7, rtol=2e-7)

    def test_old_classes_is_foreground_count_with_inclusive_old_ID_boundary(self):
        # ID2 is still old when old_classes=2; only new ID3 may move.
        value, gradient, _ = measure(torch.tensor([-1, -1, 3, -1, -1]))
        self.assertGreater(float(value), 0.)
        self.assertEqual(float(gradient[:3].abs().sum()), 0.)
        self.assertGreater(float(gradient[3].abs().sum()), 0.)

    def test_nonparticipating_zero_vectors_are_permitted(self):
        values = prototypes(); values[0].zero_(); values[2].zero_(); values[4].zero_()
        value, gradient, _ = measure(torch.tensor([-1, 3, -1, -1, -1]), values=values)
        self.assertTrue(bool(torch.isfinite(value)))
        self.assertGreater(float(gradient[3].abs().sum()), 0.)
        self.assertEqual(float(gradient[[0, 2, 4]].abs().sum()), 0.)

    def test_participating_zero_norm_vector_is_rejected(self):
        for class_id in [1, 3]:
            with self.subTest(class_id=class_id), self.assertRaises(ValueError):
                values = prototypes(); values[class_id].zero_()
                confusion_prototype_sep_loss(values, torch.tensor([-1, 3, -1, -1, -1]), 2)

    def test_class_target_length_rank_dtype_and_range_are_validated(self):
        malformed = [torch.full((4,), -1), torch.full((5, 1), -1),
                     torch.tensor([-1., 3., -1., -1., -1.]), torch.tensor([False] * 5),
                     torch.tensor([-1, 5, -1, -1, -1]), torch.tensor([-1, -2, -1, -1, -1])]
        for targets in malformed:
            with self.subTest(targets=targets.tolist()), self.assertRaises(ValueError):
                confusion_prototype_sep_loss(prototypes(), targets, 2)

    def test_nonmatrix_empty_or_nonfloating_prototypes_are_rejected(self):
        malformed = [torch.zeros(5), torch.zeros(1, 5, 3), torch.zeros(5, 0),
                     torch.zeros(1, 3), prototypes().long()]
        for p in malformed:
            with self.subTest(shape=tuple(p.shape), dtype=p.dtype), self.assertRaises(ValueError):
                confusion_prototype_sep_loss(p, torch.full((p.shape[0],), -1), 2)

    def test_nonfinite_prototypes_and_margin_are_rejected(self):
        for bad in [float('nan'), float('inf'), -float('inf')]:
            with self.subTest(prototype_value=bad), self.assertRaises(ValueError):
                values = prototypes(); values[3, 0] = bad
                confusion_prototype_sep_loss(values, torch.tensor([-1, 3, -1, -1, -1]), 2)
            with self.subTest(margin=bad), self.assertRaises(ValueError):
                confusion_prototype_sep_loss(prototypes(), torch.tensor([-1, 3, -1, -1, -1]), 2, margin=bad)

    def test_illegal_old_class_and_margin_boundaries_are_rejected(self):
        for count in [-1, 5, 6, 1.5, True]:
            with self.subTest(old_classes=count), self.assertRaises(ValueError):
                confusion_prototype_sep_loss(prototypes(), torch.full((5,), -1), count)
        for margin in [-1.1, 1.1, True, False]:
            with self.subTest(margin=margin), self.assertRaises(ValueError):
                confusion_prototype_sep_loss(prototypes(), torch.full((5,), -1), 2, margin=margin)

    def test_legal_zero_old_and_all_old_foreground_boundaries(self):
        targets = torch.tensor([-1, 3, -1, -1, -1])
        all_new = measure(targets, old_classes=0)
        self.assertEqual(all_new[2]['new_new_pair_count'], 1)
        self.assertGreater(float(all_new[1][1].abs().sum()), 0.)
        self.assertGreater(float(all_new[1][3].abs().sum()), 0.)
        all_old = measure(targets, old_classes=4)
        self.assertEqual(float(all_old[0]), 0.)
        self.assertEqual(float(all_old[1].abs().sum()), 0.)
        self.assertEqual(all_old[2]['pair_count'], 0)

    def test_nonfinite_unused_background_is_rejected_before_graph_zero(self):
        values = prototypes(); values[0, 0] = float('nan')
        with self.assertRaises(ValueError):
            confusion_prototype_sep_loss(values, torch.full((5,), -1), 2)

    def test_stats_are_json_serializable_and_training_API_has_no_GT_or_rate_weight(self):
        _, _, stats = measure(torch.tensor([-1, 3, -1, 4, -1]))
        json.dumps(stats, allow_nan=False)
        names = inspect.signature(confusion_prototype_sep_loss).parameters
        self.assertFalse(any(name in {'gt', 'ground_truth', 'gt_labels', 'rates', 'pair_weights', 'confusion_rates'} for name in names))


if __name__ == '__main__':
    unittest.main()
