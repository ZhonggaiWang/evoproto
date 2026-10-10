"""Numerical contracts for paper objectives and the per-pair hinge ablation."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from model.losses import cpa_loss, rcpl_loss
from utils.image_ald import filter_old_class_labels
from utils.prototype_confusion import (
    confusion_counterparts, confusion_update_due, symmetric_confusion,
)


class PaperPrototypeLossTests(unittest.TestCase):
    def test_confusion_normalizes_reference_rows_before_symmetrizing(self):
        counts = torch.tensor([[8, 2, 0], [1, 1, 2], [0, 0, 0]])
        original = counts.clone()
        expected = torch.tensor([[0., .45, 0.], [.45, 0., .5], [0., .5, 0.]])
        torch.testing.assert_close(symmetric_confusion(counts), expected)
        torch.testing.assert_close(counts, original)

    def test_each_class_selects_one_opposite_group_with_directed_weights(self):
        scores = torch.tensor([[0., .9, .02, .01], [.9, 0., .03, .005],
                               [.02, .03, 0., .9], [.01, .005, .9, 0.]],
                              requires_grad=True)
        counterparts, weights = confusion_counterparts(scores, old_count=2)
        torch.testing.assert_close(counterparts, torch.tensor([2, 2, 1, 0]))
        expected = torch.zeros(4, 4)
        for i, j in enumerate([2, 2, 1, 0]):
            expected[i, j] = torch.tanh(scores.detach()[i, j] / .5)
        torch.testing.assert_close(weights, expected)
        self.assertFalse(weights.requires_grad)
        self.assertGreater(weights[0, 2].item(), 0)
        self.assertEqual(weights[2, 0].item(), 0)  # Selection need not be reciprocal.

    def test_zero_confusion_does_not_add_weight_on_tied_counterparts(self):
        counterparts, weights = confusion_counterparts(torch.zeros(4, 4), 2)
        torch.testing.assert_close(counterparts, torch.tensor([2, 2, 0, 0]))
        self.assertEqual(weights.count_nonzero().item(), 0)

    def test_refresh_after_completed_4000_and_every_2000_updates(self):
        self.assertEqual([i for i in range(8000) if confusion_update_due(i)],
                         [4000, 6000])
        self.assertEqual([i for i in range(4000) if confusion_update_due(i)], [])
        self.assertEqual([i for i in range(8000) if confusion_update_due(i, 2000)], [2000, 4000, 6000])
        self.assertEqual([i for i in range(12) if confusion_update_due(i, 3, 4)], [3, 7, 11])

    def test_both_trainers_delay_confusion_independently_of_loss_warmup(self):
        root = Path(__file__).resolve().parents[1]
        args = SimpleNamespace(confusion_reweight=True, loss_warmup_iters=2000,
                               confusion_start_iter=4000, confusion_interval=2000)
        for filename in ('Trainer.py', 'Trainer_coco.py'):
            tree = ast.parse((root / 'continual' / filename).read_text())
            gate = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                        and any(isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                                and value.func.id == 'confusion_update_due' for value in ast.walk(node.test)))
            for completed in (1999, 2000, 3999, 4000, 5999, 6000):
                due = eval(compile(ast.Expression(gate.test), filename, 'eval'),
                           {'args': args, 'n_iter': completed, 'confusion_update_due': confusion_update_due})
                self.assertEqual(due, completed in (4000, 6000))

    def test_rcpl_has_base_negative_cosines_and_directed_extra_weight(self):
        # Row 0 is background. Its antipodal relation is still penalized by Eq. (8).
        prototypes = torch.tensor([[2., 0.], [-3., 0.], [0., 4.]], requires_grad=True)
        weights = torch.tensor([[0., .5, 0.], [0., 0., 0.], [0., 0., 0.]],
                               requires_grad=True)
        self.assertAlmostEqual(rcpl_loss(prototypes).item(), -2 / 3, places=6)
        loss = rcpl_loss(prototypes, weights)
        self.assertAlmostEqual(loss.item(), -2.5 / 3, places=6)
        loss.backward()
        self.assertIsNone(weights.grad)
        self.assertTrue(torch.isfinite(prototypes.grad).all())

    def test_rcpl_selected_pairs_strengthen_repulsion_gradient(self):
        prototypes = torch.tensor([[1., 0.], [1., 1.]], requires_grad=True)
        base_gradient = torch.autograd.grad(rcpl_loss(prototypes), prototypes)[0]
        weighted_gradient = torch.autograd.grad(
            rcpl_loss(prototypes, torch.tensor([[0., .6], [.6, 0.]])), prototypes,
        )[0]
        torch.testing.assert_close(weighted_gradient, 1.6 * base_gradient)
        self.assertGreater(base_gradient.abs().sum().item(), 0)

    def test_hinge_keeps_positive_pairs_when_signed_sum_is_negative(self):
        prototypes = torch.tensor([[1., 0.], [.6, .8], [-.6, -.8]], requires_grad=True)
        weights = torch.zeros(3, 3, requires_grad=True)
        with torch.no_grad():
            weights[0, 1] = .5
            weights.diagonal().fill_(10.)  # Self-pairs must still be excluded.
        self.assertLess(rcpl_loss(prototypes, weights).item(), 0.)
        loss = rcpl_loss(prototypes, weights, mode='hinge', margin=0.)
        # Only the 0<->1 pair is positive: (.6 * 1.5 + .6) / 3.
        self.assertAlmostEqual(loss.item(), .5, places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(prototypes.grad).all())
        self.assertGreater(prototypes.grad.abs().sum().item(), 0.)
        self.assertIsNone(weights.grad)

    def test_hinge_negative_pairs_have_zero_loss_and_zero_gradient(self):
        prototypes = torch.tensor([[1., 0.], [-.6, .8]], requires_grad=True)
        loss = rcpl_loss(prototypes, mode='hinge')
        self.assertEqual(loss.item(), 0.)
        gradient = torch.autograd.grad(loss, prototypes)[0]
        self.assertTrue((gradient == 0).all())
        paper_gradient = torch.autograd.grad(rcpl_loss(prototypes), prototypes)[0]
        self.assertGreater(paper_gradient.abs().sum().item(), 0.)

    def test_hinge_margin_is_applied_per_pair_before_weights_without_squaring(self):
        prototypes = torch.tensor([[1., 0.], [.6, .8]], requires_grad=True)
        weights = torch.tensor([[0., .5], [0., 0.]])
        # (.6 - .5) * (1.5 + 1.) / 2 = .125.
        self.assertAlmostEqual(rcpl_loss(prototypes, weights, mode='hinge', margin=.5).item(),
                               .125, places=6)
        self.assertEqual(rcpl_loss(prototypes, weights, mode='hinge', margin=.7).item(), 0.)
        self.assertAlmostEqual(rcpl_loss(prototypes, weights, mode='paper', margin=.7).item(),
                               .75, places=6)

    def test_both_trainers_select_hinge_for_base_and_incremental_losses(self):
        root = Path(__file__).resolve().parents[1]
        prototypes = torch.tensor([[1., 0.], [.6, .8], [-.6, -.8]])
        args = SimpleNamespace(proto_sep_mode='hinge', proto_margin=0.)
        for filename in ('Trainer.py', 'Trainer_coco.py'):
            tree = ast.parse((root / 'continual' / filename).read_text())
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name) and node.func.id == 'rcpl_loss']
            self.assertEqual(len(calls), 2)
            for call in calls:
                namespace = {'rcpl_loss': rcpl_loss, 'new_prototypes': prototypes,
                             'self': SimpleNamespace(confusion_weights=torch.zeros(3, 3)), 'args': args}
                loss = eval(compile(ast.Expression(call), filename, 'eval'), namespace)
                self.assertAlmostEqual(loss.item(), .4, places=6)

    def test_hinge_rejects_invalid_modes_and_margins(self):
        prototypes = torch.eye(2)
        with self.assertRaises(ValueError):
            rcpl_loss(prototypes, mode='unknown')
        for margin in (-.1, 1.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                rcpl_loss(prototypes, mode='hinge', margin=margin)

    def test_cpa_uses_raw_squared_l2_relaxes_confused_old_rows_only(self):
        student = torch.tensor([[2., 0.], [0., 4.], [20., 30.]], requires_grad=True)
        teacher = torch.tensor([[1., 0.], [0., 2.]], requires_grad=True)
        weights = torch.tensor([[0., 0., .8], [0., 0., .25], [0., .5, 0.]],
                               requires_grad=True)
        self.assertAlmostEqual(cpa_loss(student, teacher, mode='paper').item(), 2.5)
        loss = cpa_loss(student, teacher, weights, mode='paper')
        self.assertAlmostEqual(loss.item(), 1.6, places=6)
        loss.backward()
        torch.testing.assert_close(student.grad, torch.tensor([[.2, 0.], [0., 1.5], [0., 0.]]))
        self.assertIsNone(teacher.grad)
        self.assertIsNone(weights.grad)

    def test_image_ald_empty_singleton_mean_and_ties(self):
        candidates = torch.tensor([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 1, 1]])
        peaks = torch.tensor([[10., 20., 30.], [100., -4., 200.],
                              [2., 6., 100.], [4., 4., 4.]], requires_grad=True)
        retained, thresholds = filter_old_class_labels(candidates, peaks)
        torch.testing.assert_close(retained, torch.tensor([[0, 0, 0], [0, 1, 0],
                                                         [0, 1, 0], [1, 1, 1]]))
        torch.testing.assert_close(thresholds, torch.tensor([[0.], [-4.], [4.], [4.]]))
        self.assertFalse(thresholds.requires_grad)

    def test_invalid_splits_and_temperatures_are_rejected(self):
        for old_count in (0, 4):
            with self.assertRaises(ValueError):
                confusion_counterparts(torch.zeros(4, 4), old_count)
        for temperature in (0., -1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                confusion_counterparts(torch.zeros(4, 4), 2, temperature)


if __name__ == '__main__':
    unittest.main()
