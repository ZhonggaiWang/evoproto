"""CPU behavioral contract for old-anchor SEP suppression by new CAM.

Deploy the previously frozen SEP as sep_legacy_reference.py alongside this
script. Put the guarded candidate model package on PYTHONPATH. The guard
may change numerator strength only, never eligible class-count balancing.
"""
import unittest

import torch
import torch.nn.functional as F

from model.confusion_pair_losses import directed_pair_sep_loss as guarded_sep
from sep_legacy_reference import directed_pair_sep_loss as legacy_sep


def example(anchor=1, width=1, classes=5, teacher_classes=3):
    s = torch.linspace(-.75, 1.25, classes * width).reshape(1, classes, 1, width)
    p = s.flip(1) * .75
    t = torch.zeros(1, teacher_classes, 1, width)
    t[:, anchor if anchor < teacher_classes else 1] = 8.
    a = torch.full((1, 1, width), anchor, dtype=torch.long)
    e = {'anchors': a, 'weights': torch.full_like(a, .875, dtype=torch.float),
         'accepted': torch.ones_like(a, dtype=torch.bool), 'valid': torch.ones_like(a, dtype=torch.bool)}
    targets = torch.full((classes,), -1, dtype=torch.long)
    targets[anchor] = 2 if anchor == 1 else 1
    cams = torch.zeros(1, classes - 1, 1, width)
    return s, p, t, e, targets, cams


def measured(function, data, **kwargs):
    s, p, t, e, targets, _ = data
    a, b = s.clone().requires_grad_(), p.clone().requires_grad_()
    value, stats = function(a, b, t, e, targets, **kwargs)
    value.backward()
    return value.detach(), a.grad.detach(), b.grad.detach(), stats


class SEPNewCAMGuardTests(unittest.TestCase):
    def test_disabled_default_is_bitwise_legacy_value_both_gradients_and_stats(self):
        data = example(width=4)
        data[3]['accepted'][0, 0, -1] = False
        reference = measured(legacy_sep, data)
        for kwargs in [{}, {'new_cam_guard': False}, {'new_cam_guard': False, 'valid_cams': data[-1]}]:
            with self.subTest(kwargs=tuple(kwargs)):
                actual = measured(guarded_sep, data, **kwargs)
                for got, expected in zip(actual[:3], reference[:3]):
                    torch.testing.assert_close(got, expected, atol=0, rtol=0)
                self.assertEqual(actual[3], reference[3])

    def test_disabled_ignores_absent_or_malformed_cam_for_legacy_compatibility(self):
        data = example()
        reference = measured(legacy_sep, data)
        for cams in [None, torch.zeros(1), torch.full((1,), float('nan'))]:
            with self.subTest(cams=None if cams is None else tuple(cams.shape)):
                actual = measured(guarded_sep, data, new_cam_guard=False, valid_cams=cams)
                for got, expected in zip(actual[:3], reference[:3]):
                    torch.testing.assert_close(got, expected, atol=0, rtol=0)
                self.assertEqual(actual[3], reference[3])

    def test_zero_new_cam_is_bitwise_original_value_and_both_gradients(self):
        data = example(width=3)
        # Strong old CAMs must not be mistaken for new evidence.
        data[-1][:, :2] = 1.
        reference = measured(legacy_sep, data)
        actual = measured(guarded_sep, data, new_cam_guard=True, valid_cams=data[-1])
        for got, expected in zip(actual[:3], reference[:3]):
            torch.testing.assert_close(got, expected, atol=0, rtol=0)
        self.assertEqual(actual[3]['guard_old_removed_weight_sum'], 0.)
        self.assertEqual(actual[3]['sep_class_eligible_pixels'], reference[3]['sep_class_eligible_pixels'])

    def test_full_new_cam_zeros_old_gradients_without_removing_class_counts(self):
        data = example(width=3)
        data[-1][:, 2] = 1.
        actual = measured(guarded_sep, data, new_cam_guard=True, valid_cams=data[-1])
        self.assertEqual(float(actual[0]), 0.)
        self.assertEqual(float(actual[1].abs().sum()), 0.)
        self.assertEqual(float(actual[2].abs().sum()), 0.)
        stats = actual[3]
        self.assertEqual(stats['sep_nonzero_pixels'], 3)
        self.assertEqual(stats['sep_old_pixels'], 3)
        self.assertEqual(stats['sep_class_eligible_pixels'], [0., 3., 0., 0., 0.])
        self.assertEqual(stats['sep_weight_sum'], 0.)
        self.assertEqual(stats['guard_zero_weight_old_pixels'], 3)
        self.assertEqual(stats['guard_effective_old_pixels'], 0)

    def test_new_anchor_full_new_cam_retains_bitwise_legacy_value_and_gradients(self):
        data = example(anchor=3, width=3)
        data[-1][:, 2:] = 1.
        reference = measured(legacy_sep, data)
        actual = measured(guarded_sep, data, new_cam_guard=True, valid_cams=data[-1])
        for got, expected in zip(actual[:3], reference[:3]):
            torch.testing.assert_close(got, expected, atol=0, rtol=0)
        self.assertEqual(actual[3]['guard_new_weight_sum_before'], actual[3]['guard_new_weight_sum_after'])
        self.assertEqual(actual[3]['guard_old_removed_weight_sum'], 0.)

    def test_half_new_cam_quarters_old_value_and_each_head_gradient(self):
        data = example(width=4)
        data[-1][:, 2] = .5
        reference = measured(legacy_sep, data)
        actual = measured(guarded_sep, data, new_cam_guard=True, valid_cams=data[-1])
        for got, expected in zip(actual[:3], reference[:3]):
            torch.testing.assert_close(got, .25 * expected, atol=2e-7, rtol=2e-7)
        self.assertEqual(actual[3]['guard_old_mean_factor'], .25)
        self.assertEqual(actual[3]['sep_class_eligible_pixels'], reference[3]['sep_class_eligible_pixels'])

    def test_unequal_two_class_counts_do_not_cancel_attenuation_in_denominator(self):
        s = torch.zeros(1, 5, 1, 101)
        p = torch.zeros_like(s)
        a = torch.ones(1, 1, 101, dtype=torch.long); a[:, :, -1] = 2
        t = torch.zeros(1, 3, 1, 101); t[:, 1, :, :-1] = 8.; t[:, 2, :, -1] = 8.
        e = {'anchors': a, 'weights': torch.ones_like(a, dtype=torch.float),
             'accepted': torch.ones_like(a, dtype=torch.bool), 'valid': torch.ones_like(a, dtype=torch.bool)}
        targets = torch.tensor([-1, 2, 1, -1, -1])
        cams = torch.zeros(1, 4, 1, 101); cams[:, 2, :, :-1] = .5
        actual = measured(guarded_sep, (s, p, t, e, targets, cams), new_cam_guard=True, valid_cams=cams)
        # Original sqrt masses 10 and 1 remain the denominator. The large
        # class's numerator mass becomes 2.5, and the small class remains 1.
        expected = F.softplus(torch.tensor(1.)) * (2.5 + 1.) / (10. + 1.)
        torch.testing.assert_close(actual[0], expected, atol=2e-7, rtol=2e-7)
        self.assertEqual(actual[3]['sep_class_eligible_pixels'], [0., 100., 1., 0., 0.])
        large = -actual[1][:, 1, :, :-1].sum()
        small = -actual[1][:, 2, :, -1].sum()
        torch.testing.assert_close(large / small, torch.tensor(2.5), atol=2e-6, rtol=2e-6)

    def test_whole_class_zero_guard_retains_its_original_denominator_mass(self):
        s = torch.zeros(1, 5, 1, 101); p = torch.zeros_like(s)
        a = torch.ones(1, 1, 101, dtype=torch.long); a[:, :, -1] = 2
        t = torch.zeros(1, 3, 1, 101); t[:, 1, :, :-1] = 8.; t[:, 2, :, -1] = 8.
        e = {'anchors': a, 'weights': torch.ones_like(a, dtype=torch.float),
             'accepted': torch.ones_like(a, dtype=torch.bool), 'valid': torch.ones_like(a, dtype=torch.bool)}
        targets = torch.tensor([-1, 2, 1, -1, -1])
        cams = torch.zeros(1, 4, 1, 101); cams[:, 2, :, :-1] = 1.
        actual = measured(guarded_sep, (s, p, t, e, targets, cams), new_cam_guard=True, valid_cams=cams)
        expected = F.softplus(torch.tensor(1.)) / 11.
        torch.testing.assert_close(actual[0], expected, atol=2e-7, rtol=2e-7)
        self.assertEqual(actual[3]['sep_class_eligible_pixels'][1], 100.)
        self.assertEqual(float(actual[1][:, :, :, :-1].abs().sum()), 0.)
        self.assertGreater(float(actual[1][:, :, :, -1].abs().sum()), 0.)

    def test_teacher_contradiction_and_unaccepted_masks_are_preserved(self):
        data = example(width=3)
        data[2][:, 1, :, 0] = 0.; data[2][:, 2, :, 0] = 8.
        data[3]['accepted'][:, :, 1] = False
        data[-1][:, 2] = .5
        reference = measured(legacy_sep, data)
        actual = measured(guarded_sep, data, new_cam_guard=True, valid_cams=data[-1])
        self.assertEqual(actual[3]['sep_class_eligible_pixels'], reference[3]['sep_class_eligible_pixels'])
        self.assertEqual(actual[3]['sep_teacher_veto_pixels'], 1)
        self.assertEqual(float(actual[1][:, :, :, :2].abs().sum()), 0.)
        self.assertGreater(float(actual[1][:, :, :, -1].abs().sum()), 0.)

    def test_cam_teacher_and_evidence_are_detached(self):
        data = example(width=3)
        s = data[0].clone().requires_grad_(); p = data[1].clone().requires_grad_()
        t = data[2].requires_grad_(); e = data[3]; e['weights'].requires_grad_()
        cams = data[-1]; cams[:, 2] = .5; cams.requires_grad_()
        value, _ = guarded_sep(s, p, t, e, data[4], new_cam_guard=True, valid_cams=cams)
        value.backward()
        self.assertIsNone(t.grad); self.assertIsNone(e['weights'].grad); self.assertIsNone(cams.grad)
        self.assertGreater(float(s.grad.abs().sum()), 0.)
        self.assertGreater(float(p.grad.abs().sum()), 0.)

    def test_bilinear_cam_resize_to_native_precedes_squared_attenuation(self):
        data = example()
        cams = torch.zeros(1, 4, 2, 2)
        cams[:, 2] = torch.tensor([[0., 1.], [0., 1.]])
        reference = measured(legacy_sep, data)
        actual = measured(guarded_sep, data, new_cam_guard=True, valid_cams=cams)
        for got, expected in zip(actual[:3], reference[:3]):
            torch.testing.assert_close(got, .25 * expected, atol=2e-7, rtol=2e-7)

    def test_maximum_across_new_channels_controls_guard(self):
        data = example()
        data[-1][:, 2] = .25; data[-1][:, 3] = .75
        reference = measured(legacy_sep, data)
        actual = measured(guarded_sep, data, new_cam_guard=True, valid_cams=data[-1])
        for got, expected in zip(actual[:3], reference[:3]):
            torch.testing.assert_close(got, .0625 * expected, atol=2e-7, rtol=2e-7)

    def test_enabled_guard_rejects_missing_wrong_rank_batch_and_channels(self):
        data = example()
        for cams in [None, torch.zeros(1, 4, 1), torch.zeros(2, 4, 1, 1),
                     torch.zeros(1, 3, 1, 1), torch.zeros(1, 5, 1, 1)]:
            with self.subTest(shape=None if cams is None else tuple(cams.shape)), self.assertRaises(ValueError):
                measured(guarded_sep, data, new_cam_guard=True, valid_cams=cams)

    def test_enabled_guard_rejects_nonfinite_cam(self):
        data = example()
        for bad in [float('nan'), float('inf')]:
            with self.subTest(value=bad), self.assertRaises(ValueError):
                measured(guarded_sep, data, new_cam_guard=True, valid_cams=torch.full_like(data[-1], bad))


if __name__ == '__main__':
    unittest.main()
