"""Behavioral tests for directed SEP and confusion-guided old-class KD.

Run with the candidate source on PYTHONPATH. The legacy reference is a copy
of the previously evaluated relational KD, not a reimplementation of it.
All examples use synthetic evidence; there is no pixel-ground-truth input.
"""
import importlib
import inspect
import math
import unittest

import torch
import torch.nn.functional as F

try:
    from model.confusion_pair_losses import directed_pair_sep_loss
    from model.pixel_kd import pixel_kd_loss
except ModuleNotFoundError:
    from confusion_pair_losses import directed_pair_sep_loss
    from pixel_kd_pair import pixel_kd_loss


def evidence(anchors, weights=None, accepted=None, valid=None):
    if weights is None:
        weights = torch.ones_like(anchors, dtype=torch.float)
    if accepted is None:
        accepted = torch.ones_like(anchors, dtype=torch.bool)
    if valid is None:
        valid = torch.ones_like(anchors, dtype=torch.bool)
    return {'anchors': anchors, 'weights': weights,
            'accepted': accepted, 'valid': valid}


def one_pixel(anchor=1, partner=2, teacher_winner=1, classes=5, old_classes=3):
    s = torch.zeros(1, classes, 1, 1, requires_grad=True)
    p = torch.zeros_like(s, requires_grad=True)
    t = torch.zeros(1, old_classes, 1, 1)
    t[:, teacher_winner] = 8
    a = torch.tensor([[[anchor]]])
    targets = torch.full((classes,), -1, dtype=torch.long)
    if 0 <= anchor < classes:
        targets[anchor] = partner
    return s, p, t, evidence(a), targets


def kd_inputs(student=None, teacher=None, labels=None):
    if student is None:
        student = torch.tensor([0.1, -0.4, 1.3, 0.7, -0.2]).view(1, 5, 1, 1)
    if teacher is None:
        teacher = torch.tensor([0., 8., 0.]).view(1, 3, 1, 1)
    n, c, h, w = student.shape
    if labels is None:
        labels = torch.ones((n, h, w), dtype=torch.long)
    cams = torch.zeros((n, c - 1, h, w))
    winner = teacher.argmax(1)
    for i in range(1, teacher.shape[1]):
        cams[:, i - 1] = torch.where(winner == i, .875, .125)
    return student, teacher, labels, cams, [[0, h, 0, w] for _ in range(n)]


class DirectedPairSEPTests(unittest.TestCase):
    def test_selected_direction_pushes_both_heads_and_no_other_channel(self):
        s, p, t, e, targets = one_pixel()
        with torch.no_grad():
            s[:, 2] = 5.; s[:, 1] = -2.
            p[:, 2] = 3.; p[:, 1] = -1.
        value, _ = directed_pair_sep_loss(s, p, t, e, targets)
        value.backward()
        self.assertGreater(float(value), 0.)
        for gradient in (s.grad, p.grad):
            self.assertLess(float(gradient[0, 1, 0, 0]), 0.)
            self.assertGreater(float(gradient[0, 2, 0, 0]), 0.)
            self.assertEqual(float(gradient[:, [0, 3, 4]].abs().sum()), 0.)

    def test_direction_is_not_implicitly_symmetrized(self):
        s, p, t, e, targets = one_pixel(anchor=2, partner=1, teacher_winner=2)
        value, _ = directed_pair_sep_loss(s, p, t, e, targets)
        value.backward()
        self.assertGreater(float(s.grad[0, 1, 0, 0]), 0.)
        self.assertLess(float(s.grad[0, 2, 0, 0]), 0.)

    def test_confidently_wrong_student_remains_eligible(self):
        s, p, t, e, targets = one_pixel()
        with torch.no_grad():
            s[:, 1] = -20.; s[:, 2] = 20.
        value, _ = directed_pair_sep_loss(s, p, t, e, targets)
        value.backward()
        self.assertGreater(float(value), 10.)
        self.assertLess(float(s.grad[0, 1, 0, 0]), -.1)

    def test_new_anchor_survives_teacher_old_prediction(self):
        s, p, t, e, targets = one_pixel(anchor=3, partner=1, teacher_winner=1)
        value, _ = directed_pair_sep_loss(s, p, t, e, targets)
        value.backward()
        self.assertGreater(float(value), 0.)
        self.assertLess(float(s.grad[0, 3, 0, 0]), 0.)
        self.assertGreater(float(s.grad[0, 1, 0, 0]), 0.)

    def test_old_anchor_contradicting_teacher_is_vetoed(self):
        s, p, t, e, targets = one_pixel(anchor=1, partner=2, teacher_winner=2)
        value, _ = directed_pair_sep_loss(s, p, t, e, targets)
        value.backward()
        self.assertEqual(float(value), 0.)
        self.assertEqual(float(s.grad.abs().sum()), 0.)
        self.assertEqual(float(p.grad.abs().sum()), 0.)

    def test_background_self_missing_and_ignore_have_zero_loss_and_gradient(self):
        for anchor, partner in [(0, 1), (1, 0), (1, 1), (1, -1), (255, 1)]:
            with self.subTest(anchor=anchor, partner=partner):
                s, p, t, e, targets = one_pixel(anchor=anchor, partner=partner)
                value, _ = directed_pair_sep_loss(s, p, t, e, targets)
                value.backward()
                self.assertEqual(float(value), 0.)
                self.assertEqual(float(s.grad.abs().sum()), 0.)
                self.assertEqual(float(p.grad.abs().sum()), 0.)

    def test_unaccepted_invalid_or_zero_weight_is_differentiable_zero(self):
        for key in ['accepted', 'valid', 'weights']:
            with self.subTest(key=key):
                s, p, t, e, targets = one_pixel()
                e[key].zero_()
                value, _ = directed_pair_sep_loss(s, p, t, e, targets)
                value.backward()
                self.assertEqual(float(value), 0.)
                self.assertEqual(float(s.grad.abs().sum()), 0.)

    def test_teacher_and_evidence_are_detached(self):
        s, p, t, e, targets = one_pixel()
        t.requires_grad_(); e['weights'].requires_grad_()
        value, _ = directed_pair_sep_loss(s, p, t, e, targets)
        value.backward()
        self.assertIsNone(t.grad)
        self.assertIsNone(e['weights'].grad)
        self.assertGreater(float(s.grad.abs().sum()), 0.)
        self.assertGreater(float(p.grad.abs().sum()), 0.)

    def test_fixed_margin_and_two_head_average_have_known_loss(self):
        s, p, t, e, targets = one_pixel()
        with torch.no_grad():
            s[:, 1] = 2.; s[:, 2] = 1.
            p[:, 1] = -1.; p[:, 2] = 2.
        e['weights'].fill_(.625)
        value, _ = directed_pair_sep_loss(s, p, t, e, targets)
        expected = .625 * .5 * (F.softplus(torch.tensor(0.)) + F.softplus(torch.tensor(4.)))
        torch.testing.assert_close(value, expected, atol=1e-6, rtol=1e-6)

    def test_square_root_class_mass_limits_tiny_island_influence(self):
        # One class owns 100 pixels and another owns 1. Equal per-pixel
        # losses/derivatives must give aggregate class gradient ratio 10,
        # rather than the 100 of a pixel mean or 1 of equal class means.
        s = torch.zeros(1, 6, 1, 101, requires_grad=True)
        p = torch.zeros_like(s, requires_grad=True)
        t = torch.zeros(1, 3, 1, 101); t[:, 0] = 8.
        anchors = torch.full((1, 1, 101), 3, dtype=torch.long)
        anchors[:, :, -1] = 4
        targets = torch.tensor([-1, -1, -1, 1, 2, -1])
        value, _ = directed_pair_sep_loss(s, p, t, evidence(anchors), targets)
        value.backward()
        large = -s.grad[:, 3, :, :-1].sum()
        small = -s.grad[:, 4, :, -1].sum()
        torch.testing.assert_close(large / small, torch.tensor(10.), atol=2e-6, rtol=2e-6)
        self.assertGreater(float(value), 0.)


class ConfusionPairKDTests(unittest.TestCase):
    def test_none_and_zero_blend_are_exactly_legacy_values_and_gradients(self):
        legacy = importlib.import_module('pixel_kd_legacy_reference').pixel_kd_loss
        args = kd_inputs()
        targets = torch.tensor([-1, 2, 1, -1, -1])
        e = evidence(args[2])
        ref_s = args[0].clone().requires_grad_()
        reference, _ = legacy(ref_s, *args[1:], temperature=2.)
        reference.backward()
        for kwargs in [{}, {'pair_targets': None, 'pair_evidence': None, 'pair_blend': .5},
                       {'pair_targets': targets, 'pair_evidence': e, 'pair_blend': 0.}]:
            with self.subTest(kwargs=tuple(kwargs)):
                s = args[0].clone().requires_grad_()
                actual, _ = pixel_kd_loss(s, *args[1:], temperature=2., **kwargs)
                actual.backward()
                torch.testing.assert_close(actual, reference, atol=0, rtol=0)
                torch.testing.assert_close(s.grad, ref_s.grad, atol=0, rtol=0)

    def test_confident_teacher_pair_drives_correct_old_direction(self):
        s, t, labels, cams, boxes = kd_inputs(student=torch.zeros(1, 5, 1, 1))
        s.requires_grad_()
        value, stats = pixel_kd_loss(s, t, labels, cams, boxes,
                                    pair_targets=torch.tensor([-1, 2, 1, -1, -1]),
                                    pair_evidence=evidence(labels), pair_blend=.5)
        value.backward()
        self.assertGreater(float(value), 0.)
        self.assertLess(float(s.grad[0, 1, 0, 0]), 0.)
        self.assertGreater(float(s.grad[0, 2, 0, 0]), 0.)
        self.assertEqual(stats['kd_pair_pixels'], 1)

    def test_pair_replaces_half_of_same_pixel_loss_and_keeps_legacy_denominator(self):
        # Three old classes are essential: with only two, binary-pair KL
        # equals conditional old-class KL and cannot expose a broken blend.
        s = torch.tensor([.1, -.4, 1.3, 3.2, .7, -.2]).view(1, 6, 1, 1)
        t = torch.tensor([0., 8., 0., 2.]).view(1, 4, 1, 1)
        labels = torch.ones((1, 1, 1), dtype=torch.long)
        cams = torch.tensor([.875, .125, .125, 0., 0.]).view(1, 5, 1, 1)
        boxes = [[0, 1, 0, 1]]
        a = s.clone().requires_grad_(); b = s.clone().requires_grad_()
        legacy, _ = pixel_kd_loss(a, t, labels, cams, boxes, temperature=2.)
        actual, stats = pixel_kd_loss(b, t, labels, cams, boxes, temperature=2.,
                                    pair_targets=torch.tensor([-1, 2, -1, -1, -1, -1]),
                                    pair_evidence=evidence(labels), pair_blend=.5)
        q = torch.sigmoid((t[0, 1, 0, 0] - t[0, 2, 0, 0]) / 2.)
        r = torch.sigmoid((s[0, 1, 0, 0] - s[0, 2, 0, 0]) / 2.)
        pair_kl = 4. * (q * (q.log() - r.log()) + (1-q) * ((1-q).log() - (1-r).log()))
        reliability = (2 * torch.sigmoid(torch.tensor(8.)) - 1).square()
        expected = .5 * legacy.detach() + .5 * .875 * reliability * pair_kl
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        self.assertGreater(abs(float(actual - legacy)), 1e-3)
        legacy.backward(); actual.backward()
        # The third old channel participates only in the retained half KL.
        torch.testing.assert_close(b.grad[:, 3], .5 * a.grad[:, 3], atol=1e-6, rtol=1e-6)
        self.assertEqual(stats['kd_pair_pixels'], 1)

    def test_hybrid_kd_preserves_square_root_class_count_balance(self):
        s = torch.zeros(1, 5, 1, 101, requires_grad=True)
        t = torch.zeros(1, 4, 1, 101); t[:, 1, :, :-1] = 8.; t[:, 2, :, -1] = 8.
        labels = torch.ones((1, 1, 101), dtype=torch.long); labels[:, :, -1] = 2
        cams = torch.zeros(1, 4, 1, 101)
        cams[:, 0, :, :-1] = .875; cams[:, 1, :, -1] = .875
        value, _ = pixel_kd_loss(s, t, labels, cams, [[0, 1, 0, 101]],
                                pair_targets=torch.tensor([-1, 2, 1, -1, -1]),
                                pair_evidence=evidence(labels), pair_blend=.5)
        value.backward()
        large = -s.grad[:, 1, :, :-1].sum()
        small = -s.grad[:, 2, :, -1].sum()
        torch.testing.assert_close(large / small, torch.tensor(10.), atol=2e-6, rtol=2e-6)

    def test_weak_or_contradictory_teacher_pair_uses_exact_fallback(self):
        targets = torch.tensor([-1, 2, 1, -1, -1])
        for t, anchor in [(torch.tensor([0., 1., .8]).view(1, 3, 1, 1), 1),
                          (torch.tensor([0., 0., 8.]).view(1, 3, 1, 1), 1)]:
            with self.subTest(anchor=anchor, teacher=t.flatten().tolist()):
                args = kd_inputs(teacher=t)
                e = evidence(torch.full_like(args[2], anchor))
                a = args[0].clone().requires_grad_(); b = a.detach().clone().requires_grad_()
                expected, _ = pixel_kd_loss(a, *args[1:])
                actual, stats = pixel_kd_loss(b, *args[1:], pair_targets=targets,
                                            pair_evidence=e, pair_blend=.5)
                expected.backward(); actual.backward()
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                torch.testing.assert_close(b.grad, a.grad, atol=0, rtol=0)
                self.assertEqual(stats['kd_pair_pixels'], 0)

    def test_missing_self_background_and_new_partners_use_exact_fallback(self):
        args = kd_inputs()
        for partner in [-1, 0, 1, 3, 4]:
            with self.subTest(partner=partner):
                targets = torch.tensor([-1, partner, -1, -1, -1])
                a = args[0].clone().requires_grad_(); b = a.detach().clone().requires_grad_()
                expected, _ = pixel_kd_loss(a, *args[1:])
                actual, stats = pixel_kd_loss(b, *args[1:], pair_targets=targets,
                                            pair_evidence=evidence(args[2]), pair_blend=.5)
                expected.backward(); actual.backward()
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                torch.testing.assert_close(b.grad, a.grad, atol=0, rtol=0)
                self.assertEqual(stats['kd_pair_pixels'], 0)

    def test_no_direct_background_or_new_output_gradient_and_old_offset_invariance(self):
        args = kd_inputs()
        kw = {'pair_targets': torch.tensor([-1, 2, 1, -1, -1]),
              'pair_evidence': evidence(args[2]), 'pair_blend': .5}
        a = args[0].clone().requires_grad_()
        shifted = args[0].clone(); shifted[:, 1:3] += 7.
        b = shifted.requires_grad_()
        va, _ = pixel_kd_loss(a, *args[1:], **kw)
        vb, _ = pixel_kd_loss(b, *args[1:], **kw)
        va.backward(); vb.backward()
        torch.testing.assert_close(va, vb, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-6)
        self.assertEqual(float(a.grad[:, [0, 3, 4]].abs().sum()), 0.)

    def test_new_anchor_never_receives_old_pair_kd(self):
        s, t, labels, cams, boxes = kd_inputs(labels=torch.full((1, 1, 1), 3))
        s.requires_grad_()
        value, stats = pixel_kd_loss(s, t, labels, cams, boxes,
                                    pair_targets=torch.tensor([-1, 2, 1, 1, -1]),
                                    pair_evidence=evidence(labels), pair_blend=.5)
        value.backward()
        self.assertEqual(float(value), 0.)
        self.assertEqual(float(s.grad.abs().sum()), 0.)
        self.assertEqual(stats['kd_pair_pixels'], 0)

    def test_teacher_and_pair_evidence_are_detached(self):
        s, t, labels, cams, boxes = kd_inputs()
        s.requires_grad_(); t.requires_grad_(); cams.requires_grad_()
        e = evidence(labels); e['weights'].requires_grad_()
        value, _ = pixel_kd_loss(s, t, labels, cams, boxes,
                                pair_targets=torch.tensor([-1, 2, 1, -1, -1]),
                                pair_evidence=e, pair_blend=.5)
        value.backward()
        self.assertIsNone(t.grad); self.assertIsNone(cams.grad)
        self.assertIsNone(e['weights'].grad)
        self.assertGreater(float(s.grad.abs().sum()), 0.)

    def test_rejected_pair_evidence_uses_exact_fallback(self):
        args = kd_inputs()
        for key in ['accepted', 'valid', 'weights']:
            with self.subTest(key=key):
                e = evidence(args[2]); e[key].zero_()
                a = args[0].clone().requires_grad_(); b = a.detach().clone().requires_grad_()
                expected, _ = pixel_kd_loss(a, *args[1:])
                actual, stats = pixel_kd_loss(b, *args[1:],
                                            pair_targets=torch.tensor([-1, 2, 1, -1, -1]),
                                            pair_evidence=e, pair_blend=.5)
                expected.backward(); actual.backward()
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                torch.testing.assert_close(b.grad, a.grad, atol=0, rtol=0)
                self.assertEqual(stats['kd_pair_pixels'], 0)

    def test_blend_above_cap_or_negative_is_rejected(self):
        args = kd_inputs()
        for blend in [-.01, .5001, 1.]:
            with self.subTest(blend=blend), self.assertRaises(ValueError):
                pixel_kd_loss(*args, pair_targets=torch.tensor([-1, 2, 1, -1, -1]),
                              pair_evidence=evidence(args[2]), pair_blend=blend)


class TrainingInterfaceTests(unittest.TestCase):
    def test_training_losses_have_no_gt_parameter(self):
        for function in [directed_pair_sep_loss, pixel_kd_loss]:
            names = inspect.signature(function).parameters
            self.assertFalse(any(name in {'gt', 'ground_truth', 'gt_labels', 'pixel_gt'}
                                 for name in names))


if __name__ == '__main__':
    unittest.main()
