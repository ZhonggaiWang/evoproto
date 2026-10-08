"""Behavior tests for a proposed semantic-protected prototype SEP surrogate.

These test actual raw parameter gradients through F.normalize, rather than
only gradients in the returned normalized prototype tensor. No GPU or data
is needed. Run with the proposed module on PYTHONPATH; do not modify formal SRC.
"""
import inspect
import json
import unittest

import torch
import torch.nn.functional as F

try:
    from model.semantic_protected_sep import semantic_protected_sep
except ModuleNotFoundError:
    from semantic_protected_sep import semantic_protected_sep


def initial_raw():
    # BG0, old1/2, new3/4/5. New rows have different raw norms.
    return torch.tensor([[0., 0., 2.], [1., 1., 0.], [0., 3., 0.],
                         [2., 0., 0.], [0., 7., 0.], [0., 0., .5]], dtype=torch.float64)


def coefficients():
    sep = torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.],
                        [.9, -2., 3.], [2., .8, 1.], [1., -1., .4]], dtype=torch.float64)
    sem = torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.],
                        [.4, 1., 0.], [1., .6, 0.], [0., 0., 0.]], dtype=torch.float64)
    return sep, sem


def tangent(value, p):
    return value-(value*p).sum(1, keepdim=True)*p


def reference_projection(raw, sep, sem, old_classes=2):
    """Independent analytic first-order result, including normalize Jacobian."""
    p = F.normalize(raw.detach(), dim=1)
    gs, gm = tangent(sep, p), tangent(sem, p)
    dots = (gs*gm).sum(1, keepdim=True)
    norm2 = gm.square().sum(1, keepdim=True)
    correction = torch.where((dots < 0) & (norm2 > 0), dots/norm2.clamp_min(torch.finfo(raw.dtype).tiny), 0.)
    protected = gs-correction*gm
    protected[:old_classes+1] = 0.
    return protected/raw.detach().norm(dim=1, keepdim=True), gm/raw.detach().norm(dim=1, keepdim=True)


def measure(sep=None, sem=None, raw=None, old_classes=2):
    raw = (initial_raw() if raw is None else raw).clone().requires_grad_()
    p = F.normalize(raw, dim=1)
    default_sep, default_sem = coefficients()
    sep = default_sep if sep is None else sep
    sem = default_sem if sem is None else sem
    geometry = 10.+(p*sep).sum()
    semantic = 10.+(p*sem).sum()
    protected, stats = semantic_protected_sep(p, geometry, semantic, old_classes)
    gradient = torch.autograd.grad(protected, raw)[0]
    expected, semantic_gradient = reference_projection(raw, sep, sem, old_classes)
    return protected.detach(), geometry.detach(), gradient, expected, semantic_gradient, stats


class SemanticProtectedSeparationTests(unittest.TestCase):
    def assert_close(self, actual, expected):
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)

    def test_actual_raw_new_gradients_never_oppose_semantic_direction(self):
        result = measure()
        self.assert_close(result[2], result[3])
        dots = (result[2]*result[4]).sum(1)
        self.assertTrue(bool((dots[3:] >= -2e-13).all()))
        self.assertGreater(float(dots[4]), 0.)

    def test_only_negative_parallel_component_is_removed(self):
        gradient = measure()[2]
        # At new3, raw norm2; y opposes SEM and is removed, z survives.
        self.assert_close(gradient[3], torch.tensor([0., 0., 1.5], dtype=torch.float64))
        self.assertGreater(float(gradient[3].norm()), 0.)

    def test_positive_dot_keeps_original_raw_new_gradient(self):
        raw = initial_raw()
        sep, _ = coefficients()
        actual = measure()[2]
        original = tangent(sep, F.normalize(raw, dim=1))/raw.norm(dim=1, keepdim=True)
        self.assert_close(actual[4], original[4])

    def test_zero_semantic_row_keeps_original_sep_row(self):
        sep, _ = coefficients()
        raw = initial_raw()
        original = tangent(sep, F.normalize(raw, dim=1))/raw.norm(dim=1, keepdim=True)
        self.assert_close(measure()[2][5], original[5])

    def test_all_zero_semantic_keeps_all_new_geometry_gradients(self):
        sep, _ = coefficients()
        zero = torch.zeros_like(sep)
        result = measure(sem=zero)
        self.assert_close(result[2], result[3])
        raw = initial_raw()
        original = tangent(sep, F.normalize(raw, dim=1))/raw.norm(dim=1, keepdim=True)
        self.assert_close(result[2][3:], original[3:])

    def test_per_row_different_raw_norms_preserve_parameter_space_contract(self):
        raw = initial_raw()*torch.tensor([1., 2., .5, 11., .2, 9.], dtype=torch.float64)[:, None]
        result = measure(raw=raw)
        self.assert_close(result[2], result[3])
        self.assertTrue(bool(((result[2]*result[4]).sum(1)[3:] >= -2e-13).all()))

    def test_extreme_nonzero_raw_row_norms_keep_rowwise_direction_sign(self):
        raw = initial_raw()
        raw[3:] = F.normalize(raw[3:], dim=1)*torch.tensor([.001, 1., 200.], dtype=torch.float64)[:, None]
        result = measure(raw=raw)
        self.assert_close(result[2], result[3])
        dots = (result[2]*result[4]).sum(1)
        self.assertTrue(bool((dots[3:] >= -2e-10).all()))

    def test_non_axis_aligned_rows_apply_full_normalization_jacobian(self):
        raw = initial_raw()
        raw[3] = torch.tensor([2., 3., -1.], dtype=torch.float64)
        raw[4] = torch.tensor([-.5, 1., 4.], dtype=torch.float64)
        raw[5] = torch.tensor([1., .2, 2.], dtype=torch.float64)
        result = measure(raw=raw)
        self.assert_close(result[2], result[3])
        self.assertTrue(bool(((result[2]*result[4]).sum(1)[3:] >= -2e-13).all()))

    def test_background_and_old_rows_receive_no_surrogate_gradient(self):
        sep, sem = coefficients()
        sep[:3] = torch.tensor([[1., 2., 3.], [4., 3., 2.], [1., -2., 1.]], dtype=torch.float64)
        sem[:3] = -sep[:3]
        gradient = measure(sep=sep, sem=sem)[2]
        self.assertEqual(int(gradient[:3].count_nonzero()), 0)
        self.assertGreater(int(gradient[3:].count_nonzero()), 0)

    def test_forward_value_is_exact_original_weighted_geometry(self):
        result = measure()
        self.assertTrue(torch.equal(result[0], result[1]))

    def test_component_derivative_queries_do_not_accumulate_leaf_grads(self):
        raw = initial_raw().requires_grad_()
        feature = torch.tensor([.3, .8, -.2], dtype=torch.float64, requires_grad=True)
        p = F.normalize(raw, dim=1)
        sep, _ = coefficients()
        geometry = 10.+(p*sep).sum()
        semantic = (p[3:]*feature).sum()
        semantic_protected_sep(p, geometry, semantic, 2)
        self.assertIsNone(raw.grad)
        self.assertIsNone(feature.grad)

    def test_existing_accumulated_leaf_grads_remain_unchanged(self):
        raw = initial_raw().requires_grad_()
        raw.grad = torch.full_like(raw, 17.)
        before = raw.grad.clone()
        p = F.normalize(raw, dim=1)
        sep, sem = coefficients()
        semantic_protected_sep(p, 10.+(p*sep).sum(), 10.+(p*sem).sum(), 2)
        self.assertTrue(torch.equal(raw.grad, before))

    def test_surrogate_does_not_send_gradient_to_semantic_feature_parameter(self):
        raw = initial_raw().requires_grad_()
        feature = torch.tensor([.3, .8, -.2], dtype=torch.float64, requires_grad=True)
        p = F.normalize(raw, dim=1)
        sep, _ = coefficients()
        semantic = (p[3:]*feature).sum()
        surrogate, _ = semantic_protected_sep(p, 10.+(p*sep).sum(), semantic, 2)
        raw_grad, feature_grad = torch.autograd.grad(surrogate, (raw, feature), allow_unused=True)
        self.assertIsNone(feature_grad)
        self.assertGreater(int(raw_grad[3:].count_nonzero()), 0)

    def test_combined_loss_keeps_feature_semantic_gradient_exact(self):
        raw = initial_raw().requires_grad_()
        feature = torch.tensor([.3, .8, -.2], dtype=torch.float64, requires_grad=True)
        p = F.normalize(raw, dim=1)
        sep, _ = coefficients()
        semantic = (p[3:]*feature).square().sum()
        expected = torch.autograd.grad(semantic, feature, retain_graph=True)[0]
        surrogate, _ = semantic_protected_sep(p, 10.+(p*sep).sum(), semantic, 2)
        actual = torch.autograd.grad(semantic+surrogate, feature)[0]
        self.assert_close(actual, expected)

    def test_legitimate_zero_semantic_disconnected_from_prototypes_keeps_sep(self):
        raw = initial_raw().requires_grad_()
        feature = torch.tensor([.3, .8, -.2], dtype=torch.float64, requires_grad=True)
        p = F.normalize(raw, dim=1)
        sep, _ = coefficients()
        surrogate, _ = semantic_protected_sep(p, 10.+(p*sep).sum(), feature.square().sum()*0., 2)
        actual = torch.autograd.grad(surrogate, raw)[0]
        expected, _ = reference_projection(raw, sep, torch.zeros_like(sep))
        self.assert_close(actual, expected)
        self.assertIsNone(feature.grad)

    def test_nonzero_semantic_disconnected_from_prototypes_is_rejected(self):
        raw = initial_raw().requires_grad_()
        feature = torch.tensor([.3, .8, -.2], dtype=torch.float64, requires_grad=True)
        p = F.normalize(raw, dim=1)
        sep, _ = coefficients()
        with self.assertRaises((RuntimeError, ValueError)):
            semantic_protected_sep(p, 10.+(p*sep).sum(), feature.square().sum(), 2)

    def test_zero_geometry_without_semantic_is_graph_connected_zero(self):
        raw = initial_raw().requires_grad_()
        p = F.normalize(raw, dim=1)
        geometry = (p*0.).sum()
        surrogate, _ = semantic_protected_sep(p, geometry, None, 2)
        gradient = torch.autograd.grad(surrogate, raw)[0]
        self.assertTrue(torch.equal(surrogate.detach(), geometry.detach()))
        self.assertEqual(int(gradient.count_nonzero()), 0)

    def test_positive_semantic_rescaling_does_not_change_projection(self):
        sep, sem = coefficients()
        a = measure(sep=sep, sem=sem)[2]
        b = measure(sep=sep, sem=sem*19.)[2]
        self.assert_close(a, b)

    def test_tiny_conflicting_semantic_norm_zeroes_row_without_epsilon_residual(self):
        sep, sem = coefficients()
        result = measure(sep=sep, sem=sem*1e-160)
        self.assertEqual(int(result[2][3].count_nonzero()), 0)
        self.assertGreater(int(result[2][4].count_nonzero()), 0)
        self.assertEqual(result[5]['small_norm_conflict_zeroed_rows'], 1)
        self.assertTrue(result[5]['no_additive_denominator_epsilon'])

    def test_statistics_identify_projected_agreeing_and_zero_semantic_rows(self):
        stats = measure()[5]
        self.assertEqual(stats['conflicting_new_rows'], 1)
        self.assertEqual(stats['active_geometry_new_rows'], 3)
        self.assertTrue(stats['background_old_sep_gradient_zero'])
        rows = {row['class_id']: row for row in stats['per_class']}
        self.assertTrue(rows[3]['conflict_projected'])
        self.assertLess(rows[3]['dot_before'], 0.)
        self.assertGreaterEqual(rows[3]['dot_after'], -2e-13)
        self.assertFalse(rows[4]['conflict_projected'])
        self.assertGreater(rows[4]['dot_after'], 0.)
        self.assertEqual(rows[5]['semantic_tangent_norm'], 0.)

    def test_leaf_or_nonunit_prototype_inputs_are_rejected(self):
        raw = initial_raw().requires_grad_()
        sep, sem = coefficients()
        leaf = F.normalize(raw, dim=1).detach().requires_grad_()
        with self.assertRaises(ValueError):
            semantic_protected_sep(leaf, 10.+(leaf*sep).sum(), 10.+(leaf*sem).sum(), 2)
        nonunit = F.normalize(raw, dim=1)*2.
        with self.assertRaises(ValueError):
            semantic_protected_sep(nonunit, 10.+(nonunit*sep).sum(), 10.+(nonunit*sem).sum(), 2)

    def test_radial_only_geometry_has_zero_actual_raw_gradient(self):
        raw = initial_raw()
        p = F.normalize(raw, dim=1)
        result = measure(sep=p*torch.tensor([1., 2., 3., 4., 5., 6.], dtype=torch.float64)[:, None])
        self.assert_close(result[2], torch.zeros_like(raw))

    def test_stats_are_json_serializable_and_API_has_no_GT_argument(self):
        result = measure()
        json.dumps(result[5], allow_nan=False)
        self.assertEqual(list(inspect.signature(semantic_protected_sep).parameters),
            ['prototypes', 'weighted_geometry_loss', 'weighted_semantic_loss', 'old_classes'])


if __name__ == '__main__':
    unittest.main()
