"""Behavioral tests for observation-only directed confusion; no dataset/GT required."""
import copy
import unittest

import torch
import torch.nn.functional as F

from online_directed_confusion import (
    OnlineDirectedConfusion, build_confusion_evidence, pair_counts,
)


def make_inputs(labels, predictions, cams, classes=3, box=None):
    labels = torch.tensor(labels, dtype=torch.long).unsqueeze(0)
    predictions = torch.tensor(predictions, dtype=torch.long).unsqueeze(0)
    logits = F.one_hot(predictions, classes).permute(0, 3, 1, 2).float() * 8
    cams = torch.tensor(cams, dtype=torch.float32).unsqueeze(0)
    h, w = labels.shape[-2:]
    return logits, labels, cams, [[0, h, 0, w]] if box is None else box


class DirectedConfusionTests(unittest.TestCase):
    def test_direction_and_wrong_predictions_remain_accepted(self):
        x = make_inputs([[1, 2, 0]], [[2, 1, 2]],
                        [[[.875, .25, .125]], [[.25, .875, .125]]])
        estimator = OnlineDirectedConfusion(3)
        evidence = estimator.update(*x, synchronize=False)
        self.assertTrue(evidence['accepted'].all())
        expected = torch.zeros(3, 3, dtype=torch.float64)
        expected[1, 2] = .546875
        expected[2, 1] = .546875
        expected[0, 2] = .875
        torch.testing.assert_close(estimator.counts, expected, rtol=0, atol=0)
        self.assertEqual(estimator.export()['probabilities'][1], [0., 0., 1.])
        self.assertEqual(estimator.export()['probabilities'][2], [0., 1., 0.])

    def test_foreground_tie_and_stronger_competitor_produce_no_evidence(self):
        x = make_inputs([[1, 1]], [[2, 1]],
                        [[[.875, .75]], [[.875, .875]], [[.125, .125]]], classes=4)
        evidence = build_confusion_evidence(*x)
        self.assertEqual(evidence['weights'].tolist(), [[[0., 0.]]])
        self.assertFalse(evidence['accepted'].any())
        self.assertTrue(evidence['broad'].all())

    def test_padding_and_ignore_do_not_enter_any_count(self):
        x = make_inputs([[1, 2, 255], [0, 1, -1]], [[2, 1, 2], [0, 2, 1]],
                        [[[1., 0., 1.], [0., 1., 1.]],
                         [[0., 1., 0.], [0., 0., 0.]]], box=[[0, 2, 0, 2]])
        estimator = OnlineDirectedConfusion(3)
        evidence = estimator.update(*x, synchronize=False)
        self.assertEqual(int(evidence['valid'].sum()), 4)
        self.assertEqual(int(evidence['broad'].sum()), 4)
        self.assertEqual(float(estimator.counts.sum()), 4.)
        self.assertEqual(float(estimator.accepted_counts.sum()), 4.)
        self.assertEqual(float(estimator.broad_counts.sum()), 4.)
        self.assertEqual(estimator.image_observations.tolist(), [1., 1., 1.])
        # Ignore/out-of-range anchors also vanish when their pixels are inside
        # the valid box, rather than merely being rejected as padding.
        full_box = build_confusion_evidence(x[0], x[1], x[2], [[0, 2, 0, 3]])
        self.assertEqual(int(full_box['valid'].sum()), 6)
        self.assertEqual(int(full_box['broad'].sum()), 4)
        self.assertEqual(int(full_box['accepted'].sum()), 4)

    def test_background_uses_low_threshold_and_distinct_confidence(self):
        x = make_inputs([[0, 0, 1]], [[1, 2, 0]],
                        [[[.125, .25, .125]], [[0., 0., 0.]]])
        evidence = build_confusion_evidence(*x)
        self.assertEqual(evidence['accepted'].tolist(), [[[True, False, False]]])
        self.assertEqual(evidence['weights'].tolist(), [[[.875, 0., 0.]]])

    def test_binary_head_has_zero_runner_up(self):
        x = make_inputs([[1, 0]], [[0, 1]], [[[.875, .125]]], classes=2)
        evidence = build_confusion_evidence(*x)
        self.assertEqual(evidence['weights'].tolist(), [[[.765625, .875]]])

    def test_cam_weights_are_bounded_uncalibrated_scores(self):
        # Out-of-range CAMs are clamped by the documented evidence builder.
        x = make_inputs([[1, 2, 0]], [[2, 1, 2]],
                        [[[2., -.5, -.25]], [[-1., 3., -.5]]])
        evidence = build_confusion_evidence(*x)
        self.assertTrue((evidence['weights'] >= 0).all())
        self.assertTrue((evidence['weights'] <= 1).all())
        self.assertEqual(evidence['weights'].tolist(), [[[1., 1., 1.]]])

    def test_ema_pools_counts_instead_of_averaging_row_probabilities(self):
        estimator = OnlineDirectedConfusion(3, momentum=.5)
        first = make_inputs([[1]], [[2]], [[[1.]], [[0.]]])
        second = make_inputs([[1, 1, 1]], [[1, 1, 1]],
                             [[[1., 1., 1.]], [[0., 0., 0.]]])
        estimator.update(*first, synchronize=False)
        estimator.update(*second, synchronize=False)
        self.assertEqual(estimator.counts[1].tolist(), [0., 3., 1.])
        self.assertEqual(estimator.ema_counts[1].tolist(), [0., 1.5, .5])
        self.assertEqual(estimator.export()['ema_probabilities'][1], [0., .75, .25])
        self.assertEqual(estimator.image_observations[1].item(), 2.)
        self.assertEqual(estimator.seen_images.item(), 2)

    def test_missing_rows_keep_history_and_expose_unknown_and_staleness(self):
        estimator = OnlineDirectedConfusion(3, momentum=.5)
        estimator.update(*make_inputs([[1]], [[2]], [[[1.]], [[0.]]]), synchronize=False)
        old_row = estimator.ema_counts[1].clone()
        estimator.update(*make_inputs([[2]], [[1]], [[[0.]], [[1.]]]), synchronize=False)
        torch.testing.assert_close(estimator.ema_counts[1], old_row, rtol=0, atol=0)
        exported = estimator.export()
        self.assertEqual(exported['staleness_updates'], [None, 1, 0])
        self.assertIsNone(exported['probabilities'][0])
        self.assertIsNone(exported['ema_probabilities'][0])
        before = estimator.ema_counts.clone()
        estimator.update(*make_inputs([[255]], [[0]], [[[0.]], [[0.]]]), synchronize=False)
        torch.testing.assert_close(estimator.ema_counts, before, rtol=0, atol=0)
        self.assertEqual(estimator.export()['staleness_updates'], [None, 2, 1])

    def test_state_restore_is_exact_and_different_stage_is_rejected(self):
        estimator = OnlineDirectedConfusion(3, momentum=.5, stage=2)
        x = make_inputs([[1, 2]], [[2, 1]], [[[1., 0.]], [[0., 1.]]])
        estimator.update(*x, synchronize=False)
        resumed = OnlineDirectedConfusion(3, momentum=.5, stage=2)
        resumed.load_state_dict(copy.deepcopy(estimator.state_dict()))
        self.assertEqual(resumed.export(), estimator.export())
        estimator.update(*x, synchronize=False)
        resumed.update(*x, synchronize=False)
        self.assertEqual(resumed.export(), estimator.export())
        with self.assertRaisesRegex(ValueError, 'stage mismatch'):
            OnlineDirectedConfusion(3, momentum=.5, stage=1).load_state_dict(estimator.state_dict())

    def test_detached_observation_preserves_input_and_rng_state(self):
        x = list(make_inputs([[1, 2]], [[2, 1]], [[[1., 0.]], [[0., 1.]]]))
        x[0].requires_grad_(True)
        x[2].requires_grad_(True)
        before = [item.clone() for item in x[:3]]
        rng = torch.random.get_rng_state().clone()
        estimator = OnlineDirectedConfusion(3)
        evidence = estimator.update(*x, synchronize=False)
        for value in evidence.values():
            self.assertFalse(value.requires_grad)
        for actual, original in zip(x[:3], before):
            torch.testing.assert_close(actual, original, rtol=0, atol=0)
        self.assertIsNone(x[0].grad)
        self.assertIsNone(x[2].grad)
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
        self.assertFalse(any(v.requires_grad for v in estimator.buffers()))

    def test_count_direction_independent_of_evidence_builder(self):
        rows = torch.tensor([1, 1, 2, 255, -1])
        columns = torch.tensor([2, 2, 1, 0, 2])
        expected = torch.zeros(3, 3, dtype=torch.float64)
        expected[1, 2] = 2
        expected[2, 1] = 1
        torch.testing.assert_close(pair_counts(rows, columns, 3), expected, rtol=0, atol=0)

    def test_invalid_constructor_configuration_is_rejected_immediately(self):
        for kwargs in ({'classes': 1}, {'classes': 3, 'momentum': 1.},
                       {'classes': 3, 'momentum': -.1},
                       {'classes': 3, 'low_threshold': .8, 'high_threshold': .7},
                       {'classes': 3, 'low_threshold': -.1},
                       {'classes': 3, 'high_threshold': 1.1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                OnlineDirectedConfusion(**kwargs)

    def test_invalid_input_shapes_nonfinite_values_and_box_are_rejected(self):
        x = make_inputs([[1, 2]], [[2, 1]], [[[1., 0.]], [[0., 1.]]])
        malformed = []
        malformed.append((x[0], x[1][0], x[2], x[3]))
        malformed.append((x[0], x[1], x[2][:, :1], x[3]))
        malformed.append((x[0], x[1], x[2], [[0, 2, 0, 2]]))
        malformed.append((x[0], x[1], x[2], []))
        bad_logits = x[0].clone()
        bad_logits[0, 0, 0, 0] = float('nan')
        malformed.append((bad_logits, x[1], x[2], x[3]))
        bad_cams = x[2].clone()
        bad_cams[0, 0, 0, 0] = float('inf')
        malformed.append((x[0], x[1], bad_cams, x[3]))
        for index, inputs in enumerate(malformed):
            with self.subTest(index=index), self.assertRaises(ValueError):
                build_confusion_evidence(*inputs)
        with self.assertRaises(ValueError):
            OnlineDirectedConfusion(4).update(*x, synchronize=False)

    def test_broad_relationship_keeps_foreground_rejected_by_cam_gate(self):
        estimator = OnlineDirectedConfusion(3, momentum=.5)
        x = make_inputs([[1]], [[2]], [[[.5]], [[.25]]])
        evidence = estimator.update(*x, synchronize=False)
        self.assertTrue(evidence['broad'].all())
        self.assertFalse(evidence['accepted'].any())
        self.assertEqual(estimator.broad_counts[1, 2].item(), 1.)
        self.assertEqual(estimator.broad_ema_counts[1, 2].item(), 1.)
        self.assertEqual(estimator.broad_image_observations.tolist(), [0., 1., 0.])
        self.assertEqual(estimator.broad_pair_image_observations[1, 2].item(), 1.)
        self.assertEqual(float(estimator.counts.sum()), 0.)
        self.assertEqual(float(estimator.ema_counts.sum()), 0.)
        self.assertEqual(float(estimator.pair_image_observations.sum()), 0.)
        exported = estimator.export()
        self.assertEqual(exported['broad_ema_probabilities'][1], [0., 0., 1.])
        self.assertIsNone(exported['ema_probabilities'][1])
        self.assertEqual(exported['broad_staleness_updates'], [None, 0, None])
        self.assertIn('trusted_reference', exported['relation_views'])

    def test_pair_observations_count_images_not_repeated_pixels(self):
        x = make_inputs([[1, 1, 1, 1, 2, 2]], [[2, 2, 2, 1, 1, 1]],
                        [[[.875, .875, .875, .875, .125, .125]],
                         [[.125, .125, .125, .125, .875, .875]]])
        estimator = OnlineDirectedConfusion(3)
        estimator.update(*x, synchronize=False)
        self.assertEqual(estimator.accepted_counts[1, 2].item(), 3.)
        self.assertEqual(estimator.broad_counts[2, 1].item(), 2.)
        for field in ('pair_image_observations', 'broad_pair_image_observations'):
            actual = getattr(estimator, field)
            self.assertEqual(actual[1, 2].item(), 1.)
            self.assertEqual(actual[1, 1].item(), 1.)
            self.assertEqual(actual[2, 1].item(), 1.)
            self.assertEqual(float(actual.sum()), 3.)
        # Repeated images are separate exposures, but each image still contributes
        # at most one observation to each directed pair.
        doubled = tuple(torch.cat((value, value), dim=0) for value in x[:3]) + (x[3] * 2,)
        estimator.update(*doubled, synchronize=False)
        for field in ('pair_image_observations', 'broad_pair_image_observations'):
            actual = getattr(estimator, field)
            self.assertEqual(actual[1, 2].item(), 3.)
            self.assertEqual(actual[2, 1].item(), 3.)
            self.assertEqual(float(actual.sum()), 9.)
        self.assertEqual(estimator.image_observations.tolist(), [0., 3., 3.])
        self.assertEqual(estimator.broad_image_observations.tolist(), [0., 3., 3.])
        self.assertEqual(estimator.seen_images.item(), 3)

    def test_broad_missing_row_preserves_history_and_tracks_its_own_staleness(self):
        estimator = OnlineDirectedConfusion(3, momentum=.5)
        estimator.update(*make_inputs([[1]], [[2]], [[[.5]], [[.25]]]), synchronize=False)
        saved_row = estimator.broad_ema_counts[1].clone()
        estimator.update(*make_inputs([[2]], [[1]], [[[.25]], [[.5]]]), synchronize=False)
        torch.testing.assert_close(estimator.broad_ema_counts[1], saved_row, rtol=0, atol=0)
        self.assertEqual(estimator.export()['broad_staleness_updates'], [None, 1, 0])
        self.assertEqual(estimator.export()['staleness_updates'], [None, None, None])
        estimator.update(*make_inputs([[255]], [[0]], [[[0.]], [[0.]]]), synchronize=False)
        torch.testing.assert_close(estimator.broad_ema_counts[1], saved_row, rtol=0, atol=0)
        self.assertEqual(estimator.export()['broad_staleness_updates'], [None, 2, 1])
        estimator.update(*make_inputs([[1, 1]], [[1, 1]], [[[.5, .5]], [[.25, .25]]]), synchronize=False)
        self.assertEqual(estimator.broad_ema_counts[1].tolist(), [0., 1., .5])
        self.assertEqual(estimator.export()['broad_staleness_updates'], [None, 0, 2])
        self.assertIsNone(estimator.export()['ema_probabilities'][1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
