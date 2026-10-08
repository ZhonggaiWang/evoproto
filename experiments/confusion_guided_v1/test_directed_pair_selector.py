"""Meaningful selector checks use the real serializable confusion observer."""
import copy
import unittest

import torch

from directed_pair_selector import DirectedPairSelector
from online_directed_confusion import OnlineDirectedConfusion


class DirectedPairSelectorTests(unittest.TestCase):
    def observer(self, classes=5, updates=100):
        observer = OnlineDirectedConfusion(classes, stage=1)
        observer.updates.fill_(updates)
        observer.broad_image_observations.fill_(8)
        observer.broad_pair_image_observations.fill_(3)
        observer.broad_last_seen_update.fill_(updates-1)
        return observer

    def test_unknown_and_coldstart(self):
        observer = self.observer(updates=99)
        observer.broad_ema_counts[1, 2] = 10
        selector = DirectedPairSelector(5)
        self.assertEqual(selector.update(observer, 2099).tolist(), [-1]*5)
        self.assertEqual(selector.ramp(observer), 0.)
        observer.updates.fill_(100)
        self.assertEqual(selector.update(observer, 2100).tolist(), [-1, 2, -1, -1, -1])
        self.assertEqual(selector.ramp(observer), 0.)
        self.assertEqual(int(selector.last_refresh_iteration), 2100)

    def test_direction_background_and_self_excluded(self):
        observer = self.observer()
        observer.broad_ema_counts[0, 4] = 100
        observer.broad_ema_counts[1] = torch.tensor([100, 100, 10, 0, 0.])
        observer.broad_ema_counts[2] = torch.tensor([0, 0, 100, 5, 0.])
        selector = DirectedPairSelector(5)
        self.assertEqual(selector.update(observer, 2000).tolist(), [-1, 2, 3, -1, -1])
        self.assertAlmostEqual(float(selector.selected_rates[1]), 10/210)
        self.assertNotEqual(int(selector.targets[2]), 1)

    def test_row_exposure_threshold(self):
        observer = self.observer()
        observer.broad_ema_counts[1, 2] = 10
        observer.broad_image_observations[1] = 7
        self.assertEqual(int(DirectedPairSelector(5).update(observer, 2000)[1]), -1)

    def test_pair_exposure_threshold_and_next_eligible(self):
        observer = self.observer()
        observer.broad_ema_counts[1, 2] = 20
        observer.broad_ema_counts[1, 3] = 10
        observer.broad_pair_image_observations[1, 2] = 2
        self.assertEqual(int(DirectedPairSelector(5).update(observer, 2000)[1]), 3)

    def test_rate_threshold_uses_whole_row_mass(self):
        observer = self.observer()
        observer.broad_ema_counts[1, 0] = 999
        observer.broad_ema_counts[1, 2] = 1
        self.assertEqual(int(DirectedPairSelector(5).update(observer, 2000)[1]), -1)
        observer.broad_ema_counts[1, 0] = 99
        selector = DirectedPairSelector(5)
        self.assertEqual(int(selector.update(observer, 2000)[1]), 2)
        self.assertAlmostEqual(float(selector.selected_rates[1]), .01)

    def test_stale_and_unseen_rows(self):
        observer = self.observer(updates=400)
        observer.broad_ema_counts[1, 2] = 10
        observer.broad_ema_counts[2, 1] = 10
        observer.broad_ema_counts[3, 1] = 10
        observer.broad_last_seen_update[1] = 198  # age 201: stale
        observer.broad_last_seen_update[2] = 199  # age 200: acceptable
        observer.broad_last_seen_update[3] = -1
        self.assertEqual(DirectedPairSelector(5).update(observer, 2000).tolist(), [-1, -1, 1, -1, -1])

    def test_deterministic_tie_lowest_foreground_id(self):
        observer = self.observer()
        observer.broad_ema_counts[3, 1] = 10
        observer.broad_ema_counts[3, 2] = 10
        selector = DirectedPairSelector(5)
        self.assertEqual(int(selector.update(observer, 2000)[3]), 1)

    def test_refresh_lag_and_observer_not_mutated(self):
        observer = self.observer()
        observer.broad_ema_counts[1, 2] = 10
        selector = DirectedPairSelector(5)
        before = copy.deepcopy(observer.state_dict())
        self.assertEqual(int(selector.update(observer, 2000)[1]), 2)
        for key, value in before.items():
            if isinstance(value, torch.Tensor):
                self.assertTrue(torch.equal(value, observer.state_dict()[key]))
        observer.broad_ema_counts[1, 3] = 50
        self.assertEqual(int(selector.update(observer, 2049)[1]), 2)
        self.assertEqual(int(selector.update(observer, 2050)[1]), 3)
        self.assertEqual(int(selector.last_refresh_iteration), 2050)

    def test_returned_target_is_not_mutable_state_alias(self):
        observer = self.observer()
        observer.broad_ema_counts[1, 2] = 10
        selector = DirectedPairSelector(5)
        result = selector.update(observer, 2000)
        result[1] = 4
        self.assertEqual(int(selector.targets[1]), 2)

    def test_stale_cached_row_immediately_withdrawn(self):
        observer = self.observer(updates=300)
        observer.broad_ema_counts[1, 2] = 10
        observer.broad_last_seen_update[1] = 99  # age 200 at selection
        selector = DirectedPairSelector(5)
        self.assertEqual(int(selector.update(observer, 2000)[1]), 2)
        observer.updates.add_(1)
        self.assertEqual(int(selector.update(observer, 2001)[1]), -1)
        self.assertEqual(float(selector.selected_rates[1]), 0.)
        self.assertEqual(int(selector.last_refresh_iteration), 2000)

    def test_exact_state_resume_and_ramp(self):
        observer = self.observer(updates=200)
        observer.broad_ema_counts[1, 2] = 10
        selector = DirectedPairSelector(5)
        selector.update(observer, 2200)
        resumed = DirectedPairSelector(5)
        resumed.load_state_dict(copy.deepcopy(selector.state_dict()))
        self.assertTrue(torch.equal(selector.targets, resumed.targets))
        self.assertTrue(torch.equal(selector.selected_rates, resumed.selected_rates))
        self.assertEqual(selector.export(), resumed.export())
        self.assertEqual(selector.ramp(observer), .5)
        observer.broad_ema_counts[1, 3] = 50
        self.assertTrue(torch.equal(selector.update(observer, 2249), resumed.update(observer, 2249)))
        self.assertTrue(torch.equal(selector.update(observer, 2250), resumed.update(observer, 2250)))
        observer.updates.fill_(300)
        self.assertEqual(selector.ramp(observer), 1.)
        observer.updates.fill_(500)
        self.assertEqual(selector.ramp(observer), 1.)

    def test_load_rejects_stage_and_configuration_mismatch(self):
        state = DirectedPairSelector(5).state_dict()
        for options in [{'stage': 2}, {'refresh_interval': 10}, {'min_rate': .02},
                        {'max_stale_updates': 50}]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                DirectedPairSelector(5, **options).load_state_dict(state)

    def test_observer_stage_mismatch_and_bad_iteration_rejected(self):
        observer = self.observer()
        selector = DirectedPairSelector(5, stage=2)
        with self.assertRaises(ValueError):
            selector.update(observer, 2000)
        selector = DirectedPairSelector(5)
        with self.assertRaises(ValueError):
            selector.update(observer, -1)
        selector.update(observer, 2000)
        with self.assertRaises(ValueError):
            selector.update(observer, 1999)

    def test_invalid_configurations_and_nonfinite_counts_rejected(self):
        for options in [{'classes': 1}, {'classes': 5, 'refresh_interval': 0},
                        {'classes': 5, 'ramp_updates': 0},
                        {'classes': 5, 'min_rate': float('nan')},
                        {'classes': 5, 'min_pair_images': -1}]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                DirectedPairSelector(**options)
        observer = self.observer()
        observer.broad_ema_counts[1, 2] = float('nan')
        with self.assertRaises(ValueError):
            DirectedPairSelector(5).update(observer, 2000)


if __name__ == '__main__':
    unittest.main()
