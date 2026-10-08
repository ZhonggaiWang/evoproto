"""Meaningful CPU tests for schema4, trusted-new-source geometry selection.

Run with PYTHONPATH pointing to the frozen c_new_anchor/src. Do not run this
against the unchanged A/B schema3 module: rejecting old sources is the change.
"""
import copy
import unittest

import torch

from model.geometry_pair_selector import GeometryPairSelector
from model.online_directed_confusion import OnlineDirectedConfusion


def observer():
    obj = OnlineDirectedConfusion(5, stage=2)
    obj.updates.fill_(100)
    obj.last_seen_update.fill_(99)
    obj.broad_last_seen_update.fill_(99)
    obj.broad_image_observations.fill_(8)
    obj.image_observations.fill_(3)
    obj.pair_image_observations.fill_(3)
    obj.broad_pair_image_observations.fill_(3)
    return obj


def selector(**kwargs):
    cfg = dict(stage=2, refresh_interval=50, min_row_images=8,
               min_pair_images=3, min_rate=.01, min_updates=100,
               ramp_updates=200, max_stale_updates=5)
    cfg.update(kwargs)
    return GeometryPairSelector(5, old_classes=2, **cfg)


def assert_state_exact(test, a, b):
    test.assertEqual(a.keys(), b.keys())
    for key in a:
        if isinstance(a[key], torch.Tensor):
            torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
        else:
            test.assertEqual(a[key], b[key], key)


class NewAnchorSelectorTests(unittest.TestCase):
    def test_old_and_background_sources_never_selected_despite_max_mass_and_support(self):
        obs = observer()
        obs.broad_ema_counts[0, 3] = 1e12
        obs.broad_ema_counts[1, 3] = 1e11
        obs.broad_ema_counts[2, 4] = 1e10
        obs.broad_ema_counts[3, 1] = 10.
        obs.pair_image_observations[:3].fill_(10000)
        obs.broad_pair_image_observations[:3].fill_(10000)
        obs.image_observations[:3].fill_(10000)
        obs.broad_image_observations[:3].fill_(10000)
        chosen = selector()
        targets = chosen.update(obs, 2000)
        self.assertEqual(targets[:3].tolist(), [-1, -1, -1])
        self.assertEqual(chosen.selected_rates[:3].tolist(), [0., 0., 0.])
        self.assertEqual(int(targets[3]), 1)
        self.assertEqual(float(chosen.selected_rates[3]), 1.)

    def test_new_source_ranks_all_old_and_new_foreground_competitors(self):
        cases = [([0., 60., 20., 0., 80.], 4, .5),
                 ([0., 80., 20., 0., 60.], 1, .5),
                 ([0., 60., 60., 0., 60.], 1, 1/3)]
        for row, expected, rate in cases:
            with self.subTest(row=row):
                obs = observer()
                obs.broad_ema_counts[3] = torch.tensor(row, dtype=torch.float64)
                chosen = selector()
                self.assertEqual(int(chosen.update(obs, 2000)[3]), expected)
                self.assertAlmostEqual(float(chosen.selected_rates[3]), rate, places=14)

    def test_background_and_self_only_affect_full_row_normalization(self):
        obs = observer()
        obs.broad_ema_counts[3] = torch.tensor([80., 50., 0., 60., 10.], dtype=torch.float64)
        chosen = selector()
        self.assertEqual(int(chosen.update(obs, 2000)[3]), 1)
        self.assertAlmostEqual(float(chosen.selected_rates[3]), .25, places=14)
        # Renormalizing only the foreground competitors would incorrectly
        # pass .30. The original BG+diagonal mass must stay in the denominator.
        stricter = selector(min_rate=.30)
        self.assertEqual(int(stricter.update(obs, 2000)[3]), -1)
        self.assertEqual(float(stricter.selected_rates[3]), 0.)

    def test_trusted_and_broad_pair_support_screen_before_competitor_ranking(self):
        for unsupported in ('pair_image_observations', 'broad_pair_image_observations'):
            with self.subTest(unsupported=unsupported):
                obs = observer()
                obs.broad_ema_counts[3, 1] = 90.
                obs.broad_ema_counts[3, 4] = 10.
                getattr(obs, unsupported)[3, 1] = 2
                chosen = selector()
                self.assertEqual(int(chosen.update(obs, 2000)[3]), 4)
                self.assertAlmostEqual(float(chosen.selected_rates[3]), .1, places=14)

    def test_trusted_stale_new_source_cache_withdraws_before_next_refresh(self):
        obs = observer(); obs.broad_ema_counts[3, 1] = 10.
        chosen = selector()
        self.assertEqual(int(chosen.update(obs, 2000)[3]), 1)
        obs.updates.fill_(106)
        obs.broad_last_seen_update.fill_(105)  # broad stays fresh; trusted row ages6
        targets = chosen.update(obs, 2001)
        self.assertEqual(int(targets[3]), -1)
        self.assertEqual(float(chosen.selected_rates[3]), 0.)
        self.assertEqual(int(chosen.last_refresh_iteration), 2000)

    def test_cached_old_source_is_immediately_removed_without_refresh(self):
        obs = observer(); obs.broad_ema_counts[3, 1] = 10.
        chosen = selector()
        chosen.update(obs, 2000)
        # Deliberate cache corruption simulates forbidden old-source entries;
        # schema rejection alone must not be the only source-domain protection.
        chosen.targets[:3] = torch.tensor([3, 3, 4], dtype=torch.long)
        chosen.selected_rates[:3] = torch.tensor([.8, .9, .7], dtype=torch.float64)
        targets = chosen.update(obs, 2001)
        self.assertEqual(targets[:3].tolist(), [-1, -1, -1])
        self.assertEqual(chosen.selected_rates[:3].tolist(), [0., 0., 0.])
        self.assertEqual(int(targets[3]), 1)
        self.assertEqual(int(chosen.last_refresh_iteration), 2000)

    def test_coldstart_warmup_ramp_and_empty_new_evidence(self):
        obs = observer(); chosen = selector()
        self.assertEqual(chosen.update(obs, 2000).tolist(), [-1]*5)
        obs.broad_ema_counts[3, 1] = 10.
        obs.updates.fill_(99); obs.last_seen_update.fill_(98); obs.broad_last_seen_update.fill_(98)
        self.assertEqual(chosen.update(obs, 2001).tolist(), [-1]*5)
        obs.updates.fill_(100); obs.last_seen_update.fill_(99); obs.broad_last_seen_update.fill_(99)
        self.assertEqual(int(chosen.update(obs, 2050)[3]), 1)
        for updates, ramp in [(0, 0.), (99, 0.), (100, 0.), (200, .5), (300, 1.), (1000, 1.)]:
            obs.updates.fill_(updates)
            self.assertEqual(chosen.ramp(obs), ramp)

    def test_new_source_screening_does_not_modify_any_observer_statistic(self):
        obs = observer()
        obs.broad_ema_counts[1, 3] = 90.
        obs.broad_ema_counts[3, 1] = 10.
        obs.broad_ema_counts[3, 4] = 20.
        before = copy.deepcopy(obs.state_dict())
        selector().update(obs, 2000)
        assert_state_exact(self, obs.state_dict(), before)

    def test_schema4_exact_resume_retains_cache_and_refreshes_identically(self):
        obs = observer(); obs.broad_ema_counts[3, 1] = 90.; obs.broad_ema_counts[3, 4] = 10.
        original = selector(); original.update(obs, 2000)
        self.assertEqual(original.get_extra_state()['schema'], 4)
        self.assertEqual(original.get_extra_state()['old_classes'], 2)
        restored = selector()
        restored.load_state_dict(copy.deepcopy(original.state_dict()), strict=True)
        assert_state_exact(self, original.state_dict(), restored.state_dict())
        self.assertEqual(original.export(), restored.export())
        obs.updates.fill_(101); obs.last_seen_update.fill_(100); obs.broad_last_seen_update.fill_(100)
        obs.broad_ema_counts[3, 1] = 10.; obs.broad_ema_counts[3, 4] = 90.
        for iteration, expected in [(2001, 1), (2050, 4)]:
            self.assertEqual(int(original.update(obs, iteration)[3]), expected)
            self.assertEqual(int(restored.update(obs, iteration)[3]), expected)
            assert_state_exact(self, original.state_dict(), restored.state_dict())
            self.assertEqual(original.export(), restored.export())

    def test_schema3_old_boundary_and_stage_states_are_rejected(self):
        state = copy.deepcopy(selector().state_dict())
        schema3 = copy.deepcopy(state)
        schema3['_extra_state']['schema'] = 3
        schema3['_extra_state']['domain'] = 'foreground old-new and new-new unordered geometry'
        with self.assertRaises(ValueError):
            selector().load_state_dict(schema3, strict=True)
        for wrong in (GeometryPairSelector(5, old_classes=1, stage=2),
                      GeometryPairSelector(5, old_classes=2, stage=1)):
            with self.subTest(metadata=wrong.get_extra_state()), self.assertRaises(ValueError):
                wrong.load_state_dict(copy.deepcopy(state), strict=True)


if __name__ == '__main__':
    unittest.main()
