"""CPU tests for geometry-domain selection from past online evidence."""
import copy
import unittest

import torch

try:
    from model.geometry_pair_selector import GeometryPairSelector
    from model.online_directed_confusion import OnlineDirectedConfusion
except ModuleNotFoundError:
    from geometry_pair_selector import GeometryPairSelector
    from online_directed_confusion import OnlineDirectedConfusion


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
    config = dict(stage=2, refresh_interval=50, min_row_images=8,
                  min_pair_images=3, min_rate=.01, min_updates=100,
                  ramp_updates=200, max_stale_updates=5)
    config.update(kwargs)
    return GeometryPairSelector(5, old_classes=2, **config)


def same_modules(actual, expected):
    a, b = actual.state_dict(), expected.state_dict()
    assert a.keys() == b.keys()
    for key in a:
        if isinstance(a[key], torch.Tensor):
            torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
        else:
            assert a[key] == b[key], key
    assert actual.export() == expected.export()


class GeometryPairSelectorTests(unittest.TestCase):
    def test_domain_screening_precedes_ranking_so_old_old_cannot_hide_new_pair(self):
        obs = observer(); obs.broad_ema_counts[1, 2] = 90.; obs.broad_ema_counts[1, 3] = 10.
        chosen = selector(); target = chosen.update(obs, 2000)
        self.assertEqual(int(target[1]), 3)
        self.assertAlmostEqual(float(chosen.selected_rates[1]), .1, places=14)

    def test_full_row_mass_keeps_background_diagonal_and_old_old_counts(self):
        obs = observer()
        obs.broad_ema_counts[1] = torch.tensor([20., 10., 60., 5., 5.], dtype=torch.float64)
        chosen = selector(); target = chosen.update(obs, 2000)
        self.assertEqual(int(target[1]), 3)  # deterministic lowest ID on equal eligible rates
        self.assertAlmostEqual(float(chosen.selected_rates[1]), .05, places=14)

    def test_full_row_threshold_cannot_be_bypassed_by_renormalizing_geometry_pairs(self):
        obs = observer(); obs.broad_ema_counts[1, 2] = 1000.; obs.broad_ema_counts[1, 3] = 5.
        chosen = selector(); target = chosen.update(obs, 2000)
        self.assertEqual(int(target[1]), -1)
        self.assertEqual(float(chosen.selected_rates[1]), 0.)

    def test_trusted_pair_support_is_screened_before_ranking(self):
        obs = observer(); obs.broad_ema_counts[1, 3] = 90.; obs.broad_ema_counts[1, 4] = 10.
        obs.pair_image_observations[1, 3] = 2.
        chosen = selector(); target = chosen.update(obs, 2000)
        self.assertEqual(int(target[1]), 4)
        self.assertAlmostEqual(float(chosen.selected_rates[1]), .1, places=14)

    def test_broad_pair_support_is_still_required_after_trusted_screen(self):
        obs = observer(); obs.broad_ema_counts[1, 3] = 90.; obs.broad_ema_counts[1, 4] = 10.
        obs.broad_pair_image_observations[1, 3] = 2.
        chosen = selector(); target = chosen.update(obs, 2000)
        self.assertEqual(int(target[1]), 4)

    def test_old_new_and_new_new_are_both_valid_domains(self):
        obs = observer(); obs.broad_ema_counts[1, 3] = 10.; obs.broad_ema_counts[3, 4] = 10.
        chosen = selector(); targets = chosen.update(obs, 2000)
        self.assertEqual(int(targets[1]), 3)
        self.assertEqual(int(targets[3]), 4)

    def test_background_and_self_are_never_candidates_even_in_geometry_domain(self):
        obs = observer(); obs.broad_ema_counts[0, 3] = 100.
        obs.broad_ema_counts[3] = torch.tensor([60., 0., 10., 30., 0.], dtype=torch.float64)
        chosen = selector(); targets = chosen.update(obs, 2000)
        self.assertEqual(int(targets[0]), -1)
        self.assertEqual(int(targets[3]), 2)
        self.assertAlmostEqual(float(chosen.selected_rates[3]), .1, places=14)

    def test_trusted_stale_row_rejected_even_when_broad_row_is_current(self):
        obs = observer(); obs.broad_ema_counts[1, 3] = 10.; obs.last_seen_update[1] = 0
        chosen = selector(); targets = chosen.update(obs, 2000)
        self.assertEqual(int(targets[1]), -1)
        self.assertEqual(float(chosen.selected_rates[1]), 0.)

    def test_cached_trusted_stale_pair_is_immediately_withdrawn_without_refresh(self):
        obs = observer(); obs.broad_ema_counts[1, 3] = 10.
        chosen = selector(); self.assertEqual(int(chosen.update(obs, 2000)[1]), 3)
        obs.updates.fill_(106); obs.broad_last_seen_update.fill_(105)
        targets = chosen.update(obs, 2001)  # ranking refresh remains 49 iterations away
        self.assertEqual(int(targets[1]), -1)
        self.assertEqual(float(chosen.selected_rates[1]), 0.)
        self.assertEqual(int(chosen.last_refresh_iteration), 2000)

    def test_minimum_support_rate_and_maximum_staleness_are_inclusive(self):
        obs = observer(); obs.broad_ema_counts[1, 2] = 95.; obs.broad_ema_counts[1, 3] = 5.
        obs.updates.fill_(105); obs.broad_last_seen_update.fill_(104)
        chosen = selector(min_rate=.05)
        targets = chosen.update(obs, 2000)
        self.assertEqual(int(targets[1]), 3)
        self.assertAlmostEqual(float(chosen.selected_rates[1]), .05, places=14)

    def test_unknown_rows_remain_unknown_and_warmup_does_not_select(self):
        obs = observer(); chosen = selector()
        self.assertTrue(bool((chosen.update(obs, 2000) == -1).all()))
        obs.broad_ema_counts[1, 3] = 10.; obs.updates.fill_(99)
        self.assertTrue(bool((chosen.update(obs, 2001) == -1).all()))

    def test_geometry_screen_does_not_modify_observer_statistics(self):
        obs = observer(); obs.broad_ema_counts[1, 2] = 90.; obs.broad_ema_counts[1, 3] = 10.
        before = copy.deepcopy(obs.state_dict())
        selector().update(obs, 2000)
        after = obs.state_dict()
        for key in before:
            if isinstance(before[key], torch.Tensor):
                torch.testing.assert_close(after[key], before[key], atol=0, rtol=0)
            else:
                self.assertEqual(after[key], before[key])

    def test_schema3_resume_is_exact_and_retains_cache_then_refreshes_identically(self):
        obs = observer(); obs.broad_ema_counts[1, 3] = 90.; obs.broad_ema_counts[1, 4] = 10.
        original = selector(); original.update(obs, 2000)
        metadata = original.get_extra_state()
        self.assertEqual(metadata['schema'], 3)
        self.assertEqual(metadata['old_classes'], 2)
        restored = selector(); restored.load_state_dict(copy.deepcopy(original.state_dict()), strict=True)
        same_modules(restored, original)
        obs.updates.fill_(101); obs.last_seen_update.fill_(100); obs.broad_last_seen_update.fill_(100)
        obs.broad_ema_counts[1, 3] = 10.; obs.broad_ema_counts[1, 4] = 90.
        for iteration, expected in [(2001, 3), (2050, 4)]:
            self.assertEqual(int(original.update(obs, iteration)[1]), expected)
            self.assertEqual(int(restored.update(obs, iteration)[1]), expected)
            same_modules(restored, original)

    def test_resume_rejects_geometry_old_class_or_stage_metadata_mismatch(self):
        original = selector(); state = copy.deepcopy(original.state_dict())
        for wrong in [GeometryPairSelector(5, old_classes=1, stage=2), GeometryPairSelector(5, old_classes=2, stage=1)]:
            with self.subTest(metadata=wrong.get_extra_state()), self.assertRaises(ValueError):
                wrong.load_state_dict(copy.deepcopy(state), strict=True)

    def test_ramp_depends_on_past_observer_updates(self):
        obs = observer(); chosen = selector()
        for updates, expected in [(99, 0.), (100, 0.), (200, .5), (300, 1.), (1000, 1.)]:
            obs.updates.fill_(updates)
            self.assertEqual(chosen.ramp(obs), expected)

    def test_invalid_old_boundary_rejected(self):
        for count in [0, 4, -1, 5, True, 1.5]:
            with self.subTest(old_classes=count), self.assertRaises(ValueError):
                GeometryPairSelector(5, old_classes=count, stage=2)


if __name__ == '__main__':
    unittest.main()
