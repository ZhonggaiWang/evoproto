"""Exercise shared training targets, confusion counts and raw CAM peak extraction."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from continual.Trainer import Trainer as VOCTrainer
from continual.Trainer_coco import Trainer as COCOTrainer
from continual import prototype_supervision as supervision
from utils.camutils import multi_scale_cam2, cam_high_pass_filter
from utils.ald_stats import counts_record, supervision_counts


class IdentityPAR(torch.nn.Module):
    def forward(self, images, cams):
        return cams


class FixtureTeacher(torch.nn.Module):
    def forward(self, inputs, step0=False):
        # Zero is excluded; positive logits below 2.0 are now accepted.
        # Pixel class 1 still wins independently of the image threshold.
        cls = inputs.new_tensor([[0., .1, 1.]])
        labels = torch.ones(1, 8, 8, dtype=torch.long)
        logits = F.one_hot(labels, 4).permute(0, 3, 1, 2).float() * 10
        return cls, None, logits, None, torch.ones(4, 2)


class FixtureStudent(torch.nn.Module):
    def forward(self, inputs, cal_sim=False):
        labels = torch.full((1, 8, 8), 2, dtype=torch.long)
        labels[:, :, 4:] = 1
        logits = F.one_hot(labels, 5).permute(0, 3, 1, 2).float() * 10
        return None, logits, None, None


class PaperSupervisionPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.args = SimpleNamespace(cam_scales=[1.], ald=True, ald_mode='paper', bkg_thre=.5,
                                    high_thre=.7, low_thre=.25, ignore_index=255,
                                    crop_size=8, confusion_temperature=.01, old_cls_threshold=0.0)
        self.student, self.teacher = FixtureStudent(), FixtureTeacher()
        self.inputs = torch.zeros(1, 3, 8, 8)
        self.tags = torch.tensor([[1, 0, 0, 1]])
        self.box = torch.tensor([[0, 8, 0, 6]])
        # Old 1 (exact teacher logit 0.0) is very strong but must be excluded.
        # Old 2 masks the new CAM unless image ALD removes its low peak.
        cams = torch.zeros(1, 4, 8, 8)
        cams[:, 0] = .99
        cams[:, 1] = .95
        cams[:, 2, :, :4] = .85
        cams[:, 3, :, 4:] = .9
        self.cam_outputs = (cams, cams.clone(), torch.tensor([[100., 4., 8., 1.]]))
        self.data = [('image', self.inputs, self.tags, self.box, None)]

    def build(self, tags=None):
        with patch.object(supervision, 'multi_scale_cam2', return_value=self.cam_outputs), \
                patch.object(supervision, 'multi_scale_cam2_filter', return_value=self.cam_outputs):
            return supervision.build_incremental_supervision(
                self.student, self.teacher, self.inputs, self.tags if tags is None else tags,
                self.box, IdentityPAR(), self.args, total_classes=4, new_classes=1,
            )

    def test_threshold_and_ald_filter_image_tags_then_merge_actual_targets(self):
        targets = self.build()
        torch.testing.assert_close(targets.cls_labels_before_ald, torch.tensor([[0, 1, 1, 1]]))
        torch.testing.assert_close(targets.cls_labels, torch.tensor([[0, 0, 1, 1]]))
        torch.testing.assert_close(targets.ald_thresholds, torch.tensor([[6.]]))
        self.assertTrue((targets.old_labels == 1).all())
        self.assertTrue((targets.refined_labels[:, :, :3] == 3).all())
        # Refinement can leave an uncertain boundary; merging still uses teacher there.
        self.assertTrue((targets.refined_labels[:, :, 3] == 255).all())
        # Old CAM and image threshold do not remove teacher pixel argmax class 1.
        self.assertTrue((targets.merged_labels[:, :, :4] == 1).all())
        self.assertTrue((targets.merged_labels[:, :, 4:6] == 4).all())
        self.assertTrue((targets.merged_labels[:, :, 6:] == 255).all())

    def test_disabling_ald_keeps_teacher_presence_threshold_and_new_gt_tags(self):
        self.args.ald = False
        targets = self.build()
        torch.testing.assert_close(targets.cls_labels, torch.tensor([[0, 1, 1, 1]]))
        self.assertTrue((targets.merged_labels[:, :, :6] == 1).all())
        self.assertEqual(self.build(torch.tensor([[1, 1, 1, 0]])).cls_labels[0, -1].item(), 0)

    def fixed_fixture(self):
        self.args.ald_mode = 'fixed'
        cams = torch.zeros(1, 4, 8, 8)
        cams[:, 0] = .99  # Excluded by teacher presence, regardless of CAM peak.
        cams[:, 1] = .2
        cams[:, 2, :, :4] = .8
        cams[:, 3, :, 4:] = .9
        self.cam_outputs = (cams, cams.clone(), torch.tensor([[100., 4., 13., 4.5]]))

    def test_fixed_voc_filters_spatial_labels_without_filtering_classification_tags(self):
        self.fixed_fixture()
        self.args.dataset = 'voc'
        self.args.ald_threshold = 5.0
        targets = self.build()
        torch.testing.assert_close(targets.cls_labels, torch.tensor([[0, 1, 1, 1]]))
        torch.testing.assert_close(targets.gate_labels, torch.tensor([[0, 0, 1, 0]]))
        torch.testing.assert_close(targets.ald_thresholds, torch.tensor([[5.0]]))
        self.assertTrue((targets.refined_labels_before_ald[:, :, 4:6] == 4).all())
        self.assertTrue((targets.refined_labels[:, :, 4:6] == 255).all())
        self.assertTrue(targets.rejected_mask[:, :, 4:6].all())
        # The original fixed filter refills rejected NEW with teacher OLD.
        self.assertTrue((targets.merged_labels[:, :, :6] == 1).all())
        self.assertTrue((targets.merged_labels[:, :, 6:] == 255).all())
        self.assertFalse(targets.retained_ignore_mask.any())
        stats = counts_record(supervision_counts(
            targets.refined_labels_before_ald, targets.refined_labels, targets.old_labels,
            targets.merged_labels, targets.rejected_mask, targets.retained_ignore_mask, targets.valid_mask,
            old_classes=3, total_classes=4, cls_labels=targets.cls_labels_before_ald,
            gate_labels=targets.gate_labels, cam_peak=targets.cam_peaks,
            threshold=targets.ald_thresholds, image_adaptive=False,
        ))
        self.assertEqual(stats['gate_rejected_new_classes'], 1)
        self.assertEqual(stats['rejected_new_cam_pixels'], 16)
        self.assertEqual(stats['rejected_final_old_pixels'], 16)
        self.assertEqual(stats['padding_labeled_pixels'], 0)

    def test_fixed_coco_default_and_exact_threshold_keep_new_spatial_labels(self):
        self.fixed_fixture()
        self.args.dataset = 'coco2voc'
        self.cam_outputs[2][:, 3] = 7.5
        targets = self.build()
        torch.testing.assert_close(targets.ald_thresholds, torch.tensor([[7.5]]))
        torch.testing.assert_close(targets.gate_labels, torch.tensor([[0, 0, 1, 1]]))
        self.assertTrue((targets.merged_labels[:, :, 4:6] == 4).all())
        self.args.ald_threshold = 8.
        self.assertTrue((self.build().merged_labels[:, :, :6] == 1).all())

    def test_fixed_all_filtered_fallback_uses_strongest_positive_candidate(self):
        tags = torch.tensor([[0, 1, 1, 1], [0, 0, 0, 0]])
        peaks = torch.tensor([[100., 4., 8., 1.], [100., 4., 8., 1.]])
        gated = cam_high_pass_filter(peaks, tags, 12.5)
        torch.testing.assert_close(gated, torch.tensor([[0, 0, 1, 0], [0, 0, 0, 0]]))
        torch.testing.assert_close(tags, torch.tensor([[0, 1, 1, 1], [0, 0, 0, 0]]))

    def test_fixed_confusion_in_both_trainers_uses_same_filtered_pixel_targets(self):
        self.fixed_fixture()
        self.args.ald_threshold = 5.0
        with patch.object(supervision, 'multi_scale_cam2_filter', return_value=self.cam_outputs), \
                patch.object(supervision, 'PAR', return_value=IdentityPAR()), \
                patch.object(supervision.dist, 'is_initialized', return_value=False):
            for trainer_class in (VOCTrainer, COCOTrainer):
                trainer = SimpleNamespace(total_classes=4, new_classes=1, model_old=self.teacher,
                                          device=torch.device('cpu'), args=self.args)
                scores = trainer_class.cal_sim(trainer, self.student, self.data, self.args)
                self.assertEqual(trainer.confusion_counts[1].sum().item(), 48)
                self.assertEqual(trainer.confusion_counts[4].sum().item(), 0)
                self.assertAlmostEqual(scores[1, 2].item(), 2 / 3, places=6)

    def test_fixed_policy_is_default_and_ald_disable_removes_spatial_filter_only(self):
        self.fixed_fixture()
        del self.args.ald_mode
        targets = self.build()
        self.assertEqual(targets.ald_mode, 'fixed')
        torch.testing.assert_close(targets.ald_thresholds, torch.tensor([[0.0]]))
        torch.testing.assert_close(targets.gate_labels, targets.cls_labels_before_ald)
        self.assertFalse(targets.rejected_mask.any())
        self.assertTrue((targets.merged_labels[:, :, 4:6] == 4).all())
        self.assertTrue((targets.merged_labels[:, :, 6:] == 255).all())
        self.args.ald = False
        targets = self.build()
        self.assertEqual(targets.ald_mode, 'off')
        self.assertIsNone(targets.gate_labels)
        torch.testing.assert_close(targets.cls_labels, torch.tensor([[0, 1, 1, 1]]))
        self.assertTrue((targets.merged_labels[:, :, 4:6] == 4).all())
        self.args.ald = True
        self.args.ald_mode = 'legacy'
        self.assertEqual(self.build().ald_mode, 'fixed')
        self.args.ald_mode = 'off'
        self.assertEqual(self.build().ald_mode, 'off')
        for threshold in (-1., float('nan'), float('inf')):
            self.args.ald_mode = 'fixed'
            self.args.ald_threshold = threshold
            with self.assertRaises(ValueError):
                self.build()

    def test_both_trainers_count_merged_targets_reset_counts_and_restore_mode(self):
        expected_counts = torch.zeros(5, 5, dtype=torch.long)
        expected_counts[1, 2], expected_counts[4, 1] = 32, 16
        expected_scores = torch.zeros(5, 5)
        expected_scores[1, 2] = expected_scores[2, 1] = 1
        expected_scores[4, 1] = expected_scores[1, 4] = 1
        with patch.object(supervision, 'multi_scale_cam2', return_value=self.cam_outputs), \
                patch.object(supervision, 'PAR', return_value=IdentityPAR()), \
                patch.object(supervision.dist, 'is_initialized', return_value=False):
            for trainer_class in (VOCTrainer, COCOTrainer):
                trainer = SimpleNamespace(total_classes=4, new_classes=1, model_old=self.teacher,
                                          device=torch.device('cpu'), args=self.args)
                for training in (True, False):
                    self.student.train(training)
                    scores = trainer_class.cal_sim(trainer, self.student, self.data, self.args)
                    torch.testing.assert_close(scores, expected_scores)
                    torch.testing.assert_close(trainer.confusion_counts, expected_counts)
                    self.assertEqual(self.student.training, training)
                counterparts, weights = trainer_class.get_weight(trainer, scores, 4, 1)
                self.assertEqual(counterparts[1].item(), 4)
                self.assertEqual(counterparts[4].item(), 1)
                self.assertEqual(weights[1, 4].item(), 1)

    def test_confusion_restores_training_mode_even_when_scan_fails(self):
        trainer = SimpleNamespace(total_classes=4, new_classes=1, model_old=self.teacher,
                                  device=torch.device('cpu'))
        self.student.train()
        with patch.object(supervision, 'PAR', return_value=IdentityPAR()), \
                patch.object(supervision, 'build_incremental_supervision', side_effect=RuntimeError('fixture')):
            with self.assertRaisesRegex(RuntimeError, 'fixture'):
                supervision.compute_confusion(trainer, self.student, self.data, self.args)
        self.assertTrue(self.student.training)

    def test_distributed_counts_are_reduced_before_normalization(self):
        trainer = SimpleNamespace(total_classes=4, new_classes=1, model_old=self.teacher,
                                  device=torch.device('cpu'))
        with patch.object(supervision, 'multi_scale_cam2', return_value=self.cam_outputs), \
                patch.object(supervision, 'PAR', return_value=IdentityPAR()), \
                patch.object(supervision.dist, 'is_available', return_value=True), \
                patch.object(supervision.dist, 'is_initialized', return_value=True), \
                patch.object(supervision.dist, 'all_reduce', side_effect=lambda counts, op: counts.mul_(2)) as reduce:
            scores = supervision.compute_confusion(trainer, self.student, self.data, self.args)
        reduce.assert_called_once()
        self.assertEqual(trainer.confusion_counts.sum().item(), 96)
        self.assertEqual(scores[1, 4].item(), 1)

    def test_cam_peak_is_raw_original_branch_before_flip_scales_and_normalization(self):
        original = torch.tensor([[[[-3., -2.], [-4., -5.]], [[1., 4.], [2., 3.]]]])
        flipped = torch.full_like(original, 50.)
        base = torch.cat((original, flipped))
        scaled = torch.full_like(base, 100.)
        calls = [(base, base), (scaled, scaled)]
        with patch.object(self.student, 'forward', side_effect=calls):
            cams, aux, peaks = multi_scale_cam2(self.student, self.inputs, [1., 2.], return_peaks=True)
        torch.testing.assert_close(peaks, torch.tensor([[-2., 4.]]))
        with patch.object(self.student, 'forward', side_effect=calls):
            default_cam, default_aux = multi_scale_cam2(self.student, self.inputs, [1., 2.])
        torch.testing.assert_close(cams, default_cam)
        torch.testing.assert_close(aux, default_aux)


if __name__ == '__main__':
    unittest.main()
