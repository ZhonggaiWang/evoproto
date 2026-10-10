import unittest
import torch

from utils.ald_stats import valid_image_mask, supervision_counts, counts_record, COUNT_KEYS


class ALDSupervisionStatsTests(unittest.TestCase):
    def test_padding_is_excluded_and_class_counts_partition_real_image(self):
        before = torch.tensor([[[12, 12, 12], [255, 2, 13]]])
        after = torch.tensor([[[255, 255, 12], [255, 2, 13]]])
        teacher = torch.tensor([[[3, 0, 0], [2, 2, 0]]])
        labels = torch.tensor([[[3, 255, 255], [2, 2, 255]]])
        valid = valid_image_mask(labels, torch.tensor([[0, 2, 0, 2]]))
        rejected = (before == 12) & (after == 255)
        retained = rejected & (teacher == 0)
        cls = torch.zeros((1, 15)); cls[0, [2, 11]] = 1
        counts = supervision_counts(before, after, teacher, labels, rejected, retained, valid,
                                    old_classes=10, total_classes=15, cls_labels=cls)
        r = counts_record(counts)
        self.assertEqual(r['valid_box_pixels'], 4)
        self.assertEqual(r['before_new_cam_pixels'], 2)
        self.assertEqual(r['rejected_new_cam_pixels'], 2)
        self.assertEqual(r['rejected_teacher_old_pixels'], 1)
        self.assertEqual(r['rejected_teacher_bg_pixels'], 1)
        self.assertEqual(r['final_valid_pixels'], 3)
        self.assertEqual(r['final_old_pixels'] + r['final_bg_pixels'] + r['final_new_pixels'], 3)
        self.assertEqual(r['final_ignore_pixels'], 1)
        self.assertEqual(r['padding_labeled_pixels'], 0)
        self.assertEqual(r['supervised_fraction_of_valid_image'], .75)
        self.assertEqual(r['rejected_fraction_of_new_cam'], 1)

    def test_fallback_distinguishes_old_and_new_kept_classes(self):
        labels = torch.zeros((2, 1, 1), dtype=torch.long)
        masks = torch.zeros_like(labels, dtype=torch.bool)
        cls = torch.zeros((2, 15)); cls[:, [2, 11]] = 1
        gate = torch.zeros_like(cls); gate[0, 2] = 1; gate[1, 11] = 1
        peak = torch.ones_like(cls)
        c = supervision_counts(labels, labels, labels, labels, masks, masks, ~masks,
                               old_classes=10, total_classes=15, cls_labels=cls,
                               gate_labels=gate, cam_peak=peak, threshold=5.0)
        r = counts_record(c)
        self.assertEqual(r['fallback_images'], 2)
        self.assertEqual(r['fallback_old_images'], 1)
        self.assertEqual(r['fallback_new_images'], 1)
        self.assertEqual(r['gate_rejected_old_classes'], 1)
        self.assertEqual(r['gate_rejected_new_classes'], 1)

    def test_empty_image_box_has_explicit_null_ratios(self):
        labels = torch.full((1, 2, 2), 255)
        valid = valid_image_mask(labels, [[0, 0, 0, 2]])
        self.assertFalse(valid.any())
        r = counts_record(torch.zeros(len(COUNT_KEYS), dtype=torch.int64))
        self.assertIsNone(r['rejected_fraction_of_new_cam'])
        self.assertIsNone(r['supervised_fraction_of_valid_image'])

    def test_invalid_boxes_fail_instead_of_silently_counting_wrong_resolution(self):
        labels = torch.zeros((1, 2, 2))
        for box in [[[0, 448, 0, 448]], [[2, 1, 0, 2]], [[0., 2., 0., 2.]]]:
            with self.assertRaises(ValueError):
                valid_image_mask(labels, box)


if __name__ == '__main__':
    unittest.main()
