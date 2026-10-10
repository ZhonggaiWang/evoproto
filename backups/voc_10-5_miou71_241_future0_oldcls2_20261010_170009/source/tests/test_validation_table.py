"""The validation table must show the same mIoU as the evaluator."""
import unittest

import numpy as np

from utils.evaluate import scores
from utils.pyutils import format_tabs


class ValidationTableTests(unittest.TestCase):
    def check_display(self, prediction, expected):
        ground_truth = np.zeros((2, 2), dtype=np.int16)
        with np.errstate(divide='ignore', invalid='ignore'):
            score = scores([ground_truth], [prediction], num_classes=2)
        table = format_tabs([score], ['Seg_Pred'], cat_list=['background', 'foreground'])
        row = next(line for line in table.splitlines() if 'mIoU' in line)
        displayed = float(row.split('|')[2].strip())
        self.assertAlmostEqual(displayed, expected)
        self.assertAlmostEqual(displayed, score['miou'] * 100)

    def test_omitted_class_does_not_turn_displayed_mean_into_nan(self):
        self.check_display(np.zeros((2, 2), dtype=np.int16), 100.)

    def test_display_uses_evaluator_gt_class_policy_for_false_positive_only_class(self):
        self.check_display(np.array([[0, 1], [0, 1]], dtype=np.int16), 50.)


if __name__ == '__main__':
    unittest.main()
