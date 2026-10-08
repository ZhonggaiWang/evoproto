"""Check the actual meter without importing optional logging dependencies."""

import ast
import gc
from pathlib import Path
import unittest
import weakref

import torch


source = Path(__file__).resolve().parents[1] / "utils" / "pyutils.py"
tree = ast.parse(source.read_text())
meter_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AverageMeter")
namespace = {}
exec(compile(ast.Module(body=[meter_class], type_ignores=[]), str(source), "exec"), namespace)
AverageMeter = namespace["AverageMeter"]


class TrainingMeterTests(unittest.TestCase):
    def test_logging_an_unused_loss_does_not_retain_its_graph(self):
        meter = AverageMeter()
        parameter = torch.tensor(2.0, requires_grad=True)
        loss = parameter.square()
        parameter_reference = weakref.ref(parameter)
        meter.add({"loss": loss})
        self.assertIsInstance(meter.get("loss"), float)
        self.assertEqual(meter.get("loss"), 4.0)
        del loss, parameter
        gc.collect()
        self.assertIsNone(parameter_reference())

    def test_average_and_pop_preserve_numeric_behavior(self):
        meter = AverageMeter("loss", "score")
        meter.add({"loss": torch.tensor(2.0, requires_grad=True), "score": 0.2})
        meter.add({"loss": torch.tensor(6.0, requires_grad=True), "score": 0.6})
        loss, score = meter.get("loss", "score")
        self.assertEqual(loss, 4.0)
        self.assertAlmostEqual(score, 0.4)
        self.assertEqual(meter.pop("loss"), 4.0)
        meter.add({"loss": torch.tensor(10.0)})
        self.assertEqual(meter.get("loss"), 10.0)


if __name__ == "__main__":
    unittest.main()
