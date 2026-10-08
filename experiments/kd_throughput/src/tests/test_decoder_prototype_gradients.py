"""Exercise actual LargeFOV gradient paths and incremental checkpoint loading.

The full network requires optional training packages, so checkpoint tests load
the real Trainer.load_step_ckpt method from its AST and give it decoder-only
torch modules. No backbone, CUDA, or external data is needed.
"""
import ast
import logging
import os.path as osp
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from model.decoder.conv_head import LargeFOV
from model.losses import (
    get_seg_loss,
    prototype_distillation_loss,
    prototype_separation_loss,
)


ROOT = Path(__file__).resolve().parents[1]


def load_checkpoint_method():
    source = ROOT / "continual/Trainer.py"
    tree = ast.parse(source.read_text())
    trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Trainer")
    method = next(node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == "load_step_ckpt")
    namespace = {"torch": torch, "osp": osp, "logging": logging}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["load_step_ckpt"]


def decoder(classes):
    return LargeFOV(in_planes=4, out_planes=sum(classes), dilation=1, classes_list=classes)


def decoder_model(classes):
    model = torch.nn.Module()
    model.decoder = decoder(classes)
    return model


def assign_vectors(head, rows):
    """Ensure active, non-collinear pairs so regularizer gradients are meaningful."""
    with torch.no_grad():
        for group in head.class_prototypes:
            group.prototype.zero_()
        offset = 0
        for group in head.class_prototypes:
            for row in group.prototype:
                row[:2] = torch.tensor(rows[offset], dtype=row.dtype)
                offset += 1


class DecoderPrototypeGradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        self.rng = torch.random.fork_rng(devices=[])
        self.rng.__enter__()
        torch.manual_seed(21)
        self.addCleanup(self.rng.__exit__, None, None, None)

    def assert_nonzero_finite_gradient(self, parameter):
        self.assertIsNotNone(parameter.grad)
        self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(parameter.grad.abs().sum().item(), 0.0)

    def assert_pixel_path_has_no_gradient(self, head, features, type_seg):
        self.assertIsNone(features.grad)
        self.assertIsNone(type_seg.grad)
        self.assertIsNone(head.conv6.weight.grad)
        self.assertIsNone(head.conv7.weight.grad)
        for classifier in head.conv8:
            self.assertIsNone(classifier.weight.grad)

    def test_type_seg_supervision_updates_old_and_new_prototypes_and_feature_path(self):
        head = decoder([3, 2])
        features = torch.randn(1, 4, 3, 3, requires_grad=True)
        _, type_seg, _ = head(features)
        targets = torch.tensor([[[0, 1, 2], [3, 4, 255], [1, 3, 4]]])
        # This is the Trainer's actual prototype-pixel objective and temperature.
        loss = get_seg_loss(type_seg / 0.1, targets, class_weight=torch.ones(5))
        loss.backward()
        self.assert_nonzero_finite_gradient(features)
        self.assert_nonzero_finite_gradient(head.conv6.weight)
        self.assert_nonzero_finite_gradient(head.conv7.weight)
        for group in head.class_prototypes:
            self.assert_nonzero_finite_gradient(group.prototype)
            self.assertTrue((group.prototype.grad.abs().sum(dim=1) > 0).all())
        # The ordinary pixel classifier is a separate branch.
        for classifier in head.conv8:
            self.assertIsNone(classifier.weight.grad)

    def test_separation_alone_updates_prototypes_without_direct_pixel_gradients(self):
        head = decoder([3, 2])
        assign_vectors(head, [[-1, 0], [1, 0], [1, 1], [0, 1], [1, -1]])
        features = torch.randn(1, 4, 3, 3, requires_grad=True)
        _, type_seg, prototypes = head(features)
        type_seg.retain_grad()
        loss = prototype_separation_loss(prototypes)
        self.assertGreater(loss.item(), 0)
        loss.backward()
        self.assert_pixel_path_has_no_gradient(head, features, type_seg)
        for group in head.class_prototypes:
            self.assert_nonzero_finite_gradient(group.prototype)
        self.assertTrue(torch.equal(head.class_prototypes[0].prototype.grad[0], torch.zeros(512)))

    def test_kd_alone_updates_old_prototypes_and_detaches_teacher(self):
        student, teacher = decoder([3, 2]), decoder([3])
        assign_vectors(student, [[-1, 0], [1, 0], [1, 1], [0, 1], [1, -1]])
        assign_vectors(teacher, [[-1, 0], [0, 1], [1, -1]])
        features = torch.randn(1, 4, 3, 3, requires_grad=True)
        _, type_seg, prototypes = student(features)
        _, _, teacher_prototypes = teacher(torch.randn(1, 4, 3, 3))
        type_seg.retain_grad()
        loss = prototype_distillation_loss(prototypes, teacher_prototypes)
        self.assertGreater(loss.item(), 0)
        loss.backward()
        self.assert_pixel_path_has_no_gradient(student, features, type_seg)
        self.assert_nonzero_finite_gradient(student.class_prototypes[0].prototype)
        self.assertTrue(torch.equal(student.class_prototypes[0].prototype.grad[0], torch.zeros(512)))
        self.assertTrue(torch.equal(student.class_prototypes[1].prototype.grad, torch.zeros(2, 512)))
        for parameter in teacher.parameters():
            self.assertIsNone(parameter.grad)

    def test_previous_checkpoint_preserves_old_prototypes_and_independent_new_group(self):
        checkpoint_loader = load_checkpoint_method()
        temp_root = ROOT / "tests/.runtime-tmp"
        self.assertTrue(temp_root.resolve().is_relative_to(ROOT))
        self.assertFalse(temp_root.is_symlink())
        temp_root.mkdir(exist_ok=True)
        for old_classes in ([3], [3, 2]):
            with self.subTest(old_classes=old_classes):
                previous = decoder_model(old_classes)
                student = decoder_model([*old_classes, 2])
                teacher = decoder_model(old_classes)
                for parameter in teacher.parameters():
                    parameter.requires_grad_(False)
                old_snapshots = [group.prototype.detach().clone() for group in previous.decoder.class_prototypes]
                new_snapshot = student.decoder.class_prototypes[-1].prototype.detach().clone()
                context = SimpleNamespace(model=student, model_old=teacher, step=len(old_classes))
                with tempfile.TemporaryDirectory(dir=temp_root) as directory:
                    path = Path(directory) / "previous.pth"
                    # Exercise actual loading of legacy DDP-prefixed checkpoints.
                    state = {"module." + key: value for key, value in previous.state_dict().items()}
                    torch.save({"model_state": state}, path)
                    checkpoint_loader(context, str(path))
                for index, snapshot in enumerate(old_snapshots):
                    torch.testing.assert_close(student.decoder.class_prototypes[index].prototype, snapshot, rtol=0, atol=0)
                    torch.testing.assert_close(teacher.decoder.class_prototypes[index].prototype, snapshot, rtol=0, atol=0)
                    self.assertNotEqual(student.decoder.class_prototypes[index].prototype.data_ptr(),
                                        teacher.decoder.class_prototypes[index].prototype.data_ptr())
                torch.testing.assert_close(student.decoder.class_prototypes[-1].prototype, new_snapshot, rtol=0, atol=0)
                new_group = student.decoder.class_prototypes[-1].prototype
                self.assertNotEqual(new_group.data_ptr(), student.decoder.class_prototypes[0].prototype.data_ptr())
                with torch.no_grad():
                    new_group.add_(1)
                for index, snapshot in enumerate(old_snapshots):
                    torch.testing.assert_close(student.decoder.class_prototypes[index].prototype, snapshot, rtol=0, atol=0)
                with torch.no_grad():
                    student.decoder.class_prototypes[0].prototype.add_(2)
                torch.testing.assert_close(teacher.decoder.class_prototypes[0].prototype, old_snapshots[0], rtol=0, atol=0)
                torch.testing.assert_close(previous.decoder.class_prototypes[0].prototype, old_snapshots[0], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
