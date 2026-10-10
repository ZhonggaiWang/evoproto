"""Check the transferred prototype geometry, temperature and ViT learning rates."""
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from continual.Trainer import Trainer as VOCTrainer
from continual.Trainer_coco import Trainer as COCOTrainer
from model.checkpoint import load_incremental_checkpoint
from model.decoder import PrototypeHead
from model.losses import cpa_loss, get_seg_loss, rcpl_loss


class PrototypeGeometryTests(unittest.TestCase):
    def test_all_new_class_groups_start_on_unit_sphere(self):
        head = PrototypeHead(768, [11, 5, 5])
        for group in head.class_prototypes:
            torch.testing.assert_close(group.prototype.norm(dim=1), torch.ones(group.prototype.shape[0]))

    def test_initial_effective_scale_is_100_and_receives_pixel_ce_gradients(self):
        head = PrototypeHead(2, [2])
        with torch.no_grad():
            head.class_prototypes[0].prototype.copy_(torch.eye(2))
        logits, cosine, _ = head(torch.tensor([[[[0.8]], [[0.6]]]]))
        torch.testing.assert_close(logits, cosine * 100)
        get_seg_loss(logits, torch.ones((1, 1, 1), dtype=torch.long)).backward()
        self.assertGreater(head.logit_scale.grad.abs().item(), 0)
        self.assertTrue(torch.isfinite(head.logit_scale.grad))

    def test_legacy_temperature_checkpoint_preserves_old_predictions(self):
        previous, teacher, student = nn.Module(), nn.Module(), nn.Module()
        previous.decoder = PrototypeHead(4, [3], temperature=.25)
        teacher.decoder = PrototypeHead(4, [3])
        student.decoder = PrototypeHead(4, [3, 2])
        state = previous.state_dict()
        state.pop('decoder.logit_scale')
        state['decoder._temperature'] = torch.tensor(.25)
        features = torch.randn(1, 4, 2, 2)
        expected = previous.decoder(features)[0]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'legacy.pth'
            torch.save({'model_state': state}, path)
            load_incremental_checkpoint(student, teacher, path, 1)
        torch.testing.assert_close(teacher.decoder(features)[0], expected)
        torch.testing.assert_close(student.decoder(features)[0][:, :3], expected)

    def test_direction_cpa_ignores_lengths_background_and_new_classes(self):
        teacher = torch.tensor([[1., 0.], [0., 1.], [1., 1.]], requires_grad=True)
        student = torch.tensor([[-3., 4.], [0., 8.], [5., 5.], [7., -2.]], requires_grad=True)
        loss = cpa_loss(student, teacher)
        self.assertAlmostEqual(loss.item(), 0, places=6)
        loss.backward()
        self.assertIsNone(teacher.grad)
        torch.testing.assert_close(student.grad[[0, 3]], torch.zeros(2, 2))
        self.assertGreater(cpa_loss(student, teacher, mode='paper').item(), 0)

    def test_direction_cpa_retains_confusion_relaxation_without_geometry_term(self):
        teacher = torch.tensor([[1., 0.], [1., 0.], [0., 1.]], requires_grad=True)
        student = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [1., -1.]], requires_grad=True)
        weights = torch.zeros(4, 4, requires_grad=True)
        with torch.no_grad():
            weights[1, 3] = .75
        expected = (.25 * 2 + 2 * (1 - 1 / math.sqrt(2))) / 2
        loss = cpa_loss(student, teacher, weights)
        self.assertAlmostEqual(loss.item(), expected, places=6)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertIsNone(weights.grad)
        self.assertGreater(student.grad[1:3].abs().sum().item(), 0)
        torch.testing.assert_close(student.grad[[0, 3]], torch.zeros(2, 2))

    def test_background_only_teacher_has_differentiable_zero_cpa(self):
        student = torch.randn(3, 4, requires_grad=True)
        teacher = torch.randn(1, 4)
        loss = cpa_loss(student, teacher)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        torch.testing.assert_close(student.grad, torch.zeros_like(student))

    def test_sep_still_separates_background_from_foreground(self):
        prototypes = torch.tensor([[1., 0.], [1., 1.]], requires_grad=True)
        loss = rcpl_loss(prototypes, mode='hinge')
        self.assertGreater(loss.item(), 0)
        loss.backward()
        self.assertGreater(prototypes.grad[0].abs().sum().item(), 0)
        self.assertGreater(prototypes.grad[1].abs().sum().item(), 0)


class LayerLearningRateTests(unittest.TestCase):
    def make_trainer(self, trainer_class, step):
        model = nn.Module()
        model.encoder = nn.Module()
        model.encoder.patch_embed = nn.Linear(4, 4)
        model.encoder.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])
        model.encoder.norm = nn.LayerNorm(4)
        model.classifier = nn.Linear(4, 2)
        model.decoder = PrototypeHead(4, [3])
        groups = [list(model.encoder.parameters()), [], list(model.classifier.parameters()),
                  list(model.decoder.parameters())]
        args = SimpleNamespace(optimizer='PolyWarmupAdamW', lr=2e-5, warmup_lr=1e-6,
                               wt_decay=.01, betas=(.9, .999), warmup_iters=2000,
                               max_iters=8000, power=.9, layer_decay=.9)
        optim = trainer_class.get_optimizer(SimpleNamespace(model=model, step=step), args, groups)
        return model, optim, args

    def test_incremental_layer_decay_and_complete_unique_parameter_registration(self):
        for trainer in (VOCTrainer, COCOTrainer):
            model, optim, args = self.make_trainer(trainer, 1)
            actual = {id(p): g['lr'] for g in optim.param_groups for p in g['params']}
            flat = [id(p) for g in optim.param_groups for p in g['params']]
            self.assertEqual(len(flat), len(set(flat)))
            self.assertEqual(set(flat), {id(p) for p in model.parameters()})
            self.assertAlmostEqual(actual[id(model.encoder.patch_embed.weight)], args.lr * .9**3)
            self.assertAlmostEqual(actual[id(model.encoder.blocks[-1].weight)], args.lr * .9)
            self.assertAlmostEqual(actual[id(model.encoder.norm.weight)], args.lr)
            self.assertAlmostEqual(actual[id(model.decoder.logit_scale)], args.lr * 10)
            optim.global_step = 2000
            optim.step()
            after = {id(p): g['lr'] for g in optim.param_groups for p in g['params']}
            self.assertEqual(actual, after)

    def test_base_step_keeps_uniform_backbone_lr_and_scale_has_no_weight_decay(self):
        for trainer in (VOCTrainer, COCOTrainer):
            model, optim, args = self.make_trainer(trainer, 0)
            groups = {id(p): g for g in optim.param_groups for p in g['params']}
            for parameter in model.encoder.parameters():
                self.assertEqual(groups[id(parameter)]['lr'], args.lr)
            self.assertEqual(groups[id(model.decoder.logit_scale)]['weight_decay'], 0)


if __name__ == '__main__':
    unittest.main()
