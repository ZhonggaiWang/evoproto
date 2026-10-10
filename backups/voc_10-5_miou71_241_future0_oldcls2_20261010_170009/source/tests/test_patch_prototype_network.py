"""Check the ViT-to-prototype path and the actual Trainer loss wiring on CPU."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from model.backbone.vit import VisionTransformer
from model.model_seg_neg import network
import model.model_seg_neg as model_module
from model.losses import get_seg_loss, cpa_loss, rcpl_loss


ROOT = Path(__file__).resolve().parents[1]


class PatchPrototypeNetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(4)
            # Use the actual ViT implementation with a small CPU configuration.
            encoder = VisionTransformer(img_size=32, patch_size=16, embed_dim=8,
                                        depth=2, num_heads=2, aux_layer=-2)
            with patch.object(model_module.encoder, 'vit_base_patch16_224', return_value=encoder):
                cls.model = network('vit_base_patch16_224', num_classes=5,
                                    classes_list=[3, 2], pretrained=False,
                                    init_momentum=0.9, aux_layer=-2)
            cls.inputs = torch.randn(2, 3, 32, 48)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        self.model.zero_grad(set_to_none=True)

    def test_all_forward_interfaces_use_patch_prototype_predictions(self):
        net, inputs = self.model, self.inputs
        with torch.no_grad():
            _, tokens, _ = net.encoder.forward_features(inputs)
            prototypes = F.normalize(torch.cat(net.decoder.class_prototypes(), dim=0), dim=1)
            expected = (F.normalize(tokens, dim=-1) @ prototypes.T).transpose(1, 2).reshape(2, 5, 2, 3) / 0.01
            for options, count, seg_index in (
                ({}, 4, 1), ({'cal_sim': True}, 4, 1),
                ({'step0': True}, 5, 2), ({'cam_grad': True}, 7, 2),
                ({'crops': [inputs]}, 6, 1),
            ):
                with self.subTest(options=list(options)):
                    outputs = net(inputs, **options)
                    self.assertEqual(len(outputs), count)
                    torch.testing.assert_close(outputs[seg_index], expected)
                    if 'cam_grad' in options or 'crops' in options:
                        torch.testing.assert_close(outputs[-2] / 0.01, expected)
        self.assertFalse(any(isinstance(module, torch.nn.Conv2d) for module in net.decoder.modules()))
        self.assertFalse(any(key.startswith(('decoder.conv6.', 'decoder.conv7.', 'decoder.conv8.'))
                             for key in net.state_dict()))

    def test_pixel_supervision_reaches_vit_and_both_prototype_groups(self):
        segs = self.model(self.inputs)[1]
        labels = torch.tensor([[[0, 1, 2], [3, 4, 255]], [[4, 3, 2], [1, 0, 255]]])
        get_seg_loss(segs, labels, class_weight=torch.ones(5)).backward()
        parameters = [self.model.encoder.patch_embed.proj.weight,
                      self.model.encoder.blocks[-1].attn.qkv.weight]
        parameters += [group.prototype for group in self.model.decoder.class_prototypes]
        for parameter in parameters:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_cam_does_not_depend_on_or_evaluate_segmentation_head(self):
        with torch.no_grad(), patch.object(self.model.decoder, 'forward', side_effect=AssertionError('head called')):
            aux_cam, cam = self.model(self.inputs, cam_only=True)
            _, tokens, aux = self.model.encoder.forward_features(self.inputs)
            feature = self.model.to_2D(tokens, 2, 3)
            aux_feature = self.model.to_2D(aux, 2, 3)
            torch.testing.assert_close(cam, F.conv2d(feature, self.model.classifier.get_weight()))
            torch.testing.assert_close(aux_cam, F.conv2d(aux_feature, self.model.aux_classifier.get_weight()))

    def test_paper_losses_use_raw_prototypes_without_changing_pixel_cosine(self):
        outputs = self.model(self.inputs, step0=True)
        raw_prototypes = torch.cat(self.model.decoder.class_prototypes(), dim=0)
        torch.testing.assert_close(outputs[-1], raw_prototypes)
        teacher = (raw_prototypes[:3].detach() * 1.1).requires_grad_()
        weights = torch.zeros(5, 5, requires_grad=True)
        labels = torch.tensor([[[0, 1, 2], [3, 4, 255]], [[4, 3, 2], [1, 0, 255]]])
        loss = get_seg_loss(outputs[2], labels, class_weight=torch.ones(5))
        loss = loss + .1 * (rcpl_loss(outputs[-1], weights) + cpa_loss(outputs[-1], teacher, weights))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(self.model.encoder.patch_embed.proj.weight.grad.abs().sum().item(), 0)
        for group in self.model.decoder.class_prototypes:
            self.assertGreater(group.prototype.grad.abs().sum().item(), 0)
        self.assertIsNone(teacher.grad)
        self.assertIsNone(weights.grad)

    def test_optimizer_group_contains_all_prototypes_once(self):
        groups = self.model.get_param_groups()
        prototypes = {id(parameter) for parameter in self.model.decoder.parameters()}
        self.assertEqual({id(parameter) for parameter in groups[3]}, prototypes)
        all_ids = [id(parameter) for group in groups for parameter in group]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertTrue(all(all_ids.count(identity) == 1 for identity in prototypes))

    def test_both_incremental_warmup_losses_keep_zero_prototype_gradients_in_graph(self):
        for filename in ('Trainer.py', 'Trainer_coco.py'):
            self.model.zero_grad(set_to_none=True)
            tree = ast.parse((ROOT / 'continual' / filename).read_text())
            gate = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                        and ast.unparse(node.test) == 'n_iter < args.loss_warmup_iters')
            expression = next(node.value for statement in gate.body for node in ast.walk(statement)
                              if isinstance(node, ast.Assign)
                              and any(isinstance(target, ast.Name) and target.id == 'loss' for target in node.targets))
            outputs = self.model(self.inputs, step0=True)
            teacher = (outputs[-1][:3].detach() * 1.1).requires_grad_()
            sep = rcpl_loss(outputs[-1], torch.zeros(5, 5))
            kd = cpa_loss(outputs[-1], teacher)
            labels = torch.tensor([[[0, 1, 2], [3, 4, 255]], [[4, 3, 2], [1, 0, 255]]])
            seg = get_seg_loss(outputs[2], labels, torch.ones(5))
            namespace = {name: outputs[0].sum() * 0 for name in ('cls_loss', 'cls_loss_aux', 'ptc_loss')}
            namespace.update(prototype_sep=sep, prototype_kd=kd, contrastive_loss=sep + kd,
                             seg_loss=seg, args=SimpleNamespace(w_proto_sep=.1, w_ptc=.2))
            loss = eval(compile(ast.Expression(expression), filename, 'eval'), namespace)
            parameters = [group.prototype for group in self.model.decoder.class_prototypes]
            for gradient in torch.autograd.grad(loss, (seg, kd, sep), retain_graph=True):
                torch.testing.assert_close(gradient, torch.zeros_like(gradient))
            loss.backward()
            for parameter in parameters:
                self.assertIsNotNone(parameter.grad)
                torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter))
            self.assertIsNone(teacher.grad)

    def test_both_trainers_apply_pixel_loss_once_and_disable_it_during_warmup(self):
        for filename in ('Trainer.py', 'Trainer_coco.py'):
            tree = ast.parse((ROOT / 'continual' / filename).read_text())
            train = next(node for node in ast.walk(tree)
                         if isinstance(node, ast.FunctionDef) and node.name == 'train')
            losses = [node.value for node in ast.walk(train) if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == 'loss' for target in node.targets)]
            self.assertEqual(len(losses), 3)
            for expression in losses:
                with self.subTest(filename=filename, loss=ast.unparse(expression)):
                    namespace = {name: torch.tensor(1., requires_grad=True) for name in
                                 ('cls_loss', 'cls_loss_aux', 'ptc_loss', 'seg_loss',
                                  'prototype_kd', 'prototype_sep', 'contrastive_loss')}
                    namespace['args'] = SimpleNamespace(w_seg=0.17, w_ptc=0.2,
                                                        w_proto_kd=0.1, w_proto_sep=0.1)
                    loss = eval(compile(ast.Expression(expression), filename, 'eval'), namespace)
                    loss.backward()
                    active = any(isinstance(node, ast.Attribute) and node.attr == 'w_seg'
                                 for node in ast.walk(expression))
                    expected = 0.17 if active else 0.
                    self.assertAlmostEqual(namespace['seg_loss'].grad.item(), expected, places=6)


if __name__ == '__main__':
    unittest.main()
