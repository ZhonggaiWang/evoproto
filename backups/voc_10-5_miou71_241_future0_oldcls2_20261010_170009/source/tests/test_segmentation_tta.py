"""Check signed segmentation fusion and both Trainers' incremental validation."""
import importlib
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from model.backbone.vit import VisionTransformer
from model.model_seg_neg import network
import model.model_seg_neg as model_module
from utils.camutils import multi_scale_seg


class SegmentationTTATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_flip_alignment_preserves_signed_scores_and_batch_members(self):
        calls = []

        def model(inputs):
            calls.append((inputs.detach().clone(), torch.is_grad_enabled()))
            x = inputs[:, :1]
            position = torch.arange(x.shape[-1], dtype=x.dtype).view(1, 1, 1, -1)
            return None, torch.cat((x + position, -10 - x + 2 * position), dim=1), None, None

        inputs = torch.tensor([[-3., -2., 0., 1.], [2., 3., 5., 6.]])
        inputs = inputs[:, None, None, :].repeat(1, 1, 2, 1).requires_grad_()
        scores = multi_scale_seg(model, inputs, [1.])
        expected = torch.tensor([[[-1.5, -.5, 1.5, 2.5], [-4., -5., -7., -8.]],
                                 [[3.5, 4.5, 6.5, 7.5], [-9., -10., -12., -13.]]])
        expected = expected[:, :, None, :].repeat(1, 1, 2, 1)
        torch.testing.assert_close(scores, expected)
        self.assertEqual(len(calls), 1)
        torch.testing.assert_close(calls[0][0], torch.cat((inputs, inputs.flip(-1)), dim=0))
        self.assertFalse(calls[0][1])
        self.assertFalse(scores.requires_grad)

    def test_flip_mean_keeps_disagreeing_view_evidence(self):
        def model(inputs):
            # Max would pick class 0's isolated high score of 4, ignoring -3.
            logits = inputs.new_tensor([[4., 2.], [-3., 3.]]).view(2, 2, 1, 1)
            return None, logits, None, None

        scores = multi_scale_seg(model, torch.zeros(1, 3, 2, 4), [1.])
        expected = torch.tensor([.5, 2.5]).view(1, 2, 1, 1).expand(1, 2, 2, 4)
        torch.testing.assert_close(scores, expected)
        self.assertTrue((scores.argmax(1) == 1).all())

    def test_all_scales_change_prediction_before_final_argmax(self):
        shapes = []

        def model(inputs):
            shapes.append(tuple(inputs.shape))
            # Base view prefers class 1; the two other scales prefer class 0.
            values = [-3., -2.] if inputs.shape[-2] == 8 else [-1., -6.]
            logits = inputs.new_tensor(values).view(1, 2, 1, 1).expand(inputs.shape[0], 2, 2, 3)
            return None, logits, None, None

        inputs = torch.zeros(2, 3, 8, 12)
        single = multi_scale_seg(model, inputs, [1.])
        self.assertTrue((single.argmax(1) == 1).all())
        shapes.clear()
        scores = multi_scale_seg(model, inputs, [1., .5, 1.5])
        expected = (torch.tensor([-5., -14.]) / 3).view(1, 2, 1, 1).expand(2, 2, 8, 12)
        torch.testing.assert_close(scores, expected)
        self.assertTrue((scores.argmax(1) == 0).all())
        self.assertEqual(shapes, [(4, 3, 8, 12), (4, 3, 4, 6), (4, 3, 12, 18)])

    def test_real_vit_prototype_network_supports_all_tta_views(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(4)
            encoder = VisionTransformer(img_size=32, patch_size=16, embed_dim=8,
                                        depth=2, num_heads=2, aux_layer=-2)
            with patch.object(model_module.encoder, 'vit_base_patch16_224', return_value=encoder):
                model = network('vit_base_patch16_224', num_classes=5,
                                classes_list=[3, 2], pretrained=False,
                                init_momentum=.9, aux_layer=-2).eval()
            before = {name: value.clone() for name, value in model.state_dict().items()}
            scores = multi_scale_seg(model, torch.randn(2, 3, 32, 32), [1., .5, 1.5])
        self.assertEqual(tuple(scores.shape), (2, 5, 32, 32))
        self.assertTrue(torch.isfinite(scores).all())
        self.assertFalse(scores.requires_grad)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name])

    def test_both_trainers_use_fused_scores_only_for_incremental_validation(self):
        class FixtureModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = SimpleNamespace(embed_dim=8, temperature=.1)

            def forward(self, inputs):
                logits = inputs.new_zeros((inputs.shape[0], 3, 2, 2))
                logits[:, 0] = 1.  # Single view predicts background.
                return inputs.new_ones((inputs.shape[0], 2)), logits, None, None

        for module_name in ('continual.Trainer', 'continual.Trainer_coco'):
            module = importlib.import_module(module_name)
            for step in (0, 1):
                with self.subTest(trainer=module_name, step=step), tempfile.TemporaryDirectory(dir='/tmp') as work_dir:
                    model = FixtureModel()
                    args = SimpleNamespace(crop_size=8, cam_scales=[1., .5, 1.5], bkg_thre=.5,
                                           high_thre=.7, low_thre=.25, ignore_index=255,
                                           dataset='voc', task='10-5', w_proto_kd=.1, w_proto_sep=.1,
                                           proto_margin=0., ald=True, ald_mode='paper', seed=0,
                                           confusion_reweight=True, w_seg=.1, confusion_temperature=.01,
                                           confusion_interval=2000, local_rank=0, work_dir=work_dir)
                    trainer = SimpleNamespace(step=step, total_classes=2, current_iteration=1, model=model)
                    inputs = torch.zeros(1, 3, 4, 6)
                    labels = torch.ones(1, 3, 5, dtype=torch.long)
                    data = [(['sample'], inputs, labels, torch.ones(1, 2))]
                    cams = torch.zeros(1, 2, 8, 8)
                    cams[:, 0] = 1.
                    fused = torch.zeros(1, 3, 8, 8)
                    fused[:, 1] = 4.  # Fused views predict the correct foreground.
                    with patch.object(torch.Tensor, 'cuda', lambda tensor: tensor), \
                            patch.object(module, 'multi_scale_cam2', return_value=(cams, cams)), \
                            patch.object(module, 'multi_scale_seg', return_value=fused) as tta:
                        _, table = module.Trainer.validate(trainer, model, data, args)
                    row = next(line for line in table.splitlines() if 'mIoU' in line)
                    self.assertEqual(float(row.split('|')[-2].strip()), 100. if step else 0.)
                    self.assertEqual(tta.call_count, int(step > 0))
                    if step:
                        self.assertEqual(tuple(tta.call_args.args[1].shape), (1, 3, 8, 8))
                        self.assertEqual(tta.call_args.args[2], args.cam_scales)
                    self.assertTrue(model.training)


if __name__ == '__main__':
    unittest.main()
