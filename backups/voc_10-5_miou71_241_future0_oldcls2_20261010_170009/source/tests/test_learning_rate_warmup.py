"""Check the actual Trainer schedules against absolute warmup learning rates."""
from types import SimpleNamespace
import unittest

import torch

from continual.Trainer import Trainer as VOCTrainer
from continual.Trainer_coco import Trainer as COCOTrainer
from utils.optimizer import warmup_argument


class LearningRateWarmupTests(unittest.TestCase):
    def make_optimizer(self, trainer, base_lr, optimizer_name='PolyWarmupAdamW'):
        args = SimpleNamespace(optimizer=optimizer_name, lr=base_lr, warmup_lr=1e-6,
                               wt_decay=.01, betas=(.9, .999), warmup_iters=2000,
                               max_iters=20000 if base_lr == 6e-5 else 8000, power=.9)
        groups = [[torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))] for _ in range(4)]
        return trainer.get_optimizer(SimpleNamespace(), args, groups), args

    def test_both_trainers_start_at_absolute_lr_and_keep_poly_group_multipliers(self):
        for trainer in (VOCTrainer, COCOTrainer):
            for base_lr in (6e-5, 2e-5):
                optim, args = self.make_optimizer(trainer, base_lr)
                for step in (0, 1000, 1999, 2000, args.max_iters - 1):
                    with self.subTest(trainer=trainer.__module__, base_lr=base_lr, step=step):
                        optim.global_step = step
                        optim.step()
                        expected = (args.warmup_lr + (base_lr - args.warmup_lr) * step / 1999
                                    if step < 2000 else base_lr * (1 - (step - 2000) / (args.max_iters - 2000)) ** .9)
                        for group, multiplier in zip(optim.param_groups, (1, 1, 10, 10)):
                            self.assertAlmostEqual(group['lr'], expected * multiplier, places=13)

    def test_incremental_loss_activation_starts_decay_at_peak_without_a_drop(self):
        for trainer in (VOCTrainer, COCOTrainer):
            optim, args = self.make_optimizer(trainer, 2e-5)
            rates = []
            for step in (1999, 2000, 2001):
                optim.global_step = step
                optim.step()
                rates.append([group['lr'] for group in optim.param_groups])
            for index, multiplier in enumerate((1, 1, 10, 10)):
                self.assertAlmostEqual(rates[0][index], args.lr * multiplier, places=13)
                self.assertAlmostEqual(rates[1][index], args.lr * multiplier, places=13)
                self.assertLess(rates[2][index], rates[1][index])
                self.assertLess(rates[2][index], rates[0][index])

    def test_no_warmup_starts_at_peak_and_final_update_approaches_zero(self):
        optim, args = self.make_optimizer(VOCTrainer, 2e-5)
        optim.warmup_iter = 0
        optim.step()
        self.assertAlmostEqual(optim.param_groups[0]['lr'], args.lr, places=13)
        optim.global_step = args.max_iters - 1
        optim.step()
        self.assertGreater(optim.param_groups[0]['lr'], 0)
        self.assertLess(optim.param_groups[0]['lr'], args.lr / 1000)

    def test_reference_no_warmup_schedule_and_first_adam_update_match_both_stages(self):
        for trainer in (VOCTrainer, COCOTrainer):
            for base_lr in (6e-5, 2e-5):
                optim, args = self.make_optimizer(trainer, base_lr)
                optim.warmup_iter = 0
                for group in optim.param_groups:
                    group['params'][0].grad = torch.ones(1, dtype=torch.float64)
                optim.step()
                for group, multiplier in zip(optim.param_groups, (1, 1, 10, 10)):
                    self.assertAlmostEqual(group['params'][0].item(), -base_lr * multiplier, places=10)
                for step in (1, 2000, 4000, args.max_iters - 1):
                    optim.global_step = step
                    optim.step()
                    expected = base_lr * (1 - step / args.max_iters) ** .9
                    for group, multiplier in zip(optim.param_groups, (1, 1, 10, 10)):
                        self.assertAlmostEqual(group['lr'], expected * multiplier, places=13)

    def test_first_real_adam_update_uses_correct_lr_for_every_group(self):
        optim, _ = self.make_optimizer(VOCTrainer, 6e-5)
        for group in optim.param_groups:
            group['params'][0].grad = torch.ones(1, dtype=torch.float64)
        optim.step()
        for group, expected in zip(optim.param_groups, (1e-6, 1e-6, 1e-5, 1e-5)):
            self.assertAlmostEqual(group['params'][0].item(), -expected, places=12)

    def test_cosine_legacy_absolute_start_is_not_accidentally_converted_to_ratio(self):
        for trainer in (VOCTrainer, COCOTrainer):
            optim, _ = self.make_optimizer(trainer, 6e-5, 'CosWarmupAdamW')
            optim.step()
            for group in optim.param_groups:
                self.assertAlmostEqual(group['lr'], 1e-6, places=13)

    def test_invalid_learning_rates_are_rejected(self):
        for base_lr, warmup_lr in ((0., 1e-6), (-1., 1e-6), (float('nan'), 1e-6),
                                   (6e-5, -1e-6), (6e-5, float('inf')), (6e-5, .1)):
            with self.assertRaises(ValueError):
                warmup_argument('PolyWarmupAdamW', base_lr, warmup_lr)


if __name__ == '__main__':
    unittest.main()
