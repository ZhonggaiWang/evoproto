"""Record initialization and confusion diagnostics without changing training."""
import json
import logging
import math
from pathlib import Path
import runpy

import torch
import torch.distributed as dist
from continual.Trainer import Trainer

original_get_weight = Trainer.get_weight


def record_confusion(self, con_matrix, total_classes, new_classes):
    counterparts, weights = original_get_weight(self, con_matrix, total_classes, new_classes)
    if not dist.is_initialized() or dist.get_rank() == 0:
        completed = self.current_iteration - 1
        destination = Path(self.args.work_dir) / 'confusion'
        destination.mkdir(exist_ok=True)
        selected = weights.sum(dim=1).detach().cpu()
        old_count = total_classes - new_classes + 1
        retention = 1 - selected[:old_count]
        if self.args.proto_kd_mode == 'direction' and self.args.cpa_skip_background:
            retention[0] = 0
        state = {
            'completed_iterations': completed,
            'counts': self.confusion_counts.detach().cpu(),
            'symmetric_confusion': con_matrix.detach().cpu(),
            'counterparts': counterparts.detach().cpu(),
            'weights': weights.detach().cpu(),
            'old_class_count': old_count,
            'temperature': self.args.confusion_temperature,
            'old_cls_threshold': self.args.old_cls_threshold,
            'ald_mode': self.args.ald_mode,
            'ald_threshold': self.args.ald_threshold,
        }
        temporary = destination / f'after_{completed}.pt.tmp'
        torch.save(state, temporary)
        temporary.replace(destination / f'after_{completed}.pt')
        summary = {
            'completed_iterations': completed,
            'temperature': self.args.confusion_temperature,
            'old_cls_threshold': self.args.old_cls_threshold,
            'reference_pixels_per_class': state['counts'].sum(dim=1).tolist(),
            'counterparts': state['counterparts'].tolist(),
            'selected_weights': selected.tolist(),
            'old_cpa_retention': retention.tolist(),
            'proto_kd_mode': self.args.proto_kd_mode,
            'cpa_skip_background': self.args.cpa_skip_background,
            'classes_above_099': int((selected > .99).sum()),
        }
        (destination / f'after_{completed}.json').write_text(json.dumps(summary, indent=2) + '\n')
        logging.info('Confusion values saved after %d updates; K=%g; classes with w>0.99=%d/%d',
                     completed, self.args.confusion_temperature,
                     summary['classes_above_099'], selected.numel())
    return counterparts, weights


original_train = Trainer.train


def audited_train(self, args):
    scale = self.model.decoder.logit_scale
    scale_group = next(group for group in self.optimizer.param_groups
                       if any(parameter is scale for parameter in group['params']))
    teacher = self.model_old
    audit = {
        'step': self.step,
        'old_cls_threshold': args.old_cls_threshold,
        'future_class_label': getattr(args, 'future_class_label', 255),
        'new_prototype_norms': self.model.decoder.class_prototypes[-1].prototype.detach().norm(dim=1).tolist(),
        'effective_scale': float(scale.detach().exp()),
        'scale_in_optimizer': True,
        'scale_weight_decay': scale_group['weight_decay'],
        'teacher_frozen': teacher is None or all(not p.requires_grad for p in teacher.parameters()),
        'teacher_scale_matches_student': teacher is None or math.isclose(
            float(teacher.decoder.logit_scale.detach()), float(scale.detach()), abs_tol=1e-7),
        'layer_decay': args.layer_decay if self.step > 0 else 1.0,
        'proto_kd_mode': args.proto_kd_mode, 'cpa_skip_background': args.cpa_skip_background,
        'classification_prototype_sharing': False,
        'optimizer_groups': [{'name': group.get('name'), 'lr': group['lr'],
                              'weight_decay': group['weight_decay'],
                              'parameter_count': sum(p.numel() for p in group['params'])}
                             for group in self.optimizer.param_groups],
    }
    (Path(args.work_dir) / 'initialization_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    logging.info('Prototype initialization: scale=%g, new norms=%s, CPA=%s, skip background=%s',
                 audit['effective_scale'], audit['new_prototype_norms'], args.proto_kd_mode, args.cpa_skip_background)
    return original_train(self, args)



# Audit one training and one confusion batch without changing their targets.
import importlib
supervision_module = importlib.import_module('continual.prototype_supervision')
trainer_module = importlib.import_module('continual.Trainer')
original_build_supervision = supervision_module.build_incremental_supervision
presence_audit_seen = set()


def audited_supervision(model, model_old, inputs, cls_labels, img_box, par,
                        args, total_classes, new_classes):
    phase = 'training' if model.training else 'confusion'
    if phase in presence_audit_seen:
        return original_build_supervision(model, model_old, inputs, cls_labels, img_box,
                                          par, args, total_classes, new_classes)
    captured = []
    hook = model_old.register_forward_hook(lambda module, values, output: captured.append(output[0].detach()))
    try:
        targets = original_build_supervision(model, model_old, inputs, cls_labels, img_box,
                                             par, args, total_classes, new_classes)
    finally:
        hook.remove()
    assert len(captured) == 1
    raw = captured[0]
    expected_old = (raw > args.old_cls_threshold).long()
    expected_new = cls_labels[:, :total_classes][:, -new_classes:]
    expected = torch.cat((expected_old, expected_new), dim=1)
    torch.testing.assert_close(targets.cls_labels_before_ald, expected)
    torch.testing.assert_close(targets.old_labels, targets.old_logits.argmax(dim=1))
    audit = dict(phase=phase, old_cls_threshold=args.old_cls_threshold,
                 old_positive_at_zero=int((raw > 0).sum()),
                 old_positive_at_selected_threshold=int(expected_old.sum()),
                 new_gt_positive_count=int(expected_new.sum()),
                 candidate_labels_match=True, teacher_pixel_argmax_preserved=True)
    (Path(args.work_dir) / f'teacher_presence_audit_{phase}.json').write_text(json.dumps(audit, indent=2) + '\n')
    logging.info('Teacher presence audit %s: old_cls > %g; old candidates %d -> %d; new GT tags retained.',
                 phase, args.old_cls_threshold, audit['old_positive_at_zero'],
                 audit['old_positive_at_selected_threshold'])
    presence_audit_seen.add(phase)
    return targets


supervision_module.build_incremental_supervision = audited_supervision
trainer_module.build_incremental_supervision = audited_supervision

Trainer.train = audited_train
Trainer.get_weight = record_confusion
runpy.run_path(str(Path(__file__).resolve().parent / 'scripts/dist_train_voc_seg_neg.py'),
               run_name='__main__')
