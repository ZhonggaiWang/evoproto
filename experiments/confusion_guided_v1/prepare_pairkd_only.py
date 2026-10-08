"""Create isolated D source; leave all completed A/B/C sources untouched."""
from pathlib import Path
import ast
import json
import os
import shutil
import sys


ROOT = Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto')
EXPERIMENT = ROOT/'experiments/confusion_guided_v1'
SOURCE = EXPERIMENT/'b_pairkd/src'
TARGET = EXPERIMENT/'d_pairkd_only/src'
REFERENCE = ROOT/'runs/kd_parallel_v1/formal/b_relational'


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f'Expected one source anchor, found {text.count(before)}: {before!r}')
    return text.replace(before, after)


def patch_selector(text):
    text = replace_once(text,
        'min_updates=100, ramp_updates=200, max_stale_updates=200):',
        'min_updates=100, ramp_updates=200, max_stale_updates=200,\n'
        '                 distillation_class_limit=None):')
    text = replace_once(text, '        self.min_rate = float(min_rate)',
        '        self.min_rate = float(min_rate)\n'
        '        if distillation_class_limit is not None and (\n'
        '                isinstance(distillation_class_limit, bool)\n'
        '                or int(distillation_class_limit) != distillation_class_limit\n'
        '                or not 2 <= distillation_class_limit <= self.classes):\n'
        "            raise ValueError('Distillation class limit must include foreground and stay within classes')\n"
        '        self.distillation_class_limit = (None if distillation_class_limit is None\n'
        '                                         else int(distillation_class_limit))')
    text = replace_once(text,
        "        return {'schema': self.SCHEMA, 'classes': self.classes, 'stage': self.stage,",
        "        state = {'schema': self.SCHEMA, 'classes': self.classes, 'stage': self.stage,")
    text = replace_once(text,
        "                'class_ids': list(range(self.classes))}\n\n    def set_extra_state",
        "                'class_ids': list(range(self.classes))}\n"
        '        if self.distillation_class_limit is not None:\n'
        "            state['schema'] = 2\n"
        "            state['distillation_class_limit'] = self.distillation_class_limit\n"
        '        return state\n\n    def set_extra_state')
    text = replace_once(text, '        supported_rows[0] = False',
        '        supported_rows[0] = False\n'
        '        if self.distillation_class_limit is not None:\n'
        '            supported_rows[self.distillation_class_limit:] = False')
    text = replace_once(text, '        eligible[:, 0] = False',
        '        eligible[:, 0] = False\n'
        '        if self.distillation_class_limit is not None:\n'
        '            eligible[:, self.distillation_class_limit:] = False')
    return text


def main():
    os.chdir(ROOT)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(SOURCE))
    from kd_runtime import safe_path, atomic_json, now, digest
    source, target = safe_path(SOURCE), safe_path(TARGET)
    protocol_path = safe_path(EXPERIMENT/'pairkd_only_protocol.json')
    if target.exists() or protocol_path.exists():
        raise RuntimeError('Refuse to replace existing D source/protocol')
    base_study = json.loads((ROOT/'runs/confusion_guided_v1/formal/b_pairkd/study.json').read_text())
    original_hashes = {str(path.relative_to(source)): digest(path)
                       for path in source.rglob('*.py') if '__pycache__' not in path.parts}
    if original_hashes != base_study['source_sha256']:
        raise RuntimeError('Completed B source differs from its frozen manifest')
    legacy_selector = safe_path(EXPERIMENT/'selector_legacy_reference.py')
    if legacy_selector.exists():
        if digest(legacy_selector) != digest(source/'model/directed_pair_selector.py'):
            raise RuntimeError('Existing legacy selector reference differs from completed B')
    else:
        shutil.copyfile(source/'model/directed_pair_selector.py', legacy_selector)
    shutil.copytree(source, target)
    for path in target.rglob('*'):
        safe_path(path)
    selector = target/'model/directed_pair_selector.py'
    selector_text = patch_selector(selector.read_text())
    ast.parse(selector_text)
    selector.write_text(selector_text)
    trainer = target/'continual/Trainer.py'
    trainer_text = replace_once(trainer.read_text(),
        '            max_stale_updates=args.pair_max_stale_updates).to(self.device)',
        '            max_stale_updates=args.pair_max_stale_updates,\n'
        '            distillation_class_limit=self.old_classes+1).to(self.device)')
    ast.parse(trainer_text)
    trainer.write_text(trainer_text)
    copied_hashes = {str(path.relative_to(target)): digest(path)
                     for path in target.rglob('*.py') if '__pycache__' not in path.parts}
    changed = sorted(key for key in original_hashes if copied_hashes[key] != original_hashes[key])
    if changed != ['continual/Trainer.py', 'model/directed_pair_selector.py']:
        raise RuntimeError(f'Unexpected D source changes: {changed}')
    for path in target.rglob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    atomic_json(protocol_path, {
        'utc': now(), 'arm': 'd_pairkd_only', 'source': str(target),
        'source_parent': str(source), 'source_sha256': copied_hashes,
        'selector_legacy_reference': str(legacy_selector),
        'only_source_changes_from_B': changed,
        'hypothesis': 'Focus supported confusion-guided distillation on teacher-known foreground pairs, without directed SEP suppressing current classes.',
        'training': 'Only stage2 remaining6000iterations; reuse optimizedKD b_relational stage1teacher andstage2iter2000fullstudent/optimizer. No step0/baseline/stage1/warmup retraining.',
        'inherited_stage1': str(REFERENCE/'10-5/step1'),
        'teacher': str(REFERENCE/'10-5/step1/checkpoints/model_final.pth'),
        'shared_stage2_warmup': str(REFERENCE/'10-5/step2/checkpoints/model_iter_2000.pth'),
        'observer_initialization': 'fresh at stage2 iteration2000;6000updates',
        'selector_domain': 'D schema2, distillation_class_limit16; row/competitor IDs1..15 only. Rates retain full21-output row mass including background, self andnew classes.',
        'domain_rationale': 'Teacher cannot distill new output channels; selecting only its foreground domain prevents a new-class top competitor hiding an otherwise supported old-old pair. Default limitNone preserves original schema1 exactly.',
        'loss': 'w_pair_sep0; w_proto_sep0; w_proto_kd0; retained conditional oldKL and at most50% per-pixel teacher binarypairKL substitution within w_pixel_kd0.1,T2; completefallback outside reliable pairs.',
        'selection': 'Past broad EMA, minrow exposures8/minpair3/minrate1%, minupdates100/ramp200, refresh50/maxstale200; same previous evidence policy andthresholds; noGT.',
        'resources': '8cardonly;8GPUs*single-cardbatch1=globalbatch8, async4shardevaluation.',
        'comparison_limit': 'Fixedseed0, resumed sampler/augmentation under8GPUs instead of4; globalbatch retained but not bitwisecontinuation.',
        'smoke_limit': 'Fourstep smoke restores model/optimizer but fresh observer doesnot reach100updates; pair activation requires separate actual-gradient/coverage checks.'})
    print(json.dumps({'source': str(target), 'protocol': str(protocol_path),
                      'modified_files': changed, 'selector_schema': 2, 'teacher_class_limit': 16}))


if __name__ == '__main__':
    main()
