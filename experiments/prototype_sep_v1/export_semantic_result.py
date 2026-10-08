"""Export completed A/B evidence, without training, inference or model loading.

Run on the 8-card host only after both formal audits and both analyses pass:
  python -B export_semantic_result.py
Read-only readiness check:
  python -B export_semantic_result.py --check-only
After downloading the ZIP and manifest, verify them on any local computer:
  python export_semantic_result.py --verify implementation.zip --manifest export_manifest.json

The original A exporter is preserved and is never imported or executed here.
Only five new final deliverables under the authorized study directory are
written. Existing deliverables are never overwritten. Checkpoints stay remote.
"""
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import math
import os
import stat
import sys
import zipfile


ALLOWED = Path('/ML-vePFS/infra_rd/kun/others/wzg')
ROOT = ALLOWED / 'workspace/evoproto'
E = ROOT / 'experiments/prototype_sep_v1'
U = ROOT / 'runs/prototype_sep_v1'
A = U / 'formal/a_geometry'
B = U / 'formal/b_semantic'
ASRC = E / 'a_geometry/src'
BSRC = E / 'b_semantic/src'
OPT = ROOT / 'runs/kd_parallel_v1/formal/b_relational'
OPTSRC = ROOT / 'experiments/kd_parallel_v1/b_relational/src'
OUTPUTS = ('prototype_sep_result.txt', 'implementation.zip', 'final_decision.json',
           'precision_recall_report.json', 'export_manifest.json')
CLASSES = ('_background_', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
           'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
METRICS = ('all_miou', 'previous_foreground_miou', 'current_foreground_miou')
STEP0_SHA = '4f298e14721630cf3c66ba4f6af04bd198ebeaf77adc6b7b07ab8403ad0893b5'
OPT_FINAL_SHA = '7c7cc782c72171f18269b638c41d1759d8d03dbe7f7990c5b7c268848ce6da9a'
WEIGHT_SUFFIXES = {'.pth', '.pt', '.ckpt', '.safetensors', '.pyc', '.pyo'}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def safe_path(path):
    """No escape, symlink, hardlink or nested mount within the write boundary."""
    path = Path(path)
    require(path.is_absolute() and path.resolve().is_relative_to(ALLOWED),
            'Path escapes the authorized area: ' + str(path))
    for item in (path, *path.parents):
        if item == ALLOWED.parent:
            break
        require(not item.is_symlink(), 'Symlink in authorized path: ' + str(item))
        if item.exists():
            info = item.stat()
            require(not stat.S_ISREG(info.st_mode) or info.st_nlink == 1,
                    'Hardlinked file: ' + str(item))
            require(info.st_dev == ALLOWED.stat().st_dev, 'Nested mount: ' + str(item))
    return path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def byte_digest(data):
    return hashlib.sha256(data).hexdigest()


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def read_json(path):
    path = safe_path(path)
    require(path.is_file(), 'Required completed evidence is missing: ' + str(path))
    def invalid_constant(value):
        raise ValueError('Non-finite JSON value: ' + value)
    return json.loads(path.read_text(encoding='utf-8'), parse_constant=invalid_constant)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')


def near(a, b):
    return (type(a) in (int, float) and type(b) in (int, float)
            and math.isfinite(a) and math.isfinite(b) and abs(a - b) <= 1e-6)


def hash_matches(path, expected, cache):
    path = safe_path(path)
    require(valid_sha(expected), 'Invalid expected SHA256 for ' + str(path))
    require(path.is_file(), 'Required file is missing: ' + str(path))
    key = str(path)
    actual = cache.setdefault(key, digest(path)) if key not in cache else cache[key]
    require(actual == expected, 'SHA256 disagreement: ' + key)
    return actual


def source_map(source):
    safe_path(source)
    require(source.is_dir(), 'Frozen source directory is missing: ' + str(source))
    return {p.relative_to(source).as_posix(): digest(safe_path(p)) for p in sorted(source.rglob('*.py'))
            if '__pycache__' not in p.parts}


def histogram(values, count):
    require(isinstance(values, list) and len(values) == count
            and all(isinstance(row, list) and len(row) == count for row in values),
            f'Expected complete {count}x{count} histogram')
    require(all(type(value) is int and value >= 0 for row in values for value in row),
            'Histogram counts must be nonnegative integers')
    rows = [sum(row) for row in values]
    cols = [sum(values[i][j] for i in range(count)) for j in range(count)]
    require(all(value > 0 for value in rows), 'Every evaluated GT class must have support')
    iou = [100 * values[i][i] / (rows[i] + cols[i] - values[i][i]) for i in range(count)]
    return rows, cols, iou


def checked_endpoint(path, step):
    raw = read_json(path)
    count, old = 11 + 5 * step, 5 + 5 * step
    require(raw.get('step') == step and raw.get('iteration') == 8000
            and raw.get('images') == {1: 1240, 2: 1449}[step],
            'Not a complete final evaluation: ' + str(path))
    rows, cols, iou = histogram(raw['histogram'], count)
    require(set(raw['class_iou']) == set(CLASSES[:count])
            and all(near(raw['class_iou'][name], iou[i]) for i, name in enumerate(CLASSES[:count])),
            'Class IoUs do not reproduce from the full histogram')
    metrics = {'all_miou': sum(iou) / count,
               'previous_foreground_miou': sum(iou[1:old + 1]) / old,
               'current_foreground_miou': sum(iou[old + 1:]) / 5}
    require(all(near(raw[key], metrics[key]) for key in METRICS), 'Grouped endpoint metrics disagree')
    require(valid_sha(raw.get('checkpoint_sha256')), 'Evaluated checkpoint SHA is missing')
    return {'source': str(path), 'source_sha256': digest(safe_path(path)), 'step': step,
            'iteration': 8000, 'images': raw['images'], 'metrics': metrics,
            'histogram': raw['histogram'], 'gt_class_pixels': rows, 'pred_class_pixels': cols,
            'class_iou': dict(zip(CLASSES[:count], iou)), 'checkpoint_sha256': raw['checkpoint_sha256']}


def same_analysis_endpoint(analysed, actual):
    require(all(analysed.get(key) == actual[key] for key in
                ('source', 'step', 'iteration', 'images', 'checkpoint_sha256', 'histogram', 'gt_class_pixels', 'pred_class_pixels'))
            and all(near(analysed['metrics'][key], actual['metrics'][key]) for key in METRICS),
            'Analysis does not describe the current completed endpoint: ' + actual['source'])


def require_run_complete(run, source, runner, arm, jobs, trained_steps, cache):
    audit = read_json(run / 'completion_audit.json')
    require(audit.get('status') == 'complete'
            and audit.get('all_training_and_endpoint_requirements_complete') is True
            and not audit.get('failed_checks') and not audit.get('pending')
            and bool(audit.get('checks')) and all(value is True for value in audit['checks'].values()),
            arm + ' final audit has not passed')
    if arm == 'b_semantic':
        require(audit.get('all_semantic_sep_requirements_complete') is True, 'B semantic audit has not passed')
    require(read_json(run / 'status.json').get('status') == 'complete', arm + ' is still running or failed')
    for name in ('failure.json', 'training_failed.json'):
        require(not safe_path(run / name).exists(), arm + ' has a failure receipt')
    require(not list((run / 'evaluations').glob('*/error_rank*.json')), arm + ' has an evaluation failure receipt')
    study = read_json(run / 'study.json')
    require(study.get('arm') == arm and study.get('smoke') is False, 'Expected a formal ' + arm + ' study')
    require(study.get('source_sha256') == source_map(source), arm + ' source has changed since its freeze')
    hash_matches(runner, study.get('runner_sha256'), cache)
    cfg = study['common_config']
    require(cfg.get('seed') == 0 and cfg.get('task') == '10-5' and study.get('global_batch') == 8
            and study.get('batch_per_gpu') == 1 and study.get('train_gpus') == list(range(8)),
            'Unexpected task, seed or GPU allocation')
    training = read_json(run / 'training_complete.json')
    require(training.get('status') == 'complete', 'Training completion receipt is not successful')
    require(training.get('stages') == 2 if arm == 'a_geometry' else
            training.get('stages_trained') == 1 and training.get('inherited_stage1') == str(A),
            'Unexpected newly trained stage count')
    require(read_json(run / 'evaluation_workers_complete.json').get('returncodes') == [0] * 4,
            arm + ' evaluation workers have not all exited successfully')
    for rank in range(4):
        require(read_json(run / f'evaluator_rank{rank}.process.json').get('returncode') == 0,
                'Evaluator process did not exit successfully')
    queued = sorted(p.stem for p in (run / 'eval_queue').glob('*.json'))
    require(queued == sorted(f'step{s}_iter{i}' for s, i in jobs), 'Unexpected formal evaluation queue')
    for step in trained_steps:
        stage = run / f'10-5/step{step}'
        require(read_json(stage / 'launch.json').get('returncode') == 0, 'Trainer process did not exit successfully')
        done = read_json(stage / 'training_complete.json')
        require(done.get('returncode') == 0, 'Stage training did not exit successfully')
        hash_matches(stage / 'checkpoints/model_final.pth', done.get('checkpoint_sha256'), cache)
    for step, iteration in jobs:
        folder = run / f'evaluations/step{step}_iter{iteration}'
        result = read_json(folder / 'result.json')
        job = read_json(run / f'eval_queue/step{step}_iter{iteration}.json')
        require(result.get('step') == step and result.get('iteration') == iteration
                and result.get('images') == {1: 1240, 2: 1449}[step]
                and result.get('checkpoint_sha256') == job.get('checkpoint_sha256'),
                'Queued evaluation is incomplete or has a different checkpoint identity')
        parts = [read_json(folder / f'rank{rank}.json') for rank in range(4)]
        names = [name for part in parts for name in part['images']]
        require(len(names) == len(set(names)) == result['images'], 'Validation coverage is incomplete or duplicated')
        require(all(part.get('rank') == rank and part.get('shards') == 4
                    and part.get('step') == step and part.get('iteration') == iteration
                    and part.get('checkpoint_sha256') == result['checkpoint_sha256']
                    for rank, part in enumerate(parts)), 'Evaluation shard identities disagree')
        count = 11 + 5 * step
        summed = [[sum(part['histogram'][i][j] for part in parts) for j in range(count)] for i in range(count)]
        require(summed == result['histogram'], 'Sharded histogram does not reproduce the merged result')
    return study, audit


def collect_context():
    """All gates and hashes complete before any output or temporary file exists."""
    cache = {}
    astudy, aaudit = require_run_complete(A, ASRC, E / 'run_prototype_sep.py', 'a_geometry',
        [(1, i) for i in (4000, 6000, 8000)] + [(2, i) for i in (2000, 4000, 6000, 8000)], (1, 2), cache)
    bstudy, baudit = require_run_complete(B, BSRC, E / 'run_semantic_sep.py', 'b_semantic',
        [(2, i) for i in (4000, 6000, 8000)], (2,), cache)
    analyses = {name: read_json(U / filename) for name, filename in
                (('A', 'result_analysis.json'), ('B', 'semantic_analysis.json'))}
    for name, study, run in (('A', astudy, A), ('B', bstudy, B)):
        report = analyses[name]
        require(report.get('status') == 'complete_endpoint_analysis'
                and report.get('completion', {}).get('complete') is True
                and report.get('run') == str(run) and report.get('study') == study,
                name + ' full endpoint analysis has not completed for the frozen study')
    endpoints = {'optimized_KD': checked_endpoint(OPT / 'evaluations/step2_iter8000/result.json', 2),
                 'A_geometry': checked_endpoint(A / 'evaluations/step2_iter8000/result.json', 2),
                 'B_semantic': checked_endpoint(B / 'evaluations/step2_iter8000/result.json', 2)}
    a1 = checked_endpoint(A / 'evaluations/step1_iter8000/result.json', 1)
    opt1 = checked_endpoint(OPT / 'evaluations/step1_iter8000/result.json', 1)
    same_analysis_endpoint(analyses['A']['stages']['2']['endpoint'], endpoints['A_geometry'])
    same_analysis_endpoint(analyses['A']['stages']['2']['references']['optimized_KD'], endpoints['optimized_KD'])
    same_analysis_endpoint(analyses['A']['stages']['1']['endpoint'], a1)
    same_analysis_endpoint(analyses['A']['stages']['1']['references']['optimized_KD'], opt1)
    same_analysis_endpoint(analyses['B']['candidate_endpoint'], endpoints['B_semantic'])
    same_analysis_endpoint(analyses['B']['references']['A_geometry'], endpoints['A_geometry'])
    same_analysis_endpoint(analyses['B']['references']['optimized_KD'], endpoints['optimized_KD'])
    same_analysis_endpoint(analyses['B']['inherited_stage1']['endpoint'], a1)
    require(all(ep['gt_class_pixels'] == endpoints['optimized_KD']['gt_class_pixels'] for ep in endpoints.values()),
            'Final comparisons do not use identical original-GT class totals')
    require(a1['gt_class_pixels'] == opt1['gt_class_pixels'], 'Stage1 GT totals disagree with original optimized KD')
    require(bstudy.get('inherited_stage1_run') == str(A) and bstudy.get('teacher') == str(A / '10-5/step1/checkpoints/model_final.pth')
            and bstudy.get('resume_checkpoint') == str(A / '10-5/step2/checkpoints/model_iter_2000.pth')
            and bstudy.get('resume_iteration') == 2000
            and set(bstudy.get('restored_state', [])) == {'student', 'optimizer', 'online_confusion', 'geometry_selector'},
            'B does not inherit the audited A teacher and all four stage2/2000 states')
    require(analyses['B']['inherited_stage1'].get('new_training') is False
            and analyses['B']['stage2_resume'].get('iteration') == 2000
            and analyses['B']['stage2_resume'].get('trained_remaining_iterations') == 6000,
            'B analysis misstates the inherited training budget')
    hash_matches(Path(bstudy['teacher']), bstudy['teacher_sha256'], cache)
    hash_matches(Path(bstudy['resume_checkpoint']), bstudy['resume_checkpoint_sha256'], cache)
    require(bstudy['teacher_sha256'] == read_json(A / '10-5/step1/training_complete.json')['checkpoint_sha256']
            and bstudy['resume_checkpoint_sha256'] == read_json(A / 'eval_queue/step2_iter2000.json')['checkpoint_sha256'],
            'Inherited checkpoint identities disagree with A publications')
    hash_matches(Path(astudy['step0']), STEP0_SHA, cache)
    require(astudy['step0_sha256'] == STEP0_SHA, 'Original shared step0 identity changed')
    hash_matches(Path(astudy['step1_warmup']), astudy['step1_warmup_sha256'], cache)
    hash_matches(OPT / '10-5/step2/checkpoints/model_final.pth', OPT_FINAL_SHA, cache)
    for name, run in (('optimized_KD', OPT), ('A_geometry', A), ('B_semantic', B)):
        path = run / '10-5/step2/checkpoints/model_iter_8000.pth'
        hash_matches(path, endpoints[name]['checkpoint_sha256'], cache)
    origin = read_json(E / 'origin.json')
    require(origin.get('optimized_kd_source') == str(OPTSRC)
            and origin.get('optimized_kd_source_sha256') == source_map(OPTSRC),
            'Original optimized KD source changed')
    kd_sha = digest(safe_path(OPTSRC / 'model/pixel_kd.py'))
    require(kd_sha == astudy['source_sha256']['model/pixel_kd.py'] == bstudy['source_sha256']['model/pixel_kd.py'],
            'Optimized pixel KD is not byte-identical across the three methods')
    integration = read_json(E / 'b_semantic/semantic_integration.json')
    changed = {key: value for key, value in bstudy['source_sha256'].items()
               if astudy['source_sha256'].get(key) != value}
    require(integration.get('schema') == 1 and integration.get('candidate') == str(BSRC)
            and integration.get('primary_frozen_source') == str(ASRC)
            and integration.get('primary_source_sha256') == astudy['source_sha256']
            and integration.get('changed_source_sha256') == changed
            and integration.get('optimized_kd_sha256_unchanged') == kd_sha,
            'Actual executed semantic integration receipt disagrees with frozen sources')
    hash_matches(E / 'patch_semantic_candidate.py', integration.get('patch_script_sha256'), cache)
    for filename, study in (('preflight.json', astudy), ('semantic_preflight.json', bstudy)):
        preflight = read_json(E / filename)
        require(preflight.get('passed') is True and preflight.get('source_sha256') == study['source_sha256'],
                'Preflight does not pass for the frozen final source')
    for name, ep in endpoints.items():
        ep['final_checkpoint'] = str(({'optimized_KD': OPT, 'A_geometry': A, 'B_semantic': B}[name])
                                     / '10-5/step2/checkpoints/model_final.pth')
        ep['final_checkpoint_sha256'] = cache[ep['final_checkpoint']]
    return {'utc': now(), 'studies': {'A_geometry': astudy, 'B_semantic': bstudy},
            'audits': {'A_geometry': aaudit, 'B_semantic': baudit}, 'analyses': analyses,
            'endpoints': endpoints, 'inherited_stage1': {'A_geometry': a1, 'optimized_KD': opt1},
            'critical_file_sha256': cache, 'integration': integration, 'original_kd_sha256': kd_sha,
            'optimized_KD_source_sha256': origin['optimized_kd_source_sha256']}


def class_precision_recall(hist, rows, cols, ids):
    ids = list(ids)
    tp = sum(hist[i][i] for i in ids)
    predicted, truth = sum(cols[i] for i in ids), sum(rows[i] for i in ids)
    return {'class_ids': ids, 'precision_percent': 100 * tp / predicted if predicted else None,
            'recall_percent': 100 * tp / truth if truth else None, 'true_positive': tp,
            'false_positive': predicted - tp, 'false_negative': truth - tp,
            'predicted_pixels': predicted, 'gt_pixels': truth}


def precision_report(context):
    methods = {}
    for name, ep in context['endpoints'].items():
        hist, rows, cols = ep['histogram'], ep['gt_class_pixels'], ep['pred_class_pixels']
        per_class = []
        for i, label in enumerate(CLASSES):
            row = class_precision_recall(hist, rows, cols, [i])
            row.update(class_id=i, class_name=label, iou_percent=ep['class_iou'][label])
            per_class.append(row)
        methods[name] = {'evaluation': ep['source'], 'evaluation_sha256': ep['source_sha256'],
                        'checkpoint_sha256': ep['checkpoint_sha256'], 'metrics': ep['metrics'],
                        'all_classes': class_precision_recall(hist, rows, cols, range(21)),
                        'all_foreground': class_precision_recall(hist, rows, cols, range(1, 21)),
                        'old15': class_precision_recall(hist, rows, cols, range(1, 16)),
                        'new5': class_precision_recall(hist, rows, cols, range(16, 21)),
                        'per_class': per_class}
    return {'schema': 1, 'utc': context['utc'], 'stage': 2, 'iteration': 8000, 'images': 1449,
            'methods': methods, 'GT_role': 'Existing full validation histograms only; diagnostic, never a training input',
            'definitions': {'histogram': 'Rows are original GT, columns are main-head predictions.',
                'group_precision_recall': 'Micro multiclass: sum correct pixels of included classes / sum predictions or GT pixels of those classes. A within-group wrong class is still an error.',
                'iou': 'Per class TP/(TP+FP+FN); group mIoU is the unweighted class mean, not micro IoU.',
                'zero_denominator': 'null; no invented zero or perfect score.'}}


def decision(context):
    eps = context['endpoints']
    wins = eps['B_semantic']['metrics']['all_miou'] > eps['optimized_KD']['metrics']['all_miou']
    selected = 'B_semantic' if wins else 'optimized_KD'
    endpoint = eps[selected]
    source = BSRC if wins else OPTSRC
    rows = [{'method': name, **eps[name]['metrics'],
             'delta_all_vs_optimized_KD_pp': eps[name]['metrics']['all_miou'] - eps['optimized_KD']['metrics']['all_miou'],
             'evaluated_checkpoint_sha256': eps[name]['checkpoint_sha256'],
             'final_checkpoint_sha256': eps[name]['final_checkpoint_sha256']}
            for name in ('optimized_KD', 'A_geometry', 'B_semantic')]
    return {'schema': 2, 'utc': context['utc'], 'status': 'complete_endpoint_decision',
            'candidate_exceeds_required_endpoint': wins,
            'method': 'semantic_protected_prototype_SEP_plus_optimized_KD' if wins else 'existing_optimized_KD',
            'selected_method_id': selected, 'selected_checkpoint': endpoint['final_checkpoint'],
            'selected_checkpoint_sha256': endpoint['final_checkpoint_sha256'],
            'selected_evaluated_checkpoint_sha256': endpoint['checkpoint_sha256'],
            'selected_source': str(source), 'selected_source_sha256': source_map(source),
            'criterion': 'Select B iff full stage2/8000 all-class mIoU strictly exceeds original optimized KD; otherwise retain optimized KD. A or intermediate recovery cannot satisfy this criterion.',
            'metrics_recomputed_from_full_GT_histograms': True, 'endpoint_table': rows,
            'final_and_evaluated_checkpoint_relation': 'File SHA may differ; completed CPU audits require model, optimizer, observer and selector tensors to be exactly equal for A/B final vs iteration8000.',
            'integrity_receipts': [str(A / 'completion_audit.json'), str(B / 'completion_audit.json')],
            'analysis_receipts': [str(U / 'result_analysis.json'), str(U / 'semantic_analysis.json')],
            'B_training_budget': {'inherited_stage1': str(A), 'new_stage1_training': False,
                'stage2_resume_iteration': 2000, 'new_stage2_updates': 6000, 'global_batch': 8,
                'restored_state': ['student', 'optimizer', 'online_confusion', 'geometry_selector'],
                'final_observer_updates': 8000, 'final_observer_seen_images': 64000},
            'step0_retrained': False, 'baseline_retrained': False, 'random_seed_search': False,
            'host': '8card', '4card': 'unused', 'all_remote_outputs_and_caches_within': str(ALLOWED),
            'evidence_scope': 'One fixed seed0 adaptive development validation endpoint; no stability, significance or independent generalization claim.',
            'goal_requirement_met_by_B': wins,
            'goal_status_change_permitted_by_this_script': False,
            'limitations': context['analyses']['B']['limitations']}


def text_report(context, result):
    eps = context['endpoints']
    a1 = context['inherited_stage1']['A_geometry']['metrics']
    o1 = context['inherited_stage1']['optimized_KD']['metrics']
    names = {'optimized_KD': '原优化 KD', 'A_geometry': 'A 混淆指导原型 SEP + 原优化 KD',
             'B_semantic': 'B 语义保护原型 SEP + 原优化 KD'}
    lines = ['混淆指导原型 SEP 与优化 KD：完整终点交付',
        '最终选择：' + names[result['selected_method_id']] + '。',
        '判据：只有 B 在阶段2/8000步的整体 mIoU 严格高于原优化 KD，才选择 B；否则保留原优化 KD。',
        'VOC 10-5，固定 seed=0，完整1449张验证图；整体含背景，旧15/新5只含前景。', '',
        '方案 | 整体 mIoU | 旧15 mIoU | 新5 mIoU']
    for name in ('optimized_KD', 'A_geometry', 'B_semantic'):
        m = eps[name]['metrics']
        lines.append(f"{names[name]} | {m['all_miou']:.6f} | {m[METRICS[1]]:.6f} | {m[METRICS[2]]:.6f}")
    for name in ('A_geometry', 'B_semantic'):
        delta = [eps[name]['metrics'][key] - eps['optimized_KD']['metrics'][key] for key in METRICS]
        lines.append(f"{name} 相对原优化 KD（百分点） | {delta[0]:+.6f} | {delta[1]:+.6f} | {delta[2]:+.6f}")
    lines += ['', '继承与预算：',
        f"A 阶段1终点整体 {a1['all_miou']:.6f}，原优化 KD {o1['all_miou']:.6f}，差值 {a1['all_miou']-o1['all_miou']:+.6f} 个百分点。",
        f"该阶段新5 mIoU 差值 {a1[METRICS[2]]-o1[METRICS[2]]:+.6f} 个百分点。B 继承 A 的这个教师，包括已有质量差距；B 没有另训阶段1。",
        'A 复用共享阶段1的2000步模型和优化器，新增6000步；A 阶段2训练8000步。',
        'B 复用 A 阶段2的2000步模型、优化器、在线混淆和选择器，新增6000步。最终 observer 是8000次更新、64000次图像曝光。',
        '共享 step0、原优化 KD 和历史 baseline 均未重训，没有种子搜索。8卡机8x1保持 global batch8；4卡机未使用。',
        '', '实现：',
        '在线定向混淆只挑选需要关注的方向：完整 broad EMA 行归一化排序，可信 CAM 支持和新鲜度筛选；混淆比例不作为精确学习权重。',
        '原型 SEP 将方向去重为涉及新类的几何对；旧新对只更新新原型，新新对更新双方，排除背景、旧旧和自配对。固定 lambda0.1、margin0，对全部有效对的平方余弦 hinge 取均值。',
        'B 先计算当前新原型各行的切空间 SEP/语义梯度，对反向分量作正交投影。语义梯度按真实 global valid-pixel 分母跨卡汇合；保护在 DDP 输出包装之前计算。',
        '返回 surrogate 保留几何损失的前向值，用已脱离计算图的修正梯度执行普通一次 backward。原语义损失仍只正常加入一次。',
        '原优化 pixel KD 在原方法、A、B 中字节一致。GT 只用于已有完整验证直方图和诊断，未参与在线选择、投影或学习强度。',
        '', '证据：',
        'A：25项原型行为测试、16项选择器测试、真实图像梯度检查、8卡两阶段烟测，以及7个正式全量评估任务。',
        'B：23项语义保护CPU测试、2进程Gloo与8卡NCCL数学/真实DDP测试、8卡真实网络4步烟测，以及3个正式全量评估任务。',
        '所有正式评估任务各4分片完整覆盖；训练和评估退出码均为0。A/B完整审计通过，final与被评估8000步模型、优化器、observer及selector状态逐张量一致。',
        '三种方法的原GT类别行总量相同，IoU由已有直方图重新计算。分类及旧15/新5微平均precision/recall见precision_recall_report.json。',
        '', '结果的解释范围：',
        '投影只保证记录到的当前切空间分量梯度条件，不保证 AdamW动量、权重衰减、归一化参数更新、后续梯度或最终IoU改善。',
        '固定seed0、同一开发验证集上的逐步改进不能证明多种子稳定性、统计显著性或独立测试泛化。',
        '候选8x1与历史原优化 KD 的4x2保持global batch8，但恢复时数据采样/增强流重启，差值无法严格归因于单一模块。',
        'CAM支持计数是重复训练曝光，不是独立样本量或GT正确率；采样梯度诊断也不是全部数据的保护保证。',
        '', f"保留实现：{result['selected_source']}", f"保留检查点：{result['selected_checkpoint']}",
        f"检查点 SHA256：{result['selected_checkpoint_sha256']}",
        '检查点留在远程授权目录；implementation.zip包含A/B完整源码及实现、测试、审计和诊断证据，不包含模型权重。',
        '下载后可用随包 exporter 的 --verify 与 export_manifest.json 验证每个ZIP条目的SHA256和整个ZIP。']
    if not result['candidate_exceeds_required_endpoint']:
        lines.insert(3, 'B 本次候选未满足最终提升目标；这份交付记录完整结果，并不将未达成的优化 goal 标为完成。')
    return ('\n'.join(lines) + '\n').encode('utf-8')


def archive_name(name):
    p = PurePosixPath(name)
    require(isinstance(name, str) and '\\' not in name and not p.is_absolute()
            and name == p.as_posix() and name not in ('', '.') and '..' not in p.parts
            and ':' not in p.parts[0], 'Unsafe ZIP entry name: ' + str(name))
    return name


def collect_entries(generated):
    entries = {}
    def add(path, name):
        path = safe_path(path)
        if not path.is_file() or '__pycache__' in path.parts or path.suffix.lower() in WEIGHT_SUFFIXES:
            return
        name = archive_name(name)
        require(name not in entries, 'Duplicate ZIP entry: ' + name)
        require(stat.S_ISREG(path.stat().st_mode), 'Nonregular evidence file: ' + str(path))
        entries[name] = {'path': path, 'sha256': digest(path), 'size': path.stat().st_size}
    for label, source in (('optimized_KD', OPTSRC), ('A_geometry', ASRC), ('B_semantic', BSRC)):
        for path in sorted(source.rglob('*')):
            add(path, label + '/src/' + path.relative_to(source).as_posix())
    # Actual remote experiment scripts, including the executed CLI patch, are
    # bundled as found on this host; no local development copy replaces them.
    for path in sorted(E.rglob('*')):
        if path.is_relative_to(ASRC) or path.is_relative_to(BSRC):
            continue
        if path.suffix.lower() in {'.py', '.json', '.log', '.jsonl'}:
            add(path, 'study/' + path.relative_to(E).as_posix())
    for label, run in (('a_geometry', A), ('b_semantic', B)):
        for path in sorted(run.rglob('*')):
            if path.suffix.lower() in {'.json', '.jsonl', '.log', '.txt'}:
                add(path, 'evidence/formal/' + label + '/' + path.relative_to(run).as_posix())
    # Include diagnostic/reference JSON recursively and both completed smoke
    # receipts. Exclude formal here (already above) and previous deliverables.
    for path in sorted(U.rglob('*')):
        if path.is_relative_to(U / 'formal') or path.name in OUTPUTS:
            continue
        if path.suffix.lower() == '.json' or (path.is_relative_to(U / 'smoke')
                                            and path.suffix.lower() in {'.jsonl', '.log'}):
            add(path, 'evidence/study/' + path.relative_to(U).as_posix())
    for step in (1, 2):
        add(OPT / f'evaluations/step{step}_iter8000/result.json',
            f'evidence/reference/optimized_KD_step{step}_iter8000.json')
        for name in ('config.json', 'launch.json', 'training_complete.json'):
            add(OPT / f'10-5/step{step}' / name,
                f'evidence/reference/optimized_KD_step{step}/' + name)
    for path, name in ((OPT / 'study.json', 'optimized_KD_study.json'),
                       (ROOT / 'runs/fixed_baseline_v1/source_snapshot.json', 'baseline_source_snapshot.json'),
                       (ROOT / 'experiments/kd_pixel_v2/origin.json', 'baseline_origin.json')):
        add(path, 'evidence/reference/' + name)
    for name, payload in generated.items():
        key = archive_name('results/' + name)
        require(key not in entries, 'Duplicate generated ZIP entry')
        entries[key] = {'data': payload, 'sha256': byte_digest(payload), 'size': len(payload)}
    required_entries = ('study/b_semantic/semantic_integration.json', 'study/patch_semantic_candidate.py',
        'study/preflight.json', 'study/semantic_preflight.json', 'study/semantic_test_receipts.json',
        'evidence/formal/a_geometry/completion_audit.json', 'evidence/formal/b_semantic/completion_audit.json',
        'evidence/study/result_analysis.json', 'evidence/study/semantic_analysis.json',
        'evidence/formal/b_semantic/10-5/step2/semantic_guard_metrics.jsonl')
    require(all(name in entries for name in required_entries), 'Bundle is missing required final evidence')
    return entries


def verify_zip(archive, expected, archive_sha=None):
    """Read only, no extraction; works independently of remote path constants."""
    if archive_sha is not None:
        require(valid_sha(archive_sha) and digest(archive) == archive_sha, 'Whole ZIP SHA256 disagrees')
    require(isinstance(expected, dict) and bool(expected), 'Manifest has no ZIP entries')
    with zipfile.ZipFile(archive, 'r') as z:
        infos = z.infolist()
        names = [item.filename for item in infos]
        require(len(names) == len(set(names)) and set(names) == set(expected), 'ZIP entry set differs from manifest')
        for info in infos:
            archive_name(info.filename)
            require(not info.is_dir() and not stat.S_ISLNK(info.external_attr >> 16), 'Unexpected directory or symlink entry')
            record = expected[info.filename]
            require(valid_sha(record.get('sha256')) and type(record.get('size')) is int
                    and record['size'] >= 0 and info.file_size == record['size'], 'ZIP entry size or SHA record is invalid')
            h = hashlib.sha256()
            with z.open(info, 'r') as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    h.update(chunk)
            require(h.hexdigest() == record['sha256'], 'ZIP entry SHA256 disagrees: ' + info.filename)
    return {'verified': True, 'zip_entries': len(expected), 'sha256': digest(archive)}


def verify_local(archive, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    require(manifest.get('schema') == 2, 'Unsupported export manifest schema')
    result = verify_zip(archive, manifest['zip_entries'], manifest['files']['implementation.zip']['sha256'])
    require(Path(archive).stat().st_size == manifest['files']['implementation.zip']['size'], 'Whole ZIP size disagrees')
    checked = []
    for name in OUTPUTS:
        if name in ('implementation.zip', 'export_manifest.json'):
            continue
        record = manifest['files'][name]
        entry = manifest['zip_entries']['results/' + name]
        require(record['sha256'] == entry['sha256'] and record['size'] == entry['size'], 'Generated report manifest disagrees')
        path = Path(manifest_path).resolve().parent / name
        if path.is_file():
            require(path.stat().st_size == record['size'] and digest(path) == record['sha256'],
                    'Downloaded deliverable differs from manifest: ' + name)
            checked.append(name)
    return {**result, 'downloaded_reports_also_verified': checked, 'mode': 'read-only; no extraction'}


def export():
    context = collect_context()
    for name in OUTPUTS:
        require(not safe_path(U / name).exists(), 'Refuse to overwrite an existing final deliverable: ' + name)
    choice = decision(context)
    generated = {'final_decision.json': json_bytes(choice),
                 'precision_recall_report.json': json_bytes(precision_report(context)),
                 'prototype_sep_result.txt': text_report(context, choice)}
    entries = collect_entries(generated)
    records = {name: {'sha256': row['sha256'], 'size': row['size'],
                     'source': str(row['path']) if 'path' in row else 'generated in this export'}
               for name, row in sorted(entries.items())}
    patch = records['study/patch_semantic_candidate.py']
    require(patch['sha256'] == context['integration']['patch_script_sha256'], 'Bundle substituted a different patch script')
    temporary, committed = [], []
    try:
        zip_tmp = safe_path(U / f'.implementation.{os.getpid()}.tmp.zip')
        require(not zip_tmp.exists(), 'Refuse stale export temporary file')
        temporary.append(zip_tmp)
        with zipfile.ZipFile(zip_tmp, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for name, row in sorted(entries.items()):
                if 'path' in row:
                    z.write(safe_path(row['path']), name)
                else:
                    z.writestr(name, row['data'])
        verified = verify_zip(zip_tmp, records)
        for label, source in (('A_geometry', ASRC), ('B_semantic', BSRC)):
            require(source_map(source) == context['studies'][label]['source_sha256'], 'Frozen source changed during export')
        require(source_map(OPTSRC) == context['optimized_KD_source_sha256'], 'Original optimized KD source changed during export')
        file_records = {name: {'sha256': byte_digest(data), 'size': len(data)} for name, data in generated.items()}
        file_records['implementation.zip'] = {'sha256': verified['sha256'], 'size': zip_tmp.stat().st_size}
        manifest = {'schema': 2, 'utc': context['utc'], 'files': file_records, 'zip_entries': records,
            'source_sha256': {**{key: study['source_sha256'] for key, study in context['studies'].items()},
                              'optimized_KD': context['optimized_KD_source_sha256']},
            'critical_file_sha256': context['critical_file_sha256'],
            'executed_remote_integration_patch_sha256': context['integration']['patch_script_sha256'],
            'original_pixel_KD_byte_identical_sha256': context['original_kd_sha256'],
            'selection': {'selected_method_id': choice['selected_method_id'],
                          'candidate_exceeds_required_endpoint': choice['candidate_exceeds_required_endpoint']},
            'completion_gates': {'A_and_B_status_complete': True, 'A_and_B_audits_complete': True,
                                 'A_and_B_analyses_complete': True},
            'local_verification': 'python export_semantic_result.py --verify implementation.zip --manifest export_manifest.json',
            'exclusions': ['all model checkpoints/weights', '__pycache__', '*.pyc', '*.pyo'],
            'manifest_note': 'This manifest lists every ZIP entry and every other newly generated deliverable; it cannot contain its own SHA256.'}
        staged = {}
        for name, payload in {**generated, 'export_manifest.json': json_bytes(manifest)}.items():
            tmp = safe_path(U / f'.{name}.{os.getpid()}.tmp')
            require(not tmp.exists(), 'Refuse stale export temporary file')
            temporary.append(tmp)
            with tmp.open('xb') as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            staged[name] = tmp
        staged['implementation.zip'] = zip_tmp
        for name in OUTPUTS:
            target = safe_path(U / name)
            require(not target.exists(), 'Final deliverable appeared during export: ' + name)
            os.replace(staged[name], target)
            committed.append(target)
        return {'exported': True, 'selected_method_id': choice['selected_method_id'],
                'candidate_exceeds_required_endpoint': choice['candidate_exceeds_required_endpoint'],
                'files': [str(U / name) for name in OUTPUTS], 'zip_entries': len(records),
                'ZIP_verified_before_publication': True, 'goal_status_changed': False}
    except BaseException:
        # Only files created by this invocation are removed. Never remove an
        # existing result or any training/evaluation/source evidence.
        for path in reversed(committed + temporary):
            checked = safe_path(path)
            if checked.is_file():
                checked.unlink()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Read-only final readiness check; no deliverables written')
    parser.add_argument('--verify', metavar='ZIP', help='Read-only verification of a downloaded archive')
    parser.add_argument('--manifest', metavar='JSON', help='Downloaded export_manifest.json for --verify')
    args = parser.parse_args()
    require(bool(args.verify) == bool(args.manifest), '--verify and --manifest must be provided together')
    require(not (args.check_only and args.verify), '--check-only and --verify are mutually exclusive')
    if args.verify:
        result = verify_local(args.verify, args.manifest)
    elif args.check_only:
        collect_context()
        result = {'ready': True, 'outputs_written': False, 'A_and_B_complete': True}
    else:
        result = export()
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError, KeyError, OSError, zipfile.BadZipFile) as exc:
        print(json.dumps({'exported': False, 'error': str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
