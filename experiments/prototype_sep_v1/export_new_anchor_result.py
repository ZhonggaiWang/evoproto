"""Export completed optimizedKD/A/B/new-anchor C evidence, stdlib CPU only.

Run only after all formal audits, analyses and the own-C128 GT diagnostic:
  python -B export_new_anchor_result.py
Read-only readiness check:
  python -B export_new_anchor_result.py --check-only
After downloading the ZIP and manifest, verify them on any local computer:
  python export_new_anchor_result.py --verify implementation.zip --manifest export_manifest.json

The original A exporter is preserved and is never imported or executed here.
Only five deliverables in the new U/new_anchor_export directory are written.
Download them to the local outputs/prototype_sep_v1 subdirectory; previous
root outputs are preserved. Existing files are never overwritten. All frozen
source and GT/training evidence remains read-only. Checkpoints stay remote.
No goal state changes: a failed candidate can be archived without claiming gain.
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
C = U / 'formal/c_new_anchor'
ASRC = E / 'a_geometry/src'
BSRC = E / 'b_semantic/src'
CSRC = E / 'c_new_anchor/src'
EXPORT_DIR = U / 'new_anchor_export'
METHODS = ('optimized_KD', 'A_geometry', 'B_semantic', 'C_new_anchor')
AUDIT_FLAGS = ('all_training_and_endpoint_requirements_complete',
    'all_semantic_sep_requirements_complete', 'all_best_pipeline_requirements_complete',
    'all_new_anchor_requirements_complete')
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
    if arm == 'c_new_anchor':
        require(audit.get('arm') == arm and all(audit.get(key) is True for key in AUDIT_FLAGS),
                'New-source C independent full audit has not passed')
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
    inherited_run = OPT if arm == 'c_new_anchor' else A
    require(training.get('stages') == 2 if arm == 'a_geometry' else
            training.get('stages_trained') == 1 and training.get('inherited_stage1') == str(inherited_run),
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


def collect_parent_context():
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
    for preflight_path, study in ((U / 'preflight.json', astudy), (E / 'semantic_preflight.json', bstudy)):
        preflight = read_json(preflight_path)
        require(preflight.get('passed') is True and preflight.get('source_sha256') == study['source_sha256'],
                'Preflight does not pass for the frozen final source')
        cache[str(preflight_path)] = digest(safe_path(preflight_path))
    for name, ep in endpoints.items():
        ep['final_checkpoint'] = str(({'optimized_KD': OPT, 'A_geometry': A, 'B_semantic': B}[name])
                                     / '10-5/step2/checkpoints/model_final.pth')
        ep['final_checkpoint_sha256'] = cache[ep['final_checkpoint']]
    return {'utc': now(), 'studies': {'A_geometry': astudy, 'B_semantic': bstudy},
            'audits': {'A_geometry': aaudit, 'B_semantic': baudit}, 'analyses': analyses,
            'endpoints': endpoints, 'inherited_stage1': {'A_geometry': a1, 'optimized_KD': opt1},
            'critical_file_sha256': cache, 'integration': integration, 'original_kd_sha256': kd_sha,
            'optimized_KD_source_sha256': origin['optimized_kd_source_sha256']}


def collect_context():
    """Do not create outputs until all four endpoint and GT gates succeed."""
    context = collect_parent_context()
    cache = context['critical_file_sha256']
    study, audit = require_run_complete(C, CSRC, E/'run_new_anchor_sep.py', 'c_new_anchor',
        [(2, i) for i in (4000, 6000, 8000)], (2,), cache)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(E))
    import analyze_new_anchor_sep as new_analysis
    complete = new_analysis.completion(ROOT)
    require(complete.get('complete') is True, 'New-source C final source/smoke/audit/evaluation analysis gate did not pass')
    report_path = U/'new_anchor_analysis.json'
    report = read_json(report_path)
    require(report.get('status') == 'complete_endpoint_analysis'
            and report.get('completion', {}).get('complete') is True
            and report.get('run') == str(C) and report.get('study') == study,
            'New-source C completed analysis is missing or describes another study')
    endpoint = checked_endpoint(C/'evaluations/step2_iter8000/result.json', 2)
    same_analysis_endpoint(report['candidate_endpoint'], endpoint)
    for label in ('optimized_KD', 'A_geometry', 'B_semantic'):
        same_analysis_endpoint(report['references'][label], context['endpoints'][label])
    require(endpoint['gt_class_pixels'] == context['endpoints']['optimized_KD']['gt_class_pixels'],
            'Four methods must share every original GT source-row denominator')
    require(study.get('teacher') == str(OPT/'10-5/step1/checkpoints/model_final.pth')
            and study.get('resume_checkpoint') == str(OPT/'10-5/step2/checkpoints/model_iter_2000.pth')
            and study.get('restored_state') == ['student', 'optimizer']
            and study.get('fresh_state') == ['online_confusion', 'geometry_selector'],
            'C must restore the strongest existing pipeline and fresh online state')
    hash_matches(Path(study['teacher']), study['teacher_sha256'], cache)
    hash_matches(Path(study['resume_checkpoint']), study['resume_checkpoint_sha256'], cache)
    hash_matches(E/'launch_new_anchor_sep.py', study['launcher_sha256'], cache)
    hash_matches(E/'new_anchor_origin.json', study['new_anchor_origin_sha256'], cache)
    hash_matches(E/'new_anchor_selector_tests.json', study['new_anchor_selector_tests_sha256'], cache)
    endpoint['final_checkpoint'] = str(C/'10-5/step2/checkpoints/model_final.pth')
    endpoint['final_checkpoint_sha256'] = hash_matches(endpoint['final_checkpoint'],
        read_json(C/'10-5/step2/training_complete.json')['checkpoint_sha256'], cache)
    hash_matches(C/'10-5/step2/checkpoints/model_iter_8000.pth', endpoint['checkpoint_sha256'], cache)
    require(context['original_kd_sha256'] == study['source_sha256']['model/pixel_kd.py'],
            'Existing optimized KD must remain byte-identical in all four methods')
    geometry = report['geometry_mechanism']
    diagnostic = new_analysis.endpoint_GT_diagnostic(new_analysis.paths(ROOT), study, geometry)
    require(diagnostic.get('status') == 'complete_readonly_subset_diagnostic'
            and report.get('own_endpoint_anchor_GT_diagnostic') == diagnostic,
            'Own-C128 GT diagnostic and the current completed analysis must both exist and match')
    gt_path = U/'new_anchor_endpoint_gt_diagnostic.json'
    hash_matches(gt_path, diagnostic['sha256'], cache)
    hash_matches(C/'completion_audit.json', report['completion']['independent_C_audit']['sha256'], cache)
    require(study.get('selector_schema') == 4 and study.get('source_anchor_policy') == 'current_new_only'
            and report['new_anchor_selection_mechanism']['all_logged_sources_are_current_new'] is True,
            'Old/background sources cannot be credited to new-source C')
    context['studies']['C_new_anchor'], context['audits']['C_new_anchor'] = study, audit
    context['analyses']['C'] = report
    context['endpoints']['C_new_anchor'] = endpoint
    context['new_anchor_origin'] = read_json(E/'new_anchor_origin.json')
    context['new_anchor_selector_tests'] = read_json(E/'new_anchor_selector_tests.json')
    context['own_C_GT_diagnostic'] = diagnostic
    context['C_fresh_online_budget'] = {'restored_iteration': 2000, 'remaining_updates': 6000,
        'observer_updates': 6000, 'image_exposures': 48000, 'coldstart_updates': 100, 'ramp_updates': 200}
    return context


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
    wins = eps['C_new_anchor']['metrics']['all_miou'] > eps['optimized_KD']['metrics']['all_miou']
    require(all(eps[name]['metrics']['all_miou'] <= eps['optimized_KD']['metrics']['all_miou']
                for name in ('A_geometry', 'B_semantic')), 'A/B historical continuation gate changed')
    selected = 'C_new_anchor' if wins else 'optimized_KD'
    require(context['analyses']['C']['recommendation']['candidate_exceeds_required_endpoint'] is wins,
            'C final recommendation disagrees with the full endpoint criterion')
    endpoint = eps[selected]
    source = CSRC if wins else OPTSRC
    rows = [{'method': name, **eps[name]['metrics'],
             'delta_all_vs_optimized_KD_pp': eps[name]['metrics']['all_miou'] - eps['optimized_KD']['metrics']['all_miou'],
             'evaluated_checkpoint_sha256': eps[name]['checkpoint_sha256'],
             'final_checkpoint_sha256': eps[name]['final_checkpoint_sha256']}
            for name in METHODS]
    return {'schema': 3, 'utc': context['utc'], 'status': 'complete_endpoint_decision',
            'candidate_exceeds_required_endpoint': wins,
            'method': 'new_anchor_semantic_protected_prototype_SEP_plus_optimized_KD' if wins else 'existing_optimized_KD',
            'selected_method_id': selected, 'selected_checkpoint': endpoint['final_checkpoint'],
            'selected_checkpoint_sha256': endpoint['final_checkpoint_sha256'],
            'selected_evaluated_checkpoint_sha256': endpoint['checkpoint_sha256'],
            'selected_source': str(source), 'selected_source_sha256': source_map(source),
            'criterion': 'Select new-source C iff its full stage2/8000 all-class mIoU strictly exceeds original optimized KD; otherwise retain optimized KD. Midpoint recovery, GT128 anchor accuracy or isolated pair gains cannot select C.',
            'metrics_recomputed_from_full_GT_histograms': True, 'endpoint_table': rows,
            'final_and_evaluated_checkpoint_relation': 'File SHA may differ; completed CPU audits require model, optimizer, observer and selector tensors to be exactly equal for A/B/C final vs iteration8000.',
            'integrity_receipts': [str(run/'completion_audit.json') for run in (A, B, C)],
            'analysis_receipts': [str(U/name) for name in ('result_analysis.json', 'semantic_analysis.json', 'new_anchor_analysis.json')],
            'own_C_GT_diagnostic': {'source': context['own_C_GT_diagnostic']['source'],
                'sha256': context['own_C_GT_diagnostic']['sha256'],
                'role': 'Fixed128-prefix descriptive mechanism diagnostic; complete1449 endpoint remains the effect criterion.'},
            'C_training_budget': {'inherited_stage1': str(OPT), 'new_stage1_training': False,
                'stage2_resume_iteration': 2000, 'new_stage2_updates': 6000, 'global_batch': 8,
                'restored_state': ['student', 'optimizer'], 'fresh_state': ['online_confusion', 'geometry_selector'],
                'final_observer_updates': 6000, 'final_observer_seen_images': 48000,
                'coldstart_updates': 100, 'ramp_updates': 200},
            'C_source_change': 'Only schema4/current-new-source geometry selector differs from frozen B; all other Python sources, semantic protection and optimized KD are byte-identical.',
            'excluded_old_C': {'arm': 'c_best_teacher', 'formal_training': False,
                'state': 'smoke_only', 'included_in_four_method_endpoint_table': False,
                'note': 'Previous best-teacher proposal completed only an integration smoke; it supplies no formal effect estimate and is excluded from this bundle.'},
            'B_training_budget': {'inherited_stage1': str(A), 'new_stage1_training': False,
                'stage2_resume_iteration': 2000, 'new_stage2_updates': 6000, 'global_batch': 8,
                'restored_state': ['student', 'optimizer', 'online_confusion', 'geometry_selector'],
                'final_observer_updates': 8000, 'final_observer_seen_images': 64000},
            'step0_retrained': False, 'baseline_retrained': False, 'random_seed_search': False,
            'host': '8card', '4card': 'unused', 'all_remote_outputs_and_caches_within': str(ALLOWED),
            'evidence_scope': 'One fixed seed0 adaptive development validation endpoint; no stability, significance or independent generalization claim.',
            'observed_endpoint_improvement_by_C': wins,
            'result_interpretation': 'Small fixed-seed adaptive development-validation gain; no stability, significance, independent-test or selector-causal claim.' if wins else 'No required endpoint gain; retain optimized KD and record a stage report without completing the goal.',
            'stage_report_only_when_no_gain': not wins,
            'goal_status_change_permitted_by_this_script': False,
            'limitations': context['analyses']['C']['limitations']}


def text_report(context, result):
    eps = context['endpoints']
    a1 = context['inherited_stage1']['A_geometry']['metrics']
    o1 = context['inherited_stage1']['optimized_KD']['metrics']
    names = {'optimized_KD': '原优化 KD', 'A_geometry': 'A 混淆指导原型 SEP + 原优化 KD',
             'B_semantic': 'B 语义保护原型 SEP + 原优化 KD',
             'C_new_anchor': 'C 新类锚点指导 + 语义保护原型 SEP + 最强原优化 KD'}
    lines = ['混淆指导原型 SEP 与优化 KD：完整终点交付',
        '最终选择：' + names[result['selected_method_id']] + '。',
        '判据：只有新类锚点候选 C 在阶段2/8000步的整体 mIoU 严格高于原优化 KD，才选择 C；否则保留原优化 KD。',
        'VOC 10-5，固定 seed=0，完整1449张验证图；整体含背景，旧15/新5只含前景。', '',
        '方案 | 整体 mIoU | 旧15 mIoU | 新5 mIoU']
    for name in METHODS:
        m = eps[name]['metrics']
        lines.append(f"{names[name]} | {m['all_miou']:.6f} | {m[METRICS[1]]:.6f} | {m[METRICS[2]]:.6f}")
    for name in ('A_geometry', 'B_semantic', 'C_new_anchor'):
        delta = [eps[name]['metrics'][key] - eps['optimized_KD']['metrics'][key] for key in METRICS]
        lines.append(f"{name} 相对原优化 KD（百分点） | {delta[0]:+.6f} | {delta[1]:+.6f} | {delta[2]:+.6f}")
    lines += ['', '继承与预算：',
        f"A 阶段1终点整体 {a1['all_miou']:.6f}，原优化 KD {o1['all_miou']:.6f}，差值 {a1['all_miou']-o1['all_miou']:+.6f} 个百分点。",
        f"该阶段新5 mIoU 差值 {a1[METRICS[2]]-o1[METRICS[2]]:+.6f} 个百分点。B 继承 A 的这个教师，包括已有质量差距；B 没有另训阶段1。",
        'A 复用共享阶段1的2000步模型和优化器，新增6000步；A 阶段2训练8000步。',
        'B 复用 A 阶段2的2000步模型、优化器、在线混淆和选择器，新增6000步。最终 observer 是8000次更新、64000次图像曝光。',
        'C 复用原最强优化 KD 的阶段1最终教师及阶段2/2000模型和优化器；observer/selector从零开始，不借用A/B。仅追加6000步，最终6000更新、48000次图像曝光；原100次冷启动和200次ramp保持。',
        '旧 c_best_teacher 只完成烟测，没有正式训练终点；它不进入四方法结果表，也不打包为有效实验。',
        '共享 step0、原优化 KD 和历史 baseline 均未重训，没有种子搜索。8卡机8x1保持 global batch8；4卡机未使用。',
        '', '实现：',
        '在线定向混淆只挑选需要关注的方向：完整 broad EMA 行归一化排序，可信 CAM 支持和新鲜度筛选；混淆比例不作为精确学习权重。',
        'C 只允许当前新类作为方向来源，仍可选择旧类或其它新类为前景竞争类别。背景/自身排除；完整 broad 行分母保留。最多5个来源改变了类别对覆盖及有效对均值的梯度分配。',
        '原型 SEP 将方向去重为涉及新类的几何对；旧新对只更新新原型，新新对更新双方，排除背景、旧旧和自配对。固定 lambda0.1、margin0，对全部有效对的平方余弦 hinge 取均值。',
        'B 先计算当前新原型各行的切空间 SEP/语义梯度，对反向分量作正交投影。语义梯度按真实 global valid-pixel 分母跨卡汇合；保护在 DDP 输出包装之前计算。',
        '返回 surrogate 保留几何损失的前向值，用已脱离计算图的修正梯度执行普通一次 backward。原语义损失仍只正常加入一次。',
        'C仅替换B的类别对选择器，其它源码与语义保护保持字节一致；原优化 pixel KD 在四种方法中字节一致。GT 只用于已有完整验证直方图和诊断，未参与在线选择、投影或学习强度。',
        '', '证据：',
        'A：25项原型行为测试、16项选择器测试、真实图像梯度检查、8卡两阶段烟测，以及7个正式全量评估任务。',
        'B：23项语义保护CPU测试、2进程Gloo与8卡NCCL数学/真实DDP测试、8卡真实网络4步烟测，以及3个正式全量评估任务。',
        'C：10项新类来源选择器CPU测试、独立8卡真实恢复8步烟测；共享B已通过的损失与DDP证据不重复跑。C正式3个全量评估任务及同候选128张终点GT诊断完成。',
        '所有正式评估任务各4分片完整覆盖；训练和评估退出码均为0。A/B/C完整审计通过，final与被评估8000步模型、优化器、observer及selector状态逐张量一致。',
        '四种方法的原GT类别行总量相同，IoU由已有直方图重新计算。分类及旧15/新5微平均precision/recall见precision_recall_report.json。',
        '', '结果的解释范围：',
        '投影只保证记录到的当前切空间分量梯度条件，不保证 AdamW动量、权重衰减、归一化参数更新、后续梯度或最终IoU改善。',
        '固定seed0、同一开发验证集上的逐步改进不能证明多种子稳定性、统计显著性或独立测试泛化。',
        '候选8x1与历史原优化 KD 的4x2保持global batch8，但恢复时数据采样/增强流重启，差值无法严格归因于单一模块。',
        'CAM支持计数是重复训练曝光，不是独立样本量或GT正确率；采样梯度诊断也不是全部数据的保护保证。',
        'A的128张开发验证诊断在选中分歧像素中发现旧来源GT正确率1.601%、新来源91.832%。这些条件子集的比例不代表在线模型全体正确率、概率校准或精确学习权重。C终点诊断使用自己的模型、选择器和最强KD教师，仍只作为机制可信度证据。',
        'C相对B同时更换教师、模型/优化器轨迹、在线冷启动与来源限制，是最强现有流程的附加候选，不能严格拆成教师或新类来源的单因素因果收益。',
        '', f"保留实现：{result['selected_source']}", f"保留检查点：{result['selected_checkpoint']}",
        f"检查点 SHA256：{result['selected_checkpoint_sha256']}",
        '检查点留在远程授权目录；implementation.zip包含原优化KD/A/B/C完整源码及实现、测试、审计和诊断证据，不包含模型权重。旧交付文件保留，此包为独立阶段报告。',
        '下载后可用随包 exporter 的 --verify 与 export_manifest.json 验证每个ZIP条目的SHA256和整个ZIP。']
    if not result['candidate_exceeds_required_endpoint']:
        lines.insert(3, 'C 本次候选未满足最终提升目标；这份交付记录完整结果，goal仍待继续，本脚本不会将目标标为完成。')
    else:
        delta_all = eps['C_new_anchor']['metrics']['all_miou'] - eps['optimized_KD']['metrics']['all_miou']
        lines.insert(3, f'C 完整终点相对原优化 KD 提高 {delta_all:+.6f} 个百分点，是固定seed0的开发验证小幅收益；尚无稳定性、显著性、独立测试或选择器单因素因果结论。')
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
    for label, source in (('optimized_KD', OPTSRC), ('A_geometry', ASRC), ('B_semantic', BSRC), ('C_new_anchor', CSRC)):
        for path in sorted(source.rglob('*')):
            add(path, label + '/src/' + path.relative_to(source).as_posix())
    # Actual remote experiment scripts, including the executed CLI patch, are
    # bundled as found on this host; no local development copy replaces them.
    for path in sorted(E.rglob('*')):
        if any(path.is_relative_to(source) for source in (ASRC, BSRC, CSRC)):
            continue
        if 'best_teacher' in path.name or 'c_best_teacher' in path.parts:
            continue
        if path.suffix.lower() in {'.py', '.json', '.log', '.jsonl'}:
            add(path, 'study/' + path.relative_to(E).as_posix())
    for label, run in (('a_geometry', A), ('b_semantic', B), ('c_new_anchor', C)):
        for path in sorted(run.rglob('*')):
            if path.suffix.lower() in {'.json', '.jsonl', '.log', '.txt'}:
                add(path, 'evidence/formal/' + label + '/' + path.relative_to(run).as_posix())
    # Include diagnostic/reference JSON recursively and both completed smoke
    # receipts. Exclude formal here (already above) and previous deliverables.
    for path in sorted(U.rglob('*')):
        if path.is_relative_to(U / 'formal') or path.is_relative_to(EXPORT_DIR) or path.name in OUTPUTS:
            continue
        if 'best_teacher' in path.name or 'c_best_teacher' in path.parts:
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
                       (OPT / 'training_complete.json', 'optimized_KD_training_complete.json'),
                       (OPT / 'evaluation_workers_complete.json', 'optimized_KD_evaluation_workers_complete.json'),
                       (ROOT / 'runs/fixed_baseline_v1/source_snapshot.json', 'baseline_source_snapshot.json'),
                       (ROOT / 'experiments/kd_pixel_v2/origin.json', 'baseline_origin.json')):
        add(path, 'evidence/reference/' + name)
    for name, payload in generated.items():
        key = archive_name('results/' + name)
        require(key not in entries, 'Duplicate generated ZIP entry')
        entries[key] = {'data': payload, 'sha256': byte_digest(payload), 'size': len(payload)}
    required_entries = ('study/b_semantic/semantic_integration.json', 'study/patch_semantic_candidate.py',
        'evidence/study/preflight.json', 'study/semantic_preflight.json', 'study/semantic_test_receipts.json',
        'evidence/formal/a_geometry/completion_audit.json', 'evidence/formal/b_semantic/completion_audit.json',
        'evidence/study/result_analysis.json', 'evidence/study/semantic_analysis.json',
        'evidence/formal/b_semantic/10-5/step2/semantic_guard_metrics.jsonl',
        'C_new_anchor/src/model/geometry_pair_selector.py',
        'C_new_anchor/src/model/semantic_protected_sep.py',
        'study/new_anchor_origin.json', 'study/new_anchor_selector_tests.json',
        'study/new_anchor_geometry_selector.py', 'study/test_new_anchor_selector.py',
        'study/run_new_anchor_sep.py', 'study/launch_new_anchor_sep.py',
        'study/audit_new_anchor_sep.py', 'study/analyze_new_anchor_sep.py',
        'study/diagnose_new_anchor_gt.py', 'study/export_new_anchor_result.py',
        'evidence/formal/c_new_anchor/completion_audit.json',
        'evidence/formal/c_new_anchor/10-5/step2/semantic_guard_metrics.jsonl',
        'evidence/study/new_anchor_analysis.json',
        'evidence/study/new_anchor_endpoint_gt_diagnostic.json',
        'evidence/study/smoke/c_new_anchor/study.json')
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
    require(manifest.get('schema') == 3, 'Unsupported new-anchor export manifest schema')
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
        require(not safe_path(EXPORT_DIR / name).exists(), 'Refuse to overwrite an existing stage deliverable: ' + name)
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
    for key, path in context['critical_file_sha256'].items():
        require(digest(safe_path(key)) == path, 'Completed critical evidence changed during export preparation: ' + key)
    safe_path(EXPORT_DIR).mkdir(parents=True, exist_ok=True)
    temporary, committed = [], []
    try:
        zip_tmp = safe_path(EXPORT_DIR / f'.implementation.{os.getpid()}.tmp.zip')
        require(not zip_tmp.exists(), 'Refuse stale export temporary file')
        temporary.append(zip_tmp)
        with zipfile.ZipFile(zip_tmp, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for name, row in sorted(entries.items()):
                if 'path' in row:
                    z.write(safe_path(row['path']), name)
                else:
                    z.writestr(name, row['data'])
        verified = verify_zip(zip_tmp, records)
        for label, source in (('A_geometry', ASRC), ('B_semantic', BSRC), ('C_new_anchor', CSRC)):
            require(source_map(source) == context['studies'][label]['source_sha256'], 'Frozen source changed during export')
        require(source_map(OPTSRC) == context['optimized_KD_source_sha256'], 'Original optimized KD source changed during export')
        file_records = {name: {'sha256': byte_digest(data), 'size': len(data)} for name, data in generated.items()}
        file_records['implementation.zip'] = {'sha256': verified['sha256'], 'size': zip_tmp.stat().st_size}
        manifest = {'schema': 3, 'utc': context['utc'], 'files': file_records, 'zip_entries': records,
            'source_sha256': {**{key: study['source_sha256'] for key, study in context['studies'].items()},
                              'optimized_KD': context['optimized_KD_source_sha256']},
            'critical_file_sha256': context['critical_file_sha256'],
            'executed_remote_integration_patch_sha256': context['integration']['patch_script_sha256'],
            'original_pixel_KD_byte_identical_sha256': context['original_kd_sha256'],
            'selection': {'selected_method_id': choice['selected_method_id'],
                          'candidate_exceeds_required_endpoint': choice['candidate_exceeds_required_endpoint']},
            'completion_gates': {'A_B_C_status_complete': True, 'A_B_C_audits_complete': True,
                                 'A_B_C_analyses_complete': True, 'own_C_endpoint_GT128_complete': True,
                                 'C_source_origin_CPU_tests_and_actual_smoke_complete': True},
            'method_registry': choice['endpoint_table'], 'excluded_old_C': choice['excluded_old_C'],
            'scope': 'Independent stage report. No goal status changes; candidate failure stays failure and optimized KD is retained.',
            'remote_export_directory': str(EXPORT_DIR),
            'intended_local_directory': 'outputs/prototype_sep_v1',
            'local_verification': 'python export_new_anchor_result.py --verify implementation.zip --manifest export_manifest.json',
            'exclusions': ['all model checkpoints/weights', '__pycache__', '*.pyc', '*.pyo'],
            'manifest_note': 'This manifest lists every ZIP entry and every other newly generated deliverable; it cannot contain its own SHA256.'}
        staged = {}
        for name, payload in {**generated, 'export_manifest.json': json_bytes(manifest)}.items():
            tmp = safe_path(EXPORT_DIR / f'.{name}.{os.getpid()}.tmp')
            require(not tmp.exists(), 'Refuse stale export temporary file')
            temporary.append(tmp)
            with tmp.open('xb') as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            staged[name] = tmp
        staged['implementation.zip'] = zip_tmp
        for name in OUTPUTS:
            target = safe_path(EXPORT_DIR / name)
            require(not target.exists(), 'Final deliverable appeared during export: ' + name)
            # Exclusive creation also rejects a file that appears after the
            # preceding check. Publish the manifest last; no old file can be
            # replaced by a race between exists() and rename/replace().
            with target.open('xb') as destination:
                committed.append(target)
                with staged[name].open('rb') as source:
                    for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
                        destination.write(chunk)
                destination.flush()
                os.fsync(destination.fileno())
            require(digest(target) == digest(staged[name]), 'Published artifact digest mismatch: ' + name)
            safe_path(staged[name]).unlink()
        return {'exported': True, 'selected_method_id': choice['selected_method_id'],
                'candidate_exceeds_required_endpoint': choice['candidate_exceeds_required_endpoint'],
                'files': [str(EXPORT_DIR / name) for name in OUTPUTS], 'zip_entries': len(records),
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
        result = {'ready': True, 'outputs_written': False, 'A_B_C_and_own_C_GT_complete': True,
                  'export_directory': str(EXPORT_DIR), 'goal_status_changed': False}
    else:
        result = export()
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError, KeyError, OSError, zipfile.BadZipFile) as exc:
        print(json.dumps({'exported': False, 'error': str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
