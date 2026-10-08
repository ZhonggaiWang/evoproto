"""CPU-only integrity/endpoint audit; missing running work is INCOMPLETE.

Default invocation is read-only and prints JSON. To save the final evidence:
  python -B audit_confusion_guided.py --require-complete \
    --output /ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto/runs/confusion_guided_v1/completion_audit.json

No model forward pass, training data iteration, CUDA allocation, or statistical
accuracy claim is performed. The output distinguishes integrity from benefit.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import math
import os
import sys

# Even when the caller has training GPUs visible, this separate audit process
# must only deserialize/tensor-compare weights on CPU.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

import numpy as np
import torch

torch.set_num_threads(2)

ALLOWED = Path('/ML-vePFS/infra_rd/kun/others/wzg')
ROOT = ALLOWED / 'workspace/evoproto'
ARMS = ('a_sep', 'b_pairkd')
STEP0_SHA = '4f298e14721630cf3c66ba4f6af04bd198ebeaf77adc6b7b07ab8403ad0893b5'
ARCHIVE_SHA = 'e4a9c503c8967c631f3c53955f516744f645aefdf3dd68e71086ddb387a888d1'
CLASS_NAMES = ('_background_', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def required(path, pending):
    path = Path(path)
    if not path.exists():
        pending.append(str(path))
        return None
    try:
        return read_json(path)
    except json.JSONDecodeError:
        # A config/log can be observed while being initially written. Endpoint
        # completion is decided only after all records are readable.
        pending.append('Not yet readable JSON: ' + str(path))
        return None


def same_state(a, b):
    """Exact recursive equality for tensor buffers plus module extra state."""
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same_state(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(same_state(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def state_metadata(state, cfg, step, classes, selector=False):
    if not isinstance(state, dict):
        return False, {'reason': 'state is absent or not a mapping'}
    extra = state.get('_extra_state', {})
    expected = {'classes': classes, 'stage': step, 'class_ids': list(range(classes))}
    if selector:
        expected.update(schema=1, refresh_interval=cfg['pair_refresh_interval'],
                        min_row_images=cfg['pair_min_row_images'], min_pair_images=cfg['pair_min_pair_images'],
                        min_rate=cfg['pair_min_rate'], min_updates=cfg['pair_min_updates'],
                        ramp_updates=cfg['pair_ramp_updates'], max_stale_updates=cfg['pair_max_stale_updates'])
        shape_keys = {'targets': (classes,), 'selected_rates': (classes,), 'last_refresh_iteration': ()}
    else:
        expected.update(schema=2, momentum=cfg['confusion_momentum'],
                        high_threshold=cfg['high_thre'], low_threshold=cfg['low_thre'])
        shape_keys = {key: (classes, classes) for key in ['counts', 'accepted_counts', 'broad_counts',
                     'ema_counts', 'broad_ema_counts', 'pair_image_observations', 'broad_pair_image_observations']}
        shape_keys.update({key: (classes,) for key in ['image_observations', 'ema_image_observations',
                          'broad_image_observations', 'last_seen_update', 'broad_last_seen_update']})
        shape_keys.update(updates=(), seen_images=())
    mismatch = {key: {'expected': value, 'actual': extra.get(key)}
                for key, value in expected.items() if extra.get(key) != value}
    shape_bad = [key for key, shape in shape_keys.items()
                 if not isinstance(state.get(key), torch.Tensor) or tuple(state[key].shape) != shape]
    tensor_bad = [key for key, value in state.items() if isinstance(value, torch.Tensor)
                  and value.is_floating_point() and not bool(torch.isfinite(value).all())]
    info = {'metadata': extra, 'metadata_mismatches': mismatch,
            'invalid_tensor_shapes': shape_bad, 'nonfinite_buffers': tensor_bad}
    valid = not mismatch and not shape_bad and not tensor_bad
    if selector and valid:
        targets = state['targets'].long()
        ids = torch.arange(classes)
        valid = bool(((targets == -1) | ((targets > 0) & (targets < classes) & (targets != ids))).all())
        valid = valid and int(targets[0]) == -1
        info.update(targets=targets.tolist(), targets_valid=valid,
                    last_refresh_iteration=int(state['last_refresh_iteration']))
    elif valid:
        info.update(updates=int(state['updates']), seen_images=int(state['seen_images']))
    return valid, info


def endpoint_identity(stage, cfg, step, checks, details, pending):
    final = stage / 'checkpoints/model_final.pth'
    evaluated = stage / 'checkpoints/model_iter_8000.pth'
    if not final.exists() or not evaluated.exists():
        pending.extend(str(p) for p in (final, evaluated) if not p.exists())
        return
    a_hash, b_hash = digest(final), digest(evaluated)
    a = torch.load(final, map_location='cpu', weights_only=True, mmap=True)
    b = torch.load(evaluated, map_location='cpu', weights_only=True, mmap=True)
    am, bm = a.get('model_state'), b.get('model_state')
    checks['final_matches_evaluated_model'] = isinstance(am, dict) and isinstance(bm, dict) and same_state(am, bm)
    checks['evaluated_iteration_8000'] = b.get('iteration') == 8000
    optimizer = b.get('optimizer_state')
    checks['evaluated_optimizer_saved'] = isinstance(optimizer, dict) and bool(optimizer.get('state')) and bool(optimizer.get('param_groups'))
    details['endpoint_identity'] = {'final_sha256': a_hash, 'evaluated_sha256': b_hash,
                                  'verified_model_tensor_count': len(am) if isinstance(am, dict) else None,
                                  'all_model_tensors_exactly_equal': checks['final_matches_evaluated_model']}
    classes = 16 if step == 1 else 21
    for key, short, selector in [('online_confusion_state', 'observer', False),
                                 ('pair_selector_state', 'selector', True)]:
        valid_a, info_a = state_metadata(a.get(key), cfg, step, classes, selector=selector)
        valid_b, info_b = state_metadata(b.get(key), cfg, step, classes, selector=selector)
        checks[short + '_state_saved_and_consistent'] = valid_a and valid_b
        checks[short + '_final_and_evaluated_exact'] = same_state(a.get(key), b.get(key)) and valid_a and valid_b
        details[short + '_state'] = {'final': info_a, 'evaluated': info_b}
        if short == 'observer' and valid_a:
            expected_updates = 6000 if step == 1 else 8000
            checks['observer_updates_match_executed_budget'] = info_a['updates'] == expected_updates
            checks['observer_images_match_global_batch'] = info_a['seen_images'] == expected_updates * 8
        if short == 'selector' and valid_a:
            checks['selector_iteration_within_stage'] = 0 <= info_a['last_refresh_iteration'] <= 8000
    del a, b, am, bm


def near(a, b):
    if a is None or b is None:
        return a is None and b is None
    return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a) - float(b)) <= 1e-9


def histogram_audit(run, cfg, step, checks, details, pending):
    part = run / f'evaluations/step{step}_iter8000'
    result = required(part / 'result.json', pending)
    job = required(run / f'eval_queue/step{step}_iter8000.json', pending)
    if result is None or job is None:
        return
    evaluated = run / f'10-5/step{step}/checkpoints/model_iter_8000.pth'
    if evaluated.exists():
        checks['result_checkpoint_verified'] = (Path(job['checkpoint']) == evaluated
            and digest(evaluated) == job['checkpoint_sha256'] == result['checkpoint_sha256'])
    else:
        pending.append(str(evaluated))
    checks['result_stage_iteration_verified'] = result['step'] == job['step'] == step and result['iteration'] == job['iteration'] == 8000
    checks['evaluation_config_matches_training'] = job['config'] == cfg
    n, images, old = (16, 1240, 10) if step == 1 else (21, 1449, 15)
    hist = np.asarray(result['histogram'])
    checks['histogram_shape_valid'] = hist.shape == (n, n)
    checks['histogram_nonnegative_integer_counts'] = np.issubdtype(hist.dtype, np.integer) and bool((hist >= 0).all())
    checks['all_validation_images'] = result['images'] == images
    if not checks['histogram_shape_valid'] or not checks['histogram_nonnegative_integer_counts']:
        return
    reference = required(ROOT / f'runs/kd_parallel_v1/formal/b_relational/evaluations/step{step}_iter8000/result.json', pending)
    if reference is not None:
        rh = np.asarray(reference['histogram'], dtype=np.int64)
        checks['GT_row_totals_match_optimized_KD'] = rh.shape == hist.shape and np.array_equal(hist.sum(1), rh.sum(1))
        details['GT_row_totals'] = hist.sum(1).tolist()
    union = hist.sum(1) + hist.sum(0) - hist.diagonal()
    iou = np.divide(hist.diagonal(), union, out=np.full(n, np.nan), where=union > 0) * 100
    def mean(values):
        return float(np.nanmean(values)) if np.isfinite(values).any() else None
    computed = {'all_miou': mean(iou), 'foreground_miou': mean(iou[1:]),
                'previous_foreground_miou': mean(iou[1:old+1]), 'current_foreground_miou': mean(iou[old+1:]),
                'old_initial10_miou': mean(iou[1:11]), 'new_since_initial_miou': mean(iou[11:])}
    mismatch = {key: {'reported': result.get(key), 'computed': value}
                for key, value in computed.items() if not near(result.get(key), value)}
    checks['reported_mIoU_matches_histogram'] = not mismatch
    names = CLASS_NAMES[:n]
    per_class = {name: float(iou[i]) if np.isfinite(iou[i]) else None for i, name in enumerate(names)}
    checks['reported_class_IoU_matches_histogram'] = (set(result['class_iou']) == set(names)
        and all(near(result['class_iou'].get(name), value) for name, value in per_class.items()))
    counters = {'old_gt_to_new_pixels': int(hist[1:old+1, old+1:].sum()),
                'old_gt_to_background_pixels': int(hist[1:old+1, 0].sum()), 'old_gt_pixels': int(hist[1:old+1].sum()),
                'new_gt_to_old_pixels': int(hist[old+1:, 1:old+1].sum()), 'new_gt_pixels': int(hist[old+1:].sum())}
    checks['reported_confusion_counters_match_histogram'] = all(result.get(key) == value for key, value in counters.items())
    details['metrics'] = computed
    details['metric_mismatches'] = mismatch
    shards = [required(part / f'rank{rank}.json', pending) for rank in range(2)]
    if all(shard is not None for shard in shards):
        seen = [name for shard in shards for name in shard['images']]
        split = Path(cfg['list_folder']) / f'incremental_split/val_{cfg["task"]}_step_{step+1}.txt'
        expected = split.read_text().splitlines()
        checks['validation_shards_disjoint_and_complete'] = (len(seen) == images == len(expected)
            and len(set(seen)) == len(seen) and set(seen) == set(expected))
        checks['validation_shard_provenance_verified'] = all(shard['rank'] == rank and shard['shards'] == 2
            and shard['step'] == step and shard['iteration'] == 8000 and shard['checkpoint_sha256'] == result['checkpoint_sha256']
            for rank, shard in enumerate(shards))
        shard_hist = [np.asarray(shard['histogram']) for shard in shards]
        checks['merged_histogram_matches_shards'] = all(h.shape == (n, n) for h in shard_hist) and np.array_equal(np.sum(shard_hist, axis=0), hist)


def audit_stage(run, manifest, step, original_warmup, pending):
    stage = run / f'10-5/step{step}'
    checks, details = {}, {'step': step}
    local_pending = []
    launch = required(stage / 'launch.json', local_pending)
    cfg = required(stage / 'config.json', local_pending)
    complete = required(stage / 'training_complete.json', local_pending)
    if launch is not None:
        if 'returncode' in launch:
            checks['training_launch_exit0'] = launch['returncode'] == 0
        else:
            local_pending.append(str(stage / 'launch.json') + ': training still running')
        checks['stage_GPU_assignment_matches_manifest'] = launch['gpus'] == manifest['gpus']
    if cfg is not None:
        expected = dict(manifest['common_config'])
        expected.update(step=step, spg=2, max_iters=8000, warmup_iters=2000, loss_warmup_iters=2000,
                        seed=0, train_limit=0, val_limit=0, w_proto_kd=0., w_proto_sep=0., ald=False,
                        confusion_reweight=False, online_confusion=True, async_eval=True,
                        w_pixel_kd=.1, kd_temperature=2., w_pair_sep=.1,
                        pair_mode='sep' if manifest['arm'] == 'a_sep' else 'pairkd')
        mismatch = {key: {'expected': value, 'actual': cfg.get(key)} for key, value in expected.items() if cfg.get(key) != value}
        checks['stage_budget_and_common_config_verified'] = not mismatch
        details['config_mismatches'] = mismatch
        checks['unchanged_global_batch'] = cfg['spg'] * len(manifest['gpus']) == manifest['global_batch'] == 8
        teacher = Path(manifest['step0']['step0']) if step == 1 else run / '10-5/step1/checkpoints/model_final.pth'
        checks['expected_teacher_verified'] = Path(cfg['prev_checkpoint']) == teacher
        log = stage / 'train.log'
        if log.exists():
            text = log.read_text()
            checks['teacher_load_recorded'] = str(teacher) in text
            if step == 1:
                checks['shared_student_and_optimizer_resume_recorded'] = 'Resume student AND optimizer at iteration 2000' in text
        else:
            local_pending.append(str(log))
        if step == 1:
            checks['warmup_resume_path_verified'] = cfg['resume_checkpoint'] == original_warmup['shared_warmup_checkpoint']
        else:
            checks['fresh_stage2_optimizer'] = not bool(cfg.get('resume_checkpoint'))
        # torch.save exposes a file while writing; wait for the coordinator's
        # atomic stage-completion receipt before deserializing final weights.
        if complete is not None:
            endpoint_identity(stage, cfg, step, checks, details, local_pending)
        histogram_audit(run, cfg, step, checks, details, local_pending)
    if complete is not None:
        checks['training_stage_complete_exit0'] = complete.get('returncode') == 0
        final = stage / 'checkpoints/model_final.pth'
        if final.exists():
            checks['final_checkpoint_matches_training_receipt'] = digest(final) == complete['checkpoint_sha256']
        else:
            local_pending.append(str(final))
    details.update(checks=checks, failed_checks=[key for key, value in checks.items() if not value],
                   pending=local_pending, complete=not local_pending and bool(checks) and all(checks.values()))
    pending.extend(local_pending)
    return details


def audit():
    experiment, study = ROOT / 'experiments/confusion_guided_v1', ROOT / 'runs/confusion_guided_v1'
    pending, checks, arms = [], {}, {}
    origin = required(ROOT / 'experiments/kd_pixel_v2/origin.json', pending)
    old_manifest = required(ROOT / 'runs/kd_parallel_v1/formal/b_relational/study.json', pending)
    if origin is not None:
        checks['original_step0_hash_record_preserved'] = origin['step0_sha256'] == STEP0_SHA
        checks['shared_step0_unchanged'] = digest(origin['step0']) == STEP0_SHA
        checks['archived_baseline_source_unchanged'] = digest(ROOT / 'runs/fixed_baseline_v1/source_snapshot.tar.gz') == ARCHIVE_SHA
    if old_manifest is not None:
        checks['original_shared_warmup_unchanged'] = digest(old_manifest['shared_warmup_checkpoint']) == old_manifest['shared_warmup_sha256']
    for arm in ARMS:
        run, source = study / 'formal' / arm, experiment / arm / 'src'
        arm_pending, arm_checks = [], {}
        manifest = required(run / 'study.json', arm_pending)
        info = {'checks': arm_checks, 'pending': arm_pending, 'stages': []}
        if manifest is not None:
            arm_checks['formal_manifest_verified'] = manifest.get('arm') == arm and manifest.get('smoke') is False
            arm_checks['four_GPUs_correct_assignment'] = manifest['gpus'] == ([0, 1, 2, 3] if arm == 'a_sep' else [4, 5, 6, 7])
            bad, missing = [], []
            recorded = manifest['source_sha256']
            for rel, sha in recorded.items():
                path = source / rel
                if not path.resolve().is_relative_to(source.resolve()):
                    bad.append(rel + ': source path escapes candidate')
                elif not path.is_file():
                    missing.append(rel)
                elif digest(path) != sha:
                    bad.append(rel)
            actual = {str(path.relative_to(source)) for path in source.rglob('*.py') if '__pycache__' not in path.parts}
            arm_checks['frozen_source_verified'] = not bad and not missing and actual == set(recorded)
            info['source_mismatches'], info['missing_source_files'] = bad, missing
            info['extra_source_files'] = sorted(actual - set(recorded))
            if origin is not None:
                arm_checks['original_step0_provenance_matches'] = (manifest['step0']['step0'] == origin['step0']
                    and manifest['step0']['step0_sha256'] == STEP0_SHA
                    and manifest['step0']['source_archive_sha256'] == ARCHIVE_SHA)
            if old_manifest is not None:
                arm_checks['original_warmup_provenance_matches'] = (manifest['shared_warmup_checkpoint'] == old_manifest['shared_warmup_checkpoint']
                    and manifest['shared_warmup_sha256'] == old_manifest['shared_warmup_sha256'])
                info['stages'] = [audit_stage(run, manifest, step, old_manifest, arm_pending) for step in (1, 2)]
            complete = required(run / 'training_complete.json', arm_pending)
            evaluated = required(run / 'evaluation_workers_complete.json', arm_pending)
            if complete is not None:
                arm_checks['all_training_complete_exit0'] = complete.get('stages') == 2 and complete.get('status') == 'training_complete'
            if evaluated is not None:
                arm_checks['all_evaluation_workers_exit0'] = evaluated.get('returncodes') == [0, 0] and evaluated.get('status', 'complete') == 'complete' and not evaluated.get('failure')
            arm_checks['no_recorded_training_failure'] = not (run / 'training_failed.json').exists()
            queue = list((run / 'eval_queue').glob('*.json'))
            if complete is not None:
                expected_jobs = {f'step1_iter{iteration}' for iteration in (4000, 6000, 8000)} | {f'step2_iter{iteration}' for iteration in (2000, 4000, 6000, 8000)}
                arm_checks['all_expected_evaluation_jobs_published'] = {path.stem for path in queue} == expected_jobs
                missing_results = [str(run / 'evaluations' / job.stem / 'result.json') for job in queue
                                   if not (run / 'evaluations' / job.stem / 'result.json').exists()]
                arm_pending.extend(missing_results)
                if not missing_results:
                    arm_checks['all_queued_evaluations_completed'] = True
        info['failed_checks'] = [key for key, value in arm_checks.items() if not value]
        info['complete'] = (manifest is not None and not arm_pending and all(arm_checks.values())
                            and len(info['stages']) == 2 and all(stage['complete'] for stage in info['stages']))
        pending.extend(arm_pending)
        arms[arm] = info
    failed = [key for key, value in checks.items() if not value]
    failed.extend(f'{arm}: {key}' for arm, info in arms.items() for key in info['failed_checks'])
    failed.extend(f'{arm} step{stage["step"]}: {key}' for arm, info in arms.items()
                  for stage in info['stages'] for key in stage['failed_checks'])
    complete = not pending and not failed and len(arms) == 2 and all(info['complete'] for info in arms.values())
    status = 'complete' if complete else 'invalid' if failed else 'incomplete'
    return {'utc': now(), 'status': status, 'shared_provenance_checks': checks, 'arms': arms,
            'pending': sorted(set(pending)), 'failed_checks': failed,
            'all_training_and_endpoint_requirements_complete': complete,
            'device': 'CPU only; no CUDA allocations', 'resource_policy': '8card only; 4card inactive',
            'interpretation': 'Integrity/completeness audit. Accuracy benefit and tradeoffs require a separate comparison; missing ongoing work is incomplete, not a failed experiment.'}


def save_output(path, result):
    path = Path(path)
    if not path.is_absolute() or not path.resolve().is_relative_to(ALLOWED):
        raise RuntimeError('Audit output must remain inside the authorized workspace')
    for item in (path, *path.parents):
        if item == ALLOWED.parent:
            break
        if item.is_symlink() or (item.is_file() and item.stat().st_nlink != 1):
            raise RuntimeError('Refusing linked audit output: ' + str(item))
        if item.exists() and item.stat().st_dev != ALLOWED.stat().st_dev:
            raise RuntimeError('Refusing output through nested mount: ' + str(item))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', help='Optional evidence JSON inside the authorized workspace')
    parser.add_argument('--require-complete', action='store_true', help='Nonzero exit only when final completion is required')
    args = parser.parse_args()
    result = audit()
    if args.output:
        save_output(args.output, result)
    print(json.dumps(result, allow_nan=False))
    if args.require_complete and not result['all_training_and_endpoint_requirements_complete']:
        raise SystemExit(1)
