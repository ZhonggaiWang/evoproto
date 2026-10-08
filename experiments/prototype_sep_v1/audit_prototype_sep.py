"""CPU-only completion and integrity audit for confusion-guided prototype SEP.

Default execution only reads existing evidence and prints JSON. Missing work
is incomplete while the study is running. To save the independent final receipt:
  python -B audit_prototype_sep.py --require-complete --output \
    /ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto/runs/prototype_sep_v1/formal/a_geometry/completion_audit.json

No forward pass, data loader, training, evaluation, CUDA allocation or remote
operation is performed. Integrity does not establish an accuracy improvement.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import inspect
import json
import math
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

import numpy as np
import torch

torch.set_num_threads(2)

ALLOWED = Path('/ML-vePFS/infra_rd/kun/others/wzg')
ROOT = ALLOWED / 'workspace/evoproto'
EXPERIMENT = ROOT / 'experiments/prototype_sep_v1'
STUDY = ROOT / 'runs/prototype_sep_v1'
RUN = STUDY / 'formal/a_geometry'
SOURCE = EXPERIMENT / 'a_geometry/src'
PARENT = ROOT / 'runs/kd_parallel_v1/formal/b_relational'
PARENT_SOURCE = ROOT / 'experiments/kd_parallel_v1/b_relational/src'
RECEIPT = RUN / 'completion_audit.json'
STEP0 = ROOT / 'runs/fixed_baseline_v1/shared/10-5/step0/checkpoints/model_final.pth'
WARMUP = ROOT / 'runs/kd_pixel_v2/10-5/step1/checkpoints/model_iter_2000.pth'
STEP0_SHA = '4f298e14721630cf3c66ba4f6af04bd198ebeaf77adc6b7b07ab8403ad0893b5'
WARMUP_SHA = '4cdc0087e802524b14aa4c4a10a32dd4c20f7484f35525f249965b8d6aa5ca8f'
ARCHIVE_SHA = 'e4a9c503c8967c631f3c53955f516744f645aefdf3dd68e71086ddb387a888d1'
OPTIMIZED_ENDPOINTS = {
    'step1/checkpoints/model_final.pth': '6d3d54769b6a54e1700c7cd352b24e4bb95f4ccf398bc200b74f60795780ab33',
    'step2/checkpoints/model_final.pth': '7c7cc782c72171f18269b638c41d1759d8d03dbe7f7990c5b7c268848ce6da9a',
    'step2/checkpoints/model_iter_8000.pth': 'be237b09fbf6fdb712f245f3c0faaab949a4bd4e0529cab772b96e57201fbf3b',
}
CLASS_NAMES = ('_background_', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
JOBS = {1: (4000, 6000, 8000), 2: (2000, 4000, 6000, 8000)}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def safe_path(path):
    """The same no-link/no-hardlink/no-nested-mount write boundary as runtime."""
    path = Path(path)
    if not path.is_absolute() or not path.resolve().is_relative_to(ALLOWED):
        raise RuntimeError('Path escapes authorized workspace: ' + str(path))
    for item in (path, *path.parents):
        if item == ALLOWED.parent:
            break
        if item.is_symlink() or (item.is_file() and item.stat().st_nlink != 1):
            raise RuntimeError('Unsafe linked path: ' + str(item))
        if item.exists() and item.stat().st_dev != ALLOWED.stat().st_dev:
            raise RuntimeError('Nested mount in output path: ' + str(item))
    return path


def legal(path):
    try:
        safe_path(path)
        return True
    except (RuntimeError, OSError):
        return False


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def required(path, pending):
    path = safe_path(path)
    if not path.is_file():
        pending.append(str(path))
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        pending.append('Not yet readable JSON: ' + str(path))
        return None


def near(a, b, tolerance=1e-9):
    if a is None or b is None:
        return a is None and b is None
    return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a)-float(b)) <= tolerance


def same_state(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same_state(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(same_state(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def complete_optimizer(value):
    return isinstance(value, dict) and bool(value.get('state')) and bool(value.get('param_groups'))


def load_checkpoint(path, pending):
    path = safe_path(path)
    if not path.is_file():
        pending.append(str(path))
        return None
    return torch.load(path, map_location='cpu', weights_only=True, mmap=True)


def source_audit(source, recorded):
    bad, missing = [], []
    for rel, sha in recorded.items():
        path = source / rel
        if not path.resolve().is_relative_to(source.resolve()) or not legal(path):
            bad.append(rel + ': unsafe source path')
        elif not path.is_file():
            missing.append(rel)
        elif digest(path) != sha:
            bad.append(rel)
    actual = {str(path.relative_to(source)) for path in source.rglob('*.py') if '__pycache__' not in path.parts}
    return not bad and not missing and actual == set(recorded), {
        'mismatches': bad, 'missing': missing, 'extra': sorted(actual-set(recorded))}


def hash_check(path, expected, checks, name, pending):
    path = safe_path(path)
    if path.is_file():
        checks[name] = digest(path) == expected
    else:
        pending.append(str(path))


def preservation(manifest, checks, details, pending):
    origin = required(EXPERIMENT / 'origin.json', pending)
    previous = required(PARENT / 'study.json', pending)
    old_origin = required(ROOT / 'experiments/kd_pixel_v2/origin.json', pending)
    snapshot = required(ROOT / 'runs/fixed_baseline_v1/source_snapshot.json', pending)
    checks['candidate_source_freeze'], details['source'] = source_audit(SOURCE, manifest['source_sha256'])
    hash_check(EXPERIMENT / 'run_prototype_sep.py', manifest['runner_sha256'], checks, 'runner_freeze', pending)
    hash_check(STEP0, STEP0_SHA, checks, 'shared_step0_original_SHA_preserved', pending)
    hash_check(WARMUP, WARMUP_SHA, checks, 'shared_step1_warmup_original_SHA_preserved', pending)
    hash_check(ROOT / 'runs/fixed_baseline_v1/source_snapshot.tar.gz', ARCHIVE_SHA,
               checks, 'archived_baseline_source_preserved', pending)
    checks['manifest_original_step0_and_warmup_provenance'] = (
        manifest['step0'] == str(STEP0) and manifest['step0_sha256'] == STEP0_SHA
        and manifest['step1_warmup'] == str(WARMUP) and manifest['step1_warmup_sha256'] == WARMUP_SHA)
    for rel, sha in OPTIMIZED_ENDPOINTS.items():
        hash_check(PARENT / '10-5' / rel, sha, checks, 'original_optimized_KD_' + rel.split('/')[0] + '_' + Path(rel).stem + '_preserved', pending)
    if origin is not None:
        checks['origin_uses_original_optimized_KD_source'] = origin['optimized_kd_source'] == str(PARENT_SOURCE)
        checks['original_optimized_KD_source_preserved'], details['original_source'] = source_audit(PARENT_SOURCE, origin['optimized_kd_source_sha256'])
        prior, current = origin['optimized_kd_source_sha256'], manifest['source_sha256']
        modified = {rel for rel in set(prior) & set(current) if prior[rel] != current[rel]}
        added = set(current)-set(prior)
        checks['only_geometry_integration_source_changes'] = (
            not (set(prior)-set(current))
            and modified <= {'continual/Trainer.py', 'scripts/dist_train_voc_seg_neg.py', 'evaluate_kd.py'}
            and added == {'model/online_directed_confusion.py', 'model/directed_pair_selector.py',
                          'model/geometry_pair_selector.py', 'model/confusion_prototype_sep.py'})
        details['source_changes'] = {'modified': sorted(modified), 'added': sorted(added)}
        checks['origin_records_no_baseline_retraining_or_seed_search'] = origin.get('baseline_retraining') is False and origin.get('seed_search') is False
    if previous is not None:
        checks['warmup_matches_original_optimized_KD_manifest'] = (
            previous['shared_warmup_checkpoint'] == str(WARMUP) and previous['shared_warmup_sha256'] == WARMUP_SHA)
        if origin is not None:
            checks['origin_source_SHA_records_match_frozen_original_study'] = (
                origin['optimized_kd_source_sha256'] == previous['source_sha256'])
    if snapshot is not None and old_origin is not None:
        checks['baseline_snapshot_records_preserved'] = (
            snapshot['archive_sha256'] == ARCHIVE_SHA and snapshot['files_sha256'] == old_origin['original_source_hashes']
            and snapshot['file_count'] == len(snapshot['files_sha256']) and old_origin['step0_sha256'] == STEP0_SHA)
    kd, original_kd = SOURCE / 'model/pixel_kd.py', PARENT_SOURCE / 'model/pixel_kd.py'
    checks['optimized_pixel_KD_byte_identical'] = kd.is_file() and original_kd.is_file() and digest(kd) == digest(original_kd)
    if checks['optimized_pixel_KD_byte_identical']:
        details['pixel_KD_sha256'] = digest(kd)
    warmup = load_checkpoint(WARMUP, pending)
    if warmup is not None:
        checks['warmup_iteration2000_model_and_full_optimizer'] = (
            warmup.get('iteration') == 2000 and isinstance(warmup.get('model_state'), dict)
            and bool(warmup['model_state']) and complete_optimizer(warmup.get('optimizer_state')))
        checks['warmup_has_no_online_observer_or_geometry_selector'] = (
            warmup.get('online_confusion_state') is None and warmup.get('geometry_selector_state') is None)
        del warmup


def preflight(manifest, checks, details, pending):
    receipt = required(STUDY / 'preflight.json', pending)
    probe = required(STUDY / 'gradient_probe.json', pending)
    smoke = STUDY / 'smoke/a_geometry'
    status = required(smoke / 'status.json', pending)
    done = required(smoke / 'training_complete.json', pending)
    evaluated = required(smoke / 'evaluation_workers_complete.json', pending)
    smoke_manifest = required(smoke / 'study.json', pending)
    if receipt is not None:
        checks['preflight_passed_for_same_frozen_source'] = receipt.get('passed') is True and receipt['source_sha256'] == manifest['source_sha256']
        checks['preflight_25_prototype_and_16_selector_tests'] = all(
            receipt['unit_tests'][key].get('passed') is True and receipt['unit_tests'][key].get('count') == count
            for key, count in [('prototype_sep', 25), ('geometry_selector', 16)])
        for key, filename in [('prototype_sep', 'test_confusion_prototype_sep.py'), ('geometry_selector', 'test_geometry_selector.py')]:
            hash_check(EXPERIMENT / filename, receipt['unit_tests'][key]['source_sha256'],
                       checks, key + '_test_source_SHA_matches_preflight', pending)
        checks['preflight_pixel_KD_SHA_verified'] = digest(SOURCE / 'model/pixel_kd.py') == receipt['original_kd_byte_identical_sha256']
        if probe is not None:
            checks['preflight_gradient_probe_SHA_verified'] = digest(STUDY / 'gradient_probe.json') == receipt['gradient_probe_sha256']
        details['preflight'] = receipt
    if probe is not None:
        checks['actual_five_image_gradient_probe_passed'] = probe.get('passed') is True and len(probe.get('records', [])) == 5
        checks['probe_references_saved_original_optimized_KD'] = (
            probe['student_sha256'] == OPTIMIZED_ENDPOINTS['step2/checkpoints/model_final.pth']
            and probe['teacher_sha256'] == OPTIMIZED_ENDPOINTS['step1/checkpoints/model_final.pth'])
        for key in ['student', 'teacher', 'observer']:
            path_key = key + '_checkpoint'
            sha_key = 'observer_checkpoint_sha256' if key == 'observer' else key + '_sha256'
            hash_check(probe[path_key], probe[sha_key], checks, 'probe_' + key + '_checkpoint_SHA_verified', pending)
        records = probe.get('records', [])
        checks['probe_SEP_has_real_new_prototype_gradients'] = bool(records) and all(
            bool(row['nonzero_parameter_gradient_norms']['sep']) and all(
                name.startswith('decoder.class_prototypes.2.') and math.isfinite(value) and value > 0
                for name, value in row['nonzero_parameter_gradient_norms']['sep'].items()) for row in records)
        checks['probe_KD_has_no_direct_prototype_gradient'] = bool(records) and all(
            not any(name.startswith('decoder.class_prototypes.') for name in row['nonzero_parameter_gradient_norms']['kd'])
            and row['relationships']['sep_kd']['shared_nonzero_parameter_names'] == [] for row in records)
    if status is not None and done is not None and evaluated is not None:
        checks['two_stage_smoke_training_and_four_evaluators_complete'] = (
            status['status'] == done['status'] == 'complete' and done['stages'] == 2
            and evaluated['returncodes'] == [0]*4 and not (smoke / 'failure.json').exists())
    if smoke_manifest is not None:
        checks['smoke_used_same_source_and_8x1_global8'] = (
            smoke_manifest['source_sha256'] == manifest['source_sha256']
            and smoke_manifest['train_gpus'] == list(range(8)) and smoke_manifest['global_batch'] == 8
            and smoke_manifest['batch_per_gpu'] == 1)
    for step, iteration in [(1, 2005), (2, 5)]:
        launch = required(smoke / f'10-5/step{step}/launch.json', pending)
        completed = required(smoke / f'10-5/step{step}/training_complete.json', pending)
        result = required(smoke / f'evaluations/step{step}_iter{iteration}/result.json', pending)
        if launch is not None and completed is not None and result is not None:
            checks[f'smoke_step{step}_exit0_and_16_images'] = (
                launch.get('returncode') == completed.get('returncode') == 0
                and result['step'] == step and result['iteration'] == iteration and result['images'] == 16)


def module_states(observer_state, selector_state, cfg, step):
    """Strictly reconstruct frozen CPU modules; do not call their update methods."""
    from model.online_directed_confusion import OnlineDirectedConfusion
    from model.geometry_pair_selector import GeometryPairSelector
    classes, old = (16, 10) if step == 1 else (21, 15)
    observer = OnlineDirectedConfusion(classes, stage=step, momentum=cfg['confusion_momentum'],
        high_threshold=cfg['high_thre'], low_threshold=cfg['low_thre'])
    selector = GeometryPairSelector(classes, old_classes=old, stage=step,
        refresh_interval=cfg['pair_refresh_interval'], min_row_images=cfg['pair_min_row_images'],
        min_pair_images=cfg['pair_min_pair_images'], min_rate=cfg['pair_min_rate'],
        min_updates=cfg['pair_min_updates'], ramp_updates=cfg['pair_ramp_updates'],
        max_stale_updates=cfg['pair_max_stale_updates'])
    expected_observer, expected_selector = observer.get_extra_state(), selector.get_extra_state()
    info = {'observer_expected_metadata': expected_observer, 'selector_expected_metadata': expected_selector}
    try:
        observer.load_state_dict(observer_state, strict=True)
        selector.load_state_dict(selector_state, strict=True)
    except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
        info['restore_error'] = repr(exc)
        return False, info
    target, rates = selector.targets, selector.selected_rates
    ids = torch.arange(classes)
    directions = (target == -1) | ((target > 0) & (target < classes) & (target != ids)
                                  & ((ids > old) | (target > old)))
    buffers = list(observer.buffers()) + list(selector.buffers())
    finite = all(not value.is_floating_point() or bool(torch.isfinite(value).all()) for value in buffers)
    nonnegative_counts = all(bool((getattr(observer, name) >= 0).all()) for name in [
        'counts', 'accepted_counts', 'broad_counts', 'ema_counts', 'broad_ema_counts',
        'pair_image_observations', 'broad_pair_image_observations', 'image_observations',
        'ema_image_observations', 'broad_image_observations'])
    exact = same_state(observer_state, observer.state_dict()) and same_state(selector_state, selector.state_dict())
    valid = (exact and finite and nonnegative_counts and expected_observer['schema'] == 2
        and expected_selector['schema'] == 3 and bool(directions.all()) and int(target[0]) == -1
        and bool(((rates >= 0) & (rates <= 1)).all()) and bool((rates[target == -1] == 0).all()))
    info.update(observer_metadata=observer_state.get('_extra_state'), selector_metadata=selector_state.get('_extra_state'),
        strict_restored_state_tensor_exact=exact, finite_buffers=finite, nonnegative_counts=nonnegative_counts,
        valid_geometry_directions=bool(directions.all()), targets=target.tolist(), selected_rates=rates.tolist(),
        updates=int(observer.updates), seen_images=int(observer.seen_images),
        last_refresh_iteration=int(selector.last_refresh_iteration))
    return valid, info


def endpoint(stage, cfg, step, checks, details, pending):
    final, evaluated = stage / 'checkpoints/model_final.pth', stage / 'checkpoints/model_iter_8000.pth'
    a, b = load_checkpoint(final, pending), load_checkpoint(evaluated, pending)
    if a is None or b is None:
        return
    checks['final_and_evaluated_iteration8000'] = a.get('iteration') == b.get('iteration') == 8000
    checks['final_model_equals_evaluated_model_tensor_exact'] = (
        isinstance(a.get('model_state'), dict) and bool(a['model_state']) and same_state(a['model_state'], b.get('model_state')))
    checks['final_model_finite'] = all(not value.is_floating_point() or bool(torch.isfinite(value).all())
        for value in a.get('model_state', {}).values() if isinstance(value, torch.Tensor))
    checks['final_full_optimizer_equals_evaluated_tensor_exact'] = (
        complete_optimizer(a.get('optimizer_state')) and complete_optimizer(b.get('optimizer_state'))
        and same_state(a['optimizer_state'], b['optimizer_state']))
    va, ia = module_states(a.get('online_confusion_state'), a.get('geometry_selector_state'), cfg, step)
    vb, ib = module_states(b.get('online_confusion_state'), b.get('geometry_selector_state'), cfg, step)
    checks['observer_and_schema3_selector_saved_and_strict_restorable'] = va and vb
    checks['final_observer_equals_evaluated_tensor_exact'] = va and vb and same_state(a['online_confusion_state'], b['online_confusion_state'])
    checks['final_geometry_selector_equals_evaluated_tensor_exact'] = va and vb and same_state(a['geometry_selector_state'], b['geometry_selector_state'])
    if va:
        updates = 6000 if step == 1 else 8000
        checks['observer_updates_and_images_match_executed_budget'] = ia['updates'] == updates and ia['seen_images'] == updates*8
        checks['selector_refresh_within_executed_stage'] = (2000 if step == 1 else 0) < ia['last_refresh_iteration'] <= 8000
    details['endpoint'] = {'final_sha256': digest(final), 'evaluated_sha256': digest(evaluated),
        'model_tensor_count': sum(isinstance(v, torch.Tensor) for v in a.get('model_state', {}).values()),
        'final_online_state': ia, 'evaluated_online_state': ib}
    del a, b


def mechanism(stage, cfg, step, checks, details, pending, finished):
    def read_rows(path):
        if not safe_path(path).is_file():
            pending.append(str(path))
            return []
        rows = []
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pending.append('Not yet complete metric line: ' + str(path))
        return rows
    geometry = read_rows(stage / 'geometry_metrics.jsonl')
    kd = read_rows(stage / 'kd_metrics.jsonl')
    if not geometry or not kd:
        pending.append(str(stage) + ': geometry/KD metric logs not yet populated')
        return
    classes, old, start = (16, 10, 2000) if step == 1 else (21, 15, 0)
    expected_iterations = list(range(start+cfg['log_iters'], 8001, cfg['log_iters']))
    checks['geometry_and_KD_log_iterations_match'] = [r['iteration'] for r in geometry] == [r['iteration'] for r in kd]
    if finished:
        checks['all_expected_logged_batches_to8000'] = [r['iteration'] for r in geometry] == expected_iterations
    checks['logged_legacy_KD_weight_and_global_batch'] = all(r['step'] == step and r['weight'] == .1 and r['global_batch'] == 8 for r in kd)
    checks['no_conditional_pairKD_in_legacy_KD_log'] = all(not any(key.startswith('kd_pair_') for key in r) for r in kd)
    failures, active = [], []
    for row in geometry:
        try:
            iteration = row['iteration']
            selector, entries = row['selector'], row['pairs']
            targets = selector['targets']
            prior_updates = iteration-start-1
            expected_ramp = min(1., max(0., (prior_updates-cfg['pair_min_updates'])/cfg['pair_ramp_updates']))
            assert row['step'] == step and row['old_classes'] == old and row['total_classes_including_background'] == classes
            assert row['margin'] == 0. and row['weight'] == .1 and row['active'] is (iteration > cfg['loss_warmup_iters'])
            assert near(row['ramp'], expected_ramp) and 0 <= row['ramp'] <= 1
            assert selector['schema'] == 3 and selector['classes'] == classes and selector['stage'] == step and selector['old_classes'] == old
            assert len(targets) == classes and targets[0] == -1
            directions = [[i, j] for i, j in enumerate(targets) if j != -1]
            assert all(0 < i < classes and 0 < j < classes and i != j and (i > old or j > old) for i, j in directions)
            expected_pairs = sorted({tuple(sorted(direction)) for direction in directions})
            assert [tuple(entry['class_ids']) for entry in entries] == expected_pairs
            assert row['pair_count'] == len(entries) and row['selected_valid_directions'] == len(directions)
            assert row['duplicate_reverse_directions_removed'] == len(directions)-len(entries)
            assert row['skipped_background_directions'] == row['skipped_self_directions'] == row['skipped_old_old_directions'] == 0
            assert row['skipped_missing_directions'] == classes-len(directions)
            old_new, new_new, hinge_active = 0, 0, 0
            for entry in entries:
                first, second = entry['class_ids']
                is_old = first <= old
                assert entry['pair_type'] == ('old_new' if is_old else 'new_new')
                assert entry['old_endpoint_detached'] == (first if is_old else None)
                assert entry['gradient_class_ids'] == ([second] if is_old else [first, second])
                selected_directions = [direction for direction in directions if tuple(sorted(direction)) == (first, second)]
                assert entry['selected_directions'] == selected_directions
                cosine = entry['cosine_similarity']
                assert math.isfinite(cosine) and -1 <= cosine <= 1 and entry['margin'] == 0.
                assert entry['active'] is (cosine > 0.) and near(entry['pair_loss'], max(cosine, 0.)**2, 5e-7)
                old_new += int(is_old); new_new += int(not is_old); hinge_active += int(entry['active'])
            assert row['old_new_pair_count'] == old_new and row['new_new_pair_count'] == new_new and row['active_pair_count'] == hinge_active
            mean = sum(entry['pair_loss'] for entry in entries)/len(entries) if entries else 0.
            assert near(row['prototype_sep'], mean, 5e-7)
            assert row['old_reference'] == 'current_student_prototype_detached_in_SEP_only'
            assert row['objective'] == 'mean_over_all_selected_unique_pairs_of_relu_cosine_minus_margin_squared'
            assert row['rate_usage'] == 'Directions select unique hard pairs; confusion rates are never loss weights.'
            if row['active'] and row['ramp'] > 0 and hinge_active and row['prototype_sep'] > 0:
                active.append(row)
        except (AssertionError, KeyError, TypeError, ValueError) as exc:
            failures.append({'iteration': row.get('iteration'), 'error': repr(exc)})
    checks['logged_pair_domains_dedup_hinge_mean_detach_and_fixed_strength'] = not failures
    if active:
        checks['supported_prototype_SEP_actually_activated'] = True
    elif finished:
        checks['supported_prototype_SEP_actually_activated'] = False
    else:
        pending.append(str(stage) + ': supported prototype SEP activation pending')
    details['mechanism'] = {'logged_batches': len(geometry), 'active_logged_batches': len(active),
        'maximum_eligible_unique_pairs': max(r['pair_count'] for r in geometry),
        'maximum_active_unique_pairs': max(r['active_pair_count'] for r in geometry),
        'logged_reverse_directions_removed': sum(r['duplicate_reverse_directions_removed'] for r in geometry),
        'semantic_failures': failures, 'fixed_weight': .1, 'margin': 0.,
        'note': 'Sampled batch diagnostics; gradient endpoint IDs describe allowed differentiation, not measured nonzero norms.'}


def evaluation(cfg, step, iteration, pending):
    checks, details = {}, {'step': step, 'iteration': iteration}
    name = f'step{step}_iter{iteration}'
    part = RUN / 'evaluations' / name
    job = required(RUN / 'eval_queue' / (name + '.json'), pending)
    result = required(part / 'result.json', pending)
    if job is None or result is None:
        return {'checks': checks, 'details': details}
    checkpoint = RUN / f'10-5/step{step}/checkpoints/model_iter_{iteration}.pth'
    checks['correct_checkpoint_path_and_source_config'] = legal(job['checkpoint']) and Path(job['checkpoint']) == checkpoint and job['config'] == cfg
    checks['correct_stage_and_iteration'] = job['step'] == result['step'] == step and job['iteration'] == result['iteration'] == iteration
    if checkpoint.is_file():
        checks['checkpoint_SHA_matches_job_and_result'] = digest(checkpoint) == job['checkpoint_sha256'] == result['checkpoint_sha256']
    else:
        pending.append(str(checkpoint))
    classes, images, old = (16, 1240, 10) if step == 1 else (21, 1449, 15)
    hist = np.asarray(result['histogram'])
    hist_valid = hist.shape == (classes, classes) and np.issubdtype(hist.dtype, np.integer) and bool((hist >= 0).all())
    checks['histogram_nonnegative_integer_shape'] = hist_valid
    checks['full_validation_image_count'] = result['images'] == images
    reference = required(PARENT / f'evaluations/step{step}_iter8000/result.json', pending)
    if hist_valid:
        if reference is not None:
            rh = np.asarray(reference['histogram'], dtype=np.int64)
            checks['original_GT_row_totals_match_optimized_KD'] = rh.shape == hist.shape and np.array_equal(hist.sum(1), rh.sum(1))
        union = hist.sum(1)+hist.sum(0)-hist.diagonal()
        iou = np.divide(hist.diagonal(), union, out=np.full(classes, np.nan), where=union > 0)*100
        def mean(values):
            return float(np.nanmean(values)) if np.isfinite(values).any() else None
        computed = {'all_miou': mean(iou), 'foreground_miou': mean(iou[1:]),
            'previous_foreground_miou': mean(iou[1:old+1]), 'current_foreground_miou': mean(iou[old+1:]),
            'old_initial10_miou': mean(iou[1:11]), 'new_since_initial_miou': mean(iou[11:])}
        checks['reported_mIoU_matches_histogram'] = all(near(result.get(key), value) for key, value in computed.items())
        names = CLASS_NAMES[:classes]
        checks['reported_class_IoU_matches_histogram'] = set(result['class_iou']) == set(names) and all(
            near(result['class_iou'].get(name), float(iou[i]) if np.isfinite(iou[i]) else None) for i, name in enumerate(names))
        counters = {'old_gt_to_new_pixels': int(hist[1:old+1, old+1:].sum()),
            'old_gt_to_background_pixels': int(hist[1:old+1, 0].sum()), 'old_gt_pixels': int(hist[1:old+1].sum()),
            'new_gt_to_old_pixels': int(hist[old+1:, 1:old+1].sum()), 'new_gt_pixels': int(hist[old+1:].sum())}
        checks['reported_confusion_counters_match_histogram'] = all(result.get(key) == value for key, value in counters.items())
        details.update(metrics=computed, GT_row_totals=hist.sum(1).tolist(), checkpoint_sha256=result['checkpoint_sha256'])
    shards = [required(part / f'rank{rank}.json', pending) for rank in range(4)]
    if all(shard is not None for shard in shards):
        seen = [name for shard in shards for name in shard['images']]
        split = Path(cfg['list_folder']) / 'incremental_split' / f'val_{cfg["task"]}_step_{step+1}.txt'
        expected = split.read_text().splitlines()
        checks['four_shards_disjoint_and_full_original_split'] = len(seen) == len(set(seen)) == len(expected) == images and set(seen) == set(expected)
        checks['four_shards_provenance'] = all(shard['rank'] == rank and shard['shards'] == 4 and shard['step'] == step
            and shard['iteration'] == iteration and shard['checkpoint_sha256'] == result['checkpoint_sha256'] for rank, shard in enumerate(shards))
        shard_hist = [np.asarray(shard['histogram']) for shard in shards]
        valid_parts = all(h.shape == (classes, classes) and np.issubdtype(h.dtype, np.integer) and bool((h >= 0).all()) for h in shard_hist)
        checks['merged_histogram_exactly_matches_four_shards'] = hist_valid and valid_parts and np.array_equal(np.sum(shard_hist, axis=0), hist)
        details['shard_image_counts'] = [len(shard['images']) for shard in shards]
    checks['no_shard_error_records'] = not list(part.glob('error_rank*.json'))
    return {'checks': checks, 'details': details}


def audit_stage(manifest, step, pending):
    stage = RUN / f'10-5/step{step}'
    checks, details, local_pending = {}, {'step': step}, []
    cfg = required(stage / 'config.json', local_pending)
    launch = required(stage / 'launch.json', local_pending)
    done = required(stage / 'training_complete.json', local_pending)
    teacher = STEP0 if step == 1 else RUN / '10-5/step1/checkpoints/model_final.pth'
    if launch is not None:
        if 'returncode' in launch:
            checks['trainer_launch_exit0'] = launch['returncode'] == 0
        else:
            local_pending.append(str(stage / 'launch.json') + ': trainer still running')
        command = launch['command']
        checks['eight_rank_training_command'] = '--nproc_per_node=8' in command and str(SOURCE / 'scripts/dist_train_voc_seg_neg.py') in command
        checks['launch_teacher_path_and_SHA'] = Path(launch['teacher']) == teacher and legal(teacher) and teacher.is_file() and digest(teacher) == launch['teacher_sha256']
    if cfg is not None:
        expected = dict(manifest['common_config'])
        expected.update(step=step, spg=1, max_iters=8000, warmup_iters=2000, loss_warmup_iters=2000,
            seed=0, train_limit=0, val_limit=0, async_eval=True, w_pixel_kd=.1, kd_temperature=2.,
            w_proto_kd=0., w_proto_sep=0., w_geometry_sep=.1, proto_margin=0., ald=False,
            eval_iters=2000, log_iters=50, save_ckpt=True, num_workers=2, task='10-5', dataset='voc',
            confusion_reweight=False, confusion_momentum=.98, pair_refresh_interval=50,
            pair_min_row_images=8, pair_min_pair_images=3, pair_min_rate=.01,
            pair_min_updates=100, pair_ramp_updates=200, pair_max_stale_updates=200)
        mismatch = {key: {'expected': value, 'actual': cfg.get(key)} for key, value in expected.items() if cfg.get(key) != value}
        checks['fixed_stage_budget_optimizer_KD_and_geometry_config'] = not mismatch
        details['config_mismatches'] = mismatch
        checks['eight_times_one_global_batch8'] = cfg['spg'] == manifest['batch_per_gpu'] == 1 and manifest['train_gpus'] == list(range(8)) and manifest['global_batch'] == 8
        checks['stage_work_and_teacher_paths_legal'] = (Path(cfg['work_dir']) == stage and Path(cfg['ckpt_dir']) == stage / 'checkpoints'
            and Path(cfg['prev_checkpoint']) == teacher and all(legal(cfg[key]) for key in ['work_dir', 'ckpt_dir', 'pred_dir', 'prev_checkpoint']))
        checks['stage1_only_shared_warmup_resume'] = cfg.get('resume_checkpoint', '') == (str(WARMUP) if step == 1 else '')
        if step == 1:
            checks['resume_path_legal'] = legal(cfg['resume_checkpoint'])
        log_path = stage / 'train.log'
        if log_path.is_file():
            text = log_path.read_text()
            checks['expected_teacher_load_logged'] = str(teacher) in text
            checks['eight_GPUs_one_sample_logged'] = 'Total gpus: 8, samples per gpu: 1' in text
            if step == 1:
                checks['student_and_full_optimizer_resume_at2000_logged'] = 'Resume student AND optimizer at iteration 2000' in text and str(WARMUP) in text
            else:
                checks['stage2_no_iteration2000_resume_logged'] = 'Resume student AND optimizer' not in text
        else:
            local_pending.append(str(log_path))
        parent_cfg = required(PARENT / f'10-5/step{step}/config.json', local_pending)
        if parent_cfg is not None:
            runtime = {'local_rank', 'step', 'work_dir', 'prev_checkpoint', 'resume_checkpoint', 'ckpt_dir', 'pred_dir'}
            before = {key: value for key, value in parent_cfg.items() if key not in runtime}
            after = {key: value for key, value in cfg.items() if key not in runtime}
            changes = {key: {'optimized_KD': before.get(key), 'prototype_SEP': after.get(key)}
                for key in sorted(set(before) | set(after)) if before.get(key) != after.get(key)}
            allowed = {'spg', 'num_workers', 'w_geometry_sep', 'confusion_momentum', 'pair_refresh_interval',
                'pair_min_row_images', 'pair_min_pair_images', 'pair_min_rate', 'pair_min_updates', 'pair_ramp_updates', 'pair_max_stale_updates'}
            checks['only_explicit_geometry_and_batch_layout_changes_from_optimized_KD'] = set(changes) <= allowed
            details['changes_from_original_optimized_KD'] = changes
        mechanism(stage, cfg, step, checks, details, local_pending, done is not None)
        details['evaluations'] = [evaluation(cfg, step, iteration, local_pending) for iteration in JOBS[step]]
        if done is not None:
            endpoint(stage, cfg, step, checks, details, local_pending)
    if done is not None:
        checks['training_stage_complete_exit0'] = done.get('returncode') == 0
        hash_check(stage / 'checkpoints/model_final.pth', done['checkpoint_sha256'], checks,
            'final_checkpoint_matches_training_receipt', local_pending)
    failures = [key for key, value in checks.items() if not value]
    failures.extend(f'evaluation{item["details"]["iteration"]}: {key}' for item in details.get('evaluations', []) for key, value in item['checks'].items() if not value)
    pending.extend(local_pending)
    return {'checks': checks, 'details': details, 'failed_checks': failures,
        'pending': local_pending, 'complete': cfg is not None and done is not None and not failures and not local_pending}


def audit():
    pending, checks, details = [], {}, {}
    manifest = required(RUN / 'study.json', pending)
    completed = required(RUN / 'training_complete.json', pending)
    evaluated = required(RUN / 'evaluation_workers_complete.json', pending)
    status = required(RUN / 'status.json', pending)
    checks['all_candidate_files_remain_in_authorized_work_area'] = all(legal(path) for root in [SOURCE, RUN]
        for path in [root, *root.rglob('*')])
    caches = [ROOT / '.runtime/kd_pixel_v1/8card' / sub for sub in ['tmp', 'cache', 'cache/hf', 'cache/torch', 'mpl', 'cuda', 'triton']]
    caches += [ROOT / '.runtime/prototype_sep_v1/8card' / sub for sub in ['cache', 'cache/huggingface', 'cache/torch', 'cache/matplotlib', 'cache/cuda']]
    caches += [ROOT / '.runtime/prototype_sep_v1/8card_eval' / sub for sub in ['cache', 'mpl', 'cuda']]
    caches += [ALLOWED / '.kd8tmp', ROOT / 'pretrained']
    checks['declared_training_and_evaluation_cache_paths_in_A'] = all(legal(path) for path in caches)
    details['cache_paths_checked'] = [str(path) for path in caches]
    checks['no_recorded_coordinator_or_training_failure'] = not any((RUN / name).exists() for name in ['failure.json', 'training_failed.json', 'evaluation_failed.json'])
    stages = []
    if manifest is not None:
        checks['formal_geometry_manifest_verified'] = manifest.get('arm') == 'a_geometry' and manifest.get('smoke') is False
        preservation(manifest, checks, details, pending)
        preflight(manifest, checks, details, pending)
        # Import only frozen tensor/selector/loss modules after verifying source.
        # Their constructors/load_state_dict are CPU-only and do not create files.
        if checks['candidate_source_freeze']:
            sys.path.insert(0, str(SOURCE))
            from model.confusion_prototype_sep import confusion_prototype_sep_loss
            signature = inspect.signature(confusion_prototype_sep_loss)
            checks['prototype_loss_API_has_no_GT_or_rate_weight_input'] = list(signature.parameters) == ['prototypes', 'pair_targets', 'old_classes', 'margin']
            stages = [audit_stage(manifest, step, pending) for step in (1, 2)]
    if completed is not None:
        checks['two_stage_training_complete'] = completed.get('status') == 'complete' and completed.get('stages') == 2
        expected_jobs = {f'step{step}_iter{iteration}' for step, iterations in JOBS.items() for iteration in iterations}
        actual_jobs = {path.stem for path in (RUN / 'eval_queue').glob('*.json')}
        checks['exactly_seven_expected_evaluation_queue_jobs'] = actual_jobs == expected_jobs and len(actual_jobs) == 7
        details['evaluation_queue'] = sorted(actual_jobs)
    if evaluated is not None:
        checks['all_four_evaluator_returncodes_zero'] = evaluated.get('returncodes') == [0]*4
        hosts = []
        for rank in range(4):
            process = required(RUN / f'evaluator_rank{rank}.process.json', pending)
            if process is not None:
                command = process['command']
                hosts.append(process['host'])
                checks[f'evaluator_rank{rank}_exit0_and_four_shard_command'] = (
                    process.get('returncode') == 0 and command[command.index('--rank')+1] == str(rank)
                    and command[command.index('--shards')+1] == '4' and str(SOURCE / 'evaluate_kd.py') in command
                    and command[command.index('--run')+1] == str(RUN))
        if len(hosts) == 4:
            launches = [required(RUN / f'10-5/step{step}/launch.json', pending) for step in (1, 2)]
            if all(launch is not None for launch in launches):
                checks['both_train_stages_and_four_evaluators_on_same_8card_host'] = len(set(hosts+[launch['host'] for launch in launches])) == 1
    if status is not None:
        if completed is not None and evaluated is not None:
            checks['coordinator_status_complete'] = status.get('status') == 'complete'
        elif status.get('status') == 'failed':
            checks['coordinator_not_failed'] = False
    failed = [key for key, value in checks.items() if not value]
    failed.extend(f'step{index+1}: {key}' for index, stage in enumerate(stages) for key in stage['failed_checks'])
    complete = (not pending and not failed and len(stages) == 2 and all(stage['complete'] for stage in stages)
        and all(record is not None for record in [manifest, completed, evaluated, status]))
    return {'utc': now(), 'status': 'complete' if complete else 'invalid' if failed else 'incomplete',
        'arm': 'a_geometry', 'checks': checks, 'stages': stages, 'details': details,
        'failed_checks': failed, 'pending': sorted(set(pending)),
        'all_training_and_endpoint_requirements_complete': complete,
        'device': 'CPU only; no CUDA allocation, forward pass or evaluation',
        'budget': 'step1 shared2000 then6000 new updates; step2 full8000; 8x1 global8; seven four-shard evaluation jobs',
        'mechanism': 'Geometry only on supported foreground old-new/new-new unordered pairs; old endpoint detached in SEP; squared cosine hinge margin0; lambda0.1; rates rank only; original optimized KD byte-identical',
        'interpretation': 'Integrity/completeness evidence, not an accuracy benefit claim. Missing ongoing work remains incomplete.'}


def save_output(path, result):
    path = safe_path(path)
    if path != RECEIPT:
        raise RuntimeError('Use only independent a_geometry/completion_audit.json; preserve prior A/B/C/D receipts')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = safe_path(path.with_name(path.name + f'.{os.getpid()}.tmp'))
    with tmp.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--output', help='Only formal/a_geometry/completion_audit.json is accepted')
    args = parser.parse_args()
    result = audit()
    if args.output:
        save_output(args.output, result)
    print(json.dumps(result, allow_nan=False))
    if args.require_complete and not result['all_training_and_endpoint_requirements_complete']:
        raise SystemExit(1)
