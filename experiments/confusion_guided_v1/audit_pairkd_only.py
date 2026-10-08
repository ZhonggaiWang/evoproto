"""CPU-only integrity audit of D: online pair KD, zero external SEP.

D inherits the original optimized KD's stage1 and stage2 iteration2000 model
and optimizer. Observer/selector start fresh, so only the 6000 new updates
and 48000 new image exposures belong to D's confusion state.

Read-only by default. Final independent receipt:
  python -B audit_pairkd_only.py --require-complete --output \
    /ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto/runs/confusion_guided_v1/pairkd_only_completion_audit.json
"""
from pathlib import Path
import argparse
import json
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

ALLOWED = Path('/ML-vePFS/infra_rd/kun/others/wzg')
ROOT = ALLOWED / 'workspace/evoproto'
EXPERIMENT = ROOT / 'experiments/confusion_guided_v1'
STUDY = ROOT / 'runs/confusion_guided_v1'
RUN = STUDY / 'formal/d_pairkd_only'
PARENT = ROOT / 'runs/kd_parallel_v1/formal/b_relational'
PARENT_SOURCE = ROOT / 'experiments/kd_parallel_v1/b_relational/src'
SOURCE = EXPERIMENT / 'd_pairkd_only/src'
STAGE = RUN / '10-5/step2'
RECEIPT = STUDY / 'pairkd_only_completion_audit.json'
TEACHER_SHA = '6d3d54769b6a54e1700c7cd352b24e4bb95f4ccf398bc200b74f60795780ab33'
WARMUP_SHA = 'feb5b228b05934e5acb0e861c49f16640f13219ce7ab477aa66449b78520f153'

sys.path.insert(0, str(EXPERIMENT))
from audit_confusion_guided import (CLASS_NAMES, STEP0_SHA, ARCHIVE_SHA, digest, required,
                                   now, same_state, state_metadata, near, save_output)
import numpy as np
import torch

torch.set_num_threads(2)


def legal_path(path):
    path = Path(path)
    if not path.is_absolute() or not path.resolve().is_relative_to(ALLOWED):
        return False
    for item in (path, *path.parents):
        if item == ALLOWED.parent:
            break
        if item.is_symlink() or (item.is_file() and item.stat().st_nlink != 1):
            return False
        if item.exists() and item.stat().st_dev != ALLOWED.stat().st_dev:
            return False
    return True


def source_matches(source, recorded):
    bad, missing = [], []
    for rel, sha in recorded.items():
        path = source / rel
        if not path.resolve().is_relative_to(source.resolve()) or not legal_path(path):
            bad.append(rel + ': unsafe source path')
        elif not path.is_file():
            missing.append(rel)
        elif digest(path) != sha:
            bad.append(rel)
    actual = {str(path.relative_to(source)) for path in source.rglob('*.py') if '__pycache__' not in path.parts}
    return not bad and not missing and actual == set(recorded), {'mismatches': bad, 'missing': missing, 'extra': sorted(actual - set(recorded))}


def preservation_and_sources(manifest, checks, details, pending):
    original = required(PARENT / 'study.json', pending)
    copied = required(STUDY / 'formal/b_pairkd/study.json', pending)
    snapshot = required(ROOT / 'runs/fixed_baseline_v1/source_snapshot.json', pending)
    origin = required(ROOT / 'experiments/kd_pixel_v2/origin.json', pending)
    checks['D_source_freeze_verified'], details['D_source'] = source_matches(SOURCE, manifest['source_sha256'])
    runner = EXPERIMENT / 'run_pairkd_only.py'
    checks['D_runner_freeze_verified'] = runner.is_file() and legal_path(runner) and digest(runner) == manifest['runner_sha256']
    if copied is not None:
        prior = copied['source_sha256']
        changes = {rel for rel in set(prior) & set(manifest['source_sha256'])
                   if prior[rel] != manifest['source_sha256'][rel]}
        permitted = {'continual/Trainer.py', 'model/directed_pair_selector.py'}
        checks['D_only_expected_domain_changes_from_frozen_B'] = (changes <= permitted
            and set(manifest['source_sha256']) == set(prior))
        details['D_domain_source_changes_from_B'] = sorted(changes)
    if original is not None:
        checks['original_optimized_KD_source_preserved'], details['original_optimized_KD_source'] = source_matches(PARENT_SOURCE, original['source_sha256'])
        step0 = Path(original['step0']['step0'])
        checks['original_shared_step0_preserved'] = legal_path(step0) and digest(step0) == original['step0']['step0_sha256'] == STEP0_SHA
    archive = ROOT / 'runs/fixed_baseline_v1/source_snapshot.tar.gz'
    checks['original_baseline_archive_preserved'] = archive.is_file() and legal_path(archive) and digest(archive) == ARCHIVE_SHA
    if snapshot is not None and origin is not None:
        checks['original_baseline_snapshot_records_preserved'] = (snapshot['archive_sha256'] == ARCHIVE_SHA
            and snapshot['files_sha256'] == origin['original_source_hashes']
            and snapshot['file_count'] == len(snapshot['files_sha256']))


def inherited_provenance(manifest, checks, details, pending):
    teacher = PARENT / '10-5/step1/checkpoints/model_final.pth'
    resume = PARENT / '10-5/step2/checkpoints/model_iter_2000.pth'
    checks['only_original_optimizedKD_stage1_inherited'] = Path(manifest['inherited_stage1']) == PARENT / '10-5/step1'
    checks['expected_teacher_and_resume_paths'] = Path(manifest['teacher']) == teacher and Path(manifest['shared_stage2_warmup']) == resume
    checks['teacher_and_resume_paths_legal'] = legal_path(teacher) and legal_path(resume)
    receipt = required(PARENT / '10-5/step1/training_complete.json', pending)
    job = required(PARENT / 'eval_queue/step2_iter2000.json', pending)
    result = required(PARENT / 'evaluations/step2_iter2000/result.json', pending)
    if teacher.exists() and receipt is not None:
        checks['original_teacher_SHA_preserved'] = digest(teacher) == manifest['teacher_sha256'] == receipt['checkpoint_sha256'] == TEACHER_SHA and receipt.get('returncode') == 0
    elif not teacher.exists():
        pending.append(str(teacher))
    if resume.exists() and job is not None and result is not None:
        checks['original_stage2_warmup_SHA_preserved'] = (Path(job['checkpoint']) == resume
            and digest(resume) == manifest['shared_stage2_warmup_sha256'] == job['checkpoint_sha256'] == result['checkpoint_sha256'] == WARMUP_SHA)
        checks['original_warmup_stage2_iteration2000'] = job['step'] == result['step'] == 2 and job['iteration'] == result['iteration'] == 2000
    elif not resume.exists():
        pending.append(str(resume))
    details['inherited_stage1'] = {'path': str(PARENT / '10-5/step1'), 'teacher_sha256': manifest['teacher_sha256'],
                                  'trained_with_online_pairKD': False}
    details['warmup_restore'] = {'checkpoint': str(resume), 'sha256': manifest['shared_stage2_warmup_sha256'],
                                'model_and_optimizer': 'restore at iteration2000',
                                'observer_and_selector': 'fresh at iteration2000; not inherited from A/B/C'}


def config_and_resume(manifest, cfg, checks, details, pending):
    parent_cfg = required(PARENT / '10-5/step2/config.json', pending)
    expected = dict(manifest['common_config'])
    expected.update(step=2, spg=1, max_iters=8000, warmup_iters=2000, loss_warmup_iters=2000,
                    seed=0, train_limit=0, val_limit=0, online_confusion=True, async_eval=True,
                    pair_mode='pairkd', w_pair_sep=0., w_pixel_kd=.1, kd_temperature=2.,
                    w_proto_kd=0., w_proto_sep=0., ald=False, confusion_reweight=False,
                    confusion_momentum=.98, pair_refresh_interval=50, pair_min_row_images=8,
                    pair_min_pair_images=3, pair_min_rate=.01, pair_min_updates=100,
                    pair_ramp_updates=200, pair_max_stale_updates=200)
    mismatch = {key: {'expected': value, 'actual': cfg.get(key)} for key, value in expected.items() if cfg.get(key) != value}
    checks['D_config_and_6000_remaining_steps_verified'] = not mismatch
    details['config_mismatches'] = mismatch
    checks['SEP_external_coefficient_zero'] = cfg['w_pair_sep'] == 0. and cfg['w_proto_sep'] == 0.
    checks['eight_single_sample_ranks_global_batch8'] = manifest['gpus'] == list(range(8)) and cfg['spg'] == 1 and manifest['global_batch'] == 8
    checks['only_old_pairKD_selected'] = cfg['pair_mode'] == 'pairkd' and cfg['online_confusion'] is True and not cfg.get('sep_new_cam_guard', False)
    checks['expected_teacher_resume_work_paths'] = (cfg['prev_checkpoint'] == manifest['teacher']
        and cfg['resume_checkpoint'] == manifest['shared_stage2_warmup'] and Path(cfg['work_dir']) == STAGE
        and Path(cfg['ckpt_dir']) == STAGE / 'checkpoints'
        and all(legal_path(cfg[key]) for key in ['prev_checkpoint', 'resume_checkpoint', 'work_dir', 'ckpt_dir', 'pred_dir']))
    if parent_cfg is not None:
        runtime = {'local_rank', 'step', 'work_dir', 'prev_checkpoint', 'resume_checkpoint', 'ckpt_dir', 'pred_dir'}
        allowed = {'spg', 'online_confusion', 'pair_mode', 'w_pair_sep', 'confusion_momentum',
                   'pair_refresh_interval', 'pair_min_row_images', 'pair_min_pair_images', 'pair_min_rate',
                   'pair_min_updates', 'pair_ramp_updates', 'pair_max_stale_updates'}
        before = {key: value for key, value in parent_cfg.items() if key not in runtime}
        after = manifest['common_config']
        changes = {key: {'optimized_KD': before.get(key), 'D': after.get(key)}
                   for key in sorted(set(before) | set(after)) if before.get(key) != after.get(key)}
        checks['only_explicit_changes_from_original_optimizedKD'] = set(changes) <= allowed
        details['config_changes_from_original_optimizedKD'] = changes
    resume = Path(manifest['shared_stage2_warmup'])
    if resume.exists():
        state = torch.load(resume, map_location='cpu', weights_only=True, mmap=True)
        checks['resume_iteration2000_model_available'] = state.get('iteration') == 2000 and isinstance(state.get('model_state'), dict) and bool(state['model_state'])
        optimizer = state.get('optimizer_state')
        checks['resume_full_optimizer_available'] = isinstance(optimizer, dict) and bool(optimizer.get('state')) and bool(optimizer.get('param_groups'))
        checks['original_warmup_contains_no_observer_or_selector'] = state.get('online_confusion_state') is None and state.get('pair_selector_state') is None
        del state
    else:
        pending.append(str(resume))
    log = STAGE / 'train.log'
    if log.exists():
        text = log.read_text()
        checks['model_optimizer_resume_recorded_at2000'] = 'Resume student AND optimizer at iteration 2000' in text and manifest['shared_stage2_warmup'] in text
        checks['fresh_online_observer_initialization_recorded'] = 'Checkpoint predates online confusion; start observational state at resume iteration' in text
        checks['original_teacher_load_recorded'] = manifest['teacher'] in text
    else:
        pending.append(str(log))


def endpoint_audit(cfg, checks, details, pending):
    final = STAGE / 'checkpoints/model_final.pth'
    evaluated = STAGE / 'checkpoints/model_iter_8000.pth'
    checks['endpoint_checkpoint_paths_legal'] = legal_path(final) and legal_path(evaluated)
    if not final.exists() or not evaluated.exists():
        pending.extend(str(path) for path in (final, evaluated) if not path.exists())
        return
    a = torch.load(final, map_location='cpu', weights_only=True, mmap=True)
    b = torch.load(evaluated, map_location='cpu', weights_only=True, mmap=True)
    am, bm = a.get('model_state'), b.get('model_state')
    checks['final_model_equals_evaluated_model_tensor_exact'] = isinstance(am, dict) and isinstance(bm, dict) and same_state(am, bm)
    checks['evaluated_iteration8000'] = b.get('iteration') == 8000
    optimizer = b.get('optimizer_state')
    checks['evaluated_full_optimizer_saved'] = isinstance(optimizer, dict) and bool(optimizer.get('state')) and bool(optimizer.get('param_groups'))
    details['endpoint_identity'] = {'final_sha256': digest(final), 'evaluated_sha256': digest(evaluated),
                                  'verified_model_tensor_count': len(am) if isinstance(am, dict) else None}
    for key, short, selector in [('online_confusion_state', 'observer', False), ('pair_selector_state', 'selector', True)]:
        if selector:
            va, ia = selector_state_metadata(a.get(key), cfg)
            vb, ib = selector_state_metadata(b.get(key), cfg)
        else:
            va, ia = state_metadata(a.get(key), cfg, 2, 21, selector=False)
            vb, ib = state_metadata(b.get(key), cfg, 2, 21, selector=False)
        checks[short + '_classes_stage_and_buffers_valid'] = va and vb
        checks[short + '_final_and_evaluated_states_exact'] = va and vb and same_state(a[key], b[key])
        details[short + '_endpoint_state'] = {'final': ia, 'evaluated': ib}
        if short == 'observer' and va:
            checks['fresh_observer_6000_updates_48000_images'] = ia['updates'] == 6000 and ia['seen_images'] == 48000
        if short == 'selector' and va:
            checks['selector_online_refresh_within_D_stage'] = 2000 < ia['last_refresh_iteration'] <= 8000
    del a, b, am, bm


def selector_state_metadata(state, cfg):
    """Validate D's domain-aware metadata against its actual frozen class."""
    if not isinstance(state, dict):
        return False, {'reason': 'selector state missing or not a mapping'}
    sys.path.insert(0, str(SOURCE))
    from model.directed_pair_selector import DirectedPairSelector
    module = DirectedPairSelector(21, stage=2, distillation_class_limit=16,
        refresh_interval=cfg['pair_refresh_interval'], min_row_images=cfg['pair_min_row_images'],
        min_pair_images=cfg['pair_min_pair_images'], min_rate=cfg['pair_min_rate'],
        min_updates=cfg['pair_min_updates'], ramp_updates=cfg['pair_ramp_updates'],
        max_stale_updates=cfg['pair_max_stale_updates'])
    expected = module.get_extra_state()
    extra = state.get('_extra_state', {})
    schema_ok = expected.get('schema') == extra.get('schema') == 2 and extra.get('distillation_class_limit') == 16
    info = {'metadata': extra, 'expected_metadata': expected, 'domain_metadata_valid': schema_ok and extra == expected}
    try:
        module.load_state_dict(state, strict=True)
    except (RuntimeError, ValueError, TypeError) as exc:
        info['restore_error'] = repr(exc)
        return False, info
    targets = module.targets.detach()
    rates = module.selected_rates.detach()
    shape_ok = targets.shape == rates.shape == (21,) and module.last_refresh_iteration.shape == ()
    if not shape_ok:
        info['invalid_shapes'] = True
        return False, info
    ids = torch.arange(21)
    direction_ok = bool(((targets == -1) | ((targets > 0) & (targets < 16) & (targets != ids))).all())
    source_domain_ok = int(targets[0]) == -1 and bool((targets[16:] == -1).all())
    rates_ok = bool(torch.isfinite(rates).all() and ((rates >= 0) & (rates <= 1)).all()
                    and (rates[targets == -1] == 0).all())
    restore_exact = same_state(state, module.state_dict())
    valid = schema_ok and extra == expected and direction_ok and source_domain_ok and rates_ok and restore_exact
    info.update(targets=targets.tolist(), selected_rates=rates.tolist(),
                old_foreground_directions_valid=direction_ok and source_domain_ok,
                selected_rates_valid=rates_ok, restored_state_tensor_exact=restore_exact,
                last_refresh_iteration=int(module.last_refresh_iteration))
    return valid, info


def mechanism_audit(checks, details, pending, finished):
    kd_file = STAGE / 'kd_metrics.jsonl'
    pair_file = STAGE / 'pair_metrics.jsonl'
    if not kd_file.exists() or not pair_file.exists():
        pending.extend(str(path) for path in (kd_file, pair_file) if not path.exists())
        return
    def rows(path):
        result = []
        for line in path.read_text().splitlines():
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                pending.append('Not yet complete metric line: ' + str(path))
        return result
    kd, pairs = rows(kd_file), rows(pair_file)
    if not kd or not pairs:
        pending.append('KD/pair metric logs not yet populated')
        return
    checks['logged_KD_weight_and_global_batch_correct'] = all(row['step'] == 2 and row['global_batch'] == 8 and row['weight'] == .1 for row in kd)
    checks['all_logged_SEP_external_weights_zero'] = all(row['step'] == 2 and row['weight'] == 0. for row in pairs)
    checks['pair_blend_within_fixed_half_budget'] = all(0. <= row.get('kd_pair_blend', 0.) <= .5 for row in kd)
    checks['logged_selector_teacher_domain_correct'] = all(
        row['selector']['schema'] == 2 and row['selector']['classes'] == 21 and row['selector']['stage'] == 2
        and row['selector']['distillation_class_limit'] == 16
        and row['selector']['targets'][0] == -1 and all(target == -1 for target in row['selector']['targets'][16:])
        and all(target == -1 or (0 < target < 16 and target != anchor)
                for anchor, target in enumerate(row['selector']['targets'])) for row in pairs)
    active_pairs = [row for row in kd if row.get('active') and row.get('kd_pair_pixels', 0) > 0 and row.get('kd_pair_blend', 0.) > 0]
    if active_pairs:
        checks['pairKD_actually_activated_on_supported_pixels'] = True
    elif finished:
        checks['pairKD_actually_activated_on_supported_pixels'] = False
    else:
        pending.append('PairKD activation still pending during online observation warmup')
    if finished:
        checks['logged_KD_and_pair_endpoints_at8000'] = kd[-1]['iteration'] == pairs[-1]['iteration'] == 8000
        checks['exactly120_logged_post_resume_batches'] = len(kd) == len(pairs) == 120
    details['mechanism_evidence'] = {'logged_batches': len(kd), 'logged_pairKD_nonzero_batches': len(active_pairs),
        'logged_pairKD_pixels': sum(row.get('kd_pair_pixels', 0) for row in kd),
        'eligible_pair_old_class_ids': sorted({class_id for row in active_pairs for class_id in row.get('kd_pair_classes', [])}),
        'maximum_logged_pair_blend': max(row.get('kd_pair_blend', 0.) for row in kd),
        'SEP_external_weight': 0., 'teacher_distillation_domain': list(range(1, 16)),
        'matrix_normalization': 'Full21-column EMA row mass; only old foreground source/partner are selectable',
        'note': 'These are sampled batch supervision counts, not all-batch totals or per-class parameter-gradient norms.'}


def evaluation_audit(cfg, checks, details, pending):
    part = RUN / 'evaluations/step2_iter8000'
    result = required(part / 'result.json', pending)
    job = required(RUN / 'eval_queue/step2_iter8000.json', pending)
    if result is None or job is None:
        return
    checkpoint = STAGE / 'checkpoints/model_iter_8000.pth'
    checks['evaluated_checkpoint_path_legal'] = legal_path(job['checkpoint']) and Path(job['checkpoint']) == checkpoint
    if checkpoint.exists():
        checks['evaluated_checkpoint_SHA_verified'] = digest(checkpoint) == job['checkpoint_sha256'] == result['checkpoint_sha256']
    else:
        pending.append(str(checkpoint))
    checks['evaluation_stage2_iteration8000'] = job['step'] == result['step'] == 2 and job['iteration'] == result['iteration'] == 8000
    checks['evaluation_config_matches_training'] = job['config'] == cfg
    checks['all1449_validation_images'] = result['images'] == 1449
    hist = np.asarray(result['histogram'])
    checks['histogram_shape_integer_counts_valid'] = hist.shape == (21, 21) and np.issubdtype(hist.dtype, np.integer) and bool((hist >= 0).all())
    if not checks['histogram_shape_integer_counts_valid']:
        return
    reference = required(PARENT / 'evaluations/step2_iter8000/result.json', pending)
    if reference is not None:
        rh = np.asarray(reference['histogram'], dtype=np.int64)
        checks['original_GT_row_totals_match_optimizedKD'] = rh.shape == hist.shape and np.array_equal(hist.sum(1), rh.sum(1))
        details['GT_row_totals'] = hist.sum(1).tolist()
    union = hist.sum(1) + hist.sum(0) - hist.diagonal()
    iou = np.divide(hist.diagonal(), union, out=np.full(21, np.nan), where=union > 0) * 100
    def mean(values):
        return float(np.nanmean(values)) if np.isfinite(values).any() else None
    computed = {'all_miou': mean(iou), 'foreground_miou': mean(iou[1:]),
                'previous_foreground_miou': mean(iou[1:16]), 'current_foreground_miou': mean(iou[16:]),
                'old_initial10_miou': mean(iou[1:11]), 'new_since_initial_miou': mean(iou[11:])}
    details['metrics_recomputed_from_histogram'] = computed
    checks['reported_mIoU_matches_histogram'] = all(near(result.get(key), value) for key, value in computed.items())
    checks['reported_class_IoU_matches_histogram'] = set(result['class_iou']) == set(CLASS_NAMES) and all(
        near(result['class_iou'][name], float(iou[i]) if np.isfinite(iou[i]) else None) for i, name in enumerate(CLASS_NAMES))
    counters = {'old_gt_to_new_pixels': int(hist[1:16, 16:].sum()), 'old_gt_to_background_pixels': int(hist[1:16, 0].sum()),
                'old_gt_pixels': int(hist[1:16].sum()), 'new_gt_to_old_pixels': int(hist[16:, 1:16].sum()), 'new_gt_pixels': int(hist[16:].sum())}
    checks['reported_confusion_counters_match_histogram'] = all(result.get(key) == value for key, value in counters.items())
    shards = [required(part / f'rank{rank}.json', pending) for rank in range(4)]
    if all(shard is not None for shard in shards):
        names = [name for shard in shards for name in shard['images']]
        expected = (Path(cfg['list_folder']) / 'incremental_split/val_10-5_step_3.txt').read_text().splitlines()
        checks['four_shards_disjoint_full1449'] = len(names) == len(set(names)) == len(expected) == 1449 and set(names) == set(expected)
        checks['four_shards_stage_and_checkpoint_verified'] = all(shard['rank'] == rank and shard['shards'] == 4 and shard['step'] == 2
            and shard['iteration'] == 8000 and shard['checkpoint_sha256'] == result['checkpoint_sha256'] for rank, shard in enumerate(shards))
        shard_hist = [np.asarray(shard['histogram']) for shard in shards]
        checks['merged_histogram_matches_four_shards'] = all(h.shape == (21, 21) for h in shard_hist) and np.array_equal(np.sum(shard_hist, axis=0), hist)
        details['shard_image_counts'] = [len(shard['images']) for shard in shards]


def audit():
    pending, checks, details = [], {}, {}
    manifest = required(RUN / 'study.json', pending)
    cfg = required(STAGE / 'config.json', pending)
    launch = required(STAGE / 'launch.json', pending)
    stage_done = required(STAGE / 'training_complete.json', pending)
    training_done = required(RUN / 'training_complete.json', pending)
    eval_done = required(RUN / 'evaluation_workers_complete.json', pending)
    checks['candidate_paths_legal'] = all(legal_path(path) for path in [RUN, SOURCE, STAGE])
    checks['no_new_stage1_training'] = not (RUN / '10-5/step1').exists()
    checks['no_recorded_training_failure'] = not (RUN / 'training_failed.json').exists()
    checks['no_recorded_evaluation_failure'] = not (RUN / 'evaluation_failed.json').exists()
    if manifest is not None:
        checks['formal_D_manifest_verified'] = manifest['arm'] == 'd_pairkd_only' and manifest['smoke'] is False and manifest['start_iteration'] == 2000
        checks['fresh_observer_and_teacher_domain_protocol_recorded'] = (
            manifest.get('remaining_training_updates') == 6000
            and manifest.get('selector_schema') == 2 and manifest.get('distillation_class_limit') == 16
            and manifest.get('observer_initialization') == 'fresh at stage2 iteration2000;6000updates')
        preservation_and_sources(manifest, checks, details, pending)
        inherited_provenance(manifest, checks, details, pending)
    if launch is not None:
        if 'returncode' in launch:
            checks['training_launch_exit0'] = launch['returncode'] == 0
        else:
            pending.append(str(STAGE / 'launch.json') + ': training still running')
        checks['eight_GPU_launch_verified'] = launch['gpus'] == list(range(8)) and '--nproc_per_node=8' in launch['command']
    if cfg is not None and manifest is not None:
        config_and_resume(manifest, cfg, checks, details, pending)
        mechanism_audit(checks, details, pending, stage_done is not None)
        evaluation_audit(cfg, checks, details, pending)
        if stage_done is not None:
            endpoint_audit(cfg, checks, details, pending)
            final = STAGE / 'checkpoints/model_final.pth'
            if final.exists():
                checks['final_checkpoint_matches_training_receipt'] = digest(final) == stage_done['checkpoint_sha256']
            else:
                pending.append(str(final))
    if stage_done is not None:
        checks['stage2_training_receipt_exit0'] = stage_done.get('returncode') == 0
    if training_done is not None:
        checks['one_new_stage_only_training_complete'] = training_done['stages'] == 1 and training_done['status'] == 'training_complete' and Path(training_done['inherited_stage1']) == PARENT / '10-5/step1'
        queue = list((RUN / 'eval_queue').glob('*.json'))
        checks['three_expected_stage2_eval_jobs_published'] = {job.stem for job in queue} == {'step2_iter4000', 'step2_iter6000', 'step2_iter8000'}
        missing = [str(RUN / 'evaluations' / job.stem / 'result.json') for job in queue if not (RUN / 'evaluations' / job.stem / 'result.json').exists()]
        pending.extend(missing)
        if not missing:
            checks['all_queued_evaluations_completed'] = True
    if eval_done is not None:
        checks['all_four_evaluation_workers_exit0'] = eval_done.get('returncodes') == [0, 0, 0, 0] and eval_done.get('status') == 'complete' and not eval_done.get('failure')
    failed = [key for key, value in checks.items() if not value]
    complete = not pending and not failed and all(record is not None for record in [manifest, cfg, launch, stage_done, training_done, eval_done])
    return {'utc': now(), 'status': 'complete' if complete else 'invalid' if failed else 'incomplete',
            'arm': 'd_pairkd_only', 'new_training_stage': 2, 'new_optimizer_steps': 6000,
            'checks': checks, 'failed_checks': failed, 'pending': sorted(set(pending)), 'details': details,
            'all_pairkd_only_training_and_endpoint_requirements_complete': complete,
            'all_training_and_endpoint_requirements_complete': complete,
            'device': 'CPU only; no CUDA allocations', 'resource_policy': '8card only; 8 x batch1, globalbatch8',
            'interpretation': 'D inherits original optimizedKD stage1 and stage2 iteration2000 model+optimizer; online observer/selector start fresh for 6000 updates. Directed confusion only changes conditional old-pair KD, with external SEP zero. Integrity audit does not establish accuracy benefit.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--output', help='Only independent pairkd_only_completion_audit.json is accepted')
    args = parser.parse_args()
    result = audit()
    if args.output:
        if Path(args.output) != RECEIPT:
            raise RuntimeError('Preserve prior audits; use pairkd_only_completion_audit.json')
        save_output(args.output, result)
    print(json.dumps(result, allow_nan=False))
    if args.require_complete and not result['all_pairkd_only_training_and_endpoint_requirements_complete']:
        raise SystemExit(1)
