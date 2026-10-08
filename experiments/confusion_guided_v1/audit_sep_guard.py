"""CPU-only audit of C's stage2 continuation; inherited stage1 is NOT guarded.

This uses the already deployed read-only audit helper beside this script.
Default is read-only JSON output; final evidence can be saved independently:
  python -B audit_sep_guard.py --require-complete --output \
    /ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto/runs/confusion_guided_v1/guard_completion_audit.json

No GPU execution or baseline/stage1 training is performed.
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
RUN = STUDY / 'formal/c_newaware_sep'
PARENT = STUDY / 'formal/a_sep'
SOURCE = EXPERIMENT / 'c_newaware_sep/src'
STAGE = RUN / '10-5/step2'
RECEIPT = STUDY / 'guard_completion_audit.json'

sys.path.insert(0, str(EXPERIMENT))
from audit_confusion_guided import (CLASS_NAMES, digest, read_json, required, now,
                                   endpoint_identity, state_metadata, near, save_output)
import numpy as np
import torch

torch.set_num_threads(2)


def legal_workspace_path(path):
    """Checkpoint/output paths may not escape via links or nested mounts."""
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


def source_audit(manifest, checks, details):
    recorded = manifest['source_sha256']
    bad, missing = [], []
    for rel, sha in recorded.items():
        path = SOURCE / rel
        if not path.resolve().is_relative_to(SOURCE.resolve()) or not legal_workspace_path(path):
            bad.append(rel + ': unsafe path')
        elif not path.is_file():
            missing.append(rel)
        elif digest(path) != sha:
            bad.append(rel)
    actual = {str(path.relative_to(SOURCE)) for path in SOURCE.rglob('*.py') if '__pycache__' not in path.parts}
    checks['frozen_source_verified'] = not bad and not missing and set(recorded) == actual
    details.update(source_mismatches=bad, missing_source_files=missing,
                   extra_source_files=sorted(actual - set(recorded)))
    runner = EXPERIMENT / 'run_sep_guard.py'
    checks['runner_freeze_verified'] = runner.is_file() and legal_workspace_path(runner) and digest(runner) == manifest['runner_sha256']


def inherited_provenance(manifest, checks, details, pending):
    teacher = PARENT / '10-5/step1/checkpoints/model_final.pth'
    resume = PARENT / '10-5/step2/checkpoints/model_iter_2000.pth'
    checks['inherited_stage1_path_verified'] = Path(manifest['inherited_stage1']) == PARENT / '10-5/step1'
    checks['expected_teacher_and_resume_paths'] = Path(manifest['teacher']) == teacher and Path(manifest['shared_stage2_warmup']) == resume
    checks['teacher_and_resume_paths_legal'] = legal_workspace_path(teacher) and legal_workspace_path(resume)
    teacher_receipt = required(PARENT / '10-5/step1/training_complete.json', pending)
    warmup_job = required(PARENT / 'eval_queue/step2_iter2000.json', pending)
    warmup_result = required(PARENT / 'evaluations/step2_iter2000/result.json', pending)
    if teacher.exists() and teacher_receipt is not None:
        checks['inherited_teacher_SHA_preserved'] = (digest(teacher) == manifest['teacher_sha256']
            == teacher_receipt['checkpoint_sha256'] and teacher_receipt.get('returncode') == 0)
    elif not teacher.exists():
        pending.append(str(teacher))
    if resume.exists() and warmup_job is not None and warmup_result is not None:
        checks['original_A_stage2_warmup_SHA_preserved'] = (Path(warmup_job['checkpoint']) == resume
            and digest(resume) == manifest['shared_stage2_warmup_sha256']
            == warmup_job['checkpoint_sha256'] == warmup_result['checkpoint_sha256'])
        checks['inherited_warmup_iteration_verified'] = warmup_job['step'] == warmup_result['step'] == 2 and warmup_job['iteration'] == warmup_result['iteration'] == 2000
    elif not resume.exists():
        pending.append(str(resume))
    details['inherited_stage1'] = {'path': str(PARENT / '10-5/step1'), 'teacher_checkpoint': str(teacher),
                                  'teacher_sha256': manifest['teacher_sha256'], 'trained_with_guard': False,
                                  'interpretation': 'Previously completed A stage1 is inherited; C only changes stage2 after iteration2000.'}
    details['shared_stage2_warmup'] = {'path': str(resume), 'sha256': manifest['shared_stage2_warmup_sha256'],
                                      'model_optimizer_observer_selector_restore': True, 'start_iteration': 2000}


def resume_state_audit(manifest, cfg, checks, details, pending):
    resume = Path(manifest['shared_stage2_warmup'])
    if not resume.exists():
        pending.append(str(resume))
        return
    saved = torch.load(resume, map_location='cpu', weights_only=True, mmap=True)
    checks['resume_checkpoint_iteration2000'] = saved.get('iteration') == 2000
    checks['resume_model_state_available'] = isinstance(saved.get('model_state'), dict) and bool(saved['model_state'])
    optimizer = saved.get('optimizer_state')
    checks['resume_optimizer_state_available'] = isinstance(optimizer, dict) and bool(optimizer.get('state')) and bool(optimizer.get('param_groups'))
    for key, short, selector in [('online_confusion_state', 'observer', False), ('pair_selector_state', 'selector', True)]:
        valid, info = state_metadata(saved.get(key), cfg, 2, 21, selector=selector)
        checks['resume_' + short + '_state_valid'] = valid
        details['resume_' + short + '_state'] = info
        if short == 'observer' and valid:
            checks['resume_observer_2000_updates_16000_images'] = info['updates'] == 2000 and info['seen_images'] == 16000
        if short == 'selector' and valid:
            checks['resume_selector_iteration_within_warmup'] = 0 <= info['last_refresh_iteration'] <= 2000
    del saved


def config_audit(manifest, cfg, checks, details, pending):
    parent_cfg = required(PARENT / '10-5/step2/config.json', pending)
    expected = dict(manifest['common_config'])
    expected.update(step=2, spg=1, max_iters=8000, warmup_iters=2000, loss_warmup_iters=2000,
                    seed=0, train_limit=0, val_limit=0, online_confusion=True, async_eval=True,
                    sep_new_cam_guard=True, pair_mode='sep', w_pair_sep=.1, w_pixel_kd=.1,
                    kd_temperature=2., w_proto_kd=0., w_proto_sep=0., ald=False, confusion_reweight=False)
    mismatches = {key: {'expected': value, 'actual': cfg.get(key)} for key, value in expected.items() if cfg.get(key) != value}
    checks['stage2_budget_and_config_verified'] = not mismatches
    details['config_mismatches'] = mismatches
    checks['eight_GPUs_single_sample_global_batch8'] = manifest['gpus'] == list(range(8)) and cfg['spg'] == 1 and manifest['global_batch'] == 8
    checks['resume_and_teacher_config_verified'] = cfg['resume_checkpoint'] == manifest['shared_stage2_warmup'] and cfg['prev_checkpoint'] == manifest['teacher']
    checks['stage2_work_and_checkpoint_paths_legal'] = (Path(cfg['work_dir']) == STAGE
        and Path(cfg['ckpt_dir']) == STAGE / 'checkpoints'
        and all(legal_workspace_path(cfg[key]) for key in ['work_dir', 'ckpt_dir', 'pred_dir', 'resume_checkpoint', 'prev_checkpoint']))
    if parent_cfg is not None:
        inherited = dict(parent_cfg)
        for key in ['local_rank', 'step', 'work_dir', 'prev_checkpoint', 'resume_checkpoint', 'ckpt_dir', 'pred_dir']:
            inherited.pop(key, None)
        inherited.update(spg=1, sep_new_cam_guard=True, pair_mode='sep')
        checks['only_intended_config_changes_from_A'] = inherited == manifest['common_config']
        details['config_changes_from_A'] = {key: {'A': parent_cfg.get(key), 'C': cfg.get(key)}
            for key in sorted(set(parent_cfg) | set(cfg)) if parent_cfg.get(key) != cfg.get(key)}
    log = STAGE / 'train.log'
    if log.exists():
        text = log.read_text()
        checks['model_and_optimizer_resume_at2000_recorded'] = 'Resume student AND optimizer at iteration 2000' in text and manifest['shared_stage2_warmup'] in text
        checks['inherited_teacher_load_recorded'] = manifest['teacher'] in text
    else:
        pending.append(str(log))


def evaluation_audit(cfg, checks, details, pending):
    part = RUN / 'evaluations/step2_iter8000'
    result = required(part / 'result.json', pending)
    job = required(RUN / 'eval_queue/step2_iter8000.json', pending)
    if result is None or job is None:
        return
    checkpoint = STAGE / 'checkpoints/model_iter_8000.pth'
    checks['evaluated_checkpoint_path_legal'] = legal_workspace_path(job['checkpoint']) and Path(job['checkpoint']) == checkpoint
    if checkpoint.exists():
        checks['evaluated_checkpoint_SHA_verified'] = digest(checkpoint) == job['checkpoint_sha256'] == result['checkpoint_sha256']
    else:
        pending.append(str(checkpoint))
    checks['evaluation_stage2_iteration8000'] = job['step'] == result['step'] == 2 and job['iteration'] == result['iteration'] == 8000
    checks['evaluation_config_matches_training'] = job['config'] == cfg
    checks['all1449_validation_images'] = result['images'] == 1449
    hist = np.asarray(result['histogram'])
    valid_hist = hist.shape == (21, 21) and np.issubdtype(hist.dtype, np.integer) and bool((hist >= 0).all())
    checks['histogram_shape_and_integer_counts_valid'] = valid_hist
    if not valid_hist:
        return
    reference = required(ROOT / 'runs/kd_parallel_v1/formal/b_relational/evaluations/step2_iter8000/result.json', pending)
    if reference is not None:
        rh = np.asarray(reference['histogram'], dtype=np.int64)
        checks['original_GT_row_totals_match_optimized_KD'] = rh.shape == hist.shape and np.array_equal(hist.sum(1), rh.sum(1))
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
                'old_gt_pixels': int(hist[1:16].sum()), 'new_gt_to_old_pixels': int(hist[16:, 1:16].sum()),
                'new_gt_pixels': int(hist[16:].sum())}
    checks['reported_confusion_counters_match_histogram'] = all(result.get(key) == value for key, value in counters.items())
    shards = [required(part / f'rank{rank}.json', pending) for rank in range(4)]
    if all(shard is not None for shard in shards):
        names = [name for shard in shards for name in shard['images']]
        expected = (Path(cfg['list_folder']) / 'incremental_split/val_10-5_step_3.txt').read_text().splitlines()
        checks['four_shards_disjoint_and_full1449_coverage'] = len(names) == len(set(names)) == len(expected) == 1449 and set(names) == set(expected)
        checks['four_shards_checkpoint_and_stage_verified'] = all(shard['rank'] == rank and shard['shards'] == 4
            and shard['step'] == 2 and shard['iteration'] == 8000 and shard['checkpoint_sha256'] == result['checkpoint_sha256']
            for rank, shard in enumerate(shards))
        shard_hist = [np.asarray(shard['histogram']) for shard in shards]
        checks['merged_histogram_matches_four_shards'] = all(h.shape == (21, 21) for h in shard_hist) and np.array_equal(np.sum(shard_hist, axis=0), hist)
        details['evaluation_shard_image_counts'] = [len(shard['images']) for shard in shards]


def audit():
    pending, checks, details = [], {}, {}
    manifest = required(RUN / 'study.json', pending)
    cfg = required(STAGE / 'config.json', pending)
    launch = required(STAGE / 'launch.json', pending)
    stage_done = required(STAGE / 'training_complete.json', pending)
    training_done = required(RUN / 'training_complete.json', pending)
    eval_done = required(RUN / 'evaluation_workers_complete.json', pending)
    checks['all_candidate_paths_legal'] = all(legal_workspace_path(path) for path in [RUN, SOURCE, STAGE])
    checks['no_new_guard_stage1_run'] = not (RUN / '10-5/step1').exists()
    checks['no_recorded_training_failure'] = not (RUN / 'training_failed.json').exists()
    if manifest is not None:
        checks['formal_C_manifest_verified'] = manifest['arm'] == 'c_newaware_sep' and manifest['smoke'] is False and manifest['start_iteration'] == 2000
        source_audit(manifest, checks, details)
        inherited_provenance(manifest, checks, details, pending)
    if launch is not None:
        if 'returncode' in launch:
            checks['training_launch_exit0'] = launch['returncode'] == 0
        else:
            pending.append(str(STAGE / 'launch.json') + ': still running')
        checks['eight_GPU_launch_assignment_verified'] = launch['gpus'] == list(range(8))
        command = launch['command']
        checks['eight_rank_distributed_launch_verified'] = '--nproc_per_node=8' in command
    if cfg is not None and manifest is not None:
        config_audit(manifest, cfg, checks, details, pending)
        resume_state_audit(manifest, cfg, checks, details, pending)
        evaluation_audit(cfg, checks, details, pending)
        if stage_done is not None:
            # Wait for atomic completion receipt so plain torch.save is closed.
            endpoint_identity(STAGE, cfg, 2, checks, details, pending)
            final = STAGE / 'checkpoints/model_final.pth'
            checks['final_and_evaluated_checkpoint_paths_legal'] = all(legal_workspace_path(path) for path in [final, STAGE / 'checkpoints/model_iter_8000.pth'])
            if final.exists():
                checks['final_checkpoint_matches_training_receipt'] = digest(final) == stage_done['checkpoint_sha256']
            else:
                pending.append(str(final))
    if stage_done is not None:
        checks['stage2_training_receipt_exit0'] = stage_done.get('returncode') == 0
    if training_done is not None:
        checks['only_one_new_training_stage_completed'] = training_done['stages'] == 1 and training_done['status'] == 'training_complete' and Path(training_done['inherited_stage1']) == PARENT / '10-5/step1'
        queue = list((RUN / 'eval_queue').glob('*.json'))
        checks['only_expected_three_stage2_evaluation_jobs'] = {job.stem for job in queue} == {'step2_iter4000', 'step2_iter6000', 'step2_iter8000'}
        missing = [str(RUN / 'evaluations' / job.stem / 'result.json') for job in queue if not (RUN / 'evaluations' / job.stem / 'result.json').exists()]
        pending.extend(missing)
        if not missing:
            checks['all_queued_evaluations_completed'] = True
    if eval_done is not None:
        checks['all_four_evaluation_workers_exit0'] = eval_done.get('returncodes') == [0, 0, 0, 0] and eval_done.get('status') == 'complete' and not eval_done.get('failure')
    failed = [key for key, value in checks.items() if not value]
    complete = not failed and not pending and all(record is not None for record in [manifest, cfg, launch, stage_done, training_done, eval_done])
    return {'utc': now(), 'status': 'complete' if complete else 'invalid' if failed else 'incomplete',
            'arm': 'c_newaware_sep', 'new_training_stage': 2, 'new_optimizer_steps': 6000,
            'checks': checks, 'failed_checks': failed, 'pending': sorted(set(pending)), 'details': details,
            'all_guard_training_and_endpoint_requirements_complete': complete,
            'all_training_and_endpoint_requirements_complete': complete,
            'device': 'CPU only; no CUDA allocation', 'resource_policy': '8card only; 8 x batch1, globalbatch8',
            'interpretation': 'C inherits unguarded A stage1 and A stage2 model/optimizer/observer/selector at iteration2000, then trains 6000 stage2 steps with old-SEP new-CAM suppression. This audit verifies integrity, not an accuracy gain.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--output', help='Only the independent guard_completion_audit.json receipt is accepted')
    args = parser.parse_args()
    result = audit()
    if args.output:
        if Path(args.output) != RECEIPT:
            raise RuntimeError('Use the independent guard_completion_audit.json; preserve the original A/B audit')
        save_output(args.output, result)
    print(json.dumps(result, allow_nan=False))
    if args.require_complete and not result['all_guard_training_and_endpoint_requirements_complete']:
        raise SystemExit(1)
