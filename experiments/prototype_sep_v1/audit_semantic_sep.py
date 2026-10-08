"""CPU-only audit for the conditional stage2 semantic-protected SEP repair.

Default is read-only; missing running evidence is incomplete. Final receipt:
  python -B audit_semantic_sep.py --require-complete --output \
    /ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto/runs/prototype_sep_v1/formal/b_semantic/completion_audit.json

B inherits A's completed stage1 teacher and fully restores A stage2 iteration
2000 model, optimizer, observer and geometry selector. It trains only the6000
remaining steps. This audit does not execute a model, evaluate data or use CUDA.
"""
from pathlib import Path
import argparse
import ast
import inspect
import json
import math
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

ROOT = Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto')
EXPERIMENT = ROOT / 'experiments/prototype_sep_v1'
STUDY = ROOT / 'runs/prototype_sep_v1'
PARENT = STUDY / 'formal/a_geometry'
PARENT_SOURCE = EXPERIMENT / 'a_geometry/src'
RUN = STUDY / 'formal/b_semantic'
SOURCE = EXPERIMENT / 'b_semantic/src'
STAGE = RUN / '10-5/step2'
RECEIPT = RUN / 'completion_audit.json'
TEACHER = PARENT / '10-5/step1/checkpoints/model_final.pth'
RESUME = PARENT / '10-5/step2/checkpoints/model_iter_2000.pth'
UNIT_SHA = 'e17b1c3657d3fff41b3cb2b322ee4b3b4af964882e3fe327f8b6d0c86c488760'
DDP_SHA = 'a77840937d44f11c322b58c4f68c99d3276407aeced02d3a64d04d2f3d5e57e0'

sys.path.insert(0, str(EXPERIMENT))
from audit_prototype_sep import (ALLOWED, CLASS_NAMES, STEP0, STEP0_SHA, ARCHIVE_SHA,
    OPTIMIZED_ENDPOINTS, safe_path, legal, digest, required, near, now, same_state,
    complete_optimizer, load_checkpoint, source_audit, hash_check, module_states, endpoint)
import numpy as np
import torch

torch.set_num_threads(2)


def sources_and_preservation(manifest, checks, details, pending):
    parent_manifest = required(PARENT / 'study.json', pending)
    origin = required(EXPERIMENT / 'semantic_origin.json', pending)
    integration = required(EXPERIMENT / 'b_semantic/semantic_integration.json', pending)
    checks['B_source_freeze'], details['B_source'] = source_audit(SOURCE, manifest['source_sha256'])
    hash_check(EXPERIMENT / 'run_semantic_sep.py', manifest['runner_sha256'], checks, 'B_runner_freeze', pending)
    if parent_manifest is not None:
        checks['A_source_unchanged'], details['A_source'] = source_audit(PARENT_SOURCE, parent_manifest['source_sha256'])
        before, after = parent_manifest['source_sha256'], manifest['source_sha256']
        changed = {rel for rel in set(before) & set(after) if before[rel] != after[rel]}
        added = set(after)-set(before)
        allowed = {'continual/Trainer.py', 'model/model_seg_neg.py', 'model/decoder.py',
                   'model/decoder/conv_head.py', 'model/conv_head.py', 'scripts/dist_train_voc_seg_neg.py'}
        checks['only_explicit_SEP_forward_integration_changes'] = (
            not (set(before)-set(after)) and changed <= allowed and added == {'model/semantic_protected_sep.py'})
        unchanged = ['model/pixel_kd.py', 'model/losses.py', 'model/confusion_prototype_sep.py',
                     'model/online_directed_confusion.py', 'model/directed_pair_selector.py', 'model/geometry_pair_selector.py']
        checks['KD_semantic_ratio_pair_geometry_and_observer_modules_unchanged'] = all(before[key] == after[key] for key in unchanged)
        details['changed_source_files_from_A'] = sorted(changed | added)
        if integration is not None:
            checks['executed_integration_receipt_matches_frozen_B_changes'] = (
                integration['schema'] == 1 and integration['candidate'] == str(SOURCE)
                and integration['primary_frozen_source'] == str(PARENT_SOURCE)
                and integration['primary_source_sha256'] == before
                and integration['changed_source_sha256'] == {key: after[key] for key in changed | added}
                and integration['optimized_kd_sha256_unchanged'] == after['model/pixel_kd.py']
                and integration['rng_audit']['rng_calls_found'] == [])
            hash_check(EXPERIMENT / 'patch_semantic_candidate.py', integration['patch_script_sha256'],
                checks, 'executed_remote_patch_script_SHA_preserved', pending)
            details['integration_receipt'] = integration
        if origin is not None:
            checks['origin_parent_and_final_conditional_source_SHA_records'] = (
                origin['parent_source'] == str(PARENT_SOURCE) and origin['conditional_source'] == str(SOURCE)
                and origin['parent_source_sha256'] == before and origin['conditional_source_sha256'] == after
                and set(origin['changed_files']) == changed | added
                and origin.get('KD_byte_identical') is True and origin.get('primary_source_unchanged') is True)
            # Development may have several explicit revisions. Verify each
            # recorded transition and the final freeze without rewriting history.
            revisions = origin.get('source_revisions', [])
            if revisions:
                prior, revision_bad = before, []
                for index, row in enumerate(revisions):
                    current = row['source_sha256']
                    actual = {key for key in set(prior) | set(current) if prior.get(key) != current.get(key)}
                    if set(row['changed_files']) != actual:
                        revision_bad.append(index)
                    prior = current
                checks['all_recorded_source_revisions_consistent'] = not revision_bad and prior == after
                details['source_revisions'] = revisions
    hash_check(STEP0, STEP0_SHA, checks, 'original_shared_step0_preserved', pending)
    hash_check(ROOT / 'runs/fixed_baseline_v1/source_snapshot.tar.gz', ARCHIVE_SHA,
               checks, 'original_baseline_archive_preserved', pending)
    old_parent = ROOT / 'runs/kd_parallel_v1/formal/b_relational/10-5'
    for rel, sha in OPTIMIZED_ENDPOINTS.items():
        hash_check(old_parent / rel, sha, checks, 'optimized_KD_' + rel.split('/')[0] + '_' + Path(rel).stem + '_preserved', pending)
    original_kd = ROOT / 'experiments/kd_parallel_v1/b_relational/src/model/pixel_kd.py'
    checks['original_optimized_KD_byte_identical'] = original_kd.is_file() and digest(original_kd) == digest(SOURCE / 'model/pixel_kd.py')
    if checks['B_source_freeze']:
        tree = ast.parse((SOURCE / 'continual/Trainer.py').read_text())
        calls = [(ast.unparse(node.func), ast.unparse(node.args[0]) if node.args else '',
                  {item.arg: ast.unparse(item.value) for item in node.keywords})
                 for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                 and node.func.attr == 'load_state_dict']
        expected = [({'model.load_state_dict', 'self.model.load_state_dict'}, "saved['model_state']", True),
                    ({'optim.load_state_dict'}, "saved['optimizer_state']", False),
                    ({'self.confusion.load_state_dict'}, "saved['online_confusion_state']", True),
                    ({'self.pair_selector.load_state_dict'}, "saved['geometry_selector_state']", True)]
        checks['frozen_trainer_restores_all_four_states'] = all(any(
            func in targets and argument == key and (not strict or keywords.get('strict') == 'True')
            for func, argument, keywords in calls) for targets, key, strict in expected)
        checks['frozen_trainer_restores_optimizer_schedule_iteration'] = any(
            isinstance(node, ast.Assign) and ast.unparse(node.value) == 'start_iteration'
            and any(ast.unparse(target) == 'optim.global_step' for target in node.targets) for node in ast.walk(tree))
        checks['DDP_find_unused_parameters_true_retained'] = any(
            isinstance(node, ast.Call) and ast.unparse(node.func) == 'DistributedDataParallel'
            and any(key.arg == 'find_unused_parameters' and ast.unparse(key.value) == 'True' for key in node.keywords)
            for node in ast.walk(tree))
        callbacks = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'prototype_objective']
        checks['guard_computed_inside_model_callback_before_sibling_wrapping'] = len(callbacks) == 1 and any(
            isinstance(node, ast.Call) and ast.unparse(node.func) == 'semantic_protected_sep' for node in ast.walk(callbacks[0]))
        def model_structure(path):
            model_tree = ast.parse(path.read_text())
            network = next(node for node in model_tree.body if isinstance(node, ast.ClassDef) and node.name == 'network')
            return {node.name: ast.dump(node, include_attributes=False) for node in network.body
                    if isinstance(node, ast.FunctionDef) and node.name in ['__init__', 'get_param_groups']}
        checks['model_constructor_state_registry_and_optimizer_groups_unchanged'] = (
            model_structure(SOURCE / 'model/model_seg_neg.py') == model_structure(PARENT_SOURCE / 'model/model_seg_neg.py'))


def inheritance(manifest, checks, details, pending):
    primary = required(PARENT / 'completion_audit.json', pending)
    status = required(PARENT / 'status.json', pending)
    analysis = required(STUDY / 'result_analysis.json', pending)
    teacher_receipt = required(PARENT / '10-5/step1/training_complete.json', pending)
    job = required(PARENT / 'eval_queue/step2_iter2000.json', pending)
    result = required(PARENT / 'evaluations/step2_iter2000/result.json', pending)
    checks['only_stage2_repair_and_no_new_stage1'] = not (RUN / '10-5/step1').exists()
    checks['manifest_full_state_restore_at2000'] = (
        manifest['arm'] == 'b_semantic' and manifest['smoke'] is False and manifest['resume_iteration'] == 2000
        and manifest['inherited_stage1_run'] == str(PARENT) and Path(manifest['teacher']) == TEACHER
        and Path(manifest['resume_checkpoint']) == RESUME
        and manifest['restored_state'] == ['student', 'optimizer', 'online_confusion', 'geometry_selector'])
    if primary is not None and status is not None:
        checks['primary_A_completed_and_passed_integrity_audit'] = (primary.get('status') == 'complete'
            and primary.get('all_training_and_endpoint_requirements_complete') is True and status['status'] == 'complete')
    if analysis is not None:
        checks['evidence_driven_conditional_repair_gate'] = analysis['recommendation']['candidate_exceeds_required_endpoint'] is False
    if teacher_receipt is not None:
        hash_check(TEACHER, manifest['teacher_sha256'], checks, 'inherited_A_stage1_teacher_SHA', pending)
        checks['inherited_teacher_from_successful_A_stage1'] = teacher_receipt['returncode'] == 0 and teacher_receipt['checkpoint_sha256'] == manifest['teacher_sha256']
    if job is not None and result is not None:
        hash_check(RESUME, manifest['resume_checkpoint_sha256'], checks, 'A_stage2_warmup_SHA_preserved', pending)
        checks['resume_SHA_matches_A_publication_and_full_evaluation'] = (
            Path(job['checkpoint']) == RESUME and job['checkpoint_sha256'] == result['checkpoint_sha256'] == manifest['resume_checkpoint_sha256']
            and job['step'] == result['step'] == 2 and job['iteration'] == result['iteration'] == 2000 and result['images'] == 1449)
    details['inheritance'] = {'teacher': str(TEACHER), 'resume': str(RESUME),
        'new_optimizer_steps': 6000, 'new_stage1_training': False,
        'final_online_budget': 'Restored2000 plus6000, final8000 updates/64000 globally exposed images'}


def preflight(manifest, checks, details, pending):
    pre = required(EXPERIMENT / 'semantic_preflight.json', pending)
    tests = required(EXPERIMENT / 'semantic_test_receipts.json', pending)
    if pre is not None and tests is not None:
        checks['exact_source_tests_and_actual_smoke_preflight_passed'] = (pre.get('passed') is True
            and tests.get('passed') is True and pre['source_sha256'] == tests['source_sha256'] == manifest['source_sha256']
            and pre['runner_sha256'] == manifest['runner_sha256'] and pre['smoke_complete'] is True)
        module_sha = digest(SOURCE / 'model/semantic_protected_sep.py')
        for key, filename, source_sha in [('unit_tests', 'test_semantic_protected_sep.py', UNIT_SHA),
                                         ('ddp_2rank', 'test_semantic_protected_sep_ddp.py', DDP_SHA),
                                         ('ddp_8rank', 'test_semantic_protected_sep_ddp.py', DDP_SHA)]:
            record = pre[key]
            checks[key + '_passed_against_exact_module'] = (record == tests[key] and record.get('passed') is True
                and record.get('returncode') == 0 and record['module_sha256'] == module_sha and record['source_sha256'] == source_sha)
            hash_check(EXPERIMENT / filename, source_sha, checks, key + '_test_source_frozen', pending)
            hash_check(record['log'], record['evidence_sha256'], checks, key + '_execution_log_SHA', pending)
            if key == 'unit_tests':
                checks['exactly23_CPU_unit_tests_passed'] = record.get('count') == 23
            else:
                outcome, ranks = record['result'], 2 if key == 'ddp_2rank' else 8
                checks[key + '_true_DDP_topology_and_pooled_normalization_passed'] = (
                    outcome.get('passed') is True and outcome['ranks'] == ranks
                    and outcome['backend'] == ('gloo' if ranks == 2 else 'nccl')
                    and outcome['sibling_output_probe']['find_unused_parameters'] is True
                    and outcome['sibling_output_probe']['semantic_to_returned_p_disconnected'] is True
                    and outcome['sibling_output_probe']['unsafe_external_helper_explicitly_rejected'] is True
                    and {row['case'] for row in outcome['cases']} == {'uneven_all_supported', 'uneven_with_zero_rank', 'all_zero_semantic'}
                    and all(row['inside_forward_partial_leaf_hooks_zero'] and row['final_leaf_hooks_once']
                            and row['unused_projector_grad_none'] for row in outcome['cases']))
        details['preflight'] = pre
    smoke = STUDY / 'smoke/b_semantic'
    smoke_status = required(smoke / 'status.json', pending)
    smoke_eval = required(smoke / 'evaluation_workers_complete.json', pending)
    smoke_launch = required(smoke / '10-5/step2/launch.json', pending)
    smoke_stage = required(smoke / '10-5/step2/training_complete.json', pending)
    smoke_result = required(smoke / 'evaluations/step2_iter2004/result.json', pending)
    smoke_manifest = required(smoke / 'study.json', pending)
    if all(value is not None for value in [smoke_status, smoke_eval, smoke_launch, smoke_stage, smoke_result, smoke_manifest]):
        checks['actual_8rank_network_smoke_four_shards_exit0'] = (
            smoke_status['status'] == 'complete' and smoke_eval['returncodes'] == [0]*4
            and smoke_launch['returncode'] == smoke_stage['returncode'] == 0 and smoke_result['images'] == 16
            and smoke_result['iteration'] == 2004 and smoke_manifest['source_sha256'] == manifest['source_sha256'])
    if pre is not None:
        checks['smoke_guard_exercised_real_conflict_projection'] = (
            pre['smoke_run'] == str(smoke) and pre['smoke_guard_rows'] == 4
            and len(pre['smoke_conflict_rows']) == 4 and any(value > 0 for value in pre['smoke_conflict_rows']))


def configuration_and_resume(manifest, cfg, checks, details, pending):
    parent_cfg = required(PARENT / '10-5/step2/config.json', pending)
    expected = dict(manifest['common_config'])
    expected.update(step=2, spg=1, max_iters=8000, warmup_iters=2000, loss_warmup_iters=2000,
        log_iters=50, eval_iters=2000, seed=0, train_limit=0, val_limit=0, async_eval=True,
        save_ckpt=True, w_geometry_sep=.1, w_pixel_kd=.1, kd_temperature=2., w_proto_seg=.1,
        w_proto_sep=0., w_proto_kd=0., proto_margin=0., ald=False, confusion_reweight=False)
    mismatch = {key: {'expected': value, 'actual': cfg.get(key)} for key, value in expected.items() if cfg.get(key) != value}
    checks['same_fixed_geometry_KD_semantic_strength_and_remaining_budget'] = not mismatch
    details['config_mismatches'] = mismatch
    checks['eight_times_one_global_batch8'] = manifest['train_gpus'] == list(range(8)) and manifest['batch_per_gpu'] == cfg['spg'] == 1 and manifest['global_batch'] == 8
    checks['expected_teacher_resume_and_safe_output_paths'] = (Path(cfg['prev_checkpoint']) == TEACHER
        and Path(cfg['resume_checkpoint']) == RESUME and Path(cfg['work_dir']) == STAGE
        and Path(cfg['ckpt_dir']) == STAGE / 'checkpoints'
        and all(legal(cfg[key]) for key in ['prev_checkpoint', 'resume_checkpoint', 'work_dir', 'ckpt_dir', 'pred_dir']))
    if parent_cfg is not None:
        runtime = {'step', 'local_rank', 'work_dir', 'ckpt_dir', 'pred_dir', 'prev_checkpoint', 'resume_checkpoint'}
        differences = {key: {'A': parent_cfg.get(key), 'B': cfg.get(key)}
            for key in set(parent_cfg) | set(cfg) if key not in runtime and parent_cfg.get(key) != cfg.get(key)}
        checks['no_unrelated_config_or_threshold_changes_from_A'] = not differences
        details['config_changes_from_A'] = differences
    resume = load_checkpoint(RESUME, pending)
    if resume is not None:
        checks['resume_iteration2000_full_model_optimizer_available'] = (resume.get('iteration') == 2000
            and isinstance(resume.get('model_state'), dict) and bool(resume['model_state']) and complete_optimizer(resume.get('optimizer_state')))
        valid, info = module_states(resume.get('online_confusion_state'), resume.get('geometry_selector_state'), cfg, 2)
        checks['resume_observer_selector_exact_restorable_with2000_updates16000_images'] = valid and info.get('updates') == 2000 and info.get('seen_images') == 16000
        details['resume_online_state'] = info
        del resume
    log = STAGE / 'train.log'
    if log.is_file():
        text = log.read_text()
        checks['model_optimizer_resume_at2000_logged'] = 'Resume student AND optimizer at iteration 2000' in text and str(RESUME) in text
        checks['inherited_A_stage1_teacher_and_global8_logged'] = str(TEACHER) in text and 'Total gpus: 8, samples per gpu: 1' in text
    else:
        pending.append(str(log))


def metric_rows(path, pending):
    if not safe_path(path).is_file():
        pending.append(str(path))
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pending.append('Not yet readable metric line: ' + str(path))
    return rows


def guard_logs(checks, details, pending, finished):
    guard = metric_rows(STAGE / 'semantic_guard_metrics.jsonl', pending)
    geometry = metric_rows(STAGE / 'geometry_metrics.jsonl', pending)
    kd = metric_rows(STAGE / 'kd_metrics.jsonl', pending)
    if not guard or not geometry or not kd:
        pending.append(str(STAGE) + ': guard/geometry/KD logs not yet populated')
        return
    expected = list(range(2050, 8001, 50))
    checks['guard_geometry_KD_iterations_match'] = [row['iteration'] for row in guard] == [row['iteration'] for row in geometry] == [row['iteration'] for row in kd]
    if finished:
        checks['exactly120_logged_post_resume_batches_to8000'] = [row['iteration'] for row in guard] == expected
    checks['legacy_KD_weight_global8_unchanged'] = all(row['step'] == 2 and row['weight'] == .1 and row['global_batch'] == 8
        and not any(key.startswith('kd_pair_') for key in row) for row in kd)
    failures, corrected_batches = [], []
    for row, pair in zip(guard, geometry):
        try:
            assert row['step'] == 2 and row['iteration'] > 2000
            assert row['old_classes'] == 15 and row['classes_including_background'] == 21 and row['world_size'] == 8
            assert row['forward_value_preserved'] is True and row['background_old_sep_gradient_zero'] is True
            assert row['active'] is True and row['enabled'] is True and row['weight'] == .1 and row['ramp'] == 1.
            assert row['effective_geometry_weight'] == .1
            assert row['no_additive_denominator_epsilon'] is True and row['no_norm_cap'] is True and row['higher_order_gradients'] is False
            assert pair['step'] == 2 and pair['weight'] == .1 and pair['ramp'] == 1. and pair['active'] is True
            assert near(row['weighted_geometry_forward_value'], .1*pair['prototype_sep'], 5e-7)
            assert pair['semantic_protection'] == {key: value for key, value in row.items()
                if key not in ['step', 'iteration', 'active', 'weight', 'ramp']}
            selector = pair['selector']
            assert selector['schema'] == 3 and selector['classes'] == 21 and selector['stage'] == 2 and selector['old_classes'] == 15
            targets = selector['targets']
            assert len(targets) == 21 and targets[0] == -1
            directions = [[i, j] for i, j in enumerate(targets) if j != -1]
            assert all(0 < i < 21 and 0 < j < 21 and i != j and (i > 15 or j > 15) for i, j in directions)
            selected = sorted({tuple(sorted(direction)) for direction in directions})
            assert [tuple(entry['class_ids']) for entry in pair['pairs']] == selected
            assert pair['pair_count'] == len(selected) and pair['selected_valid_directions'] == len(directions)
            assert pair['duplicate_reverse_directions_removed'] == len(directions)-len(selected)
            assert pair['skipped_background_directions'] == pair['skipped_self_directions'] == pair['skipped_old_old_directions'] == 0
            assert pair['rate_usage'] == 'Directions select unique hard pairs; confusion rates are never loss weights.'
            per_class = row['per_class']
            assert [item['class_id'] for item in per_class] == list(range(21))
            for item in per_class:
                is_new = item['class_id'] > 15
                assert item['is_new'] is is_new
                assert math.isfinite(item['geometry_tangent_norm_before']) and math.isfinite(item['geometry_tangent_norm_after'])
                if not is_new:
                    assert item['geometry_tangent_norm_before'] == item['geometry_tangent_norm_after'] == 0.
                    assert item['conflict_projected'] is False
                elif row['semantic_gradient_used']:
                    assert item['dot_before'] is not None and item['dot_after'] is not None
                    assert math.isfinite(item['dot_after']) and item['dot_after'] >= -item['roundoff_dot_tolerance']*(1+1e-6)
                    assert item['conflict_projected'] is (item['dot_before'] < 0.)
                else:
                    assert item['semantic_tangent_norm'] is None and item['dot_after'] is None
            new = per_class[16:]
            assert row['conflicting_new_rows'] == sum(item['conflict_projected'] for item in new)
            assert row['active_geometry_new_rows'] == sum(item['geometry_tangent_norm_before'] > 0 for item in new)
            if row['semantic_gradient_used']:
                assert row['local_semantic_gradient_connected'] is True
                assert near(row['new_tangent_dot_sum_before'], sum(item['dot_before'] for item in new), 1e-9)
                assert near(row['new_tangent_dot_sum_after'], sum(item['dot_after'] for item in new), 1e-9)
            if row['conflicting_new_rows'] > 0 and row['removed_geometry_tangent_norm'] > 0:
                corrected_batches.append(row)
        except (AssertionError, KeyError, TypeError, ValueError) as exc:
            failures.append({'iteration': row.get('iteration'), 'error': repr(exc)})
    checks['guard_logs_validate_domains_forward_identity_rowwise_protection_and_scope'] = not failures
    if corrected_batches:
        checks['semantic_guard_actually_removed_conflicting_SEP_components'] = True
    elif finished:
        checks['semantic_guard_actually_removed_conflicting_SEP_components'] = False
    else:
        pending.append(str(STAGE) + ': actual semantic-conflict projection pending')
    details['guard_evidence'] = {'logged_batches': len(guard), 'logged_conflict_corrected_batches': len(corrected_batches),
        'maximum_conflicting_new_rows': max(row['conflicting_new_rows'] for row in guard),
        'semantic_failures': failures, 'note': 'Logged first-order Euclidean component protection, not an AdamW or IoU guarantee.'}


def evaluation(cfg, iteration, pending):
    checks, details = {}, {'iteration': iteration}
    name = f'step2_iter{iteration}'
    part = RUN / 'evaluations' / name
    job, result = required(RUN / 'eval_queue' / (name + '.json'), pending), required(part / 'result.json', pending)
    if job is None or result is None:
        return {'checks': checks, 'details': details}
    checkpoint = STAGE / f'checkpoints/model_iter_{iteration}.pth'
    checks['checkpoint_config_and_stage_provenance'] = (Path(job['checkpoint']) == checkpoint and legal(checkpoint)
        and job['config'] == cfg and job['step'] == result['step'] == 2 and job['iteration'] == result['iteration'] == iteration)
    if checkpoint.is_file():
        checks['checkpoint_SHA_matches_job_and_result'] = digest(checkpoint) == job['checkpoint_sha256'] == result['checkpoint_sha256']
    else:
        pending.append(str(checkpoint))
    hist = np.asarray(result['histogram'])
    valid = hist.shape == (21, 21) and np.issubdtype(hist.dtype, np.integer) and bool((hist >= 0).all())
    checks['histogram_valid_and_full1449'] = valid and result['images'] == 1449
    original = required(ROOT / 'runs/kd_parallel_v1/formal/b_relational/evaluations/step2_iter8000/result.json', pending)
    if valid:
        if original is not None:
            reference = np.asarray(original['histogram'], dtype=np.int64)
            checks['GT_row_totals_match_original_optimized_KD'] = reference.shape == (21, 21) and np.array_equal(hist.sum(1), reference.sum(1))
        union = hist.sum(1)+hist.sum(0)-hist.diagonal()
        iou = np.divide(hist.diagonal(), union, out=np.full(21, np.nan), where=union > 0)*100
        def mean(values):
            return float(np.nanmean(values)) if np.isfinite(values).any() else None
        computed = {'all_miou': mean(iou), 'foreground_miou': mean(iou[1:]),
            'previous_foreground_miou': mean(iou[1:16]), 'current_foreground_miou': mean(iou[16:]),
            'old_initial10_miou': mean(iou[1:11]), 'new_since_initial_miou': mean(iou[11:])}
        checks['reported_mIoU_and_class_IoU_recomputed'] = (all(near(result.get(key), value) for key, value in computed.items())
            and set(result['class_iou']) == set(CLASS_NAMES) and all(near(result['class_iou'][name],
                float(iou[i]) if np.isfinite(iou[i]) else None) for i, name in enumerate(CLASS_NAMES)))
        counters = {'old_gt_to_new_pixels': int(hist[1:16, 16:].sum()), 'old_gt_to_background_pixels': int(hist[1:16, 0].sum()),
            'old_gt_pixels': int(hist[1:16].sum()), 'new_gt_to_old_pixels': int(hist[16:, 1:16].sum()), 'new_gt_pixels': int(hist[16:].sum())}
        checks['reported_confusion_counters_recomputed'] = all(result.get(key) == value for key, value in counters.items())
        details.update(metrics=computed, GT_row_totals=hist.sum(1).tolist(), checkpoint_sha256=result['checkpoint_sha256'])
    parts = [required(part / f'rank{rank}.json', pending) for rank in range(4)]
    if all(value is not None for value in parts):
        seen = [name for record in parts for name in record['images']]
        expected = (Path(cfg['list_folder']) / 'incremental_split' / f'val_{cfg["task"]}_step_3.txt').read_text().splitlines()
        checks['four_shards_disjoint_complete_original1449_split'] = len(seen) == len(set(seen)) == len(expected) == 1449 and set(seen) == set(expected)
        checks['four_shards_correct_rank_checkpoint_stage'] = all(record['rank'] == rank and record['shards'] == 4
            and record['step'] == 2 and record['iteration'] == iteration and record['checkpoint_sha256'] == result['checkpoint_sha256'] for rank, record in enumerate(parts))
        hist_parts = [np.asarray(record['histogram']) for record in parts]
        valid_parts = all(h.shape == (21, 21) and np.issubdtype(h.dtype, np.integer) and bool((h >= 0).all()) for h in hist_parts)
        checks['merged_histogram_matches_four_shards_exactly'] = valid and valid_parts and np.array_equal(np.sum(hist_parts, axis=0), hist)
    checks['no_shard_error_records'] = not list(part.glob('error_rank*.json'))
    return {'checks': checks, 'details': details}


def audit():
    checks, details, pending = {}, {}, []
    manifest = required(RUN / 'study.json', pending)
    cfg = required(STAGE / 'config.json', pending)
    launch = required(STAGE / 'launch.json', pending)
    stage_done = required(STAGE / 'training_complete.json', pending)
    complete = required(RUN / 'training_complete.json', pending)
    evaluated = required(RUN / 'evaluation_workers_complete.json', pending)
    status = required(RUN / 'status.json', pending)
    checks['candidate_files_outputs_and_cache_paths_stay_in_A'] = all(legal(path) for base in [SOURCE, RUN]
        for path in [base, *base.rglob('*')]) and all(legal(path) for path in [ALLOWED / '.kd8tmp',
        ROOT / '.runtime/kd_pixel_v1/8card', ROOT / '.runtime/prototype_sep_v1/8card',
        ROOT / '.runtime/prototype_sep_v1/8card_eval', ROOT / 'pretrained'])
    checks['no_recorded_training_or_evaluation_failure'] = not any((RUN / name).exists() for name in ['failure.json', 'training_failed.json', 'evaluation_failed.json'])
    if manifest is not None:
        sources_and_preservation(manifest, checks, details, pending)
        inheritance(manifest, checks, details, pending)
        preflight(manifest, checks, details, pending)
        if checks['B_source_freeze']:
            sys.path.insert(0, str(SOURCE))
            from model.semantic_protected_sep import semantic_protected_sep
            checks['guard_API_has_no_GT_or_optimizer_or_rate_input'] = list(inspect.signature(semantic_protected_sep).parameters) == [
                'prototypes', 'weighted_geometry_loss', 'weighted_semantic_loss', 'old_classes']
            if cfg is not None:
                configuration_and_resume(manifest, cfg, checks, details, pending)
                guard_logs(checks, details, pending, stage_done is not None)
                details['evaluations'] = [evaluation(cfg, iteration, pending) for iteration in [4000, 6000, 8000]]
                if stage_done is not None:
                    endpoint(STAGE, cfg, 2, checks, details, pending)
    if launch is not None:
        if 'returncode' in launch:
            checks['trainer_launch_exit0'] = launch['returncode'] == 0
        else:
            pending.append(str(STAGE / 'launch.json') + ': trainer still running')
        command = launch['command']
        checks['eight_rank_launch_inherited_teacher_resume'] = ('--nproc_per_node=8' in command
            and command[command.index('--step')+1] == '2' and command[command.index('--prev_checkpoint')+1] == str(TEACHER)
            and command[command.index('--resume_checkpoint')+1] == str(RESUME)
            and str(SOURCE / 'scripts/dist_train_voc_seg_neg.py') in command)
        checks['launch_teacher_SHA_verified'] = Path(launch['teacher']) == TEACHER and digest(TEACHER) == launch['teacher_sha256']
    if stage_done is not None:
        checks['stage2_training_complete_exit0'] = stage_done.get('returncode') == 0
        hash_check(STAGE / 'checkpoints/model_final.pth', stage_done['checkpoint_sha256'], checks, 'final_checkpoint_matches_stage_receipt', pending)
    if complete is not None:
        checks['exactly_one_new_training_stage_completed'] = complete.get('status') == 'complete' and complete.get('stages_trained') == 1 and complete['inherited_stage1'] == str(PARENT)
        jobs = {path.stem for path in (RUN / 'eval_queue').glob('*.json')}
        checks['exactly_three_post_resume_evaluation_jobs'] = jobs == {'step2_iter4000', 'step2_iter6000', 'step2_iter8000'}
    if evaluated is not None:
        checks['all_four_evaluator_returncodes_zero'] = evaluated.get('returncodes') == [0]*4
        hosts = []
        for rank in range(4):
            process = required(RUN / f'evaluator_rank{rank}.process.json', pending)
            if process is not None:
                command = process['command']; hosts.append(process['host'])
                checks[f'evaluator_rank{rank}_exit0_and_four_shard_command'] = (process.get('returncode') == 0
                    and command[command.index('--rank')+1] == str(rank) and command[command.index('--shards')+1] == '4'
                    and str(SOURCE / 'evaluate_kd.py') in command and command[command.index('--run')+1] == str(RUN))
        if launch is not None and len(hosts) == 4:
            checks['training_and_evaluation_on_same8card_host'] = len(set(hosts+[launch['host']])) == 1
    if status is not None:
        if stage_done is not None and complete is not None and evaluated is not None:
            checks['coordinator_completed'] = status.get('status') == 'complete'
        elif status.get('status') == 'failed':
            checks['coordinator_not_failed'] = False
    failed = [key for key, value in checks.items() if not value]
    failed.extend(f'evaluation{record["details"]["iteration"]}: {key}' for record in details.get('evaluations', [])
        for key, value in record['checks'].items() if not value)
    done = not pending and not failed and all(record is not None for record in [manifest, cfg, launch, stage_done, complete, evaluated, status])
    return {'utc': now(), 'status': 'complete' if done else 'invalid' if failed else 'incomplete', 'arm': 'b_semantic',
        'checks': checks, 'failed_checks': failed, 'pending': sorted(set(pending)), 'details': details,
        'all_training_and_endpoint_requirements_complete': done, 'all_semantic_sep_requirements_complete': done,
        'device': 'CPU only; no CUDA, model forward, optimizer, evaluation or remote operations',
        'budget': 'A stage1 inherited; A stage2 iteration2000 all four states restored; only6000 new steps, final observer8000/64000; 8x1 global8',
        'interpretation': 'Completion/integrity and logged first-order semantic protection. This does not establish AdamW update protection or improved final IoU.'}


def save_output(path, result):
    path = safe_path(path)
    if path != RECEIPT:
        raise RuntimeError('Use only independent formal/b_semantic/completion_audit.json; preserve A and earlier study receipts')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = safe_path(path.with_name(path.name + f'.{os.getpid()}.tmp'))
    with temporary.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--output', help='Only independent formal/b_semantic/completion_audit.json is accepted')
    args = parser.parse_args()
    result = audit()
    if args.output:
        save_output(args.output, result)
    print(json.dumps(result, allow_nan=False))
    if args.require_complete and not result['all_semantic_sep_requirements_complete']:
        raise SystemExit(1)
