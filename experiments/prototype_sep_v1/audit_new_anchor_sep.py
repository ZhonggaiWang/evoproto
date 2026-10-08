"""CPU-only final audit for the best-pipeline stage2 add-on C.

Default: read only; missing running evidence remains incomplete. Final receipt:
  python -B audit_new_anchor_sep.py --require-complete --output \
    /ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto/runs/prototype_sep_v1/formal/c_new_anchor/completion_audit.json

C copies frozen B and changes only its geometry selector to schema4/current-new
source rows. Its teacher and iteration2000 student and
optimizer come from the archived optimized-KD pipeline. That checkpoint has no
online states: the observer and schema4 geometry selector start fresh, yielding
6000 updates/48000 image exposures at iteration8000. Only trusted current-new
source rows may select geometry competitors. A/B sources and receipts are preserved.
No CUDA, model forward, new tests, training, inference or remote operations.
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
E = ROOT / 'experiments/prototype_sep_v1'
U = ROOT / 'runs/prototype_sep_v1'
OPT = ROOT / 'runs/kd_parallel_v1/formal/b_relational'
PRIOR_A = U / 'formal/a_geometry'
PRIOR_B = U / 'formal/b_semantic'
PRIOR_SOURCE = E / 'b_semantic/src'
SOURCE = E / 'c_new_anchor/src'
RUN = U / 'formal/c_new_anchor'
STAGE = RUN / '10-5/step2'
SMOKE = U / 'smoke/c_new_anchor'
RECEIPT = RUN / 'completion_audit.json'
TEACHER = OPT / '10-5/step1/checkpoints/model_final.pth'
RESUME = OPT / '10-5/step2/checkpoints/model_iter_2000.pth'
TEACHER_SHA = '6d3d54769b6a54e1700c7cd352b24e4bb95f4ccf398bc200b74f60795780ab33'
RESUME_SHA = 'feb5b228b05934e5acb0e861c49f16640f13219ce7ab477aa66449b78520f153'
SELECTOR_TEST_SHA = 'dc6b25d14b59bc53999f47999efebe8921d1295549069363f502a6e2296a88a3'
UNIT_SHA = 'e17b1c3657d3fff41b3cb2b322ee4b3b4af964882e3fe327f8b6d0c86c488760'
DDP_SHA = 'a77840937d44f11c322b58c4f68c99d3276407aeced02d3a64d04d2f3d5e57e0'
DERIVED_KEYS = {'local_rank', 'step', 'work_dir', 'prev_checkpoint', 'resume_checkpoint', 'ckpt_dir', 'pred_dir'}
GEOMETRY_CONFIG = {'w_geometry_sep': .1, 'confusion_momentum': .98,
    'pair_refresh_interval': 50, 'pair_min_row_images': 8, 'pair_min_pair_images': 3,
    'pair_min_rate': .01, 'pair_min_updates': 100, 'pair_ramp_updates': 200,
    'pair_max_stale_updates': 200}
SMOKE_OVERRIDES = {'max_iters': 2008, 'log_iters': 1, 'eval_iters': 2008,
    'val_limit': 16, 'train_limit': 64, 'pair_min_updates': 0, 'pair_ramp_updates': 1,
    'pair_min_row_images': 1, 'pair_min_pair_images': 1, 'pair_min_rate': 0., 'pair_refresh_interval': 1}

sys.path.insert(0, str(E))
from audit_prototype_sep import (ALLOWED, CLASS_NAMES, STEP0, STEP0_SHA, ARCHIVE_SHA,
    OPTIMIZED_ENDPOINTS, safe_path, legal, digest, required, near, now, same_state,
    complete_optimizer, load_checkpoint, source_audit, hash_check)
import numpy as np
import torch

torch.set_num_threads(2)


def metric_rows(path, pending):
    path = safe_path(path)
    if not path.is_file():
        pending.append(str(path))
        return []
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pending.append(f'Not yet readable metric line: {path}:{number}')
    return rows


def module_states(observer_state, selector_state, cfg, step):
    """Reconstruct CPU modules; no observation, selection update or forward."""
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
    targets, rates = selector.targets, selector.selected_rates
    ids = torch.arange(classes)
    valid_directions = (targets == -1) | ((ids > old) & (targets > 0) & (targets < classes) & (targets != ids))
    buffers = list(observer.buffers()) + list(selector.buffers())
    finite = all(not value.is_floating_point() or bool(torch.isfinite(value).all()) for value in buffers)
    nonnegative = all(bool((getattr(observer, name) >= 0).all()) for name in [
        'counts', 'accepted_counts', 'broad_counts', 'ema_counts', 'broad_ema_counts',
        'pair_image_observations', 'broad_pair_image_observations', 'image_observations',
        'ema_image_observations', 'broad_image_observations'])
    exact = same_state(observer_state, observer.state_dict()) and same_state(selector_state, selector.state_dict())
    old_sources_empty = bool((targets[:old + 1] == -1).all()) and bool((rates[:old + 1] == 0).all())
    valid = (exact and finite and nonnegative and expected_observer['schema'] == 2
        and expected_selector['schema'] == 4 and expected_selector['source_anchor_policy'] == 'current_new_only'
        and old_sources_empty and bool(valid_directions.all()) and bool(((rates >= 0) & (rates <= 1)).all())
        and bool((rates[targets == -1] == 0).all()))
    info.update(observer_metadata=observer_state.get('_extra_state'), selector_metadata=selector_state.get('_extra_state'),
        strict_restored_state_tensor_exact=exact, finite_buffers=finite, nonnegative_counts=nonnegative,
        valid_new_source_directions=bool(valid_directions.all()), background_and_old_sources_empty=old_sources_empty,
        targets=targets.tolist(), selected_rates=rates.tolist(), updates=int(observer.updates), seen_images=int(observer.seen_images),
        last_refresh_iteration=int(selector.last_refresh_iteration))
    return valid, info


def sources_and_shared_preflight(manifest, checks, details, pending):
    bstudy = required(PRIOR_B / 'study.json', pending)
    checks['C_isolated_new_anchor_source_freeze'], details['source'] = source_audit(SOURCE, manifest['source_sha256'])
    checks['source_path_is_isolated_new_anchor_candidate'] = manifest.get('source') == str(SOURCE)
    hash_check(E / 'run_new_anchor_sep.py', manifest['runner_sha256'], checks, 'C_runner_freeze', pending)
    hash_check(E / 'launch_new_anchor_sep.py', manifest['launcher_sha256'], checks, 'C_launcher_freeze', pending)
    pre = required(E / 'semantic_preflight.json', pending)
    tests = required(E / 'semantic_test_receipts.json', pending)
    if bstudy is not None:
        checks['B_source_unchanged'], details['B_source'] = source_audit(PRIOR_SOURCE, bstudy['source_sha256'])
        before, after = bstudy['source_sha256'], manifest['source_sha256']
        changed = [key for key in sorted(set(before) | set(after)) if before.get(key) != after.get(key)]
        checks['C_only_selector_changed_from_frozen_B'] = set(before) == set(after) and changed == ['model/geometry_pair_selector.py']
        checks['manifest_parent_link_and_new_source_domain'] = (
            manifest.get('parent_source') == str(PRIOR_SOURCE) and manifest.get('parent_source_sha256') == before
            and manifest.get('changed_files_from_B') == ['model/geometry_pair_selector.py']
            and manifest.get('source_reused_without_modification') is False
            and manifest.get('selector_schema') == 4 and manifest.get('source_anchor_policy') == 'current_new_only'
            and manifest.get('source_domain') == 'new_foreground_rows_only'
            and manifest.get('KD_byte_identical_to_B') is True and manifest.get('semantic_guard_byte_identical_to_B') is True)
        details['changes_from_B'] = changed
        hash_check(E / 'run_semantic_sep.py', bstudy['runner_sha256'], checks, 'prior_B_runner_preserved', pending)
    if pre is not None and tests is not None and bstudy is not None:
        checks['shared_B_preflight_exact_source_passed_no_retests'] = (pre.get('passed') is True and tests.get('passed') is True
            and pre['source_sha256'] == tests['source_sha256'] == bstudy['source_sha256']
            and pre['runner_sha256'] == bstudy['runner_sha256'] and pre['smoke_complete'] is True)
        module_sha = digest(SOURCE / 'model/semantic_protected_sep.py')
        for key, filename, sha in [('unit_tests', 'test_semantic_protected_sep.py', UNIT_SHA),
                                  ('ddp_2rank', 'test_semantic_protected_sep_ddp.py', DDP_SHA),
                                  ('ddp_8rank', 'test_semantic_protected_sep_ddp.py', DDP_SHA)]:
            record = tests[key]
            checks[key + '_shared_passed_exact_module'] = (pre[key] == record and record.get('passed') is True
                and record.get('returncode') == 0 and record['source_sha256'] == sha and record['module_sha256'] == module_sha)
            hash_check(E / filename, sha, checks, key + '_test_source_preserved', pending)
            hash_check(record['log'], record['evidence_sha256'], checks, key + '_test_log_preserved', pending)
            if key == 'unit_tests':
                checks['shared23_CPU_tests_passed'] = record.get('count') == 23
            else:
                result, ranks = record['result'], 2 if key == 'ddp_2rank' else 8
                checks[key + '_shared_true_DDP_pooled_semantic_contract'] = (result.get('passed') is True
                    and result['ranks'] == ranks and result['backend'] == ('gloo' if ranks == 2 else 'nccl')
                    and result['sibling_output_probe']['find_unused_parameters'] is True
                    and result['sibling_output_probe']['semantic_to_returned_p_disconnected'] is True
                    and result['sibling_output_probe']['unsafe_external_helper_explicitly_rejected'] is True
                    and {row['case'] for row in result['cases']} == {'uneven_all_supported', 'uneven_with_zero_rank', 'all_zero_semantic'}
                    and all(row['inside_forward_partial_leaf_hooks_zero'] and row['final_leaf_hooks_once']
                            and row['unused_projector_grad_none'] for row in result['cases']))
        hash_check(E / 'semantic_preflight.json', manifest['semantic_preflight_sha256'], checks, 'shared_preflight_receipt_SHA', pending)
        checks['shared_preflight_path_is_B_receipt'] = manifest['semantic_preflight'] == str(E / 'semantic_preflight.json')
        details['shared_B_preflight'] = pre
    origin = required(E / 'new_anchor_origin.json', pending)
    selected_tests = required(E / 'new_anchor_selector_tests.json', pending)
    if origin is not None and bstudy is not None:
        checks['new_anchor_origin_exact_parent_and_candidate_hash_link'] = (
            origin.get('parent_source') == str(PRIOR_SOURCE) and origin.get('candidate_source') == str(SOURCE)
            and origin.get('parent_source_sha256') == bstudy['source_sha256']
            and origin.get('source_sha256') == origin.get('candidate_source_sha256') == manifest['source_sha256']
            and origin.get('changed_files') == ['model/geometry_pair_selector.py']
            and origin.get('selector_schema') == 4 and origin.get('source_anchor_policy') == 'current_new_only'
            and origin.get('source_domain') == 'new_foreground_rows_only'
            and origin.get('parent_source_unchanged') is True and origin.get('KD_byte_identical') is True
            and origin.get('semantic_guard_byte_identical') is True)
        details['new_anchor_origin'] = origin
    if selected_tests is not None:
        checks['ten_new_anchor_selector_CPU_tests_passed_exact_candidate'] = (
            selected_tests.get('passed') is True and selected_tests.get('returncode') == 0
            and selected_tests.get('count') == 10 and selected_tests.get('test_source_sha256') == SELECTOR_TEST_SHA
            and selected_tests.get('module_sha256') == manifest['source_sha256']['model/geometry_pair_selector.py']
            and selected_tests.get('source_sha256') == manifest['source_sha256'])
        hash_check(E / 'test_new_anchor_selector.py', SELECTOR_TEST_SHA, checks, 'new_selector_tests_source_frozen', pending)
        hash_check(E / 'new_anchor_geometry_selector.py', manifest['source_sha256']['model/geometry_pair_selector.py'],
            checks, 'isolated_selector_matches_prepared_module', pending)
        if selected_tests.get('log') and selected_tests.get('evidence_sha256'):
            hash_check(selected_tests['log'], selected_tests['evidence_sha256'], checks, 'new_selector_execution_log_SHA', pending)
        details['new_anchor_selector_CPU_receipt'] = selected_tests
    for key, path in [('new_anchor_origin', E / 'new_anchor_origin.json'),
                      ('new_anchor_selector_tests', E / 'new_anchor_selector_tests.json')]:
        if key in manifest and key + '_sha256' in manifest:
            checks[key + '_manifest_receipt_path'] = manifest[key] == str(path)
            hash_check(path, manifest[key + '_sha256'], checks, key + '_manifest_receipt_SHA', pending)
        else:
            checks[key + '_manifest_receipt_hash_link_present'] = False
    origin = required(E / 'origin.json', pending)
    if origin is not None:
        original = ROOT / 'experiments/kd_parallel_v1/b_relational/src'
        checks['original_optimized_KD_source_preserved'], details['original_source'] = source_audit(original, origin['optimized_kd_source_sha256'])
        checks['optimized_pixel_KD_byte_identical'] = digest(original / 'model/pixel_kd.py') == manifest['source_sha256']['model/pixel_kd.py']
    hash_check(STEP0, STEP0_SHA, checks, 'shared_step0_original_SHA_preserved', pending)
    hash_check(ROOT / 'runs/fixed_baseline_v1/source_snapshot.tar.gz', ARCHIVE_SHA, checks, 'original_baseline_archive_preserved', pending)
    for rel, sha in OPTIMIZED_ENDPOINTS.items():
        hash_check(OPT / '10-5' / rel, sha, checks, 'original_optimized_KD_' + rel.replace('/', '_') + '_preserved', pending)
    if checks['C_isolated_new_anchor_source_freeze']:
        sys.path.insert(0, str(SOURCE))
        from model.semantic_protected_sep import semantic_protected_sep
        checks['guard_API_has_no_GT_optimizer_or_rate_input'] = list(inspect.signature(semantic_protected_sep).parameters) == [
            'prototypes', 'weighted_geometry_loss', 'weighted_semantic_loss', 'old_classes']
        tree = ast.parse((SOURCE / 'continual/Trainer.py').read_text())
        calls = [(ast.unparse(node.func), ast.unparse(node.args[0]) if node.args else '',
                  {key.arg: ast.unparse(key.value) for key in node.keywords}) for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'load_state_dict']
        checks['frozen_source_restores_student_strict_and_full_optimizer'] = (
            any(func in {'model.load_state_dict', 'self.model.load_state_dict'} and value == "saved['model_state']"
                and keywords.get('strict') == 'True' for func, value, keywords in calls)
            and any(func == 'optim.load_state_dict' and value == "saved['optimizer_state']" for func, value, _ in calls))
        checks['frozen_source_restores_optimizer_schedule_iteration'] = any(isinstance(node, ast.Assign)
            and ast.unparse(node.value) == 'start_iteration' and any(ast.unparse(target) == 'optim.global_step' for target in node.targets)
            for node in ast.walk(tree))


def inheritance_and_resume(manifest, checks, details, pending):
    checks['only_stage2_addon_no_new_stage1'] = not (RUN / '10-5/step1').exists()
    checks['manifest_best_pipeline_model_optimizer_resume_fresh_online'] = (
        manifest.get('arm') == 'c_new_anchor' and manifest.get('smoke') is False
        and manifest.get('inherited_stage1_run') == str(OPT) and Path(manifest['teacher']) == TEACHER
        and Path(manifest['resume_checkpoint']) == Path(manifest['shared_stage2_warmup']) == RESUME
        and manifest.get('resume_iteration') == 2000 and manifest.get('remaining_training_updates') == 6000
        and manifest.get('restored_state') == ['student', 'optimizer']
        and manifest.get('fresh_state') == ['online_confusion', 'geometry_selector']
        and manifest.get('observer_initialization') == 'fresh at stage2 iteration2000; no borrowed A/B online state'
        and manifest.get('expected_final_observer_updates') == 6000 and manifest.get('expected_final_observer_seen_images') == 48000
        and manifest.get('formal_online_coldstart_updates') == 100 and manifest.get('formal_online_ramp_updates') == 200
        and manifest.get('smoke_overrides') == {} and manifest.get('new_stage1_training') is False
        and manifest.get('step0_retrained') is False and manifest.get('baseline_retrained') is False)
    hash_check(TEACHER, TEACHER_SHA, checks, 'original_optimized_KD_stage1_teacher_SHA', pending)
    hash_check(RESUME, RESUME_SHA, checks, 'original_optimized_KD_stage2_warmup_SHA', pending)
    checks['manifest_original_checkpoint_SHA_identities'] = manifest['teacher_sha256'] == TEACHER_SHA and manifest['resume_checkpoint_sha256'] == manifest['shared_stage2_warmup_sha256'] == RESUME_SHA
    teacher_done = required(OPT / '10-5/step1/training_complete.json', pending)
    job = required(OPT / 'eval_queue/step2_iter2000.json', pending)
    result = required(OPT / 'evaluations/step2_iter2000/result.json', pending)
    old_complete = required(OPT / 'training_complete.json', pending)
    old_evaluated = required(OPT / 'evaluation_workers_complete.json', pending)
    if teacher_done is not None:
        checks['teacher_from_completed_original_stage1'] = teacher_done.get('returncode') == 0 and teacher_done['checkpoint_sha256'] == TEACHER_SHA
    if old_complete is not None and old_evaluated is not None:
        checks['original_optimized_KD_history_complete'] = (old_complete.get('status') == 'training_complete'
            and old_complete.get('stages') == 2 and old_evaluated.get('returncodes') == [0, 0])
    for step in (1, 2):
        old_launch = required(OPT / f'10-5/step{step}/launch.json', pending)
        if old_launch is not None:
            checks[f'original_optimized_KD_step{step}_launch_exit0'] = old_launch.get('returncode') == 0
    if job is not None and result is not None:
        checks['warmup_SHA_matches_original_job_and_full1449_result'] = (Path(job['checkpoint']) == RESUME
            and job['step'] == result['step'] == 2 and job['iteration'] == result['iteration'] == 2000
            and result['images'] == 1449 and job['checkpoint_sha256'] == result['checkpoint_sha256'] == RESUME_SHA)
    resume = load_checkpoint(RESUME, pending)
    if resume is not None:
        optimizer = resume.get('optimizer_state', {})
        checks['resume_cpu2000_full_student_optimizer_available'] = (resume.get('iteration') == 2000
            and isinstance(resume.get('model_state'), dict) and bool(resume['model_state']) and complete_optimizer(optimizer))
        checks['resume_has_no_online_state_to_borrow'] = all(resume.get(key) is None for key in
            ['online_confusion_state', 'geometry_selector_state', 'pair_selector_state'])
        checks['original_four_group_AdamW_state_at2000'] = (len(optimizer.get('param_groups', [])) == 4
            and bool(optimizer.get('state')) and all(int(value['step']) == 2000 for value in optimizer.get('state', {}).values() if 'step' in value))
        verified = manifest['restoration_verified']
        checks['runner_CPU_restore_metadata_matches_actual_original_checkpoint'] = (
            verified.get('iteration') == 2000 and verified.get('model_tensors') == len(resume.get('model_state', {}))
            and verified.get('optimizer_parameter_groups') == len(optimizer.get('param_groups', []))
            and verified.get('optimizer_state_entries') == len(optimizer.get('state', {}))
            and verified.get('observer_state_present') is False and verified.get('selector_state_present') is False
            and verified.get('teacher_sha256') == TEACHER_SHA and verified.get('resume_checkpoint_sha256') == RESUME_SHA)
        details['resume'] = {'iteration': resume.get('iteration'), 'checkpoint_keys': sorted(resume),
            'model_tensors': len(resume.get('model_state', {})), 'optimizer_parameter_groups': len(optimizer.get('param_groups', [])),
            'optimizer_state_entries': len(optimizer.get('state', {})), 'observer_and_selector_initialized_fresh': checks['resume_has_no_online_state_to_borrow']}
        del resume
    for name, path in [('A_geometry', PRIOR_A), ('B_semantic', PRIOR_B)]:
        audit, status = required(path / 'completion_audit.json', pending), required(path / 'status.json', pending)
        if audit is not None and status is not None:
            checks[name + '_audit_and_complete_status_preserved'] = audit.get('status') == status.get('status') == 'complete' and audit.get('all_training_and_endpoint_requirements_complete') is True
    analysis = required(U / 'semantic_analysis.json', pending)
    if analysis is not None:
        checks['completed_B_failure_supports_conditional_addon'] = (analysis.get('status') == 'complete_endpoint_analysis'
            and analysis.get('completion', {}).get('complete') is True and analysis.get('run') == str(PRIOR_B)
            and analysis['recommendation'].get('candidate_exceeds_required_endpoint') is False
            and analysis['candidate_endpoint']['metrics']['all_miou'] <= analysis['references']['optimized_KD']['metrics']['all_miou'])
        hash_check(U / 'semantic_analysis.json', manifest['B_decision_analysis_sha256'], checks, 'B_gate_analysis_SHA_preserved', pending)
        checks['B_gate_receipt_matches_decision'] = manifest['B_decision_analysis'] == str(U / 'semantic_analysis.json') and manifest['B_decision'] == analysis['recommendation']
    details['inheritance'] = {'teacher': str(TEACHER), 'teacher_sha256': TEACHER_SHA, 'resume': str(RESUME),
        'resume_sha256': RESUME_SHA, 'new_stage1_training': False, 'new_training_updates': 6000,
        'final_online_budget': 'Fresh6000 updates/48000 repeated image exposures; no inherited A/B online history',
        'interpretation': 'Best-pipeline add-on; teacher, student/optimizer trajectory and online initialization change together. Not a teacher causal ablation.'}


def configuration(manifest, cfg, checks, details, pending):
    original = required(OPT / '10-5/step2/config.json', pending)
    prior = required(PRIOR_B / '10-5/step2/config.json', pending)
    if original is not None:
        common = {key: value for key, value in original.items() if key not in DERIVED_KEYS}
        common.update(GEOMETRY_CONFIG)
        common.update(spg=1, num_workers=2, async_eval=True)
        checks['common_config_only_explicit_addon_changes_to_best_pipeline'] = common == manifest['common_config']
        details['config_changes_from_original'] = {key: {'original': original.get(key), 'C': common.get(key)}
            for key in set(original) | set(common) if key not in DERIVED_KEYS and original.get(key) != common.get(key)}
    expected = {**manifest['common_config'], **GEOMETRY_CONFIG,
        'step': 2, 'spg': 1, 'max_iters': 8000, 'warmup_iters': 2000, 'loss_warmup_iters': 2000,
        'log_iters': 50, 'eval_iters': 2000, 'seed': 0, 'train_limit': 0, 'val_limit': 0,
        'async_eval': True, 'save_ckpt': True, 'w_pixel_kd': .1, 'kd_temperature': 2.,
        'w_proto_seg': .1, 'w_proto_sep': 0., 'w_proto_kd': 0., 'proto_margin': 0., 'ald': False, 'confusion_reweight': False}
    mismatch = {key: {'expected': value, 'actual': cfg.get(key)} for key, value in expected.items() if cfg.get(key) != value}
    checks['formal_original100_coldstart200_ramp_and_fixed_strength_budget'] = not mismatch
    details['config_mismatches'] = mismatch
    checks['eight_times_one_global_batch8'] = manifest['train_gpus'] == list(range(8)) and manifest['batch_per_gpu'] == cfg['spg'] == 1 and manifest['global_batch'] == 8
    checks['expected_original_teacher_resume_and_safe_output_paths'] = (Path(cfg['prev_checkpoint']) == TEACHER
        and Path(cfg['resume_checkpoint']) == RESUME and Path(cfg['work_dir']) == STAGE
        and Path(cfg['ckpt_dir']) == STAGE / 'checkpoints'
        and all(legal(cfg[key]) for key in ['prev_checkpoint', 'resume_checkpoint', 'work_dir', 'ckpt_dir', 'pred_dir']))
    if prior is not None:
        differences = {key: {'B': prior.get(key), 'C': cfg.get(key)} for key in set(prior) | set(cfg)
            if key not in DERIVED_KEYS and prior.get(key) != cfg.get(key)}
        checks['no_new_loss_or_formal_threshold_changes_from_tested_B'] = not differences
        details['config_changes_from_B'] = differences
    log = STAGE / 'train.log'
    if log.is_file():
        text = log.read_text()
        checks['actual_original_model_optimizer_resume_at2000_logged'] = 'Resume student AND optimizer at iteration 2000' in text and str(RESUME) in text
        checks['original_stage1_teacher_and_global8_logged'] = str(TEACHER) in text and 'Total gpus: 8, samples per gpu: 1' in text
    else:
        pending.append(str(log))


def logs(stage, cfg, final_iteration, checks, details, pending, finished, prefix):
    guard = metric_rows(stage / 'semantic_guard_metrics.jsonl', pending)
    geometry = metric_rows(stage / 'geometry_metrics.jsonl', pending)
    kd = metric_rows(stage / 'kd_metrics.jsonl', pending)
    if not guard or not geometry or not kd:
        pending.append(str(stage) + ': guard/geometry/KD logs not yet populated')
        return
    iterations = [row['iteration'] for row in guard]
    aligned = iterations == [row['iteration'] for row in geometry] == [row['iteration'] for row in kd]
    if finished or aligned:
        checks[prefix + '_guard_geometry_KD_iterations_match'] = aligned
    else:
        pending.append(str(stage) + ': metric append synchronization pending')
    if finished:
        checks[prefix + '_all_post_resume_logged_batches'] = iterations == list(range(2000 + cfg['log_iters'], final_iteration + 1, cfg['log_iters']))
    checks[prefix + '_legacy_KD_weight_global8_unchanged'] = all(row['step'] == 2 and row['weight'] == .1
        and row['global_batch'] == 8 and not any(key.startswith('kd_pair_') for key in row) for row in kd)
    failures, active_geometry, used_semantic, corrected = [], [], [], []
    for row, pair in zip(guard, geometry):
        try:
            iteration = row['iteration']
            prior_updates = iteration - 2000 - 1
            ramp = min(1., max(0., (prior_updates - cfg['pair_min_updates']) / cfg['pair_ramp_updates']))
            assert row['step'] == pair['step'] == 2 and row['iteration'] == pair['iteration']
            assert row['old_classes'] == 15 and row['classes_including_background'] == 21 and row['world_size'] == 8
            assert pair['old_classes'] == 15 and pair['total_classes_including_background'] == 21
            assert row['active'] is True and row['enabled'] is True and pair['active'] is True
            assert row['weight'] == pair['weight'] == .1 and near(row['ramp'], ramp) and near(pair['ramp'], ramp)
            assert near(row['effective_geometry_weight'], .1*ramp)
            assert row['forward_value_preserved'] is True and row['background_old_sep_gradient_zero'] is True
            assert near(row['weighted_geometry_forward_value'], .1*ramp*pair['prototype_sep'], 5e-7)
            assert row['no_additive_denominator_epsilon'] is True and row['no_norm_cap'] is True and row['higher_order_gradients'] is False
            assert pair['semantic_protection'] == {key: value for key, value in row.items() if key not in ['step', 'iteration', 'active', 'weight', 'ramp']}
            selector = pair['selector']
            assert selector['schema'] == 4 and selector['classes'] == 21 and selector['stage'] == 2 and selector['old_classes'] == 15
            assert selector['source_anchor_policy'] == 'current_new_only'
            for key in ('refresh_interval', 'min_row_images', 'min_pair_images', 'min_rate', 'min_updates', 'ramp_updates', 'max_stale_updates'):
                assert selector[key] == cfg['pair_' + key]
            targets = selector['targets']
            assert len(targets) == 21 and targets[:16] == [-1]*16
            assert selector['selected_rates'][:16] == [0.]*16
            directions = [[i, j] for i, j in enumerate(targets) if j != -1]
            assert all(15 < i < 21 and 0 < j < 21 and i != j for i, j in directions)
            selected = sorted({tuple(sorted(direction)) for direction in directions})
            assert [tuple(entry['class_ids']) for entry in pair['pairs']] == selected
            assert pair['pair_count'] == len(selected) and pair['selected_valid_directions'] == len(directions)
            assert pair['duplicate_reverse_directions_removed'] == len(directions)-len(selected)
            assert pair['skipped_background_directions'] == pair['skipped_self_directions'] == pair['skipped_old_old_directions'] == 0
            assert pair['skipped_missing_directions'] == 21-len(directions)
            assert pair['margin'] == 0. and pair['old_reference'] == 'current_student_prototype_detached_in_SEP_only'
            assert pair['objective'] == 'mean_over_all_selected_unique_pairs_of_relu_cosine_minus_margin_squared'
            assert pair['rate_usage'] == 'Directions select unique hard pairs; confusion rates are never loss weights.'
            old_new, new_new, hinge_active = 0, 0, 0
            for entry in pair['pairs']:
                first, second = entry['class_ids']
                is_old = first <= 15
                assert entry['pair_type'] == ('old_new' if is_old else 'new_new')
                assert entry['old_endpoint_detached'] == (first if is_old else None)
                assert entry['gradient_class_ids'] == ([second] if is_old else [first, second])
                assert entry['selected_directions'] == [direction for direction in directions if tuple(sorted(direction)) == (first, second)]
                cosine = entry['cosine_similarity']
                assert math.isfinite(cosine) and -1 <= cosine <= 1 and entry['margin'] == 0.
                assert entry['active'] is (cosine > 0) and near(entry['pair_loss'], max(cosine, 0.)**2, 5e-7)
                old_new += int(is_old); new_new += int(not is_old); hinge_active += int(entry['active'])
            assert pair['old_new_pair_count'] == old_new and pair['new_new_pair_count'] == new_new and pair['active_pair_count'] == hinge_active
            mean = sum(entry['pair_loss'] for entry in pair['pairs'])/len(selected) if selected else 0.
            assert near(pair['prototype_sep'], mean, 5e-7)
            if prior_updates < cfg['pair_min_updates']:
                assert not selected and ramp == 0.
            if ramp == 0. or pair['prototype_sep'] == 0.:
                assert row['semantic_gradient_used'] is False
            per_class = row['per_class']
            assert [item['class_id'] for item in per_class] == list(range(21))
            for item in per_class:
                is_new = item['class_id'] > 15
                assert item['is_new'] is is_new
                assert math.isfinite(item['geometry_tangent_norm_before']) and math.isfinite(item['geometry_tangent_norm_after'])
                if not is_new:
                    assert item['geometry_tangent_norm_before'] == item['geometry_tangent_norm_after'] == 0. and item['conflict_projected'] is False
                elif row['semantic_gradient_used']:
                    assert item['dot_before'] is not None and item['dot_after'] is not None
                    assert math.isfinite(item['dot_after']) and item['dot_after'] >= -item['roundoff_dot_tolerance']*(1+1e-6)
                    assert item['conflict_projected'] is (item['dot_before'] < 0.)
                else:
                    assert item['semantic_tangent_norm'] is None and item['dot_after'] is None and item['conflict_projected'] is False
            new = per_class[16:]
            assert row['conflicting_new_rows'] == sum(item['conflict_projected'] for item in new)
            assert row['active_geometry_new_rows'] == sum(item['geometry_tangent_norm_before'] > 0 for item in new)
            if row['semantic_gradient_used']:
                assert isinstance(row['local_semantic_gradient_connected'], bool)
                assert near(row['new_tangent_dot_sum_before'], sum(item['dot_before'] for item in new), 1e-9)
                assert near(row['new_tangent_dot_sum_after'], sum(item['dot_after'] for item in new), 1e-9)
                used_semantic.append(row)
            if ramp > 0 and hinge_active and pair['prototype_sep'] > 0:
                active_geometry.append(pair)
            if row['conflicting_new_rows'] > 0 and row['removed_geometry_tangent_norm'] > 0:
                corrected.append(row)
        except (AssertionError, KeyError, TypeError, ValueError) as exc:
            failures.append({'iteration': row.get('iteration'), 'error': repr(exc)})
    checks[prefix + '_logs_validate_fresh_ramp_geometry_and_semantic_protection'] = not failures
    if active_geometry and used_semantic:
        checks[prefix + '_geometry_and_semantic_protection_really_activated'] = True
    elif finished:
        checks[prefix + '_geometry_and_semantic_protection_really_activated'] = False
    else:
        pending.append(str(stage) + ': later nonzero geometry/semantic activation pending')
    details[prefix + '_mechanism'] = {'logged_batches': len(guard), 'nonzero_geometry_batches': len(active_geometry),
        'semantic_gradient_used_batches': len(used_semantic), 'conflict_corrected_batches': len(corrected),
        'maximum_conflicting_new_rows': max(row['conflicting_new_rows'] for row in guard),
        'validation_failures': failures, 'initial_ramp': guard[0]['ramp'], 'final_ramp': guard[-1]['ramp'],
        'scope': 'First-order logged pooled-semantic tangent protection, not AdamW or IoU guarantee; rates rank only.'}


def smoke(manifest, cfg, checks, details, pending):
    sstudy = required(SMOKE / 'study.json', pending)
    scfg = required(SMOKE / '10-5/step2/config.json', pending)
    status = required(SMOKE / 'status.json', pending)
    done = required(SMOKE / '10-5/step2/training_complete.json', pending)
    launch = required(SMOKE / '10-5/step2/launch.json', pending)
    evaluated = required(SMOKE / 'evaluation_workers_complete.json', pending)
    if any(value is None for value in (sstudy, scfg, status, done, launch, evaluated)):
        return
    if status.get('status') != 'complete':
        if status.get('status') == 'failed':
            checks['C_specific_restore_smoke_not_failed'] = False
        else:
            pending.append(str(SMOKE) + ': C-specific smoke incomplete')
        return
    checks['C_specific_8rank_real_restore_smoke_exit0'] = launch.get('returncode') == done.get('returncode') == 0 and evaluated.get('returncodes') == [0]*4
    checks['C_smoke_matches_same_source_scripts_and_original_checkpoints'] = (
        sstudy.get('arm') == 'c_new_anchor' and sstudy.get('smoke') is True
        and sstudy['source_sha256'] == manifest['source_sha256'] and sstudy['runner_sha256'] == manifest['runner_sha256']
        and sstudy['launcher_sha256'] == manifest['launcher_sha256'] and sstudy['teacher_sha256'] == TEACHER_SHA
        and sstudy['resume_checkpoint_sha256'] == RESUME_SHA and sstudy.get('restored_state') == ['student', 'optimizer']
        and sstudy.get('fresh_state') == ['online_confusion', 'geometry_selector'] and sstudy.get('expected_final_observer_updates') == 8
        and sstudy.get('expected_final_observer_seen_images') == 64 and sstudy.get('smoke_overrides') == SMOKE_OVERRIDES)
    expected = dict(manifest['common_config']); expected.update(SMOKE_OVERRIDES)
    checks['only_smoke_uses_low_support_thresholds'] = sstudy['common_config'] == expected and all(scfg.get(key) == value for key, value in expected.items())
    checks['C_formal_gate_requires_own_smoke'] = manifest.get('smoke_restoration_receipt') == {
        'run': str(SMOKE), 'eight_updates': True, 'active_protection_observed': True}
    stage = SMOKE / '10-5/step2'
    logs(stage, scfg, 2008, checks, details, pending, True, 'smoke')
    geometry = metric_rows(stage / 'geometry_metrics.jsonl', pending)
    guard = metric_rows(stage / 'semantic_guard_metrics.jsonl', pending)
    if geometry and guard:
        checks['smoke_first_update_proves_empty_past_state_and_zero_ramp'] = (
            geometry[0]['iteration'] == guard[0]['iteration'] == 2001 and geometry[0]['ramp'] == guard[0]['ramp'] == 0.
            and geometry[0]['pair_count'] == 0 and geometry[0]['selector']['targets'] == [-1]*21
            and guard[0]['semantic_gradient_used'] is False)
    final = stage / 'checkpoints/model_final.pth'
    hash_check(final, done['checkpoint_sha256'], checks, 'smoke_final_matches_training_receipt', pending)
    saved = load_checkpoint(final, pending)
    if saved is not None:
        valid, info = module_states(saved.get('online_confusion_state'), saved.get('geometry_selector_state'), scfg, 2)
        checks['smoke_saved_online_states_are_fresh8_updates64_images'] = valid and saved.get('iteration') == 2008 and info.get('updates') == 8 and info.get('seen_images') == 64
        details['smoke_final_online_state'] = info
        del saved
    result = evaluation(SMOKE, scfg, 2008, 16, pending, compare_original_GT=False)
    details['smoke_evaluation'] = result
    checks.update({'smoke_eval_' + key: value for key, value in result['checks'].items()})


def endpoint(cfg, checks, details, pending):
    final, evaluated = STAGE / 'checkpoints/model_final.pth', STAGE / 'checkpoints/model_iter_8000.pth'
    a, b = load_checkpoint(final, pending), load_checkpoint(evaluated, pending)
    if a is None or b is None:
        return
    checks['final_and_evaluated_iteration8000'] = a.get('iteration') == b.get('iteration') == 8000
    checks['final_model_equals_evaluated_model_tensor_exact'] = isinstance(a.get('model_state'), dict) and bool(a['model_state']) and same_state(a['model_state'], b.get('model_state'))
    checks['final_model_finite'] = all(not value.is_floating_point() or bool(torch.isfinite(value).all())
        for value in a.get('model_state', {}).values() if isinstance(value, torch.Tensor))
    checks['final_full_optimizer_equals_evaluated_tensor_exact'] = complete_optimizer(a.get('optimizer_state')) and complete_optimizer(b.get('optimizer_state')) and same_state(a['optimizer_state'], b['optimizer_state'])
    va, ia = module_states(a.get('online_confusion_state'), a.get('geometry_selector_state'), cfg, 2)
    vb, ib = module_states(b.get('online_confusion_state'), b.get('geometry_selector_state'), cfg, 2)
    checks['observer_and_schema4_selector_saved_strict_restorable'] = va and vb
    checks['final_observer_equals_evaluated_tensor_exact'] = va and vb and same_state(a['online_confusion_state'], b['online_confusion_state'])
    checks['final_geometry_selector_equals_evaluated_tensor_exact'] = va and vb and same_state(a['geometry_selector_state'], b['geometry_selector_state'])
    if va and vb:
        checks['fresh_observer_executed6000_updates48000_images'] = ia['updates'] == ib['updates'] == 6000 and ia['seen_images'] == ib['seen_images'] == 48000
        checks['selector_refresh_in_fresh_post2000_stage'] = 2100 < ia['last_refresh_iteration'] <= 8000
    details['endpoint'] = {'final_sha256': digest(final), 'evaluated_sha256': digest(evaluated),
        'model_tensor_count': sum(isinstance(value, torch.Tensor) for value in a.get('model_state', {}).values()),
        'final_online_state': ia, 'evaluated_online_state': ib,
        'file_identity_note': 'Final and evaluated file SHA may differ; tensors/state are required exactly equal.'}
    del a, b


def evaluation(run, cfg, iteration, images, pending, compare_original_GT=True):
    checks, details = {}, {'iteration': iteration, 'images': images}
    folder = run / f'evaluations/step2_iter{iteration}'
    job = required(run / f'eval_queue/step2_iter{iteration}.json', pending)
    result = required(folder / 'result.json', pending)
    if job is None or result is None:
        return {'checks': checks, 'details': details}
    checkpoint = run / f'10-5/step2/checkpoints/model_iter_{iteration}.pth'
    checks['checkpoint_config_and_stage_provenance'] = (Path(job['checkpoint']) == checkpoint and legal(checkpoint)
        and job['config'] == cfg and job['step'] == result['step'] == 2 and job['iteration'] == result['iteration'] == iteration)
    if checkpoint.is_file():
        checks['checkpoint_SHA_matches_job_and_result'] = digest(checkpoint) == job['checkpoint_sha256'] == result['checkpoint_sha256']
    else:
        pending.append(str(checkpoint))
    hist = np.asarray(result['histogram'])
    valid = hist.shape == (21, 21) and np.issubdtype(hist.dtype, np.integer) and bool((hist >= 0).all())
    checks['histogram_valid_and_expected_image_count'] = valid and result['images'] == images
    if valid:
        if compare_original_GT:
            original = required(OPT / 'evaluations/step2_iter8000/result.json', pending)
            if original is not None:
                reference = np.asarray(original['histogram'], dtype=np.int64)
                checks['original_GT_row_totals_match_optimized_KD'] = reference.shape == (21, 21) and np.array_equal(hist.sum(1), reference.sum(1))
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
    parts = [required(folder / f'rank{rank}.json', pending) for rank in range(4)]
    if all(value is not None for value in parts):
        seen = [name for record in parts for name in record['images']]
        validation = safe_path(Path(cfg['list_folder']) / 'incremental_split' / f'val_{cfg["task"]}_step_3.txt')
        if validation.is_file():
            expected = validation.read_text().splitlines()[:images] if cfg['val_limit'] else validation.read_text().splitlines()
            checks['four_shards_disjoint_complete_original_split'] = len(seen) == len(set(seen)) == len(expected) == images and set(seen) == set(expected)
        else:
            pending.append(str(validation))
        checks['four_shards_correct_rank_checkpoint_stage'] = all(record['rank'] == rank and record['shards'] == 4
            and record['step'] == 2 and record['iteration'] == iteration and record['checkpoint_sha256'] == result['checkpoint_sha256'] for rank, record in enumerate(parts))
        hist_parts = [np.asarray(record['histogram']) for record in parts]
        valid_parts = all(h.shape == (21, 21) and np.issubdtype(h.dtype, np.integer) and bool((h >= 0).all()) for h in hist_parts)
        checks['merged_histogram_equals_four_shards_exactly'] = valid and valid_parts and np.array_equal(np.sum(hist_parts, axis=0), hist)
    checks['no_shard_error_receipts'] = not list(folder.glob('error_rank*.json'))
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
    checks['candidate_source_outputs_and_cache_paths_stay_in_A'] = all(legal(path) for base in [SOURCE, RUN, SMOKE]
        for path in [base, *base.rglob('*')]) and all(legal(path) for path in [E / 'run_new_anchor_sep.py',
        E / 'launch_new_anchor_sep.py', ALLOWED / '.kd8tmp', ROOT / '.runtime/kd_pixel_v1/8card',
        ROOT / '.runtime/prototype_sep_v1/8card', ROOT / '.runtime/prototype_sep_v1/8card_eval', ROOT / 'pretrained'])
    checks['no_recorded_C_training_or_evaluation_failure'] = not any((RUN / name).exists() for name in ['failure.json', 'training_failed.json', 'evaluation_failed.json'])
    if manifest is not None:
        sources_and_shared_preflight(manifest, checks, details, pending)
        inheritance_and_resume(manifest, checks, details, pending)
        if cfg is not None and checks['C_isolated_new_anchor_source_freeze']:
            configuration(manifest, cfg, checks, details, pending)
            smoke(manifest, cfg, checks, details, pending)
            logs(STAGE, cfg, 8000, checks, details, pending, stage_done is not None, 'formal')
            details['evaluations'] = [evaluation(RUN, cfg, iteration, 1449, pending) for iteration in (4000, 6000, 8000)]
            if stage_done is not None:
                endpoint(cfg, checks, details, pending)
    if launch is not None:
        if 'returncode' in launch:
            checks['trainer_launch_exit0'] = launch['returncode'] == 0
        else:
            pending.append(str(STAGE / 'launch.json') + ': trainer still running')
        command = launch['command']
        checks['eight_rank_launch_original_teacher_and_resume'] = ('--nproc_per_node=8' in command
            and command[command.index('--step')+1] == '2' and command[command.index('--prev_checkpoint')+1] == str(TEACHER)
            and command[command.index('--resume_checkpoint')+1] == str(RESUME)
            and str(SOURCE / 'scripts/dist_train_voc_seg_neg.py') in command)
        checks['launch_original_teacher_SHA_verified'] = Path(launch['teacher']) == TEACHER and launch['teacher_sha256'] == TEACHER_SHA and digest(TEACHER) == TEACHER_SHA
    if stage_done is not None:
        checks['stage2_training_complete_exit0'] = stage_done.get('returncode') == 0
        hash_check(STAGE / 'checkpoints/model_final.pth', stage_done['checkpoint_sha256'], checks, 'final_checkpoint_matches_training_receipt', pending)
    if complete is not None:
        checks['one_new_stage_best_pipeline_stage1_inherited'] = complete.get('status') == 'complete' and complete.get('stages_trained') == 1 and complete.get('inherited_stage1') == str(OPT)
        checks['exactly_three_post_resume_evaluation_jobs'] = {path.stem for path in (RUN / 'eval_queue').glob('*.json')} == {'step2_iter4000', 'step2_iter6000', 'step2_iter8000'}
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
            checks['training_and_evaluation_share8card_host'] = len(set(hosts + [launch['host']])) == 1
    if status is not None:
        if status.get('status') == 'failed':
            checks['coordinator_not_failed'] = False
        elif status.get('status') == 'complete':
            checks['coordinator_complete'] = True
        else:
            pending.append(str(RUN / 'status.json') + ': coordinator not complete')
    failed = [key for key, value in checks.items() if not value]
    failed.extend(f'evaluation{row["details"]["iteration"]}: {key}' for row in details.get('evaluations', [])
        for key, value in row['checks'].items() if not value)
    done = not pending and not failed and all(value is not None for value in [manifest, cfg, launch, stage_done, complete, evaluated, status])
    return {'utc': now(), 'status': 'complete' if done else 'invalid' if failed else 'incomplete', 'arm': 'c_new_anchor',
        'checks': checks, 'failed_checks': failed, 'pending': sorted(set(pending)), 'details': details,
        'all_training_and_endpoint_requirements_complete': done, 'all_semantic_sep_requirements_complete': done,
        'all_best_pipeline_requirements_complete': done, 'all_new_anchor_requirements_complete': done,
        'device': 'CPU only; no CUDA, forward, optimizer, new tests, evaluation or remote operations',
        'budget': 'Original optimizedKD stage1 teacher and stage2/2000 student+optimizer inherited; fresh observer/selector;6000 new updates/48000 exposures;8x1 global8',
        'interpretation': 'Integrity and logged tangent protection only. Trusted current-new-source policy, teacher, student/optimizer and online history differ versus B; not teacher causal ablation, AdamW or final-IoU guarantee. GT diagnostic-prefix evidence does not establish population accuracy.'}


def save_output(path, result):
    path = safe_path(path)
    if path != RECEIPT:
        raise RuntimeError('Only independent formal/c_new_anchor/completion_audit.json is allowed; preserve A/B and earlier audits')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = safe_path(path.with_name(path.name + f'.{os.getpid()}.tmp'))
    with temporary.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--output', help='Only independent formal/c_new_anchor/completion_audit.json is accepted')
    parser.add_argument('--compact', action='store_true', help='Print compact status; saved receipt still contains the complete CPU audit')
    args = parser.parse_args()
    result = audit()
    if args.output:
        save_output(args.output, result)
    printed = result
    if args.compact:
        printed = {key: value for key, value in result.items() if key in
            ['utc', 'status', 'arm', 'failed_checks', 'pending', 'all_training_and_endpoint_requirements_complete',
             'all_semantic_sep_requirements_complete', 'all_best_pipeline_requirements_complete', 'all_new_anchor_requirements_complete']}
        printed['checks_passed'] = sum(bool(value) for value in result['checks'].values())
        printed['checks_total'] = len(result['checks'])
        printed['fresh_endpoint_online_state'] = result['details'].get('endpoint', {}).get('final_online_state')
        printed['formal_mechanism'] = result['details'].get('formal_mechanism')
    print(json.dumps(printed, allow_nan=False))
    if args.require_complete and not result['all_best_pipeline_requirements_complete']:
        raise SystemExit(1)
