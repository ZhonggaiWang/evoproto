"""CPU/stdlib analysis of C, the trusted-new-source prototype-SEP add-on.

Require completed stage2, all four evaluators, all three full1449-image jobs,
frozen B-derived sources, exact optimized-KD inheritance and independent audit.
Only the geometry selector may differ from B: schema4 never selects an old or
background source. Its partner may be any other foreground class. GT is never
an input to online selection or a calibrated learning strength.
Only the predefined8000 endpoint may select C; no midpoint or pair cherry-pick.
No model, GPU, dataset inference, optimizer or training operation is imported.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_prototype_sep as primary
import analyze_semantic_sep as semantic


ROOT = primary.ROOT
ITERATIONS = [4000, 6000, 8000]
TARGET = 69.14929219468897
AUDIT_FLAGS = ['all_training_and_endpoint_requirements_complete',
               'all_semantic_sep_requirements_complete',
               'all_best_pipeline_requirements_complete',
               'all_new_anchor_requirements_complete']
SELECTOR_FILE = 'model/geometry_pair_selector.py'
SELECTOR_TEST_SHA = 'dc6b25d14b59bc53999f47999efebe8921d1295549069363f502a6e2296a88a3'
SMOKE_OVERRIDES = {'max_iters': 2008, 'log_iters': 1, 'eval_iters': 2008,
    'val_limit': 16, 'train_limit': 64, 'pair_min_updates': 0,
    'pair_ramp_updates': 1, 'pair_min_row_images': 1, 'pair_min_pair_images': 1,
    'pair_min_rate': 0., 'pair_refresh_interval': 1}
EXPECTED_LOG_ITERATIONS = list(range(2050, 8001, 50))
GEOMETRY_CONFIG = {'w_geometry_sep': .1, 'proto_margin': 0.,
    'confusion_momentum': .98, 'pair_refresh_interval': 50,
    'pair_min_row_images': 8, 'pair_min_pair_images': 3, 'pair_min_rate': .01,
    'pair_min_updates': 100, 'pair_ramp_updates': 200, 'pair_max_stale_updates': 200}


def paths(root):
    unit = root/'runs/prototype_sep_v1'
    return {'unit': unit, 'run': unit/'formal/c_new_anchor',
            'A': unit/'formal/a_geometry', 'B': unit/'formal/b_semantic',
            'optimized': root/'runs/kd_parallel_v1/formal/b_relational',
            'experiment': root/'experiments/prototype_sep_v1',
            'smoke': unit/'smoke/c_new_anchor'}


def new_source_domain(selector):
    """Validate every snapshot, including empty/cold-start cached targets."""
    targets = selector.get('targets')
    rates = selector.get('selected_rates')
    if (selector.get('schema') != 4 or selector.get('source_anchor_policy') != 'current_new_only' or
            selector.get('old_classes') != 15 or
            selector.get('classes') != 21 or selector.get('stage') != 2 or
            not isinstance(targets, list) or len(targets) != 21 or
            not isinstance(rates, list) or len(rates) != 21):
        raise ValueError('New-source selector requires schema4 stage2 full21-class state')
    directions = []
    for i, j in enumerate(targets):
        if isinstance(j, bool) or not isinstance(j, int) or not -1 <= j < 21:
            raise ValueError('New-source selector target must be -1 or a valid class ID')
        rate = primary.finite(rates[i], 'New-source diagnostic ranking rate')
        if not 0 <= rate <= 1 + primary.EPS:
            raise ValueError('Selected full-row rate must be a finite fraction')
        if i <= 15 and (j != -1 or rate != 0):
            raise ValueError('Background/old sources must have no target or selected rate')
        if j >= 0:
            if j == 0 or i == j:
                raise ValueError('New sources cannot select background or themselves')
            directions.append((i, j))
        elif rate != 0:
            raise ValueError('Unselected sources must have zero diagnostic ranking rate')
    return directions


def new_anchor_mechanism(stage, geometry):
    """Describe observed new-source directions without a pixel-accuracy claim."""
    rows = primary.read_rows(stage/'geometry_metrics.jsonl')
    directions_seen, active_seen = {}, {}
    snapshots = []
    for row in rows:
        directions = new_source_domain(row['selector'])
        unique = {tuple(sorted(pair)) for pair in directions}
        if unique != {tuple(entry['class_ids']) for entry in row['pairs']}:
            raise ValueError('New-source selected directions do not match deduplicated geometry pairs')
        active_pairs = {tuple(entry['class_ids']) for entry in row['pairs']
                        if entry['active'] and row['active'] and row['ramp'] > 0}
        for pair in directions:
            directions_seen[pair] = directions_seen.get(pair, 0) + 1
            active_seen[pair] = active_seen.get(pair, 0) + int(tuple(sorted(pair)) in active_pairs)
        snapshots.append({'iteration': row['iteration'], 'new_source_direction_count': len(directions),
                          'deduplicated_pair_count': len(unique),
                          'effective_active_direction_count': sum(tuple(sorted(pair)) in active_pairs for pair in directions),
                          'new_to_old_directions': sum(j <= 15 for _, j in directions),
                          'new_to_new_directions': sum(j > 15 for _, j in directions),
                          'directions': [list(pair) for pair in directions]})
    return {'selector_schema': 4, 'source_class_ids': list(range(16, 21)),
        'partner_domain': 'Any foreground class1..20 except the source itself; old-old pairs impossible.',
        'all_logged_sources_are_current_new': True,
        'background_and_old_source_targets_always_missing': True,
        'all_logged_selected_directed_relations': [list(pair) for pair in sorted(directions_seen)],
        'final_logged_selected_directed_relations': [list(pair) for pair in new_source_domain(rows[-1]['selector'])],
        'direction_frequencies': [{'source_id': i, 'target_id': j,
            'source_name': primary.CLASSES[i], 'target_name': primary.CLASSES[j],
            'partner_group': 'old' if j <= 15 else 'new',
            'logged_snapshots_selected': directions_seen[i, j],
            'logged_snapshots_with_effectively_active_pair': active_seen.get((i, j), 0)}
            for i, j in sorted(directions_seen)],
        'new_source_direction_count_distribution': primary.distribution([row['new_source_direction_count'] for row in snapshots]),
        'sampled_snapshots_with_new_source_directions': sum(row['new_source_direction_count'] > 0 for row in snapshots),
        'sampled_snapshots_with_effectively_active_new_source_pairs': sum(row['effective_active_direction_count'] > 0 for row in snapshots),
        'geometry_unique_pairs_observed': geometry['unique_pairs_observed_in_logs'],
        'snapshots': snapshots,
        'selection_definition': 'Past full broad pseudo-anchor row EMA ranks supported foreground competitors. Trusted CAM pair exposure and freshness screen candidates; only current-new source rows can propose a direction. The full21-class broad row denominator is preserved.',
        'rates_usage': 'Ranking and support gates only; neither GT diagnostic accuracy nor online rates become loss weights.',
        'loss_definition': 'Directions choose unordered prototype pairs once; inactive hinge pairs remain in the mean denominator. Current student old endpoints are detached, new endpoints differentiable; SEM tangent protection is unchanged.',
        'scope': '120 sampled rank0 synchronized training moments; selection counts are not independent images, pixel coverage, population anchor accuracy or measured raw parameter-gradient norms.'}


def actual_new_smoke(p, study, actual):
    """The previous best-teacher smoke cannot satisfy this new-source gate."""
    run, stage = p['smoke'], p['smoke']/'10-5/step2'
    smoke_study = primary.read_json(run/'study.json')
    smoke_cfg = primary.read_json(stage/'config.json')
    expected_cfg = {**study['common_config'], **SMOKE_OVERRIDES}
    geometry = primary.read_rows(stage/'geometry_metrics.jsonl')
    guard = primary.read_rows(stage/'semantic_guard_metrics.jsonl')
    expected_iterations = list(range(2001, 2009))
    if ([row.get('iteration') for row in geometry] != expected_iterations or
            [row.get('iteration') for row in guard] != expected_iterations):
        raise ValueError('New-source smoke must log its own8 restored updates2001..2008')
    for row in geometry:
        new_source_domain(row['selector'])
        if (row['selector'].get('min_updates') != 0 or row['selector'].get('ramp_updates') != 1
                or row.get('ramp') != min(1., float(row['iteration']-2001))):
            raise ValueError('Only the own smoke may use its accelerated cold-start configuration')
    checks = {
        'own_c_new_anchor_smoke_complete': primary.read_json(run/'status.json').get('status') == 'complete',
        'own_smoke_correct_arm_and_sources': smoke_study.get('arm') == 'c_new_anchor'
            and smoke_study.get('smoke') is True and smoke_study.get('source_sha256') == actual
            and smoke_study.get('source') == study['source'],
        'own_smoke_exact_teacher_resume_scripts': all(smoke_study.get(key) == study.get(key) for key in
            ['teacher', 'teacher_sha256', 'resume_checkpoint', 'resume_checkpoint_sha256', 'runner_sha256', 'launcher_sha256']),
        'own_smoke_model_optimizer_only_fresh_observer': smoke_study.get('restored_state') == ['student', 'optimizer']
            and smoke_study.get('fresh_state') == ['online_confusion', 'geometry_selector']
            and smoke_study.get('expected_final_observer_updates') == 8
            and smoke_study.get('expected_final_observer_seen_images') == 64,
        'own_smoke_train_and_four_evaluators_exit0': primary.read_json(stage/'launch.json').get('returncode') == 0
            and primary.read_json(stage/'training_complete.json').get('returncode') == 0
            and primary.read_json(run/'evaluation_workers_complete.json').get('returncodes') == [0]*4,
        'own_smoke_only_declared_overrides': smoke_study.get('smoke_overrides') == SMOKE_OVERRIDES
            and smoke_study.get('common_config') == expected_cfg
            and all(smoke_cfg.get(key) == value for key, value in expected_cfg.items()),
        'own_smoke_first_update_has_no_past_online_selection': geometry[0].get('pair_count') == 0
            and geometry[0]['selector']['targets'] == [-1]*21 and geometry[0]['ramp'] == 0
            and guard[0].get('semantic_gradient_used') is False,
        'own_smoke_reaches_actual_SEM_protection_gradient': any(row.get('semantic_gradient_used') is True for row in guard),
        'own_smoke_reported_by_formal_gate': study.get('smoke_restoration_receipt') == {
            'run': str(run), 'eight_updates': True, 'active_protection_observed': True}}
    protection = semantic.protection_statistics(stage)
    checks['own_smoke_nested_and_independent_protection_logs_exact'] = (
        protection['logged_snapshots'] == 8 and protection['independent_and_nested_logs_exactly_match'])
    return {'complete': all(checks.values()), 'checks': checks, 'run': str(run),
            'iterations': expected_iterations, 'scope': 'Actual8-rank candidate restoration/integration smoke with accelerated support gates; not a performance result or formal selection policy.'}


def endpoint_GT_diagnostic(p, study, geometry):
    """Read the optional own-C GT audit after the complete endpoint; no inference."""
    path = p['unit']/'new_anchor_endpoint_gt_diagnostic.json'
    if not path.is_file():
        return {'status': 'pending_optional_mechanism_diagnostic', 'source': str(path),
            'effect_criterion': 'Complete1449-image endpoint only; an unrun128-image diagnostic supplies no anchor-accuracy claim.'}
    record = primary.read_json(path)
    final = p['run']/'10-5/step2/checkpoints/model_final.pth'
    cfg = primary.read_json(p['run']/'10-5/step2/config.json')
    names = (Path(cfg['list_folder'])/'incremental_split'/f"val_{cfg['task']}_step_3.txt").read_text().splitlines()[:128]
    expected_directions = {f'{i}->{j}' for i, j in new_source_domain(geometry['final_logged_selector'])}
    if (record.get('passed') is not True or record.get('mode') != 'gt'
            or record.get('candidate') != 'c_new_anchor' or record.get('stage') != 2
            or record.get('iteration') != 8000 or record.get('final_checkpoint') is not True
            or record.get('student_checkpoint') != str(final) or record.get('student_sha256') != primary.digest(final)
            or record.get('teacher_checkpoint') != study['teacher'] or record.get('teacher_sha256') != study['teacher_sha256']
            or record.get('observer_and_selector_checkpoint') != str(final)
            or record.get('saved_observer_updates') != 6000 or record.get('saved_observer_seen_images') != 48000
            or record.get('source_sha256') != study['source_sha256']
            or record.get('source_anchor_policy') != 'current_new_only'
            or record.get('saved_selector') != geometry['final_logged_selector']
            or record.get('sample_indices') != list(range(128)) or record.get('sample_names') != names
            or record.get('validation_prefix128_sha256') != hashlib.sha256('\n'.join(names).encode()).hexdigest()
            or record.get('completion_audit_sha256') != primary.digest(p['run']/'completion_audit.json')
            or record.get('diagnostic_script_sha256') != primary.digest(p['experiment']/'diagnose_new_anchor_gt.py')
            or record.get('reused_GT_protocol_sha256') != '743677a64fa835a91ff4652beab3c4485a5d18004457c9e7788289dcc68ea2b0'
            or not record.get('state_checks') or any(value is not True for value in record['state_checks'].values())
            or set(record.get('selected_directions', {})) != {'trusted', 'broad'}
            or any(set(directions) != expected_directions for directions in record['selected_directions'].values())):
        raise ValueError('Own-C128 GT diagnostic identity/prefix/source/state contract disagrees with completed endpoint')
    return {'status': 'complete_readonly_subset_diagnostic', 'source': str(path), 'sha256': primary.digest(path),
        'student_sha256': record['student_sha256'], 'teacher_sha256': record['teacher_sha256'],
        'validation_prefix128_sha256': record['validation_prefix128_sha256'],
        'anchor_group_summary': record['anchor_group_summary'],
        'selected_directions': record['selected_directions'],
        'anchor_vs_GT_counts': record['anchor_vs_gt_counts'],
        'anchor_vs_GT_evidence_weight': record['anchor_vs_gt_evidence_weight'],
        'GT_role': record['GT_role'], 'scope': record['scope'], 'definitions': record['definitions'],
        'limitations': record['limitations'],
        'effect_criterion': 'These descriptive128-image subset diagnostics never replace the full1449-image all/old/new endpoint selection criterion.'}


def coldstart_logs(stage):
    """Validate the real global iteration -> prior-online-update ramp mapping."""
    geometry = primary.read_rows(stage/'geometry_metrics.jsonl')
    guard = primary.read_rows(stage/'semantic_guard_metrics.jsonl')
    if ([row.get('iteration') for row in geometry] != EXPECTED_LOG_ITERATIONS or
            [row.get('iteration') for row in guard] != EXPECTED_LOG_ITERATIONS):
        raise ValueError('C must contain exactly120 ordered2050..8000 snapshots in each log')
    phases = {'zero_ramp': [], 'ramping': [], 'full_ramp': []}
    for row in geometry:
        iteration = primary.integer(row['iteration'], 'C logged iteration')
        prior = iteration-2000-1
        expected = min(1., max(0., (prior-100)/200))
        ramp = primary.finite(row['ramp'], 'C selector ramp')
        selector = row['selector']
        info = row.get('semantic_protection', {})
        if (row.get('step') != 2 or row.get('active') is not True or
                abs(ramp-expected) > primary.EPS or
                selector.get('schema') != 4 or selector.get('old_classes') != 15 or
                selector.get('classes') != 21 or selector.get('stage') != 2 or
                selector.get('min_updates') != 100 or selector.get('ramp_updates') != 200):
            raise ValueError('C fresh-online ramp or selector stage/configuration disagrees')
        directions = new_source_domain(selector)
        if info.get('enabled') is not True:
            raise ValueError('All C resumed logged iterations are past the original loss warmup')
        if expected == 0 and info.get('semantic_gradient_used') is not False:
            raise ValueError('Zero-ramp geometry must skip semantic gradient extraction')
        phase = 'zero_ramp' if expected == 0 else 'full_ramp' if expected == 1 else 'ramping'
        phases[phase].append({'iteration': iteration, 'prior_online_updates': prior,
            'selector_ramp': ramp, 'semantic_gradient_used': info.get('semantic_gradient_used'),
            'pair_count': row['pair_count'], 'active_pair_count': row['active_pair_count'],
            'new_source_directed_relations': [list(pair) for pair in directions]})
    # Reuse B's exact per-row mathematical/log consistency audit. Its zero-SEP
    # branch permits the deliberately cold-started early C snapshots.
    protection = semantic.protection_statistics(stage)
    if protection['logged_snapshots'] != 120 or not protection['independent_and_nested_logs_exactly_match']:
        raise ValueError('C complete independent/nested protection logs are required')
    online = primary.read_json(stage/'online_confusion.json')
    if (online.get('training_iteration') != 8000 or online.get('stage') != 2 or
            online.get('classes') != 21 or online.get('updates') != 6000 or
            online.get('seen_images') != 48000):
        raise ValueError('C final observer must be fresh6000 updates/48000 image exposures')
    return {'passed': True, 'logged_snapshots': 120,
        'ramp_rule': 'prior_online_updates=global_iteration-2000-1; ramp=clip((prior-100)/200,0,1)',
        'first_positive_ramp_iteration': 2102, 'first_full_ramp_iteration': 2301,
        'phases': {name: {'logged_snapshots': len(rows),
                         'first_iteration': rows[0]['iteration'] if rows else None,
                         'last_iteration': rows[-1]['iteration'] if rows else None,
                         'semantic_gradient_skipped_snapshots': sum(row['semantic_gradient_used'] is False for row in rows),
                         'snapshots': rows} for name, rows in phases.items()},
        'final_observer_updates': 6000, 'final_seen_image_exposures': 48000,
        'zero_branch_meaning': 'Coldstart ramp0 or zero new-row SEP tangent gradient legitimately avoids SEM extraction. This is not evidence of a missing semantic ancestor or a failed protection mechanism.',
        'exposure_caveat': 'Repeated/augmented training visits are exposures, not independent images.'}


def completion(root):
    p = paths(root)
    run, opt, experiment = p['run'], p['optimized'], p['experiment']
    stage = run/'10-5/step2'
    teacher = opt/'10-5/step1/checkpoints/model_final.pth'
    resume = opt/'10-5/step2/checkpoints/model_iter_2000.pth'
    required = [run/name for name in ['study.json', 'status.json', 'training_complete.json',
                 'evaluation_workers_complete.json', 'completion_audit.json']]
    required += [stage/name for name in ['config.json', 'launch.json', 'training_complete.json',
                 'checkpoints/model_final.pth', 'geometry_metrics.jsonl',
                 'semantic_guard_metrics.jsonl', 'online_confusion.json']]
    required += [opt/'training_complete.json', opt/'evaluation_workers_complete.json',
                 opt/'10-5/step1/launch.json', opt/'10-5/step2/launch.json',
                 opt/'10-5/step1/training_complete.json',
                 opt/'eval_queue/step2_iter2000.json', opt/'evaluations/step2_iter2000/result.json',
                 opt/'evaluations/step1_iter8000/result.json', opt/'evaluations/step2_iter8000/result.json',
                 teacher, resume, p['B']/'study.json', experiment/'semantic_preflight.json',
                 experiment/'new_anchor_origin.json', experiment/'new_anchor_selector_tests.json',
                 experiment/'test_new_anchor_selector.py', experiment/'new_anchor_geometry_selector.py',
                 experiment/'run_new_anchor_sep.py', experiment/'launch_new_anchor_sep.py']
    smoke_stage = p['smoke']/'10-5/step2'
    required += [p['smoke']/name for name in ['study.json', 'status.json', 'evaluation_workers_complete.json']]
    required += [smoke_stage/name for name in ['config.json', 'launch.json', 'training_complete.json',
                                              'geometry_metrics.jsonl', 'semantic_guard_metrics.jsonl']]
    required += [run/f'evaluator_rank{rank}.process.json' for rank in range(4)]
    for iteration in ITERATIONS:
        required += [run/f'eval_queue/step2_iter{iteration}.json',
                     run/f'evaluations/step2_iter{iteration}/result.json',
                     stage/f'checkpoints/model_iter_{iteration}.pth']
        required += [run/f'evaluations/step2_iter{iteration}/rank{rank}.json' for rank in range(4)]
    missing = [str(path) for path in required if not path.is_file()]
    failures = [str(run/name) for name in ['failure.json', 'training_failed.json', 'evaluation_failed.json']
                if (run/name).exists()]
    if missing or failures:
        return {'complete': False, 'missing': missing, 'failure_receipts': failures,
                'reason': 'No final recommendation or report is written while artifacts are missing or failed.'}
    study, cfg = primary.read_json(run/'study.json'), primary.read_json(stage/'config.json')
    training = primary.read_json(run/'training_complete.json')
    audit = primary.read_json(run/'completion_audit.json')
    opt_stage1 = primary.endpoint(opt/'evaluations/step1_iter8000/result.json', 1)
    optimized = primary.endpoint(opt/'evaluations/step2_iter8000/result.json', 2)
    bstudy = primary.read_json(p['B']/'study.json')
    preflight = primary.read_json(experiment/'semantic_preflight.json')
    origin = primary.read_json(experiment/'new_anchor_origin.json')
    selector_tests = primary.read_json(experiment/'new_anchor_selector_tests.json')
    inherited = study.get('restoration_verified', {})
    teacher_sha, resume_sha = primary.digest(teacher), primary.digest(resume)
    original_resume_job = primary.read_json(opt/'eval_queue/step2_iter2000.json')
    original_resume_result = primary.read_json(opt/'evaluations/step2_iter2000/result.json')
    checks = {
        'formal_C_not_smoke': study.get('arm') == 'c_new_anchor' and study.get('smoke') is False,
        'C_status_complete': primary.read_json(run/'status.json').get('status') == 'complete',
        'one_stage_trained_original_optimized_stage1_inherited': training.get('status') == 'complete'
            and training.get('stages_trained') == 1 and training.get('inherited_stage1') == str(opt),
        'C_stage2_train_exit0': primary.read_json(stage/'launch.json').get('returncode') == 0
            and primary.read_json(stage/'training_complete.json').get('returncode') == 0,
        'four_evaluators_exit0': primary.read_json(run/'evaluation_workers_complete.json').get('returncodes') == [0]*4,
        'independent_C_audit_complete': audit.get('arm') == 'c_new_anchor' and audit.get('status') == 'complete'
            and all(audit.get(key) is True for key in AUDIT_FLAGS),
        'original_optimized_KD_complete': primary.read_json(opt/'training_complete.json').get('status') == 'training_complete'
            and primary.read_json(opt/'training_complete.json').get('stages') == 2
            and primary.read_json(opt/'evaluation_workers_complete.json').get('returncodes') == [0, 0]
            and all(primary.read_json(opt/f'10-5/step{step}/launch.json').get('returncode') == 0 for step in [1, 2]),
        'original_optimized_endpoint_matches_required_reference': abs(optimized['metrics']['all_miou']-TARGET) < 1e-8,
        'own_teacher_optimized_stage1_final': study.get('teacher') == str(teacher)
            and study.get('inherited_stage1_run') == str(opt)
            and Path(cfg.get('prev_checkpoint', '')) == teacher,
        'teacher_identity_matches_original_receipt': teacher_sha == study.get('teacher_sha256')
            == inherited.get('teacher_sha256')
            == primary.read_json(opt/'10-5/step1/training_complete.json').get('checkpoint_sha256'),
        'resume_original_optimized_step2_2000': study.get('resume_checkpoint') == str(resume)
            and study.get('shared_stage2_warmup') == str(resume)
            and Path(cfg.get('resume_checkpoint', '')) == resume and study.get('resume_iteration') == 2000,
        'resume_identity_matches_original_job_result_and_restoration': resume_sha == study.get('resume_checkpoint_sha256')
            == study.get('shared_stage2_warmup_sha256') == inherited.get('resume_checkpoint_sha256')
            == original_resume_job.get('checkpoint_sha256') == original_resume_result.get('checkpoint_sha256')
            and original_resume_job.get('checkpoint') == str(resume)
            and original_resume_job.get('iteration') == original_resume_result.get('iteration') == 2000
            and original_resume_result.get('images') == 1449,
        'restored_model_optimizer_only_fresh_online_selector': set(study.get('restored_state', [])) == {'student', 'optimizer'}
            and set(study.get('fresh_state', [])) == {'online_confusion', 'geometry_selector'}
            and inherited.get('observer_state_present') is False and inherited.get('selector_state_present') is False,
        'original_iteration2000_full_optimizer_state_verified': inherited.get('iteration') == 2000
            and inherited.get('optimizer_parameter_groups') == 4 and inherited.get('optimizer_state_entries', 0) > 0,
        'fresh6000_updates48000_exposures': study.get('remaining_training_updates') == 6000
            and study.get('expected_final_observer_updates') == 6000
            and study.get('expected_final_observer_seen_images') == 48000,
        'formal100_coldstart200_ramp': study.get('formal_online_coldstart_updates') == 100
            and study.get('formal_online_ramp_updates') == 200,
        'new_source_location': study.get('source') == str(experiment/'c_new_anchor/src'),
        'study_only_new_source_policy_and_frozen_parent': study.get('source_reused_without_modification') is False
            and study.get('parent_source') == str(experiment/'b_semantic/src')
            and study.get('parent_source_sha256') == bstudy.get('source_sha256')
            and study.get('changed_files_from_B') == [SELECTOR_FILE] and study.get('selector_schema') == 4
            and study.get('source_anchor_policy') == 'current_new_only'
            and study.get('source_domain') == 'new_foreground_rows_only'
            and study.get('KD_byte_identical_to_B') is True
            and study.get('semantic_guard_byte_identical_to_B') is True,
        'task_seed_global8_topology': cfg.get('task') == '10-5' and cfg.get('step') == 2 and cfg.get('seed') == 0
            and cfg.get('spg') == 1 and study.get('global_batch') == 8 and study.get('batch_per_gpu') == 1
            and study.get('train_gpus') == list(range(8)),
        'original_budget_and_KD_preserved': cfg.get('max_iters') == 8000
            and cfg.get('warmup_iters') == cfg.get('loss_warmup_iters') == 2000
            and cfg.get('w_pixel_kd') == cfg.get('w_proto_seg') == .1 and cfg.get('kd_temperature') == 2
            and cfg.get('w_proto_kd') == cfg.get('w_proto_sep') == 0,
        'formal_geometry_policy_preserved': all(cfg.get(key) == value and study['common_config'].get(key) == value
                                                for key, value in GEOMETRY_CONFIG.items()),
        'no_new_stage1_step0_baseline_training': study.get('new_stage1_training') is False
            and study.get('step0_retrained') is False and study.get('baseline_retrained') is False
            and not (run/'10-5/step1/launch.json').exists(),
        'no_formal_smoke_overrides': study.get('smoke_overrides') == {},
    }
    for rank in range(4):
        checks[f'evaluator{rank}_record_exit0'] = primary.read_json(run/f'evaluator_rank{rank}.process.json').get('returncode') == 0
    source = experiment/'c_new_anchor/src'
    bsource = experiment/'b_semantic/src'
    actual = {str(path.relative_to(source)): primary.digest(path) for path in source.rglob('*.py')
              if '__pycache__' not in path.parts}
    bactual = {str(path.relative_to(bsource)): primary.digest(path) for path in bsource.rglob('*.py')
               if '__pycache__' not in path.parts}
    checks['C_source_equals_own_full_freeze'] = bool(actual) and actual == study.get('source_sha256')
    checks['parent_B_source_equals_its_freeze_and_semantic_preflight'] = bool(bactual) and bactual == bstudy.get('source_sha256') == preflight.get('source_sha256')
    checks['only_geometry_selector_changed_from_frozen_B'] = (actual.keys() == bactual.keys()
        and SELECTOR_FILE in actual and actual[SELECTOR_FILE] != bactual[SELECTOR_FILE]
        and all(actual[name] == bactual[name] for name in actual if name != SELECTOR_FILE))
    checks['new_source_origin_exact_parent_candidate_maps'] = (
        origin.get('parent_source') == str(bsource) and origin.get('candidate_source') == str(source)
        and origin.get('parent_source_sha256') == bactual
        and origin.get('source_sha256') == origin.get('candidate_source_sha256') == actual
        and origin.get('changed_files') == [SELECTOR_FILE]
        and origin.get('selector_schema') == 4 and origin.get('source_anchor_policy') == 'current_new_only'
        and origin.get('source_domain') == 'new_foreground_rows_only'
        and all(origin.get(key) is True for key in
            ['parent_source_unchanged', 'KD_byte_identical', 'semantic_guard_byte_identical']))
    checks['new_selector_CPU_tests_passed_for_exact_source'] = (
        selector_tests.get('passed') is True and selector_tests.get('returncode') == 0
        and selector_tests.get('count') == 10 and selector_tests.get('source_sha256') == actual
        and selector_tests.get('module_sha256') == actual.get(SELECTOR_FILE)
        and selector_tests.get('test_source_sha256') == SELECTOR_TEST_SHA
        and primary.digest(experiment/'test_new_anchor_selector.py') == SELECTOR_TEST_SHA
        and primary.digest(experiment/'new_anchor_geometry_selector.py') == actual.get(SELECTOR_FILE))
    for key, name in [('new_anchor_origin', 'new_anchor_origin.json'),
                      ('new_anchor_selector_tests', 'new_anchor_selector_tests.json')]:
        checks[f'{key}_receipt_identity'] = (study.get(key) == str(experiment/name)
            and study.get(key+'_sha256') == primary.digest(experiment/name))
    checks['tested_semantic_source_preflight_passed'] = preflight.get('passed') is True
    for key, name in [('runner_sha256', 'run_new_anchor_sep.py'), ('launcher_sha256', 'launch_new_anchor_sep.py')]:
        checks[f'C_{key}_frozen'] = primary.digest(experiment/name) == study.get(key)
    checks['preflight_receipt_identity'] = primary.digest(experiment/'semantic_preflight.json') == study.get('semantic_preflight_sha256')
    smoke = actual_new_smoke(p, study, actual)
    checks.update(smoke['checks'])
    final_receipt = primary.read_json(stage/'training_complete.json')
    checks['C_final_file_matches_training_receipt'] = primary.digest(stage/'checkpoints/model_final.pth') == final_receipt.get('checkpoint_sha256')
    queue = sorted(path.stem for path in (run/'eval_queue').glob('*.json'))
    checks['exact_three_remaining_evaluation_jobs'] = queue == [f'step2_iter{i}' for i in ITERATIONS]
    checks['no_evaluation_error_receipts'] = not list((run/'evaluations').glob('*/error_rank*.json'))
    split = Path(cfg['list_folder'])/'incremental_split'/f"val_{cfg['task']}_step_3.txt"
    expected_names = split.read_text().splitlines() if split.is_file() else []
    evaluations = {}
    for iteration in ITERATIONS:
        folder = run/f'evaluations/step2_iter{iteration}'
        result = semantic.checked_evaluation(folder/'result.json', iteration)
        job = primary.read_json(run/f'eval_queue/step2_iter{iteration}.json')
        parts = [primary.read_json(folder/f'rank{rank}.json') for rank in range(4)]
        names = [name for part in parts for name in part['images']]
        checkpoint = stage/f'checkpoints/model_iter_{iteration}.pth'
        checks[f'iter{iteration}_full1449_exact_unique_names'] = len(names) == len(set(names)) == len(expected_names) == 1449 and set(names) == set(expected_names)
        checks[f'iter{iteration}_four_shard_identifiers'] = all(part.get('rank') == rank and part.get('shards') == 4
            and part.get('step') == 2 and part.get('iteration') == iteration for rank, part in enumerate(parts))
        checks[f'iter{iteration}_checkpoint_identity'] = (job.get('step') == 2 and job.get('iteration') == iteration
            and job.get('checkpoint') == str(checkpoint)
            and primary.digest(checkpoint) == job.get('checkpoint_sha256') == result['checkpoint_sha256']
            and all(part.get('checkpoint_sha256') == result['checkpoint_sha256'] for part in parts))
        for part in parts:
            if not isinstance(part['histogram'], list) or len(part['histogram']) != 21 or any(len(row) != 21 for row in part['histogram']):
                raise ValueError('Malformed C shard histogram')
            for row in part['histogram']:
                for value in row:
                    primary.integer(value, 'C shard count')
        merged = [[sum(part['histogram'][i][j] for part in parts) for j in range(21)] for i in range(21)]
        checks[f'iter{iteration}_histogram_additive_consistency'] = merged == result['histogram']
        checks[f'iter{iteration}_GT_rows_equal_original_optimized'] = result['gt_class_pixels'] == optimized['gt_class_pixels']
        evaluations[str(iteration)] = {key: result[key] for key in ['source', 'images', 'metrics', 'checkpoint_sha256']}
    coldstart = coldstart_logs(stage)
    checks['120_protection_logs_fresh_state_ramp_exact'] = coldstart['passed']
    a_complete, b_complete = primary.completion(p['A']), semantic.completion(root)
    checks['A_complete_for_comparison'] = a_complete['complete']
    checks['B_complete_for_comparison'] = b_complete['complete']
    return {'complete': all(checks.values()), 'checks': checks, 'missing': missing,
        'failure_receipts': failures, 'evaluations': evaluations,
        'independent_C_audit': {'source': str(run/'completion_audit.json'),
                              'sha256': primary.digest(run/'completion_audit.json'),
                              'status': audit.get('status'),
                              **{key: audit.get(key) for key in AUDIT_FLAGS}},
        'inherited_stage1': str(opt), 'inherited_stage1_metrics': opt_stage1['metrics'],
        'validation_list': str(split), 'coldstart': coldstart,
        'own_new_anchor_smoke': smoke,
        'new_selector_CPU_receipt': {'path': str(experiment/'new_anchor_selector_tests.json'),
            'sha256': primary.digest(experiment/'new_anchor_selector_tests.json'), 'receipt': selector_tests},
        'new_source_origin': {'path': str(experiment/'new_anchor_origin.json'),
            'sha256': primary.digest(experiment/'new_anchor_origin.json'), 'receipt': origin}}


def analyze(root):
    p = paths(root)
    done = completion(root)
    if not done['complete']:
        raise RuntimeError('C final analysis withheld until successful complete endpoint/audit: '+json.dumps(done))
    run, stage = p['run'], p['run']/'10-5/step2'
    study = primary.read_json(run/'study.json')
    candidate = primary.endpoint(run/'evaluations/step2_iter8000/result.json', 2)
    references = {name: primary.endpoint(path/'evaluations/step2_iter8000/result.json', 2)
                  for name, path in [('optimized_KD', p['optimized']), ('A_geometry', p['A']), ('B_semantic', p['B'])]}
    inherited = primary.endpoint(p['optimized']/'evaluations/step1_iter8000/result.json', 1)
    geometry = primary.mechanism(stage, 2, .1)
    new_anchor = new_anchor_mechanism(stage, geometry)
    GT_diagnostic = endpoint_GT_diagnostic(p, study, geometry)
    protection = semantic.protection_statistics(stage)
    comparisons = {name: primary.comparison(candidate, reference, geometry['all_logged_selected_pairs'])
                   for name, reference in references.items()}
    final_pairs = {name: primary.comparison(candidate, reference, geometry['final_snapshot_pairs'])['selected_pair_bidirectional_deltas']
                   for name, reference in references.items()}
    selected_directions = {name: [{'selection_source_id': i, 'selection_partner_id': j,
        'selection_source_name': primary.CLASSES[i], 'selection_partner_name': primary.CLASSES[j],
        'selected_direction': primary.direction_delta(candidate, reference, i, j),
        'reverse_direction': primary.direction_delta(candidate, reference, j, i),
        'direction_caveat': 'These are GT-source evaluation errors; the direction was selected using pseudo/CAM anchors, not GT.'}
        for i, j in new_anchor['all_logged_selected_directed_relations']]
        for name, reference in references.items()}
    focus_pairs = {name: [{'class_ids': [i, j], 'class_names': [primary.CLASSES[i], primary.CLASSES[j]],
                          'forward': primary.direction_delta(candidate, reference, i, j),
                          'reverse': primary.direction_delta(candidate, reference, j, i),
                          'selection_note': 'Fixed historical focus; this does not imply an old-old pair participates in the geometry loss.'}
                         for i, j in [(8, 12), (9, 18), (10, 13)]] for name, reference in references.items()}
    delta = comparisons['optimized_KD']['metric_delta_pp']['all_miou']
    wins = delta > 0.
    selected_run = run if wins else p['optimized']
    grouped = comparisons['optimized_KD']['metric_delta_pp']
    losses = [name for name in ['previous_foreground_miou', 'current_foreground_miou'] if grouped[name] < 0]
    recommendation = {'method': 'best_pipeline_new_anchor_semantic_protected_prototype_SEP_addon' if wins else 'existing_optimized_KD',
        'selected_checkpoint': str(selected_run/'10-5/step2/checkpoints/model_final.pth'),
        'candidate_exceeds_required_endpoint': wins, 'candidate_all_miou': candidate['metrics']['all_miou'],
        'optimized_KD_all_miou': references['optimized_KD']['metrics']['all_miou'], 'delta_all_pp': delta,
        'criterion': 'Select C iff its complete stage2/8000 all-class mIoU strictly exceeds the existing optimizedKD69.14929219468897; intermediate or isolated-pair gains cannot select it.',
        'status': 'observed_development_endpoint_gain' if wins else 'candidate_did_not_improve_required_endpoint',
        'foreground_group_tradeoff': {'delta_old15_pp': grouped['previous_foreground_miou'],
                                    'delta_new5_pp': grouped['current_foreground_miou'],
                                    'groups_with_lower_mean_IoU': losses},
        'evidence_limit': 'Fixedseed0 adaptive development validation; no significance, seed-stability or fresh-test claim.'}
    return {'utc': datetime.now(timezone.utc).isoformat(), 'status': 'complete_endpoint_analysis',
        'run': str(run), 'study': study, 'completion': done, 'candidate_endpoint': candidate,
        'references': references, 'comparisons': comparisons,
        'final_logged_selected_pair_bidirectional_deltas': final_pairs,
        'fixed_focus_pair_bidirectional_deltas': focus_pairs,
        'selected_new_source_direction_and_reverse_GT_deltas': selected_directions,
        'new_anchor_selection_mechanism': new_anchor,
        'own_endpoint_anchor_GT_diagnostic': GT_diagnostic,
        'geometry_mechanism': geometry, 'semantic_protection_mechanism': protection,
        'fresh_online_coldstart': done['coldstart'],
        'inherited_stage1': {'run': str(p['optimized']), 'endpoint': inherited, 'new_training': False,
            'teacher': study['teacher'], 'teacher_sha256': study['teacher_sha256'],
            'note': 'Original completed optimizedKD stage1 teacher (approximately74.207 all-class mIoU); C does not inherit the weaker A/B teacher or retrain stage1.'},
        'stage2_resume': {'checkpoint': study['resume_checkpoint'], 'sha256': study['resume_checkpoint_sha256'],
            'iteration': 2000, 'trained_remaining_iterations': 6000,
            'restored_state': study['restored_state'], 'fresh_state': study['fresh_state'],
            'restoration_verified': study['restoration_verified'],
            'observer_note': 'Original optimizedKD warmup has model/optimizer only. Fresh observer and selector start at global iteration2000, first positive ramp2102, full ramp2301; no A/B online history is borrowed.'},
        'recommendation': recommendation,
        'scope': {'task': 'VOC10-5', 'seed': 0, 'stage': 2, 'iteration': 8000, 'images': 1449,
            'all_miou_includes_background': True, 'units': 'IoU/error percentages; delta percentage points.',
            'GT_direction_definition': 'GT i->prediction j divided by the entire GT source row; each reverse direction has its own denominator.',
            'GT_role': 'Existing evaluation histograms only, never supervision for selector/projection/learning weights.',
            'training_selection': 'Only current-new source rows of past broad PAR-pseudo->main-prediction EMA propose foreground competitors, with trusted CAM exposure support. Old-new/new-new prototype pairs, rate ranking only.',
            'intermediate_evaluations': '4000/6000 complete1449 receipts and histograms are audited, but winner selection uses8000 only.',
            'comparison_type': 'C is an add-on to the original strongest completed pipeline, not a teacher causal ablation.'},
        'limitations': ['One fixedseed0 task under adaptive development; prior128-image GT diagnostics and complete endpoints use the same development validation, not a fresh independent test.',
            'Relative to B, teacher, student/model-optimizer trajectory, fresh online history and the source-domain restriction change together. This best-pipeline add-on cannot isolate teacher quality or the selector restriction as a single causal factor.',
            'C restores original optimizedKD stage2/2000 model and optimizer, then restarts sampling/augmentation with8x1 versus archived reference4x2. Global batch8 is preserved, but this is not a bitwise continuation or strict single-factor attribution.',
            'The original optimizedKD stage1 teacher and shared warmup are inherited; no stage1, step0 or baseline retraining is attributed to C.',
            'SEM protection changes SEP gradients while preserving its scalar forward value; Euclidean tangent alignment does not ensure AdamW update protection, future teacher/encoder behavior or final IoU.',
            'Protection logs are120 sampled rank0 training moments, not120 independent observations, pixel coverage or measured raw-parameter/optimizer-gradient norms.',
            'The128-image validation-prefix diagnostic motivating new-only sources concerned trusted selected-disagreement pixels: old-source GT accuracy1.601% over63266 valid pixels and current-new-source91.832% over36876, with model matching the GT competitor60.845% in the old-source subset. These figures do not estimate overall online anchor accuracy, all-image population precision, probability calibration or exact learning weights.',
            'GT was used for a development diagnostic and candidate design, never for selecting training pixels/pairs or loss strengths. The same validation subsequently supplies the complete endpoint, so it is not a fresh independent test.',
            'Online pseudo-anchor rates and full GT confusion use different anchor spaces. Restricting sources can lower pair coverage; selected-pair direction changes cannot replace all-class and old/new endpoint tradeoff assessment.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--compact', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if args.check_only:
        try:
            state = completion(root)
        except (ValueError, KeyError, OSError, TypeError) as exc:
            state = {'complete': False, 'invalid': True, 'error': str(exc)}
        print(json.dumps(state, allow_nan=False))
        raise SystemExit(0 if state['complete'] else 3 if state.get('invalid') else 2)
    report = analyze(root)
    output = args.output or root/'runs/prototype_sep_v1/new_anchor_analysis.json'
    primary.write_report(output, report)
    result = {'report': str(output), 'status': report['status'],
        'stage2': report['candidate_endpoint']['metrics'],
        'delta_vs_optimizedKD': report['comparisons']['optimized_KD']['metric_delta_pp'],
        'delta_vs_A': report['comparisons']['A_geometry']['metric_delta_pp'],
        'delta_vs_B': report['comparisons']['B_semantic']['metric_delta_pp'],
        'recommendation': report['recommendation']}
    if not args.compact:
        result['completion'] = report['completion']
        result['protection_summary'] = {key: value for key, value in report['semantic_protection_mechanism'].items()
                                       if key not in ['snapshots', 'conflict_occurrences_by_class']}
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
