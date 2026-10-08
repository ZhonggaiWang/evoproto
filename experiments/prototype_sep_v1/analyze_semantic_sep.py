"""Analyze the completed B semantic-protected stage2 repair, without GPU.

Uses the primary analysis helpers. B inherits A stage1 and A stage2/2000;
there is no newly trained B stage1. Refuses reporting until its one training
stage, four evaluators and all three full1449-image jobs exit successfully.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import math
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_prototype_sep as primary

ROOT = primary.ROOT
ITERATIONS = [4000, 6000, 8000]


def checked_evaluation(path, iteration):
    raw = primary.read_json(path)
    if raw.get('step') != 2 or raw.get('iteration') != iteration or raw.get('images') != 1449:
        raise ValueError(f'Not a full stage2/{iteration} evaluation: {path}')
    matrix, rows, columns, iou = primary.histogram(raw['histogram'], 21)
    if set(raw['class_iou']) != set(primary.CLASSES):
        raise ValueError('Stage2 evaluation class names disagree')
    reported = [primary.finite(raw['class_iou'][name], 'IoU') for name in primary.CLASSES]
    if max(abs(a-b) for a, b in zip(reported, iou)) > 1e-6:
        raise ValueError('Evaluation histogram/class IoU disagreement')
    metrics = {'all_miou': sum(iou)/21, 'previous_foreground_miou': sum(iou[1:16])/15,
               'current_foreground_miou': sum(iou[16:])/5}
    for key, value in metrics.items():
        if abs(primary.finite(raw[key], key)-value) > 1e-6:
            raise ValueError('Evaluation grouping/metric disagreement')
    return {'source': str(path), 'step': 2, 'iteration': iteration, 'images': 1449,
            'checkpoint_sha256': raw['checkpoint_sha256'], 'metrics': metrics,
            'histogram': matrix, 'gt_class_pixels': rows, 'pred_class_pixels': columns}


def completion(root):
    run = root/'runs/prototype_sep_v1/formal/b_semantic'
    parent = root/'runs/prototype_sep_v1/formal/a_geometry'
    required = [run/'study.json', run/'status.json', run/'training_complete.json',
                run/'evaluation_workers_complete.json', run/'10-5/step2/launch.json',
                run/'10-5/step2/training_complete.json',
                run/'10-5/step2/checkpoints/model_final.pth',
                parent/'status.json', parent/'10-5/step1/training_complete.json',
                parent/'eval_queue/step2_iter2000.json']
    required += [run/f'evaluator_rank{rank}.process.json' for rank in range(4)]
    for iteration in ITERATIONS:
        required += [run/f'eval_queue/step2_iter{iteration}.json',
                     run/f'evaluations/step2_iter{iteration}/result.json']
        required += [run/f'evaluations/step2_iter{iteration}/rank{rank}.json' for rank in range(4)]
    missing = [str(path) for path in required if not path.is_file()]
    failures = [str(run/name) for name in ['failure.json', 'training_failed.json'] if (run/name).exists()]
    if missing or failures:
        return {'complete': False, 'missing': missing, 'failure_receipts': failures}
    study = primary.read_json(run/'study.json')
    training = primary.read_json(run/'training_complete.json')
    cfg = study['common_config']
    teacher = parent/'10-5/step1/checkpoints/model_final.pth'
    resume = parent/'10-5/step2/checkpoints/model_iter_2000.pth'
    checks = {
        'formal_B_not_smoke': study.get('arm') == 'b_semantic' and study.get('smoke') is False,
        'B_status_complete': primary.read_json(run/'status.json').get('status') == 'complete',
        'one_stage_trained_stage1_inherited': training.get('status') == 'complete'
                and training.get('stages_trained') == 1 and training.get('inherited_stage1') == str(parent),
        'B_stage2_train_exit0': primary.read_json(run/'10-5/step2/launch.json').get('returncode') == 0
                and primary.read_json(run/'10-5/step2/training_complete.json').get('returncode') == 0,
        'four_eval_workers_exit0': primary.read_json(run/'evaluation_workers_complete.json').get('returncodes') == [0]*4,
        'parent_A_complete': primary.read_json(parent/'status.json').get('status') == 'complete',
        'parent_stage1_teacher_trained_exit0': primary.read_json(parent/'10-5/step1/training_complete.json').get('returncode') == 0,
        'teacher_and_warmup_owned_A': Path(study.get('teacher', '')) == teacher
                and Path(study.get('resume_checkpoint', '')) == resume,
        'stage1_inheritance_path_A': study.get('inherited_stage1_run') == str(parent),
        'resume_iteration2000': study.get('resume_iteration') == 2000,
        'teacher_identity_matches_A_stage1_receipt': study.get('teacher_sha256')
                == primary.read_json(parent/'10-5/step1/training_complete.json').get('checkpoint_sha256'),
        'warmup_identity_matches_A_step2_2000_job': study.get('resume_checkpoint_sha256')
                == primary.read_json(parent/'eval_queue/step2_iter2000.json').get('checkpoint_sha256'),
        'model_optimizer_online_selector_restored': set(study.get('restored_state', []))
                == {'student', 'optimizer', 'online_confusion', 'geometry_selector'},
        'task_seed_globalbatch': cfg.get('task') == '10-5' and cfg.get('seed') == 0
                and study.get('global_batch') == 8 and study.get('batch_per_gpu') == 1,
        'no_B_stage1_training_receipt': not (run/'10-5/step1/launch.json').exists(),
    }
    for rank in range(4):
        checks[f'evaluator{rank}_process_exit0'] = primary.read_json(run/f'evaluator_rank{rank}.process.json').get('returncode') == 0
    experiment = root/'experiments/prototype_sep_v1'
    source, runner = experiment/'b_semantic/src', experiment/'run_semantic_sep.py'
    actual = {str(path.relative_to(source)): primary.digest(path) for path in source.rglob('*.py')
              if '__pycache__' not in path.parts}
    checks['B_source_frozen'] = bool(study.get('source_sha256')) and actual == study['source_sha256']
    checks['B_runner_frozen'] = runner.is_file() and primary.digest(runner) == study.get('runner_sha256')
    queue = sorted((run/'eval_queue').glob('*.json'))
    checks['exact_three_remaining_evaluation_jobs'] = [path.stem for path in queue] == [f'step2_iter{iteration}' for iteration in ITERATIONS]
    checks['no_evaluation_error_receipts'] = not list((run/'evaluations').glob('*/error_rank*.json'))
    split = Path(cfg['list_folder'])/'incremental_split'/f"val_{cfg['task']}_step_3.txt"
    expected_names = split.read_text().splitlines() if split.is_file() else []
    evaluations, first_rows = {}, None
    for iteration in ITERATIONS:
        folder = run/f'evaluations/step2_iter{iteration}'
        result = checked_evaluation(folder/'result.json', iteration)
        job = primary.read_json(run/f'eval_queue/step2_iter{iteration}.json')
        parts = [primary.read_json(folder/f'rank{rank}.json') for rank in range(4)]
        names = [name for part in parts for name in part['images']]
        checks[f'iter{iteration}_full1449_exact_val_names'] = len(names) == len(set(names)) == 1449
        checks[f'iter{iteration}_full1449_exact_val_names'] &= len(expected_names) == 1449 and set(names) == set(expected_names)
        checks[f'iter{iteration}_part_identifiers'] = all(part.get('rank') == rank and part.get('shards') == 4
                and part.get('step') == 2 and part.get('iteration') == iteration for rank, part in enumerate(parts))
        checks[f'iter{iteration}_checkpoint_identity'] = (job.get('step') == 2 and job.get('iteration') == iteration
                and job.get('checkpoint_sha256') == result['checkpoint_sha256']
                and all(part.get('checkpoint_sha256') == result['checkpoint_sha256'] for part in parts))
        for part in parts:
            if len(part['histogram']) != 21 or any(len(row) != 21 for row in part['histogram']):
                raise ValueError('Malformed B shard histogram')
            for row in part['histogram']:
                for value in row:
                    primary.integer(value, 'shard count')
        summed = [[sum(part['histogram'][i][j] for part in parts) for j in range(21)] for i in range(21)]
        checks[f'iter{iteration}_histogram_additive_consistency'] = summed == result['histogram']
        if first_rows is None:
            first_rows = result['gt_class_pixels']
        checks[f'iter{iteration}_same_full_GT_rows'] = result['gt_class_pixels'] == first_rows
        evaluations[str(iteration)] = {key: result[key] for key in ['source','images','metrics','checkpoint_sha256']}
    return {'complete': all(checks.values()), 'checks': checks, 'missing': missing,
            'failure_receipts': failures, 'evaluations': evaluations,
            'inherited_stage1': str(parent), 'validation_list': str(split)}


def protection_statistics(stage):
    path = stage/'geometry_metrics.jsonl'
    rows = primary.read_rows(path)
    guard_path = stage/'semantic_guard_metrics.jsonl'
    independent = primary.read_rows(guard_path) if guard_path.is_file() else None
    independent_map = {}
    if independent is not None:
        for entry in independent:
            key = entry.get('step'), entry.get('iteration')
            if key in independent_map:
                raise ValueError('Duplicated independent semantic guard snapshot')
            independent_map[key] = entry
        if set(independent_map) != {(row['step'],row['iteration']) for row in rows}:
            raise ValueError('Independent protection log and geometry log iterations disagree')
    snapshots, class_conflicts = [], [0]*21
    failures = []
    for row in rows:
        info = row.get('semantic_protection')
        if not isinstance(info, dict) or not isinstance(info.get('enabled'), bool):
            raise ValueError('Missing explicit semantic_protection stats in B geometry log')
        if independent is not None:
            separate = independent_map[row['step'],row['iteration']]
            if any(separate.get(key) != row[key] for key in ['weight','ramp','active']):
                raise ValueError('Independent protection activation/coefficient metadata disagrees')
            payload = {key:value for key,value in separate.items()
                       if key not in ['step','iteration','weight','ramp','active']}
            if payload != info:
                raise ValueError('Independent and nested protection stats disagree')
            info = payload
        expected_weight = row['weight']*row['ramp'] if row['active'] else 0.
        effective_weight = primary.finite(info['effective_geometry_weight'], 'effective protected coefficient')
        if abs(effective_weight-expected_weight) > primary.EPS:
            raise ValueError('Protected coefficient disagrees with original geometry activation/ramp')
        if not info['enabled']:
            if row['active'] or info.get('reason') != 'loss_warmup' or effective_weight != 0:
                raise ValueError('Protection warmup state disagrees with active loss')
            snapshots.append({'iteration': row['iteration'], 'enabled': False, 'effective_geometry_weight': 0.})
            continue
        if not row['active']:
            raise ValueError('SEM protection enabled during inactive geometry loss')
        if info.get('world_size') != 8 or info.get('old_classes') != 15 or info.get('classes_including_background') != 21:
            raise ValueError('Protection world/class grouping mismatch')
        if info.get('forward_value_preserved') is not True or info.get('background_old_sep_gradient_zero') is not True:
            raise ValueError('Protection changed forward loss or allowed old/BG SEP gradients')
        actual_loss = primary.finite(info['weighted_geometry_forward_value'], 'protected forward scalar')
        if abs(actual_loss-expected_weight*row['prototype_sep']) > 1e-6:
            raise ValueError('Protected forward value differs from the original weighted SEP')
        used = info.get('semantic_gradient_used')
        if not isinstance(used, bool) or len(info.get('per_class', [])) != 21:
            raise ValueError('Protection gradient-use/per-class stats invalid')
        conflict_count, tiny_count = 0, 0
        minimum_dot, minimum_cosine = None, None
        for i, entry in enumerate(info['per_class']):
            is_new = i > 15
            if entry.get('class_id') != i or entry.get('is_new') is not is_new:
                raise ValueError('Protected class IDs/new mask disagree')
            before = primary.finite(entry['geometry_tangent_norm_before'], 'geometry norm before')
            after = primary.finite(entry['geometry_tangent_norm_after'], 'geometry norm after')
            if min(before, after) < 0 or (not is_new and (before != 0 or after != 0)):
                raise ValueError('Protected tangent norm/old gradient invalid')
            conflict, tiny = entry['conflict_projected'], entry['small_norm_conflict_zeroed']
            if not isinstance(conflict, bool) or not isinstance(tiny, bool) or (tiny and not conflict):
                raise ValueError('Projection flags invalid')
            if used:
                sem_norm = primary.finite(entry['semantic_tangent_norm'], 'semantic norm')
                dot_before = primary.finite(entry['dot_before'], 'dot before')
                dot_after = primary.finite(entry['dot_after'], 'dot after')
                tolerance = primary.finite(entry['roundoff_dot_tolerance'], 'roundoff tolerance')
                if sem_norm < 0 or tolerance < 0 or conflict != (is_new and dot_before < 0):
                    raise ValueError('Conflict flag/semantic norm inconsistent')
                if is_new and dot_after < -tolerance:
                    failures.append({'iteration':row['iteration'],'class_id':i,'dot_after':dot_after,'tolerance':tolerance})
                if is_new:
                    minimum_dot = dot_after if minimum_dot is None else min(minimum_dot, dot_after)
                    cosine = entry.get('cosine_after')
                    if cosine is not None:
                        cosine = primary.finite(cosine, 'after cosine')
                        minimum_cosine = cosine if minimum_cosine is None else min(minimum_cosine, cosine)
            elif any(entry.get(key) is not None for key in ['semantic_tangent_norm','dot_before','dot_after','roundoff_dot_tolerance']):
                raise ValueError('Zero-geometry path must not claim measured semantic gradient')
            conflict_count += int(conflict)
            tiny_count += int(tiny)
            class_conflicts[i] += int(conflict)
        if conflict_count != info['conflicting_new_rows'] or tiny_count != info['small_norm_conflict_zeroed_rows']:
            raise ValueError('Protected conflict summary disagrees with class entries')
        before = primary.finite(info['geometry_tangent_norm_before'], 'new geometry norm before')
        after = primary.finite(info['geometry_tangent_norm_after'], 'new geometry norm after')
        removed = primary.finite(info['removed_geometry_tangent_norm'], 'removed geometry norm')
        if min(before, after, removed) < 0:
            raise ValueError('Negative protected norm summary')
        snapshots.append({'iteration': row['iteration'], 'enabled': True,
                          'effective_geometry_weight': effective_weight,
                          'semantic_gradient_used': used,
                          'local_semantic_gradient_connected': info.get('local_semantic_gradient_connected'),
                          'active_geometry_new_rows': info['active_geometry_new_rows'],
                          'conflicting_new_rows': conflict_count,
                          'small_norm_conflict_zeroed_rows': tiny_count,
                          'weighted_geometry_forward_value': actual_loss,
                          'geometry_tangent_norm_before': before,
                          'geometry_tangent_norm_after': after,
                          'removed_geometry_tangent_norm': removed,
                          'removed_over_before_norm': removed/before if before > 0 else None,
                          'minimum_new_dot_after': minimum_dot,
                          'minimum_new_cosine_after': minimum_cosine})
    if failures:
        raise ValueError('Logged protection failed its declared roundoff bound: '+json.dumps(failures))
    enabled = [row for row in snapshots if row['enabled']]
    nonzero = [row for row in enabled if row['geometry_tangent_norm_before'] > 0]
    semantic = [row for row in enabled if row['semantic_gradient_used']]
    return {'source': str(guard_path if independent is not None else path),
            'nested_geometry_source':str(path),
            'independent_and_nested_logs_exactly_match':True if independent is not None else None,
            'independent_log_available':independent is not None,
            'logged_snapshots':len(rows), 'enabled_snapshots':len(enabled),
            'semantic_gradient_used_snapshots':len(semantic),
            'zero_geometry_skip_semantic_snapshots':sum(not row['semantic_gradient_used'] for row in enabled),
            'snapshots_with_conflicting_new_rows':sum(row['conflicting_new_rows'] > 0 for row in enabled),
            'total_logged_conflicting_row_occurrences':sum(row['conflicting_new_rows'] for row in enabled),
            'conflict_occurrences_by_class':{primary.CLASSES[i]:class_conflicts[i] for i in range(21)},
            'removed_over_before_norm':primary.distribution([row['removed_over_before_norm'] for row in nonzero]),
            'before_tangent_norm':primary.distribution([row['geometry_tangent_norm_before'] for row in enabled]),
            'after_tangent_norm':primary.distribution([row['geometry_tangent_norm_after'] for row in enabled]),
            'logged_roundoff_bound_violations':failures, 'snapshots':snapshots,
            'scope':'Rank0 recorded moments of replica geometry and pooled semantic tangent gradients; not population coverage, raw parameter-gradient norms, AdamW updates or GT reliability.',
            'value_caveat':'The surrogate keeps the original weighted SEP forward value while changing its gradient. Equal losses do not establish equal optimization strength.',
            'guarantee_limit':'Current Euclidean per-row component-gradient alignment, under unit normalization and replica contracts. No AdamW momentum/preconditioning, convergence, teacher quality or IoU guarantee.'}


def analyze(root):
    run = root/'runs/prototype_sep_v1/formal/b_semantic'
    parent = root/'runs/prototype_sep_v1/formal/a_geometry'
    complete = completion(root)
    if not complete['complete']:
        raise RuntimeError('B final analysis withheld until successful completion: '+json.dumps(complete))
    study = primary.read_json(run/'study.json')
    candidate = primary.endpoint(run/'evaluations/step2_iter8000/result.json', 2)
    references = {
        'optimized_KD':primary.endpoint(root/'runs/kd_parallel_v1/formal/b_relational/evaluations/step2_iter8000/result.json',2),
        'A_geometry':primary.endpoint(parent/'evaluations/step2_iter8000/result.json',2),
    }
    inherited = primary.endpoint(parent/'evaluations/step1_iter8000/result.json',1)
    weight = primary.finite(study['common_config']['w_geometry_sep'],'geometry coefficient')
    if abs(weight-.1) > primary.EPS or abs(study['common_config']['proto_margin']) > primary.EPS:
        raise ValueError('Unexpected original geometry coefficient or margin')
    geometry = primary.mechanism(run/'10-5/step2',2,weight)
    protection = protection_statistics(run/'10-5/step2')
    comparisons = {name:primary.comparison(candidate,reference,geometry['all_logged_selected_pairs']) for name,reference in references.items()}
    final_pairs = {name:primary.comparison(candidate,reference,geometry['final_snapshot_pairs'])['selected_pair_bidirectional_deltas']
                   for name,reference in references.items()}
    delta = comparisons['optimized_KD']['metric_delta_pp']['all_miou']
    wins = delta > primary.EPS
    selected = (run if wins else root/'runs/kd_parallel_v1/formal/b_relational')/'10-5/step2/checkpoints/model_final.pth'
    recommendation = {'method':'semantic_protected_prototype_SEP_plus_optimized_KD' if wins else 'existing_optimized_KD',
                      'selected_checkpoint':str(selected), 'candidate_exceeds_required_endpoint':wins,
                      'candidate_all_miou':candidate['metrics']['all_miou'],
                      'optimized_KD_all_miou':references['optimized_KD']['metrics']['all_miou'],
                      'delta_all_pp':delta,
                      'criterion':'Complete stage2/8000 all-class mIoU must exceed existing optimizedKD69.1492921947. Recovery relative to A alone is insufficient.',
                      'status':'observed_development_endpoint_gain' if wins else 'candidate_did_not_improve_required_endpoint'}
    return {'utc':datetime.now(timezone.utc).isoformat(),'status':'complete_endpoint_analysis',
            'run':str(run),'study':study,'completion':complete,'candidate_endpoint':candidate,
            'references':references,'comparisons':comparisons,
            'final_logged_selected_pair_bidirectional_deltas':final_pairs,
            'geometry_mechanism':geometry,'semantic_protection_mechanism':protection,
            'inherited_stage1':{'run':str(parent),'endpoint':inherited,
                               'new_training':False,'teacher':study['teacher'],
                               'teacher_sha256':study['teacher_sha256'],
                               'note':'B inherits A stage1, including its already-observed quality deficit; this is not the original optimizedKD stage1 teacher.'},
            'stage2_resume':{'checkpoint':study['resume_checkpoint'],'sha256':study['resume_checkpoint_sha256'],
                             'iteration':2000,'trained_remaining_iterations':6000,
                             'restored_state':study['restored_state'],
                             'observer_note':'Own A stage2 warmup online/selector state is restored; no new observer coldstart is claimed.'},
            'recommendation':recommendation,
            'scope':{'task':'VOC10-5','seed':0,'stage':2,'iteration':8000,'images':1449,
                     'all_miou_includes_background':True,
                     'GT_direction_definition':'GT i->prediction j divided by the whole GT i row, including correct, background and other predictions.',
                     'GT_role':'Existing evaluation only; never input to the selector, protection projection or exact learning weights.',
                     'selection_role':'Past pseudo/PAR->prediction broad EMA with trusted CAM support; this is a different anchor space from GT confusion.',
                     'intermediate_evaluations':'4000/6000 receipts are checked for full coverage and histogram consistency; they do not decide the winner.'},
            'limitations':['One fixedseed0 adaptively developed task; same validation set is development evidence, not a new independent test.',
                           'B resumes A stage2/2000 student, optimizer and online states and restarts sampling/augmentations; it is not a bitwise continuation of A.',
                           'Candidate uses8GPUs x batch1/global8; archived optimizedKD used4GPUs x batch2. Differences cannot strictly isolate the protection as one causal factor.',
                           'B and A share the A stage1 teacher. Their comparison targets the stage2 repair; comparison with optimizedKD also includes the inherited stage1 teacher difference.',
                           'The tangent projection protects the current Euclidean semantic component. AdamW state, preconditioning, weight decay, encoder dynamics and future IoU remain unconstrained.',
                           'Protection statistics describe logged tangent gradients and eligibility, not population error accuracy or measured optimizer steps.',
                           'A selected pair can improve one GT direction and worsen its reverse or other classes; full endpoint mIoU governs retention.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--compact',action='store_true')
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if args.check_only:
        state = completion(root)
        print(json.dumps(state,allow_nan=False))
        sys.exit(0 if state['complete'] else 2)
    report = analyze(root)
    output = args.output or root/'runs/prototype_sep_v1/semantic_analysis.json'
    primary.write_report(output,report)
    compact = {'report':str(output),'status':report['status'],
               'stage2':report['candidate_endpoint']['metrics'],
               'delta_vs_optimizedKD':report['comparisons']['optimized_KD']['metric_delta_pp'],
               'delta_vs_A':report['comparisons']['A_geometry']['metric_delta_pp'],
               'recommendation':report['recommendation']}
    if not args.compact:
        compact['completion'] = report['completion']
        compact['protection_summary'] = {key:value for key,value in report['semantic_protection_mechanism'].items()
                                         if key not in ['snapshots','conflict_occurrences_by_class']}
    print(json.dumps(compact,allow_nan=False))


if __name__ == '__main__':
    main()
