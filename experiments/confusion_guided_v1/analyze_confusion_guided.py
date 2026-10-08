"""Analyze final confusion-guided endpoints using existing evaluation counts.

Pure standard-library reporting: no model loading, training, pixel-GT feedback
or GPU use. Confusion rates are row-normalized GT -> prediction error rates.
Only iteration-8000 stage-2 endpoints are compared; intermediate best pairs
never determine the recommendation.
"""
from pathlib import Path
import argparse
import datetime
import json
import math
import os


DEFAULT_ROOT = Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto')
WRITE_BOUNDARY = Path('/ML-vePFS/infra_rd/kun/others/wzg')
CLASSES = ['_background_', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
           'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
METRICS = ['all_miou', 'previous_foreground_miou', 'current_foreground_miou']
HIGHLIGHT_PAIRS = [(12, 8), (8, 12), (18, 9), (9, 18), (13, 10), (10, 13)]
EPSILON = 1e-10
BASE_CANDIDATES = ['a_sep', 'b_pairkd']
GUARD_CANDIDATE = 'c_newaware_sep'
PAIRKD_ONLY_CANDIDATE = 'd_pairkd_only'


def read_json(path):
    return json.loads(path.read_text())


def last_json(path):
    if not path.exists():
        return None
    for line in reversed(path.read_text().splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def validated_class_iou(values):
    if not isinstance(values, dict) or set(values) != set(CLASSES):
        raise ValueError('Expected exact VOC21 class names in endpoint IoU')
    ordered = [float(values[name]) for name in CLASSES]
    if any(not math.isfinite(value) or not 0 <= value <= 100 for value in ordered):
        raise ValueError('Endpoint IoU is nonfinite or outside [0, 100]')
    return ordered


def performance(iou):
    return {'all_miou': sum(iou)/21,
            'previous_foreground_miou': sum(iou[1:16])/15,
            'current_foreground_miou': sum(iou[16:21])/5}


def validated_histogram(hist):
    if not isinstance(hist, list) or len(hist) != 21 or any(len(row) != 21 for row in hist):
        raise ValueError('Expected a complete 21x21 endpoint histogram')
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value < 0 or int(value) != value
           for row in hist for value in row):
        raise ValueError('Histogram must contain finite nonnegative integer counts')
    result = [[int(value) for value in row] for row in hist]
    rows = [sum(row) for row in result]
    if any(value == 0 for value in rows):
        raise ValueError('Every VOC21 GT class must have evaluation support')
    columns = [sum(result[i][j] for i in range(21)) for j in range(21)]
    iou = [100*result[i][i]/(rows[i]+columns[i]-result[i][i]) for i in range(21)]
    return result, rows, iou


def load_endpoint(path):
    raw = read_json(path)
    if raw.get('step') != 2 or raw.get('iteration') != 8000:
        raise ValueError(f'Not the final stage-2 endpoint: {path}')
    iou = validated_class_iou(raw['class_iou'])
    hist, rows = None, None
    if 'histogram' in raw:
        hist, rows, recomputed = validated_histogram(raw['histogram'])
        error = max(abs(a-b) for a, b in zip(iou, recomputed))
        if error > 1e-6:
            raise ValueError(f'Endpoint IoU/histogram disagree by {error}: {path}')
    metrics = performance(iou)
    # Historical old_miou/new_miou may refer to initial10/cumulative10; use
    # explicit previous15/current5 grouping calculated from class IoU.
    for key in METRICS:
        if key in raw and abs(float(raw[key])-metrics[key]) > 1e-6:
            raise ValueError(f'Endpoint {key} disagrees with class IoU: {path}')
    return {'source': str(path), 'step': 2, 'iteration': 8000,
            'images': raw.get('images'), 'checkpoint_sha256': raw.get('checkpoint_sha256'),
            'metrics': metrics, 'class_iou': dict(zip(CLASSES, iou)),
            'histogram': hist, 'gt_class_pixels': rows,
            'histogram_available': hist is not None}


def load_historical_iou(root, method):
    path = root/f'runs/fixed_baseline_v1/{method}/10-5/step2/metrics.jsonl'
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    final = [row for row in rows if row.get('iteration') == 8000]
    if not final:
        return None
    iou = validated_class_iou(final[-1]['class_iou'])
    return {'source': str(path), 'step': 2, 'iteration': 8000,
            'images': None, 'checkpoint_sha256': None,
            'metrics': performance(iou), 'class_iou': dict(zip(CLASSES, iou)),
            'histogram': None, 'gt_class_pixels': None, 'histogram_available': False}


def direction(endpoint, source, target):
    hist = endpoint['histogram']
    pixels = hist[source][target]
    denominator = endpoint['gt_class_pixels'][source]
    return {'source_id': source, 'source_class': CLASSES[source],
            'target_id': target, 'target_class': CLASSES[target],
            'error_pixels': pixels, 'gt_source_pixels': denominator,
            'error_percent': 100*pixels/denominator}


def directional_delta(candidate, reference, source, target):
    current = direction(candidate, source, target)
    baseline = direction(reference, source, target)
    return {'source_id': source, 'source_class': CLASSES[source],
            'target_id': target, 'target_class': CLASSES[target],
            'candidate_error_percent': current['error_percent'],
            'reference_error_percent': baseline['error_percent'],
            'delta_error_pp': current['error_percent']-baseline['error_percent'],
            'candidate_error_pixels': current['error_pixels'],
            'reference_error_pixels': baseline['error_pixels'],
            'gt_source_pixels': current['gt_source_pixels']}


def aggregate_errors(endpoint):
    hist = endpoint['histogram']
    groups = {
        'old_to_background': (range(1, 16), [0]),
        'new_to_old': (range(16, 21), range(1, 16)),
        'old_to_new': (range(1, 16), range(16, 21)),
        'new_to_background': (range(16, 21), [0]),
    }
    result = {}
    for name, (sources, targets) in groups.items():
        numerator = sum(hist[i][j] for i in sources for j in targets)
        denominator = sum(endpoint['gt_class_pixels'][i] for i in sources)
        result[name] = {'error_pixels': numerator, 'gt_source_pixels': denominator,
                        'error_percent': 100*numerator/denominator}
    return result


def selected_pairs(run):
    path = run/'10-5/step2/pair_metrics.jsonl'
    row = last_json(path)
    if row is None:
        return {'available': False, 'source': str(path), 'directions': []}
    selector = row.get('selector', {})
    targets = selector.get('targets')
    if not isinstance(targets, list) or len(targets) != 21:
        raise ValueError(f'Final selector target shape is invalid: {path}')
    rates = selector.get('selected_rates', [None]*21)
    if len(rates) != 21:
        raise ValueError(f'Final selector rate shape is invalid: {path}')
    directions = []
    for i, j in enumerate(targets):
        if not isinstance(j, int) or not -1 <= j < 21:
            raise ValueError(f'Invalid selector class ID: {path}')
        if i > 0 and j > 0 and i != j:
            directions.append({'source_id': i, 'source_class': CLASSES[i],
                               'target_id': j, 'target_class': CLASSES[j],
                               'selected_estimated_rate': rates[i]})
    return {'available': True, 'source': str(path), 'logged_iteration': row.get('iteration'),
            'is_final_iteration': row.get('iteration') == 8000,
            'last_refresh_iteration': selector.get('last_refresh_iteration'),
            'directions': directions,
            'caveat': 'Targets are the last logged selector directions; they may have changed during training. Rates ranked directions and did not become calibrated loss weights.'}


def comparison(candidate, reference, selected):
    metrics = {key: candidate['metrics'][key]-reference['metrics'][key] for key in METRICS}
    classes = [{'class_id': i, 'class_name': name,
                'candidate_iou': candidate['class_iou'][name],
                'reference_iou': reference['class_iou'][name],
                'delta_iou_pp': candidate['class_iou'][name]-reference['class_iou'][name]}
               for i, name in enumerate(CLASSES)]
    result = {'metric_delta_pp': metrics, 'per_class_delta': classes,
              'classes_improved': sorted([row for row in classes if row['delta_iou_pp'] > EPSILON],
                                         key=lambda row: (-row['delta_iou_pp'], row['class_id'])),
              'classes_worsened': sorted([row for row in classes if row['delta_iou_pp'] < -EPSILON],
                                         key=lambda row: (row['delta_iou_pp'], row['class_id'])),
              'confusion_comparison_available': candidate['histogram_available'] and reference['histogram_available']}
    if not result['confusion_comparison_available']:
        result['confusion_limitation'] = 'Reference has only endpoint IoU; no direction error rates can be inferred from IoU.'
        return result
    if candidate['gt_class_pixels'] != reference['gt_class_pixels']:
        raise ValueError('Cannot compare directional rates: GT row totals differ across evaluations')
    if (candidate['images'] is not None and reference['images'] is not None
            and candidate['images'] != reference['images']):
        raise ValueError('Cannot compare endpoints: evaluated image counts differ')
    directions = [directional_delta(candidate, reference, i, j)
                  for i in range(1, 21) for j in range(1, 21) if i != j]
    result['foreground_direction_deltas'] = directions
    result['directions_improved'] = sorted(
        [row for row in directions if row['delta_error_pp'] < -EPSILON],
        key=lambda row: (row['delta_error_pp'], row['source_id'], row['target_id']))
    result['directions_worsened'] = sorted(
        [row for row in directions if row['delta_error_pp'] > EPSILON],
        key=lambda row: (-row['delta_error_pp'], row['source_id'], row['target_id']))
    result['directions_unchanged_count'] = sum(abs(row['delta_error_pp']) <= EPSILON for row in directions)
    result['highlight_directions'] = [directional_delta(candidate, reference, i, j) for i, j in HIGHLIGHT_PAIRS]
    result['selected_final_directions'] = [
        {**row, **directional_delta(candidate, reference, row['source_id'], row['target_id'])}
        for row in selected['directions']]
    current_aggregate, reference_aggregate = aggregate_errors(candidate), aggregate_errors(reference)
    result['aggregate_error_deltas'] = {
        name: {'candidate': current_aggregate[name], 'reference': reference_aggregate[name],
               'delta_error_pp': current_aggregate[name]['error_percent']-reference_aggregate[name]['error_percent']}
        for name in current_aggregate}
    result['direction_delta_sign'] = 'Negative means less GT->prediction confusion; positive means more confusion.'
    result['same_gt_row_totals_verified'] = True
    return result


def dominates(a, b):
    return (all(a[key] >= b[key]-EPSILON for key in METRICS)
            and any(a[key] > b[key]+EPSILON for key in METRICS))


def recommendation(endpoints, all_candidates_available, candidates=None):
    # Overall final mIoU is the primary deployment criterion. Old/new results
    # and the Pareto set make tradeoffs explicit; no highlighted pair enters
    # this ranking or substitutes for complete endpoint performance.
    ranked = sorted(endpoints, key=lambda name: (
        -endpoints[name]['metrics']['all_miou'],
        -endpoints[name]['metrics']['previous_foreground_miou'],
        -endpoints[name]['metrics']['current_foreground_miou'], name))
    best = ranked[0]
    pareto = [name for name in ranked if not any(
        other != name and dominates(endpoints[other]['metrics'], endpoints[name]['metrics'])
        for other in ranked)]
    tradeoffs = {}
    for other in ranked[1:]:
        delta = {key: endpoints[best]['metrics'][key]-endpoints[other]['metrics'][key] for key in METRICS}
        tradeoffs[other] = {'best_minus_reference_pp': delta,
                            'lower_metrics': [key for key in METRICS if delta[key] < -EPSILON]}
    candidates = candidates or BASE_CANDIDATES
    result = {'status': 'final' if all_candidates_available else 'provisional_pending_final_endpoint_or_evaluation_workers',
            'recommended_retained_method': best if all_candidates_available else None,
            'best_available_endpoint_method_pending_completion': None if all_candidates_available else best,
            'ranking_by_final_overall_miou': ranked,
            'pareto_methods_overall_old_new': pareto, 'tradeoffs_against_other_methods': tradeoffs,
            'selection_rule': 'Retain the best complete final overall mIoU endpoint; inspect old15/new5 tradeoffs. No class-pair result is used to pick the method.',
            'candidate_gain_over_optimized_kd_pp': {
                name: {key: endpoints[name]['metrics'][key]-endpoints['optimized_KD']['metrics'][key] for key in METRICS}
                for name in candidates if name in endpoints and 'optimized_KD' in endpoints},
            'statistical_claim': 'Descriptive fixed-seed, single-task endpoint comparison. This is not a significance test or proof of generalization.'}
    if GUARD_CANDIDATE in endpoints and 'a_sep' in endpoints:
        result['guard_candidate_gain_over_parent_a_pp'] = {
            key: endpoints[GUARD_CANDIDATE]['metrics'][key]-endpoints['a_sep']['metrics'][key]
            for key in METRICS}
    return result


def evaluation_completion(run, endpoint_exists, expected_workers=None):
    """Do not certify a candidate while its evaluation workers remain live."""
    receipt_path = run/'evaluation_workers_complete.json'
    receipt = read_json(receipt_path) if receipt_path.exists() else None
    returncodes = receipt.get('returncodes') if isinstance(receipt, dict) else None
    successful_workers = (isinstance(returncodes, list) and bool(returncodes)
                          and all(isinstance(code, int) and not isinstance(code, bool) and code == 0
                                  for code in returncodes))
    if expected_workers is not None:
        successful_workers = successful_workers and len(returncodes or []) == expected_workers
    missing_queued_results = [str(run/'evaluations'/path.stem/'result.json')
                             for path in sorted((run/'eval_queue').glob('*.json'))
                             if not (run/'evaluations'/path.stem/'result.json').exists()]
    return {'complete': bool(endpoint_exists and successful_workers and not missing_queued_results),
            'final_stage2_endpoint_exists': endpoint_exists,
            'evaluation_worker_receipt': str(receipt_path),
            'evaluation_workers_exited_successfully': successful_workers,
            'evaluation_worker_returncodes': returncodes,
            'expected_evaluation_workers': expected_workers,
            'missing_queued_evaluations': missing_queued_results}


def guard_development_diagnostic(run_root):
    """Read the saved diagnostic receipt; never infer training correctness."""
    path = run_root/'diagnostics/sep_new_conflict/merged.json'
    if not path.exists():
        return {'available': False, 'source': str(path)}
    raw = read_json(path)
    categories = ['all_old_eligible', 'correct_old_anchor', 'wrong_new_gt', 'direct_new_target']
    checkpoints = []
    for row in raw.get('results', []):
        if row.get('images') != 128:
            raise ValueError('Guard development diagnostic must use the recorded128-image subset')
        checkpoints.append({'checkpoint_label': row.get('checkpoint_label'), 'images': row['images'],
                            'checkpoint_sha256': row.get('checkpoint_sha256'),
                            'category_totals': {name: row.get('category_totals', {}).get(name) for name in categories}})
    return {'available': True, 'source': str(path), 'images': 128, 'checkpoints': checkpoints,
            'same_images_and_checkpoint_identity_verified': raw.get('same128_images_and_checkpoint_identity_verified'),
            'gt_enters_gates': raw.get('gt_enters_gates'),
            'usage': 'These128GT diagnostic images informed selection of the protected candidate. Static attenuation selectivity motivates a hypothesis; it does not establish trained endpoint gain.',
            'push_definition': 'Evidence_weight*sigmoid(1-native_pair_logit_margin), before class normalization,0.5 branch averaging and outer lambda/ramp; not a parameter-gradient norm.',
            'limitations': raw.get('limitations', [])}


def atomic_json(path, value):
    boundary = WRITE_BOUNDARY.resolve()
    target = path.resolve()
    if not target.is_relative_to(boundary):
        raise ValueError(f'Report output leaves authorized workspace: {path}')
    if path.exists() and path.stat().st_nlink > 1:
        raise ValueError('Refuse report write through a multiply linked file')
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name+f'.tmp.{os.getpid()}')
    if temporary.exists():
        raise ValueError('Refuse existing temporary output path')
    with temporary.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
    os.replace(temporary, target)


def analyze(root):
    run_root = root/'runs/confusion_guided_v1'
    endpoints, selections, missing = {}, {}, []
    candidates = list(BASE_CANDIDATES)
    c_study_path = run_root/'formal'/GUARD_CANDIDATE/'study.json'
    c_study = read_json(c_study_path) if c_study_path.exists() else None
    if c_study_path.exists():
        candidates.append(GUARD_CANDIDATE)
    d_study_path = run_root/'formal'/PAIRKD_ONLY_CANDIDATE/'study.json'
    d_study = read_json(d_study_path) if d_study_path.exists() else None
    if d_study_path.exists():
        candidates.append(PAIRKD_ONLY_CANDIDATE)
    completion = {}
    for arm in candidates:
        run = run_root/'formal'/arm
        path = run/'evaluations/step2_iter8000/result.json'
        if path.exists():
            endpoints[arm] = load_endpoint(path)
            selections[arm] = selected_pairs(run)
        else:
            missing.append(str(path))
        completion[arm] = evaluation_completion(run, path.exists(),
            expected_workers=4 if arm == PAIRKD_ONLY_CANDIDATE else None)
    optimized = root/'runs/kd_parallel_v1/formal/b_relational/evaluations/step2_iter8000/result.json'
    if optimized.exists():
        endpoints['optimized_KD'] = load_endpoint(optimized)
    else:
        missing.append(str(optimized))
    for method in ['base', 'kd', 'sep', 'kd_sep']:
        hist_path = root/f'runs/kd_pixel_v1/evaluations/baseline_{method}_step2/result.json'
        endpoint = load_endpoint(hist_path) if hist_path.exists() else load_historical_iou(root, method)
        if endpoint is not None:
            endpoints[method] = endpoint
    comparisons = {
        arm: {reference: comparison(endpoints[arm], endpoint, selections[arm])
              for reference, endpoint in endpoints.items() if reference != arm}
        for arm in candidates if arm in endpoints}
    warnings = []
    for arm, selection in selections.items():
        if not selection['available'] or not selection.get('is_final_iteration'):
            warnings.append(f'{arm}: final-iteration selector log unavailable; listed directions are not confirmed final')
    complete = ('optimized_KD' in endpoints
                and all(arm in endpoints and completion[arm]['complete'] for arm in candidates))
    pending_completion = {arm: status for arm, status in completion.items() if not status['complete']}
    if pending_completion:
        warnings.append('Recommendation remains pending until every registered formal candidate has a final endpoint and successful evaluation-worker completion.')
    guard_lineage = None
    if GUARD_CANDIDATE in candidates:
        guard_lineage = {
            'parent_arm': 'a_sep', 'registered_study': str(c_study_path),
            'stage1': 'Inherited completed parent A stage1; no new stage1 training.',
            'stage2': 'Resume parent A iteration2000 model, optimizer, observer and selector; train only remaining6000 iterations to the same8000 endpoint.',
            'gpu_strategy': '8GPUs x single-card batch1, globalbatch8; parent A used4GPUs x batch2.',
            'mechanism': 'Only old SEP numerator evidence weights multiply (1-max_current_new_CAM)^2; new anchors, eligible counts, matrix-selection rule and lambda unchanged.',
            'study': c_study,
            'development_diagnostic': guard_development_diagnostic(run_root),
            'causal_limitations': ['The factor also lowers effective old SEP supervision mass; lambda unchanged does not mean equal effective strength.',
                                  'GPU grouping and restarted data/augmentation sampling prevent a bitwise matched continuation; parent A differences are descriptive, not isolated causal or significance evidence.',
                                  'The protected candidate was selected after observing A/B endpoints and128-image GT diagnostics; it is an adaptive exploration within this task, not an independent confirmation.',
                                  'The same VOC validation set supplies the full endpoint evaluation. This is development validation, not a new independent test set.']}
    pairkd_only_lineage = None
    if PAIRKD_ONLY_CANDIDATE in candidates:
        optimized_run = root/'runs/kd_parallel_v1/formal/b_relational'
        pairkd_only_lineage = {
            'parent_method': 'optimized_KD', 'parent_arm': 'b_relational',
            'registered_study': str(d_study_path), 'study': d_study,
            'stage1': 'Inherited completed optimized-KD b_relational stage1; no new stage1 training and no inherited A/C teacher.',
            'stage1_teacher_checkpoint': str(optimized_run/'10-5/step1/checkpoints/model_final.pth'),
            'stage2': 'Resume optimized-KD step2 iteration2000 student and optimizer; train only remaining6000 iterations to the8000 endpoint.',
            'shared_stage2_warmup_checkpoint': str(optimized_run/'10-5/step2/checkpoints/model_iter_2000.pth'),
            'observer_state': 'Cold start at resumed iteration2000 with zero observations; no observer/selector history is inherited from warmup.',
            'mechanism': 'Teacher-compatible directed pair KD only: select both source and partner within1..15 from the full21-class EMA, preserve teacher binary soft relations and cap blend0.5; w_pair_sep=0, existing relational KD gates and normalization retained.',
            'selection_domain': {'selector_schema': 2, 'distillation_class_limit': 16,
                                 'source_class_ids': list(range(1, 16)), 'partner_class_ids': list(range(1, 16)),
                                 'ranking_counts': 'Full21-class online EMA with original support, rate, staleness and ramp constraints.',
                                 'rate_denominator': 'Full21-class source-row mass including background, diagonal and new classes; no old-submatrix renormalization.',
                                 'teacher_unknown_classes': list(range(16, 21))},
            'gpu_strategy': '8GPUs x single-card batch1, globalbatch8; optimized-KD parent used4GPUs x batch2.',
            'evaluation_workers_required': 4,
            'causal_limitations': ['D and optimized KD share the stage1 teacher and step2 warmup model/optimizer, reducing initialization differences.',
                                  'GPU partitioning and sampler/augmentation restart still differ; endpoint deltas do not strictly isolate pair KD as a single causal factor.',
                                  'Confusion observer cold-starts after resume and ramps independently; it did not observe the first2000 steps.',
                                  'D adapts candidate selection to the teacher-known class domain as well as omitting SEP; it is not merely a SEP-off copy of arm B.',
                                  'D was selected after A/B/C development results on the same task. Full endpoint validation is not a new independent test.']}
    return {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'scope': {'task': 'VOC 10-5', 'seed': 0, 'step': 2, 'iteration': 8000,
                      'classes': CLASSES, 'old_foreground_ids': list(range(1, 16)),
                      'new_foreground_ids': list(range(16, 21)),
                      'all_miou_includes_background': True,
                      'direction_definition': 'GT source class i -> predicted class j, normalized by all GT pixels of class i',
                      'units': 'IoU and error rates in percent; changes in percentage points',
                      'gt_usage': 'Existing held-out evaluation histogram only; no GT enters training or pair selection',
                      'limitations': ['One fixed seed and one task; no statistical significance or generalization claim.',
                                      'Reused warmup and multi-GPU sampler restart are not a bitwise continuation of archived baseline.',
                                      'Class-pair errors can improve while other classes worsen; complete endpoint metrics govern retention.',
                                      'Selected final directions summarize the last selector state, not all earlier targeted directions.',
                                      'The128-image GT diagnostic informed selection of the C guard; full endpoints on the same validation set are development validation, not an independent unseen test.',
                                      'C changes GPU partitioning from4rank batch2 to8rank batch1 and restarts sampler/augmentation. Its parent-A differences cannot strictly isolate the guard as a single causal factor.',
                                      'Static guard selectivity is a mechanism diagnostic and cannot substitute for final trained accuracy gain.',
                                      'D reuses the optimized-KD teacher/warmup student/optimizer but changes GPU partitioning and restarts sampling; its observer cold-starts at iteration2000. This is descriptive development exploration, not single-factor causal proof.']},
            'status': 'complete_endpoint_analysis' if complete else 'waiting_for_final_endpoints',
            'missing_endpoints': missing, 'warnings': warnings, 'endpoints': endpoints,
            'required_candidates': candidates, 'candidate_evaluation_completion': completion,
            'pending_candidate_completion': pending_completion, 'guard_candidate_lineage': guard_lineage,
            'pairkd_only_candidate_lineage': pairkd_only_lineage,
            'final_selected_pairs': selections, 'comparisons': comparisons,
            'guard_vs_parent_a': comparisons.get(GUARD_CANDIDATE, {}).get('a_sep'),
            'pairkd_only_vs_optimized_kd': comparisons.get(PAIRKD_ONLY_CANDIDATE, {}).get('optimized_KD'),
            'recommendation': recommendation(endpoints, complete, candidates) if endpoints else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    report = analyze(args.root.resolve())
    if args.require_complete and report['status'] != 'complete_endpoint_analysis':
        raise RuntimeError('Final analysis not ready: '+json.dumps({
            'missing_endpoints': report['missing_endpoints'],
            'pending_candidate_completion': report['pending_candidate_completion']}))
    output = args.output or args.root/'runs/confusion_guided_v1/result_analysis.json'
    atomic_json(output, report)
    print(json.dumps({'output': str(output), 'status': report['status'],
                      'metrics': {name: endpoint['metrics'] for name, endpoint in report['endpoints'].items()},
                      'recommendation': report['recommendation'], 'warnings': report['warnings']}, allow_nan=False))


if __name__ == '__main__':
    main()
