"""Analyze completed prototype SEP endpoints; standard library, no GPU.

Refuse final reporting until both train stages and every evaluation worker
exit successfully. Read existing evaluation histograms and mechanism logs;
never load a model, train, infer, optimize, or use pixel GT as supervision.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import uuid


AREA = Path('/ML-vePFS/infra_rd/kun/others/wzg')
ROOT = AREA/'workspace/evoproto'
CLASSES = ['_background_', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
           'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
METRICS = ['all_miou', 'previous_foreground_miou', 'current_foreground_miou']
EXPECTED_IMAGES = {1: 1240, 2: 1449}
FINAL_ITERATION = 8000
EPS = 1e-8


def read_json(path):
    return json.loads(Path(path).read_text())


def finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'Expected finite numeric {label}')
    return float(value)


def integer(value, label):
    number = finite(value, label)
    if int(number) != number or number < 0:
        raise ValueError(f'Expected nonnegative integer {label}')
    return int(number)


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for data in iter(lambda: handle.read(8*1024*1024), b''):
            hasher.update(data)
    return hasher.hexdigest()


def read_rows(path):
    rows = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f'Malformed completed log {path}:{number}') from exc
    return rows


def completion(run):
    """Known runner receipts, success exits, all queued results and full shards."""
    required = [run/'study.json', run/'status.json', run/'training_complete.json',
                run/'evaluation_workers_complete.json']
    for step in [1, 2]:
        stage = run/f'10-5/step{step}'
        required += [stage/'launch.json', stage/'training_complete.json',
                     stage/'checkpoints/model_final.pth',
                     run/f'evaluations/step{step}_iter8000/result.json']
    required += [run/f'evaluator_rank{rank}.process.json' for rank in range(4)]
    missing = [str(path) for path in required if not path.is_file()]
    failures = []
    for name in ['failure.json', 'training_failed.json']:
        if (run/name).exists():
            failures.append({'file': str(run/name), 'receipt': read_json(run/name)})
    if missing or failures:
        return {'complete': False, 'missing': missing, 'failures': failures}
    study = read_json(run/'study.json')
    checks = {
        'formal_not_smoke': study.get('smoke') is False,
        'coordinator_complete_status': read_json(run/'status.json').get('status') == 'complete',
        'both_train_stages_complete': read_json(run/'training_complete.json').get('status') == 'complete'
                                      and read_json(run/'training_complete.json').get('stages') == 2,
        'all_four_evaluators_exit0': read_json(run/'evaluation_workers_complete.json').get('returncodes') == [0]*4,
    }
    experiment = run.parents[2].parent/'experiments/prototype_sep_v1'
    source = experiment/'a_geometry/src'
    recorded_sources = study.get('source_sha256', {})
    actual_sources = {str(path.relative_to(source)): digest(path) for path in source.rglob('*.py')
                      if '__pycache__' not in path.parts}
    checks['formal_source_frozen'] = bool(recorded_sources) and actual_sources == recorded_sources
    runner = experiment/'run_prototype_sep.py'
    checks['formal_runner_frozen'] = runner.is_file() and digest(runner) == study.get('runner_sha256')
    stage_receipts = {}
    for step in [1, 2]:
        stage = run/f'10-5/step{step}'
        launch, receipt = read_json(stage/'launch.json'), read_json(stage/'training_complete.json')
        checks[f'step{step}_train_exit0'] = launch.get('returncode') == 0 and receipt.get('returncode') == 0
        stage_receipts[str(step)] = {'launch': str(stage/'launch.json'),
                                     'returncode': launch.get('returncode'),
                                     'final_checkpoint_sha256': receipt.get('checkpoint_sha256')}
    for rank in range(4):
        checks[f'evaluator{rank}_record_exit0'] = read_json(run/f'evaluator_rank{rank}.process.json').get('returncode') == 0
    queue = sorted((run/'eval_queue').glob('*.json'))
    checks['evaluation_queue_nonempty'] = bool(queue)
    checks['every_queued_result_exists'] = all((run/'evaluations'/job.stem/'result.json').is_file() for job in queue)
    errors = list((run/'evaluations').glob('*/error_rank*.json'))
    checks['no_sharded_evaluation_error_receipts'] = not errors
    full_shards = {}
    for step in [1, 2]:
        base = run/f'evaluations/step{step}_iter8000'
        files = [base/f'rank{rank}.json' for rank in range(4)]
        checks[f'step{step}_all_four_shards_present'] = all(path.is_file() for path in files)
        if not checks[f'step{step}_all_four_shards_present']:
            continue
        records = [read_json(path) for path in files]
        names = [name for row in records for name in row['images']]
        checks[f'step{step}_full_unique_image_count'] = len(names) == EXPECTED_IMAGES[step] and len(set(names)) == len(names)
        cfg = study['common_config']
        split = Path(cfg['list_folder'])/'incremental_split'/f"val_{cfg['task']}_step_{step+1}.txt"
        expected_names = split.read_text().splitlines() if split.is_file() else []
        checks[f'step{step}_exact_full_validation_names'] = len(expected_names) == EXPECTED_IMAGES[step] and set(names) == set(expected_names)
        checks[f'step{step}_shard_ids_and_count'] = all(row.get('rank') == rank and row.get('shards') == 4 for rank, row in enumerate(records))
        checks[f'step{step}_shard_endpoint_iteration'] = all(row.get('step') == step and row.get('iteration') == FINAL_ITERATION for row in records)
        result = read_json(base/'result.json')
        checks[f'step{step}_shard_checkpoint_identity'] = all(row.get('checkpoint_sha256') == result.get('checkpoint_sha256') for row in records)
        full_shards[str(step)] = {'image_count': len(names), 'unique_image_count': len(set(names)),
                                  'validation_list': str(split), 'shards': [str(path) for path in files]}
    return {'complete': all(checks.values()), 'checks': checks,
            'missing': missing, 'failures': failures,
            'error_receipts': [str(path) for path in errors],
            'stage_receipts': stage_receipts, 'full_endpoint_shards': full_shards,
            'all_queued_jobs': [path.stem for path in queue]}


def histogram(values, classes):
    if not isinstance(values, list) or len(values) != classes or any(len(row) != classes for row in values):
        raise ValueError(f'Expected {classes}x{classes} full GT histogram')
    matrix = [[integer(value, 'histogram count') for value in row] for row in values]
    rows = [sum(row) for row in matrix]
    cols = [sum(matrix[i][j] for i in range(classes)) for j in range(classes)]
    if any(total == 0 for total in rows):
        raise ValueError('Every GT class must have evaluation support')
    iou = [100*matrix[i][i]/(rows[i]+cols[i]-matrix[i][i]) for i in range(classes)]
    return matrix, rows, cols, iou


def endpoint(path, step, require_histogram=True):
    raw = read_json(path)
    count, old = 11+5*step, 5+5*step
    if raw.get('step') != step or raw.get('iteration') != FINAL_ITERATION:
        raise ValueError(f'Not a complete stage{step}/8000 endpoint: {path}')
    if raw.get('images') != EXPECTED_IMAGES[step]:
        raise ValueError(f'Incomplete validation image count in {path}')
    names = CLASSES[:count]
    if not isinstance(raw.get('class_iou'), dict) or set(raw['class_iou']) != set(names):
        raise ValueError(f'Endpoint class IDs/names disagree: {path}')
    iou = [finite(raw['class_iou'][name], 'IoU') for name in names]
    if any(not 0 <= score <= 100 for score in iou):
        raise ValueError('IoU outside [0,100]')
    hist, rows, cols = None, None, None
    if 'histogram' in raw:
        hist, rows, cols, recomputed = histogram(raw['histogram'], count)
        if max(abs(a-b) for a, b in zip(iou, recomputed)) > 1e-6:
            raise ValueError(f'IoU does not reproduce from histogram: {path}')
    elif require_histogram:
        raise ValueError(f'Full endpoint histogram missing: {path}')
    metrics = {'all_miou': sum(iou)/count,
               'previous_foreground_miou': sum(iou[1:old+1])/old,
               'current_foreground_miou': sum(iou[old+1:])/5}
    for name in METRICS:
        if name in raw and abs(finite(raw[name], name)-metrics[name]) > 1e-6:
            raise ValueError(f'Grouping/metric disagreement: {path}, {name}')
    return {'source': str(path), 'step': step, 'iteration': FINAL_ITERATION,
            'images': raw['images'], 'class_count': count,
            'old_foreground_ids': list(range(1, old+1)),
            'new_foreground_ids': list(range(old+1, count)),
            'metrics': metrics, 'class_iou': dict(zip(names, iou)),
            'histogram': hist, 'gt_class_pixels': rows, 'pred_class_pixels': cols,
            'histogram_available': hist is not None,
            'checkpoint_sha256': raw.get('checkpoint_sha256')}


def historical(root, method, step):
    for tree in ['runs/kd_pixel_v1', 'runs/kd_pixel_v2',
                 'runs/kd_parallel_v1/formal/b_relational']:
        path = root/tree/f'evaluations/baseline_{method}_step{step}/result.json'
        if path.is_file():
            return endpoint(path, step)
    path = root/f'runs/fixed_baseline_v1/{method}/10-5/step{step}/metrics.jsonl'
    if not path.is_file():
        raise FileNotFoundError(f'Existing historical endpoint missing: {path}')
    rows = [row for row in read_rows(path) if row.get('iteration') == FINAL_ITERATION and row.get('step') == step]
    if not rows:
        raise ValueError(f'Historical final metrics missing: {path}')
    raw, count, old = rows[-1], 11+5*step, 5+5*step
    names = CLASSES[:count]
    if set(raw['class_iou']) != set(names):
        raise ValueError('Historical class names disagree')
    iou = [finite(raw['class_iou'][name], 'historical IoU') for name in names]
    if any(not 0 <= score <= 100 for score in iou):
        raise ValueError('Historical IoU outside [0,100]')
    return {'source': str(path), 'step': step, 'iteration': FINAL_ITERATION,
            'images': None, 'class_count': count,
            'old_foreground_ids': list(range(1, old+1)), 'new_foreground_ids': list(range(old+1, count)),
            'metrics': {'all_miou': sum(iou)/count,
                        'previous_foreground_miou': sum(iou[1:old+1])/old,
                        'current_foreground_miou': sum(iou[old+1:])/5},
            'class_iou': dict(zip(names, iou)), 'histogram': None,
            'gt_class_pixels': None, 'pred_class_pixels': None,
            'histogram_available': False, 'checkpoint_sha256': None,
            'limitation': 'Archived endpoint class IoU only; image completeness is not re-audited here and direction ratios cannot be reconstructed from IoU.'}


def error_direction(result, source, target):
    numerator = result['histogram'][source][target]
    denominator = result['gt_class_pixels'][source]
    return {'source_id': source, 'target_id': target,
            'source_class': CLASSES[source], 'target_class': CLASSES[target],
            'error_pixels': numerator, 'gt_source_pixels': denominator,
            'error_percent': 100*numerator/denominator}


def direction_delta(candidate, reference, i, j):
    current, original = error_direction(candidate, i, j), error_direction(reference, i, j)
    return {**current, 'candidate_error_percent': current['error_percent'],
            'reference_error_percent': original['error_percent'],
            'delta_error_pp': current['error_percent']-original['error_percent'],
            'reference_error_pixels': original['error_pixels']}


def aggregate_errors(result):
    old, new = result['old_foreground_ids'], result['new_foreground_ids']
    groups = {'old_to_new': (old, new), 'new_to_old': (new, old),
              'old_to_background': (old, [0]), 'new_to_background': (new, [0]),
              'new_to_other_new': (new, new)}
    return {name: {'error_pixels': sum(result['histogram'][i][j] for i in sources for j in targets if i != j),
                   'gt_source_pixels': sum(result['gt_class_pixels'][i] for i in sources),
                   'error_percent': 100*sum(result['histogram'][i][j] for i in sources for j in targets if i != j)/sum(result['gt_class_pixels'][i] for i in sources)}
            for name, (sources, targets) in groups.items()}


def comparison(candidate, reference, selected_pairs):
    class_rows = [{'class_id': i, 'class_name': name,
                   'candidate_iou': score, 'reference_iou': reference['class_iou'][name],
                   'delta_iou_pp': score-reference['class_iou'][name]}
                  for i, (name, score) in enumerate(candidate['class_iou'].items())]
    result = {'metric_delta_pp': {key: candidate['metrics'][key]-reference['metrics'][key] for key in METRICS},
              'per_class_delta': class_rows,
              'classes_improved': sorted([row for row in class_rows if row['delta_iou_pp'] > EPS], key=lambda row: -row['delta_iou_pp']),
              'classes_worsened': sorted([row for row in class_rows if row['delta_iou_pp'] < -EPS], key=lambda row: row['delta_iou_pp']),
              'direction_ratios_available': candidate['histogram_available'] and reference['histogram_available']}
    if not result['direction_ratios_available']:
        result['direction_limitation'] = 'Reference class IoU does not identify off-diagonal confusion counts.'
        return result
    if candidate['gt_class_pixels'] != reference['gt_class_pixels']:
        raise ValueError('Evaluation GT row denominators differ; direction comparison invalid')
    if candidate['images'] != reference['images']:
        raise ValueError('Evaluation image counts differ')
    count = candidate['class_count']
    directions = [direction_delta(candidate, reference, i, j)
                  for i in range(count) for j in range(count) if i != j]
    result['all_direction_deltas'] = directions
    result['directions_improved'] = sorted([row for row in directions if row['delta_error_pp'] < -EPS], key=lambda row: row['delta_error_pp'])
    result['directions_worsened'] = sorted([row for row in directions if row['delta_error_pp'] > EPS], key=lambda row: -row['delta_error_pp'])
    result['selected_pair_bidirectional_deltas'] = [
        {'class_ids': [i, j], 'class_names': [CLASSES[i], CLASSES[j]],
         'pair_type': 'old_new' if i in candidate['old_foreground_ids'] else 'new_new',
         'forward': direction_delta(candidate, reference, i, j),
         'reverse': direction_delta(candidate, reference, j, i),
         'interpretation': 'Each direction uses its own full GT source-row denominator; reduction in one direction does not establish joint separation or overall IoU gain.'}
        for i, j in selected_pairs]
    before, after = aggregate_errors(reference), aggregate_errors(candidate)
    result['aggregate_group_errors'] = {name: {'candidate': after[name], 'reference': before[name],
                                              'delta_error_pp': after[name]['error_percent']-before[name]['error_percent']}
                                       for name in after}
    return result


def distribution(values):
    if not values:
        return {'count': 0}
    ordered = sorted(values)
    return {'count': len(values), 'min': ordered[0], 'mean': statistics.mean(values),
            'median': statistics.median(values), 'max': ordered[-1]}


def mechanism(stage, step, expected_weight):
    path = stage/'geometry_metrics.jsonl'
    if not path.is_file():
        raise FileNotFoundError(path)
    rows, old, count = read_rows(path), 5+5*step, 11+5*step
    if not rows or rows[-1].get('iteration') != FINAL_ITERATION:
        raise ValueError('Geometry log must include the final iteration8000 snapshot')
    iterations = [integer(row['iteration'], 'logged iteration') for row in rows]
    if len(set(iterations)) != len(iterations) or iterations != sorted(iterations):
        raise ValueError('Geometry snapshots duplicated or out of order')
    pair_frequency, active_frequency, direction_frequency = Counter(), Counter(), Counter()
    class_frequency, active_endpoint_frequency = Counter(), Counter()
    previous_targets, target_changes = None, 0
    snapshot_rows, all_pairs = [], set()
    detach_checks = {'old_new_all_old_endpoints_detached': True,
                     'new_new_all_both_endpoints_differentiable': True,
                     'no_old_old_or_background_pairs': True}
    for row in rows:
        if row.get('step') != step or row.get('old_classes') != old or row.get('total_classes_including_background') != count:
            raise ValueError('Geometry log stage/class grouping disagrees')
        pair_count = integer(row['pair_count'], 'pair count')
        active_count = integer(row['active_pair_count'], 'active pair count')
        entries = row['pairs']
        if pair_count != len(entries) or active_count != sum(entry['active'] is True for entry in entries):
            raise ValueError('Geometry pair/active summary disagrees with entries')
        targets = row['selector']['targets']
        if len(targets) != count or any(isinstance(j, bool) or not isinstance(j, int) or not -1 <= j < count for j in targets):
            raise ValueError('Geometry selector targets invalid')
        if targets[0] != -1 or any(j >= 0 and (i == j or j == 0 or (i <= old and j <= old)) for i, j in enumerate(targets)):
            raise ValueError('Geometry selector includes disallowed background/self/oldold direction')
        if previous_targets is not None and targets != previous_targets:
            target_changes += 1
        previous_targets = targets
        if not isinstance(row['active'], bool):
            raise ValueError('Loss activation flag must be bool')
        ramp, weight = finite(row['ramp'], 'ramp'), finite(row['weight'], 'SEP weight')
        raw_loss = finite(row['prototype_sep'], 'geometry loss')
        if not 0 <= ramp <= 1 or abs(weight-expected_weight) > EPS or raw_loss < -EPS:
            raise ValueError('Invalid geometry ramp, coefficient or loss')
        seen, entry_losses, directions = set(), [], []
        for entry in entries:
            i, j = entry['class_ids']
            if not 0 < i < j < count or (i <= old and j <= old) or (i, j) in seen:
                raise ValueError('Duplicate or disallowed geometric pair')
            seen.add((i, j))
            all_pairs.add((i, j))
            cosine = finite(entry['cosine_similarity'], 'pair cosine')
            margin = finite(entry['margin'], 'pair margin')
            loss = finite(entry['pair_loss'], 'pair loss')
            if not -1-EPS <= cosine <= 1+EPS or abs(margin) > EPS:
                raise ValueError('Cosine/margin does not match frozen margin0 objective')
            if abs(loss-max(0., cosine-margin)**2) > 1e-6 or entry['active'] != (cosine > margin):
                raise ValueError('Pair hinge loss/active flag inconsistent')
            expected_type, endpoints = ('old_new', [j]) if i <= old else ('new_new', [i, j])
            expected_detached = i if i <= old else None
            if entry['pair_type'] != expected_type or entry['gradient_class_ids'] != endpoints or entry['old_endpoint_detached'] != expected_detached:
                raise ValueError('SEP detached endpoint semantics disagree')
            entry_losses.append(loss)
            pair_frequency[i, j] += 1
            active_frequency[i, j] += int(entry['active'])
            class_frequency.update([i, j])
            if entry['active'] and row['active'] and ramp > 0:
                active_endpoint_frequency.update(endpoints)
            for a, b in entry['selected_directions']:
                if (a, b) not in [(i, j), (j, i)] or targets[a] != b:
                    raise ValueError('Pair source direction disagrees with selector')
                directions.append((a, b))
                direction_frequency[a, b] += 1
        if seen != {tuple(sorted((i, j))) for i, j in enumerate(targets) if j > 0}:
            raise ValueError('Selected unique pairs disagree with selector targets')
        if len(directions) != row['selected_valid_directions'] or len(directions)-pair_count != row['duplicate_reverse_directions_removed']:
            raise ValueError('Geometry direction/deduplication totals disagree')
        if row['old_new_pair_count'] != sum(i <= old for i, j in seen) or row['new_new_pair_count'] != sum(i > old for i, j in seen):
            raise ValueError('Geometry pair-type counts disagree')
        recomputed_loss = statistics.mean(entry_losses) if entry_losses else 0.
        if abs(raw_loss-recomputed_loss) > 1e-6:
            raise ValueError('Geometry objective is not mean over all eligible unique pairs')
        effective_lambda = weight*ramp if row['active'] else 0.
        snapshot_rows.append({'iteration': row['iteration'], 'base_loss_active': row['active'],
                              'selector_ramp': ramp, 'effective_lambda': effective_lambda,
                              'pair_count': pair_count, 'active_pair_count': active_count,
                              'old_new_pair_count': row['old_new_pair_count'],
                              'new_new_pair_count': row['new_new_pair_count'],
                              'raw_sep_loss': raw_loss,
                              'weighted_sep_loss': effective_lambda*raw_loss,
                              'targets': targets,
                              'selected_rates_diagnostic_only': row['selector'].get('selected_rates')})
    valid = [row for row in snapshot_rows if row['effective_lambda'] > 0]
    counts = sum(row['pair_count'] for row in valid)
    active_counts = sum(row['active_pair_count'] for row in valid)
    frequencies = [{'class_ids': [i, j], 'class_names': [CLASSES[i], CLASSES[j]],
                    'pair_type': 'old_new' if i <= old else 'new_new',
                    'logged_snapshots_selected': pair_frequency[i, j],
                    'logged_snapshots_active': active_frequency[i, j],
                    'snapshot_selection_fraction': pair_frequency[i, j]/len(rows)}
                   for i, j in sorted(all_pairs)]
    periods = {}
    for label, predicate in [('base_loss_inactive', lambda row: not row['base_loss_active']),
                             ('loss_active_selector_ramping', lambda row: row['base_loss_active'] and row['selector_ramp'] < 1),
                             ('loss_active_full_ramp', lambda row: row['base_loss_active'] and row['selector_ramp'] == 1)]:
        selected = [row for row in snapshot_rows if predicate(row)]
        periods[label] = {'logged_snapshots': len(selected),
                          'first_iteration': selected[0]['iteration'] if selected else None,
                          'last_iteration': selected[-1]['iteration'] if selected else None,
                          'pair_count': distribution([row['pair_count'] for row in selected]),
                          'weighted_sep_loss': distribution([row['weighted_sep_loss'] for row in selected])}
    return {'source': str(path), 'step': step, 'logged_snapshots': len(rows),
            'first_logged_iteration': iterations[0], 'last_logged_iteration': iterations[-1],
            'snapshots_with_effective_sep_weight': len(valid),
            'snapshots_with_effective_weight_and_pairs': sum(row['pair_count'] > 0 for row in valid),
            'snapshots_with_effective_weight_and_active_pairs': sum(row['active_pair_count'] > 0 for row in valid),
            'sampled_target_change_events': target_changes,
            'unique_pairs_observed_in_logs': len(all_pairs), 'pair_frequencies': frequencies,
            'direction_frequencies': [{'source_id': i, 'target_id': j, 'logged_snapshots': frequency}
                                      for (i, j), frequency in sorted(direction_frequency.items())],
            'class_pair_occurrence_counts': {CLASSES[i]: class_frequency[i] for i in range(count)},
            'active_allowed_gradient_endpoint_occurrence_counts': {CLASSES[i]: active_endpoint_frequency[i] for i in range(count)},
            'effective_weight_snapshot_pair_occurrences': counts,
            'effective_weight_snapshot_active_pair_occurrences': active_counts,
            'effective_weight_snapshot_active_pair_fraction': active_counts/counts if counts else None,
            'pair_count_distribution_when_enabled': distribution([row['pair_count'] for row in valid]),
            'raw_sep_loss_when_enabled': distribution([row['raw_sep_loss'] for row in valid]),
            'weighted_sep_loss_when_enabled': distribution([row['weighted_sep_loss'] for row in valid]),
            'effective_lambda_when_enabled': distribution([row['effective_lambda'] for row in valid]),
            'detach_semantics_checks': detach_checks, 'activation_periods': periods,
            'snapshots': snapshot_rows,
            'final_logged_pairs': rows[-1]['pairs'], 'final_logged_selector': rows[-1]['selector'],
            'final_snapshot_pairs': sorted(tuple(entry['class_ids']) for entry in rows[-1]['pairs']),
            'all_logged_selected_pairs': sorted(all_pairs),
            'scope': 'Recorded rank0 snapshots of globally synchronized prototype geometry and selector. These are repeated logged training moments, not independent samples, population pixel coverage, measured parameter-gradient norms, or GT confusion accuracy.',
            'activation_caveat': 'Observer updates and selected pairs during stage2 warmup do not enable the loss. Weighted SEP is zero until base_loss_active, and additionally follows the selector ramp.',
            'rate_caveat': 'Selected rates rank pseudo/CAM-supported relations; they are not precise weights and are not comparable to calibrated GT full-row error probabilities.'}


def safe_output(path):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError('Report output must be absolute')
    target, allowed = path.resolve(strict=False), AREA.resolve(strict=True)
    if target == allowed or allowed not in target.parents:
        raise ValueError('Report output leaves the authorized A directory')
    if path.exists() and (path.is_symlink() or path.stat().st_nlink != 1):
        raise ValueError('Refuse symlinked or hardlinked report file')
    return target


def write_report(path, value):
    target = safe_output(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target = safe_output(target)
    temp = safe_output(target.with_name(target.name+'.'+uuid.uuid4().hex+'.tmp'))
    with temp.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
    os.replace(temp, target)


def analyze(root):
    run = root/'runs/prototype_sep_v1/formal/a_geometry'
    done = completion(run)
    if not done['complete']:
        raise RuntimeError('Final analysis withheld until train/eval completion and exit0: '+json.dumps(done))
    study = read_json(run/'study.json')
    cfg = study['common_config']
    if cfg.get('task') != '10-5' or cfg.get('seed') != 0 or study.get('global_batch') != 8 or study.get('batch_per_gpu') != 1:
        raise ValueError('Unexpected formal task/seed/global batch')
    expected_weight = finite(cfg['w_geometry_sep'], 'formal SEP weight')
    if abs(expected_weight-.1) > EPS or abs(finite(cfg['proto_margin'], 'formal margin')) > EPS:
        raise ValueError('Unexpected frozen geometry coefficient/margin')
    stages = {}
    for step in [1, 2]:
        candidate = endpoint(run/f'evaluations/step{step}_iter8000/result.json', step)
        # Verify the additive histograms underlying the merged endpoint.
        parts = [read_json(run/f'evaluations/step{step}_iter8000/rank{rank}.json') for rank in range(4)]
        merged = [[sum(part['histogram'][i][j] for part in parts) for j in range(candidate['class_count'])]
                  for i in range(candidate['class_count'])]
        if merged != candidate['histogram']:
            raise ValueError('Merged endpoint histogram disagrees with full shards')
        optimized = endpoint(root/f'runs/kd_parallel_v1/formal/b_relational/evaluations/step{step}_iter8000/result.json', step)
        refs = {'optimized_KD': optimized,
                **{name: historical(root, name, step) for name in ['base', 'sep', 'kd', 'kd_sep']}}
        signals = mechanism(run/f'10-5/step{step}', step, expected_weight)
        comparisons = {name: comparison(candidate, ref, signals['all_logged_selected_pairs']) for name, ref in refs.items()}
        stages[str(step)] = {'endpoint': candidate, 'references': refs, 'comparisons': comparisons,
                             'mechanism': signals,
                             'final_snapshot_selected_pair_bidirectional_deltas_vs_optimizedKD':
                             comparison(candidate, optimized, signals['final_snapshot_pairs']).get('selected_pair_bidirectional_deltas')}
    final = stages['2']['endpoint']
    reference = stages['2']['references']['optimized_KD']
    delta = final['metrics']['all_miou']-reference['metrics']['all_miou']
    wins = delta > EPS
    selected_checkpoint = (run if wins else root/'runs/kd_parallel_v1/formal/b_relational')/'10-5/step2/checkpoints/model_final.pth'
    recommendation = {'method': 'confusion_guided_prototype_SEP_plus_optimized_KD' if wins else 'existing_optimized_KD',
                      'selected_checkpoint': str(selected_checkpoint),
                      'candidate_stage2_final_all_miou': final['metrics']['all_miou'],
                      'reference_stage2_final_all_miou': reference['metrics']['all_miou'],
                      'candidate_delta_pp': delta, 'candidate_exceeds_required_endpoint': wins,
                      'criterion': 'Complete stage2/8000 all-class mIoU exceeds existing optimizedKD69.1492921947; stage1 improvements or isolated class-pair changes cannot decide this goal.',
                      'status': 'observed_development_endpoint_gain' if wins else 'candidate_did_not_improve_required_endpoint',
                      'evidence_limit': 'One fixed seed and one adaptively developed task; no significance, seed stability or independent generalization claim.'}
    return {'utc': datetime.now(timezone.utc).isoformat(), 'status': 'complete_endpoint_analysis',
            'run': str(run), 'study': study, 'completion': done,
            'scope': {'task': 'VOC10-5', 'seed': 0, 'iteration': FINAL_ITERATION,
                      'stages': {'1': {'images':1240,'previous_foreground':10,'current_foreground':5},
                                 '2': {'images':1449,'previous_foreground':15,'current_foreground':5}},
                      'all_miou_includes_background': True,
                      'units': 'IoU/error ratios are percentages; deltas are percentage points.',
                      'gt_direction_definition': 'GT class i -> main-head prediction j divided by all GT pixels in row i, including correct predictions, background and every competitor.',
                      'training_selection_definition': 'Past broad EMA of pseudo/PAR anchor -> main prediction, plus trusted CAM pair support and row freshness; geometry candidates involve a current-new class.',
                      'rate_warning': 'GT and online selector use different anchor spaces and sample measures. GT ratios are diagnostic only and neither are calibrated learning strengths.',
                      'gt_usage': 'Existing full validation histogram only; no GT feeds prototype SEP, online selection or learning weights.'},
            'mechanism_description': {
                'objective': 'lambda0.1*ramp*mean_over_all_eligible_unique_pairs(relu(cosine-margin0)^2); inactive pairs remain in the denominator.',
                'old_new': 'Detach current student old prototype only inside SEP; update new endpoint. Old prototypes remain trainable through existing losses.',
                'new_new': 'Both current-new endpoints differentiate; reverse selected directions deduplicate to one symmetric geometric term.',
                'old_old': 'Skip in selection and SEP to leave preservation to unchanged optimized KD and existing losses.',
                'direct_gradient_scope': 'Prototype-only SEP directly updates prototype parameters only; it has no direct encoder/conv6/conv7/main-head gradient.',
                'indirect_feature_effect': 'Changed prototypes can alter later prototype-segmentation feature gradients; the final main-head improvement is empirical, not guaranteed by geometric separation.',
                'KD': 'Existing optimized conditional old-foreground KD is retained; no new binary pairKD substitution or pixel-margin SEP.'},
            'stages': stages, 'recommendation': recommendation,
            'limitations': ['One fixedseed0 task and adaptive development validation; not a new independent test set.',
                            'Candidate8GPUs x batch1 versus archived reference4GPUs x batch2 preserves globalbatch8 but restarts sampler/augmentations; deltas do not strictly isolate SEP as one causal factor.',
                            'Stage1 reuses full model/optimizer at iteration2000. Stage2 inherits the candidate stage1 teacher, so its endpoint is the whole trajectory rather than an isolated stage2 intervention.',
                            'Prototype cosine is symmetric. Directed confusion supplies priority; reducing one GT direction may increase its reverse or false positives elsewhere.',
                            'Logged pair and active endpoint counts are sampled moments and permitted gradient paths, not measured optimizer-gradient magnitudes or population coverage.',
                            'Some historical methods retain only class IoU; no unavailable GT direction statistics are inferred or fabricated.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-only', action='store_true', help='Check successful completion only; no output write')
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if args.check_only:
        state = completion(root/'runs/prototype_sep_v1/formal/a_geometry')
        print(json.dumps(state, allow_nan=False))
        sys.exit(0 if state['complete'] else 2)
    report = analyze(root)
    output = args.output or root/'runs/prototype_sep_v1/result_analysis.json'
    write_report(output, report)
    print(json.dumps({'report': str(output), 'status': report['status'],
                      'stage1': report['stages']['1']['endpoint']['metrics'],
                      'stage2': report['stages']['2']['endpoint']['metrics'],
                      'recommendation': report['recommendation']}, allow_nan=False))


if __name__ == '__main__':
    main()
