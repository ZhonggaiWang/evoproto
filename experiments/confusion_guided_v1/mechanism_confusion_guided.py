"""Summarize observed confusion-guided mechanisms; no GPU or training changes.

Reads existing formal training samples only. Selection and margin histories
are sparse batch observations, not population estimates or accuracy evidence.
The report never treats a selected confusion rate as a precise loss weight.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import argparse
import json
import math
import os
import statistics


DEFAULT_ROOT = '/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto/runs/confusion_guided_v1'
ALLOWED_ROOT = Path('/ML-vePFS/infra_rd/kun/others/wzg')


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def read_samples(path):
    if not path.exists():
        return [], {'path': str(path), 'exists': False, 'samples': 0}
    text = path.read_text()
    lines = text.splitlines()
    rows = []
    partial_tail = False
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines)-1 and not text.endswith('\n'):
                partial_tail = True
                continue
            raise ValueError(f'Invalid complete JSON line {i+1} in {path}')
        if not isinstance(row, dict) or not isinstance(row.get('iteration'), int):
            raise ValueError(f'Missing integer training iteration in {path}:{i+1}')
        rows.append(row)
    deduplicated = {row['iteration']: row for row in rows}
    result = [deduplicated[key] for key in sorted(deduplicated)]
    return result, {'path': str(path), 'exists': True, 'samples': len(result),
                    'duplicate_iterations_replaced_by_latest': len(rows)-len(result),
                    'incomplete_trailing_line_ignored': partial_tail}


def ratio(numerator, denominator):
    return numerator/denominator if denominator > 0 else None


def distribution(values):
    finite = sorted(float(x) for x in values
                    if x is not None and isinstance(x, (int, float)) and math.isfinite(x))
    if not finite:
        return {'samples': 0, 'mean': None, 'min': None, 'p25': None,
                'median': None, 'p75': None, 'max': None}
    def quantile(q):
        index = q*(len(finite)-1)
        lower = math.floor(index)
        upper = math.ceil(index)
        return finite[lower]*(upper-index)+finite[upper]*(index-lower) if upper != lower else finite[lower]
    return {'samples': len(finite), 'mean': statistics.fmean(finite),
            'min': finite[0], 'p25': quantile(.25), 'median': quantile(.5),
            'p75': quantile(.75), 'max': finite[-1]}


def selection_summary(samples):
    directions = Counter()
    row_known = Counter()
    changes = Counter()
    change_types = Counter()
    previous = None
    row_denominators = Counter()
    for sample in samples:
        targets = sample.get('selector', {}).get('targets', [])
        for anchor, target in enumerate(targets):
            if anchor == 0:
                continue
            row_denominators[anchor] += 1
            if target > 0 and target != anchor:
                directions[(anchor, target)] += 1
                row_known[anchor] += 1
            if previous is not None and anchor < len(previous) and previous[anchor] != target:
                changes[anchor] += 1
                category = ('unknown_to_known' if previous[anchor] < 0 and target > 0 else
                            'known_to_unknown' if previous[anchor] > 0 and target < 0 else
                            'known_to_other_known')
                change_types[category] += 1
        previous = targets
    return {
        'logged_samples': len(samples),
        'direction_frequency': [
            {'anchor': i, 'competitor': j, 'selected_samples': n,
             'fraction_of_logged_row_samples': ratio(n, row_denominators[i]),
             'fraction_of_known_row_samples': ratio(n, row_known[i])}
            for (i, j), n in sorted(directions.items(), key=lambda item: (-item[1], item[0]))],
        'row_known_fraction': {str(i): ratio(row_known[i], n)
                               for i, n in sorted(row_denominators.items())},
        'sampled_target_change_count': sum(changes.values()),
        'sampled_target_changes_by_anchor': {str(i): n for i, n in sorted(changes.items())},
        'sampled_target_change_types': dict(change_types),
        'interpretation': 'Adjacent logged snapshots only; unobserved changes between samples can be missed. Frequency measures selected snapshots, not loss or gradient mass.'}


def class_coverage(samples, key, classes):
    mass = [0.0]*classes
    present = [0]*classes
    for sample in samples:
        for i, amount in enumerate(sample.get(key, [])):
            if i < classes:
                mass[i] += amount
                present[i] += int(amount > 0)
    return {'active_class_ids': [i for i in range(1, classes) if present[i] > 0],
            'sampled_eligible_pixel_occurrences_by_class': mass,
            'positive_logged_batches_by_class': present,
            'meaning': 'Region eligibility, not per-class gradient norms; batches are sparse observations and repeated pixel/image exposures.'}


def sep_coverage(samples, kd_by_iteration):
    totals = Counter()
    invalid = []
    per_batch = {key: [] for key in ['old_candidate', 'new_candidate', 'teacher_veto',
                                    'old_eligible', 'new_eligible', 'all_eligible',
                                    'eligible_fraction_of_valid', 'veto_fraction_of_old_candidate']}
    for sample in samples:
        old = int(sample.get('sep_old_pixels', 0))
        new = int(sample.get('sep_new_pixels', 0))
        veto = int(sample.get('sep_teacher_veto_pixels', 0))
        candidate = int(sample.get('sep_candidate_pixels', 0))
        eligible = int(sample.get('sep_nonzero_pixels', 0))
        old_candidate, new_candidate = old+veto, new
        if candidate != old_candidate+new_candidate or eligible != old+new:
            invalid.append(sample['iteration'])
        values = {'old_candidate': old_candidate, 'new_candidate': new_candidate,
                  'teacher_veto': veto, 'old_eligible': old, 'new_eligible': new,
                  'all_candidate': candidate, 'all_eligible': eligible}
        totals.update(values)
        kd = kd_by_iteration.get(sample['iteration'])
        valid = kd.get('valid_pixels') if kd else None
        if valid is not None:
            totals['valid_pixels_in_joined_batches'] += int(valid)
            totals['eligible_in_joined_batches'] += eligible
        for key in per_batch:
            if key in values:
                per_batch[key].append(values[key])
        per_batch['eligible_fraction_of_valid'].append(ratio(eligible, valid) if valid is not None else None)
        per_batch['veto_fraction_of_old_candidate'].append(ratio(veto, old_candidate))
    return {'logged_batches': len(samples), 'totals_of_sampled_batch_pixel_occurrences': dict(totals),
            'pooled_old_teacher_veto_fraction': ratio(totals['teacher_veto'], totals['old_candidate']),
            'pooled_old_eligible_fraction_of_old_candidate': ratio(totals['old_eligible'], totals['old_candidate']),
            'pooled_new_eligible_fraction_of_new_candidate': ratio(totals['new_eligible'], totals['new_candidate']),
            'pooled_eligible_fraction_of_valid_joined_pixels': ratio(
                totals['eligible_in_joined_batches'], totals['valid_pixels_in_joined_batches']),
            'batch_distributions': {key: distribution(values) for key, values in per_batch.items()},
            'candidate_reconstruction_inconsistent_iterations': invalid,
            'reconstruction': 'All teacher vetoes are old anchors: old candidate=old eligible+teacher veto; new candidate=new eligible.',
            'coverage_caveat': 'These are selected, trusted anchor regions, not all old/new GT pixels. No GT denominators or precision are inferred.'}


def metric_trends(samples):
    if not samples:
        return {'logged_samples': 0, 'early_window': None, 'late_window': None}
    window = min(10, max(1, math.ceil(len(samples)*.2)))
    early, late = samples[:window], samples[-window:]
    names = ['pair_sep', 'pair_sep_main', 'pair_sep_proto']
    names += [f'sep_{age}_{branch}_{suffix}' for age in ['old', 'new']
              for branch in ['main', 'proto']
              for suffix in ['margin', 'error_rate', 'margin_violation_rate']]
    metrics = {}
    for name in names:
        # Empty old/new support has logged zero statistics; exclude these zeros.
        age = 'old' if name.startswith('sep_old_') else 'new' if name.startswith('sep_new_') else None
        def values(rows):
            return [row[name] for row in rows if name in row and
                    (not age or row.get(f'sep_{age}_pixels', 0) > 0)]
        a, b = distribution(values(early)), distribution(values(late))
        metrics[name] = {'all_observed': distribution(values(samples)), 'early': a, 'late': b,
                         'late_minus_early_mean': b['mean']-a['mean']
                         if b['mean'] is not None and a['mean'] is not None else None}
    return {'logged_samples': len(samples), 'window_size_batches': window,
            'early_iterations': [early[0]['iteration'], early[-1]['iteration']],
            'late_iterations': [late[0]['iteration'], late[-1]['iteration']], 'metrics': metrics,
            'interpretation': 'Descriptive sparse-batch trends; anchor composition, selected pairs and confidence change. No causal improvement or GT accuracy follows from lower training loss/violations.'}


def kd_summary(samples, classes, pair_mode):
    enabled = [row for row in samples if 'kd_pair_pixels' in row]
    fractions = [ratio(row['kd_pair_pixels'], row.get('kd_nonzero_pixels', 0)) for row in enabled]
    total_pair = sum(row.get('kd_pair_pixels', 0) for row in samples)
    total_kd = sum(row.get('kd_nonzero_pixels', 0) for row in samples)
    missing_pair = [row['iteration'] for row in samples if 'kd_pair_pixels' not in row]
    return {'logged_batches': len(samples), 'pair_mode': pair_mode,
            'pair_branch_logged_batches': len(enabled),
            'positive_pair_batches': sum(row['kd_pair_pixels'] > 0 for row in enabled),
            'pair_eligible_pixels_distribution': distribution(row.get('kd_pair_eligible_pixels') for row in enabled),
            'pair_pixels_distribution': distribution(row['kd_pair_pixels'] for row in enabled),
            'base_kd_nonzero_pixels_distribution': distribution(row.get('kd_nonzero_pixels') for row in samples),
            'pair_fraction_of_nonzero_kd_distribution': distribution(fractions),
            'pooled_pair_fraction_of_nonzero_kd': ratio(total_pair, total_kd) if enabled else None,
            'sampled_pair_pixel_occurrences': total_pair,
            'sampled_nonzero_kd_pixel_occurrences': total_kd,
            'pair_coverage': class_coverage(enabled, 'kd_pair_class_pixels', classes),
            'base_kd_region_coverage': class_coverage(samples, 'class_eligible_pixels', classes),
            'teacher_pair_confidence_distribution_on_positive_batches': distribution(
                row.get('kd_pair_mean_teacher_confidence') for row in enabled if row['kd_pair_pixels'] > 0),
            'teacher_pair_margin_distribution_on_positive_batches': distribution(
                row.get('kd_pair_mean_teacher_margin') for row in enabled if row['kd_pair_pixels'] > 0),
            'iterations_without_pair_branch_fields': missing_pair,
            'missing_pair_field_meaning': 'Arm A has no pair KD; in arm B, zero blend disables pair calculation and leaves its fields absent. Missing fields are not fabricated observations.',
            'interpretation': 'Positive region coverage indicates eligible replay pixels, not measured gradient magnitude or GT correctness.'}


def outer_objectives(pair_samples, kd_by_iteration, config):
    rows = []
    for pair in pair_samples:
        kd = kd_by_iteration.get(pair['iteration'])
        ramp = pair.get('ramp', 0.)
        coefficient = pair.get('weight', config.get('w_pair_sep', .1))*ramp
        injected_sep = coefficient*pair.get('pair_sep', 0.)
        injected_kd = kd.get('weight', config.get('w_pixel_kd', .1))*kd.get('pixel_kd', 0.) if kd else None
        rows.append({'iteration': pair['iteration'], 'sep_ramp': ramp,
                     'sep_outer_coefficient': coefficient,
                     'sep_weighted_objective': injected_sep, 'kd_weighted_objective': injected_kd,
                     'sep_to_kd_weighted_objective_ratio': ratio(injected_sep, injected_kd)
                     if injected_kd is not None else None,
                     'pair_kd_blend': kd.get('kd_pair_blend') if kd else None})
    return {'active_logged_batches': len(rows), 'latest_active_sample': rows[-1] if rows else None,
            'distributions': {key: distribution(row.get(key) for row in rows)
                              for key in ['sep_ramp', 'sep_outer_coefficient', 'sep_weighted_objective',
                                          'kd_weighted_objective', 'sep_to_kd_weighted_objective_ratio',
                                          'pair_kd_blend']},
            'definition': 'Injected SEP=logged weight*ramp*pair_sep; injected KD=logged weight*pixel_kd. main/prototype SEP are averaged inside pair_sep.',
            'limitation': 'Full training loss is absent from these logs; SEP share of total training loss and its gradient share cannot be computed.'}


def guard_summary(samples):
    """Report sampled weight attenuation and native-logit local push only."""
    guarded = [row for row in samples if row.get('new_cam_guard') is True]
    if not guarded:
        return {'available': False, 'guard_logged_batches': 0,
                'interpretation': 'No enabled new-CAM guard observations in this sample set; missing fields are not zero measurements.'}
    names = ['guard_old_weight_sum_before', 'guard_old_weight_sum_after',
             'guard_old_removed_weight_sum', 'guard_old_mean_new_cam', 'guard_old_mean_factor',
             'guard_new_weight_sum_before', 'guard_new_weight_sum_after',
             'guard_removed_main_push', 'guard_removed_proto_push',
             'guard_total_before_main_push', 'guard_total_after_main_push',
             'guard_total_before_proto_push', 'guard_total_after_proto_push',
             'guard_effective_pixels', 'guard_effective_old_pixels', 'guard_effective_new_pixels',
             'guard_attenuated_old_pixels', 'guard_zero_weight_old_pixels']
    totals = {name: sum(row[name] for row in guarded if name in row)
              for name in names if 'mean' not in name and any(name in row for row in guarded)}
    missing = {name: [row['iteration'] for row in guarded if name not in row]
               for name in names if any(name not in row for row in guarded)}
    per_batch_old_removed = []
    per_batch_main_removed, per_batch_proto_removed = [], []
    inconsistent = []
    for row in guarded:
        before, after = row.get('guard_old_weight_sum_before'), row.get('guard_old_weight_sum_after')
        removed = row.get('guard_old_removed_weight_sum')
        new_before, new_after = row.get('guard_new_weight_sum_before'), row.get('guard_new_weight_sum_after')
        if before is not None and after is not None:
            tolerance = 1e-5*max(1, abs(before))
            if after < -tolerance or after > before+tolerance:
                inconsistent.append({'iteration': row['iteration'], 'invariant': 'old weights cannot increase or become negative'})
            if removed is not None and abs((before-after)-removed) > tolerance:
                inconsistent.append({'iteration': row['iteration'], 'invariant': 'removed old mass equals before minus after'})
        if new_before is not None and new_after is not None and abs(new_before-new_after) > 1e-6*max(1, abs(new_before)):
            inconsistent.append({'iteration': row['iteration'], 'invariant': 'new-anchor evidence weights unchanged'})
        if row.get('guard_effective_old_pixels', 0) > row.get('sep_old_pixels', 0):
            inconsistent.append({'iteration': row['iteration'], 'invariant': 'effective old pixels cannot exceed original eligibility'})
        if row.get('guard_effective_new_pixels', 0) != row.get('sep_new_pixels', 0):
            inconsistent.append({'iteration': row['iteration'], 'invariant': 'new-anchor effective pixels equal unchanged eligibility'})
        per_batch_old_removed.append(ratio(removed, before) if removed is not None and before is not None else None)
        for branch, destination in [('main', per_batch_main_removed), ('proto', per_batch_proto_removed)]:
            numerator = row.get(f'guard_removed_{branch}_push')
            denominator = row.get(f'guard_total_before_{branch}_push')
            destination.append(ratio(numerator, denominator) if numerator is not None and denominator is not None else None)
    distributions = {}
    for name in names:
        values = [row.get(name) for row in guarded if 'mean' not in name or row.get('sep_old_pixels', 0) > 0]
        distributions[name] = distribution(values)
    def pooled_fraction(numerator, denominator):
        return ratio(totals[numerator], totals[denominator]) if numerator in totals and denominator in totals else None
    return {'available': True, 'guard_logged_batches': len(guarded),
            'logged_iteration_range': [guarded[0]['iteration'], guarded[-1]['iteration']],
            'totals_of_sampled_masses_or_pixel_occurrences': totals,
            'batch_distributions': distributions,
            'pooled_old_evidence_weight_removed_fraction': pooled_fraction(
                'guard_old_removed_weight_sum', 'guard_old_weight_sum_before'),
            'pooled_main_native_push_removed_fraction': pooled_fraction(
                'guard_removed_main_push', 'guard_total_before_main_push'),
            'pooled_proto_native_push_removed_fraction': pooled_fraction(
                'guard_removed_proto_push', 'guard_total_before_proto_push'),
            'batch_fraction_distributions': {'old_evidence_weight_removed': distribution(per_batch_old_removed),
                                             'main_native_push_removed': distribution(per_batch_main_removed),
                                             'proto_native_push_removed': distribution(per_batch_proto_removed)},
            'guard_invariant_inconsistent_observations': inconsistent,
            'missing_fields_by_logged_iteration': missing,
            'eligibility_definition': 'Original eligible gates and class-count denominator remain unchanged; sep_old/new/nonzero pixels count original eligibility. guard_effective pixels count positive weights after attenuation.',
            'factor_definition': 'Only old-anchor weights multiply (1-max_current_new_CAM)^2. New-anchor numerator weights are unchanged.',
            'local_push_definition': 'Logged push is sum evidence_weight*sigmoid(1-native_pair_logit_margin), separately for main/prototype. It is the magnitude of one native-logit pair derivative before class normalization, branch0.5 averaging, outer lambda/ramp and optimizer effects.',
            'local_push_denominator': 'The logged before/after main/prototype push totals include all eligible old and new anchors. Removed push is old-anchor-only; pooled fractions use the all-anchor before total.',
            'interpretation': 'Local push is not a parameter-gradient norm, backbone-gradient allocation, probability of correctness, or GT accuracy. Sparse sums count repeated training exposures, not full-run totals. No GT error/correct fractions can be inferred from these logs.'}


def summarize_stage(stage_dir, stage):
    config = read_json(stage_dir/'config.json')
    if config is None:
        return {'stage': stage, 'status': 'not_started_or_config_not_yet_written',
                'data_generated': False}
    pair, pair_source = read_samples(stage_dir/'pair_metrics.jsonl')
    kd, kd_source = read_samples(stage_dir/'kd_metrics.jsonl')
    kd_by_iteration = {row['iteration']: row for row in kd}
    warmup = int(config['loss_warmup_iters'])
    active_pair = [row for row in pair if row['iteration']-1 >= warmup]
    warmup_pair = [row for row in pair if row['iteration']-1 < warmup]
    active_kd = [row for row in kd if row.get('active', row['iteration']-1 >= warmup)]
    flag_mismatch = [row['iteration'] for row in kd
                     if row.get('active', row['iteration']-1 >= warmup) != (row['iteration']-1 >= warmup)]
    confusion_path = next((stage_dir/name for name in ['online_confusion_latest.json', 'online_confusion.json']
                           if (stage_dir/name).exists()), None)
    confusion = read_json(confusion_path) if confusion_path is not None else None
    latest_selector = pair[-1].get('selector', {}) if pair else {}
    classes = int(latest_selector.get('classes', confusion['classes'] if confusion else (16 if stage == 1 else 21)))
    observed_offset = (confusion['training_iteration']-confusion['updates']) if confusion else None
    latest_iteration = max([row['iteration'] for row in pair+kd], default=None)
    completed = (stage_dir/'training_complete.json').exists()
    last_is_endpoint = latest_iteration == config.get('max_iters') and completed
    per_stage = {
        'stage': stage, 'status': 'training_complete' if completed else 'in_progress',
        'data_generated': bool(pair or kd or confusion),
        'config': {key: config.get(key) for key in ['pair_mode', 'w_pair_sep', 'w_pixel_kd',
                    'loss_warmup_iters', 'pair_min_updates', 'pair_ramp_updates', 'pair_refresh_interval',
                    'pair_min_row_images', 'pair_min_pair_images', 'pair_min_rate', 'pair_max_stale_updates',
                    'max_iters', 'log_iters', 'resume_checkpoint', 'new_cam_guard', 'sep_new_cam_guard', 'spg', 'seed']},
        'sample_sources': {'pair': pair_source, 'kd': kd_source, 'latest_confusion': str(confusion_path) if confusion_path else None},
        'latest_logged_iteration': latest_iteration,
        'latest_logged_sample_is_completed_endpoint': last_is_endpoint,
        'phase_accounting': {'observation_training_iteration_offset': observed_offset,
            'latest_observer_updates': confusion.get('updates') if confusion else None,
            'latest_observer_seen_image_exposures': confusion.get('seen_images') if confusion else None,
            'loss_first_active_training_iteration': warmup+1,
            'warmup_pair_samples': len(warmup_pair), 'active_pair_samples': len(active_pair),
            'active_kd_flag_inconsistent_iterations': flag_mismatch,
            'first_logged_positive_ramp_iteration': next((row['iteration'] for row in pair if row.get('ramp', 0) > 0), None),
            'first_logged_active_nonzero_sep_iteration': next((row['iteration'] for row in active_pair
                if row.get('ramp', 0) > 0 and row.get('sep_nonzero_pixels', 0) > 0), None),
            'interpretation': 'Stage1 observer cold-starts after reused iteration2000 warmup; stage2 observer runs from its first batch, including loss warmup. Statistics/selection/ramp presence does not imply losses are active. Logged first occurrence is an upper bound on actual first activation.'},
        'selection_all_logged': selection_summary(pair),
        'selection_objective_active': selection_summary(active_pair),
        'sep_warmup_observations': sep_coverage(warmup_pair, kd_by_iteration),
        'sep_objective_active': sep_coverage(active_pair, kd_by_iteration),
        'sep_active_class_coverage': class_coverage(active_pair, 'sep_class_eligible_pixels', classes),
        'sep_active_trends': metric_trends([row for row in active_pair
                                          if row.get('ramp', 0) > 0 and row.get('sep_nonzero_pixels', 0) > 0]),
        'kd_objective_active': kd_summary(active_kd, classes, config.get('pair_mode')),
        'outer_objectives_when_loss_active': outer_objectives(active_pair, kd_by_iteration, config),
        'latest_logged_selector': {'iteration': pair[-1]['iteration'] if pair else None,
            'is_completed_endpoint': last_is_endpoint,
            'last_refresh_iteration': latest_selector.get('last_refresh_iteration'),
            'targets': latest_selector.get('targets'), 'selected_rates_diagnostic_only': latest_selector.get('selected_rates'),
            'rates_warning': 'Selected rates rank supported directions and describe EMA confusion. They are not loss multipliers or calibrated GT error probabilities.'}}
    if confusion:
        targets = latest_selector.get('targets', [])
        per_stage['latest_selected_relation_support'] = [
            {'anchor': i, 'competitor': j,
             'broad_row_image_exposures': confusion['broad_image_observations'][i],
             'broad_pair_image_exposures': confusion['broad_pair_image_observations'][i][j],
             'broad_row_staleness_updates': confusion['broad_staleness_updates'][i]}
            for i, j in enumerate(targets) if i > 0 and j > 0 and j != i]
        per_stage['relation_support_snapshot_caveat'] = 'Observer snapshot and latest logged selector can differ by one or more samples; counts are exposures including repeated images.'
    if latest_selector.get('distillation_class_limit') is not None:
        limit = int(latest_selector['distillation_class_limit'])
        per_stage['selection_domain'] = {
            'selector_schema': latest_selector.get('schema'), 'distillation_class_limit': limit,
            'source_class_ids': list(range(1, limit)), 'partner_class_ids': list(range(1, limit)),
            'excluded_teacher_unknown_class_ids': list(range(limit, classes)),
            'row_mass_definition': 'Rates still divide by the full matrix source row, including background, diagonal and teacher-unknown new classes; no old-only renormalization.',
            'unknown_target_meaning': 'Teacher-unknown new source rows deliberately have target-1; this does not imply their online confusion evidence is missing.',
            'support_policy': 'Existing image-exposure support, absolute full-row minrate, staleness and ramp remain unchanged.'}
    if config.get('new_cam_guard') or config.get('sep_new_cam_guard') or any(row.get('new_cam_guard') is True for row in pair):
        per_stage['new_cam_guard_all_logged'] = guard_summary(pair)
        per_stage['new_cam_guard_objective_active'] = guard_summary(active_pair)
    if config.get('w_pair_sep', .1) == 0:
        # Pair selection may still drive KD, and raw SEP diagnostics may be
        # computed, but a zero outer coefficient cannot optimize SEP.
        per_stage['sep_observational_only'] = {
            'coefficient': 0,
            'coverage_during_active_loss_phase': per_stage['sep_objective_active'],
            'region_coverage_during_active_loss_phase': per_stage['sep_active_class_coverage'],
            'raw_margin_and_loss_trends': per_stage['sep_active_trends'],
            'meaning': 'SEP quantities are computed observations only; zero w_pair_sep supplies no SEP optimization gradient. The same selected directions can actively drive pair KD.'}
        per_stage['sep_objective_active'] = sep_coverage([], kd_by_iteration)
        per_stage['sep_active_class_coverage'] = class_coverage([], 'sep_class_eligible_pixels', classes)
        per_stage['sep_active_trends'] = metric_trends([])
        per_stage['phase_accounting']['loss_phase_pair_observation_samples'] = len(active_pair)
        per_stage['phase_accounting']['active_pair_samples'] = 0
        per_stage['phase_accounting']['first_logged_active_nonzero_sep_iteration'] = None
    return per_stage


def report(root):
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'schema': 1,
              'run_root': str(root), 'scope': 'Existing formal sparse training observations only',
              'global_caveats': ['No missing future stages, batches or GT accuracy measurements are synthesized.',
                'Logged batches are sampled, normally every50iterations; totals count sampled pixel exposures, not unique pixels or full-run totals.',
                'Pair switching and direction frequency describe sampled snapshots, not exact intervening updates or gradient allocation.',
                'Training loss/margins depend on changing samples and pair selections; accuracy improvement must come from independent endpoint evaluation.',
                'Selected confusion proportions are diagnostics and rankings only; learning strength uses bounded ramp and evidence gates.',
                'Guard local push is a native-logit pair-derivative mass before class normalization and coefficients, not a parameter-gradient norm or GT measure.'],
              'arms': {}}
    for arm in ['a_sep', 'b_pairkd']:
        result['arms'][arm] = {'stages': {str(stage): summarize_stage(root/'formal'/arm/f'10-5/step{stage}', stage)
                                        for stage in [1, 2]}}
    c_run = root/'formal/c_newaware_sep'
    if (c_run/'study.json').exists():
        parent_stage = root/'formal/a_sep/10-5/step1'
        inherited_complete = (parent_stage/'training_complete.json').exists()
        stage2 = summarize_stage(c_run/'10-5/step2', 2)
        stage2['continuation_caveat'] = 'Restores parent A stage2 iteration2000 student, optimizer, observer and selector; only remaining6000 steps are trained.8GPU x batch1 replaces parent4GPU x batch2, globalbatch8 unchanged; sampler/augmentation restart prevents bitwise causal isolation.'
        result['arms']['c_newaware_sep'] = {
            'lineage': {'study': str(c_run/'study.json'), 'parent_arm': 'a_sep',
                        'stage1_inherited': True, 'stage2_resume_iteration': 2000,
                        'stage2_remaining_training_iterations': 6000,
                        'scope': 'Old-anchor SEP numerator protection only; matrix direction selection, all original gates/class counts, lambda and new-anchor supervision retained.',
                        'development_diagnostic_source': str(root/'diagnostics/sep_new_conflict/merged.json'),
                        'development_validation_caveat': '128GT diagnostic images informed the C change; full endpoints use the same validation set and are development validation rather than a new independent test. Static push selectivity is not final accuracy gain.'},
            'stages': {'1': {'stage': 1, 'status': 'inherited_training_complete' if inherited_complete else 'inherited_parent_not_yet_confirmed',
                             'data_generated': False, 'inherited_from_arm': 'a_sep',
                             'inherited_source_stage_dir': str(parent_stage),
                             'meaning': 'C reuses the completed A stage1 teacher; no separate C stage1 training or synthesized observations.'},
                       '2': stage2}}
    d_run = root/'formal/d_pairkd_only'
    if (d_run/'study.json').exists():
        parent_stage = root.parent/'kd_parallel_v1/formal/b_relational/10-5/step1'
        parent_warmup = root.parent/'kd_parallel_v1/formal/b_relational/10-5/step2/checkpoints/model_iter_2000.pth'
        inherited_complete = (parent_stage/'training_complete.json').exists()
        stage2 = summarize_stage(d_run/'10-5/step2', 2)
        stage2['continuation_caveat'] = 'Restores optimized-KD b_relational stage2 iteration2000 student and optimizer with the same optimized-KD stage1 teacher.8GPU x batch1 changes distributed partitioning from parent4GPU x batch2; restarted sampling/augmentation prevents strict single-factor causal isolation.'
        stage2['observer_initialization'] = {
            'type': 'cold_start', 'training_iteration_at_start': 2000,
            'inherited_observer_or_selector_state': False,
            'warmup2000_observations_seen': False,
            'expected_observation_training_iteration_offset': 2000,
            'actual_observation_training_iteration_offset': stage2.get('phase_accounting', {}).get('observation_training_iteration_offset'),
            'interpretation': 'D starts its online matrix from zero at resume. Observation warmup and pair-blend ramp occur after iteration2000; optimizer/model warmup state does not imply existing confusion evidence.'}
        result['arms']['d_pairkd_only'] = {
            'lineage': {'study': str(d_run/'study.json'), 'parent_method': 'optimized_KD',
                        'parent_arm': 'b_relational', 'stage1_inherited': True,
                        'shared_stage2_warmup_checkpoint': str(parent_warmup),
                        'stage2_resume_iteration': 2000, 'stage2_remaining_training_iterations': 6000,
                        'observer_and_selector_cold_start': True,
                        'scope': 'Teacher-compatible old-class pair KD alone, maxblend0.5; w_pair_sep0. Selector source and partner IDs1..15 use full21-row rates. Raw SEP logs are observational only.',
                        'selection_domain': {'selector_schema': 2, 'distillation_class_limit': 16,
                                             'source_class_ids': list(range(1, 16)), 'partner_class_ids': list(range(1, 16)),
                                             'rate_denominator': 'Full21-class EMA row; original support/rate/staleness/ramp, no submatrix renormalization.',
                                             'teacher_soft_relations_preserved': True},
                        'evaluation_workers_required': 4,
                        'development_validation_caveat': 'Selected after A/B/C outcomes on the same validation task; full endpoint comparison is development exploration, not independent confirmation.'},
            'stages': {'1': {'stage': 1, 'status': 'inherited_training_complete' if inherited_complete else 'inherited_parent_not_yet_confirmed',
                             'data_generated': False, 'inherited_from_method': 'optimized_KD',
                             'inherited_from_arm': 'b_relational', 'inherited_source_stage_dir': str(parent_stage),
                             'meaning': 'D reuses optimized-KD b_relational stage1 teacher, not A or C. No separate D stage1 training or synthesized observations.'},
                       '2': stage2}}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', default=DEFAULT_ROOT)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    if os.name != 'nt' and not root.is_relative_to(ALLOWED_ROOT.resolve()):
        raise ValueError('Run root must stay inside the authorized workspace')
    output = Path(args.output).resolve() if args.output else root/'mechanism_report.json'
    if not output.is_relative_to(root):
        raise ValueError('Report output must stay inside run root')
    if output.exists() and output.stat().st_nlink != 1:
        raise ValueError('Report target must not be hard linked')
    result = report(root)
    temporary = output.with_name(output.name+f'.{os.getpid()}.tmp')
    with temporary.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    os.replace(temporary, output)
    compact = {'output': str(output), 'arms': {}}
    for arm, arm_report in result['arms'].items():
        compact['arms'][arm] = {stage: {'status': row['status'],
            'last_logged_iteration': row.get('latest_logged_iteration'),
            'active_sep_batches': row.get('sep_objective_active', {}).get('logged_batches'),
            'active_kd_pair_batches': row.get('kd_objective_active', {}).get('positive_pair_batches')}
            for stage, row in arm_report['stages'].items()}
    print(json.dumps(compact, ensure_ascii=False))


if __name__ == '__main__':
    main()
