"""CPU-only checkpoint geometry audit for a proposed confusion-guided SEP.

No forward image pass, pixel GT, optimizer, training or source modification.
Saved selector directions are diagnostics, not calibrated loss weights.
Cross-checkpoint selector transfers are explicitly post-hoc observations and
must not be confused with the selector available at that training iteration.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import gc
import hashlib
import json
import math
import os
import re
import sys
import uuid


AREA = Path('/ML-vePFS/infra_rd/kun/others/wzg')
ROOT = AREA/'workspace/evoproto'
EXPERIMENT = ROOT/'experiments/prototype_sep_v1'
NAMES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
         'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
         'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
         'tvmonitor']
OPT = ROOT/'runs/kd_parallel_v1/formal/b_relational/10-5'
LEGACY = ROOT/'runs/fixed_baseline_v1/sep/10-5'
FORMAL = ROOT/'runs/confusion_guided_v1/formal'
DEFAULT_CHECKPOINTS = {
    'opt_s1_warmup2000': ROOT/'runs/kd_pixel_v2/10-5/step1/checkpoints/model_iter_2000.pth',
    'opt_s1_final': OPT/'step1/checkpoints/model_final.pth',
    'opt_s2_warmup2000': OPT/'step2/checkpoints/model_iter_2000.pth',
    'opt_s2_final': OPT/'step2/checkpoints/model_final.pth',
    'legacy_sep_s1_final': LEGACY/'step1/checkpoints/model_final.pth',
    'legacy_sep_s2_final': LEGACY/'step2/checkpoints/model_final.pth',
    'A_s2_warmup2000': FORMAL/'a_sep/10-5/step2/checkpoints/model_iter_2000.pth',
    'A_s2_final': FORMAL/'a_sep/10-5/step2/checkpoints/model_final.pth',
    'B_s2_final': FORMAL/'b_pairkd/10-5/step2/checkpoints/model_final.pth',
    'C_s2_final': FORMAL/'c_newaware_sep/10-5/step2/checkpoints/model_final.pth',
    'D_s2_final': FORMAL/'d_pairkd_only/10-5/step2/checkpoints/model_final.pth',
}
DEFAULT_REFERENCES = {key: value for key, value in DEFAULT_CHECKPOINTS.items()
                      if key in ['A_s2_final', 'B_s2_final', 'C_s2_final', 'D_s2_final']}


def named_paths(values, defaults):
    if not values:
        return dict(defaults)
    result = {}
    for value in values:
        if '=' not in value:
            raise ValueError('Use LABEL=/absolute/checkpoint/path for checkpoint inputs')
        name, path = value.split('=', 1)
        if not name or name in result:
            raise ValueError('Checkpoint labels must be nonempty and unique')
        result[name] = Path(path)
    return result


def safe_output(path):
    """Resolve every destination; reject links escaping A and hardlinked files."""
    path = Path(path)
    if not path.is_absolute():
        path = ROOT/path
    allowed = AREA.resolve(strict=True)
    resolved = path.resolve(strict=False)
    if resolved == allowed or allowed not in resolved.parents:
        raise ValueError('Output must resolve inside the authorized A directory')
    if path.exists() and (path.is_symlink() or path.stat().st_nlink != 1):
        raise ValueError('Refuse to replace symlinked or hardlinked output')
    return resolved


def sha256(path):
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def load_checkpoint(path, torch, include_hash=False):
    path = Path(path).resolve(strict=True)
    value = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    state = value.get('model_state', value)
    found = []
    for name, tensor in state.items():
        match = re.search(r'(?:^|\.)decoder\.class_prototypes\.(\d+)\.prototype$', name)
        if match:
            if tensor.ndim != 2:
                raise ValueError(f'Invalid prototype tensor shape: {name}')
            found.append((int(match.group(1)), tensor.detach().to(dtype=torch.float64).clone()))
    found.sort(key=lambda item: item[0])
    if not found or [key for key, _ in found] != list(range(len(found))):
        raise ValueError('Expected consecutive incremental prototype blocks')
    raw = torch.cat([tensor for _, tensor in found], dim=0)
    if not torch.isfinite(raw).all() or (raw.norm(dim=1) <= 0).any():
        raise ValueError('Prototype rows must be finite and nonzero')
    previous_channels = sum(t.shape[0] for _, t in found[:-1])
    selector = value.get('pair_selector_state')
    observer = value.get('online_confusion_state')
    def detached_dict(dictionary):
        if dictionary is None:
            return None
        return {key: val.detach().clone() if isinstance(val, torch.Tensor) else val
                for key, val in dictionary.items()}
    result = {'path': str(path), 'size_bytes': path.stat().st_size,
              'iteration': value.get('iteration'), 'raw': raw,
              'stage': len(found)-1, 'previous_channels': previous_channels,
              'blocks': [t.shape[0] for _, t in found],
              'selector': detached_dict(selector), 'observer': detached_dict(observer)}
    if include_hash:
        result['sha256'] = sha256(path)
    del state, value
    gc.collect()
    return result


def summary(values, torch, margin):
    tensor = torch.as_tensor(values, dtype=torch.float64)
    if tensor.numel() == 0:
        return {'pairs': 0, 'active_pairs': 0, 'active_fraction': None,
                'mean_squared_hinge': None, 'cosine': None}
    excess = (tensor-margin).clamp_min(0)
    return {'pairs': tensor.numel(), 'active_pairs': int((excess > 0).sum()),
            'active_fraction': float((excess > 0).double().mean()),
            'mean_squared_hinge': float(excess.square().mean()),
            'cosine': {'min': float(tensor.min()), 'q25': float(tensor.quantile(.25)),
                       'median': float(tensor.median()), 'q75': float(tensor.quantile(.75)),
                       'q90': float(tensor.quantile(.9)), 'q95': float(tensor.quantile(.95)),
                       'max': float(tensor.max()), 'mean': float(tensor.mean()),
                       'std_population': float(tensor.std(unbiased=False))}}


def pair_group(i, j, previous):
    i_old, j_old = i < previous, j < previous
    return ('old_old' if i_old and j_old else 'new_new' if not i_old and not j_old
            else 'old_new' if i_old else 'new_old')


def pair_record(i, j, cosine, previous, margin):
    score = float(cosine[i, j])
    violation = max(0., score-margin)
    return {'anchor': i, 'competitor': j,
            'anchor_name': NAMES[i] if i < len(NAMES) else str(i),
            'competitor_name': NAMES[j] if j < len(NAMES) else str(j),
            'group': pair_group(i, j, previous), 'cosine': score,
            'margin': margin, 'violation': violation, 'active': violation > 0,
            'squared_hinge': violation**2, 'd_loss_d_cosine_unnormalized': 2*violation}


def geometry(checkpoint, torch, margin):
    raw = checkpoint['raw']
    normalized = torch.nn.functional.normalize(raw, dim=1)
    cosine = normalized @ normalized.T
    previous = checkpoint['previous_channels']
    records = [pair_record(i, j, cosine, previous, margin)
               for i in range(1, raw.shape[0]) for j in range(i+1, raw.shape[0])]
    groups = {name: summary([p['cosine'] for p in records
                            if name == 'all_foreground' or p['group'] == name], torch, margin)
              for name in ['all_foreground', 'old_old', 'old_new', 'new_new']}
    meta = {key: checkpoint[key] for key in ['path', 'size_bytes', 'iteration',
                                            'stage', 'previous_channels', 'blocks']}
    if 'sha256' in checkpoint:
        meta['sha256'] = checkpoint['sha256']
    return cosine, {**meta, 'class_count': raw.shape[0], 'feature_dimension': raw.shape[1],
                    'raw_prototype_norms': raw.norm(dim=1).tolist(),
                    'groups': groups, 'pairs': records,
                    'highest_cosine_pairs': sorted(records, key=lambda p: (-p['cosine'], p['anchor'], p['competitor']))[:20]}


def selection(checkpoint, reference, cosine, torch, margin, same_checkpoint):
    selector = reference['selector']
    if selector is None:
        return {'status': 'no_saved_selector'}
    targets = selector['targets'].tolist()
    if len(targets) != cosine.shape[0]:
        return {'status': 'class_domain_mismatch', 'selector_classes': len(targets),
                'geometry_classes': cosine.shape[0]}
    rates = selector.get('selected_rates')
    observer = reference['observer']
    previous = checkpoint['previous_channels']
    records, invalid = [], []
    for i, j in enumerate(targets):
        if j == -1:
            continue
        if i == 0 or j == 0 or j == i or not 0 <= j < len(targets):
            invalid.append({'anchor': i, 'competitor': j})
            continue
        row = pair_record(i, j, cosine, previous, margin)
        row['selected_rate_diagnostic_only'] = float(rates[i]) if rates is not None else None
        if observer is not None:
            for key, target_key in [('broad_image_observations', 'row_image_exposures'),
                                    ('broad_last_seen_update', 'row_last_seen_update')]:
                if key in observer:
                    row[target_key] = int(observer[key][i])
            if 'broad_pair_image_observations' in observer:
                row['pair_image_exposures'] = int(observer['broad_pair_image_observations'][i, j])
            if 'pair_image_observations' in observer:
                row['trusted_pair_image_exposures'] = int(observer['pair_image_observations'][i, j])
            if 'image_observations' in observer:
                row['trusted_row_image_exposures'] = int(observer['image_observations'][i])
            if 'last_seen_update' in observer:
                row['trusted_row_last_seen_update'] = int(observer['last_seen_update'][i])
        records.append(row)
    policy = [p for p in records if p['group'] != 'old_old']
    unique = {tuple(sorted([p['anchor'], p['competitor']])): p for p in policy}
    counts = {group: sum(p['group'] == group for p in records)
              for group in ['old_old', 'old_new', 'new_old', 'new_new']}
    # A real autograd check on the proposed geometric objective: old rows are
    # detached, current rows retain gradients. No network is loaded or trained.
    grad_raw = checkpoint['raw'].detach().clone().requires_grad_(True)
    pnorm = torch.nn.functional.normalize(grad_raw, dim=1)
    terms = []
    for i, j in sorted(unique):
        left = pnorm[i].detach() if i < previous else pnorm[i]
        right = pnorm[j].detach() if j < previous else pnorm[j]
        terms.append(torch.relu(left.dot(right)-margin).square())
    objective = torch.stack(terms).mean() if terms else grad_raw.sum()*0
    gradient, = torch.autograd.grad(objective, grad_raw)
    return {'status': 'complete',
            'provenance': 'saved_selector_from_this_checkpoint' if same_checkpoint else 'posthoc_selector_transfer_not_available_during_this_checkpoint_training',
            'reference_path': reference['path'], 'selector_metadata': selector.get('_extra_state'),
            'last_refresh_iteration': int(selector['last_refresh_iteration']),
            'observer_updates': int(observer['updates']) if observer is not None else None,
            'targets': targets, 'invalid_targets': invalid, 'directions': records,
            'direction_counts': counts,
            'selected_directed_geometry': summary([p['cosine'] for p in records], torch, margin),
            'proposal_skip_oldold_detach_old_deduplicate': {
                'old_old_directions_skipped': counts['old_old'],
                'retained_directions': len(policy), 'unique_pairs': len(unique),
                'geometry': summary([p['cosine'] for p in unique.values()], torch, margin),
                'hypothetical_unweighted_mean_loss': float(objective.detach()),
                'per_class_prototype_gradient_norms': gradient.norm(dim=1).tolist(),
                'old_prototype_gradient_norm': float(gradient[:previous].norm()),
                'new_prototype_gradient_norm': float(gradient[previous:].norm()),
                'direct_encoder_or_main_head_gradient': 'none: those tensors are absent from the prototype-only loss graph',
                'definition': 'Unweighted mean over unique selected non-oldold pairs; detach old rows only inside SEP; zero hinge after reaching margin. This is an audit proposal, not an efficacy result.'}}


def replay_geometry_selector(checkpoint, reference, cosine, torch, margin, same_checkpoint):
    """Preview broad ranking plus trusted support, with old-old removed first.

    This is a diagnostic recomputation from saved observer state, not an
    assertion that the original saved selector followed the proposed policy.
    Geometry is assessed after selection, never used to force negative margin.
    """
    observer = reference['observer']
    selector = reference['selector']
    if observer is None or selector is None:
        return {'status': 'no_saved_observer_or_selector'}
    classes = cosine.shape[0]
    if observer['broad_ema_counts'].shape != (classes, classes):
        return {'status': 'class_domain_mismatch'}
    cfg = selector['_extra_state']
    updates = int(observer['updates'])
    counts = observer['broad_ema_counts'].to(torch.float64)
    mass = counts.sum(1)
    rates = counts / mass.clamp_min(torch.finfo(counts.dtype).tiny)[:, None]
    def fresh(name):
        seen = observer[name]
        age = updates-1-seen
        return (seen >= 0) & (age >= 0) & (age <= cfg['max_stale_updates'])
    row_support = fresh('broad_last_seen_update') & (observer['broad_image_observations'] >= cfg['min_row_images'])
    broad_eligible = ((observer['broad_pair_image_observations'] >= cfg['min_pair_images'])
                      & (rates >= cfg['min_rate']) & (counts > 0)
                      & row_support[:, None] & (mass > 0)[:, None])
    broad_eligible[0, :] = False
    broad_eligible[:, 0] = False
    broad_eligible.fill_diagonal_(False)
    previous = checkpoint['previous_channels']
    broad_eligible[:previous, :previous] = False
    trusted_eligible = broad_eligible & (observer['pair_image_observations'] >= cfg['min_pair_images'])
    dual_eligible = trusted_eligible & fresh('last_seen_update')[:, None]
    if updates < cfg['min_updates']:
        dual_eligible.fill_(False)
    best_rates, targets = rates.masked_fill(~dual_eligible, -1.).max(1)
    targets = torch.where(best_rates >= 0, targets, torch.full_like(targets, -1))
    preview_reference = dict(reference)
    preview_reference['selector'] = {
        'targets': targets, 'selected_rates': best_rates.clamp_min(0),
        'last_refresh_iteration': selector['last_refresh_iteration'],
        '_extra_state': {**cfg, 'diagnostic_replay_policy': 'oldold excluded before ranking; broad supports plus trusted pair exposures>=3 and trusted row freshness; no teacher distillation class cap'}}
    result = selection(checkpoint, preview_reference, cosine, torch, margin, same_checkpoint)
    result['provenance'] = ('diagnostic_replay_from_this_checkpoint_observer' if same_checkpoint
                            else 'posthoc_replay_from_different_checkpoint_observer_not_training_targets')
    result['candidate_counts'] = {
        'broad_supported_non_oldold_directed': int(broad_eligible.sum()),
        'after_trusted_pair_support': int(trusted_eligible.sum()),
        'after_trusted_row_freshness_and_warmup': int(dual_eligible.sum()),
        'selected_rows': int((targets >= 0).sum())}
    domain = torch.ones_like(broad_eligible)
    domain[0, :] = False
    domain[:, 0] = False
    domain.fill_diagonal_(False)
    domain[:previous, :previous] = False
    remaining = domain.clone()
    filters = [
        ('broad_row_freshness', fresh('broad_last_seen_update')[:, None]),
        ('broad_row_image_support', (observer['broad_image_observations'] >= cfg['min_row_images'])[:, None]),
        ('positive_row_mass', (mass > 0)[:, None]),
        ('positive_pair_mass', counts > 0),
        ('broad_pair_image_support', observer['broad_pair_image_observations'] >= cfg['min_pair_images']),
        ('minimum_full_row_rate', rates >= cfg['min_rate']),
        ('trusted_pair_image_support', observer['pair_image_observations'] >= cfg['min_pair_images']),
        ('trusted_row_freshness', fresh('last_seen_update')[:, None]),
        ('observer_warmup', torch.full_like(domain, updates >= cfg['min_updates'])),
    ]
    rejections = []
    for label, gate in filters:
        after = remaining & gate
        rejections.append({'filter': label, 'rejected_at_this_filter': int((remaining & ~gate).sum()),
                           'remaining_directed_candidates': int(after.sum())})
        remaining = after
    if not torch.equal(remaining, dual_eligible):
        raise AssertionError('Candidate replay diagnostics disagree with the proposed eligibility mask')
    result['sequential_candidate_filter_diagnostics'] = {
        'initial_non_oldold_foreground_domain': int(domain.sum()), 'filters': rejections,
        'note': 'Rejection counts follow this stated sequence; overlapping causes are not independent causal attributions.'}
    result['trust_limit'] = 'Pair supports are cumulative repeated image exposures. Row freshness does not establish pair-specific recency or GT correctness.'
    result['geometry_gate'] = 'No geometry filter in ranking; hinge itself stops when cosine reaches margin.'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', action='append', help='LABEL=PATH; replaces default checkpoint list')
    parser.add_argument('--selector-reference', action='append', help='LABEL=PATH; replaces default saved selector references')
    parser.add_argument('--margin', type=float, default=0.)
    parser.add_argument('--output', default=str(EXPERIMENT/'prototype_geometry.json'))
    parser.add_argument('--hash', action='store_true', help='Hash large checkpoints; off to minimize filesystem IO')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.margin) or not -1 < args.margin < 1:
        raise ValueError('Cosine margin must be finite and strictly inside (-1, 1)')
    output = safe_output(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError('Report already exists; select a new output or explicitly --overwrite')
    os.chdir(ROOT)
    sys.dont_write_bytecode = True
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TMPDIR'] = str(AREA/'.kd8tmp')
    os.environ['XDG_CACHE_HOME'] = str(AREA/'.cache')
    import torch
    torch.set_num_threads(1)
    checkpoints = named_paths(args.checkpoint, DEFAULT_CHECKPOINTS)
    references = named_paths(args.selector_reference, DEFAULT_REFERENCES)
    missing, ref_data = [], {}
    for name, path in references.items():
        if not path.is_file():
            missing.append({'role': 'selector_reference', 'label': name, 'path': str(path)})
            continue
        ref_data[name] = load_checkpoint(path, torch, args.hash)
    report = {'utc': datetime.now(timezone.utc).isoformat(), 'device': 'cpu',
              'scope': 'Saved checkpoint prototype geometry and selector diagnostics only; no images, GT, optimizer, training or accuracy estimates.',
              'margin': args.margin, 'dtype': 'float64', 'background_excluded': True,
              'direct_gradient_fact': 'Prototype-only SEP differentiates class_prototypes only. Encoder/conv6/conv7/main_conv8 have no direct SEP gradient; later prototype-segmentation updates may indirectly change feature learning.',
              'direction_fact': 'Prototype cosine is symmetric; selection direction supplies priority only. Duplicate directed pairs are deduplicated for the proposed geometric objective. Old/new detach gives asymmetric parameter updates.',
              'selected_rates_usage': 'diagnostic ranking information only; never exact learning weights',
              'gt_usage': 'none in this script; any later pixel GT must remain diagnostic only',
              'checkpoints': {}, 'missing_inputs': missing}
    for name, path in checkpoints.items():
        if not path.is_file():
            missing.append({'role': 'checkpoint', 'label': name, 'path': str(path)})
            continue
        checkpoint = load_checkpoint(path, torch, args.hash)
        cosine, entry = geometry(checkpoint, torch, args.margin)
        entry['own_saved_selector'] = selection(checkpoint, checkpoint, cosine, torch, args.margin, True)
        entry['own_proposed_dual_view_selector_replay'] = replay_geometry_selector(checkpoint, checkpoint, cosine, torch, args.margin, True)
        entry['transferred_selectors'] = {}
        entry['transferred_proposed_selector_replays'] = {}
        for ref_name, reference in ref_data.items():
            if reference['path'] != checkpoint['path']:
                entry['transferred_selectors'][ref_name] = selection(checkpoint, reference, cosine, torch, args.margin, False)
                entry['transferred_proposed_selector_replays'][ref_name] = replay_geometry_selector(checkpoint, reference, cosine, torch, args.margin, False)
        report['checkpoints'][name] = entry
        group = entry['groups']['all_foreground']
        print(json.dumps({'checkpoint': name, 'classes': entry['class_count'],
                          'active_all': group['active_pairs'], 'pairs_all': group['pairs'],
                          'mean_squared_hinge': group['mean_squared_hinge'],
                          'own_selector': entry['own_saved_selector']['status']}), flush=True)
        del checkpoint, cosine
        gc.collect()
    if not report['checkpoints']:
        raise RuntimeError('No checkpoint was analyzed')
    output.parent.mkdir(parents=True, exist_ok=True)
    output = safe_output(output)
    temporary = safe_output(output.with_name(output.name+'.'+uuid.uuid4().hex+'.tmp'))
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, output)
    print(json.dumps({'report': str(output), 'checkpoints': len(report['checkpoints']),
                      'missing_inputs': len(missing)}), flush=True)


if __name__ == '__main__':
    main()
