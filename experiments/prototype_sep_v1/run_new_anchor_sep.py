"""One conditional stage2 new-anchor SEP add-on to the best optimized KD.

The isolated c_new_anchor source differs from tested B only in selector policy:
current new foreground sources rank any supported foreground competitor.
Original optimized-KD teacher and stage2/2000 student+optimizer are loaded;
absent online states start fresh. This is not a teacher causal ablation.
Development GT128 disagreement evidence motivates source eligibility, never
training labels, selection scores or exact learning weights.
"""
from pathlib import Path
import argparse
import ast
import json
import os
import signal
import socket
import subprocess
import sys
import time

R = Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto')
E = R / 'experiments/prototype_sep_v1'
U = R / 'runs/prototype_sep_v1'
S = E / 'c_new_anchor/src'
BSRC = E / 'b_semantic/src'
B = U / 'formal/b_semantic'
OPT = R / 'runs/kd_parallel_v1/formal/b_relational'
PY = R / '.runtime/env/bin/python'
RUNNER = E / 'run_new_anchor_sep.py'
LAUNCHER = E / 'launch_new_anchor_sep.py'
TEACHER = OPT / '10-5/step1/checkpoints/model_final.pth'
RESUME = OPT / '10-5/step2/checkpoints/model_iter_2000.pth'
TARGET = 69.14929219468897
DERIVED_KEYS = ('local_rank', 'step', 'work_dir', 'prev_checkpoint',
                'resume_checkpoint', 'ckpt_dir', 'pred_dir')
GEOMETRY_CONFIG = {
    'w_geometry_sep': .1, 'confusion_momentum': .98,
    'pair_refresh_interval': 50, 'pair_min_row_images': 8,
    'pair_min_pair_images': 3, 'pair_min_rate': .01,
    'pair_min_updates': 100, 'pair_ramp_updates': 200,
    'pair_max_stale_updates': 200,
}
SMOKE_OVERRIDES = {
    'max_iters': 2008, 'log_iters': 1, 'eval_iters': 2008,
    'val_limit': 16, 'train_limit': 64,
    'pair_min_updates': 0, 'pair_ramp_updates': 1,
    'pair_min_row_images': 1, 'pair_min_pair_images': 1,
    'pair_min_rate': 0., 'pair_refresh_interval': 1,
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def initialize():
    sys.path.insert(0, str(R / 'experiments/kd_pixel_v2'))
    from run_kd import environment
    sys.path.insert(0, str(S))
    from kd_runtime import safe_path, atomic_json, now, digest
    return environment, safe_path, atomic_json, now, digest


def read_json(path, services):
    safe_path = services[1]
    return json.loads(safe_path(path).read_text())


def sources(services, source=None):
    safe_path, digest = services[1], services[4]
    source = S if source is None else source
    require(safe_path(source).is_dir(), 'Frozen source is missing: ' + str(source))
    for path in source.rglob('*'):
        safe_path(path)
    return {str(path.relative_to(source)): digest(path) for path in sorted(source.rglob('*.py'))
            if '__pycache__' not in path.parts}


def gates(services):
    """Read-only completed-B decision and new-selector-specific test gates."""
    safe_path, digest = services[1], services[4]
    require(read_json(B / 'status.json', services).get('status') == 'complete',
            'B must complete before any new-anchor smoke or formal launch')
    study = read_json(B / 'study.json', services)
    require(study.get('arm') == 'b_semantic' and study.get('smoke') is False,
            'Conditional gate requires the formal B study')
    parent_code = sources(services, BSRC)
    preflight = read_json(E / 'semantic_preflight.json', services)
    require(parent_code == study['source_sha256'] == preflight['source_sha256']
            and preflight.get('passed') is True,
            'Parent B must retain its exact passed loss/DDP source freeze')
    code = sources(services)
    changed = sorted(key for key in set(code) | set(parent_code)
                     if code.get(key) != parent_code.get(key))
    require(changed == ['model/geometry_pair_selector.py'],
            'New-anchor candidate may change only the geometry selector: ' + repr(changed))
    origin_path = E / 'new_anchor_origin.json'
    origin = read_json(origin_path, services)
    require(origin.get('parent_source') == str(BSRC)
            and origin.get('parent_source_sha256') == parent_code
            and origin.get('candidate_source') == str(S)
            and origin.get('candidate_source_sha256') == code
            and origin.get('source_sha256') == code
            and origin.get('changed_files') == changed
            and origin.get('selector_schema') == 4
            and origin.get('source_anchor_policy') == 'current_new_only'
            and origin.get('source_domain') == 'new_foreground_rows_only'
            and origin.get('parent_source_unchanged') is True
            and origin.get('KD_byte_identical') is True
            and origin.get('semantic_guard_byte_identical') is True,
            'New selector source/origin linkage disagrees with frozen B')
    tests_path = E / 'new_anchor_selector_tests.json'
    tests = read_json(tests_path, services)
    require(tests.get('passed') is True and tests.get('returncode') == 0
            and tests.get('count') == 10
            and tests.get('module_sha256') == code['model/geometry_pair_selector.py']
            and tests.get('source_sha256') == code
            and tests.get('test_source_sha256') == digest(safe_path(E / 'test_new_anchor_selector.py')),
            'The isolated schema4 selector must pass its exact10 CPU tests')
    report_path = safe_path(U / 'semantic_analysis.json')
    report = read_json(report_path, services)
    require(report.get('status') == 'complete_endpoint_analysis'
            and report.get('completion', {}).get('complete') is True
            and report.get('run') == str(B) and report.get('study') == study,
            'B complete endpoint analysis is missing or stale')
    recommendation = report['recommendation']
    candidate = report['candidate_endpoint']
    reference = report['references']['optimized_KD']
    candidate_all = candidate['metrics']['all_miou']
    reference_all = reference['metrics']['all_miou']
    require(abs(reference_all - TARGET) <= 1e-8
            and recommendation.get('candidate_exceeds_required_endpoint') is False
            and candidate_all <= reference_all,
            'New-anchor continuation is unnecessary: B exceeds the best endpoint')
    actual_b = read_json(B / 'evaluations/step2_iter8000/result.json', services)
    require(actual_b.get('step') == 2 and actual_b.get('iteration') == 8000
            and actual_b.get('images') == 1449
            and actual_b['histogram'] == candidate['histogram']
            and actual_b['checkpoint_sha256'] == candidate['checkpoint_sha256']
            and abs(actual_b['all_miou'] - candidate_all) <= 1e-8,
            'B decision analysis does not describe its final full evaluation')
    require(safe_path(RUNNER).is_file() and safe_path(LAUNCHER).is_file(),
            'Deploy both new-anchor scripts before checking readiness')
    return {'source_sha256': code, 'B_study': study,
            'parent_source': str(BSRC), 'parent_source_sha256': parent_code,
            'changed_files_from_B': changed,
            'selector_schema': 4, 'source_anchor_policy': 'current_new_only',
            'source_domain': 'new_foreground_rows_only',
            'new_anchor_origin': str(origin_path), 'new_anchor_origin_sha256': digest(origin_path),
            'new_anchor_selector_tests': str(tests_path), 'new_anchor_selector_tests_sha256': digest(tests_path),
            'selector_CPU_test_count': 10,
            'B_decision': recommendation,
            'B_decision_analysis': str(report_path),
            'B_decision_analysis_sha256': digest(report_path),
            'semantic_preflight': str(E / 'semantic_preflight.json'),
            'semantic_preflight_sha256': digest(E / 'semantic_preflight.json'),
            'runner_sha256': digest(RUNNER), 'launcher_sha256': digest(LAUNCHER)}


def configuration(smoke, services, gate):
    cfg = dict(read_json(OPT / '10-5/step2/config.json', services))
    for key in DERIVED_KEYS:
        cfg.pop(key, None)
    require(cfg.get('spg') == 2 and cfg.get('max_iters') == 8000
            and cfg.get('warmup_iters') == cfg.get('loss_warmup_iters') == 2000
            and cfg.get('seed') == 0 and cfg.get('w_pixel_kd') == .1
            and cfg.get('kd_temperature') == 2 and cfg.get('w_proto_seg') == .1
            and cfg.get('w_proto_kd') == cfg.get('w_proto_sep') == 0
            and cfg.get('proto_margin') == 0 and cfg.get('ald') is False
            and cfg.get('confusion_reweight') is False,
            'Archived best pipeline budget, KD or original SEP settings changed')
    bcfg = dict(gate['B_study']['common_config'])
    require(all(bcfg.get(key) == value for key, value in GEOMETRY_CONFIG.items()),
            'Do not change the already tested B geometry policy')
    cfg.update(GEOMETRY_CONFIG)
    cfg.update(spg=1, num_workers=2, async_eval=True)
    # Apart from topology/logistical spg, inherited KD settings equal B.
    differing = {key: (value, bcfg.get(key)) for key, value in cfg.items()
                 if key not in DERIVED_KEYS and value != bcfg.get(key)}
    require(not differing, 'Best pipeline/B common settings disagree: ' + repr(differing))
    if smoke:
        cfg.update(SMOKE_OVERRIDES)
    require(cfg['spg'] * 8 == 8 and cfg['proto_margin'] == 0
            and cfg['w_geometry_sep'] == .1, 'Preserve global8, margin0 and SEP.1')
    return cfg


def command_for(run, cfg):
    script = S / 'scripts/dist_train_voc_seg_neg.py'
    options = set()
    for node in ast.walk(ast.parse(script.read_text())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
            options.update(arg.value for arg in node.args if isinstance(arg, ast.Constant)
                           and isinstance(arg.value, str))
    command = [str(PY), '-B', '-m', 'torch.distributed.run', '--standalone',
               '--nproc_per_node=8', str(script), '--work_dir', str(run), '--step', '2',
               '--prev_checkpoint', str(TEACHER), '--resume_checkpoint', str(RESUME)]
    for key, value in cfg.items():
        if value is None:
            continue
        require('--' + key in options, 'Config has no argparse option: ' + key)
        if isinstance(value, bool):
            if value:
                command.append('--' + key)
            else:
                require('--no-' + key in options or key != 'pretrained',
                        'False pretrained requires explicit --no-pretrained')
                if '--no-' + key in options:
                    command.append('--no-' + key)
        elif isinstance(value, (list, tuple)):
            command.extend(['--' + key, *map(str, value)])
        else:
            command.extend(['--' + key, str(value)])
    return command


def restoration(services):
    """CPU-only metadata; all inherited checkpoint identities are published."""
    safe_path, digest = services[1], services[4]
    completed = read_json(OPT / 'training_complete.json', services)
    evaluators = read_json(OPT / 'evaluation_workers_complete.json', services)
    stages = [read_json(OPT / f'10-5/step{step}/launch.json', services) for step in (1, 2)]
    require(completed.get('status') == 'training_complete' and completed.get('stages') == 2
            and evaluators.get('returncodes') == [0, 0]
            and all(stage.get('returncode') == 0 for stage in stages),
            'Original optimized KD must have two successful stages and both historical evaluators exit0')
    teacher_done = read_json(OPT / '10-5/step1/training_complete.json', services)
    job = read_json(OPT / 'eval_queue/step2_iter2000.json', services)
    result = read_json(OPT / 'evaluations/step2_iter2000/result.json', services)
    teacher_sha, resume_sha = digest(safe_path(TEACHER)), digest(safe_path(RESUME))
    require(teacher_done.get('returncode') == 0
            and teacher_done.get('checkpoint_sha256') == teacher_sha,
            'Teacher SHA differs from successful optimized-KD stage1 receipt')
    require(job.get('step') == result.get('step') == 2
            and job.get('iteration') == result.get('iteration') == 2000
            and result.get('images') == 1449 and job.get('checkpoint') == str(RESUME)
            and job.get('checkpoint_sha256') == result.get('checkpoint_sha256') == resume_sha,
            'Resume SHA must agree with both published 2000 job and full result')
    import torch
    saved = torch.load(safe_path(RESUME), map_location='cpu', weights_only=True, mmap=True)
    require(saved.get('iteration') == 2000 and saved.get('model_state')
            and saved.get('optimizer_state'), 'Restore both student and optimizer at2000')
    require(not any(saved.get(key) is not None for key in
                    ('online_confusion_state', 'geometry_selector_state', 'pair_selector_state')),
            'Best-KD checkpoint must have no online states; C initializes fresh ones')
    optim = saved['optimizer_state']
    require(len(optim['param_groups']) == 4 and bool(optim['state'])
            and all(int(state['step']) == 2000 for state in optim['state'].values() if 'step' in state),
            'Expected full four-group AdamW state at2000')
    metadata = {'iteration': 2000, 'model_tensors': len(saved['model_state']),
                'optimizer_parameter_groups': len(optim['param_groups']),
                'optimizer_state_entries': len(optim['state']),
                'observer_state_present': False, 'selector_state_present': False,
                'teacher_sha256': teacher_sha, 'resume_checkpoint_sha256': resume_sha,
                'teacher_training_receipt': str(OPT / '10-5/step1/training_complete.json'),
                'resume_job': str(OPT / 'eval_queue/step2_iter2000.json'),
                'resume_result': str(OPT / 'evaluations/step2_iter2000/result.json')}
    del saved
    return metadata


def smoke_gate(services, gate, inherited):
    smoke = U / 'smoke/c_new_anchor'
    require(read_json(smoke / 'status.json', services).get('status') == 'complete',
            'Complete the new best-KD restoration smoke before formal C')
    study = read_json(smoke / 'study.json', services)
    require(study.get('arm') == 'c_new_anchor' and study.get('smoke') is True
            and study['source_sha256'] == gate['source_sha256']
            and study['runner_sha256'] == gate['runner_sha256']
            and study['launcher_sha256'] == gate['launcher_sha256']
            and study['teacher_sha256'] == inherited['teacher_sha256']
            and study['resume_checkpoint_sha256'] == inherited['resume_checkpoint_sha256'],
            'C smoke must use these exact sources/scripts and best-KD checkpoints')
    require(read_json(smoke / '10-5/step2/launch.json', services).get('returncode') == 0
            and read_json(smoke / 'evaluation_workers_complete.json', services).get('returncodes') == [0] * 4,
            'C restoration smoke trainer/evaluators did not exit0')
    stage = smoke / '10-5/step2'
    geometry = [json.loads(line) for line in services[1](stage / 'geometry_metrics.jsonl').read_text().splitlines() if line.strip()]
    guard = [json.loads(line) for line in services[1](stage / 'semantic_guard_metrics.jsonl').read_text().splitlines() if line.strip()]
    require(len(geometry) == len(guard) == 8
            and [row['iteration'] for row in guard] == list(range(2001, 2009)),
            'C smoke must log all8 resumed updates')
    require(any(row.get('semantic_gradient_used') is True for row in guard),
            'Smoke did not reach an active SEP semantic protection gradient')
    return {'run': str(smoke), 'eight_updates': True, 'active_protection_observed': True}


def stop_owned(children):
    for child in children:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 10
    while any(child is not None and child.poll() is None for child in children) and time.monotonic() < deadline:
        time.sleep(.1)
    for child in children:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=10)


def evaluation_complete(run, smoke, services):
    expected = ['step2_iter2008'] if smoke else ['step2_iter4000', 'step2_iter6000', 'step2_iter8000']
    require(sorted(path.stem for path in (run / 'eval_queue').glob('*.json')) == expected,
            'Unexpected C evaluation queue')
    previous_rows = None
    for name in expected:
        job = read_json(run / 'eval_queue' / (name + '.json'), services)
        folder = run / 'evaluations' / name
        result = read_json(folder / 'result.json', services)
        parts = [read_json(folder / f'rank{rank}.json', services) for rank in range(4)]
        names = [name for part in parts for name in part['images']]
        require(result.get('step') == 2 and result.get('iteration') == job['iteration']
                and result.get('images') == (16 if smoke else 1449)
                and len(names) == len(set(names)) == result['images']
                and result.get('checkpoint_sha256') == job['checkpoint_sha256'],
                'C evaluation is incomplete or has inconsistent checkpoint identity')
        require(all(part.get('rank') == rank and part.get('shards') == 4
                    and part.get('step') == 2 and part.get('iteration') == result['iteration']
                    and part.get('checkpoint_sha256') == result['checkpoint_sha256']
                    for rank, part in enumerate(parts)), 'C shard identity mismatch')
        summed = [[sum(part['histogram'][i][j] for part in parts) for j in range(21)] for i in range(21)]
        require(summed == result['histogram'], 'C merged histogram is not its four-shard sum')
        rows = [sum(row) for row in summed]
        require(previous_rows is None or rows == previous_rows, 'C full GT row totals changed across evaluations')
        previous_rows = rows


def main(smoke=False, check_only=False, print_command=False):
    services = initialize()
    environment, safe_path, atomic_json, now, digest = services
    gate = gates(services)
    cfg = configuration(smoke, services, gate)
    inherited = restoration(services)
    run = safe_path(U / ('smoke' if smoke else 'formal') / 'c_new_anchor')
    smoke_receipt = None if smoke else smoke_gate(services, gate, inherited)
    command = command_for(run, cfg)
    if check_only or print_command:
        print(json.dumps({'ready': True, 'outputs_written': False, 'run': str(run),
                          'configuration': cfg, 'restoration': inherited,
                          'command': command if print_command else None}, allow_nan=False))
        return
    import fcntl
    run.mkdir(parents=True, exist_ok=True)
    lock = open(safe_path(run / 'coordinator.lock'), 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    require(not (run / 'study.json').exists(), 'Refuse duplicate C training')
    env = environment('8card')
    env['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    atomic_json(run / 'study.json', {
        'created_utc': now(), 'arm': 'c_new_anchor', 'smoke': smoke,
        'common_config': cfg, 'train_gpus': list(range(8)), 'batch_per_gpu': 1, 'global_batch': 8,
        'source': str(S), 'source_reused_without_modification': False,
        'KD_byte_identical_to_B': True, 'semantic_guard_byte_identical_to_B': True,
        **{key: value for key, value in gate.items() if key != 'B_study'},
        'inherited_stage1_run': str(OPT), 'teacher': str(TEACHER),
        'teacher_sha256': inherited['teacher_sha256'],
        'resume_checkpoint': str(RESUME), 'shared_stage2_warmup': str(RESUME),
        'resume_checkpoint_sha256': inherited['resume_checkpoint_sha256'],
        'shared_stage2_warmup_sha256': inherited['resume_checkpoint_sha256'],
        'resume_iteration': 2000, 'restored_state': ['student', 'optimizer'],
        'fresh_state': ['online_confusion', 'geometry_selector'],
        'restoration_verified': inherited, 'smoke_restoration_receipt': smoke_receipt,
        'remaining_training_updates': cfg['max_iters'] - 2000,
        'observer_initialization': 'fresh at stage2 iteration2000; no borrowed A/B online state',
        'expected_final_observer_updates': cfg['max_iters'] - 2000,
        'expected_final_observer_seen_images': (cfg['max_iters'] - 2000) * 8,
        'formal_online_coldstart_updates': 100, 'formal_online_ramp_updates': 200,
        'smoke_overrides': SMOKE_OVERRIDES if smoke else {},
        'new_stage1_training': False, 'step0_retrained': False, 'baseline_retrained': False,
        'mechanism': 'Schema4 trusted current-new-source confusion selection; otherwise byte-identical B semantic-protected geometry and optimized KD, added in stage2 to the archived best pipeline',
        'scope': 'One fixed-seed best-pipeline add-on, not teacher causal ablation; new-source policy supported by A development GT128 disagreement diagnostic, not population anchor precision; no seed/grid/new-loss search',
        'development_GT_diagnostic': str(U / 'endpoint_gt_diagnostic.json'),
        'GT_role': 'Development diagnostic only, never an input to online evidence, scores, targets, protected gradients or precise loss weights. Directional disagreement subsets differ from all foreground anchor accuracy.',
        'limits': 'Teacher, student/optimizer trajectory, fresh online history and source domain change together versus B. At most5 new-source rows can select partners, changing pair coverage and mean-eligible gradient allocation. GT128 fixed-prefix disagreement evidence is neither population accuracy nor causal proof. Original4x2 to8x1 preserves global8 but restarts sampler/augmentation RNG. Euclidean component protection is not an AdamW or final-IoU guarantee.',
        'evaluation': 'same8card host;4 asynchronous shards;formal4000/6000/8000 full1449',
    })
    workers, train_child, train_record = [], None, None
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'Signal{signum}')
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    def frozen():
        require(sources(services) == gate['source_sha256']
                and sources(services, BSRC) == gate['parent_source_sha256']
                and digest(RUNNER) == gate['runner_sha256']
                and digest(LAUNCHER) == gate['launcher_sha256'], 'Frozen C source/scripts changed')
    try:
        frozen()
        for rank in range(4):
            cmd = [str(PY), '-B', str(S / 'evaluate_kd.py'), '--run', str(run), '--rank', str(rank), '--shards', '4']
            with open(safe_path(run / f'evaluator_rank{rank}.log'), 'x') as log:
                child = subprocess.Popen(cmd, cwd=R, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            record = {'pid': child.pid, 'starttime': Path(f'/proc/{child.pid}/stat').read_text().split()[21],
                      'command': cmd, 'host': socket.gethostname(), 'utc': now()}
            atomic_json(run / f'evaluator_rank{rank}.process.json', record)
            workers.append((child, record))
        stage = safe_path(run / '10-5/step2')
        stage.mkdir(parents=True, exist_ok=True)
        with open(safe_path(stage / 'launcher.log'), 'x') as log:
            train_child = subprocess.Popen(command, cwd=R, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        train_record = {'pid': train_child.pid, 'starttime': Path(f'/proc/{train_child.pid}/stat').read_text().split()[21],
                        'command': command, 'host': socket.gethostname(), 'start_utc': now(),
                        'teacher': str(TEACHER), 'teacher_sha256': inherited['teacher_sha256']}
        atomic_json(stage / 'launch.json', train_record)
        atomic_json(run / 'status.json', {'status': 'training', 'step': 2, 'utc': now()})
        while train_child.poll() is None:
            frozen()
            require(not any(child.poll() is not None for child, _ in workers),
                    'An evaluator exited before C training completed')
            require(not (run / 'evaluation_failed.json').exists()
                    and not list((run / 'evaluations').glob('*/error_rank*.json')), 'C evaluation failure')
            time.sleep(2)
        rc = train_child.returncode
        atomic_json(stage / 'launch.json', {**train_record, 'returncode': rc, 'end_utc': now()})
        require(rc == 0, f'C stage2 failed:{rc}')
        frozen()
        final = safe_path(stage / 'checkpoints/model_final.pth')
        require(final.is_file(), 'Successful C trainer did not publish final checkpoint')
        atomic_json(stage / 'training_complete.json', {'returncode': 0, 'utc': now(), 'checkpoint_sha256': digest(final)})
        atomic_json(run / 'training_complete.json', {'status': 'complete', 'stages_trained': 1,
                                                     'inherited_stage1': str(OPT), 'utc': now()})
        atomic_json(run / 'status.json', {'status': 'finishing_evaluation', 'utc': now()})
        deadline = time.monotonic() + 600
        while not all(child.poll() is not None for child, _ in workers):
            frozen()
            require(time.monotonic() < deadline, 'C evaluation timeout after training')
            require(all(child.poll() in (None, 0) for child, _ in workers), 'C evaluator failed')
            time.sleep(2)
        for rank, (child, record) in enumerate(workers):
            require(child.returncode == 0, 'C evaluator did not exit0')
            atomic_json(run / f'evaluator_rank{rank}.process.json', {**record, 'returncode': 0, 'end_utc': now()})
        evaluation_complete(run, smoke, services)
        frozen()
        atomic_json(run / 'evaluation_workers_complete.json', {'returncodes': [0] * 4, 'utc': now()})
        atomic_json(run / 'status.json', {'status': 'complete', 'utc': now()})
        print(json.dumps({'status': 'complete', 'run': str(run)}, allow_nan=False), flush=True)
    except BaseException as exc:
        stop_owned([train_child, *[child for child, _ in workers]])
        if train_child is not None and train_record is not None:
            atomic_json(run / '10-5/step2/launch.json', {**train_record, 'returncode': train_child.poll(),
                                                       'end_utc': now(), 'failure': repr(exc)})
        for rank, (child, record) in enumerate(workers):
            atomic_json(run / f'evaluator_rank{rank}.process.json', {**record, 'returncode': child.poll(),
                                                                  'end_utc': now(), 'failure': repr(exc)})
        atomic_json(run / 'failure.json', {'exception': repr(exc), 'utc': now()})
        atomic_json(run / 'status.json', {'status': 'failed', 'utc': now()})
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--check-only', action='store_true', help='Read-only gates/config/checkpoint verification')
    parser.add_argument('--print-command', action='store_true', help='Read-only verification plus concrete train command')
    args = parser.parse_args()
    main(args.smoke, args.check_only, args.print_command)

