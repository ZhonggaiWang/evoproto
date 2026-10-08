"""D: stage2 only, teacher-domain directed pair KD and zero SEP weight."""
from pathlib import Path
import argparse
import ast
import json
import os
import signal
import subprocess
import sys
import time


ROOT = Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto')
EXPERIMENT = ROOT/'experiments/confusion_guided_v1'
SOURCE = EXPERIMENT/'d_pairkd_only/src'
REFERENCE = ROOT/'runs/kd_parallel_v1/formal/b_relational'
PYTHON = ROOT/'.runtime/env/bin/python'
DERIVED_KEYS = ['local_rank', 'step', 'work_dir', 'prev_checkpoint',
                'resume_checkpoint', 'ckpt_dir', 'pred_dir']
PAIR_CONFIGURATION = {'online_confusion': True, 'confusion_momentum': .98,
    'pair_mode': 'pairkd', 'w_pair_sep': 0., 'pair_refresh_interval': 50,
    'pair_min_row_images': 8, 'pair_min_pair_images': 3, 'pair_min_rate': .01,
    'pair_min_updates': 100, 'pair_ramp_updates': 200, 'pair_max_stale_updates': 200}


def initialize():
    os.chdir(ROOT)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT/'experiments/kd_pixel_v2'))
    from run_kd import environment
    sys.path.insert(0, str(SOURCE))
    from kd_runtime import safe_path, atomic_json, now, digest
    return environment, safe_path, atomic_json, now, digest


def source_hashes(digest):
    return {str(path.relative_to(SOURCE)): digest(path) for path in SOURCE.rglob('*.py')
            if '__pycache__' not in path.parts}


def configuration(smoke):
    cfg = json.loads((REFERENCE/'10-5/step2/config.json').read_text())
    for key in DERIVED_KEYS:
        cfg.pop(key, None)
    cfg.update(PAIR_CONFIGURATION)
    cfg.update(spg=1, async_eval=True)
    if cfg['max_iters'] != 8000 or cfg['loss_warmup_iters'] != 2000:
        raise ValueError('Reference stage2 budget/warmup must remain8000/2000')
    if cfg['w_pixel_kd'] != .1 or cfg['kd_temperature'] != 2 or cfg['seed'] != 0:
        raise ValueError('Unexpected reference KD orseed configuration')
    if cfg['w_proto_kd'] or cfg['w_proto_sep'] or cfg['ald'] or cfg['confusion_reweight']:
        raise ValueError('Expected existing prototypeKD/SEP,ALD,legacyreweight disabled')
    if smoke:
        cfg.update(max_iters=2004, eval_iters=2004, log_iters=1, val_limit=16, train_limit=64)
    if cfg['spg']*8 != 8 or cfg['w_pair_sep'] != 0 or cfg['pair_mode'] != 'pairkd':
        raise ValueError('D must preserve globalbatch8 anddisable SEP')
    return cfg


def command_for(run, cfg, teacher, resume):
    script = SOURCE/'scripts/dist_train_voc_seg_neg.py'
    options = set()
    for node in ast.walk(ast.parse(script.read_text())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
            options.update(arg.value for arg in node.args if isinstance(arg, ast.Constant)
                           and isinstance(arg.value, str))
    cmd = [str(PYTHON), '-B', '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8',
           str(script), '--work_dir', str(run), '--step', '2',
           '--prev_checkpoint', str(teacher), '--resume_checkpoint', str(resume)]
    for key, value in cfg.items():
        if value is None:
            continue
        if '--'+key not in options:
            raise ValueError(f'Configuration has no argparse option: {key}')
        if isinstance(value, bool):
            if value:
                cmd.append('--'+key)
            elif key == 'pretrained':
                cmd.append('--no-pretrained')
        elif isinstance(value, list):
            if not value:
                raise ValueError(f'Empty argument list: {key}')
            cmd.extend(['--'+key, *map(str, value)])
        else:
            cmd.extend(['--'+key, str(value)])
    return cmd


def terminate(children):
    for child in children:
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for child in children:
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=15)


def process_start(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split()[21]
    except FileNotFoundError:
        return None


def run_directory(smoke, safe_path):
    return safe_path(ROOT/'runs/confusion_guided_v1'/('pairkd_smoke' if smoke else 'formal')/'d_pairkd_only')


def train(smoke, services):
    import fcntl
    environment, safe_path, atomic_json, now, digest = services
    run = run_directory(smoke, safe_path)
    run.mkdir(parents=True, exist_ok=True)
    lock = open(safe_path(run/'train.lock'), 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (run/'study.json').exists():
        raise RuntimeError('Refuse duplicate D training; existing study retained')
    cfg = configuration(smoke)
    teacher = safe_path(REFERENCE/'10-5/step1/checkpoints/model_final.pth')
    resume = safe_path(REFERENCE/'10-5/step2/checkpoints/model_iter_2000.pth')
    code = source_hashes(digest)
    protocol = json.loads((EXPERIMENT/'pairkd_only_protocol.json').read_text())
    if not code or code != protocol['source_sha256']:
        raise RuntimeError('D source differs from prepared protocol manifest')
    cmd = command_for(run, cfg, teacher, resume)
    env = environment('8card')
    env['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    os.environ.update(env)
    import torch
    checkpoint = torch.load(resume, map_location='cpu', weights_only=True, mmap=True)
    if int(checkpoint['iteration']) != 2000 or not checkpoint.get('optimizer_state'):
        raise ValueError('D requires full student ANDoptimizer at stage2iteration2000')
    if checkpoint.get('online_confusion_state') is not None or checkpoint.get('pair_selector_state') is not None:
        raise ValueError('OptimizedKD warmup must have no observer/selector; D initializes its own fresh state')
    restored = {'iteration': int(checkpoint['iteration']), 'model_tensors': len(checkpoint['model_state']),
                'optimizer_parameter_groups': len(checkpoint['optimizer_state']['param_groups']),
                'observer_state_present': False, 'selector_state_present': False}
    del checkpoint
    atomic_json(run/'study.json', {'arm': 'd_pairkd_only', 'utc': now(), 'created_utc': now(),
        'gpus': list(range(8)), 'global_batch': 8, 'common_config': cfg,
        'source_sha256': code, 'runner_sha256': digest(Path(__file__)),
        'inherited_stage1': str(REFERENCE/'10-5/step1'), 'teacher': str(teacher),
        'teacher_sha256': digest(teacher), 'shared_stage2_warmup': str(resume),
        'shared_stage2_warmup_sha256': digest(resume), 'start_iteration': 2000,
        'remaining_training_updates': cfg['max_iters']-2000, 'smoke': smoke,
        'observer_initialization': 'fresh at stage2 iteration2000;6000updates' if not smoke
                                   else 'fresh at stage2 iteration2000;4updates smoke only',
        'selector_schema': 2, 'distillation_class_limit': 16,
        'resume_checkpoint_verified': restored, 'protocol': str(EXPERIMENT/'pairkd_only_protocol.json'),
        'scope': 'Only old-old directedKD; zeroSEP. Reuse originaloptimizedKD stage1 andstage2warmup; no baseline/step0/stage1/warmuptraining.',
        'resume_caveat': 'Student ANDoptimizer restored; sampler restarts under8GPUs versus4, not exact RNGcontinuation.',
        'smoke_caveat': 'Fresh observer remains below100updates; fourstep smoke doesnot establish pairKDactivation.'})
    stage = safe_path(run/'10-5/step2')
    stage.mkdir(parents=True, exist_ok=True)
    child = None
    record = None
    try:
        if source_hashes(digest) != code:
            raise RuntimeError('D source changed before launch')
        with open(safe_path(stage/'launcher.log'), 'x') as handle:
            child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        def interrupt(signum, frame):
            raise KeyboardInterrupt(f'Signal{signum}')
        signal.signal(signal.SIGTERM, interrupt)
        signal.signal(signal.SIGINT, interrupt)
        record = {'pid': child.pid, 'starttime': process_start(child.pid), 'command': cmd,
                  'gpus': list(range(8)), 'utc': now()}
        atomic_json(stage/'launch.json', record)
        while child.poll() is None:
            if (run/'evaluation_failed.json').exists():
                raise RuntimeError('Stop D training because its evaluation workers failed')
            time.sleep(2)
        returncode = child.wait()
        atomic_json(stage/'launch.json', {**record, 'returncode': returncode, 'end_utc': now()})
        if returncode:
            raise RuntimeError(f'D training exited{returncode}')
        if source_hashes(digest) != code:
            raise RuntimeError('Frozen D source changed during training')
        final = safe_path(stage/'checkpoints/model_final.pth')
        if not final.exists():
            raise RuntimeError('D final checkpoint absent after successful launcher')
        atomic_json(stage/'training_complete.json', {'utc': now(), 'returncode': 0,
                    'checkpoint_sha256': digest(final)})
        atomic_json(run/'training_complete.json', {'utc': now(), 'stages': 1, 'status': 'training_complete',
                    'inherited_stage1': str(REFERENCE/'10-5/step1')})
    except BaseException as exc:
        if child is not None:
            terminate([child])
            if record is not None:
                atomic_json(stage/'launch.json', {**record, 'returncode': child.poll(),
                            'end_utc': now(), 'failure': repr(exc)})
        atomic_json(run/'training_failed.json', {'utc': now(), 'error': repr(exc)})
        raise
    finally:
        lock.close()


def evaluate(smoke, services):
    import fcntl
    environment, safe_path, atomic_json, now, digest = services
    run = run_directory(smoke, safe_path)
    run.mkdir(parents=True, exist_ok=True)
    lock = open(safe_path(run/'evaluation.lock'), 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = environment('8card')
    env['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    children = []
    failure = None
    try:
        def interrupt(signum, frame):
            raise KeyboardInterrupt(f'Signal{signum}')
        signal.signal(signal.SIGTERM, interrupt)
        signal.signal(signal.SIGINT, interrupt)
        for rank in range(4):
            cmd = [str(PYTHON), '-B', str(SOURCE/'evaluate_kd.py'), '--run', str(run),
                   '--rank', str(rank), '--shards', '4']
            with open(safe_path(run/f'evaluator_rank{rank}.log'), 'a') as handle:
                child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle,
                                         stderr=subprocess.STDOUT, start_new_session=True)
            children.append(child)
            atomic_json(run/f'evaluator_rank{rank}.process.json', {
                'pid': child.pid, 'starttime': process_start(child.pid), 'command': cmd, 'utc': now()})
        while True:
            codes = [child.poll() for child in children]
            if (run/'training_failed.json').exists():
                raise RuntimeError('D training failed')
            if any(code not in (None, 0) for code in codes):
                raise RuntimeError(f'D evaluator failure{codes}')
            if all(code == 0 for code in codes):
                if not (run/'training_complete.json').exists():
                    raise RuntimeError('Evaluators exited before successful training completion')
                expected = ['step2_iter2004'] if smoke else ['step2_iter4000', 'step2_iter6000', 'step2_iter8000']
                for stem in expected:
                    if not (run/f'eval_queue/{stem}.json').exists() or not (run/f'evaluations/{stem}/result.json').exists():
                        raise RuntimeError(f'Missing completed evaluation{stem}')
                break
            time.sleep(2)
    except BaseException as exc:
        failure = repr(exc)
        atomic_json(run/'evaluation_failed.json', {'utc': now(), 'error': failure})
        terminate(children)
        raise
    finally:
        atomic_json(run/'evaluation_workers_complete.json', {'utc': now(),
            'returncodes': [child.poll() for child in children], 'status': 'failed' if failure else 'complete',
            'failure': failure})
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('role', choices=['train', 'eval'])
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--print-command', action='store_true', help='Validate/print trainingargv without launching')
    args = parser.parse_args()
    services = initialize()
    if args.print_command:
        run = run_directory(args.smoke, services[1])
        cfg = configuration(args.smoke)
        print(json.dumps({'command': command_for(run, cfg,
            REFERENCE/'10-5/step1/checkpoints/model_final.pth',
            REFERENCE/'10-5/step2/checkpoints/model_iter_2000.pth'), 'configuration': cfg}))
        return
    if args.role == 'train':
        try:
            train(args.smoke, services)
        except BaseException as exc:
            run = run_directory(args.smoke, services[1])
            if not isinstance(exc, BlockingIOError) and not (run/'study.json').exists():
                services[2](run/'training_failed.json', {'utc': services[3](), 'error': repr(exc)})
            raise
    else:
        evaluate(args.smoke, services)


if __name__ == '__main__':
    main()
