"""Observe an existing ALD study without starting or controlling training.

The explicit PID/starttime pair anchors one runner. Physical GPU readings are
shared totals, never an attribution of memory or utilization to this study.
"""
import argparse
import csv
import fcntl
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('_evoproto_ald_monitor_base',
                                             ROOT / 'tools/run_ald_study.py')
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)


def checked(path):
    path = Path(path)
    return base.writable_path(path if path.is_absolute() else ROOT / path)


def identity(pid):
    try:
        raw = Path(f'/proc/{pid}/stat').read_text()
    except FileNotFoundError:
        return None
    fields = raw[raw.rfind(')') + 2:].split()
    return {'pid': pid, 'state': fields[0], 'starttime': int(fields[19])}


def finite(value):
    if isinstance(value, dict):
        return {key: finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite(item) for item in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def read_json(path):
    path = checked(path)
    if not path.is_file():
        return None
    try:
        return finite(json.loads(path.read_text()))
    except json.JSONDecodeError:
        return {'read_error': 'JSON is incomplete or malformed at this observation'}


def tail(path):
    path = checked(path)
    if not path.is_file():
        return ''
    with path.open('rb') as handle:
        size = path.stat().st_size
        start = max(0, size - 262144)
        handle.seek(start)
        data = handle.read().decode(errors='replace')
    return data.split('\n', 1)[1] if start and '\n' in data else data


def last_record(path):
    for line in reversed(tail(path).splitlines()):
        if not line.strip():
            continue
        try:
            return finite(json.loads(line))
        except json.JSONDecodeError:
            continue  # A writer may not yet have finished the final line.
    return {}


def stage_snapshot(study, arm, step, maximum):
    directory = checked(study / arm / '10-5' / f'step{step}')
    log = tail(directory / 'train.log')
    iterations = [int(value) for value in re.findall(r'\bIter:\s*(\d+)\b', log)]
    coverage = last_record(directory / 'ald_metrics.jsonl')
    metrics = last_record(directory / 'metrics.jsonl')
    complete = read_json(directory / 'completion.json') or {}
    process = read_json(directory / 'process.json') or {}
    checkpoint = checked(directory / 'checkpoints/model_final.pth').is_file()
    observed_complete = (complete.get('returncode') == 0 and process.get('returncode') == 0
                         and complete.get('iteration') == maximum
                         and metrics.get('iteration') == maximum and checkpoint)
    progress = max(iterations + [coverage.get('iteration', 0), metrics.get('iteration', 0)])
    return {'arm': arm, 'step': step, 'latest_observed_iteration': progress,
            'latest_log_iteration': max(iterations, default=0),
            'latest_coverage_iteration': coverage.get('iteration'),
            'latest_evaluation_iteration': metrics.get('iteration'),
            'latest_metrics': {key: metrics.get(key) for key in
                               ('all_miou', 'old_miou', 'new_miou', 'h_miou')},
            'segmentation_loss_active': coverage.get('segmentation_loss_active'),
            'completion_returncode': complete.get('returncode'),
            'launcher_returncode': process.get('returncode'),
            'final_checkpoint_present': checkpoint, 'observed_complete': observed_complete,
            'status': 'completed_recorded' if observed_complete else
                      'started' if directory.exists() else 'not_started',
            'error_markers': [marker for marker in ('Traceback (most recent call last)',
                                                   'CUDA out of memory')
                              if marker in tail(study / arm / f'launcher-step{step}.log')]}


def physical_gpus(mapping):
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu',
         '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=15)
    selected = set(mapping.values())
    rows = []
    for fields in csv.reader(io.StringIO(result.stdout)):
        index, total, used, free, utilization = (value.strip() for value in fields)
        if index in selected:
            rows.append({'gpu': index, 'memory_total_mib': int(total),
                         'memory_used_mib': int(used), 'memory_free_mib': int(free),
                         'utilization_gpu_percent': int(utilization)})
    return {'scope': 'physical GPU totals include all shared jobs; '
                     'not this study\'s allocation or utilization', 'gpus': rows}


def open_owned(path, flags):
    path = checked(path)
    fd = os.open(path, flags | os.O_NOFOLLOW, 0o600)
    try:
        checked(path)
        stamp, current = os.fstat(fd), path.stat()
        if stamp.st_nlink != 1 or (stamp.st_dev, stamp.st_ino) != (current.st_dev, current.st_ino):
            raise RuntimeError(f'Monitor output identity changed: {path}')
    except BaseException:
        os.close(fd)
        raise
    return fd


def monitor(args):
    study = checked(args.study)
    if not study.is_dir() or not checked(study / 'study.json').is_file():
        raise RuntimeError('Study must already exist with study.json; monitor never creates it')
    manifest = base.read_json(study / 'study.json')
    variants, mapping = manifest['variants'], manifest['gpu_mapping']
    if (not variants or any(not isinstance(arm, str) or Path(arm).name != arm
                            or arm in ('.', '..') for arm in variants)):
        raise RuntimeError('Invalid study variants')
    maximum = manifest['incremental_iters']
    log = checked(args.log or ROOT / '.runtime' /
                  f'ald-monitor-{study.name}-pid{args.pid}-start{args.starttime}.jsonl')
    if not log.is_relative_to(ROOT / '.runtime'):
        raise RuntimeError('Monitor log must stay under this project\'s .runtime')
    base.make_directory(log.parent)
    lock_fd = open_owned(log.with_suffix(log.suffix + '.lock'), os.O_RDWR | os.O_CREAT)
    log_fd = None
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        log_fd = open_owned(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
        while True:
            current = identity(args.pid)
            terminal = ('pid_missing' if current is None else
                        'pid_reused' if current['starttime'] != args.starttime else
                        'pid_terminated' if current['state'] in ('Z', 'X', 'x') else None)
            stages = [stage_snapshot(study, arm, step, maximum)
                      for arm in variants for step in (1, 2)]
            complete_count = sum(stage['observed_complete'] for stage in stages)
            row = {'utc': base.utc_now(), 'event': 'terminal' if terminal else 'sample',
                   'study': str(study), 'runner_pid': args.pid,
                   'runner_starttime': args.starttime, 'runner_identity': current,
                   'terminal_reason': terminal, 'interval_seconds': args.interval,
                   'study_manifest_sha256': base.file_sha256(study / 'study.json'),
                   'gpu_mapping': mapping, 'stages': stages,
                   'completed_stage_count': complete_count, 'expected_stage_count': len(stages),
                   'results_present': checked(study / 'results.json').is_file(),
                   'completion_scope': 'observed final artifacts and exit-0 records; '
                                       'runner provenance verification remains authoritative',
                   'physical_gpu': None}
            if not terminal:
                try:
                    row['physical_gpu'] = physical_gpus(mapping)
                except (OSError, subprocess.SubprocessError, ValueError) as exc:
                    row['physical_gpu'] = {'read_error': str(exc),
                                           'scope': 'physical totals of all shared jobs'}
            checked(log)
            if os.fstat(log_fd).st_nlink != 1:
                raise RuntimeError('Monitor log acquired another hard link')
            encoded = (json.dumps(row, sort_keys=True, allow_nan=False) + '\n').encode()
            while encoded:
                encoded = encoded[os.write(log_fd, encoded):]
            os.fsync(log_fd)
            progress = ' '.join(f"{stage['arm']}/s{stage['step']}="
                                f"{stage['latest_observed_iteration']}" for stage in stages)
            print(f"[{row['utc']}] {row['event']} {complete_count}/{len(stages)} {progress}"
                  + (f' {terminal}' if terminal else ''), flush=True)
            if terminal:
                return 0 if (terminal != 'pid_reused' and complete_count == len(stages)
                             and row['results_present']) else 1
            time.sleep(args.interval)
    finally:
        if log_fd is not None:
            os.close(log_fd)
        os.close(lock_fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True, help='Existing study directory inside the project')
    parser.add_argument('--pid', required=True, type=int)
    parser.add_argument('--starttime', required=True, type=int, help='Linux /proc PID stat field 22')
    parser.add_argument('--interval', type=int, default=45)
    parser.add_argument('--log', help='Own JSONL output under the project .runtime directory')
    args = parser.parse_args()
    if min(args.pid, args.starttime, args.interval) <= 0 or args.interval > 60:
        parser.error('PID/starttime/interval must be positive and interval must be at most 60 seconds')
    try:
        return monitor(args)
    except Exception as exc:
        print(f'Monitor failed: {exc}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
