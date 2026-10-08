"""Run an isolated two-arm current-class gate study; never alter the primary run.

Both arms use legacy fusion and KD/SEP weights zero. Each GPU independently
trains step 1 then step 2 from the original seed-0 initialization. Formal work
requires the primary arm on each requested GPU to have finished both stages;
smoke may share a GPU only after an actual free-memory check.
"""
import argparse
import concurrent.futures
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / 'runs/ald_fusion_v1'
ISOLATED = ROOT / 'experiments/ald_gate_new_v1'
ENTRY = ISOLATED / 'train.py'
ARMS = ('legacy', 'new_fallback')
EXTRA_SOURCES = (ENTRY, ISOLATED / 'trainer.py', ISOLATED / 'ald_stats.py',
                 ROOT / 'tools/ald_gate_policies.py', Path(__file__).resolve())

# A separate module namespace keeps all overrides local to this runner process.
_spec = importlib.util.spec_from_file_location('_evoproto_gate_runner_base',
                                             ROOT / 'tools/run_ald_study.py')
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
_original_writable = base.writable_path


def checked_path(path):
    path = _original_writable(path)
    if path.exists() and not (path.is_dir() or path.is_file()):
        raise RuntimeError(f'Unsupported filesystem node: {path}')
    return path


base.writable_path = checked_path


def read_file(path):
    path = checked_path(path)
    if not path.is_file():
        raise RuntimeError(f'Missing ordinary project file: {path}')
    return path


def primary_manifest():
    return base.read_json(read_file(PRIMARY / 'study.json'))


def gate_fingerprint():
    """Anchor the entire primary frozen source set, then add isolated sources."""
    primary = primary_manifest()
    old = primary['code_sha256']
    for relative, expected in old.items():
        path = read_file(ROOT / relative)
        if base.file_sha256(path) != expected:
            raise RuntimeError(f'Primary frozen source changed: {relative}')
    files = {ROOT / relative for relative in old} | set(EXTRA_SOURCES)
    return {str(path.relative_to(ROOT)): base.file_sha256(read_file(path))
            for path in sorted(files)}


base.ENTRY = ENTRY
base.VARIANTS = ARMS
base.source_fingerprint = gate_fingerprint


def proc_fields(pid):
    """Read numeric state/group fields only, without exposing another job's args."""
    try:
        data = Path(f'/proc/{pid}/stat').read_text()
    except FileNotFoundError:
        return None
    fields = data[data.rfind(')') + 2:].split()
    return {'state': fields[0], 'ppid': int(fields[1]), 'group': int(fields[2])}


def verify_primary_gpu_complete(gpu):
    """Verify raw final outputs and exit-0 provenance, not lock/status alone."""
    primary = primary_manifest()
    matches = [arm for arm, value in primary['gpu_mapping'].items() if value == gpu]
    if len(matches) != 1 or primary.get('smoke') or primary.get('incremental_iters') != 8000:
        raise RuntimeError(f'Cannot establish formal primary arm for GPU {gpu}')
    arm = matches[0]
    previous = Path(primary['shared_initialization']['checkpoint'])
    validator = object.__new__(base.Study)
    validator.args = SimpleNamespace(smoke=False)
    validator.iterations = 8000
    stages, groups = [], set()
    for step in (1, 2):
        directory = PRIMARY / arm / '10-5' / f'step{step}'
        config = base.read_json(read_file(directory / 'config.json'))
        inputs = base.read_json(read_file(directory / 'inputs.json'))
        expected = dict(primary['common_training_config'], step=step,
                        ald_mode=arm, ald=arm != 'off', prev_checkpoint=str(previous),
                        work_dir=str(directory), ckpt_dir=str(directory / 'checkpoints'),
                        pred_dir=str(directory / 'predictions'))
        if config != expected or inputs.get('expected_config') != expected:
            raise RuntimeError(f'Primary configuration mismatch: {directory}')
        if (inputs.get('previous_checkpoint') != str(previous)
                or inputs.get('previous_sha256') != base.file_sha256(read_file(previous))
                or inputs.get('code_sha256') != primary['code_sha256']
                or inputs.get('dataset_lists_sha256') != primary['dataset_lists_sha256']
                or inputs.get('gpu') != gpu):
            raise RuntimeError(f'Primary input provenance mismatch: {directory}')
        metrics = base.Study.final_metrics(validator, directory, expected)
        coverage = base.Study.final_ald_metrics(validator, directory, expected)
        complete = base.read_json(read_file(directory / 'completion.json'))
        process = base.read_json(read_file(directory / 'process.json'))
        final = read_file(directory / 'checkpoints/model_final.pth')
        digests = {'inputs_sha256': base.file_sha256(read_file(directory / 'inputs.json')),
                   'config_sha256': base.file_sha256(read_file(directory / 'config.json')),
                   'checkpoint_sha256': base.file_sha256(final),
                   'metrics_sha256': base.file_sha256(read_file(directory / 'metrics.jsonl')),
                   'ald_metrics_sha256': base.file_sha256(read_file(directory / 'ald_metrics.jsonl'))}
        if (complete.get('returncode') != 0 or complete.get('iteration') != 8000
                or process.get('returncode') != 0
                or process.get('command') != inputs.get('command')
                or complete.get('command') != inputs.get('command')
                or complete.get('config') != config
                or complete.get('final_metrics') != metrics
                or complete.get('final_ald_metrics') != coverage
                or any(complete.get(key) != value for key, value in digests.items())):
            raise RuntimeError(f'Primary stage is not verified exit-0 complete: {directory}')
        pid = process.get('pid')
        if type(pid) is not int or pid <= 0:
            raise RuntimeError(f'Missing primary process identity: {directory}')
        live = proc_fields(pid)
        if live and live['state'] not in ('Z', 'X') and live['group'] == pid:
            raise RuntimeError(f'Primary launcher is still live on GPU {gpu}')
        groups.add(pid)
        stages.append({'stage': str(directory.relative_to(ROOT)),
                       'completion_sha256': base.file_sha256(directory / 'completion.json'),
                       'process_sha256': base.file_sha256(directory / 'process.json')})
        previous = final
    return {'primary_arm': arm, 'verified_stages': stages}, groups


def gpu_admission(gpus, *, smoke, minimum_free):
    """Query physical memory; check primary process groups without controlling jobs."""
    query = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.used,memory.free',
                            '--format=csv,noheader,nounits'], check=True,
                           capture_output=True, text=True, timeout=15)
    cards = {}
    uuid_to_gpu = {}
    for row in csv.reader(io.StringIO(query.stdout)):
        index, uuid, total, used, free = (value.strip() for value in row)
        uuid_to_gpu[uuid] = index
        cards[index] = {'gpu': index, 'memory_total_mib': int(total),
                        'memory_used_mib': int(used), 'memory_free_mib': int(free),
                        'shared_physical_gpu': int(used) > 256}
    primary_rows, groups = {}, {}
    for gpu in gpus:
        if gpu not in cards or cards[gpu]['memory_free_mib'] < minimum_free:
            raise RuntimeError(f'GPU {gpu} lacks the required {minimum_free} MiB free memory')
        if not smoke:
            primary_rows[gpu], groups[gpu] = verify_primary_gpu_complete(gpu)
    if not smoke:
        apps = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                               '--format=csv,noheader,nounits'], check=True,
                              capture_output=True, text=True, timeout=15)
        for row in csv.reader(io.StringIO(apps.stdout)):
            if not row:
                continue
            uuid, pid = (value.strip() for value in row)
            gpu = uuid_to_gpu.get(uuid)
            if gpu in groups:
                state = proc_fields(int(pid))
                if state and state['group'] in groups[gpu] and state['state'] not in ('Z', 'X'):
                    raise RuntimeError(f'Primary GPU process remains active on GPU {gpu}')
    return {'observed_utc': base.utc_now(), 'smoke_shared_allowed': smoke,
            'minimum_free_mib': minimum_free,
            'gpu_usage_scope': 'physical totals include any pre-existing jobs',
            'gpus': [cards[gpu] for gpu in gpus], 'primary_exit0_evidence': primary_rows}


class GateStudy(base.Study):
    def __init__(self, args, output, source_config, source, environment):
        super().__init__(args, output, source_config, source, environment)
        self.config.update(study='isolated current-class gate fallback', expected_stages=4,
                           fusion_mode='legacy', primary_study=str(PRIMARY),
                           primary_study_sha256=base.file_sha256(read_file(PRIMARY / 'study.json')),
                           minimum_free_mib=args.min_free_mib)

    def assert_frozen(self):
        super().assert_frozen()
        if base.file_sha256(read_file(PRIMARY / 'study.json')) != self.config['primary_study_sha256']:
            raise RuntimeError('Primary immutable study manifest changed')

    def stage_plan(self, variant, step, gpu, previous):
        directory = self.output / variant / '10-5' / f'step{step}'
        expected = dict(self.common, step=step, ald_mode='legacy', ald=True,
                        ald_gate_policy=variant, prev_checkpoint=str(previous.resolve()),
                        work_dir=str(directory), ckpt_dir=str(directory / 'checkpoints'),
                        pred_dir=str(directory / 'predictions'))
        command = [sys.executable, '-B', '-m', 'torch.distributed.run', '--standalone',
                   '--nnodes=1', '--nproc_per_node=1', str(ENTRY),
                   '--work_dir', str(self.output / variant), '--step', str(step),
                   '--prev_checkpoint', str(previous.resolve()), '--ald_mode', 'legacy',
                   '--ald_gate_policy', variant]
        for key, value in self.common.items():
            if key in ('confusion_reweight', 'save_ckpt', 'pretrained'):
                if value:
                    command.append(f'--{key}')
                elif key == 'pretrained':
                    command.append('--no-pretrained')
            elif isinstance(value, (tuple, list)):
                command.extend([f'--{key}', *(str(item) for item in value)])
            else:
                command.extend([f'--{key}', str(value)])
        inputs = {'schema_version': 1, 'previous_checkpoint': str(previous.resolve()),
                  'previous_sha256': base.file_sha256(read_file(previous)),
                  'pretrained_sha256': self.source['pretrained_sha256'],
                  'shared_initialization': self.source, 'code_sha256': self.code,
                  'dataset_lists_sha256': self.data_hashes, 'gpu': gpu,
                  'command': command, 'expected_config': expected}
        return directory, expected, command, inputs

    def final_metrics(self, directory, expected):
        final = super().final_metrics(directory, expected)
        if any(row.get('ald_gate_policy') != expected['ald_gate_policy']
               for row in base.read_metrics(read_file(directory / 'metrics.jsonl'))):
            raise RuntimeError(f'Metric gate-policy mismatch: {directory}')
        return final

    def final_ald_metrics(self, directory, expected):
        final = super().final_ald_metrics(directory, expected)
        for row in base.read_metrics(read_file(directory / 'ald_metrics.jsonl')):
            extra = ('new_rescue_images', 'new_rescued_class_occurrences',
                     'direct_threshold_new_classes')
            valid = (row.get('ald_gate_policy') == expected['ald_gate_policy']
                     and all(type(row.get(key)) is int and row[key] >= 0 for key in extra))
            if valid:
                rescued = row['new_rescue_images']
                direct = row['direct_threshold_new_classes']
                valid = (rescued == row['new_rescued_class_occurrences'] <= row['batch_images']
                         and direct <= row['positive_new_classes']
                         and row['positive_new_classes'] == row['gate_rejected_new_classes']
                         + direct + row['fallback_new_images'] + rescued
                         and (expected['ald_gate_policy'] != 'legacy' or rescued == 0))
            if not valid:
                raise RuntimeError(f'Invalid gate rescue counts at {row.get("iteration")}: {directory}')
        return final

    def record_admission(self, phase, gpus):
        sample = gpu_admission(gpus, smoke=self.args.smoke, minimum_free=self.args.min_free_mib)
        sample['phase'] = phase
        path = self.output / 'resource_admission.json'
        record = base.read_json(read_file(path)) if path.exists() else {'attempts': []}
        record['attempts'].append(sample)
        base.write_json(path, record, replace=path.exists())

    def train(self, variant, step, gpu, previous):
        directory = self.output / variant / '10-5' / f'step{step}'
        if not (directory / 'checkpoints/model_final.pth').is_file():
            with self.process_lock:
                self.record_admission(f'{variant}/step{step}', [gpu])
        return super().train(variant, step, gpu, previous)

    def record_failure(self, error):
        if not self.accepted_manifest:
            return
        path = self.output / 'failures.json'
        previous = base.read_json(read_file(path)) if path.exists() else {'attempts': []}
        previous['attempts'].append({'time_utc': base.utc_now(), 'pid': os.getpid(),
                                    'error': str(error), 'type': type(error).__name__})
        base.write_json(path, previous, replace=path.exists())
        base.write_json(self.output / 'status.json', {'status': 'failed',
                        'time_utc': base.utc_now(), 'expected_stages': 4,
                        'error': str(error)}, replace=True)

    def run(self):
        self.admit_output()
        self.record_admission('before_any_launch', self.args.gpus)
        base.write_json(self.output / 'status.json', {'status': 'in_progress',
                        'time_utc': base.utc_now(), 'expected_stages': 4}, replace=True)
        errors = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            jobs = {pool.submit(self.run_variant, arm, gpu): arm
                    for arm, gpu in zip(ARMS, self.args.gpus)}
            try:
                for future in concurrent.futures.as_completed(jobs):
                    try:
                        print('Finished arm:', future.result(), flush=True)
                    except Exception as error:
                        errors.append(f'{jobs[future]}: {error}')
                        self.stop_owned_processes()
                if errors:
                    raise RuntimeError('; '.join(errors))
            except BaseException:
                self.stop_owned_processes()
                raise
        self.assert_frozen()
        results = {}
        for arm, gpu in zip(ARMS, self.args.gpus):
            results[arm] = {}
            previous = Path(self.source['checkpoint'])
            for step in (1, 2):
                directory, expected, _, inputs = self.stage_plan(arm, step, gpu, previous)
                previous = self.reuse_completed(directory, expected, inputs)
                if previous is None:
                    raise RuntimeError(f'Missing completed gate stage: {directory}')
                results[arm][str(step)] = self.final_metrics(directory, expected)
        path = self.output / 'results.json'
        if path.exists():
            if base.read_json(read_file(path)) != results:
                raise RuntimeError('Existing gate results differ from verified final records')
        else:
            base.write_json(path, results)
        base.write_json(self.output / 'status.json', {'status': 'complete',
                        'time_utc': base.utc_now(), 'completed_stages': 4,
                        'expected_stages': 4, 'all_stage_returncodes': 0,
                        'results_sha256': base.file_sha256(path)}, replace=True)
        print('Gate study complete:', path, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=None)
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--min_free_mib', type=int, default=20000,
                        help='minimum observed free GPU memory before each new stage')
    args = parser.parse_args()
    args.gpus = [value.strip() for value in args.gpus.split(',')]
    if (len(args.gpus) != 2 or len(set(args.gpus)) != 2
            or not set(args.gpus).issubset({'0', '1', '2', '3'})):
        parser.error('Choose two distinct authorized GPU IDs from 0,1,2,3')
    if args.workers < 1 or args.seed < 0 or args.min_free_mib < 20000:
        parser.error('workers > 0, seed >= 0 and min_free_mib >= 20000 are required')
    output = ROOT / (args.output or ('runs/ald_gate_new_v1_smoke' if args.smoke
                                     else 'runs/ald_gate_new_v1'))
    if not output.is_relative_to(ROOT / 'runs') or output.resolve() != output:
        parser.error('output must be a canonical path under this project/runs')
    for protected in (PRIMARY, base.BASELINE):
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            parser.error('output must be separate from both preserved source studies')
    base.make_directory(output)
    with base.runner_lock(output):
        source_config, source = base.verify_shared_initialization()
        study = GateStudy(args, output, source_config, source, base.private_environment())
        old_handlers = {}

        def handle_signal(signum, _frame):
            study.cancel.set()
            study.signal_owned_processes(signal.SIGTERM)
            raise KeyboardInterrupt(f'Gate runner received signal {signum}')

        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.signal(signum, handle_signal)
        try:
            study.run()
        except BaseException as error:
            study.stop_owned_processes()
            study.record_failure(error)
            raise
        finally:
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)


if __name__ == '__main__':
    main()
