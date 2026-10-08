"""Run four matched ALD variants from the completed fixed baseline's step 0.

Use the project's private Python environment. Formal runs train eight 8k stages;
``--smoke`` trains eight 4-iteration stages in a separate output directory. Both
reuse the same original seed-0 step-0 checkpoint without copying or modifying it.
``--seed`` changes only incremental training; it does not make a new step 0.
An incomplete stage is preserved and refused, rather than silently restarted.
"""

import argparse
import concurrent.futures
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import threading
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
AUTHORIZED_ROOT = Path('/ML-vePFS/infra_rd/kun/others/wzg').resolve()
ENTRY = ROOT / 'scripts/dist_train_voc_seg_neg.py'
BASELINE = ROOT / 'runs/fixed_baseline_v1'
RUNTIME = ROOT / '.runtime'
PRETRAINED = ROOT / 'pretrained/checkpoints/jx_vit_base_p16_224-80ecf9dd.pth'
PRETRAINED_SHA256 = '80ecf9dd5e3a58895e959af554c5666c4e7b4da4410de4f1f2b0025e93435d8c'
VARIANTS = ('off', 'legacy', 'preserve_rejected', 'preserve_background')
OLD_VARIANTS = ('base', 'kd', 'sep', 'kd_sep')
_MOUNTS = None
_HASH_CACHE = {}
_HASH_LOCK = threading.Lock()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def reject_json_constant(value):
    raise ValueError(f'Nonfinite JSON number: {value}')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'),
                      parse_constant=reject_json_constant)


def writable_path(path):
    """Reject links and nested mounts before any project-owned output write."""
    global _MOUNTS
    path = Path(path)
    if not ROOT.is_relative_to(AUTHORIZED_ROOT):
        raise RuntimeError(f'Project is outside the authorized workspace: {ROOT}')
    if not path.is_absolute() or not path.is_relative_to(ROOT):
        raise RuntimeError(f'Output is outside this project: {path}')
    if not path.resolve().is_relative_to(ROOT):
        raise RuntimeError(f'Output resolves outside this project: {path}')
    for candidate in (path, *path.parents):
        if candidate.is_symlink():
            raise RuntimeError(f'Writable path contains a symlink: {candidate}')
        if candidate == ROOT:
            break
    if path.exists():
        info = path.stat()
        if stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
            raise RuntimeError(f'Refusing to write a hard-linked file: {path}')
    if _MOUNTS is None:
        _MOUNTS = []
        for line in Path('/proc/self/mountinfo').read_text().splitlines():
            mount = re.sub(r'\\([0-7]{3})',
                           lambda match: chr(int(match.group(1), 8)), line.split()[4])
            mount = Path(mount)
            if (mount.is_relative_to(AUTHORIZED_ROOT)
                    and (mount.is_relative_to(ROOT) or ROOT.is_relative_to(mount))):
                _MOUNTS.append(mount)
    if any(path == mount or path.is_relative_to(mount) for mount in _MOUNTS):
        raise RuntimeError(f'Output is on a nested mount: {path}')
    return path


def make_directory(path):
    writable_path(path).mkdir(parents=True, exist_ok=True)
    writable_path(path)


def write_json(path, data, *, replace=False):
    path = writable_path(path)
    if path.exists() and not replace:
        raise RuntimeError(f'Refusing to overwrite an existing manifest: {path}')
    temporary = writable_path(path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp'))
    with temporary.open('x', encoding='utf-8') as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    writable_path(path)
    os.replace(temporary, path)


def file_sha256(path):
    """Cache only while inode, size and nanosecond timestamps remain unchanged."""
    path = Path(path)
    before = path.stat()
    stamp = (before.st_dev, before.st_ino, before.st_size,
             before.st_mtime_ns, before.st_ctime_ns)
    with _HASH_LOCK:
        cached = _HASH_CACHE.get(str(path))
        if cached and cached[0] == stamp:
            return cached[1]
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    after = path.stat()
    if stamp != (after.st_dev, after.st_ino, after.st_size,
                 after.st_mtime_ns, after.st_ctime_ns):
        raise RuntimeError(f'File changed while hashing: {path}')
    value = digest.hexdigest()
    with _HASH_LOCK:
        _HASH_CACHE[str(path)] = (stamp, value)
    return value


def source_fingerprint():
    required = (ENTRY, Path(__file__).resolve(), ROOT / 'tasks.py',
                ROOT / 'utils/ald.py', ROOT / 'utils/ald_stats.py',
                ROOT / 'utils/camutils.py')
    for path in required:
        if not path.is_file():
            raise RuntimeError(f'Missing required ALD source: {path}')
    paths = set(required)
    for directory in ('model', 'continual', 'datasets', 'utils'):
        paths.update((ROOT / directory).rglob('*.py'))
    result = {}
    for path in sorted(paths):
        if not path.resolve().is_relative_to(ROOT):
            raise RuntimeError(f'Core source resolves outside this project: {path}')
        result[str(path.relative_to(ROOT))] = file_sha256(path)
    return result


def list_fingerprint(directory):
    paths = sorted(path for path in directory.rglob('*')
                   if path.is_file() and path.suffix in ('.txt', '.npy'))
    if not paths:
        raise RuntimeError(f'Missing VOC split/label lists: {directory}')
    return {str(path.resolve()): file_sha256(path) for path in paths}


def verify_shared_initialization():
    """Match the actual checkpoint against provenance saved by all old variants."""
    baseline_study = read_json(BASELINE / 'study.json')
    stage = BASELINE / 'shared/10-5/step0'
    checkpoint = stage / 'checkpoints/model_final.pth'
    config = read_json(stage / 'config.json')
    inputs = read_json(stage / 'inputs.json')
    results = read_json(BASELINE / 'results.json')
    expected_study = {'task': '10-5', 'seed': 0, 'global_batch_size': 8,
                      'step0_iters': 20000, 'incremental_iters': 8000,
                      'smoke': False, 'ald': False, 'confusion_reweight': False}
    if any(baseline_study.get(key) != value for key, value in expected_study.items()):
        raise RuntimeError('The source study is not the completed formal seed-0 baseline')
    expected_config = {'task': '10-5', 'step': 0, 'seed': 0, 'max_iters': 20000,
                       'w_proto_kd': 0.0, 'w_proto_sep': 0.0, 'w_proto_seg': 0.1,
                       'proto_margin': 0.0, 'ald': False, 'confusion_reweight': False,
                       'train_limit': 0, 'val_limit': 0, 'spg': 2}
    if any(config.get(key) != value for key, value in expected_config.items()):
        raise RuntimeError('The source step-0 configuration does not match the formal baseline')
    records = read_metrics(stage / 'metrics.jsonl')
    if not records or records[-1].get('iteration') != 20000:
        raise RuntimeError('Source step 0 lacks its final 20k evaluation')
    if results.get('shared', {}).get('0') != records[-1]:
        raise RuntimeError('Source step-0 final metrics do not match baseline results.json')
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise RuntimeError(f'Missing original step-0 checkpoint: {checkpoint}')
    checkpoint = checkpoint.resolve()
    checkpoint_sha = file_sha256(checkpoint)
    anchor_paths = [BASELINE / 'study.json', BASELINE / 'results.json',
                    stage / 'config.json', stage / 'inputs.json', stage / 'metrics.jsonl']
    for variant in OLD_VARIANTS:
        anchor = BASELINE / variant / '10-5/step1/inputs.json'
        recorded = read_json(anchor)
        if (Path(recorded.get('previous_checkpoint', '')).resolve() != checkpoint
                or recorded.get('previous_sha256') != checkpoint_sha):
            raise RuntimeError(f'Original step-0 checkpoint SHA/provenance mismatch: {anchor}')
        anchor_paths.append(anchor)
    pretrained_sha = file_sha256(PRETRAINED)
    if inputs.get('pretrained_sha256') != pretrained_sha or pretrained_sha != PRETRAINED_SHA256:
        raise RuntimeError('Pretrained weights do not match the original baseline SHA')
    data = {}
    for key in ('data_folder', 'list_folder', 'seg_label_dir', 'val_label_dir'):
        path = Path(config[key]).resolve()
        if not path.is_dir():
            raise RuntimeError(f'Missing source dataset directory: {path}')
        data[key] = str(path)
    provenance = {'checkpoint': str(checkpoint), 'checkpoint_sha256': checkpoint_sha,
                  'checkpoint_bytes': checkpoint.stat().st_size, 'initialization_seed': 0,
                  'step0_iterations': 20000, 'pretrained_checkpoint': str(PRETRAINED.resolve()),
                  'pretrained_sha256': pretrained_sha, 'data': data,
                  'source_code_sha256': baseline_study['code_sha256'],
                  'source_manifests_sha256': {str(path.resolve()): file_sha256(path)
                                              for path in anchor_paths}}
    return config, provenance


def read_metrics(path):
    return [json.loads(line, parse_constant=reject_json_constant)
            for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


@contextmanager
def runner_lock(output):
    lock_path = writable_path(output / 'runner.lock')
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'r+', encoding='utf-8') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Another runner holds {lock_path}') from error
        writable_path(lock_path)
        handle.seek(0)
        handle.truncate()
        json.dump({'pid': os.getpid(), 'started_utc': utc_now()}, handle)
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def private_environment():
    executable = Path(sys.executable).resolve()
    if not executable.is_relative_to((RUNTIME / 'env').resolve()):
        raise RuntimeError('Run this study with .runtime/env/bin/python from this project')
    if not executable.is_relative_to(ROOT):
        raise RuntimeError('The private Python executable resolves outside this project')
    writable_path(RUNTIME / 'env')
    writable_path(ROOT / 'pretrained')
    writable_path(PRETRAINED)
    env = dict(os.environ)
    directories = {'TMPDIR': 'tmp', 'XDG_CACHE_HOME': 'cache/xdg',
                   'HF_HOME': 'cache/huggingface', 'TORCH_HOME': 'cache/torch',
                   'CUDA_CACHE_PATH': 'cache/cuda', 'TRITON_CACHE_DIR': 'cache/triton',
                   'MPLCONFIGDIR': 'cache/matplotlib', 'PIP_CACHE_DIR': 'cache/pip',
                   'CONDA_PKGS_DIRS': 'pkgs', 'CONDA_ENVS_PATH': 'envs'}
    for variable, relative in directories.items():
        directory = RUNTIME / relative
        make_directory(directory)
        env[variable] = str(directory)
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
               OMP_NUM_THREADS='4', MPLBACKEND='Agg', CONDA_ALWAYS_COPY='true',
               PIP_DISABLE_PIP_VERSION_CHECK='1')
    for variable in ('MASTER_ADDR', 'MASTER_PORT', 'RANK', 'WORLD_SIZE',
                     'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'GROUP_RANK', 'ROLE_RANK',
                     'ROLE_WORLD_SIZE', 'TORCHELASTIC_RUN_ID'):
        env.pop(variable, None)
    return env


class Study:
    def __init__(self, args, output, source_config, source, environment):
        self.args, self.output = args, output
        self.source, self.environment = source, environment
        self.iterations = 4 if args.smoke else 8000
        self.code = source_fingerprint()
        self.data_hashes = list_fingerprint(Path(source['data']['list_folder']))
        self.common = dict(source_config)
        for key in ('step', 'prev_checkpoint', 'work_dir', 'ckpt_dir', 'pred_dir', 'ald'):
            self.common.pop(key, None)
        self.common.update(source['data'])
        self.common.update(max_iters=self.iterations, lr=2e-5, spg=8, seed=args.seed,
                           num_workers=args.workers, w_proto_kd=0.0, w_proto_sep=0.0,
                           w_proto_seg=0.1, proto_margin=0.0, confusion_reweight=False,
                           warmup_iters=1 if args.smoke else 2000,
                           loss_warmup_iters=1 if args.smoke else 2000,
                           eval_iters=4 if args.smoke else 2000,
                           log_iters=1 if args.smoke else 50,
                           train_limit=32 if args.smoke else 0,
                           val_limit=8 if args.smoke else 0, local_rank=0, save_ckpt=True)
        self.config = {'schema_version': 1, 'study': 'matched ALD fusion', 'task': '10-5',
                       'smoke': args.smoke, 'seed': args.seed,
                       'seed_scope': 'incremental only; shared initialization is original seed 0',
                       'global_batch_size': 8, 'incremental_iters': self.iterations,
                       'variants': list(VARIANTS), 'gpu_mapping': dict(zip(VARIANTS, args.gpus)),
                       'python': sys.executable, 'common_training_config': self.common,
                       'shared_initialization': source, 'code_sha256': self.code,
                       'dataset_lists_sha256': self.data_hashes,
                       'historical_baseline_role': 'reference only; matched off is trained anew'}
        self.cancel = threading.Event()
        self.process_lock = threading.RLock()
        self.processes = {}
        self.accepted_manifest = False

    def assert_frozen(self):
        if source_fingerprint() != self.code:
            raise RuntimeError('Core source changed during the study; no further stage may start')
        if list_fingerprint(Path(self.source['data']['list_folder'])) != self.data_hashes:
            raise RuntimeError('VOC split/label lists changed during the study')
        for name, digest in self.source['source_manifests_sha256'].items():
            if file_sha256(Path(name)) != digest:
                raise RuntimeError(f'Original baseline provenance changed: {name}')
        if file_sha256(Path(self.source['checkpoint'])) != self.source['checkpoint_sha256']:
            raise RuntimeError('Original shared step-0 checkpoint changed')
        if file_sha256(PRETRAINED) != self.source['pretrained_sha256']:
            raise RuntimeError('Pretrained weights changed')

    def admit_output(self):
        manifest = self.output / 'study.json'
        if manifest.exists():
            if read_json(manifest) != self.config:
                raise RuntimeError('Existing study configuration/provenance differs; choose a new output')
        else:
            if any(path.name != 'runner.lock' for path in self.output.iterdir()):
                raise RuntimeError('Existing output lacks matching study.json; choose an empty output')
            write_json(manifest, self.config)
        self.accepted_manifest = True
        for variant in VARIANTS:
            for step in (1, 2):
                directory = self.output / variant / '10-5' / f'step{step}'
                writable_path(directory)
                if directory.exists() and any(directory.iterdir()):
                    if not (directory / 'checkpoints/model_final.pth').is_file():
                        raise RuntimeError(f'Incomplete stage is preserved, not overwritten: {directory}')
                    if not (directory / 'completion.json').is_file():
                        raise RuntimeError(f'Existing stage lacks exit-0 completion provenance: {directory}')
                if step == 2 and directory.exists() and any(directory.iterdir()):
                    previous = directory.parent / 'step1/checkpoints/model_final.pth'
                    if not previous.is_file():
                        raise RuntimeError(f'Orphan step-2 stage: {directory}')
        # Validate every reusable stage before any of the four GPU jobs starts.
        for variant, gpu in zip(VARIANTS, self.args.gpus):
            previous = Path(self.source['checkpoint'])
            for step in (1, 2):
                directory, expected, _, inputs = self.stage_plan(variant, step, gpu, previous)
                if (directory / 'checkpoints/model_final.pth').is_file():
                    previous = self.reuse_completed(directory, expected, inputs)
                else:
                    log = self.output / variant / f'launcher-step{step}.log'
                    writable_path(log)
                    if log.exists():
                        raise RuntimeError(f'Existing incomplete launcher log is preserved: {log}')
                    break

    def stage_plan(self, variant, step, gpu, previous):
        variant_root = self.output / variant
        directory = variant_root / '10-5' / f'step{step}'
        expected = dict(self.common, step=step, ald_mode=variant, ald=variant != 'off',
                        prev_checkpoint=str(previous.resolve()), work_dir=str(directory),
                        ckpt_dir=str(directory / 'checkpoints'),
                        pred_dir=str(directory / 'predictions'))
        command = [sys.executable, '-B', '-m', 'torch.distributed.run', '--standalone',
                   '--nnodes=1', '--nproc_per_node=1', str(ENTRY),
                   '--work_dir', str(variant_root), '--step', str(step),
                   '--prev_checkpoint', str(previous.resolve()), '--ald_mode', variant]
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
                  'previous_sha256': file_sha256(previous),
                  'pretrained_sha256': self.source['pretrained_sha256'],
                  'shared_initialization': self.source, 'code_sha256': self.code,
                  'dataset_lists_sha256': self.data_hashes, 'gpu': gpu,
                  'command': command, 'expected_config': expected}
        return directory, expected, command, inputs

    def final_metrics(self, directory, expected):
        records = read_metrics(directory / 'metrics.jsonl')
        required_iterations = ([4] if self.args.smoke else [2000, 4000, 6000, 8000])
        if [record.get('iteration') for record in records] != required_iterations:
            raise RuntimeError(f'Missing or unexpected evaluations: {directory}')
        keys = ('step', 'ald_mode', 'ald', 'seed', 'confusion_reweight',
                'w_proto_kd', 'w_proto_sep', 'proto_margin')
        for record in records:
            if any(record.get(key) != expected[key] for key in keys):
                raise RuntimeError(f'Metric configuration mismatch: {directory}')
            value = record.get('all_miou')
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise RuntimeError(f'Nonfinite/missing mIoU: {directory}')
            if not self.args.smoke:
                for key in ('old_miou', 'new_miou'):
                    value = record.get(key)
                    if not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise RuntimeError(f'Nonfinite/missing {key}: {directory}')
        return records[-1]

    def validate_configuration(self, directory, expected):
        actual = read_json(directory / 'config.json')
        if any(actual.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f'Training configuration mismatch: {directory}')
        return actual

    def final_ald_metrics(self, directory, expected):
        records = read_metrics(directory / 'ald_metrics.jsonl')
        period, maximum = expected['log_iters'], expected['max_iters']
        iterations = list(range(period, maximum + 1, period))
        if not iterations or iterations[-1] != maximum:
            iterations.append(maximum)
        if [record.get('iteration') for record in records] != iterations:
            raise RuntimeError(f'Missing or unexpected ALD coverage intervals: {directory}')
        count_keys = ('batch_images', 'total_pixels', 'valid_box_pixels',
                      'before_new_cam_pixels', 'after_new_cam_pixels', 'rejected_new_cam_pixels',
                      'rejected_teacher_old_pixels', 'rejected_teacher_bg_pixels',
                      'retained_ignore_pixels', 'rejected_final_ignore_pixels',
                      'rejected_final_old_pixels', 'rejected_final_bg_pixels',
                      'final_valid_pixels', 'final_bg_pixels', 'final_old_pixels',
                      'final_new_pixels', 'final_ignore_pixels', 'padding_labeled_pixels',
                      'positive_old_classes', 'positive_new_classes',
                      'gate_rejected_old_classes', 'gate_rejected_new_classes',
                      'fallback_images', 'fallback_old_images', 'fallback_new_images')
        previous = 0
        for row, iteration in zip(records, iterations):
            def require(condition):
                if not condition:
                    raise RuntimeError(f'Invalid ALD coverage at iteration {iteration}: {directory}')
            require(all(type(row.get(key)) is int and row[key] >= 0 for key in count_keys))
            interval = iteration - previous
            require(type(row.get('iteration')) is int and type(row.get('step')) is int)
            require(row.get('step') == expected['step'] and row.get('ald_mode') == expected['ald_mode'])
            require(type(row.get('interval_batches')) is int and row['interval_batches'] == interval)
            require(row['batch_images'] == 8 * interval)
            require(row.get('segmentation_loss_active') is (iteration > expected['loss_warmup_iters']))
            valid, rejected = row['valid_box_pixels'], row['rejected_new_cam_pixels']
            old, background = row['rejected_teacher_old_pixels'], row['rejected_teacher_bg_pixels']
            require(row['total_pixels'] == row['batch_images'] * expected['crop_size'] ** 2)
            require(0 <= valid <= row['total_pixels'] and row['before_new_cam_pixels'] <= valid)
            require(row['before_new_cam_pixels'] == row['after_new_cam_pixels'] + rejected)
            require(rejected == old + background)
            require(rejected == row['rejected_final_ignore_pixels'] + row['rejected_final_old_pixels'] + row['rejected_final_bg_pixels'])
            require(row['retained_ignore_pixels'] == row['rejected_final_ignore_pixels'] == row['final_ignore_pixels'])
            require(row['final_valid_pixels'] == row['final_bg_pixels'] + row['final_old_pixels'] + row['final_new_pixels'])
            require(valid == row['final_valid_pixels'] + row['final_ignore_pixels'])
            require(row['final_new_pixels'] == row['after_new_cam_pixels'] and row['padding_labeled_pixels'] == 0)
            mode = expected['ald_mode']
            retained = rejected if mode == 'preserve_rejected' else background if mode == 'preserve_background' else 0
            require(row['retained_ignore_pixels'] == retained)
            require(row['rejected_final_old_pixels'] == (0 if mode == 'preserve_rejected' else old))
            require(row['rejected_final_bg_pixels'] == (background if mode == 'legacy' else 0))
            require(row['fallback_images'] == row['fallback_old_images'] + row['fallback_new_images'] <= row['batch_images'])
            require(row['gate_rejected_old_classes'] <= row['positive_old_classes'])
            require(row['gate_rejected_new_classes'] <= row['positive_new_classes'])
            if mode == 'off':
                require(rejected == 0 and row['gate_rejected_old_classes'] == 0
                        and row['gate_rejected_new_classes'] == 0 and row['fallback_images'] == 0)
            ratios = {'rejected_fraction_of_new_cam': (rejected, row['before_new_cam_pixels']),
                      'rejected_fraction_of_valid_image': (rejected, valid),
                      'supervised_fraction_of_valid_image': (row['final_valid_pixels'], valid),
                      'retained_ignore_fraction_of_rejected': (retained, rejected)}
            for key, (numerator, denominator) in ratios.items():
                value = row.get(key)
                require(value is None if denominator == 0 else
                        type(value) in (int, float) and math.isfinite(value)
                        and math.isclose(value, numerator / denominator, rel_tol=1e-12, abs_tol=1e-12))
            previous = iteration
        return records[-1]

    def reuse_completed(self, directory, expected, inputs):
        final = directory / 'checkpoints/model_final.pth'
        if not final.is_file():
            return None
        if read_json(directory / 'inputs.json') != inputs:
            raise RuntimeError(f'Input/command provenance mismatch: {directory}')
        actual = self.validate_configuration(directory, expected)
        metrics = self.final_metrics(directory, expected)
        ald_metrics = self.final_ald_metrics(directory, expected)
        complete = read_json(directory / 'completion.json')
        digests = {'inputs_sha256': file_sha256(directory / 'inputs.json'),
                   'config_sha256': file_sha256(directory / 'config.json'),
                   'checkpoint_sha256': file_sha256(final),
                   'metrics_sha256': file_sha256(directory / 'metrics.jsonl'),
                   'ald_metrics_sha256': file_sha256(directory / 'ald_metrics.jsonl')}
        if (complete.get('returncode') != 0 or complete.get('iteration') != self.iterations
                or complete.get('command') != inputs['command']
                or complete.get('config') != actual or complete.get('final_metrics') != metrics
                or complete.get('final_ald_metrics') != ald_metrics
                or any(complete.get(key) != value for key, value in digests.items())):
            raise RuntimeError(f'Completed stage provenance mismatch: {directory}')
        print(f'Reusing completed {directory.relative_to(self.output)}', flush=True)
        return final

    def signal_owned_processes(self, signum):
        with self.process_lock:
            for process in self.processes.values():
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signum)
                    except ProcessLookupError:
                        pass

    def stop_owned_processes(self):
        self.cancel.set()
        self.signal_owned_processes(signal.SIGTERM)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            with self.process_lock:
                running = any(process.poll() is None for process in self.processes.values())
            if not running:
                return
            time.sleep(0.25)
        self.signal_owned_processes(signal.SIGKILL)

    def train(self, variant, step, gpu, previous):
        if self.cancel.is_set():
            raise RuntimeError(f'Cancelled before {variant} step {step}')
        self.assert_frozen()
        directory, expected, command, inputs = self.stage_plan(variant, step, gpu, previous)
        reused = self.reuse_completed(directory, expected, inputs)
        if reused:
            return reused
        if directory.exists() and any(directory.iterdir()):
            raise RuntimeError(f'Incomplete stage is preserved, not overwritten: {directory}')
        make_directory(directory)
        write_json(directory / 'inputs.json', inputs)
        log = writable_path(self.output / variant / f'launcher-step{step}.log')
        started = utc_now()
        env = dict(self.environment, CUDA_VISIBLE_DEVICES=gpu)
        print(f'Starting {variant} step {step}, GPU {gpu}, {self.iterations} iterations', flush=True)
        with log.open('x', encoding='utf-8') as handle:
            handle.write('COMMAND ' + json.dumps(command) + '\n')
            handle.flush()
            with self.process_lock:
                if self.cancel.is_set():
                    raise RuntimeError(f'Cancelled before launching {variant} step {step}')
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                self.processes[(variant, step)] = process
            returncode = process.wait()
        finished = utc_now()
        write_json(directory / 'process.json', {'pid': process.pid, 'returncode': returncode,
                   'started_utc': started, 'finished_utc': finished, 'command': command})
        if returncode != 0:
            raise RuntimeError(f'{variant} step {step} exited {returncode}; see {log}')
        if self.cancel.is_set():
            raise RuntimeError(f'Cancelled after {variant} step {step}; artifacts are preserved')
        self.assert_frozen()
        actual = self.validate_configuration(directory, expected)
        metrics = self.final_metrics(directory, expected)
        ald_metrics = self.final_ald_metrics(directory, expected)
        final = directory / 'checkpoints/model_final.pth'
        writable_path(final)
        if not final.is_file() or final.stat().st_size == 0:
            raise RuntimeError(f'Missing final checkpoint: {final}')
        completion = {'returncode': 0, 'iteration': self.iterations,
                      'started_utc': started, 'finished_utc': finished,
                      'command': command, 'config': actual, 'final_metrics': metrics,
                      'final_ald_metrics': ald_metrics,
                      'inputs_sha256': file_sha256(directory / 'inputs.json'),
                      'config_sha256': file_sha256(directory / 'config.json'),
                      'checkpoint_sha256': file_sha256(final),
                      'metrics_sha256': file_sha256(directory / 'metrics.jsonl'),
                      'ald_metrics_sha256': file_sha256(directory / 'ald_metrics.jsonl')}
        write_json(directory / 'completion.json', completion)
        print(f'Completed {variant} step {step}: all_miou={metrics["all_miou"]:.4f}', flush=True)
        return final

    def run_variant(self, variant, gpu):
        previous = Path(self.source['checkpoint'])
        for step in (1, 2):
            previous = self.train(variant, step, gpu, previous)
        return variant

    def record_failure(self, error):
        if not self.accepted_manifest:
            return
        path = self.output / 'failures.json'
        previous = read_json(path) if path.exists() else {'attempts': []}
        previous['attempts'].append({'time_utc': utc_now(), 'pid': os.getpid(),
                                     'error': str(error), 'type': type(error).__name__})
        write_json(path, previous, replace=path.exists())
        write_json(self.output / 'status.json', {'status': 'failed', 'time_utc': utc_now(),
                    'expected_stages': 8, 'error': str(error)}, replace=True)

    def run(self):
        self.admit_output()
        write_json(self.output / 'status.json', {'status': 'in_progress',
                   'time_utc': utc_now(), 'expected_stages': 8}, replace=True)
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        errors = []
        try:
            jobs = {pool.submit(self.run_variant, variant, gpu): variant
                    for variant, gpu in zip(VARIANTS, self.args.gpus)}
            for future in concurrent.futures.as_completed(jobs):
                try:
                    print('Finished variant:', future.result(), flush=True)
                except Exception as error:
                    errors.append(f'{jobs[future]}: {error}')
                    self.stop_owned_processes()
            if errors:
                raise RuntimeError('; '.join(errors))
        except BaseException:
            self.stop_owned_processes()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        self.assert_frozen()
        results = {}
        for variant, gpu in zip(VARIANTS, self.args.gpus):
            results[variant] = {}
            previous = Path(self.source['checkpoint'])
            for step in (1, 2):
                directory, expected, _, inputs = self.stage_plan(variant, step, gpu, previous)
                previous = self.reuse_completed(directory, expected, inputs)
                if previous is None:
                    raise RuntimeError(f'Missing final completed stage: {directory}')
                results[variant][str(step)] = self.final_metrics(directory, expected)
        destination = self.output / 'results.json'
        if destination.exists():
            if read_json(destination) != results:
                raise RuntimeError('Existing results.json differs from verified stage results')
        else:
            write_json(destination, results)
        write_json(self.output / 'status.json', {'status': 'complete', 'time_utc': utc_now(),
                   'completed_stages': 8, 'expected_stages': 8, 'all_stage_returncodes': 0,
                   'results_sha256': file_sha256(destination)}, replace=True)
        print('Study complete:', destination, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=None,
                        help='default: runs/ald_fusion_v1, or runs/ald_fusion_v1_smoke')
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0,
                        help='incremental seed; shared step-0 initialization remains seed 0')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    args.gpus = [item.strip() for item in args.gpus.split(',')]
    if len(args.gpus) != 4 or len(set(args.gpus)) != 4 or any(not item for item in args.gpus):
        parser.error('The study requires four distinct authorized GPU device IDs')
    if set(args.gpus) != {'0', '1', '2', '3'}:
        parser.error('This study is authorized for physical GPUs 0, 1, 2 and 3 only')
    if args.workers < 1 or args.seed < 0:
        parser.error('workers must be positive and seed must be nonnegative')
    output = ROOT / (args.output or ('runs/ald_fusion_v1_smoke' if args.smoke else 'runs/ald_fusion_v1'))
    if not output.is_relative_to(ROOT / 'runs') or output.resolve() != output:
        parser.error('output must be a canonical path within this project/runs')
    if output.is_relative_to(BASELINE) or BASELINE.is_relative_to(output):
        parser.error('output must be separate from the preserved fixed baseline')
    make_directory(output)
    with runner_lock(output):
        source_config, source = verify_shared_initialization()
        study = Study(args, output, source_config, source, private_environment())
        previous_handlers = {}

        def handle_signal(signum, _frame):
            study.cancel.set()
            study.signal_owned_processes(signal.SIGTERM)
            raise KeyboardInterrupt(f'Runner received signal {signum}')

        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, handle_signal)
        try:
            study.run()
        except BaseException as error:
            study.stop_owned_processes()
            study.record_failure(error)
            raise
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)


if __name__ == '__main__':
    main()
