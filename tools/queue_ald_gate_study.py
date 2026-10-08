"""Queue the isolated gate study after primary GPU 0/1 stages finish.

This process never signals or restarts any job. It writes only its private lock
and event log until the independently verified formal child is launched.
"""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    '_evoproto_gate_queue_runner', ROOT / 'tools/run_ald_gate_study.py')
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)

PRIMARY_PID = 983111
PRIMARY_STARTTIME = 803948214
PRIMARY_SHA = 'ffed1e09af01f80dbfda933b03c73d2ffda26ee428864865ee931bdf133f7619'
SMOKE_SHA = '3b4ac28d029a2311db995c61b64ff3eb768e3bae0d3a86e8983f85365d645e90'
SMOKE_MANIFEST = ROOT / 'runs/ald_gate_new_v1_smoke/study.json'
OUTPUT = ROOT / 'runs/ald_gate_new_v1'
PRIMARY_ENTRY = ROOT / 'scripts/dist_train_voc_seg_neg.py'
GPUS = ['0', '1']
POLL_SECONDS = 45
HEARTBEAT_SECONDS = 900


def process_identity(pid):
    try:
        raw = Path(f'/proc/{pid}/stat').read_text()
    except FileNotFoundError:
        return None
    fields = raw[raw.rfind(')') + 2:].split()
    return {'pid': pid, 'state': fields[0], 'starttime': int(fields[19])}


def primary_alive(pid, starttime):
    identity = process_identity(pid)
    return bool(identity and identity['starttime'] == starttime
                and identity['state'] not in ('Z', 'X', 'x'))


def safe_fd(path, flags):
    path = gate.checked_path(path)
    fd = os.open(path, flags | os.O_NOFOLLOW, 0o600)
    try:
        gate.checked_path(path)
        stat = os.fstat(fd)
        current = path.stat()
        if stat.st_nlink != 1 or (stat.st_dev, stat.st_ino) != (current.st_dev, current.st_ino):
            raise RuntimeError(f'Private queue file identity changed: {path}')
    except BaseException:
        os.close(fd)
        raise
    return fd


class Events:
    def __init__(self):
        self.path = ROOT / '.runtime/ald-gate-launch-queue.jsonl'
        self.fd = safe_fd(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND)

    def emit(self, event, **fields):
        gate.checked_path(self.path)
        if os.fstat(self.fd).st_nlink != 1:
            raise RuntimeError('Queue event log acquired another hard link')
        row = dict(utc=gate.base.utc_now(), event=event, queue_pid=os.getpid(), **fields)
        encoded = (json.dumps(row, sort_keys=True, allow_nan=False) + '\n').encode()
        while encoded:
            encoded = encoded[os.write(self.fd, encoded):]
        os.fsync(self.fd)
        print(f"[{row['utc']}] {event}: {fields.get('reason', fields.get('returncode', ''))}",
              flush=True)


def assert_anchors(expected_code):
    primary = gate.read_file(gate.PRIMARY / 'study.json')
    smoke = gate.read_file(SMOKE_MANIFEST)
    if gate.base.file_sha256(primary) != PRIMARY_SHA:
        raise RuntimeError('Primary immutable study manifest SHA changed')
    if gate.base.file_sha256(smoke) != SMOKE_SHA:
        raise RuntimeError('Completed gate smoke manifest SHA changed')
    actual = gate.gate_fingerprint()
    if actual != expected_code or len(actual) != 42:
        raise RuntimeError('Frozen 42-source fingerprint differs from completed gate smoke')
    return actual


def pending_stages():
    return [gate.PRIMARY / arm / '10-5' / f'step{step}'
            for arm in ('off', 'legacy') for step in (1, 2)
            if not gate.checked_path(gate.PRIMARY / arm / '10-5' / f'step{step}'
                                     / 'completion.json').is_file()]


def transient_rejection(reason, pending):
    if reason in [f'GPU {gpu} lacks the required 20000 MiB free memory' for gpu in GPUS]:
        return True
    if reason in [f'Primary launcher is still live on GPU {gpu}' for gpu in GPUS]:
        return True
    if reason in [f'Primary GPU process remains active on GPU {gpu}' for gpu in GPUS]:
        return True
    for directory in pending:
        if reason in (f'Missing or unexpected evaluations: {directory}',
                      f'Missing or unexpected ALD coverage intervals: {directory}'):
            return True
        prefix = 'Missing ordinary project file: '
        if reason.startswith(prefix):
            missing = Path(reason[len(prefix):])
            if missing.is_relative_to(directory):
                return True
    return False


def namespace_target_processes():
    """Match only complete primary ENTRY/work_dir tokens; never expose other argv."""
    targets = {str(gate.PRIMARY / arm): arm for arm in ('off', 'legacy')}
    mapping = gate.primary_manifest()['gpu_mapping']
    matches = []
    for directory in Path('/proc').iterdir():
        if not directory.name.isdecimal():
            continue
        try:
            tokens = [token.decode(errors='surrogateescape')
                      for token in (directory / 'cmdline').read_bytes().split(b'\0') if token]
            if str(PRIMARY_ENTRY) not in tokens:
                continue
            values = []
            for index, token in enumerate(tokens):
                if token == '--work_dir' and index + 1 < len(tokens):
                    values.append(tokens[index + 1])
                elif token.startswith('--work_dir='):
                    values.append(token.split('=', 1)[1])
            if not any(value in targets for value in values):
                continue
            if len(values) != 1:
                raise RuntimeError(f'Ambiguous target work_dir tokens for PID {directory.name}')
            raw = (directory / 'stat').read_text()
            fields = raw[raw.rfind(')') + 2:].split()
            if fields[0] in ('Z', 'X', 'x'):
                continue
            arm = targets[values[0]]
            matches.append({'pid': int(directory.name), 'ppid': int(fields[1]),
                            'process_group': int(fields[2]), 'state': fields[0],
                            'starttime': int(fields[19]), 'arm': arm,
                            'gpu_from_primary_manifest': mapping[arm],
                            'matched_entry_token': str(PRIMARY_ENTRY),
                            'matched_work_dir_token': values[0]})
        except FileNotFoundError:
            continue  # A disappearing /proc entry represents an exited process.
        except OSError as exc:
            raise RuntimeError('Cannot establish target-process absence: '
                               f'unreadable /proc/{directory.name}') from exc
    return {'pid_namespace': str(Path('/proc/self/ns/pid').readlink()),
            'scope': 'current /proc, exact primary ENTRY and off/legacy work_dir tokens; '
                     'includes inherited DataLoader argv',
            'target_match_count': len(matches),
            'target_matches': sorted(matches, key=lambda row: row['pid'])}


def require_live_primary_or_verified_targets(pid, starttime):
    alive = primary_alive(pid, starttime)
    if not alive:
        for gpu in GPUS:
            try:
                gate.verify_primary_gpu_complete(gpu)
            except (RuntimeError, FileNotFoundError) as verify_error:
                raise RuntimeError('Anchored primary runner ended before target stages '
                                   f'verified complete: {verify_error}') from verify_error
    return alive


def readiness(pid, starttime):
    # Rank sessions differ from their launchers, and NVIDIA PIDs may be outside
    # this /proc view. Establish own target-process absence independently first.
    namespace = namespace_target_processes()
    if namespace['target_match_count']:
        alive = require_live_primary_or_verified_targets(pid, starttime)
        return False, {'reason': 'waiting_found_target_processes', 'primary_alive': alive,
                       'namespace_target_processes': namespace,
                       'pending_stages': [str(path.relative_to(ROOT)) for path in pending_stages()]}
    # Admission also checks final metrics, provenance, exit codes and physical
    # memory. Its host-PID group check proves only groups visible in this /proc.
    try:
        evidence = gate.gpu_admission(GPUS, smoke=False, minimum_free=20000)
    except (RuntimeError, FileNotFoundError) as exc:
        pending = pending_stages()
        if isinstance(exc, FileNotFoundError):
            if not exc.filename:
                raise
            missing = gate.checked_path(Path(exc.filename))
            if not any(missing.is_relative_to(directory) for directory in pending):
                raise
            reason = f'Missing ordinary project file: {missing}'
        else:
            reason = str(exc)
        if not transient_rejection(reason, pending):
            raise
        alive = require_live_primary_or_verified_targets(pid, starttime)
        return False, {'reason': reason, 'primary_alive': alive,
                       'namespace_target_processes': namespace,
                       'pending_stages': [str(path.relative_to(ROOT)) for path in pending]}
    evidence['namespace_target_processes'] = namespace
    evidence['driver_pid_process_group_scope'] = (
        'Only NVIDIA-reported PIDs visible in current /proc support process-group checks; '
        'invisible driver PIDs provide no group proof. Exact namespace target absence '
        'is checked separately.')
    return True, evidence


def run(args, events):
    private_env = (ROOT / '.runtime/env').resolve()
    if not Path(sys.executable).resolve().is_relative_to(private_env):
        raise RuntimeError('Activate the project private environment before running this queue')
    smoke = gate.base.read_json(gate.read_file(SMOKE_MANIFEST))
    if not smoke.get('smoke') or smoke.get('seed') != 0:
        raise RuntimeError('Expected the completed seed-0 gate smoke manifest')
    expected_code = smoke['code_sha256']
    assert_anchors(expected_code)
    initial = process_identity(args.primary_pid)
    if args.primary_pid == PRIMARY_PID:
        starttime = PRIMARY_STARTTIME
    elif initial and initial['state'] not in ('Z', 'X', 'x'):
        starttime = initial['starttime']
    else:
        raise RuntimeError('An alternate primary PID must be alive when it is anchored')
    events.emit('anchored', primary_pid=args.primary_pid, primary_starttime=starttime,
                initial_identity=initial, primary_study_sha256=PRIMARY_SHA,
                gate_smoke_study_sha256=SMOKE_SHA, code_sha256=expected_code,
                poll_seconds=POLL_SECONDS, check_once=args.check_once)
    last_state, last_log = None, 0.0
    while True:
        assert_anchors(expected_code)  # Source failures are fatal, never ordinary waiting.
        ready, evidence = readiness(args.primary_pid, starttime)
        if ready:
            if args.check_once:
                events.emit('check_ready', admission=evidence)
                return 0
            # Recheck immediately before the only launch. Never precreate output.
            assert_anchors(expected_code)
            ready, evidence = readiness(args.primary_pid, starttime)
            if ready:
                if gate.checked_path(OUTPUT).exists():
                    raise RuntimeError('Formal gate output already exists; queue will not restart it')
                command = [sys.executable, '-B', str(ROOT / 'tools/run_ald_gate_study.py'),
                           '--output', 'runs/ald_gate_new_v1', '--gpus', '0,1',
                           '--workers', '4', '--seed', '0']
                events.emit('launch_admitted', command=command, admission=evidence)
                child = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(),
                                         start_new_session=True)
                events.emit('child_started', child_pid=child.pid, command=command)
                returncode = child.wait()
                events.emit('child_finished', child_pid=child.pid, returncode=returncode)
                return returncode if returncode >= 0 else 128 - returncode
        state = json.dumps(evidence, sort_keys=True)
        if args.check_once:
            events.emit('check_waiting', **evidence)
            return 0
        now = time.monotonic()
        if state != last_state or now - last_log >= HEARTBEAT_SECONDS:
            events.emit('waiting', **evidence)
            last_state, last_log = state, now
        time.sleep(POLL_SECONDS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-once', action='store_true',
                        help='Check readiness once without sleeping or launching any child')
    parser.add_argument('--primary-pid', type=int, default=PRIMARY_PID,
                        help='Primary runner PID; its /proc starttime is anchored against reuse')
    args = parser.parse_args()
    if args.primary_pid <= 0:
        parser.error('--primary-pid must be positive')
    lock_fd = safe_fd(ROOT / '.runtime/ald-gate-launch-queue.lock', os.O_RDWR | os.O_CREAT)
    events = None
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another gate launch queue holds the exclusive lock') from exc
        events = Events()
        return run(args, events)
    except Exception as exc:
        if events:
            events.emit('fatal', reason=str(exc))
        else:
            print(f'Queue fatal: {exc}', file=sys.stderr, flush=True)
        return 1
    finally:
        if events:
            os.close(events.fd)
        os.close(lock_fd)


if __name__ == '__main__':
    sys.exit(main())
