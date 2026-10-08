"""Conditionally launch C after complete B failure and exact-source checks.

Use --smoke first, then formal after the new restoration smoke completes.
Only the authorized c_new_anchor run receives new files. The runner never copies or edits a source.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

R = Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto')
E = R / 'experiments/prototype_sep_v1'


def main(smoke=False, check_only=False):
    sys.path.insert(0, str(E))
    import run_new_anchor_sep as runner
    services = runner.initialize()
    environment, safe_path, atomic_json, now, digest = services
    gate = runner.gates(services)
    cfg = runner.configuration(smoke, services, gate)
    inherited = runner.restoration(services)
    if not smoke:
        runner.smoke_gate(services, gate, inherited)
    run = safe_path(runner.U / ('smoke' if smoke else 'formal') / 'c_new_anchor')
    require = runner.require
    require(not (run / 'study.json').exists() and not (run / 'coordinator.process.json').exists(),
            'Refuse duplicate C launch')
    cmd = [str(runner.PY), '-B', str(runner.RUNNER)]
    if smoke:
        cmd.append('--smoke')
    if check_only:
        print(json.dumps({'ready': True, 'outputs_written': False, 'smoke': smoke,
                          'run': str(run), 'coordinator_command': cmd,
                          'train_command': runner.command_for(run, cfg),
                          'restoration': inherited}, allow_nan=False))
        return
    import fcntl
    run.mkdir(parents=True, exist_ok=True)
    with open(safe_path(run / 'launch.lock'), 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (run / 'study.json').exists() and not (run / 'coordinator.process.json').exists(),
                'Another C launch acquired this run')
        with open(safe_path(run / 'coordinator.log'), 'x') as log:
            child = subprocess.Popen(cmd, cwd=R, env=environment('8card'), stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        atomic_json(run / 'coordinator.process.json', {
            'pid': child.pid, 'starttime': Path(f'/proc/{child.pid}/stat').read_text().split()[21],
            'command': cmd, 'utc': now(), 'smoke': smoke,
            'runner_sha256': gate['runner_sha256'], 'launcher_sha256': gate['launcher_sha256'],
        })
    print(json.dumps({'pid': child.pid, 'run': str(run), 'smoke': smoke}, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--check-only', action='store_true', help='Read-only; do not create files or launch')
    args = parser.parse_args()
    main(args.smoke, args.check_only)


