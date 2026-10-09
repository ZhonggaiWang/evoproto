"""Finite CPU queue for completed formal checkpoints; never occupies a GPU."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / 'runs/restore_proto_v1/control'
FORMAL = ROOT / 'runs/restore_proto_v1/formal'
child = None


def read(path):
    return json.loads(path.read_text())


def status(**data):
    data.update(pid=os.getpid(), time=time.time())
    path = CONTROL / 'postprocess_state.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)
    print(json.dumps(data), flush=True)


def coordinator_alive():
    try:
        pid = int((CONTROL / 'formal.pid').read_text())
        return b'restore_proto_v1/run.py' in Path(f'/proc/{pid}/cmdline').read_bytes()
    except (OSError, ValueError):
        return False


def wait_for(path):
    status(state='waiting', path=str(path))
    while not path.exists():
        if not coordinator_alive():
            raise RuntimeError('Formal coordinator ended before ' + str(path))
        time.sleep(15)


def stop(*unused):
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
    raise KeyboardInterrupt


def run(script, arguments, logfile):
    global child
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='8', PYTHONDONTWRITEBYTECODE='1')
    command = [sys.executable, '-B', str(ROOT / 'tools' / script)] + arguments
    status(state='running', command=command)
    with logfile.open('a') as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        code = child.wait()
    if code:
        raise RuntimeError(f'{script} exited {code}; inspect {logfile}')


def main():
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for variant in read(FORMAL / 'manifest.json')['variants']:
        for stage in (1, 2):
            directory = FORMAL / variant / '10-5' / f'step{stage}'
            wait_for(directory / 'final_receipt.json')
            run('audit_restore_proto_checkpoint.py', ['--stage-dir', str(directory)],
                directory / 'checkpoint_audit.log')
        wait_for(directory / 'evaluation.json')
        official = read(directory / 'evaluation.json')
        output = directory / 'fusion_evaluation.json'
        if not output.exists():
            run('evaluate_restore_proto_fusion.py', ['--checkpoint', str(directory / 'checkpoints/model_final.pth'),
                '--stage', '2', '--output', str(output), '--device', 'cpu', '--threads', '8',
                '--expected-main', str(official['results']['square448']['main']['miou'])], directory / 'fusion_evaluation.log')
        fusion = read(output)
        assert fusion['images'] == official['images'] == 1449
        assert fusion['checkpoint_sha256'] == official['checkpoint_sha256']
        assert fusion['image_tags_used'] is False and fusion['alphas'] == [0, .25, .5, .75, 1]
        differences = {}
        for mode in ('square448', 'aspect672'):
            differences[mode] = {}
            for alpha, head in [('0.0', 'main'), ('1.0', 'prototype')]:
                delta = fusion['results'][mode][alpha]['miou'] - official['results'][mode][head]['miou']
                assert abs(delta) < .2, (variant, mode, head, delta)
                differences[mode][head] = delta
        import numpy as np
        with np.load(output.with_suffix('.npz'), allow_pickle=False) as data:
            assert len(data['names']) == len(set(data['names'])) == 1449
            assert np.array_equal(data['alphas'], fusion['alphas'])
            for mode in ('square448', 'aspect672'):
                assert data[mode].shape == (1449, 5, 21, 21)
                aggregate = data[mode].sum(axis=0)
                for index, alpha in enumerate(fusion['alphas']):
                    assert np.array_equal(aggregate[index], fusion['results'][mode][str(alpha)]['histogram'])
        verification = {'checkpoint_sha256': fusion['checkpoint_sha256'], 'images': 1449,
                        'cpu_minus_gpu_miou': differences, 'per_image_histograms_verified': True}
        (directory / 'fusion_verification.json').write_text(json.dumps(verification, indent=2))
        status(state='variant_completed', variant=variant)
    status(state='completed')


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        status(state='failed', error=repr(error))
        raise
