"""Run the fixed-prototype VOC 10-5 study with one shared initialization.

Each incremental variant owns its checkpoints. ALD and confusion reweighting
are absent from every command, so both stay disabled. Existing completed steps
are reused only when their recorded configuration matches this study.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / 'scripts/dist_train_voc_seg_neg.py'
VARIANTS = {'base': (0.0, 0.0), 'kd': (0.1, 0.0),
            'sep': (0.0, 0.1), 'kd_sep': (0.1, 0.1)}


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
        raise RuntimeError(f'Unsafe output: {path}')
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='runs/fixed_baseline_v1')
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--step0_iters', type=int, default=20000)
    parser.add_argument('--incremental_iters', type=int, default=8000)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    gpus = args.gpus.split(',')
    if len(gpus) != 4 or len(set(gpus)) != 4:
        parser.error('The four-way study requires four distinct authorized GPUs')
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT):
        parser.error('output must stay inside this project')
    output.mkdir(parents=True, exist_ok=True)
    config = {'task': '10-5', 'seed': 0, 'global_batch_size': 8,
              'step0_iters': 4 if args.smoke else args.step0_iters,
              'incremental_iters': 4 if args.smoke else args.incremental_iters,
              'workers': args.workers, 'smoke': args.smoke,
              'prototype_margin': 0.0, 'prototype_seg_weight': 0.1,
              'ald': False, 'confusion_reweight': False,
              'variants': VARIANTS, 'python': sys.executable,
              'code_sha256': {str(p.relative_to(ROOT)): file_sha256(p)
                              for p in [ENTRY, ROOT / 'continual/Trainer.py', ROOT / 'model/losses.py',
                                        ROOT / 'datasets/voc.py', ROOT / 'model/decoder/conv_head.py',
                                        ROOT / 'utils/optimizer.py', ROOT / 'utils/pyutils.py']}}
    manifest = output / 'study.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != json.loads(json.dumps(config)):
            raise RuntimeError('Existing study has a different configuration; choose a new output directory')
    else:
        write_json(manifest, config)

    def train(variant, step, devices, previous=None):
        variant_root = output / variant
        step_dir = variant_root / '10-5' / f'step{step}'
        final = step_dir / 'checkpoints/model_final.pth'
        iterations = config['step0_iters'] if step == 0 else config['incremental_iters']
        kd, sep = (0.0, 0.0) if step == 0 else VARIANTS[variant]
        step_dir.mkdir(parents=True, exist_ok=True)
        inputs = {'previous_checkpoint': str(previous) if previous else None,
                  'previous_sha256': file_sha256(previous) if previous else None,
                  'pretrained_sha256': file_sha256(ROOT / 'pretrained/checkpoints/jx_vit_base_p16_224-80ecf9dd.pth')}
        input_manifest = step_dir / 'inputs.json'
        if final.exists() and not input_manifest.exists():
            raise RuntimeError(f'Existing checkpoint lacks predecessor provenance: {final}')
        if input_manifest.exists() and json.loads(input_manifest.read_text()) != inputs:
            raise RuntimeError(f'Previous checkpoint mismatch: {step_dir}')
        if not input_manifest.exists():
            write_json(input_manifest, inputs)
        if final.exists():
            recorded = json.loads((step_dir / 'config.json').read_text())
            expected = {'step': step, 'max_iters': iterations, 'w_proto_kd': kd,
                        'w_proto_sep': sep, 'ald': False, 'confusion_reweight': False}
            if any(recorded.get(k) != v for k, v in expected.items()):
                raise RuntimeError(f'Existing checkpoint configuration mismatch: {final}')
            print(f'Reusing completed {variant} step {step}: {final}', flush=True)
            return final
        command = [sys.executable, '-B', '-m', 'torch.distributed.run', '--standalone',
                   '--nnodes=1', f'--nproc_per_node={len(devices)}', str(ENTRY),
                   '--task', '10-5', '--step', str(step), '--work_dir', str(variant_root),
                   '--max_iters', str(iterations), '--lr', '6e-5' if step == 0 else '2e-5',
                   '--spg', str(8 // len(devices)), '--seed', '0', '--save_ckpt',
                   '--num_workers', str(args.workers), '--w_proto_kd', str(kd),
                   '--w_proto_sep', str(sep), '--w_proto_seg', '0.1', '--proto_margin', '0']
        if previous:
            command.extend(['--prev_checkpoint', str(previous)])
        if args.smoke:
            command.extend(['--train_limit', '32', '--val_limit', '8', '--eval_iters', '4',
                            '--log_iters', '1', '--warmup_iters', '1', '--loss_warmup_iters', '1'])
        variant_root.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(devices),
                   PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4',
                   TMPDIR=str(ROOT / '.runtime/tmp'))
        log = variant_root / f'launcher-step{step}.log'
        started = time.time()
        print(f'Starting {variant} step {step}, GPU {devices}, {iterations} iterations', flush=True)
        with log.open('a', encoding='utf-8') as handle:
            handle.write('\nCOMMAND ' + json.dumps(command) + '\n')
            handle.flush()
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode or not final.exists():
            raise RuntimeError(f'{variant} step {step} failed ({result.returncode}); see {log}')
        print(f'Completed {variant} step {step} in {time.time() - started:.1f}s', flush=True)
        return final

    shared = train('shared', 0, gpus)
    def run_variant(item):
        variant, gpu = item
        previous = shared
        for step in [1, 2]:
            previous = train(variant, step, [gpu], previous)
        return variant
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {pool.submit(run_variant, item): item[0] for item in zip(VARIANTS, gpus)}
        for future in concurrent.futures.as_completed(jobs):
            try:
                print('Finished variant:', future.result(), flush=True)
            except Exception as error:
                errors.append(str(error))
                print('FAILED:', error, flush=True)
    if errors:
        write_json(output / 'failures.json', errors)
        raise RuntimeError('Some variants failed; completed variants are preserved')
    summary = {}
    for variant in ['shared', *VARIANTS]:
        summary[variant] = {}
        steps = [0] if variant == 'shared' else [1, 2]
        for step in steps:
            records = (output / variant / '10-5' / f'step{step}' / 'metrics.jsonl').read_text().splitlines()
            final = json.loads(records[-1])
            if final['iteration'] != (config['step0_iters'] if step == 0 else config['incremental_iters']):
                raise RuntimeError(f'Missing final evaluation for {variant} step {step}')
            summary[variant][str(step)] = final
    write_json(output / 'results.json', summary)
    print('Study complete:', output / 'results.json', flush=True)


if __name__ == '__main__':
    main()
