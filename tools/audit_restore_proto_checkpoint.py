"""CPU audit of completed restoration checkpoints and their recorded lineage."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from model.model_seg_neg import network


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inside(path):
    path = Path(path).resolve(strict=True)
    if not path.is_relative_to(ROOT):
        raise ValueError('Checkpoint audit path outside project: ' + str(path))
    return path


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage-dir', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    stage_dir = inside(args.stage_dir)
    config = read(stage_dir / 'config.json')
    stage = config['step']
    assert stage in (1, 2)
    checkpoint = inside(stage_dir / 'checkpoints/model_final.pth')
    receipt = read(stage_dir / 'final_receipt.json')
    digest = sha(checkpoint)
    assert digest == receipt['sha256'], 'Final checkpoint digest mismatch'
    assert inside(receipt['path']) == checkpoint
    for relative, expected in receipt['source_sha256'].items():
        assert sha(inside(ROOT / relative)) == expected, 'Source changed: ' + relative
    predecessor = read(stage_dir / 'predecessor.json')
    parent_path = inside(predecessor['path'])
    assert sha(parent_path) == predecessor['sha256'], 'Predecessor digest mismatch'
    raw = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert raw['iteration'] == config['max_iters'] == 8000
    assert raw['variant'] == config['variant']
    state = {k.removeprefix('module.'): v for k, v in raw['model_state'].items()}
    assert all(torch.isfinite(v).all().item() for v in state.values()), 'Nonfinite model tensor'
    classes = [11] + [5] * stage
    net = network('vit_base_patch16_224', num_classes=sum(classes), classes_list=classes,
                  pretrained=False, init_momentum=.9, aux_layer=-3)
    net.load_state_dict(state, strict=True)
    del net
    parent = torch.load(parent_path, map_location='cpu', weights_only=True)
    parent_state = {k.removeprefix('module.'): v for k, v in parent['model_state'].items()}
    prototypes = []
    for index, count in enumerate(classes):
        key = f'decoder.class_prototypes.{index}.prototype'
        value = state[key]
        assert tuple(value.shape) == (count, 512)
        norms = value.norm(dim=1)
        assert (norms > 0).all().item()
        row = {'key': key, 'shape': list(value.shape), 'min_norm': norms.min().item(),
               'max_norm': norms.max().item(), 'inherited_block': key in parent_state}
        if key in parent_state:
            row['delta_l2_from_predecessor'] = (value - parent_state[key]).norm().item()
        prototypes.append(row)
    result = {'checkpoint': str(checkpoint), 'sha256': digest, 'variant': config['variant'],
              'stage': stage, 'iteration': raw['iteration'], 'strict_model_load': True,
              'all_tensors_finite': True, 'source_matches_receipt': True,
              'predecessor': predecessor, 'prototypes': prototypes,
              'note': 'Parameter presence/change alone does not prove causal benefit; use loss/gradient evidence and full-chain ablations.'}
    output = stage_dir / 'checkpoint_audit.json'
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False))
    temporary.replace(output)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
