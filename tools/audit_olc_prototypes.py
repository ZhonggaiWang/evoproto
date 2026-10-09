"""Confirm inherited prototypes changed direction, beyond weight-decay shrinkage."""
import argparse,hashlib,json
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def prototypes(path):
    raw=torch.load(path,map_location='cpu',weights_only=True)
    state={n.removeprefix('module.'):v for n,v in raw['model_state'].items()}
    keys=sorted((n for n in state if n.startswith('decoder.class_prototypes.') and n.endswith('.prototype')),key=lambda n:int(n.split('.')[2]))
    assert keys
    values=[state[n].double() for n in keys]
    assert all(v.ndim==2 and v.shape[1]==512 and torch.isfinite(v).all() for v in values)
    return torch.cat(values),{n:list(v.shape) for n,v in zip(keys,values)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage-dir',required=True);a=p.parse_args()
    directory=Path(a.stage_dir).resolve();assert directory.is_relative_to(ROOT)
    receipt=json.loads((directory/'final_receipt.json').read_text());parent=json.loads((directory/'predecessor.json').read_text())
    current=Path(receipt['path']).resolve();previous=Path(parent['path']).resolve()
    assert current.is_relative_to(ROOT) and previous.is_relative_to(ROOT)
    assert sha(current)==receipt['sha256'] and sha(previous)==parent['sha256']
    new,shapes=prototypes(current);old,_=prototypes(previous);assert len(new)==len(old)+5
    # Exclude background; normalized direction change cannot be explained by scalar decay.
    x=F.normalize(new[1:len(old)],dim=1);y=F.normalize(old[1:],dim=1)
    displacement=(x-y).norm(dim=1);cos=(x*y).sum(1)
    result=dict(checkpoint_sha256=receipt['sha256'],predecessor_sha256=parent['sha256'],prototype_blocks=shapes,
                old_foreground_classes=len(old)-1,mean_old_cosine=float(cos.mean()),max_old_direction_displacement=float(displacement.max()),
                per_old_class_displacement=displacement.tolist(),direction_changed_beyond_decay=bool(displacement.max()>1e-3),
                interpretation='Direction changes plus active loss/source checks support actual prototype optimization; this is not evidence of segmentation benefit by itself.')
    assert result['direction_changed_beyond_decay'],'No substantial inherited prototype direction update detected'
    (directory/'prototype_direction_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':
    torch.set_num_threads(2);main()
