"""Offline join of saved OLC predictions and image annotations; never used in training."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from tools.audit_old_image_labels import metrics

def main():
    p=argparse.ArgumentParser();p.add_argument('--memory',required=True);p.add_argument('--annotation-audit',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    memory=Path(a.memory).resolve();out=Path(a.output).resolve()
    assert memory.is_relative_to(ROOT) and out.is_relative_to(ROOT)
    data=torch.load(memory,map_location='cpu',weights_only=True)
    names=np.asarray(data['names']);prob=data['probabilities'].numpy();k=data['old_classes']
    # Training exports predictions only. This separate offline process joins GT.
    with np.load(a.annotation_audit,allow_pickle=False) as diagnostic:
        index={str(n):i for i,n in enumerate(diagnostic['names'])}
        assert len(index)==len(diagnostic['names']) and all(str(n) in index for n in names)
        gt=diagnostic['diagnostic_gt'][[index[str(n)] for n in names],:k].astype(bool)
        original=diagnostic['probabilities'][[index[str(n)] for n in names],0,:k]>.5
    positive=(prob>.5).all(1);known=positive|(prob<=.5).all(1)
    main=prob[:,0]>.5
    policies={'olc_memory':(positive,known),'main_head_same_ema':(main,np.ones_like(main)),
              'original_full_image_main_different_view':(original,np.ones_like(original))}
    result=dict(iteration=data['updates'],images=len(names),old_classes=k,momentum=data['momentum'],
                minimum_visits=int(data['visits'].min()),maximum_visits=int(data['visits'].max()),
                mean_visits=float(data['visits'].float().mean()),
                annotation_use='Offline only; never used by OLC or trainer',
                limitation='Image presence annotations describe the full image. EMA uses randomly augmented crops; original full-image main is a different-view reference and at stage 2 can also use a different predecessor teacher. It is not an isolated EMA ablation.',
                policies={n:metrics(*v,gt) for n,v in policies.items()},
                per_class={n:[metrics(v[0][:,c],v[1][:,c],gt[:,c]) for c in range(k)] for n,v in policies.items()})
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2))
    print(json.dumps({n:v for n,v in result.items() if n!='per_class'}))
if __name__=='__main__':main()
