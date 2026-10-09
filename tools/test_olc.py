"""Behavioral OLC tests, including two-rank CPU synchronization."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
import torch.nn.functional as F
from experiments.restore_proto_olc_v1.olc import OldLabelMemory, masked_classification_loss, mask_untrusted_old

def run():
    logits=torch.tensor([[1.,2.,-1.,.3]],requires_grad=True)
    targets=torch.tensor([[1.,0.,0.,1.]])
    known=torch.tensor([[True,False,True,True]])
    masked_classification_loss(logits,targets,known).backward()
    assert logits.grad[0,1]==0
    expected=(logits.detach().sigmoid()-targets)/4
    assert torch.allclose(logits.grad[known],expected[known])
    assert torch.allclose(masked_classification_loss(logits,targets,torch.ones_like(known)),F.multilabel_soft_margin_loss(logits,targets))
    labels=torch.tensor([[[0,1,2,3,255]]])
    masked=mask_untrusted_old(labels,torch.tensor([[False,True]]))
    assert masked.tolist()==[[[0,255,2,3,255]]] and labels.tolist()==[[[0,1,2,3,255]]]
    m=OldLabelMemory(2,.5)
    positive,known=m.update(['a'],torch.tensor([[2.,2.]]),torch.tensor([[2.,-2.]]))
    assert positive.tolist()==[[True,False]] and known.tolist()==[[True,False]]
    m.update(['a'],torch.tensor([[-4.,-4.]]),torch.tensor([[-4.,-4.]]))
    assert m.visits['a']==2 and m.updates==2
    assert torch.allclose(m.bank['a'][0],(torch.tensor([2.,2.]).sigmoid()+torch.tensor([-4.,-4.]).sigmoid())/2)
    # No annotations are accepted by the memory API.
    import inspect
    assert list(inspect.signature(m.update).parameters)==['names','main_logits','auxiliary_logits']
    print('OLC local behavior passed',flush=True)

if __name__=='__main__':
    run()
    if '--ddp' in sys.argv:
        dist.init_process_group('gloo');rank=dist.get_rank();m=OldLabelMemory(1,.5)
        m.update(['shared',str(rank)],torch.tensor([[2. if rank==0 else -2.],[1.]]),torch.ones(2,1))
        assert m.visits['shared']==1 and len(m.bank)==3 and abs(float(m.bank['shared'][0])-.5)<1e-6
        m.update(['shared',str(rank)],torch.ones(2,1)*2,torch.ones(2,1)*2)
        states=[None]*dist.get_world_size();dist.all_gather_object(states,{k:v.tolist() for k,v in m.bank.items()})
        assert states[0]==states[1] and m.visits['shared']==2
        dist.destroy_process_group()
        print('OLC two-rank memory and duplicate handling passed',rank,flush=True)
