import sys,os,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
import torch.distributed as dist
from experiments.restore_proto_simplekd_v1.seeds import SeedCorrection,flood


def fixture(mode='correct'):
    # New-class seeds at left, matching novel pixels next to them, a gap,
    # and a disconnected matching region at right. Teacher calls novel class old1.
    k,oc=4,3;f=torch.zeros(1,2,3,8);f[:,0]=1
    f[:,:,1,0:3]=torch.tensor([0.,1.])[:,None]
    f[:,:,1,6:8]=torch.tensor([0.,1.])[:,None]
    anchor=torch.full((1,3,8),255,dtype=torch.long);anchor[0,1,0:2]=3;anchor[0,0]=0
    tp=torch.ones_like(anchor);tp[:,0]=0
    ca=torch.zeros(1,3,3,8);ca[:,2]=.4;cb=ca.clone()
    labels=tp.clone();labels[0,1,0:2]=3;valid=torch.ones_like(labels,dtype=torch.bool)
    m=SeedCorrection(k,oc,'cpu',mode)
    for name in ['a','a','b']:m.update(anchor,tp,[name])
    assert not m.graph().any(),'Repeated image must not create three-image support'
    m.update(anchor,tp,['c']);assert m.graph()[3,1] and not m.graph()[3,0]
    e=(f,ca,cb,tp,anchor,valid)
    return m,e,labels,valid

m,e,y,v=fixture();original=y.clone();out,mask,proposal,stats=m.correct(e,y,v)
assert torch.equal(y,original)
assert mask.sum()==1 and mask[0,1,2] and out[0,1,2]==3
assert not mask[0,1,6:].any(),'No jump to disconnected lookalike'
from model.losses import get_seg_loss
logits=torch.zeros(1,4,3,8,requires_grad=True)
loss=get_seg_loss(logits,out,torch.ones(4));loss.backward()
assert logits.grad[0,3,1,2]<0 and logits.grad[0,1,1,2]>0,'Correction must train new, not old'
assert logits.grad[0,1,1,6]<0 and logits.grad[0,3,1,6]>0,'Unchanged old targets retain their gradients'

assert torch.equal(m.correct(e,y,v,active=False)[0],y)
for mode in ['ignore','off','no_graph']:
    m.mode=mode;z,mask,_,_=m.correct(e,y,v)
    assert z[0,1,2]==(255 if mode=='ignore' else 1 if mode=='off' else 3)
m.mode='correct'
e[4][0,1,2]=1
assert not m.correct(e,y,v)[1].any(),'Preserve reliable old anchor'
# Missing references or missing seeds cannot propagate.
m,e,y,v=fixture();e[4][:]=255
assert not m.correct(e,y,v)[1].any()
# Padded targets cannot be edited.
m,e,y,v=fixture();v[0,1,2]=False
assert not m.correct(e,y,v)[1].any()
print('PASS CPU: new->old direction, unique image support, connected propagation, modes, old anchors, no seeds, padding, immutability',flush=True)

if '--ddp' in sys.argv:
    dist.init_process_group('gloo');rank=dist.get_rank()
    m=SeedCorrection(4,3,'cpu')
    a=torch.tensor([[[3,3],[255,0]]]);t=torch.tensor([[[1,0],[2,0]]])
    m.update(a,t,[f'image-{rank}']);m.update(a,t,['shared'])
    assert m.graph()[3,1] and not m.graph()[3,0]
    states=[None]*dist.get_world_size();dist.all_gather_object(states,m.state());assert states[0]==states[1]
    assert m.state()['unique_image_support'][3][1]==3
    assert torch.allclose(m.count[3,1],m.mass[3]/2)
    print('PASS DDP',rank,flush=True);dist.destroy_process_group()

# Verify actual KD gradients, both branches, with all former reliability signals zero.
from experiments.restore_proto_simplekd_v1.relation import ConfusionProto
m,e,y,v=fixture('no_graph');m.correct(e,y,v)
assert m.novel_region[0,1,0:3].all() and not m.novel_region[0,1,6:].any()
m.correct(e,y,v,active=False);assert not m.novel_region.any()
r=ConfusionProto(4,3,'cpu');r.novel_mask=torch.tensor([[[False,True,False,False]]])
a=torch.full((1,1,4),255,dtype=torch.long);tp=torch.tensor([[[1,1,0,1]]]);valid=torch.ones_like(tp,dtype=torch.bool)
proto=torch.zeros(1,4,1,4,requires_grad=True);main=torch.zeros_like(proto,requires_grad=True)
teacher=torch.zeros_like(proto);teacher[:,1]=2;teacher[:,2]=-2
cams=torch.zeros(1,3,1,4);par=torch.full_like(tp,3)
sp=torch.randn(4,512,requires_grad=True);oldp=sp.detach().clone()
loss,stats=r.loss(proto,teacher,main,teacher,a,cams,valid,tp,par,sp,oldp)
assert stats['kd_pixels']==2 and stats['kd_main']>0 and stats['kd_proto']>0
loss.backward()
for z in [proto,main]:
 assert z.grad[...,0].abs().sum()>0 and z.grad[...,3].abs().sum()>0
 assert z.grad[...,1:3].abs().sum()==0,'Novel veto and teacher BG must have zero KD gradient'
print('PASS simplified KD: both-head gradients, novel veto, BG exclusion, no confidence/CAM/PAR gate')
