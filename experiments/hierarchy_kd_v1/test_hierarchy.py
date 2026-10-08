"""Gradient semantics, routing invariants and exact eight-rank global reduction."""
from pathlib import Path
import os,sys,json
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto'); E=R/'experiments/hierarchy_kd_v1'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
os.environ.update(environment('8card'))
sys.path.insert(0,str(E/'src'))
import torch
import torch.distributed as dist
import model.hierarchical_kd as m
from model.online_directed_confusion import OnlineDirectedConfusion
from kd_runtime import atomic_json,digest,safe_path
torch.set_num_threads(1)

def batch(n=2,offset=0,device='cpu'):
    t=torch.full((n,4,1,4),-2.,device=device);s=torch.full((n,6,1,4),-1.,device=device)
    t[:,1,0,0]=3;t[:,2,0,1]=3;t[:,0,0,2:]=3
    s[:,1,0,0]=2;s[:,0,0,1]=2;s[:,4,0,2]=2;s[:,5,0,3]=2
    s=s+torch.arange(n,device=device)[:,None,None,None]*.01+offset*.01
    cam=torch.zeros(n,5,1,4,device=device);cam[:,0,0,0]=.9;cam[:,1,0,1]=.9
    y=torch.tensor([[[1,2,0,0]]],device=device).repeat(n,1,1)
    tags=torch.tensor([[0,1]],device=device).repeat(n,1)
    directions=torch.ones(6,dtype=torch.bool,device=device);directions[0]=False
    boxes=[[0,1,0,4]]*n
    return s,t,y,cam,boxes,tags,directions,directions.clone()

def call(b):return m.hierarchical_kd_loss(*b)

def unit():
    b=list(batch());b[0].requires_grad_();b[1].requires_grad_()
    a,z,stats=call(b);(a+z).backward()
    assert b[1].grad is None and torch.isfinite(b[0].grad).all()
    assert stats['mass_pixels']==4 and stats['absent_new_pixels']==2 and stats['background_pair_pixels']==2
    assert b[0].grad[:,4,0,2].min()>0 # reject absent new class
    assert b[0].grad[:,5,0,3].min()>0 and b[0].grad[:,0,0,3].max()<0
    assert torch.all(b[0].grad[:,:,0,2][:,[0,1,2,3,5]]<0) # no forced BG target
    shifted=b[0].detach().clone();shifted[:,1:4]+=2
    assert not torch.allclose(m.group_log_prob(shifted,3),m.group_log_prob(b[0],3))
    assert torch.allclose(shifted[:,1:4].softmax(1),b[0][:,1:4].softmax(1))
    # Reallocate student BG/new probability mass with unchanged total: group KL invariant.
    x=torch.randn(2,6,2,2);q=x.clone();q[:,[0,4,5]]=x[:,[5,0,4]]
    assert torch.allclose(m.group_log_prob(x,3),m.group_log_prob(q,3),atol=1e-6)
    empty=list(batch());empty[0].requires_grad_();empty[4]=[[0,0,0,0]]*2
    a,z,st=call(empty);(a+z).backward()
    assert float(a+z)==0 and torch.count_nonzero(empty[0].grad)==0
    assert st['valid_pixels']==0
    # New PAR veto remains effective even when teacher claims old foreground.
    veto=list(batch());veto[2].fill_(4);a,z,st=call(veto);assert st['mass_pixels']==0
    observer=OnlineDirectedConfusion(6,stage=2)
    o,g=m.select_mass_directions(observer,3);assert not o.any() and not g.any()
    observer.updates.fill_(100);observer.broad_last_seen_update.fill_(99)
    observer.broad_image_observations.fill_(20);observer.broad_pair_image_observations.fill_(5)
    observer.broad_ema_counts.copy_(torch.eye(6)*100)
    observer.broad_ema_counts[1:4,0]=torch.tensor([1.,3.,2.]);observer.broad_ema_counts[0,1:]=torch.tensor([1.,2.,3.,4.,5.])
    o,g=m.select_mass_directions(observer,3);assert o.nonzero().flatten().tolist()==[1,2,3] and g.nonzero().flatten().tolist()==[3,4,5]
    observer.broad_last_seen_update[0]=-1;o,g=m.select_mass_directions(observer,3);assert not g.any()
    clone=OnlineDirectedConfusion(6,stage=2);clone.load_state_dict(observer.state_dict(),strict=True)
    x=m.select_mass_directions(observer,3);y=m.select_mass_directions(clone,3)
    assert all(torch.equal(a,b) for a,b in zip(x,y))
    return ['teacher_detached','old_mass_shift_sensitive','conditional_invariant',
            'BG_new_group_invariance','absent_class_complement_gradient','BG_pair_gradient',
            'zero_ROI_gradient','new_PAR_veto','past_routing_support_staleness','routing_resume_exact']

checks=unit()
rank=int(os.environ.get('LOCAL_RANK',0));torch.cuda.set_device(rank)
dist.init_process_group('nccl');world=dist.get_world_size();assert world==8
b=list(batch(n=2,offset=rank,device=f'cuda:{rank}'))
# Create unequal per-rank eligibility and an entirely empty rank.
if rank==0:b[4]=[[0,0,0,0]]*2
elif rank%2:b[2][0,0,0]=255
b[0].requires_grad_();a,z,stats=call(b);(a+z).backward()
actual=b[0].grad.detach()/world;reported=torch.stack((a,z)).detach()
# Every rank independently evaluates the identical concatenated global batch,
# with reductions disabled ONLY after all collective production calls finish.
dist.barrier()
parts=[]
for i in range(world):
    v=list(batch(n=2,offset=i,device=f'cuda:{rank}'))
    if i==0:v[4]=[[0,0,0,0]]*2
    elif i%2:v[2][0,0,0]=255
    parts.append(v)
full=[torch.cat([p[j] for p in parts],0) if j in (0,1,2,3,5) else
      sum([p[j] for p in parts],[]) if j==4 else parts[0][j] for j in range(8)]
m.distributed_sum=lambda x:x.detach().clone();m.world_size=lambda:1
full[0].requires_grad_();aa,zz,_=call(full);(aa+zz).backward()
error=(actual-full[0].grad[rank*2:(rank+1)*2]).abs().max()
value_error=(reported-torch.stack((aa,zz)).detach()).abs().max()
assert float(error)<1e-7 and float(value_error)<1e-6,(error,value_error)
dist.all_reduce(error,op=dist.ReduceOp.MAX);dist.all_reduce(value_error,op=dist.ReduceOp.MAX)
if rank==0:
    src=E/'src';hashes={str(p.relative_to(src)):digest(safe_path(p)) for p in sorted(src.rglob('*.py'))}
    report={'passed':True,'checks':checks+['8rank_NCCL_values_and_DDP_averaged_gradients_unequal_support_empty_rank'],
        'ranks':world,'maximum_gradient_error':float(error),'maximum_value_error':float(value_error),
        'source_sha256':hashes,'test_sha256':digest(Path(__file__))}
    atomic_json(E/'preflight.json',report);print(json.dumps(report),flush=True)
dist.destroy_process_group()
