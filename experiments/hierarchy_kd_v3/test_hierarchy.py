from pathlib import Path
import os,sys,json
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/hierarchy_kd_v3'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
os.environ.update(environment('8card'));sys.path.insert(0,str(E/'src'))
import torch
import torch.distributed as dist
import model.new_class_transfer_kd as m
import model.hierarchical_kd as reduction
from kd_runtime import atomic_json,digest,safe_path
torch.set_num_threads(1)

def batch(rank=0,device='cpu'):
    n=2;t=torch.full((n,4,1,4),-2.,device=device);s=torch.full((n,6,1,4),-1.,device=device)
    t[:,1,0,:2]=3;t[:,2,0,2:]=2
    s[:,1,0,:2]=2;s[:,2,0,2:]=2
    s[:,4,0,:2]+=rank*.1;s[:,5,0,2:]+=rank*.07
    y=torch.tensor([[[4,4,5,5]]],device=device).repeat(n,1,1)
    cam=torch.zeros(n,5,1,4,device=device);cam[:,3,0,:2]=.9;cam[:,4,0,2:]=.8
    targets=torch.tensor([-1,-1,-1,-1,1,2],device=device)
    boxes=[[0,1,0,4]]*2
    if rank==1:boxes[0]=[0,1,1,4]
    return [s,t,y,cam,boxes,targets]

def unit():
    b=batch();b[0].requires_grad_();b[1].requires_grad_();loss,st=m.new_class_transfer_kd(*b);loss.backward()
    assert st['transfer_pixels']==8 and float(loss)>0 and b[1].grad is None
    assert (b[0].grad[:,4,0,:2]<0).all() and (b[0].grad[:,1,0,:2]>0).all()
    assert (b[0].grad[:,5,0,2:]<0).all() and (b[0].grad[:,2,0,2:]>0).all()
    assert (b[0].grad[:,0]>0).all() # rejecting BG with trusted new foreground, never teaching BG
    for mode in ['teacher_BG','wrong_pair','old_anchor','low_CAM','empty_ROI','already_stronger']:
        x=batch();x[0].requires_grad_()
        if mode=='teacher_BG':x[1][:,0]=20
        elif mode=='wrong_pair':x[5][4:]=3
        elif mode=='old_anchor':x[2].fill_(1)
        elif mode=='low_CAM':x[3].mul_(.1)
        elif mode=='empty_ROI':x[4]=[[0,0,0,0]]*2
        else:
            with torch.no_grad():x[0][:,4,0,:2]=30;x[0][:,5,0,2:]=30
        a,stats=m.new_class_transfer_kd(*x);a.backward()
        assert float(a)==0 and int(torch.count_nonzero(x[0].grad))==0,mode
    x=batch();a,_=m.new_class_transfer_kd(*x);x[0]+=17;x[1]+=13;b,_=m.new_class_transfer_kd(*x)
    # Logit distributions are translation invariant; evidence reliability is
    # intentionally NOT: teacher sigmoid confidence retains its native scale.
    assert torch.allclose(m.binary_class_log_prob(x[0],x[2],2),m.binary_class_log_prob(x[0]-17,x[2],2),atol=1e-6)
    return ['foreground_mass_transfer_gradient','teacher_detached','no_BG_promotion',
        'teacher_BG_veto','wrong_direction_veto','old_anchor_veto','low_CAM_veto',
        'empty_ROI_zero_gradient','stronger_student_not_degraded','class_probability_shift_invariance']

checks=unit()
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);dist.init_process_group('nccl');world=dist.get_world_size();assert world==8
b=batch(rank,f'cuda:{rank}')
if rank==0:b[4]=[[0,0,0,0]]*2
b[0].requires_grad_();loss,stats=m.new_class_transfer_kd(*b);loss.backward();actual=b[0].grad/world
parts=[batch(i,f'cuda:{rank}') for i in range(world)];parts[0][4]=[[0,0,0,0]]*2
full=[torch.cat([x[j] for x in parts]) if j<4 else sum([x[j] for x in parts],[]) if j==4 else parts[0][j] for j in range(6)]
dist.barrier();reduction.distributed_sum=lambda x:x.detach().clone();reduction.world_size=lambda:1
m.distributed_sum=lambda x:x.detach().clone()
full[0].requires_grad_();reference,_=m.new_class_transfer_kd(*full);reference.backward()
error=(actual-full[0].grad[2*rank:2*(rank+1)]).abs().max()
value_error=(loss-reference).abs().detach();assert float(error)<1e-7 and float(value_error)<1e-6
dist.all_reduce(error,op=dist.ReduceOp.MAX);dist.all_reduce(value_error,op=dist.ReduceOp.MAX)
if rank==0:
    hashes={str(p.relative_to(E/'src')):digest(safe_path(p)) for p in sorted((E/'src').rglob('*.py'))}
    report={'passed':True,'checks':checks+['8rank_NCCL_loss_DDP_gradient_unequal_support_empty_rank'],
        'ranks':world,'maximum_gradient_error':float(error),'maximum_value_error':float(value_error),
        'source_sha256':hashes,'test_sha256':digest(Path(__file__)),
        'unchanged_hierarchy_and_background_tests':str(R/'experiments/hierarchy_kd_v2/preflight.json')}
    atomic_json(E/'preflight.json',report);print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'}),flush=True)
dist.destroy_process_group()
