from pathlib import Path
import sys,os,math
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/local_sep_v3';S=E/'src'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
os.environ.update(environment('8card'));sys.path.insert(0,str(S))
import torch
import torch.distributed as dist
from model.boundary_sep import targets,separation_loss,PairEvidence
from kd_runtime import atomic_json,digest
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);dist.init_process_group('nccl');torch.set_num_threads(1)
dev='cuda';ref=torch.full((1,21,1,3),-3.,device=dev);ref[:,16]=.2;ref[:,9]=0;ref[:,0]=-4
ref[:,16,0,1]=3;ref[:,0,0,2]=4;ref.requires_grad_(True)
y=torch.full((1,1,3),16,device=dev);cam=torch.zeros(1,20,1,3,device=dev);cam[:,15]=.9
valid=torch.ones_like(y,dtype=torch.bool);t=targets(y,cam,cam,valid,ref)
assert t['eligible'].tolist()==[[[True,False,False]]]
assert t['rival'][0,0,0]==9 and not t['margin'].requires_grad
assert ((t['margin']-t['reference_gap'])<=math.log(2)+1e-6).all()
z=ref.detach().clone().requires_grad_(True);support=torch.ones(21,21,device=dev,dtype=torch.bool)
l,stats=separation_loss(z,t,support);l.backward()
assert l>0 and z.grad[0,16,0,0]<0 and z.grad[0,9,0,0]>0
assert z.grad[:,0].abs().sum()==0 and z.grad[:,:,:,1:].abs().sum()==0 and ref.grad is None
assert torch.allclose(z.grad.sum(1),torch.zeros_like(y,dtype=torch.float),atol=1e-6)
done=ref.detach().clone();done[:,16,0,0]=2;done.requires_grad_(True)
zero,_=separation_loss(done,t,support);zero.backward();assert zero==0 and done.grad.abs().sum()==0
no,_=separation_loss(z,t,~support);assert no==0
aux=cam.clone();aux[:,15]=0;aux[:,8]=.9
assert not targets(y,cam,aux,valid,ref)['eligible'].any()
old=y.clone();old.fill_(9);assert not targets(old,cam,cam,valid,ref)['eligible'].any()
invalid=valid.clone();invalid[:,:,0]=False;assert not targets(y,cam,cam,invalid,ref)['eligible'].any()
# All ranks share one scalar; local support differs. DDP-averaged gradient must
# match a single global cohort including one denominator pseudo-count/image.
p=torch.tensor(.2,device=dev,requires_grad=True);src=torch.nn.functional.one_hot(torch.tensor(16,device=dev),21).float()[None,:,None,None]
zz=src*p;tt={k:v[:,:,:1].clone() for k,v in t.items()};tt['eligible'].fill_(rank%2==0)
loss,_=separation_loss(zz,tt,support);loss.backward();g=p.grad.clone();dist.all_reduce(g);g/=dist.get_world_size()
pp=torch.tensor(.2,device=dev,requires_grad=True);expected=((math.log(2)-pp)/math.log(2))**2/3;expected.backward()
assert torch.allclose(loss,expected,atol=1e-6) and torch.allclose(g,pp.grad,atol=1e-5)
evidence=PairEvidence().cuda();assert not evidence.supported().any();evidence.update(tt,zz)
assert evidence.image_exposures[16,9]==4 and evidence.supported()[16,9]
evidence.updates+=51;assert not evidence.supported()[16,9]
# Resolving a second sample must not magnify the first sample's gradient.
tt2={k:v[:,:,:1].expand(1,1,2).clone() for k,v in t.items()}
grads=[]
for second in [.2,2.]:
 a=torch.zeros(1,21,1,2,device=dev);a[:,16,0,0]=.2;a[:,16,0,1]=second;a.requires_grad_(True)
 ll,_=separation_loss(a,tt2,support);ll.backward();grads.append(a.grad[0,16,0,0].clone())
assert torch.equal(grads[0],grads[1])
# Exact identity initialization and native-grid probability constraints.
from model.pair_adapter import PairAdapter,redistribute
torch.manual_seed(19)
base=torch.randn(8,21,3,5,device=dev)*2
raw=(torch.randn_like(base)*5).requires_grad_(True)
corrected=redistribute(base,raw)
assert torch.equal(redistribute(base,torch.zeros_like(base)),base)
idx=base[:,1:].topk(2,1).indices+1
chosen=torch.zeros_like(base,dtype=torch.bool).scatter_(1,idx,True)
assert torch.equal(corrected[~chosen],base[~chosen])
assert torch.allclose(torch.logsumexp(corrected.gather(1,idx),1),torch.logsumexp(base.gather(1,idx),1),atol=2e-6)
assert torch.allclose(corrected.softmax(1)[:,0],base.softmax(1)[:,0],atol=2e-7)
assert torch.equal(corrected.argmax(1)==0,base.argmax(1)==0)
assert (corrected-base).abs().max()<=math.log(2)+1e-6
oldpair=(idx<16).all(1);assert torch.equal(corrected.permute(0,2,3,1)[oldpair],base.permute(0,2,3,1)[oldpair])
corrected[:,16:].sum().backward();assert torch.isfinite(raw.grad).all()
assert raw.grad[~chosen].abs().sum()==0
adapter=PairAdapter().cuda();feat=torch.randn(8,512,3,5,device=dev)
assert torch.equal(adapter(feat,base),base)
opt=torch.optim.AdamW(adapter.parameters(),lr=.001)
a=adapter(feat,base);a[:,16:].sum().backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter.parameters())
opt.step();assert not torch.equal(adapter(feat,base),base)
# Confirm the entire new architecture is exactly the original checkpoint's
# classifier for a real backbone forward, before adapter optimization.
import json
from model.model_seg_neg import network
cfg=json.loads((R/'runs/ald_calibration_v9/formal/study.json').read_text())['config']
ck=torch.load(R/'runs/ald_calibration_v9/formal/10-5/step2/checkpoints/model_final.pth',map_location='cpu',weights_only=True,mmap=True)
m=network(backbone=cfg['backbone'],num_classes=21,classes_list=[11,5,5],pretrained=False,init_momentum=cfg['momentum'],aux_layer=cfg['aux_layer'])
missing=m.load_state_dict(ck['model_state'],strict=False)
assert not missing.unexpected_keys and len(missing.missing_keys)==3 and all(k.startswith('decoder.local_sep.') for k in missing.missing_keys)
m=m.cuda().eval().requires_grad_(False);capture={}
def hook(mod,args):capture['f']=args[0]
h=m.decoder.conv8.register_forward_pre_hook(hook)
with torch.no_grad():
 output=m(torch.randn(1,3,448,448,device=dev))[1]
 expected=m.decoder.conv8(capture['f'])
 assert torch.equal(output,expected)
h.remove();del m,ck

dist.barrier()
if rank==0:
 atomic_json(E/'preflight.json',{'passed':True,'ranks':8,'tests':['source_rival_only_logit_gradients','background_gradient_zero','bounded_reference_margin_detached','already_resolved_zero_gradient','no_support_zero','main_aux_disagreement_veto','old_source_excluded','padding_excluded','unequal_DDP_global_gradient','pair_exposure_global_count_and_staleness','fixed_cohort_no_remaining_example_amplification','adapter_exact_zero_identity_full_backbone','adapter_nonpair_logits_exact','adapter_pair_probability_mass','adapter_native_BG_probability_and_decision','adapter_bounded_delta','adapter_old_only_pairs_untouched','adapter_finite_gradient_and_trainability'],
 'source_sha256':{str(p.relative_to(S)):digest(p) for p in S.rglob('*.py')},'test_sha256':digest(E/'test_local.py')})
 print('PASS 8 GPU boundary SEP invariants',flush=True)
dist.destroy_process_group()
