from common import *
import torch
from mechanism import *
from network import load_parent

def test():
 device='cuda';torch.cuda.set_device(0);torch.set_num_threads(1);torch.manual_seed(0)
 q=torch.tensor([.1,.1,.5,.2,.1],device=device).view(1,5,1,1)
 p=torch.tensor([.1,.2,.4,.2,.1],device=device).view(1,5,1,1)
 ref=q.log();z=p.log().requires_grad_()
 e=dict(labels=torch.ones(1,1,1,device=device,dtype=torch.long),valid=torch.ones(1,1,1,device=device,dtype=torch.bool),foreground=torch.ones(1,1,1,device=device,dtype=torch.bool),allowed=torch.ones(1,4,device=device,dtype=torch.bool))
 support=torch.zeros(5,5,device=device,dtype=torch.bool);support[1,2]=True
 kd,sep,detail=relation_loss(z,ref,e,support)
 assert kd.abs()<1e-6 and sep>0
 (kd+sep).backward();assert z.grad[0,0].abs()<1e-6 and z.grad[0,1]<0 and z.grad[0,2]>0
 p2=torch.tensor([.1,.35,.25,.2,.1],device=device).view(1,5,1,1)
 z2=p2.log().requires_grad_()
 kd2,sep2,_=relation_loss(z2,ref,e,support)
 assert torch.allclose(kd,kd2,atol=1e-6) and sep2==0
 # The supported KD term is exactly the KL of the collapsed categories.
 p3=torch.tensor([.1,.2,.3,.25,.15],device=device).view(1,5,1,1)
 z3=p3.log()
 k3,_,_=relation_loss(z3,ref,e,support)
 a=torch.tensor([.1,.6,.2,.1],device=device);b=torch.tensor([.1,.5,.25,.15],device=device)
 assert torch.allclose(k3,(a*(a.log()-b.log())).sum(),atol=1e-6)
 k0,s0,_=relation_loss(z,ref,e,torch.zeros_like(support))
 assert torch.allclose(k0,(q*(q.log()-p.log())).sum(),atol=1e-6) and s0==0
 # BG drift is now constrained by the same KD; no separate BG regularizer.
 zb=p.log().clone();zb[:,0]+=1;zb.requires_grad_()
 kb,_,_=relation_loss(zb,ref,e,support);kb.backward();assert zb.grad[:,0]>0
 # Image-absent classes get zero target mass but are not masked in the student.
 ea={**e,'allowed':e['allowed'].clone()};ea['allowed'][:,-1]=False
 za=p.log().clone().requires_grad_();ka,_,da=relation_loss(za,ref,ea,support);ka.backward()
 assert da['q'][:,-1]==0 and za.grad[:,-1]>0
 g=DirectedEvidence(5,8,device)
 for _ in range(5):g.update(e,ref,torch.tensor([0],device=device))
 assert g.counts()[1,2]==1 and not g.supported().any()
 for i in [1,2]:g.update(e,ref,torch.tensor([i],device=device))
 assert g.supported()[1,2] and not g.supported()[2,1]
 rb=torch.tensor([.5,.1,.1,.2,.1],device=device).view(1,5,1,1).log()
 gb=DirectedEvidence(5,8,device)
 for i in range(3):gb.update(e,rb,torch.tensor([i],device=device))
 assert gb.supported()[1,0]
 kbg,sbg,_=relation_loss(rb,rb,e,gb.supported());assert kbg.abs()<1e-6 and sbg>0
 ev={**e,'valid':torch.zeros_like(e['valid']),'foreground':torch.zeros_like(e['foreground'])}
 k,s,_=relation_loss(z,ref,ev,support);assert k==s==0
 # Snapshot conversion must preserve native logits / classifier outputs.
 slim=load_parent().cuda().eval()
 sys.path.append(str(R/'experiments/ald_calibration_v9/src'))
 from model.model_seg_neg import network as Original
 old=Original(backbone='vit_base_patch16_224',num_classes=21,classes_list=[11,5,5],pretrained=False,init_momentum=.9,aux_layer=-3).cuda().eval()
 old.load_state_dict(torch.load(PARENT,map_location='cpu',weights_only=True,mmap=True)['model_state'],strict=True)
 with torch.no_grad():
  for h,w in [(448,448),(384,512)]:
   x=torch.randn(1,3,h,w,device=device);a=slim(x);b=old(x)
   assert torch.equal(a[0],b[1]);assert torch.equal(a[3],b[0]);assert torch.equal(a[4],b[3])
 checks=['collapsed_KD_exact_KL','preserves_pair_mass_not_wrong_internal_order','SEP_correct_direction','SEP_stops_when_order_correct','unsupported_pair_full_KD','background_mass_retained_by_same_KD','absent_class_suppressed_by_same_KD','padding_zero_relation_loss','distinct_image_support_not_exposures','directed_not_symmetric_graph','slim_model_exact_native_logits_and_classifiers_two_shapes']
 atomic_json(E/'preflight.json',dict(passed=True,checks=checks,source_sha256={p.name:digest(p) for p in E.glob('*.py')},utc=now()))
 print(checks)

if __name__=='__main__':test()
