import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
import torch.nn.functional as F
from experiments.restore_proto_cas_v1.supervision import ambiguity_loss,select_conflicts
from model.losses import get_seg_loss

torch.manual_seed(23)
z=torch.randn(2,5,3,4,requires_grad=True)
y=torch.randint(0,5,(2,3,4));y[0,0,0]=255
mask=torch.zeros_like(y,dtype=torch.bool);a=torch.ones_like(y);b=a*3;w=torch.ones(5)
base=get_seg_loss(z,y,w)
for mode in ['off','ignore','confusion_pair','local_pair']:
    l=ambiguity_loss(z,y,mask,a,b,mode,w)
    assert torch.allclose(base,l,atol=2e-6), (base,l)
    assert torch.allclose(torch.autograd.grad(base,z,retain_graph=True)[0],torch.autograd.grad(l,z,retain_graph=True)[0],atol=1e-7)
mask[1,1,1]=True
li=ambiguity_loss(z,y,mask,a,b,'ignore',w)
gi=torch.autograd.grad(li,z,retain_graph=True)[0]
assert torch.equal(gi[1,:,1,1],torch.zeros(5))
l=ambiguity_loss(z,y,mask,a,b,'confusion_pair',w)
g=torch.autograd.grad(l,z)[0]
assert torch.all(g[1,[0,2,4],1,1]>0)
assert torch.equal(g[0,:,0,0],torch.zeros(5))
# Exact likelihood: two mutually exclusive Bernoulli configurations.
v=torch.tensor([.2,-.8,1.1,.4,-1.2],dtype=torch.float64)
p=v.sigmoid();prob=0.
for c in [1,3]:
    t=1-p.clone();t[c]=p[c];prob+=t.prod()
formula=F.softplus(v).sum()-torch.logsumexp(v[[1,3]],0)
assert torch.allclose(formula,-prob.log(),atol=1e-12)
# Equal candidates receive equal gradients, with no preference for the hard label.
v=torch.zeros(1,5,1,1,requires_grad=True)
l=ambiguity_loss(v,torch.tensor([[[3]]]),torch.ones(1,1,1,dtype=torch.bool),torch.tensor([[[1]]]),torch.tensor([[[3]]]),'confusion_pair',w)
g=torch.autograd.grad(l,v)[0]
assert g[0,1,0,0]==g[0,3,0,0]
# No GT interface; prevent background, absent tags, weak evidence and invalid padding.
teacher=torch.tensor([[[[-3.]*6],[[3.]*6],[[-3.]*6]]])
cams=torch.zeros(1,4,1,6);cams[:,2]=.5;aux=cams.clone();tags=torch.ones(1,4)
par=torch.full((1,1,6),3);valid=torch.ones_like(par,dtype=torch.bool);valid[:,:,0]=False
cams[:,:,0,1]=0;aux[:,:,0,2]=0;par[:,:,3]=0;teacher[:,:,0,4]=0
G=torch.zeros(5,5);G[1,3]=1
proto=torch.zeros(1,5,1,6);proto[:,1]=1
m,_,_,local=select_conflicts(teacher,cams,aux,tags,par,valid,G,3,'confusion_pair',prototype=proto)
assert m.flatten().tolist()==[False,False,False,False,False,True],m
assert not select_conflicts(teacher,cams,aux,tags,par,valid,G*0,3,'confusion_pair',prototype=proto)[0].any()
assert select_conflicts(teacher,cams,aux,tags,par,valid,G*0,3,'local_pair',prototype=proto)[0].sum()==1
assert not select_conflicts(teacher,cams,aux,tags,par,valid,G.T,3,'confusion_pair',prototype=proto)[0].any()
assert not select_conflicts(teacher,cams,aux,tags,par,valid,G,3,'local_pair',prototype=-proto)[0].any()
strong=cams.clone();strong[:,2]=.9
assert not select_conflicts(teacher,strong,strong,tags,par,valid,G,3,'local_pair',prototype=proto)[0].any()
tags[:,0]=0
assert not select_conflicts(teacher,cams,aux,tags,par,valid,G,3,'local_pair',prototype=proto)[0].any()
# All-ignore stays finite and differentiable.
l=ambiguity_loss(z,torch.full_like(y,255),mask,a,b,'confusion_pair',w)
assert l==0
l.backward()
print('PASS: baseline loss/gradient equivalence; exact marginal likelihood; ignore/unknown gradients; conflict evidence guards; all-ignore')
