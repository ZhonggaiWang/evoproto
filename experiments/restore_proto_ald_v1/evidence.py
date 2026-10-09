"""Three-state old-class presence from heterogeneous frozen weak evidence.

No pixel annotations or old-class image annotations are accepted. A state of
-1 is uncertainty, not a negative label. Existing high-CAM threshold .7 and
three native pixels are an explicit conservative support rule, not a fitted
probability calibration or a claim of independent evidence.
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def calibrate_presence(old_cls_views,old_aux_views,teacher_views,reference_views,cams,new_tags,high=.7):
    n,v,k=old_cls_views.shape
    assert k==15 and v==2 and old_aux_views.shape==(n,v,k)
    assert teacher_views.shape[:3]==(n,v,16) and reference_views.shape[:3]==(n,v,21)
    h,w=reference_views.shape[-2:]
    cam=F.interpolate(cams,(h,w),mode='bilinear',align_corners=False).clamp(0,1)
    new_strength=(cam[:,15:]*new_tags[:,:,None,None]).amax(1)
    tp=teacher_views.argmax(2);rp=reference_views.argmax(2)
    support=[];teacher_present=[];reference_present=[]
    for c in range(1,16):
        stable=(rp[:,0]==c)&(rp[:,1]==c)&(cam[:,c-1]>=high)&(new_strength<high)
        support.append(stable.sum((1,2)))
        teacher_present.append((tp==c).flatten(1).any(1))
        reference_present.append((rp==c).flatten(1).any(1))
    support=torch.stack(support,1)
    teacher_present=torch.stack(teacher_present,1);reference_present=torch.stack(reference_present,1)
    positive_vote=((old_cls_views>0)|(old_aux_views>0)).any(1)
    negative_consensus=((old_cls_views<=0)&(old_aux_views<=0)).all(1)
    positive=(support>=3)&positive_vote
    negative=negative_consensus&~teacher_present&~reference_present
    assert not (positive&negative).any()
    state=torch.full((n,k),-1,device=cams.device,dtype=torch.long)
    state[positive]=1;state[negative]=0
    return {'state':state,'positive':positive,'negative':negative,'unknown':~(positive|negative),
        'support':support,'raw_positive':old_cls_views[:,0]>0,
        'positive_vote':positive_vote,'negative_consensus':negative_consensus,
        'teacher_present':teacher_present,'reference_present':reference_present}
