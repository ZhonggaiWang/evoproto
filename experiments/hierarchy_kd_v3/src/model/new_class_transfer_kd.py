"""Transfer teacher foreground mass along a trusted new->old confusion.

An online selected old competitor i is the teacher's foreground prediction,
while independent CAM/PAR evidence identifies current-new class j. Preserve
teacher foreground recognition by assigning its i probability mass to j.
The target is a lower bound: never soften an already stronger student j.
Pixel GT is not accepted. Teacher-BG pixels cannot trigger this transfer.
"""
import torch
import torch.nn.functional as F
from model.hierarchical_kd import balanced_mean
from model.pixel_kd import distributed_sum


def binary_class_log_prob(logits, labels, temperature):
    z=logits/temperature
    target=z.gather(1,labels[:,None])[:,0]
    others=z.masked_fill(F.one_hot(labels,z.shape[1]).permute(0,3,1,2).bool(),-torch.inf).logsumexp(1)
    return torch.stack((target,others),1).log_softmax(1)


def new_class_transfer_kd(student,teacher,par_labels,cams,img_box,pair_targets,temperature=2.,high=.7):
    n,c,h,w=student.shape;k=teacher.shape[1];old=k-1
    if not 1<k<c or temperature<=0 or pair_targets.shape!=(c,):
        raise ValueError('Invalid incremental mapping or temperature')
    with torch.no_grad():
        t=F.interpolate(teacher.detach(),(h,w),mode='bilinear',align_corners=False)
        p=t.argmax(1)
        y=F.interpolate(par_labels[:,None].float(),(h,w),mode='nearest')[:,0].long()
        safe_y=y.clamp(0,c-1)
        target=pair_targets[safe_y]
        cam=F.interpolate(cams.detach(),(h,w),mode='bilinear',align_corners=False).clamp(0,1)
        top=cam.topk(2,dim=1)
        strength=top.values[:,0];gap=(strength-top.values[:,1]).clamp_min(0)
        valid=torch.zeros_like(par_labels,dtype=torch.float)
        for i,box in enumerate(img_box):
            a,b,l,r=[int(x) for x in box];valid[i,a:b,l:r]=1
        valid=F.interpolate(valid[:,None],(h,w),mode='nearest')[:,0]>0
        anchor=(valid & (y>=k) & (y<c) & (top.indices[:,0]+1==y)
                & (strength>=high) & (gap>0) & (target>0) & (target<=old) & (p==target))
        reliability=(2*t.sigmoid().gather(1,p[:,None])[:,0]-1).clamp_min(0).square()
        weights=strength*gap*reliability
        target_log=binary_class_log_prob(t,p,temperature)
        target_probability=target_log.exp()
    student_log=binary_class_log_prob(student,safe_y,temperature)
    active=anchor & (weights>0) & (student_log[:,0].detach()<target_log[:,0])
    values=(target_probability*(target_log-student_log)).sum(1)*temperature**2
    loss,counts,reported=balanced_mean(values,active,safe_y,c,weights)
    packed=distributed_sum(torch.stack((valid.sum(),anchor.sum(),active.sum(),(active*weights).sum())).float())
    return loss,{'transfer_kd':float(reported),'transfer_valid_pixels':int(packed[0]),
        'transfer_anchor_pixels':int(packed[1]),'transfer_pixels':int(packed[2]),
        'transfer_weight_sum':float(packed[3]),'transfer_class_counts':counts.tolist(),
        'transfer_definition':'teacher old competitor foreground mass -> trusted new anchor mass lower bound',
        'transfer_teacher_BG_used':False}
